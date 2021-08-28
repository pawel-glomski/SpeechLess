import av
import json
import argparse
import numpy as np
import pytsmod as tsm

from typing import List, Tuple
from fractions import Fraction
from logging import Logger
from pathlib import Path

from .utils import Real, rangesOfTruth, intLinspaceStepsByLimit

VIDEO_STREAM_TYPE = 'video'
AUDIO_STREAM_TYPE = 'audio'
SUPPORTED_STREAM_TYPES = [VIDEO_STREAM_TYPE, AUDIO_STREAM_TYPE]

DROP_FRAME_PTS = -1

AUD_WIN_TYPE = 'hann'
AUD_WIN_SIZE = 1024
AUD_HOP_SIZE = int(AUD_WIN_SIZE/4)
AUD_WS_PAD_SIZE = 512
AUD_WS_SIZE_MAX = AUD_WS_PAD_SIZE + 1000*AUD_WIN_SIZE + AUD_WS_PAD_SIZE
AUD_VFRAME_SIZE_MAX = 10*1024


def seekBeginning(stream):
    stream.container.seek(np.iinfo(np.int32).min, stream=stream, backward=True, any_frame=False)


class Range:
    @staticmethod
    def fromNumpy(arr: np.ndarray) -> List[np.ndarray]:
        return [Range(*r) for r in arr]

    def __init__(self, beg, end, multi) -> None:
        self.beg = beg
        self.end = end
        self.multi = multi

    @property
    def beg(self):
        return self._beg

    @beg.setter
    def beg(self, value):
        self._beg = Real(value) if value is not None else None

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = Real(value) if value is not None else None

    @property
    def multi(self):
        return self._multi

    @multi.setter
    def multi(self, value):
        self._multi = Real(value) if value is not None else None


class EditCtx:
    def __init__(self, srcStream, dstStream):
        self.srcStream = srcStream
        self.dstStream = dstStream
        self.isDone = False

    def __bool__(self):
        return self.isDone

    def _prepareSrcDurs(self) -> np.ndarray:
        """Collects pts of each frame """
        seekBeginning(self.srcStream)
        pts = []
        for packet in self.srcStream.container.demux(self.srcStream):
            if packet.dts is not None:
                pts.append(packet.pts)
        if len(pts) < 2:  # there must be at least 2 frames
            return [], False

        pts = sorted(set(pts))
        # convert to seconds and set the end frame pts
        pts = np.array(pts, dtype=Real) * Real(self.srcStream.time_base)
        durs = pts[1:] - pts[:-1]
        # add virtual empty first frame that lasts until the first real frame
        virtualFirst = ([pts[0]] if pts[0] > 0 else [])
        # assume duration of the last frame is equal to the average duration of real frames
        return (np.concatenate([virtualFirst, durs, [np.mean(durs)]], dtype=Real),
                len(virtualFirst) == 1)

    def _prepareRawDstDurations(self, srcDurs: np.ndarray, ranges: List[Range]) -> np.ndarray:
        dstDurs = []
        rIdx = 0
        frEnd = Real(0)
        srcStreamEnd = np.sum(srcDurs, dtype=Real)
        for oldDur in srcDurs:
            frBeg = frEnd
            frEnd = frBeg + oldDur
            newDur = 0

            while rIdx < len(ranges) and frBeg < ranges[rIdx].end and frEnd > ranges[rIdx].beg:
                clampedBeg = np.max([ranges[rIdx].beg, frBeg])
                clampedEnd = np.min([ranges[rIdx].end, frEnd])
                partDur = clampedEnd - clampedBeg
                assert np.round(oldDur - partDur, 12) >= 0

                oldDur -= partDur
                newDur += partDur * ranges[rIdx].multi
                if clampedEnd < ranges[rIdx].end:
                    # this range extends beyond this frame (was clamped), move to the next frame
                    break
                else:
                    # this range ends on this frame (wasn't clamped), move to the next range
                    rIdx += 1
            newDur += oldDur

            # early stop when the recording has trimmed end
            if newDur == 0 and rIdx < len(ranges) and ranges[rIdx].end >= srcStreamEnd:
                break
            dstDurs.append(newDur)

        return np.array(dstDurs, dtype=Real) if len(dstDurs) >= 2 else np.array([])


class VidCtx(EditCtx):
    def __init__(self, srcStream, dstStream, maxFPS: float):
        super().__init__(srcStream, dstStream)
        self.maxFPS = maxFPS
        self.frIdx = 0
        self.dstPTS = None

    def prepareTimelineEdits(self, ranges: List[Range]) -> bool:
        durs, virtualFirst = self._prepareSrcDurs()
        if len(durs) == 0:
            return False

        if len(ranges) > 0:
            durs = self._prepareRawDstDurations(durs, ranges)
            durs = self._constrainRawDstDurations(durs)
            if len(durs) == 0 or np.sum(durs) == 0:
                return False

        self.dstPTS = np.concatenate([[0], np.cumsum(durs[:-1])])
        self.dstPTS[durs <= 0] = DROP_FRAME_PTS

        # if first frame is virtual, discard it
        if virtualFirst:
            self.dstPTS = self.dstPTS[1:]

        return len(self.dstPTS) > 0

    def _constrainRawDstDurations(self, dstDurs: np.ndarray) -> np.ndarray:
        minimalDur = Real(1/self.maxFPS)
        lastNZ = None
        for i, dur in enumerate(dstDurs):
            if dur <= 0 or dur >= minimalDur:
                continue
            if lastNZ is None:
                lastNZ = i
            elif dur < minimalDur:
                dstDurs[lastNZ] += dur
                dstDurs[i] = Real(0)
                if dstDurs[lastNZ] >= minimalDur:
                    # move the surplus to the current frame
                    surplus = dstDurs[lastNZ] - minimalDur
                    dstDurs[lastNZ] = minimalDur
                    dstDurs[i] = surplus
                    lastNZ = i if surplus > 0.0 else None
        # introduce a "small" desync: (0, minimalDur/2> in order to stay true to the max fps
        if lastNZ is not None:
            assert dstDurs[lastNZ] < minimalDur
            dstDurs[lastNZ] = np.round(dstDurs[lastNZ] / minimalDur) * minimalDur
        return dstDurs if np.sum(dstDurs) > 0 else np.array([])

    def decodeEditEncode(self, srcPacket: av.Packet) -> List[av.Packet]:
        for frame in srcPacket.decode():
            frIdx = self.frIdx
            self.frIdx += 1
            self.isDone = (self.frIdx == len(self.dstPTS))
            if self.dstPTS[frIdx] != DROP_FRAME_PTS:
                frame.pts = int(round(self.dstPTS[frIdx] / frame.time_base))
                frame.pict_type = av.video.frame.PictureType.NONE
                yield self.dstStream.encode(frame)


class AudCtx(EditCtx):
    class Workspace:
        @staticmethod
        def createWorkspaces(srcDurs: np.ndarray, dstDurs: np.ndarray,
                             wsRange: Tuple[int, int]) -> List['AudCtx.Workspace']:
            """Creates workspaces for a specified range of frames. The range should already include
            padding frames.

            Args:
                srcDurs (np.ndarray): Durations of the original frames
                dstDurs (np.ndarray): Durations of the edited frames
                wsRange (Tuple[int, int]): Range of frames, including the padding frames

            Returns:
                List['AudCtx.Workspace']: Workspaces for the specified range
            """
            wsRanges = AudCtx.Workspace.splitWsRange(srcDurs, wsRange)
            assert len(wsRanges) > 0

            workspaces = []
            if len(wsRanges) == 1:
                workspaces.append(AudCtx.Workspace(srcDurs, dstDurs, wsRange, True, True))
            else:
                workspaces.append(AudCtx.Workspace(srcDurs, dstDurs, wsRanges[0], True, False))
                for subWsRange in wsRanges[1:-1]:
                    workspaces.append(AudCtx.Workspace(srcDurs, dstDurs, subWsRange, False, False))
                workspaces.append(AudCtx.Workspace(srcDurs, dstDurs, wsRanges[-1], False, True))
                for ws, nextWs in zip(workspaces[:-1], workspaces[1:]):
                    ws.nextWorkspace = nextWs
            return workspaces

        @staticmethod
        def splitWsRange(srcDurs: np.ndarray, wsRange: Tuple[int, int]) -> List[Tuple[int, int]]:
            assert wsRange[0] < wsRange[1]
            start, end = wsRange

            wsRanges = []
            rangeSamples = np.sum(srcDurs[start:end])
            for targetWsSize in intLinspaceStepsByLimit(0, rangeSamples, AUD_WS_SIZE_MAX):
                wsSize = 0
                for frIdx in range(start, end):
                    wsSize += srcDurs[frIdx]
                    if wsSize >= targetWsSize:
                        break
                wsRanges.append((start, frIdx + 1))
                start = frIdx + 1
                if start == end:
                    break
            return wsRanges

        @staticmethod
        def modifyWsRange(durs: np.ndarray, wsRange: Tuple[int, int],
                          changes: Tuple[int, int]) -> Tuple[int, int]:
            """Expands or contracts the range of a workspace by a specified number of samples for
            each side. This operates on frames, so the number of added/removed samples will be equal
            to or greater (less only on ends) than specified (which would be 2*`samples`).

            Args:
                durs: (np.ndarray): Array with durations of frames
                wsRange (int): Range of a workspace
                changes (Tuple[int, int]): Number of samples to add to or remove from each side of
                the range: (leftSide, rightSide)

            Returns:
                Tuple[int, int]: Modified range
            """
            assert wsRange[0] < wsRange[1]
            wsRangeInclusive = (wsRange[0], wsRange[1] - 1)
            directions = np.sign(changes) * np.array([-1, 1])
            starts = np.array(wsRangeInclusive) + directions*(np.sign(changes) > 0)

            modified = [*wsRange]
            for sideIdx, direction in enumerate(directions):
                targetSamples = changes[sideIdx]
                start = starts[sideIdx]
                if targetSamples > 0:
                    end = len(durs) if direction == 1 else -1
                elif targetSamples < 0:
                    end = wsRange[1] if direction == 1 else wsRange[0] - 1
                if targetSamples == 0 or start == end:
                    continue

                samples = 0
                for frIdx in range(start, end, direction):
                    samples += durs[frIdx]
                    if samples >= abs(targetSamples):
                        break
                modified[sideIdx] = frIdx + (direction > 0)
            modified = (min(modified), max(modified))
            assert modified[0] <= modified[1]
            return modified

        def __init__(self, srcDurs: np.ndarray, dstDurs: np.ndarray, wsRange: Tuple[int, int],
                     encodeLeftPad: bool, encodeRightPad: bool):
            """Creates a workspace for a specified range of frames.

            Args:
                srcDurs (np.ndarray): Durations of all the original frames
                dstDurs (np.ndarray): Durations of all the edited frames
                wsRange (Tuple[int, int]): Range of frames
                encodeLeftPad (bool): Should this workspace encode the left padding during
                editing. When True, wsRange should already include the left padding
                encodeRightPad (bool): Should this workspace encode the right padding during
                editing. When True, wsRange should already include the right padding
            """
            leftChange = (-1 if encodeLeftPad else 1) * AUD_WS_PAD_SIZE
            rightChange = (-1 if encodeRightPad else 1) * AUD_WS_PAD_SIZE
            encWsRange = AudCtx.Workspace.modifyWsRange(srcDurs, wsRange,
                                                        (leftChange, rightChange))
            leftPad = min(wsRange[0], encWsRange[0]), max(wsRange[0], encWsRange[0])
            rightPad = min(wsRange[1], encWsRange[1]), max(wsRange[1], encWsRange[1])

            self.beg = leftPad[0]
            self.end = rightPad[1]
            self.leftPad = (leftPad[0] - self.beg, leftPad[1] - self.beg)
            self.rightPad = (rightPad[0] - self.beg, rightPad[1] - self.beg)
            self.encodeLeftPad = encodeLeftPad
            self.encodeRightPad = encodeRightPad
            self.dstDurs = dstDurs[self.beg:self.end]
            self.frameCache = []
            self.frameTemplate = None  # first pushed frame
            self.nextWorkspace = None

            assert self.leftPad[0] == 0 and self.rightPad[1] == self.dstDurs.shape[0]

        def pushFrame(self, frIdx: int, frame: av.AudioFrame, frameData: np.ndarray) -> bool:
            assert frIdx < self.end
            if frIdx < self.beg:
                return False

            if len(self.frameCache) == 0:
                self.frameTemplate = frame
            if self.dstDurs[frIdx-self.beg] > 0:
                frameData = frame.to_ndarray() if frameData is None else frameData
            else:
                frameData = np.ndarray(shape=(0, 0))
            self.frameCache.append((frIdx, frameData))
            return True

        def pullFrame(self) -> av.AudioFrame:
            if len(self.frameCache) == 0 or self.frameCache[-1][0] < (self.end - 1):
                return None
            assert (len({idx for idx, frame in self.frameCache}) == len(self.frameCache) and
                    len(self.frameCache) == (self.end - self.beg))

            dstLPadLen = np.sum(self.dstDurs[:self.leftPad[1]])
            dstRPadLen = np.sum(self.dstDurs[self.rightPad[0]:])

            # add virtual frames to separate padding from real frames
            if dstLPadLen > 0:
                self.frameCache.append((self.beg+self.leftPad[1]-0.5, np.ndarray((0, 0))))
            if dstRPadLen > 0:
                self.frameCache.append((self.beg+self.rightPad[0]-0.25, np.ndarray((0, 0))))
            self.frameCache = sorted(self.frameCache, key=lambda kv: kv[0])

            srcDurs = np.array([frame.shape[1] for idx, frame in self.frameCache])
            dstDurs = np.concatenate([self.dstDurs[self.leftPad[0]:self.leftPad[1]],
                                      [0],  # virtual, the end of the left padding
                                      self.dstDurs[self.leftPad[1]:self.rightPad[0]],
                                      [0],  # virtual, the begining of the right padding
                                      self.dstDurs[self.rightPad[0]:self.rightPad[1]]])

            # # speed of frames next to deleted ones is unchanged (but they are trimmed)
            # for beg, end in rangesOfTruth(srcDurs == 0):
            #     left, right = max(beg-1, 0), min(end, len(srcDurs)-1)
            #     # the right side of the left frame and the left side of the right frame are trimmed
            #     srcDurs[left] = dstDurs[left]
            #     srcDurs[right] = dstDurs[right]
            #     self.frameCache[left][1] = self.frameCache[left][1][:, :srcDurs[left]]
            #     self.frameCache[right][1] = self.frameCache[right][1][:, -srcDurs[right]:]

            signal = np.concatenate([f for i, f in self.frameCache if f.shape[1] > 0], axis=1)
            leftPad = signal[:, :dstLPadLen]
            rightPad = signal[:, (signal.shape[1]-dstRPadLen):]

            # Time-Scale Modification
            # padding is included here for the calculations but its modifications are discarded
            srcSP = np.concatenate([[0], np.cumsum(srcDurs[srcDurs > 0])])
            dstSP = np.concatenate([[0], np.cumsum(dstDurs[dstDurs > 0])])
            assert srcSP[-1] == signal.shape[1]
            assert srcSP.shape == dstSP.shape
            if not np.array_equal(srcSP, dstSP):
                srcSP[-1] -= 1
                dstSP[-1] -= 1
                signal = tsm.phase_vocoder(signal, np.array([srcSP, dstSP]), AUD_WIN_TYPE,
                                           AUD_WIN_SIZE, AUD_HOP_SIZE, phase_lock=True)

            # discard modifications made to padding
            signal[:, :dstLPadLen] = leftPad
            signal[:, (signal.shape[1]-dstRPadLen):] = rightPad

            # soften transitions between frames next to deleted (or virtual) ones
            fullDstSP = np.cumsum(dstDurs)  # includes deleted frames
            for beg, end in rangesOfTruth(dstDurs == 0):
                if not (0 < beg < end < len(dstDurs)):
                    continue
                winSize = min(64, fullDstSP[beg])
                winSize = min(winSize, fullDstSP[-1] - fullDstSP[beg])
                window = -np.hamming(winSize*2) + 1
                signal[:, fullDstSP[beg] - winSize:fullDstSP[beg] + winSize] *= window

            if not self.encodeLeftPad and dstLPadLen > 0:
                signal = signal[:, dstLPadLen:]
            if not self.encodeRightPad and dstRPadLen > 0:
                signal = signal[:, :(signal.shape[1]-dstRPadLen)]
                # if encodeRightPad == False, there must be a next sub-workspace
                # transfer common frames to the next workspace
                assert self.nextWorkspace is not None and len(self.nextWorkspace.frameCache) == 0
                for f in reversed(self.frameCache):
                    if f[0] != int(f[0]):  # discard virtual
                        continue
                    if not (self.nextWorkspace.beg <= f[0] < self.end):
                        break
                    self.nextWorkspace.frameCache.append(f)
                self.nextWorkspace.frameCache.reverse()
                self.nextWorkspace.frameTemplate = self.frameTemplate

            dtype = self.frameCache[0][1].dtype
            return AudCtx._createFrame(self.frameTemplate, signal.astype(dtype))

    @staticmethod
    def _createFrame(template, data):
        frame = av.AudioFrame.from_ndarray(data, template.format.name,
                                           template.layout.name)
        frame.sample_rate = template.sample_rate
        frame.time_base = template.time_base
        frame.pts = None
        return frame

    def __init__(self, srcStream, dstStream):
        super().__init__(srcStream, dstStream)
        self.workspaces = []  # sorted (by pts) workspaces
        self.dstVFrames = []
        self.dstFramesNo = None
        self.past = None
        self.frIdx = 0

    def prepareTimelineEdits(self, ranges: List[Range]):
        # return False
        srcDurs, firstVirtual = self._prepareSrcDurs()
        if len(srcDurs) == 0:
            return False

        if len(ranges) > 0:
            dstDurs = self._prepareRawDstDurations(srcDurs, ranges)
            srcDurs = srcDurs[:len(dstDurs)]

            # convert from seconds to samples
            pts = (np.cumsum(srcDurs) * self.srcStream.sample_rate).astype(int)
            srcDurs[1:] = (pts[1:] - pts[:-1])
            srcDurs[0] = pts[0]
            srcDurs = srcDurs.astype(int)
            pts = np.round(np.cumsum(dstDurs) * self.srcStream.sample_rate).astype(int)
            dstDurs[1:] = (pts[1:] - pts[:-1])
            dstDurs[0] = pts[0]
            dstDurs = dstDurs.astype(int)
            self.dstFramesNo = len(dstDurs)

            if len(dstDurs) == 0 or np.sum(dstDurs) == 0:
                return False

            # PyAV expects the audio streams to start at 0, so the virtual first frame (if present)
            # will be actually encoded - if its too big, it must be split into many frames
            if len(dstDurs) > 0:
                if firstVirtual:
                    self.dstVFrames = intLinspaceStepsByLimit(0, dstDurs[0], AUD_VFRAME_SIZE_MAX)
                    # srcVFrames = intLinspaceStepsByNo(0, srcDurs[0], len(self.dstVFrames))
                    dstDurs = np.concatenate([self.dstVFrames, dstDurs[1:]])
                    srcDurs = np.concatenate([self.dstVFrames, srcDurs[1:]])
                self.workspaces = self._prepareWorkspaces(srcDurs, dstDurs)
        # there must be at lease one valid frame
        return len(dstDurs) > 0

    def _prepareWorkspaces(self, srcDurs: np.ndarray, dstDurs: np.ndarray):
        toEdit = srcDurs[:len(dstDurs)] != dstDurs
        srcDurs[dstDurs == 0] = 0  # deleted frames will not be included in workspaces

        for beg, end in rangesOfTruth(toEdit):
            # extend the ranges by padding (this might merge adjacent ranges into one)
            extBeg, extEnd = AudCtx.Workspace.modifyWsRange(srcDurs, (beg, end),
                                                            (AUD_WS_PAD_SIZE, AUD_WS_PAD_SIZE))
            assert extBeg <= extBeg < extEnd <= extEnd
            srcSamples = np.sum(srcDurs[extBeg:extEnd])
            dstSamples = np.sum(dstDurs[extBeg:extEnd])
            if abs(srcSamples-dstSamples) <= 2:
                dstDurs[beg:end] = srcDurs[beg:end]
                toEdit[beg:end] = False
            else:
                toEdit[extBeg:beg] = True
                toEdit[end:extEnd] = True

        return [ws for wsRange in rangesOfTruth(toEdit) for ws in
                AudCtx.Workspace.createWorkspaces(srcDurs, dstDurs, wsRange)]

    def decodeEditEncode(self, srcPacket: av.Packet) -> List[av.Packet]:
        for frIdx, frame, frameData in self._decode(srcPacket):
            self.isDone = frIdx + 1 >= self.dstFramesNo
            if len(self.workspaces) > 0 and self.workspaces[0].pushFrame(frIdx, frame, frameData):
                while len(self.workspaces) > 0:
                    frame = self.workspaces[0].pullFrame()
                    if frame is not None:
                        self.workspaces.pop(0)
                        yield self.dstStream.encode(frame)
                    else:
                        assert not self.isDone
                        break
            else:
                yield self.dstStream.encode(frame)

    def _decode(self, srcPacket: av.Packet) -> Tuple[int, av.AudioFrame, np.ndarray]:
        frames = srcPacket.decode()
        if self.frIdx < len(self.dstVFrames) and len(frames) > 0:
            srcFrame = frames[0]
            srcData = srcFrame.to_ndarray()
            while self.frIdx < len(self.dstVFrames):
                frIdx = self.frIdx
                self.frIdx += 1

                data = np.zeros((srcData.shape[0], self.dstVFrames[frIdx]), dtype=srcData.dtype)
                vFrame = AudCtx._createFrame(srcFrame, data)
                yield frIdx, vFrame, data

        for frame in frames:
            frIdx = self.frIdx
            self.frIdx += 1

            frame.pts = None
            yield frIdx, frame, None


class Editor:
    @staticmethod
    def fromJSON(jsonSpecs: dict, logger: Logger) -> 'Editor':
        """Constructs an Editor from a dictionary with edit specifications.
        Ensures correct types of specs' values

        Returns:
            Editor: Editor instance prepared to edit
        """
        editor = Editor(logger)
        for identifier, specs in jsonSpecs.get('specs', {}).items():
            identifier = identifier.lower()
            if identifier in [VIDEO_STREAM_TYPE, AUDIO_STREAM_TYPE]:
                especs = editor.specs.setdefault(identifier, {})  # stream type
            elif identifier.isnumeric():
                especs = editor.specs.setdefault(int(identifier), {})  # stream idx
            else:
                logger.warning(f'Skipping unrecognized stream identifier: {identifier}')
                continue
            for key, value in specs.items():
                if key == 'codec':
                    especs[key] = str(value)
                elif key == 'codec_options':
                    especs[key] = value
                    for optionKey, optionValue in value.items():
                        value[optionKey] = str(optionValue)
                elif key == 'bitrate':
                    especs[key] = int(value)  # bitrate in b/s
                    if especs[key] <= 0:
                        raise ValueError('"bitrate" must be a positive number')
                elif key == 'resolution':
                    especs[key] = [int(dim) for dim in value]  # [width, height]
                    if especs[key][0] * especs[key][1] <= 0:
                        raise ValueError('"resolution" must consist of positive numbers')
                elif key == 'maxfps':
                    especs[key] = float(value)
                    if especs[key] <= 0:
                        raise ValueError('"maxfps" must be a positive number')
                elif key == 'samplerate':
                    especs[key] = int(value)
                    if especs[key] <= 0:
                        raise ValueError('"samplerate" must be a positive number')
                elif key == 'mono':
                    especs[key] = bool(value)
                else:
                    logger.warning(f'Skipping unrecognized option: {key}:')
        return editor

    @staticmethod
    def parseJSONRanges(jsonRanges: List[List[str]]) -> List[Range]:
        ranges = [Range(*r) for r in jsonRanges]
        if len(jsonRanges) > 0:
            for r1, r2 in zip(ranges[:-1], ranges[1:]):
                if not (0 <= r1.beg < r1.end <= r2.beg):
                    raise ValueError('Ranges must be sorted, mutually exclusive, and of length > 0')
            if not (0 <= ranges[-1].beg < ranges[-1].end):
                raise ValueError('Ranges must be of length > 0')
        return ranges

    def __init__(self, logger):
        self.logger = logger
        self.specs = {}

    def edit(self, srcPath: Path, ranges: List[Range], dstPath: Path):
        self.logger.info(f'Started editing: "{srcPath}"')
        srcPath = str(Path(srcPath).resolve())
        dstPath = str(Path(dstPath).resolve())
        source = av.open(str(srcPath))
        dest, ctxMap = self.prepareDestination(source, dstPath)

        # find dts of first packets (to know which one to seek for begining)
        firstPkts = {}
        for stream in source.streams:
            seekBeginning(stream)
            firstPkt = next(source.demux(stream))
            firstPkts[stream.index] = Real(firstPkt.dts * firstPkt.time_base)
        streamsOrdered = [k for k, v in sorted(firstPkts.items(), key=lambda kv: kv[1])]

        # prepare contexts of streams for editing
        validStreams = {}
        for idx, ctx in ctxMap.items():
            if ctx.prepareTimelineEdits(ranges):
                validStreams[idx] = ctx.srcStream

        # edit
        if len(validStreams) > 0:
            firstStream = [idx for idx in streamsOrdered if idx in validStreams][0]
            seekBeginning(source.streams[firstStream])
            for srcPacket in source.demux(list(validStreams.values())):
                ctx = ctxMap[srcPacket.stream.index]
                assert srcPacket.stream is ctx.srcStream
                if ctx.isDone:
                    continue
                for dstPacket in ctx.decodeEditEncode(srcPacket):
                    dest.mux(dstPacket)
                if all(ctxMap.values()):  # early stop when all are done
                    break
            for dstStream in dest.streams:
                dest.mux(dstStream.encode())
        else:
            self.logger.warning(f'Editing: "{srcPath}" resulted in an empty recording')
        self.logger.info(f'Finished editing: "{srcPath}" -> "{dstPath}"')

        dest.close()
        source.close()

    def prepareDestination(self, source, dstPath):
        dst = av.open(dstPath, mode='w')
        ctxMap = {}

        validStreams = []
        for stream in source.streams:
            if stream.type in SUPPORTED_STREAM_TYPES:
                if stream.codec_context is not None:
                    validStreams.append(stream)
                else:
                    self.logger.warning(f'Skipping #{stream.index} stream (no decoder available)')
            else:
                self.logger.warning(
                    f'Skipping #{stream.index} stream ({stream.type} not supported)')

        for srcStream in validStreams:
            # stream-specific settings take precedence over type-specific settings
            specs = self.specs.get(srcStream.type, {})
            specs.update(self.specs.get(srcStream.index, {}))

            if srcStream.type == VIDEO_STREAM_TYPE:
                codec = specs.get('codec', srcStream.codec_context.name)
                codecOptions = specs.get('codec_options', srcStream.codec_context.options)
                bitrate = specs.get('bitrate', srcStream.bit_rate)
                resolution = specs.get('resolution', [srcStream.width, srcStream.height])
                maxFPS = Fraction(specs.get('maxfps', srcStream.guessed_rate))

                dstStream = dst.add_stream(codec_name=codec, options=codecOptions)
                dstStream.codec_context.time_base = Fraction(1, 60000)
                dstStream.time_base = Fraction(1, 60000)  # might not work
                dstStream.pix_fmt = srcStream.pix_fmt
                dstStream.bit_rate = bitrate
                dstStream.width, dstStream.height = resolution
                ctxMap[srcStream.index] = VidCtx(srcStream, dstStream, maxFPS)

            elif srcStream.type == AUDIO_STREAM_TYPE:
                codec = specs.get('codec', srcStream.codec_context.name)
                codecOptions = specs.get('codec_options', srcStream.codec_context.options)
                bitrate = specs.get('bitrate', srcStream.bit_rate)
                samplerate = specs.get('samplerate', srcStream.sample_rate)
                channels = 1 if specs.get('mono', False) else srcStream.channels

                dstStream = dst.add_stream(codec_name=codec, rate=samplerate)
                dstStream.options = codecOptions
                dstStream.bit_rate = bitrate
                dstStream.channels = channels
                ctxMap[srcStream.index] = AudCtx(srcStream, dstStream)

            srcStream.thread_type = 'AUTO'
            dstStream.thread_type = 'AUTO'

        dst.start_encoding()
        for vidCtx in [c for c in ctxMap.values() if isinstance(c, VidCtx)]:
            possibleFPS = 1 / vidCtx.dstStream.time_base
            if possibleFPS < vidCtx.maxFPS:
                self.logger.warning(f'Low time base resolution of #{dstStream.index} video stream - '
                                    f'maxfps must be limited from {vidCtx.maxFPS} to {possibleFPS}')
                vidCtx.maxFPS = possibleFPS

        return dst, ctxMap

    def exportJSON(self, ranges: List[Range], path):
        with open(path, 'w') as fp:
            specsDict = {'specs': self.specs,
                         'ranges': [[r.beg, r.end, r.multi] for r in ranges]}
            json.dump(specsDict, fp)


DESCRIPTION = 'Edits recordings according to the specification'
ARG_SRC = 'src'
ARG_DST = 'dst'
DEFAULT_ARGS = {}


def setupArgParser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Creates CLI argument parser for editor submodule

    Returns:
        argparse.ArgumentParser: Argument parser of this submodule
    """
    parser.description = DESCRIPTION
    parser.add_argument(ARG_SRC,
                        help='Path of the recording to edit',
                        type=Path, action='store')
    parser.add_argument(ARG_DST,
                        help='Path of the edited recording',
                        type=Path, action='store')
    parser.set_defaults(run=runSubmodule)
    # TODO
    return parser


def runSubmodule(args: object, logger: Logger) -> None:
    # TODO
    args = args.__dict__
    with open('test.json', 'r') as fp:
        jsonSpecs = json.load(fp)
    editor = Editor.fromJSON(jsonSpecs, logger=logger)
    ranges = Editor.parseJSONRanges(jsonSpecs['ranges'])
    # editor.exportJSON(ranges, 'test2.json')
    editor.edit(args[ARG_SRC], ranges, args[ARG_DST])
