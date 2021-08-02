import av
import json
import logging
import argparse
import numpy as np
import pytsmod as tsm
from typing import List
from fractions import Fraction
from pathlib import Path

Real = np.double
Int = np.int64

VIDEO_STREAM_TYPE = 'video'
AUDIO_STREAM_TYPE = 'audio'
SUPPORTED_STREAM_TYPES = [VIDEO_STREAM_TYPE, AUDIO_STREAM_TYPE]
DROP_FRAME_PTS = -1
AUDIO_WINDOW = 30  # in seconds


def seekBeginning(container):
    container.seek(np.iinfo(Int).min, backward=True, any_frame=False)


class Editor:
    DESCRIPTION = 'Edits recordings according to the specification'
    DEFAULT_ARGS = {}

    class VidCtx:
        def __init__(self, srcStream, dstStream, maxFPS):
            self.srcStream = srcStream
            self.dstStream = dstStream
            self.maxFPS = maxFPS
            self.dstFrPTS = None

        def captureEditData(self):
            assert self.dstFrPTS is None
            dstFrPTS = []
            # collect pts of each packet
            seekBeginning(self.srcStream.container)
            for packet in self.srcStream.container.demux(self.srcStream):
                if packet.dts is not None:
                    dstFrPTS.append(packet.pts)

            # sort pts and convert to seconds
            dstFrPTS = sorted(set(dstFrPTS))
            dstFrPTS.append(dstFrPTS[-1])  # add dummy end frame
            dstFrPTS = np.array(dstFrPTS, dtype=Real) * Real(self.srcStream.time_base)
            dstFrPTS[-1] += np.mean(dstFrPTS[1:] - dstFrPTS[:-1])  # fix end
            self.dstFrPTS = dstFrPTS

        def applyEditRanges(self, ranges: List[List[Real]]):
            if len(ranges) == 0:
                return True
            streamStart = self.srcStream.start_time * self.srcStream.time_base

            # prepare modified durations of frames
            durs = []
            # move to the relevant ranges
            for eRangeIdx, eRange in enumerate(ranges):
                if eRange[1] > streamStart:
                    eRange[0] = max(eRange[0], streamStart)
                    break
            if eRange[1] <= streamStart:
                return True # no edits to make

            for pts, nextPTS in zip(self.dstFrPTS[:-1], self.dstFrPTS[1:]):
                oldDur = nextPTS - pts
                newDur = 0

                while eRange[0] < nextPTS and eRangeIdx < len(ranges):
                    assert eRange[0] >= pts

                    clampedEnd = min(eRange[1], nextPTS)
                    partDur = clampedEnd - eRange[0]
                    assert partDur <= oldDur

                    oldDur -= partDur
                    newDur += partDur * eRange[2]
                    if clampedEnd < eRange[1]:
                        eRange[0] = nextPTS
                        break
                    else:
                        eRangeIdx += 1
                        eRange = ranges[eRangeIdx]

                newDur += oldDur

                # early stop when the recording has trimmed end
                if newDur == 0 and eRange[1] >= self.dstFrPTS[-1]:
                    break
                durs.append(newDur)

            durs = np.array(durs)

            # collapse too small frames
            minimalDur = Real(1/self.maxFPS)
            lastNZ = -1
            for i, dur in enumerate(durs):
                if dur <= 0 or dur >= minimalDur:
                    continue
                if lastNZ < 0:
                    lastNZ = i
                elif dur < minimalDur:
                    durs[lastNZ] += dur
                    durs[i] = 0
                    if durs[lastNZ] >= minimalDur:
                        # move the surplus to the current frame
                        surplus = durs[lastNZ] - minimalDur
                        durs[lastNZ] = minimalDur
                        durs[i] = surplus
                        lastNZ = i if surplus > 0.0 else -1
            # introduce a "small" desync: (0, minimalDur/2> in order to stay true to the max fps
            if lastNZ > 0:
                assert durs[lastNZ] < minimalDur
                durs[lastNZ] = round(durs[lastNZ] / minimalDur) * minimalDur

            # generate final pts
            self.dstFrPTS = np.cumsum(durs)
            self.dstFrPTS[durs <= 0] = DROP_FRAME_PTS

            return self.dstFrPTS[-1] > 0

    class AudCtx:
        def __init__(self, srcStream, dstStream) -> None:
            self.srcStream = srcStream
            self.dstStream = dstStream
            self.srcDuration = None
            self.timePoints = []
            self.audioFrames = []

        def captureEditData(self):
            assert self.srcDuration is None
            self.srcDuration = self.srcStream.duration
            if self.srcDuration is None:
                endPTS = 0
                seekBeginning(self.srcStream.container)
                for packet in self.srcStream.container.demux(self.srcStream):
                    if packet.dts is None:
                        continue
                    endPTS = max(endPTS, packet.pts + packet.duration)
                self.srcDuration = endPTS
            self.srcDuration *= self.srcStream.time_base

        def applyEditRanges(self, ranges: List[List[Real]]):
            if len(ranges) == 0:
                return True
            streamStart = self.srcStream.start_time * self.srcStream.time_base
            timePoints = [[Real(streamStart*self.dstStream.sample_rate)],
                          [Real(streamStart*self.dstStream.sample_rate)]]
            for b, e, m in ranges:
                if e < streamStart:
                    continue
                b = max(b, streamStart) * self.dstStream.sample_rate
                e = min(e, self.srcDuration) * self.dstStream.sample_rate
                betweenPrev = b - timePoints[0][-1]
                dur = e - b
                if betweenPrev > 0:
                    timePoints[0].append(Real(b))
                    timePoints[1].append(Real(timePoints[1][-1] + betweenPrev))
                timePoints[0].append(Real(e))
                timePoints[1].append(Real(timePoints[1][-1] + dur*m))
            rest = self.srcDuration * self.dstStream.sample_rate - e
            if rest > 0:
                timePoints[0].append(Real(e+rest))
                timePoints[1].append(Real(timePoints[1][-1] + rest))
            self.timePoints = timePoints
            return timePoints[1][-1] - timePoints[1][0] > 0

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.specs = {}

    def edit(self, srcPath, ranges, dstPath):
        self.logger.info(f'Editing: "{srcPath}"')
        source = av.open(srcPath)
        dest, ctxMap = self.prepareDestination(source, dstPath)

        validStreams = {}
        for idx, ctx in ctxMap.items():
            ctx.captureEditData()
            if ctx.applyEditRanges(ranges):
                validStreams[idx] = ctx.srcStream

        if len(validStreams) > 0:
            seekBeginning(source)
            for srcPacket in source.demux(list(validStreams.values())):
                if srcPacket.stream_index not in ctxMap or srcPacket.dts is None:
                    if len(ctxMap) == 0:
                        break
                    continue
                ctx = ctxMap[srcPacket.stream_index]
                assert srcPacket.stream is ctx.srcStream
                for frame in srcPacket.decode():
                    if srcPacket.stream.type == VIDEO_STREAM_TYPE:
                        if frame.index == len(ctx.dstFrPTS) - 1:
                            del ctxMap[srcPacket.stream_index]
                        if ctx.dstFrPTS[frame.index] == DROP_FRAME_PTS:
                            continue
                        frame.pts = Int(ctx.dstFrPTS[frame.index] / Real(frame.time_base))
                        frame.pict_type = av.video.frame.PictureType.NONE
                        dest.mux(ctx.dstStream.encode(frame))
                    elif srcPacket.stream.type == AUDIO_STREAM_TYPE:
                        ...

            for dstStream in dest.streams:
                dest.mux(dstStream.encode())
        else:
            self.logger.warning(f'Editing {srcPath} resulted in an empty recording')
        self.logger.info(f'Finished editing: {srcPath}')

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
                ctxMap[srcStream.index] = Editor.VidCtx(srcStream, dstStream, maxFPS)

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
                ctxMap[srcStream.index] = Editor.AudCtx(srcStream, dstStream)

            srcStream.thread_type = 'AUTO'
            dstStream.thread_type = 'AUTO'

        dst.start_encoding()
        for vidCtx in [c for c in ctxMap.values() if isinstance(c, Editor.VidCtx)]:
            possibleFPS = 1 / vidCtx.dstStream.time_base
            if possibleFPS < vidCtx.maxFPS:
                self.logger.warning(f'Low time base resolution of #{dstStream.index} video stream - '
                                    f'maxfps must be limited from {vidCtx.maxFPS} to {possibleFPS}')
                vidCtx.maxFPS = possibleFPS

        return dst, ctxMap

    def toJSON(self, ranges, path) -> None:
        with open(path, 'w') as fp:
            specsDict = {'specs': self.specs,
                         'ranges': ranges}
            json.dump(specsDict, fp)

    @staticmethod
    def fromJSON(jsonSpecs: dict, logger: logging.Logger) -> 'Editor':
        """Constructs an Editor from a dictionary with edit specifications.
        Ensures correct types of specs' values

        Returns:
            [Editor]: Editor instance prepared to edit
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
                    especs[key] = bool(value.lower() in ['yes', 'true', '1'])
                else:
                    logger.warning(f'Skipping unrecognized specification: {key}:')
        return editor

    @staticmethod
    def parseJSONRanges(jsonRanges: List[List[str]]):
        ranges = [[Real(b), Real(e), Real(m)] for (b, e, m) in jsonRanges]
        if len(jsonRanges) > 0:
            for (b, e, _), (b_1, _, _) in zip(ranges[:-1], ranges[1:]):
                if not (b < e <= b_1):
                    raise ValueError('Ranges must be sorted, mutually exclusive, and of length > 0')
            if not (ranges[-1][0] < ranges[-1][1]):
                raise ValueError('Ranges must be of length > 0')
        return ranges

    @staticmethod
    def setupArgParser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Creates CLI argument parser for editor submodule

        Returns:
            argparse.ArgumentParser: Argument parser of this submodule
        """
        parser.description = Editor.DESCRIPTION
        parser.add_argument('src',
                            help=f'Path to the recordding for editing',
                            type=Path, action='store')
        parser.add_argument('dst',
                            help=f'Destination directory for edited recording',
                            type=Path, action='store')
        # TODO
        return parser
