import av
import json
import argparse
import numpy as np
import pytsmod as tsm

from typing import List, Tuple
from fractions import Fraction
from logging import Logger
from pathlib import Path

from .utils import Real, ranges_of_truth, int_linspace_steps_by_limit

VIDEO_STREAM_TYPE = 'video'
AUDIO_STREAM_TYPE = 'audio'
SUPPORTED_STREAM_TYPES = [VIDEO_STREAM_TYPE, AUDIO_STREAM_TYPE]

DROP_FRAME_PTS = -1

AUD_WIN_TYPE = 'hann'
AUD_WIN_SIZE = 1024
AUD_HOP_SIZE = int(AUD_WIN_SIZE / 4)
AUD_WS_PAD_SIZE = 512
AUD_WS_SIZE_MAX = AUD_WS_PAD_SIZE + 1000 * AUD_WIN_SIZE + AUD_WS_PAD_SIZE
AUD_VFRAME_SIZE_MAX = 10 * 1024


def seek_beginning(stream):
  stream.container.seek(np.iinfo(np.int32).min, stream=stream, backward=True, any_frame=False)


class Range:

  @staticmethod
  def from_numpy(arr: np.ndarray) -> List[np.ndarray]:
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

  def __init__(self, src_stream, dst_stream):
    self.src_stream = src_stream
    self.dst_stream = dst_stream
    self.is_done = False

  def __bool__(self):
    return self.is_done

  def _prepare_src_durs(self) -> np.ndarray:
    """Collects pts of each frame """
    seek_beginning(self.src_stream)
    pts = []
    for packet in self.src_stream.container.demux(self.src_stream):
      if packet.dts is not None:
        pts.append(packet.pts)
    if len(pts) < 2:  # there must be at least 2 frames
      return [], False

    pts = sorted(set(pts))
    # convert to seconds and set the end frame pts
    pts = np.array(pts, dtype=Real) * Real(self.src_stream.time_base)
    durs = pts[1:] - pts[:-1]
    # add virtual empty first frame that lasts until the first real frame
    virtual_first = ([pts[0]] if pts[0] > 0 else [])
    # assume duration of the last frame is equal to the average duration of real frames
    return (np.concatenate([virtual_first, durs, [np.mean(durs)]]), len(virtual_first) == 1)

  def _prepare_raw_dst_durations(self, src_durs: np.ndarray, ranges: List[Range]) -> np.ndarray:
    dst_durs = []
    r_idx = 0
    fr_end = Real(0)
    src_stream_end = np.sum(src_durs, dtype=Real)
    for old_dur in src_durs:
      fr_beg = fr_end
      fr_end = fr_beg + old_dur
      new_dur = 0

      while r_idx < len(ranges) and fr_beg < ranges[r_idx].end and fr_end > ranges[r_idx].beg:
        clamped_beg = np.max([ranges[r_idx].beg, fr_beg])
        clamped_end = np.min([ranges[r_idx].end, fr_end])
        part_dur = clamped_end - clamped_beg
        assert np.round(old_dur - part_dur, 12) >= 0

        old_dur -= part_dur
        new_dur += part_dur * ranges[r_idx].multi
        if clamped_end < ranges[r_idx].end:
          # this range extends beyond this frame (was clamped), move to the next frame
          break
        else:
          # this range ends on this frame (wasn't clamped), move to the next range
          r_idx += 1
      new_dur += old_dur

      # early stop when the recording has trimmed end
      if new_dur == 0 and r_idx < len(ranges) and ranges[r_idx].end >= src_stream_end:
        break
      dst_durs.append(new_dur)

    return np.array(dst_durs, dtype=Real) if len(dst_durs) >= 2 else np.array([])


class VidCtx(EditCtx):

  def __init__(self, src_stream, dst_stream, max_fps: float):
    super().__init__(src_stream, dst_stream)
    self.max_fps = max_fps
    self.fr_idx = 0
    self.dst_pts = None

  def prepare_timeline_edits(self, ranges: List[Range]) -> bool:
    durs, virtual_first = self._prepare_src_durs()
    if len(durs) == 0:
      return False

    if len(ranges) > 0:
      durs = self._prepare_raw_dst_durations(durs, ranges)
      durs = self._constrain_raw_dst_durations(durs)
      if len(durs) == 0 or np.sum(durs) == 0:
        return False

    self.dst_pts = np.concatenate([[0], np.cumsum(durs[:-1])])
    self.dst_pts[durs <= 0] = DROP_FRAME_PTS

    # if first frame is virtual, discard it
    if virtual_first:
      self.dst_pts = self.dst_pts[1:]

    return len(self.dst_pts) > 0

  def _constrain_raw_dst_durations(self, dst_durs: np.ndarray) -> np.ndarray:
    minimal_dur = Real(1 / self.max_fps)
    last_nz = None
    for i, dur in enumerate(dst_durs):
      if dur <= 0 or dur >= minimal_dur:
        continue
      if last_nz is None:
        last_nz = i
      elif dur < minimal_dur:
        dst_durs[last_nz] += dur
        dst_durs[i] = Real(0)
        if dst_durs[last_nz] >= minimal_dur:
          # move the surplus to the current frame
          surplus = dst_durs[last_nz] - minimal_dur
          dst_durs[last_nz] = minimal_dur
          dst_durs[i] = surplus
          last_nz = i if surplus > 0.0 else None
    # introduce a "small" desync: (0, minimal_dur/2> in order to stay true to the max fps
    if last_nz is not None:
      assert dst_durs[last_nz] < minimal_dur
      dst_durs[last_nz] = np.round(dst_durs[last_nz] / minimal_dur) * minimal_dur
    return dst_durs if np.sum(dst_durs) > 0 else np.array([])

  def decode_edit_encode(self, src_packet: av.Packet) -> List[av.Packet]:
    for frame in src_packet.decode():
      fr_idx = self.fr_idx
      self.fr_idx += 1
      self.is_done = (self.fr_idx == len(self.dst_pts))
      if self.dst_pts[fr_idx] != DROP_FRAME_PTS:
        frame.pts = int(round(self.dst_pts[fr_idx] / frame.time_base))
        frame.pict_type = av.video.frame.PictureType.NONE
        yield self.dst_stream.encode(frame)


class AudCtx(EditCtx):

  class Workspace:

    @staticmethod
    def create_workspaces(src_durs: np.ndarray, dst_durs: np.ndarray,
                          ws_range: Tuple[int, int]) -> List['AudCtx.Workspace']:
      """Creates workspaces for a specified range of frames. The range should already include
            padding frames.

            Args:
                src_durs (np.ndarray): Durations of the original frames
                dst_durs (np.ndarray): Durations of the edited frames
                ws_range (Tuple[int, int]): Range of frames, including the padding frames

            Returns:
                List['AudCtx.Workspace']: Workspaces for the specified range
            """
      ws_ranges = AudCtx.Workspace.split_ws_range(src_durs, ws_range)
      assert len(ws_ranges) > 0

      workspaces = []
      if len(ws_ranges) == 1:
        workspaces.append(AudCtx.Workspace(src_durs, dst_durs, ws_range, True, True))
      else:
        workspaces.append(AudCtx.Workspace(src_durs, dst_durs, ws_ranges[0], True, False))
        for sub_ws_range in ws_ranges[1:-1]:
          workspaces.append(AudCtx.Workspace(src_durs, dst_durs, sub_ws_range, False, False))
        workspaces.append(AudCtx.Workspace(src_durs, dst_durs, ws_ranges[-1], False, True))
        for ws, next_ws in zip(workspaces[:-1], workspaces[1:]):
          ws.next_workspace = next_ws
      return workspaces

    @staticmethod
    def split_ws_range(src_durs: np.ndarray, ws_range: Tuple[int, int]) -> List[Tuple[int, int]]:
      assert ws_range[0] < ws_range[1]
      start, end = ws_range

      ws_ranges = []
      range_samples = np.sum(src_durs[start:end])
      for target_ws_size in int_linspace_steps_by_limit(0, range_samples, AUD_WS_SIZE_MAX):
        ws_size = 0
        for fr_idx in range(start, end):
          ws_size += src_durs[fr_idx]
          if ws_size >= target_ws_size:
            break
        ws_ranges.append((start, fr_idx + 1))
        start = fr_idx + 1
        if start == end:
          break
      return ws_ranges

    @staticmethod
    def modify_ws_range(durs: np.ndarray, ws_range: Tuple[int, int],
                        changes: Tuple[int, int]) -> Tuple[int, int]:
      """Expands or contracts the range of a workspace by a specified number of samples for
            each side. This operates on frames, so the number of added/removed samples will be equal
            to or greater (less only on ends) than specified (which would be 2*`samples`).

            Args:
                durs: (np.ndarray): Array with durations of frames
                ws_range (int): Range of a workspace
                changes (Tuple[int, int]): Number of samples to add to or remove from each side of
                the range: (left_side, right_side)

            Returns:
                Tuple[int, int]: Modified range
            """
      assert ws_range[0] < ws_range[1]
      ws_range_inclusive = (ws_range[0], ws_range[1] - 1)
      directions = np.sign(changes) * np.array([-1, 1])
      starts = np.array(ws_range_inclusive) + directions * (np.sign(changes) > 0)

      modified = [*ws_range]
      for side_idx, direction in enumerate(directions):
        target_samples = changes[side_idx]
        start = starts[side_idx]
        if target_samples > 0:
          end = len(durs) if direction == 1 else -1
        elif target_samples < 0:
          end = ws_range[1] if direction == 1 else ws_range[0] - 1
        if target_samples == 0 or start == end:
          continue

        samples = 0
        for fr_idx in range(start, end, direction):
          samples += durs[fr_idx]
          if samples >= abs(target_samples):
            break
        modified[side_idx] = fr_idx + (direction > 0)
      modified = (min(modified), max(modified))
      assert modified[0] <= modified[1]
      return modified

    def __init__(self, src_durs: np.ndarray, dst_durs: np.ndarray, ws_range: Tuple[int, int],
                 encode_left_pad: bool, encode_right_pad: bool):
      """Creates a workspace for a specified range of frames.

            Args:
                src_durs (np.ndarray): Durations of all the original frames
                dst_durs (np.ndarray): Durations of all the edited frames
                ws_range (Tuple[int, int]): Range of frames
                encode_left_pad (bool): Should this workspace encode the left padding during
                editing. When True, ws_range should already include the left padding
                encode_right_pad (bool): Should this workspace encode the right padding during
                editing. When True, ws_range should already include the right padding
            """
      left_change = (-1 if encode_left_pad else 1) * AUD_WS_PAD_SIZE
      right_change = (-1 if encode_right_pad else 1) * AUD_WS_PAD_SIZE
      enc_ws_range = AudCtx.Workspace.modify_ws_range(src_durs, ws_range,
                                                      (left_change, right_change))
      left_pad = min(ws_range[0], enc_ws_range[0]), max(ws_range[0], enc_ws_range[0])
      right_pad = min(ws_range[1], enc_ws_range[1]), max(ws_range[1], enc_ws_range[1])

      self.beg = left_pad[0]
      self.end = right_pad[1]
      self.left_pad = (left_pad[0] - self.beg, left_pad[1] - self.beg)
      self.right_pad = (right_pad[0] - self.beg, right_pad[1] - self.beg)
      self.encode_left_pad = encode_left_pad
      self.encode_right_pad = encode_right_pad
      self.dst_durs = dst_durs[self.beg:self.end]
      self.frame_cache = []
      self.frame_template = None  # first pushed frame
      self.next_workspace = None

      assert self.left_pad[0] == 0 and self.right_pad[1] == self.dst_durs.shape[0]

    def push_frame(self, fr_idx: int, frame: av.AudioFrame, frame_data: np.ndarray) -> bool:
      assert fr_idx < self.end
      if fr_idx < self.beg:
        return False

      if len(self.frame_cache) == 0:
        self.frame_template = frame
      if self.dst_durs[fr_idx - self.beg] > 0:
        frame_data = frame.to_ndarray() if frame_data is None else frame_data
      else:
        if len(self.frame_cache) > 0:
          frame_data = np.ndarray((0, 0), dtype=self.frame_cache[0][1].dtype)
        else:
          frame_data = np.ndarray((0, 0), dtype=frame.to_ndarray().dtype)
      self.frame_cache.append((fr_idx, frame_data))
      return True

    def pull_frame(self) -> av.AudioFrame:
      if len(self.frame_cache) == 0 or self.frame_cache[-1][0] < (self.end - 1):
        return None
      assert (len({idx for idx, frame in self.frame_cache}) == len(self.frame_cache) and
              len(self.frame_cache) == (self.end - self.beg))

      dst_lpad_len = np.sum(self.dst_durs[:self.left_pad[1]])
      dst_rpad_len = np.sum(self.dst_durs[self.right_pad[0]:])

      # add virtual frames to separate padding from real frames
      if dst_lpad_len > 0:
        self.frame_cache.append((self.beg + self.left_pad[1] - 0.5, np.ndarray((0, 0))))
      if dst_rpad_len > 0:
        self.frame_cache.append((self.beg + self.right_pad[0] - 0.25, np.ndarray((0, 0))))
      self.frame_cache = sorted(self.frame_cache, key=lambda kv: kv[0])

      src_durs = np.array([frame.shape[1] for idx, frame in self.frame_cache])
      dst_durs = np.concatenate([
          self.dst_durs[self.left_pad[0]:self.left_pad[1]],
          [0],  # virtual, the end of the left padding
          self.dst_durs[self.left_pad[1]:self.right_pad[0]],
          [0],  # virtual, the begining of the right padding
          self.dst_durs[self.right_pad[0]:self.right_pad[1]]
      ])

      # # speed of frames next to deleted ones is unchanged (but they are trimmed)
      # for beg, end in ranges_of_truth(src_durs == 0):
      #     left, right = max(beg-1, 0), min(end, len(src_durs)-1)
      #     # the right side of the left frame and the left side of the right frame are trimmed
      #     src_durs[left] = dst_durs[left]
      #     src_durs[right] = dst_durs[right]
      #     self.frame_cache[left][1] = self.frame_cache[left][1][:, :src_durs[left]]
      #     self.frame_cache[right][1] = self.frame_cache[right][1][:, -src_durs[right]:]

      signal = np.concatenate([f for i, f in self.frame_cache if f.shape[1] > 0], axis=1)
      left_pad = signal[:, :dst_lpad_len]
      right_pad = signal[:, (signal.shape[1] - dst_rpad_len):]

      # Time-Scale Modification
      # padding is included here for the calculations but its modifications are discarded
      src_sp = np.concatenate([[0], np.cumsum(src_durs[src_durs > 0])])
      dst_sp = np.concatenate([[0], np.cumsum(dst_durs[dst_durs > 0])])
      assert src_sp[-1] == signal.shape[1]
      assert src_sp.shape == dst_sp.shape
      if not np.array_equal(src_sp, dst_sp):
        src_sp[-1] -= 1
        dst_sp[-1] -= 1
        signal = tsm.phase_vocoder(signal,
                                   np.array([src_sp, dst_sp]),
                                   AUD_WIN_TYPE,
                                   AUD_WIN_SIZE,
                                   AUD_HOP_SIZE,
                                   phase_lock=True)

      # discard modifications made to padding
      signal[:, :dst_lpad_len] = left_pad
      signal[:, (signal.shape[1] - dst_rpad_len):] = right_pad

      # soften transitions between frames next to deleted (or virtual) ones
      full_dst_sp = np.cumsum(dst_durs)  # includes deleted frames
      for beg, end in ranges_of_truth(dst_durs == 0):
        if not (0 < beg < end < len(dst_durs)):
          continue
        win_size = min(64, full_dst_sp[beg])
        win_size = min(win_size, full_dst_sp[-1] - full_dst_sp[beg])
        window = -np.hamming(win_size * 2) + 1
        signal[:, full_dst_sp[beg] - win_size:full_dst_sp[beg] + win_size] *= window

      if not self.encode_left_pad and dst_lpad_len > 0:
        signal = signal[:, dst_lpad_len:]
      if not self.encode_right_pad and dst_rpad_len > 0:
        signal = signal[:, :(signal.shape[1] - dst_rpad_len)]
        # if encode_right_pad == False, there must be a next sub-workspace
        # transfer common frames to the next workspace
        assert self.next_workspace is not None and len(self.next_workspace.frame_cache) == 0
        for f in reversed(self.frame_cache):
          if f[0] != int(f[0]):  # discard virtual
            continue
          if not (self.next_workspace.beg <= f[0] < self.end):
            break
          self.next_workspace.frame_cache.append(f)
        self.next_workspace.frame_cache.reverse()
        self.next_workspace.frame_template = self.frame_template

      dtype = self.frame_cache[0][1].dtype
      return AudCtx._create_frame(self.frame_template, signal.astype(dtype))  # pylint: disable=protected-access

  @staticmethod
  def _create_frame(template, data):
    frame = av.AudioFrame.from_ndarray(data, template.format.name, template.layout.name)
    frame.sample_rate = template.sample_rate
    frame.time_base = template.time_base
    frame.pts = None
    return frame

  def __init__(self, src_stream, dst_stream):
    super().__init__(src_stream, dst_stream)
    self.workspaces = []  # sorted (by pts) workspaces
    self.dst_vframes = []
    self.dst_frames_no = None
    self.past = None
    self.fr_idx = 0

  def prepare_timeline_edits(self, ranges: List[Range]):
    # return False
    src_durs, first_virtual = self._prepare_src_durs()
    if len(src_durs) == 0:
      return False

    if len(ranges) > 0:
      dst_durs = self._prepare_raw_dst_durations(src_durs, ranges)
      src_durs = src_durs[:len(dst_durs)]

      # convert from seconds to samples
      pts = (np.cumsum(src_durs) * self.src_stream.sample_rate).astype(int)
      src_durs[1:] = (pts[1:] - pts[:-1])
      src_durs[0] = pts[0]
      src_durs = src_durs.astype(int)
      pts = np.round(np.cumsum(dst_durs) * self.src_stream.sample_rate).astype(int)
      dst_durs[1:] = (pts[1:] - pts[:-1])
      dst_durs[0] = pts[0]
      dst_durs = dst_durs.astype(int)
      self.dst_frames_no = len(dst_durs)

      if len(dst_durs) == 0 or np.sum(dst_durs) == 0:
        return False

      # PyAV expects the audio streams to start at 0, so the virtual first frame (if present)
      # will be actually encoded - if its too big, it must be split into many frames
      if len(dst_durs) > 0:
        if first_virtual:
          self.dst_vframes = int_linspace_steps_by_limit(0, dst_durs[0], AUD_VFRAME_SIZE_MAX)
          # srcV_frames = int_linspace_steps_by_no(0, src_durs[0], len(self.dst_vframes))
          dst_durs = np.concatenate([self.dst_vframes, dst_durs[1:]])
          src_durs = np.concatenate([self.dst_vframes, src_durs[1:]])
        self.workspaces = self._prepare_workspaces(src_durs, dst_durs)
    # there must be at lease one valid frame
    return len(dst_durs) > 0

  def _prepare_workspaces(self, src_durs: np.ndarray, dst_durs: np.ndarray):
    to_edit = src_durs[:len(dst_durs)] != dst_durs
    src_durs[dst_durs == 0] = 0  # deleted frames will not be included in workspaces

    for beg, end in ranges_of_truth(to_edit):
      # extend the ranges by padding (this might merge adjacent ranges into one)
      ext_beg, ext_end = AudCtx.Workspace.modify_ws_range(src_durs, (beg, end),
                                                          (AUD_WS_PAD_SIZE, AUD_WS_PAD_SIZE))
      assert ext_beg <= ext_beg < ext_end <= ext_end
      src_samples = np.sum(src_durs[ext_beg:ext_end])
      dst_samples = np.sum(dst_durs[ext_beg:ext_end])
      if abs(src_samples - dst_samples) <= 2:
        dst_durs[beg:end] = src_durs[beg:end]
        to_edit[beg:end] = False
      else:
        to_edit[ext_beg:beg] = True
        to_edit[end:ext_end] = True

    return [
        ws for ws_range in ranges_of_truth(to_edit)
        for ws in AudCtx.Workspace.create_workspaces(src_durs, dst_durs, ws_range)
    ]

  def decode_edit_encode(self, src_packet: av.Packet) -> List[av.Packet]:
    for fr_idx, frame, frame_data in self._decode(src_packet):
      self.is_done = fr_idx + 1 >= self.dst_frames_no
      if len(self.workspaces) > 0 and self.workspaces[0].push_frame(fr_idx, frame, frame_data):
        while len(self.workspaces) > 0:
          frame = self.workspaces[0].pull_frame()
          if frame is not None:
            self.workspaces.pop(0)
            yield self.dst_stream.encode(frame)
          else:
            assert not self.is_done
            break
      else:
        yield self.dst_stream.encode(frame)

  def _decode(self, src_packet: av.Packet) -> Tuple[int, av.AudioFrame, np.ndarray]:
    frames = src_packet.decode()
    if self.fr_idx < len(self.dst_vframes) and len(frames) > 0:
      src_frame = frames[0]
      src_data = src_frame.to_ndarray()
      while self.fr_idx < len(self.dst_vframes):
        fr_idx = self.fr_idx
        self.fr_idx += 1

        data = np.zeros((src_data.shape[0], self.dst_vframes[fr_idx]), dtype=src_data.dtype)
        v_frame = AudCtx._create_frame(src_frame, data)
        yield fr_idx, v_frame, data

    for frame in frames:
      fr_idx = self.fr_idx
      self.fr_idx += 1

      frame.pts = None
      yield fr_idx, frame, None


class Editor:

  @staticmethod
  def from_json(json_specs: dict, logger: Logger) -> 'Editor':
    """Constructs an Editor from a dictionary with edit specifications.
        Ensures correct types of specs' values

        Returns:
            Editor: Editor instance prepared to edit
        """
    editor = Editor(logger)
    for identifier, specs in json_specs.get('specs', {}).items():
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
          for option_key, option_value in value.items():
            value[option_key] = str(option_value)
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
  def parse_json_ranges(json_ranges: List[List[str]]) -> List[Range]:
    ranges = [Range(*r) for r in json_ranges]
    if len(json_ranges) > 0:
      for r1, r2 in zip(ranges[:-1], ranges[1:]):
        if not (0 <= r1.beg < r1.end <= r2.beg):
          raise ValueError('Ranges must be sorted, mutually exclusive, and of length > 0')
      if not (0 <= ranges[-1].beg < ranges[-1].end):
        raise ValueError('Ranges must be of length > 0')
    return ranges

  def __init__(self, logger):
    self.logger = logger
    self.specs = {}

  def edit(self, src_path: Path, ranges: List[Range], dst_path: Path):
    self.logger.info(f'Started editing: "{src_path}"')
    src_path = str(Path(src_path).resolve())
    dst_path = str(Path(dst_path).resolve())
    source = av.open(str(src_path))
    dest, ctx_map = self.prepare_destination(source, dst_path)

    # find dts of first packets (to know which one to seek for begining)
    first_pkts = {}
    for stream in source.streams:
      seek_beginning(stream)
      first_pkt = next(source.demux(stream))
      first_pkts[stream.index] = Real(first_pkt.dts * first_pkt.time_base)
    streams_ordered = [k for k, v in sorted(first_pkts.items(), key=lambda kv: kv[1])]

    # prepare contexts of streams for editing
    valid_streams = {}
    for idx, ctx in ctx_map.items():
      if ctx.prepare_timeline_edits(ranges):
        valid_streams[idx] = ctx.src_stream

    # edit
    if len(valid_streams) > 0:
      first_stream = [idx for idx in streams_ordered if idx in valid_streams][0]
      seek_beginning(source.streams[first_stream])
      for src_packet in source.demux(list(valid_streams.values())):
        ctx = ctx_map[src_packet.stream.index]
        assert src_packet.stream is ctx.src_stream
        if ctx.is_done:
          continue
        for dst_packet in ctx.decode_edit_encode(src_packet):
          dest.mux(dst_packet)
        if all(ctx_map.values()):  # early stop when all are done
          break
      for dst_stream in dest.streams:
        dest.mux(dst_stream.encode())
    else:
      self.logger.warning(f'Editing: "{src_path}" resulted in an empty recording')
    self.logger.info(f'Finished editing: "{src_path}" -> "{dst_path}"')

    dest.close()
    source.close()

  def prepare_destination(self, source, dst_path):
    dst = av.open(dst_path, mode='w')
    ctx_map = {}

    valid_streams = []
    for stream in source.streams:
      if stream.type in SUPPORTED_STREAM_TYPES:
        if stream.codec_context is not None:
          valid_streams.append(stream)
        else:
          self.logger.warning(f'Skipping #{stream.index} stream (no decoder available)')
      else:
        self.logger.warning(f'Skipping #{stream.index} stream ({stream.type} not supported)')

    for src_stream in valid_streams:
      # stream-specific settings take precedence over type-specific settings
      specs = self.specs.get(src_stream.type, {})
      specs.update(self.specs.get(src_stream.index, {}))

      if src_stream.type == VIDEO_STREAM_TYPE:
        codec = specs.get('codec', src_stream.codec_context.name)
        codec_options = specs.get('codec_options', src_stream.codec_context.options)
        bitrate = specs.get('bitrate', src_stream.bit_rate)
        resolution = specs.get('resolution', [src_stream.width, src_stream.height])
        max_fps = Fraction(specs.get('maxfps', src_stream.guessed_rate))

        dst_stream = dst.add_stream(codec_name=codec, options=codec_options)
        dst_stream.codec_context.time_base = Fraction(1, 60000)
        dst_stream.time_base = Fraction(1, 60000)  # might not work
        dst_stream.pix_fmt = src_stream.pix_fmt
        dst_stream.bit_rate = bitrate
        dst_stream.width, dst_stream.height = resolution
        ctx_map[src_stream.index] = VidCtx(src_stream, dst_stream, max_fps)

      elif src_stream.type == AUDIO_STREAM_TYPE:
        codec = specs.get('codec', src_stream.codec_context.name)
        codec_options = specs.get('codec_options', src_stream.codec_context.options)
        bitrate = specs.get('bitrate', src_stream.bit_rate)
        samplerate = specs.get('samplerate', src_stream.sample_rate)
        channels = 1 if specs.get('mono', False) else src_stream.channels

        dst_stream = dst.add_stream(codec_name=codec, rate=samplerate)
        dst_stream.options = codec_options
        dst_stream.bit_rate = bitrate
        dst_stream.channels = channels
        ctx_map[src_stream.index] = AudCtx(src_stream, dst_stream)

      src_stream.thread_type = 'AUTO'
      dst_stream.thread_type = 'AUTO'

    dst.start_encoding()
    for vid_ctx in [c for c in ctx_map.values() if isinstance(c, VidCtx)]:
      possible_fps = 1 / vid_ctx.dst_stream.time_base
      if possible_fps < vid_ctx.max_fps:
        self.logger.warning(f'Low time base resolution of #{dst_stream.index} video stream - '
                            f'maxfps must be limited from {vid_ctx.max_fps} to {possible_fps}')
        vid_ctx.max_fps = possible_fps

    return dst, ctx_map

  def export_json(self, ranges: List[Range], path):
    with open(path, 'w', encoding='UTF-8') as fp:
      specs_dict = {'specs': self.specs, 'ranges': [[r.beg, r.end, r.multi] for r in ranges]}
      json.dump(specs_dict, fp)


############################################### CLI ################################################

NAME = 'editor'
DESCRIPTION = 'Edits recordings according to the specification'
ARG_SRC = 'src'
ARG_DST = 'dst'
DEFAULT_ARGS = {}


def setup_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  """Creates CLI argument parser for editor submodule

    Returns:
        argparse.ArgumentParser: Argument parser of this submodule
    """
  parser.description = DESCRIPTION
  parser.add_argument(ARG_SRC, help='Path of the recording to edit', type=Path, action='store')
  parser.add_argument(ARG_DST, help='Path of the edited recording', type=Path, action='store')
  parser.set_defaults(run=run_submodule)
  # TODO
  return parser


def run_submodule(args: object, logger: Logger) -> None:
  # TODO
  args = args.__dict__
  with open('test.json', 'r', encoding='UTF-8') as fp:
    json_specs = json.load(fp)
  editor = Editor.from_json(json_specs, logger=logger)
  ranges = Editor.parse_json_ranges(json_specs['ranges'])
  # editor.export_json(ranges, 'test2.json')
  editor.edit(args[ARG_SRC], ranges, args[ARG_DST])
