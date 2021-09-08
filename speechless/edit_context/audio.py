import av
import numpy as np
import pytsmod as tsm
from typing import Generator, List, Tuple
from av.audio.stream import AudioStream
from av.audio.frame import format_dtypes

from ..utils import ranges_of_truth, int_linspace_steps_by_limit
from .common import EditCtx, TimelineChange, Real

WIN_TYPE = 'hann'
WIN_SIZE = 1024
HOP_SIZE = int(WIN_SIZE / 4)
WS_PAD_SIZE = 512
WS_SIZE_MAX = WS_PAD_SIZE + 1000 * WIN_SIZE + WS_PAD_SIZE
VFRAME_SIZE_MAX = 10 * 1024


class Workspace:

  def __init__(self, src_durs: np.ndarray, dst_durs: np.ndarray, ws_range: Tuple[int, int],
               encode_left_pad: bool, encode_right_pad: bool):
    """Audio editing workspace for a specific range of frames

    Args:
        src_durs (np.ndarray): Durations of all the original frames
        dst_durs (np.ndarray): Durations of all the edited frames
        ws_range (Tuple[int, int]): Range of frames
        encode_left_pad (bool): Should this workspace encode the left padding during
        editing. When True, ws_range should already include the left padding
        encode_right_pad (bool): Should this workspace encode the right padding during
        editing. When True, ws_range should already include the right padding
    """
    left_change = (-1 if encode_left_pad else 1) * WS_PAD_SIZE
    right_change = (-1 if encode_right_pad else 1) * WS_PAD_SIZE
    enc_ws_range = Workspace.modify_workspace_range(src_durs, ws_range, (left_change, right_change))
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

  def push_frame(self, frame_idx: int, frame: av.AudioFrame, frame_data: np.ndarray) -> bool:
    """Pushes a frame to the workspace's cache if the frame is in this workspace's range

    Args:
        frame_idx (int): Index of the frame to push
        frame (av.AudioFrame): Frame to push
        frame_data (np.ndarray): Numpy representation of the frame's data

    Returns:
        bool: Whether the frame was in the workspace's range
    """
    assert frame_idx < self.end
    if frame_idx < self.beg:
      return False

    if len(self.frame_cache) == 0:
      self.frame_template = frame

    if self.dst_durs[frame_idx - self.beg] > 0:
      frame_data = frame.to_ndarray() if frame_data is None else frame_data
      if self.frame_template.format.is_packed and len(self.frame_template.layout.channels) > 1:
        frame_data = frame_data.reshape((-1, 2)).T  # from packed to planar
    else:
      if len(self.frame_cache) > 0:
        frame_data = np.ndarray((0, 0), dtype=self.frame_cache[0][1].dtype)
      else:
        # this will only happen when the first frame is deleted
        frame_data = np.ndarray((0, 0), dtype=np.dtype(format_dtypes[frame.format.name]))

    self.frame_cache.append([frame_idx, frame_data])
    return True

  def pull_frame(self) -> av.AudioFrame:
    """When all the frames of this workspace's range were pushed, edits them and returns the output
    as a single frame

    Returns:
        av.AudioFrame: Edited frames combined to a single frame
    """
    if len(self.frame_cache) == 0 or self.frame_cache[-1][0] < (self.end - 1):
      return None
    assert (len({idx for idx, frame in self.frame_cache}) == len(self.frame_cache) and
            len(self.frame_cache) == (self.end - self.beg))

    dst_lpad_len = np.sum(self.dst_durs[:self.left_pad[1]])
    dst_rpad_len = np.sum(self.dst_durs[self.right_pad[0]:])
    src_durs = np.array([frame.shape[1] for idx, frame in self.frame_cache])
    dst_durs = self.dst_durs

    # speed of frames next to deleted ones is unchanged (but they are trimmed)
    for beg, end in ranges_of_truth(src_durs == 0):
      left, right = max(beg - 1, 0), min(end, len(src_durs) - 1)
      # the right side of the left frame and the left side of the right frame are trimmed
      if dst_durs[left] < src_durs[left]:
        src_durs[left] = dst_durs[left]
        self.frame_cache[left][1] = self.frame_cache[left][1][:, :src_durs[left]]
      if dst_durs[right] < src_durs[right]:
        src_durs[right] = dst_durs[right]
        self.frame_cache[right][1] = self.frame_cache[right][1][:, -src_durs[right]:]

    signal = np.concatenate([f for i, f in self.frame_cache if f.shape[1] > 0], axis=1).astype(Real)

    # soften transitions between frames next to deleted ones

    def soften_transition(signal, point, length):
      win_size = min(length, point)
      win_size = max(min(win_size, signal.shape[1] - point), 0)
      window = -np.hamming(win_size * 2) + 1
      signal[:, point - win_size:point + win_size] *= window

    full_src_sp = np.cumsum(src_durs)  # includes deleted frames
    for beg, end in ranges_of_truth(src_durs == 0):
      if (0 < beg < end < len(src_durs)):
        soften_transition(signal, full_src_sp[beg], 64)

    # save current padding
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
      signal = tsm.wsola(
          signal,
          np.array([src_sp, dst_sp]),
          WIN_TYPE,
          WIN_SIZE,
          HOP_SIZE,
      ).reshape((signal.shape[0], -1))

      # discard modifications made to padding
      signal[:, :dst_lpad_len] = left_pad
      signal[:, (signal.shape[1] - dst_rpad_len):] = right_pad

      # soften the transition between the padding and the modified signal
      soften_transition(signal, dst_lpad_len, 64)
      soften_transition(signal, (signal.shape[1] - dst_rpad_len), 64)

    # prepare the output frame
    if not self.encode_left_pad and dst_lpad_len > 0:
      signal = signal[:, dst_lpad_len:]
    if not self.encode_right_pad and dst_rpad_len > 0:
      signal = signal[:, :(signal.shape[1] - dst_rpad_len)]
      # if encode_right_pad == False, there must be a next sub-workspace
      # transfer common frames to the next workspace
      assert self.next_workspace is not None and len(self.next_workspace.frame_cache) == 0
      for f in reversed(self.frame_cache):
        if not (self.next_workspace.beg <= f[0] < self.end):
          break
        self.next_workspace.frame_cache.append(f)
      self.next_workspace.frame_cache.reverse()
      self.next_workspace.frame_template = self.frame_template

    dtype = self.frame_cache[0][1].dtype
    return create_audio_frame(self.frame_template, signal.astype(dtype))

  @staticmethod
  def create_workspaces(src_durs: np.ndarray, dst_durs: np.ndarray,
                        ws_range: Tuple[int, int]) -> List['Workspace']:
    """Creates a workspace for a specified range of frames. If the range is too large, it will be
    split into smaller ones. A group of sub-workspaces created this way will exchange necessary
    frames (padding) among themselves

    Args:
        src_durs (np.ndarray): Durations of the original frames
        dst_durs (np.ndarray): Durations of the edited frames
        ws_range (Tuple[int, int]): Range of frames to work on (includes padding)

    Returns:
        List[Workspace]: Workspaces for the specified range
    """
    sub_ws_ranges = Workspace.split_workspace_range(src_durs, ws_range)
    assert len(sub_ws_ranges) > 0

    workspaces = []
    if len(sub_ws_ranges) == 1:
      workspaces.append(Workspace(src_durs, dst_durs, ws_range, True, True))
    else:
      workspaces.append(Workspace(src_durs, dst_durs, sub_ws_ranges[0], True, False))
      for sub_ws_range in sub_ws_ranges[1:-1]:
        workspaces.append(Workspace(src_durs, dst_durs, sub_ws_range, False, False))
      workspaces.append(Workspace(src_durs, dst_durs, sub_ws_ranges[-1], False, True))
      for ws, next_ws in zip(workspaces[:-1], workspaces[1:]):
        ws.next_workspace = next_ws
    return workspaces

  @staticmethod
  def split_workspace_range(durs: np.ndarray, ws_range: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Splits the range of a workspace on many smaller ones if its too large. Created ranges will be
    roughly the same size

    Args:
        durs (np.ndarray): Array with durations of frames
        ws_range (Tuple[int, int]): Workspace range to split

    Returns:
        List[Tuple[int, int]]: List of ranges of smaller workspaces
    """
    assert ws_range[0] < ws_range[1]
    start, end = ws_range

    range_samples = np.sum(durs[start:end])
    target_ws_sizes = int_linspace_steps_by_limit(0, range_samples, WS_SIZE_MAX)
    if len(target_ws_sizes) == 1:
      return [ws_range]

    sub_ws_ranges = []
    for target_ws_size in target_ws_sizes:
      sub_ws_size = 0
      for frame_idx in range(start, end):
        sub_ws_size += durs[frame_idx]
        if sub_ws_size >= target_ws_size:
          break
      sub_ws_ranges.append((start, frame_idx + 1))
      start = frame_idx + 1
      if start == end:
        break
    return sub_ws_ranges

  @staticmethod
  def modify_workspace_range(durs: np.ndarray, ws_range: Tuple[int, int],
                             changes: Tuple[int, int]) -> Tuple[int, int]:
    """Expands or contracts the range of a workspace by a specified number of samples for
    each side. This operates on frames, so the number of added/removed samples will be equal
    to or greater (less only on ends) than specified (which would be 2*`samples`).

    Args:
        durs: (np.ndarray): Array with durations of frames
        ws_range (Tuple[int, int]): Range of a workspace
        changes (Tuple[int, int]): Number of samples to add to or remove from each side of the
        range: (+/-left side samples, +/-right side samples)

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
      for frame_idx in range(start, end, direction):
        samples += durs[frame_idx]
        if samples >= abs(target_samples):
          break
      modified[side_idx] = frame_idx + (direction > 0)
    modified = (min(modified), max(modified))
    assert modified[0] <= modified[1]
    return modified


class AudioEditContext(EditCtx):

  def __init__(self, src_stream: AudioStream, dst_stream: AudioStream):
    """Context of an audio stream during editing

    Args:
        src_stream (VideoStream): Source video stream to edit
        dst_stream (VideoStream): Destination video stream of the edit
    """
    super().__init__(src_stream, dst_stream)
    self.workspaces = []  # sorted by pts
    self.dst_vframes = []
    self.dst_frames_no = None
    self.frame_idx = 0

  def prepare_for_editing(self, ranges: List[TimelineChange]) -> bool:
    """Prepares this context for editing by creating editing workspaces

    Args:
        changes (List[TimelineChange]): Timeline changes to make

    Returns:
        bool: Whether the destination stream will be empty
    """
    src_durs, first_is_virtual = self._prepare_src_durations()
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

      # PyAV expects audio streams to start at 0, so the virtual first frame (if present)
      # will be actually encoded - if its too big, it must be split into many frames
      if first_is_virtual:
        self.dst_vframes = int_linspace_steps_by_limit(0, dst_durs[0], VFRAME_SIZE_MAX)
        assert len(self.dst_vframes) > 0
        dst_durs = np.concatenate([self.dst_vframes, dst_durs[1:]])
        src_durs = np.concatenate([self.dst_vframes, src_durs[1:]])
      self.workspaces = self._create_workspaces(src_durs, dst_durs)
    # there must be at lease one valid frame
    return len(dst_durs) > 0

  def _create_workspaces(self, src_durs: np.ndarray, dst_durs: np.ndarray) -> List[Workspace]:
    """Creates editing workspaces according to the differences between the source stream and
    destination stream frame durations

    Args:
        src_durs (np.ndarray): Durations of the original frames
        dst_durs (np.ndarray): Durations of the edited frames

    Returns:
        List[Workspace]: List of workspaces
    """
    to_edit = src_durs[:len(dst_durs)] != dst_durs
    # floating-point precision error can introduce a duration diffrence of 1 sample
    for beg, end in ranges_of_truth(to_edit):
      src_samples = np.sum(src_durs[beg:end])
      dst_samples = np.sum(dst_durs[beg:end])
      if abs(src_samples - dst_samples) <= 1:
        dst_durs[beg:end] = src_durs[beg:end]
        to_edit[beg:end] = False

    src_durs[dst_durs == 0] = 0  # deleted frames will not be edited (just dropped)
    for beg, end in ranges_of_truth(to_edit):
      # extend the ranges by padding (this might merge adjacent ranges into one)
      ext_beg, ext_end = Workspace.modify_workspace_range(src_durs, (beg, end),
                                                          (WS_PAD_SIZE, WS_PAD_SIZE))
      assert ext_beg <= beg < end <= ext_end
      to_edit[ext_beg:beg] = True
      to_edit[end:ext_end] = True

    return [
        ws for ws_range in ranges_of_truth(to_edit)
        for ws in Workspace.create_workspaces(src_durs, dst_durs, ws_range)
    ]

  def decode_edit_encode(self, src_packet: av.Packet) -> Generator[av.AudioFrame, None, None]:
    """Decodes the packet, edits the frame and encodes it to a packet of the destination stream

    Args:
        src_packet (av.Packet): Packet of the source stream to decode

    Yields:
        av.Packet: Packet to mux into the destination container
    """
    assert src_packet.stream is self.src_stream

    for frame_idx, frame, frame_data in self._decode(src_packet):
      self.is_done = frame_idx + 1 >= self.dst_frames_no
      if len(self.workspaces) > 0 and self.workspaces[0].push_frame(frame_idx, frame, frame_data):
        while len(self.workspaces) > 0:
          frame = self.workspaces[0].pull_frame()
          if frame is None:
            break
          self.workspaces.pop(0)
          yield self.dst_stream.encode(frame)
      else:
        yield self.dst_stream.encode(frame)

    assert not self.is_done or len(self.workspaces) == 0

  def _decode(self, src_packet: av.Packet) \
    -> Generator[Tuple[int, av.Packet, np.ndarray], None, None]:
    """Decodes a packet and (if available) returns a frame (returned frames are ordered by PTS).The
    returned frame might not be the frame encoded in the provided packet. If the source stream does
    not start at 0 second, silent frames will be generated to make sure that each audio stream
    starts at 0 (imposed by PyAV)

    Args:
        src_packet (av.Packet): Packet to decode

    Yields:
        Tuple[int, av.Packet, np.ndarray]: Index of the frame, frame, numpy representation of the
        frame's data (not None only for the generated silent frames)
    """
    frames = src_packet.decode()
    if self.frame_idx < len(self.dst_vframes) and len(frames) > 0:
      # generate silent frames
      src_frame = frames[0]
      src_data = src_frame.to_ndarray()
      while self.frame_idx < len(self.dst_vframes):
        frame_idx = self.frame_idx
        self.frame_idx += 1

        data_shape = (len(src_frame.layout.channels), self.dst_vframes[frame_idx])
        data = np.zeros(data_shape, dtype=src_data.dtype)
        v_frame = create_audio_frame(src_frame, data)
        yield frame_idx, v_frame, data

    for frame in frames:
      frame_idx = self.frame_idx
      self.frame_idx += 1

      frame.pts = None
      yield frame_idx, frame, None


def create_audio_frame(template: av.AudioFrame, data: np.ndarray) -> av.AudioFrame:
  """Creates an audio frame from a provided frame template and data

  Args:
      template (av.AudioFrame): Frame template with set format, layout, sample_rate and time_base
      data (np.ndarray): Frame data

  Returns:
      av.AudioFrame: Frame with the specified data and configuration of the template frame
  """
  if template.format.is_packed and len(template.layout.channels) > 1:
    data = data.T.reshape((1, -1))  # packed: [c1_1, c2_1, c1_2, c2_2, ..., c1_n, c2_n]
  frame = av.AudioFrame.from_ndarray(data, template.format.name, template.layout.name)
  frame.sample_rate = template.sample_rate
  frame.time_base = template.time_base
  frame.pts = None
  return frame
