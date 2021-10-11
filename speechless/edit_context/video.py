import av
import numpy as np
from typing import Generator, List
from av.video.stream import VideoStream

from ..utils import Real
from .common import EditCtx, TimelineChange

DROP_FRAME_PTS = -1


class VideoEditContext(EditCtx):

  def __init__(self, src_stream: VideoStream, dst_stream: VideoStream, max_fps: float):
    """Context of a video stream during editing

    Args:
        src_stream (VideoStream): Source video stream to edit
        dst_stream (VideoStream): Destination video stream of the edit
        max_fps (float): Maximum FPS of the destination stream. This value might be used to generate
        a CFR video by providing the FPS of the source stream (this is usually the case)
    """
    super().__init__(src_stream, dst_stream)
    self.max_fps = max_fps
    self.dst_pts = None

  def prepare_for_editing(self, changes: List[TimelineChange]) -> bool:
    """Prepares this context for editing by calculating the durations of the destination stream's
    frames

    Args:
        changes (List[TimelineChange]): Timeline changes to make

    Returns:
        bool: Whether the destination stream will be empty
    """
    durs, first_is_virtual = self._prepare_src_durations()
    if len(durs) == 0:
      return False

    if len(changes) > 0:
      durs = self._prepare_raw_dst_durations(durs, changes)
      durs = self._constrain_raw_dst_durations(durs)
      if len(durs) == 0 or np.max(durs) <= 0:
        return False

    self.dst_pts = np.concatenate([[0], np.cumsum(durs[:-1])])
    self.dst_pts[durs <= 0] = DROP_FRAME_PTS

    # if first frame is virtual, discard it
    if first_is_virtual:
      self.dst_pts = self.dst_pts[1:]

    self.num_frames_to_encode = len(self.dst_pts)
    return len(self.dst_pts) > 0

  def _constrain_raw_dst_durations(self, dst_durs: np.ndarray) -> np.ndarray:
    """Constrains the durations of the destination stream's frames to stay true to the set max FPS.
    This operation might introduce a "small" (up to 1/MAX_FPS/2 seconds) desync

    Args:
        dst_durs (np.ndarray): Unconstrained durations of the destination stream's frames

    Returns:
        np.ndarray: Constrained durations of the destination stream's frames
    """
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

  def decode_edit_encode(self, src_packet: av.Packet) -> Generator[av.Packet, None, None]:
    """Decodes the packet, edits the frame and encodes it to a packet of the destination stream

    Args:
        src_packet (av.Packet): Packet of the source stream to decode

    Yields:
        av.Packet: Packet to mux into the destination container
    """
    assert src_packet.stream is self.src_stream

    for frame in src_packet.decode():
      if self.is_done():
        break
      frame_idx = self.num_frames_encoded
      self.num_frames_encoded += 1
      if self.dst_pts[frame_idx] != DROP_FRAME_PTS:
        frame.pts = int(round(self.dst_pts[frame_idx] / frame.time_base))
        frame.pict_type = av.video.frame.PictureType.NONE
        yield self.dst_stream.encode(frame)
