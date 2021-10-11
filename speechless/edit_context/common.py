import av
import numpy as np
from typing import Dict, List, Tuple
from speechless.utils import Real


class TimelineChange:

  def __init__(self, beg: float, end: float, multi: float):
    """Change in the timeline of a recording

    Args:
        beg (float): Start time of the modification in seconds
        end (float): End time of the modification in seconds
        multi (float): Duration multiplier - values less than one increase the speed of the selected
        fragment and values greater than one will slow it down
    """
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

  @staticmethod
  def from_numpy(arr: np.ndarray) -> List['TimelineChange']:
    """Creates a list of timeline changes from a numpy ndarray

    Args:
        arr (np.ndarray): Array of shape (N, 3) where the first dimention is the number of changes
        and the second is a change in the form of a 3-element array:
        [start time, end time, duration multiplier]. The times are in seconds

    Raises:
        ValueError: When changes are not sorted properly or one of them has a range of length <= 0

    Returns:
        List[TimelineChange]: List of timeline changes
    """
    changes = [TimelineChange(*r) for r in arr]
    if len(changes) > 0:
      for r1, r2 in zip(changes[:-1], changes[1:]):
        if not (0 <= r1.beg < r1.end <= r2.beg):
          raise ValueError('Changes must be sorted, mutually exclusive, and with ranges of length '
                           'greater than zero')
      if not (0 <= changes[-1].beg < changes[-1].end):
        raise ValueError('Change range must be of length greater than 0')
    return changes


class EditCtx:

  def __init__(self, src_stream: av.stream.Stream, dst_stream: av.stream.Stream):
    """Generic context of a stream during editing

    Args:
        src_stream (av.stream.Stream): Source (audio or video) stream to edit
        dst_stream (av.stream.Stream): Destination (audio or video) stream of the edit
    """
    self.src_stream = src_stream
    self.dst_stream = dst_stream
    self.num_frames_to_encode = None
    self.num_frames_encoded = 0

  def is_done(self) -> bool:
    """Returns whether editing is done for this context

    Returns:
        bool: True when done, False otherwise
    """
    # if num_frames_to_encode is None, then this function was called too early
    return self.num_frames_encoded >= self.num_frames_to_encode

  def _prepare_src_durations(self) -> Tuple[np.ndarray, bool]:
    """Demuxes each packet in the source stream in order to calculate the duration of each frame
    using Presentation TimeStamp of packets

    Returns:
        Tuple[np.ndarray, bool]: An array of frame durations and whether the first duration is
        virtual - duration of a non-existent frame added to streams that do not start at 0 second
    """
    pts = []
    for packet in self.src_stream.container.demux(self.src_stream):
      if packet.pts is not None and packet.pts >= 0:
        pts.append(packet.pts)
    if len(pts) < 2:  # there must be at least 2 frames
      return (np.ndarray(shape=(0,)), False)

    pts = sorted(set(pts))
    # convert to seconds and set the end frame pts
    pts = np.array(pts, dtype=Real) * Real(self.src_stream.time_base)
    durs = (pts[1:] - pts[:-1])
    # add virtual empty first frame that lasts until the first real frame
    virtual_first = ([pts[0]] if pts[0] > 0 else [])
    # assume duration of the last frame is equal to the average duration of real frames

    src_durs = np.concatenate([virtual_first, durs, [np.mean(durs)]])
    first_is_virtual = len(virtual_first) == 1
    return (src_durs, first_is_virtual)

  def _prepare_raw_dst_durations(self, src_durs: np.ndarray,
                                 changes: List[TimelineChange]) -> np.ndarray:
    """Prepares the durations of the destination stream's frames according to the specified timeline
    changes.

    Args:
        src_durs (np.ndarray): Durations of the source stream's frames
        changes (List[TimelineChange]): List of timeline changes to make. The ranges of changes must
        be mutually exclusive and sorted ascending order of their starts

    Returns:
        np.ndarray: Durations of the destination stream's frames
    """
    dst_durs = []
    r_idx = 0
    fr_end = Real(0)
    src_stream_end = np.sum(src_durs, dtype=Real)
    for old_dur in src_durs:
      fr_beg = fr_end
      fr_end = fr_beg + old_dur
      new_dur = 0

      # go through all timeline changes that modify current frame
      while r_idx < len(changes) and fr_beg < changes[r_idx].end and fr_end > changes[r_idx].beg:
        clamped_beg = np.max([changes[r_idx].beg, fr_beg])
        clamped_end = np.min([changes[r_idx].end, fr_end])
        part_dur = clamped_end - clamped_beg
        assert np.round(old_dur - part_dur, 12) >= 0

        old_dur -= part_dur
        new_dur += part_dur * changes[r_idx].multi
        if clamped_end < changes[r_idx].end:
          # this change extends beyond this frame (was clamped), move to the next frame
          break
        else:
          # this change ends on this frame (wasn't clamped), move to the next change
          r_idx += 1
      new_dur += old_dur

      # early stop when the recording has trimmed end
      if (new_dur == 0 and r_idx < len(changes) and fr_end > changes[r_idx].beg and
          changes[r_idx].end >= src_stream_end):
        break
      dst_durs.append(new_dur)

    return np.array(dst_durs, dtype=Real) if len(dst_durs) >= 2 else np.array([])


def restart_container(container: av.container.InputContainer, ctx_map: Dict[int, EditCtx]) \
  -> av.container.InputContainer:
  """Restarts a container and updates the affected contexts. The next demuxed packed of the returned
  container will be its first one

  Args:
      container (av.container.InputContainer): A container to restart
      ctx_map (Dict[int, EditCtx]): Map of contexts to update (mapping: stream_index -> context)

  Returns:
      av.container.InputContainer: A restarted container
  """
  for ctx in ctx_map.values():
    assert ctx.src_stream.container is container

  filename = container.name
  container.close()

  container = av.open(filename)
  for idx, ctx in ctx_map.items():
    ctx_map[idx].src_stream = container.streams[idx]
  return container
