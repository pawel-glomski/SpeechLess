import numpy as np
from typing import Dict
from enum import Enum
from threading import Thread
from queue import Empty, Queue
from fractions import Fraction
from collections import namedtuple
from math import copysign

import av
import pyglet
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, VideoFormat, AudioData

AUDIO_FRAME_CACHE_SIZE = 64
VIDEO_FRAME_CACHE_SIZE = 32

MAX_COMPENSATION_RATIO = Fraction(1, 10)
MAX_AUDIO_CHANNELS = 2
AUDIO_FORMAT = av.AudioFormat('s16').packed
VIDEO_FORMAT = 'rgba'

SLFrame = namedtuple('SLFrame', ['timestamp', 'data'])


class MediaType(Enum):
  Audio = 'audio'
  Video = 'video'


class StreamDecoder:

  def __init__(self, filepath: str, stream_idx: int):
    self.container = av.open(filepath)
    self.stream = self.container.streams[stream_idx]
    self.start_time = self.stream.start_time
    assert isinstance(self.stream, (av.audio.AudioStream, av.video.VideoStream))

    if isinstance(self.stream, av.audio.AudioStream):
      self.frame_buffer: 'Queue[SLFrame]' = Queue(AUDIO_FRAME_CACHE_SIZE)

      format_resampler = av.AudioResampler(format=AUDIO_FORMAT,
                                           layout=self.stream.layout,
                                           rate=self.stream.sample_rate)

      self.process_frame = lambda frame: [
          SLFrame(subframe.time, subframe.to_ndarray())
          for subframe in format_resampler.resample(frame)
      ]
    else:
      self.frame_buffer: 'Queue[SLFrame]' = Queue(VIDEO_FRAME_CACHE_SIZE)
      self.process_frame = lambda frame: [
          SLFrame(frame.time, frame.to_ndarray(format=VIDEO_FORMAT))
      ]

    self.tasks = Queue()
    Thread(target=self._start, name='Decoding Thread', daemon=True).start()

  def __del__(self):

    def stop_decoding():
      raise StopIteration()

    self.put_task(stop_decoding)
    self.tasks.join()
    self.container.close()

  def put_task(self, task):
    self.tasks.put(task)
    # When buffer is full, the decoding thread is waiting to put the next decoded frame. To trigger
    # the task, get one frame from the buffer
    try:
      self.frame_buffer.get_nowait()
    except Empty:
      pass

  def start_seek(self, offset: int):

    def av_seek():
      self.frame_buffer.queue.clear()
      self.container.seek(offset)

    self.put_task(av_seek)

  def finish_seek(self):
    self.tasks.join()

  def _start(self):
    wait_for_task = False  # true only when finished decoding
    while True:
      try:
        self.tasks.get(wait_for_task).__call__()
      except Empty:
        pass
      else:
        self.tasks.task_done()
        wait_for_task = False
        continue

      for pkt in self.container.demux(self.stream):
        for frame in pkt.decode():
          if self.tasks.unfinished_tasks > 0:
            break
          for sub_frame in self.process_frame(frame):
            self.frame_buffer.put(sub_frame)
        if self.tasks.unfinished_tasks > 0:
          break
      else:
        self.frame_buffer.put(None)
        wait_for_task = True


class SLDecoder:

  def __init__(self, container: av.container.InputContainer):
    self.audio_decoder: StreamDecoder = None
    self.video_decoder: StreamDecoder = None
    self.decoders: Dict[MediaType, StreamDecoder] = {}
    if container.streams.audio:
      self.audio_decoder = StreamDecoder(container.name, container.streams.audio[0].index)
      self.decoders[MediaType.Audio] = self.audio_decoder
    if container.streams.video:
      self.video_decoder = StreamDecoder(container.name, container.streams.video[0].index)
      self.decoders[MediaType.Video] = self.video_decoder
    self.start_time = min([dec.start_time for dec in self.decoders.values()])

  def seek(self, timestamp: float):
    offset = int(self.start_time + timestamp) * av.time_base
    for decoder in self.decoders.values():
      decoder.start_seek(offset)
    for decoder in self.decoders.values():
      decoder.finish_seek()


class SLSource(StreamingSource):

  def __init__(self, filepath: str):
    self.container = av.open(filepath)
    self.decoder = SLDecoder(self.container)
    if self.decoder.audio_decoder is not None:
      audio_stream = self.decoder.audio_decoder.stream
      self.audio_format = AudioFormat(channels=min(2, audio_stream.channels),
                                      sample_size=AUDIO_FORMAT.bits,
                                      sample_rate=audio_stream.sample_rate)
    if self.decoder.video_decoder is not None:
      video_stream = self.decoder.video_decoder.stream
      self.video_format = VideoFormat(video_stream.width, video_stream.height)
      if video_stream.sample_aspect_ratio is not None:
        self.video_format.sample_aspect = video_stream.sample_aspect_ratio
      if video_stream.framerate is not None and float(video_stream.framerate) > 0:
        self.video_format.frame_rate = float(video_stream.framerate)

  def __del__(self):
    self.container.close()

  def seek(self, timestamp):
    self.decoder.seek(timestamp)
    decoders_to_seek = list(self.decoder.decoders.values())

    while len(decoders_to_seek) > 0:
      for decoder in [*decoders_to_seek]:
        if decoder.frame_buffer.qsize() < 2:
          with decoder.frame_buffer.not_empty:
            decoder.frame_buffer.not_empty.wait()
            if decoder.frame_buffer.queue[-1] is None:
              decoders_to_seek.remove(decoder)
        else:
          if decoder.frame_buffer.queue[1].timestamp >= timestamp:
            decoders_to_seek.remove(decoder)
          if decoder.frame_buffer.queue[1].timestamp <= timestamp:
            decoder.frame_buffer.get()  # remove frame at idx == 0

  def get_audio_data(self, num_bytes, compensation_time=0.0) -> AudioData:
    frames = []
    bytes_sum = 0
    timestamp = None
    while bytes_sum < num_bytes:
      frame = self._pop_frame(MediaType.Audio)
      frame_data: np.ndarray = frame.data
      if frame is None:
        break
      if timestamp is None:
        timestamp = frame.timestamp

      frames.append(frame_data)
      bytes_sum += frame_data.size * frame_data.dtype.itemsize

    if len(frames) == 0:
      return None

    assert AUDIO_FORMAT.is_packed
    audio_stream = self.decoder.audio_decoder.stream
    data = np.concatenate(frames, axis=1).reshape((-1, audio_stream.channels))
    data = data[:, :MAX_AUDIO_CHANNELS]
    samples = data.shape[0]

    duration = Fraction(samples, audio_stream.sample_rate)
    if compensation_time != 0:
      max_comp_time = duration * MAX_COMPENSATION_RATIO
      comp_time = -Fraction(compensation_time)
      comp_time = int(copysign(1, comp_time)) * min(abs(comp_time), max_comp_time)
      out_sample_rate = samples / (duration + comp_time)
      duration = duration - comp_time
      resampler = av.AudioResampler(format=self.format_resampler.format,
                                    layout=self.format_resampler.layout,
                                    rate=out_sample_rate)

      frame = av.AudioFrame.from_ndarray(data.reshape((1, -1)), audio_stream.format.packed.name,
                                         audio_stream.layout.name)
      frame.sample_rate = audio_stream.sample_rate
      frame.time_base = audio_stream.codec_context.time_base
      frames = resampler.resample(frame)

      data = [frame.to_ndarray() for frame in frames]
      data = np.concatenate(data, axis=1)

    data = data.tobytes()
    return AudioData(data, len(data), timestamp, duration, [])

  def _pop_frame(self, frame_type: MediaType) -> SLFrame:
    buffer = self.decoder.decoders[frame_type].frame_buffer
    frame = buffer.get()
    if frame is None:
      assert len(buffer.queue) == 0
      buffer.put(None)
    return frame

  def get_next_video_timestamp(self):
    buffer = self.decoder.decoders[MediaType.Video].frame_buffer
    with buffer.not_empty:
      if len(buffer.queue) == 0:
        buffer.not_empty.wait()
    if buffer.queue[0] is None:
      return None
    return buffer.queue[0].timestamp

  def get_next_video_frame(self, skip_empty_frame=True):
    frame = self._pop_frame(MediaType.Video)
    if frame is None:
      return None
    frame = frame.data
    height, width = frame.shape[:2]
    image = pyglet.image.ImageData(width, height, VIDEO_FORMAT, frame.tobytes(),
                                   width * len(VIDEO_FORMAT))
    return image
