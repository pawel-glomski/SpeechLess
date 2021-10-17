import subprocess
import av
import numpy as np
from av.audio.frame import format_dtypes
from typing import Generator
from logging import Logger

from .utils import NULL_LOGGER


class AudioReader:

  def __init__(self, file_path: str, logger: Logger = NULL_LOGGER):
    """Audio frame reader

    Args:
        file_path (str): Path to the recording
        logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER
    """
    self.file_path = file_path
    self.logger = logger
    self._container = None

  def __del__(self):
    if self._container is not None:
      self._container.close()

  def read_stream(self, aud_stream_idx: int = None) -> Generator[np.ndarray, None, None]:
    """Creates a generator of audio frames of a given audio stream in the recording

    Args:
        aud_stream_idx (int, optional): Index of an audio stream to read (not its index in the \
          container). Defaults to None (first audio stream)

    Returns:
        Generator[np.ndarray, None, None]: A generator of audio frames in numpy format
    """
    if self._container is not None:
      self._container.close()
    self._container = av.open(self.file_path)

    if aud_stream_idx is None:
      aud_stream_idx = 0
      if len(self._container.streams.audio) > 1:
        self.logger.warning(
            'Unspecified audio stream for a file with multiple audio streams. Reading '
            f'the first one (stream #{self._container.streams.audio[self._stream_idx].index})')

    def audio_iter():
      for frame in self._container.decode(audio=aud_stream_idx):
        yield frame.to_ndarray()

    return audio_iter()


def read_entire_audio(file_path: str,
                      aud_stream_idx: int = None,
                      aud_format: str = 'f32le',
                      logger: Logger = NULL_LOGGER) -> np.ndarray:
  """Reads an entire audio stream from a recording

  Args:
      file_path (str): Path to a recording
      aud_stream_idx (int, optional): Index of an audio stream (not its index in the container). \
        Defaults to None (first audio stream)
      aud_format (str, optional): A desired audio format. Defaults to 'f32le'
      logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER

  Returns:
      np.ndarray: An entire audio stream in a specified format
  """
  with av.open(file_path) as container:
    if aud_stream_idx is None:
      aud_stream_idx = 0
      if len(container.streams.audio) > 1:
        logger.warning('Unspecified audio stream for a file with multiple audio streams. Reading '
                       f'the first one (stream #{container.streams.audio[aud_stream_idx].index})')

    acodec = f'pcm_{aud_format}'
    process = subprocess.Popen(stdout=subprocess.PIPE,
                               args=[
                                   'ffmpeg', '-i', f'{file_path}', '-map', f'0:a:{aud_stream_idx}',
                                   '-f', f'{aud_format}', '-acodec', f'{acodec}', 'pipe:1'
                               ])
    buffer, _ = process.communicate()

    astream = container.streams.audio[aud_stream_idx]
    acodec = av.Codec(acodec, 'r')
    dtype = np.dtype(format_dtypes[acodec.audio_formats[0].name])
    if acodec.audio_formats[0].is_planar:
      return np.frombuffer(buffer, dtype).reshape((astream.channels, -1))
    else:
      return np.frombuffer(buffer, dtype).reshape((-1, astream.channels)).T
