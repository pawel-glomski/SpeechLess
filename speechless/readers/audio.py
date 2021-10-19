import subprocess
import numpy as np
import av
from av.audio.frame import format_dtypes
from typing import Dict, Generator, Tuple
from logging import Logger
from pathlib import Path
from enum import Enum, auto

from speechless.utils.logging import NULL_LOGGER


class StreamInfo(Enum):
  SAMPLE_RATE = auto()
  FRAME_SIZE = auto()


class AudioReader:

  def __init__(self, file_path: str, logger: Logger = NULL_LOGGER):
    """Audio frame reader

    Args:
        file_path (str): Path to the recording
        logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER
    """
    self.file_path = str(Path(file_path).resolve())
    self.logger = logger
    self._container = None

  def __del__(self):
    if self._container is not None:
      self._container.close()

  def read_stream(self, aud_stream_idx: int = None) \
    -> Tuple[Generator[np.ndarray, None, None], Dict[StreamInfo, object]]:
    """Creates a generator of audio frames of a given audio stream in the recording

    Args:
        aud_stream_idx (int, optional): Index of the audio stream (0 -> first, 1 -> second). \
          Defaults to None (the first audio stream)

    Returns:
        Tuple[Generator[np.ndarray, None, None], Dict[StreamInfo, object]]: A generator of the \
          audio frames and a dictionary with info about the stream
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

    return (audio_iter(), AudioReader.prepare_stream_info(self._container, aud_stream_idx))

  @staticmethod
  def prepare_stream_info(container: av.container.InputContainer, aud_stream_idx: int) \
    -> Dict[StreamInfo, object]:
    """Creates a dictionary with info about a specific audio stream in the container

    Args:
        container (av.container.InputContainer): A container with the audio stream
        aud_stream_idx (int): The index of the audio stream (0 -> first, 1 -> second)

    Returns:
        Dict[StreamInfo, object]: A dictionary with info about the stream
    """
    astream = container.streams.audio[aud_stream_idx]
    return {
        StreamInfo.SAMPLE_RATE: astream.sample_rate,
        StreamInfo.FRAME_SIZE: astream.codec_context.frame_size
    }


def read_entire_audio(file_path: str,
                      aud_stream_idx: int = None,
                      aud_format: str = 'f32le',
                      logger: Logger = NULL_LOGGER) -> Tuple[np.ndarray, Dict[StreamInfo, object]]:
  """Reads an entire audio stream from a recording

  Args:
      file_path (str): Path to the recording
      aud_stream_idx (int, optional): Index of the audio stream (0 -> first, 1 -> second). \
        Defaults to None (the first audio stream)
      aud_format (str, optional): A desired audio format. Defaults to 'f32le'
      logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER

  Returns:
      Tuple[np.ndarray, Dict[StreamInfo, object]]: The entire audio stream in the specified format \
        and a dictionary with info about the stream
  """
  file_path = str(Path(file_path).resolve())
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
      data = np.frombuffer(buffer, dtype).reshape((astream.channels, -1))
    else:
      data = np.frombuffer(buffer, dtype).reshape((-1, astream.channels)).T

    return (data, AudioReader.prepare_stream_info(container, aud_stream_idx))
