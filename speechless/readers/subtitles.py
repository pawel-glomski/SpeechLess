import webvtt
from typing import Callable, List
from pathlib import Path

from speechless.processing.tokenization import EditToken

REGISTERED_READERS: List[Callable[[str], List[EditToken]]] = []


def read_subtitles(sub_path: str) -> List[EditToken]:
  path = Path(sub_path).resolve()
  for reader, extensions in REGISTERED_READERS:
    if path.suffix[1:].lower() in extensions:
      transcript = reader(sub_path)
      if transcript is not None:
        return transcript
  return None


def sub_reader(extensions: List[str]):

  def register(reader_func: Callable[[str], List[EditToken]]):
    assert isinstance(extensions, list)
    REGISTERED_READERS.append((reader_func, extensions))
    return reader_func

  return register


@sub_reader(extensions=['vtt'])
def vtt_reader(sub_path: str) -> List[EditToken]:
  vtt = webvtt.read(sub_path)
  transcript = [EditToken(c.text, c.start_in_seconds, c.end_in_seconds) for c in vtt.captions]
  transcript = [t for t in transcript if len(t.text) > 0]
  return transcript
