import argparse
import numpy as np
import ffmpeg
from typing import List

import logging
from logging import Logger
from pathlib import Path
from speechless.transcription import speech_to_text, transcript_to_tokens
from speechless.utils import NULL_LOGGER, Real, Token
from tfidf import Tfidf

SAMPLE_RATE = 22050


def load_signal(file_name):
  out, _ = (ffmpeg.input(file_name).output(
      '-', format='s16le', acodec='pcm_s16le', ac=1,
      ar=SAMPLE_RATE).overwrite_output().run(capture_stdout=True))
  out = np.frombuffer(out, np.int16)
  return out


class Director:

  def __init__(self, logger: Logger = NULL_LOGGER, sentence_threshold: Real = 0.7) -> None:
    self.logger = logger
    self.sentence_threshold = sentence_threshold

  def run(self, input_file: Path, method: str):
    if method == 'tfidf':
      audio = load_signal(input_file)
      transcript = speech_to_text(audio)
      tokens = transcript_to_tokens(transcript)
      for t in tokens:
        print(f'{t}: {t.timestamps}')
      sentences = self.split_into_sentences(tokens)
      for s in sentences:
        for w in s:
          print(f'{w}', end=', ')
        print('')
      tfidf = Tfidf(logger=self.logger)
      changes = tfidf.classify(sentences)
      print(changes)
      #TODO run editor
    else:
      print('not supported')

  def split_into_sentences(self, tokens: List[Token]) -> List[List[Token]]:
    sentences = list()
    sentence = list()
    for i in range(len(tokens) - 1):
      end = tokens[i].end
      start = tokens[i + 1].start
      assert end < start
      sentence.append(tokens[i])
      if start - end > self.sentence_threshold:
        sentences.append(sentence.copy())
        sentence.clear()
    sentence.append(tokens[-1])
    sentences.append(sentence.copy())
    return sentences


############################################### CLI ################################################

NAME = 'classifier'
DESCRIPTION = 'TODO'
ARG_METHOD = 'method'
ARG_INPUT = 'input'
DEFAULT_ARGS = {}


def setup_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  """Sets up a CLI argument parser for this submodule

  Returns:
      argparse.ArgumentParser: Configured parser
  """
  parser.description = DESCRIPTION
  parser.add_argument(ARG_METHOD, help='Method to use when classifying', type=str, action='store')
  parser.add_argument(ARG_INPUT,
                      help='Path of the audio/video file to classify',
                      type=Path,
                      action='store')
  parser.set_defaults(run=run_submodule)
  return parser


def run_submodule(args: object, logger: Logger) -> None:
  """Runs this submodule

  Args:
      args (object): Arguments of this submodule (defined in setup_arg_parser)
      logger (Logger): Logger for messages
  """
  # TODO
  args = args.__dict__
  director = Director(logger)
  director.run(args[ARG_INPUT], args[ARG_METHOD])


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      handlers=[logging.StreamHandler()],
                      format='%(asctime)s %(levelname)s %(message)s')
  mylogger = logging.getLogger()
  mydirector = Director(mylogger)
  mydirector.run('/opt/workspace/SpeechLess/nagranie1.wav', 'tfidf')
