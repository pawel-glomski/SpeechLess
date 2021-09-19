from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import List

import gensim.downloader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from speechless.edit_context.common import TimelineChange
from speechless.processing.analysis.analysis import (ARG_PREPARE_METHOD, AnalysisDomain,
                                                     AnalysisMethod, analysis_method_cli)
from speechless.processing.tokenization import EditToken, sentence_segmentation
from speechless.readers.subtitles import vtt_reader
from speechless.utils.logging import NULL_LOGGER

DICTIONARY_FILE = 'tfidf_dictionary.dic'
MODEL_FILE = 'tfidf_model.model'


class TfidfAnalysis(AnalysisMethod):

  def __init__(self, dataset: str, sentence_threshold: float, logger: Logger = NULL_LOGGER) -> None:
    super().__init__('Tf-idf Analysis', [AnalysisDomain.TEXT])
    self.logger = logger
    self.sentence_threshold = sentence_threshold
    gensim.downloader.BASE_DIR = '/opt/workspace/SpeechLess/gensim-data'
    gensim.downloader.base_dir = '/opt/workspace/SpeechLess/gensim-data'
    if Path(DICTIONARY_FILE).exists() and Path(MODEL_FILE).exists():
      self.data_dict = Dictionary.load(DICTIONARY_FILE)
      self.model = TfidfModel.load(MODEL_FILE)
    else:
      data = gensim.downloader.load(dataset)
      self.data_dict = Dictionary(data)
      self.data_dict.save(DICTIONARY_FILE)
      corpus = [self.data_dict.doc2bow(line) for line in data]
      self.model = TfidfModel(corpus)
      self.model.save(MODEL_FILE)

  def analyze(self, recording_path: str, subtitles_path: str) -> List[TimelineChange]:
    if subtitles_path is None:
      raise NotImplementedError
    else:
      sentences = sentence_segmentation(vtt_reader(subtitles_path))
    tokens = self.set_labels(sentences)
    changes = [token.as_timeline_change(0.0) for token in tokens if token.label]
    return changes

  def set_labels(self, sentences: List[List[EditToken]]) -> List[List[EditToken]]:
    for s in sentences:
      tokens = [word for token in s for word in token.text.split()]
      bow = self.data_dict.doc2bow(tokens, allow_update=True)
      sentence_vector = self.model[bow]
      sentence_value = sum([val for id, val in sentence_vector]) / len(tokens)
      for t in s:
        t.label = float(sentence_value < self.sentence_threshold)
    return [token for s in sentences for token in s]

  def benchmark(self, transcript: List[EditToken]) -> List[float]:
    sentences = sentence_segmentation(vtt_reader(transcript))
    tokens = self.set_labels(sentences)
    changes = [token.label for token in tokens]
    return changes


############################################### CLI ################################################


@analysis_method_cli
class CLI:
  COMMAND = 'tfidf'
  DESCRIPTION = 'Tf-idf analysis'
  ARG_DATASET = 'dataset'
  ARG_SENTENCE_THRESHOLD = 'sentence_threshold'
  DEFAULT_ARGS = {ARG_DATASET: 'text8', ARG_SENTENCE_THRESHOLD: 0.2}

  @staticmethod
  def prepare_method(args) -> 'TfidfAnalysis':
    return TfidfAnalysis(args[CLI.ARG_DATASET], args[CLI.ARG_SENTENCE_THRESHOLD])

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument('-d',
                        f'--{CLI.ARG_DATASET}',
                        help='Dataset (TODO)',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_DATASET])
    parser.add_argument('-t',
                        f'--{CLI.ARG_SENTENCE_THRESHOLD}',
                        help='Sentence threshold',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_SENTENCE_THRESHOLD])
    parser.set_defaults(**{ARG_PREPARE_METHOD: CLI.prepare_method})
