import gensim.downloader

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import List

from speechless.edit_context.common import TimelineChange
from speechless.processing.analysis.analysis import (ARG_PREPARE_METHOD_FN, AnalysisDomain,
                                                     AnalysisMethod, analysis_method_cli)
from speechless.processing.tokenization import EditToken, make_timeline_changes, sentence_segmentation
from speechless.readers.subtitles import read_subtitles
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.storage import make_cache_dir_rel

GENSIM_CACHE_DIR = make_cache_dir_rel('gensim')
CORPUS_DIR = str(GENSIM_CACHE_DIR / 'gensim-data/')
DICTIONARY_FILE = str(GENSIM_CACHE_DIR / 'tfidf_dictionary.dic')
MODEL_FILE = str(GENSIM_CACHE_DIR / 'tfidf_model.model')


class TfidfAnalysis(AnalysisMethod):

  def __init__(self, corpus: str, sentence_threshold: float, logger: Logger = NULL_LOGGER) -> None:
    super().__init__('Tf-idf Analysis', [AnalysisDomain.TEXT], logger)

    self.sentence_threshold = sentence_threshold
    gensim.downloader.BASE_DIR = CORPUS_DIR
    gensim.downloader.base_dir = CORPUS_DIR
    if Path(DICTIONARY_FILE).exists() and Path(MODEL_FILE).exists():
      self.data_dict = Dictionary.load(DICTIONARY_FILE)
      self.model = TfidfModel.load(MODEL_FILE)
    else:
      data = gensim.downloader.load(corpus)
      self.data_dict = Dictionary(data)
      self.data_dict.save(DICTIONARY_FILE)
      corpus = [self.data_dict.doc2bow(line) for line in data]
      self.model = TfidfModel(corpus)
      self.model.save(MODEL_FILE)

  def analyze(self, recording_path: str, subtitles_path: str) -> List[TimelineChange]:
    if subtitles_path is None:
      raise NotImplementedError
    else:
      sentences = sentence_segmentation(read_subtitles(subtitles_path))
    tokens = self.set_labels(sentences)
    return make_timeline_changes(tokens)

  def set_labels(self, sentences: List[List[EditToken]]) -> List[EditToken]:
    for s in sentences:
      tokens = [word for token in s for word in token.text.split()]
      bow = self.data_dict.doc2bow(tokens, allow_update=True)
      sentence_vector = self.model[bow]
      sentence_value = sum([val for id, val in sentence_vector]) / len(tokens)
      for t in s:
        t.label = float(sentence_value < self.sentence_threshold)
    return [token for s in sentences for token in s]

  def score_transcription(self, transcript: List[EditToken]) -> List[float]:
    sentences = sentence_segmentation(transcript)
    tokens = self.set_labels(sentences)
    changes = [token.label for token in tokens]
    return changes


############################################### CLI ################################################


@analysis_method_cli
class CLI:
  COMMAND = 'tfidf'
  DESCRIPTION = 'Tf-idf analysis'
  ARG_CORPUS = 'corpus'
  ARG_SENTENCE_THRESHOLD = 'sentence_threshold'
  DEFAULT_ARGS = {ARG_CORPUS: 'text8', ARG_SENTENCE_THRESHOLD: 0.2}

  @staticmethod
  def prepare_method(args, logger: Logger) -> 'TfidfAnalysis':
    return TfidfAnalysis(args[CLI.ARG_CORPUS], args[CLI.ARG_SENTENCE_THRESHOLD], logger=logger)

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument('-c',
                        f'--{CLI.ARG_CORPUS}',
                        help='Corpus from gensim',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_CORPUS])
    parser.add_argument('-t',
                        f'--{CLI.ARG_SENTENCE_THRESHOLD}',
                        help='Sentence threshold',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_SENTENCE_THRESHOLD])
    parser.set_defaults(**{ARG_PREPARE_METHOD_FN: CLI.prepare_method})
