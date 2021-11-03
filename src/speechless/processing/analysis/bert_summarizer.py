from logging import Logger
from typing import List
from argparse import ArgumentParser
from summarizer import Summarizer

from .analysis import AnalysisMethod, AnalysisDomain, analysis_method_cli, ARG_PREPARE_METHOD_FN
from speechless.processing.tokenization import make_timeline_changes, sentence_segmentation
from speechless.readers import read_subtitles
from speechless.utils.logging import NULL_LOGGER
from speechless.edit_context import TimelineChange


class BertSentenceAnalysis(AnalysisMethod):

  def __init__(self, ratio: float, logger: Logger = NULL_LOGGER):
    super().__init__('BERT-based sentence analysis', [AnalysisDomain.AUDIO], logger)
    self.summarizer = Summarizer('distilbert-base-uncased')
    self.ratio = ratio

  def analyze(self, recording_path: str, subtitles_path: str) -> List[TimelineChange]:
    if subtitles_path is None:
      raise NotImplementedError
    sentences = sentence_segmentation(read_subtitles(subtitles_path))
    document = ''.join([token.text for sentence in sentences for token in sentence])
    summary = self.summarizer(document, ratio=self.ratio)

    for sentence in sentences:
      sentence_text = ''.join([token.text for token in sentence])
      label = sentence_text in summary
      for token in sentence:
        token.label = not label
    return make_timeline_changes([token for sentence in sentences for token in sentence])


############################################### CLI ################################################


@analysis_method_cli
class CLI:
  COMMAND = 'bert_sentence'
  DESCRIPTION = 'BERT-based sentence analysis'
  ARG_RATIO = 'ratio'
  DEFAULT_ARGS = {ARG_RATIO: 0.7}

  @staticmethod
  def prepare_method(args, logger) -> 'BertSentenceAnalysis':
    return BertSentenceAnalysis(args[CLI.ARG_RATIO], logger=logger)

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument('-r',
                        f'--{CLI.ARG_RATIO}',
                        help='Threshold value: greater value = less aggresive cuts',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_RATIO])
    parser.set_defaults(**{ARG_PREPARE_METHOD_FN: CLI.prepare_method})
