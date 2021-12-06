import gensim.downloader
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from typing import List

from speechless.edit_context.common import TimelineChange
from speechless.processing.analysis.analysis import (ARG_PREPARE_METHOD_FN, AnalysisDomain,
                                                     AnalysisMethod, analysis_method_cli)
from speechless.processing.tokenization import (EditToken, make_timeline_changes,
                                                sentence_segmentation, spacy_nlp)
from speechless.readers import read_subtitles, read_entire_audio
from speechless.transcription import speech_to_text
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.storage import make_cache_dir_rel

GENSIM_CACHE_DIR = make_cache_dir_rel('gensim')
CORPUS_DIR = str(GENSIM_CACHE_DIR / 'gensim-data/')
DICTIONARY_FILE = str(GENSIM_CACHE_DIR / 'tfidf_dictionary.dic')
MODEL_FILE = str(GENSIM_CACHE_DIR / 'tfidf_model.model')


class TfidfAnalysis(AnalysisMethod):

  def __init__(self,
               corpus: str,
               sent_th_ratio: float,
               remove_sw: bool,
               lemmatize: bool,
               logger: Logger = NULL_LOGGER) -> None:
    super().__init__('Tf-idf Analysis', [AnalysisDomain.TEXT], logger)

    self.sent_th_ratio = sent_th_ratio
    self.remove_sw = remove_sw
    self.lemmatize = lemmatize

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
      audio, _ = read_entire_audio(recording_path,
                                   aud_format='s16le',
                                   sample_rate=16000,
                                   logger=self.logger)
      transcript = speech_to_text(audio[0] if len(audio.shape) > 1 else audio)
    else:
      transcript = read_subtitles(subtitles_path)
    sentences = sentence_segmentation(transcript)
    tokens = self.set_labels(sentences)
    return make_timeline_changes(tokens)

  def set_labels(self, sentences: List[List[EditToken]]) -> List[EditToken]:
    spacy_pipes = ['tagger', 'attribute_ruler'] + (['lemmatizer'] if self.lemmatize else [])
    doc_text = ''.join([token.text for sentence in sentences for token in sentence])
    doc_tokens = spacy_nlp(doc_text, spacy_pipes)

    doc_tokens = [token for token in doc_tokens if not token.is_punct]
    if self.remove_sw:
      doc_tokens = [token for token in doc_tokens if not token.is_stop]
    if self.lemmatize:
      tfidf_doc = [token.lemma_ for token in doc_tokens]
    else:
      tfidf_doc = [token.norm_ for token in doc_tokens]

    sent_scores = [[] for _ in range(len(sentences))]
    if len(tfidf_doc) > 0:
      bow = self.data_dict.doc2bow(tfidf_doc,)
      doc_scores = self.model[bow]
      doc_scores = {self.data_dict[key]: score for key, score in doc_scores}

      sent_idx = -1
      sent_start, sent_end = 0, 0
      for token_idx, token in enumerate(doc_tokens):
        while sent_idx + 1 < len(sentences) and not (sent_start <= token.idx < sent_end):
          sent_idx += 1
          first_sent_token = sentences[sent_idx][0]
          last_sent_token = sentences[sent_idx][-1]
          sent_start = first_sent_token.start_pos
          sent_end = last_sent_token.start_pos + len(last_sent_token)
        if sent_idx >= len(sentences):
          break
        sent_scores[sent_idx].append(doc_scores.get(tfidf_doc[token_idx], 0.0))

    sent_scores = np.array([np.mean(s_sc) if len(s_sc) > 0 else 0 for s_sc in sent_scores])
    sent_scores = (sent_scores >= self.sent_th_ratio * np.mean(sent_scores)).astype(float)
    for sent_idx, sentence in enumerate(sentences):
      for t in sentence:
        t.label = sent_scores[sent_idx]

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
  ARG_SENT_TH_RATIO = 'sent_th_ratio'
  ARG_REMOVE_SW = 'remove_sw'
  ARG_LEMMATIZE = 'lemmatize'
  DEFAULT_ARGS = {
      ARG_CORPUS: 'text8',
      ARG_SENT_TH_RATIO: 1.0,
      ARG_REMOVE_SW: False,
      ARG_LEMMATIZE: False
  }

  @staticmethod
  def prepare_method(args, logger: Logger) -> 'TfidfAnalysis':
    return TfidfAnalysis(args.get(CLI.ARG_CORPUS, CLI.DEFAULT_ARGS[CLI.ARG_CORPUS]),
                         args.get(CLI.ARG_SENT_TH_RATIO, CLI.DEFAULT_ARGS[CLI.ARG_SENT_TH_RATIO]),
                         args.get(CLI.ARG_REMOVE_SW, CLI.DEFAULT_ARGS[CLI.ARG_REMOVE_SW]),
                         args.get(CLI.ARG_LEMMATIZE, CLI.DEFAULT_ARGS[CLI.ARG_LEMMATIZE]),
                         logger=logger)

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
    parser.add_argument('-tr',
                        f'--{CLI.ARG_SENT_TH_RATIO}',
                        help='Sentence threshold ratio. Sentences with a score lower than \
                          `mean sentence score`*`ratio` will be removed from the recording',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_SENT_TH_RATIO])
    parser.add_argument('-rsw',
                        f'--{CLI.ARG_REMOVE_SW}',
                        help='Remove stopwords',
                        action='store_true',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_REMOVE_SW])
    parser.add_argument('-l',
                        f'--{CLI.ARG_LEMMATIZE}',
                        help='Use lemmatization',
                        action='store_true',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_LEMMATIZE])
    parser.set_defaults(**{ARG_PREPARE_METHOD_FN: CLI.prepare_method})
