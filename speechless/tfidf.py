from logging import Logger
import math
from pathlib import Path
from typing import List
import numpy
import json
import itertools

from utils import NULL_LOGGER, Real, Token


class Tfidf:

  def __init__(self,
               input_json: Path = None,
               output_json: Path = None,
               logger: Logger = NULL_LOGGER) -> None:
    self.logger = logger
    self.output_json = output_json
    if input_json is not None:
      self.dictionaries = self.load_json(input_json)
      assert isinstance(self.dictionaries, List[dict])
    else:
      self.dictionaries = []

  def term_frequency(self, term: Token, document: dict) -> Real:
    return Real(document[str(term)] / sum(document.values()))

  def load_json(self, input_file: Path):
    with open(input_file, 'r') as file:
      return json.load(file)

  def save_json(self) -> None:
    if self.output_json is not None:
      with open(self.output_json, 'w') as file:
        json.dump(self.dictionaries, file)

  def build_dictionary(self, document: List[Token]) -> dict:
    d = {}
    for word in document:
      if d.get(str(word)) is None:
        d[str(word)] = 1.0
      else:
        d[str(word)] += 1.0
    return d

  def inverse_document_frequency(self, term: Token) -> float:
    denominator = 1.0
    for dic in self.dictionaries:
      if dic.get(str(term)) is not None:
        denominator += 1.0
    return math.log10((float)(len(self.dictionaries)) / denominator)

  def tfidf_dictionaries(self) -> List[dict]:
    tfidf_list = []
    for dic in self.dictionaries:
      tfidf = {}
      for term in dic.keys():
        tfidf[term] = (self.term_frequency(term, dic) * self.inverse_document_frequency(term))
      tfidf = dict(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))
      tfidf_list.append(tfidf.copy())
    return tfidf_list

  def classify(self,
               sentences: List[List[Token]],
               sentence_threshold: Real = 1.0,
               speed_ratio: Real = 3.0) -> numpy.array:
    new_document = self.build_dictionary(list(itertools.chain.from_iterable(sentences)))
    self.dictionaries.append(new_document)
    self.save_json()
    dicts = self.tfidf_dictionaries()
    tfidf_for_sentences = dicts[-1]
    changes = []
    for s in sentences:
      sentence_value = sum([tfidf_for_sentences[str(token)] for token in s]) / len(s)
      if sentence_value < sentence_threshold:
        changes.append([s[0].start, s[-1].end, speed_ratio * (sentence_threshold - sentence_value)])
    return numpy.array(changes)
