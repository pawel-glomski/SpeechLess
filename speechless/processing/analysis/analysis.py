import numpy as np

from abc import ABC, abstractmethod
from typing import List
from enum import Enum, auto
from logging import Logger

from speechless.processing.tokenization import EditToken


class AnalysisDomain(Enum):
  TEXT = auto()
  AUDIO = auto()
  VIDEO = auto()


class AnalysisMethod(ABC):

  def __init__(self, name: str, domain: List[AnalysisDomain]):
    self.name = name
    self.domain = domain
    assert isinstance(domain, list)

  @abstractmethod
  def analyze(self, recording_path: str, sentences: List[List[EditToken]], logger: Logger) \
    -> np.ndarray:
    raise NotImplementedError()


############################################### CLI ################################################

ARG_PREPARE_METHOD = 'prepare_fn'
ANALYSIS_METHODS = []


def analysis_method_cli(method_class):
  assert hasattr(method_class, 'prepare_method') and hasattr(method_class, 'setup_arg_parser')

  ANALYSIS_METHODS.append(method_class)
  return method_class
