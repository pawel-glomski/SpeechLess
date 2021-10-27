from abc import ABC, abstractmethod
from typing import List, Type
from enum import Enum, auto
from logging import Logger

from speechless.edit_context import TimelineChange


class AnalysisDomain(Enum):
  TEXT = auto()
  AUDIO = auto()
  VIDEO = auto()


class AnalysisMethod(ABC):

  def __init__(self, name: str, domain: List[AnalysisDomain], logger: Logger):
    self.name = name
    self.domain = domain
    self.logger = logger
    assert isinstance(domain, list)

  @abstractmethod
  def analyze(self, recording_path: str, subtitles_path: str = None) \
    -> List[TimelineChange]:
    """Analyzes the recording (and subtitles if provided) and generates timeline changes for the
    editor

    Args:
        recording_path (str): Path to the recording
        subtitles_path (str, optional): Path to the subtitles. Defaults to None.

    Returns:
        List[TimelineChange]: Timeline changes for the editor
    """
    raise NotImplementedError()


############################################### CLI ################################################

ARG_PREPARE_METHOD = 'prepare_fn'
ANALYSIS_METHODS = []


def analysis_method_cli(method_class: Type) -> Type:
  """Registers a new analysis method in the CLI. A method, to be properly registered, must be
  defined in a module located in the analysis module

  Args:
      method_class (Type): A class (acting as a namespace) with proper static functions and \
        attributes defined

  Returns:
      Type: The provided (unchanged) class
  """
  assert hasattr(method_class, 'prepare_method') and hasattr(method_class, 'setup_arg_parser')

  ANALYSIS_METHODS.append(method_class)
  return method_class
