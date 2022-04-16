from abc import ABC, abstractmethod
from typing import List
from logging import Logger

from speechless.edit_context import TimelineChange

ARG_PREPARE_ANALYSIS_METHOD_FN = 'prepare_fn'


class AnalysisMethod(ABC):

  def __init__(self, name: str, logger: Logger):
    self.name = name
    self.logger = logger

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
