from .analysis import AnalysisMethod, ARG_PREPARE_ANALYSIS_METHOD_FN
from .spectrogram import SpectrogramAnalysis, CLI as SpectrogramCLI
from .tfidf import TfidfAnalysis, CLI as TfidfCLI

ANALYSIS_METHODS = [SpectrogramCLI, TfidfCLI]
