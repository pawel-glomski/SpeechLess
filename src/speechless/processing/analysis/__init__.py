from .analysis import AnalysisDomain, AnalysisMethod, ANALYSIS_METHODS, ARG_PREPARE_METHOD

from pathlib import Path
from importlib import import_module

# import all analysis methods
for module_file in Path(__file__).parent.glob('*'):
  if module_file.is_file() and str(module_file).endswith('.py') and '__' not in str(module_file):
    import_module(f'{__name__}.{module_file.stem}')
