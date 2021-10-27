import librosa
import numpy as np

from logging import Logger
from typing import Dict, List
from argparse import ArgumentParser

from .analysis import AnalysisMethod, AnalysisDomain, analysis_method_cli, ARG_PREPARE_METHOD
from speechless.readers import StreamInfo, read_entire_audio
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.math import ranges_of_truth
from speechless.edit_context import TimelineChange

N_FFT = 2048


class SpectrogramAnalysis(AnalysisMethod):

  def __init__(self, threshold: float, dur_multi: float, logger: Logger = NULL_LOGGER):
    super().__init__('Spectrogram Analysis', [AnalysisDomain.AUDIO], logger)
    self.threshold = threshold
    self.dur_multi = dur_multi
    self.logger = logger

  def analyze(self, recording_path: str, _) -> List[TimelineChange]:
    sig, stream_info = read_entire_audio(recording_path, logger=self.logger)
    scored_segments = SpectrogramAnalysis.optimize_signal(sig[0], stream_info, self.threshold)
    segment_size = stream_info[StreamInfo.FRAME_SIZE] / stream_info[StreamInfo.SAMPLE_RATE]
    changes = ranges_of_truth(scored_segments != 1) * segment_size
    return np.concatenate([changes, np.ones((changes.shape[0], 1)) * self.dur_multi], axis=1)

  @staticmethod
  def optimize_signal(sig: np.ndarray, stream_info: Dict[StreamInfo, object], threshold: float) \
    -> np.ndarray:
    SEG_LEN = stream_info[StreamInfo.FRAME_SIZE]

    min_true_seq = 6
    min_false_seq = 4

    sig = sig[:int(len(sig) / SEG_LEN) * SEG_LEN].reshape((-1, SEG_LEN))
    indicator = SpectrogramAnalysis.calc_indicator(sig.reshape(-1), stream_info)
    labels = indicator >= ((np.median(indicator) + np.mean(indicator)) / 2 * (1 / threshold))

    ##################### Reduce aggresive cuts #####################

    ranges = ranges_of_truth(labels == False)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= min_false_seq]
    labels[:] = True
    for r in ranges:
      labels[r[0]:r[1]] = False

    ##################### Cut out small segments #####################

    ranges = ranges_of_truth(labels)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= min_true_seq]
    labels[:] = False
    for r in ranges:
      labels[r[0]:r[1]] = True

    return labels

  @staticmethod
  def calc_indicator(sig: np.ndarray, stream_info: Dict[StreamInfo, object]) -> np.ndarray:
    SEG_LEN = stream_info[StreamInfo.FRAME_SIZE]
    SAMPLE_RATE = stream_info[StreamInfo.SAMPLE_RATE]

    # it generates one more for some reason
    y = librosa.power_to_db(
        librosa.feature.melspectrogram(sig, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=SEG_LEN).T)
    y -= np.median(y)
    y[y < 0] = 0
    y /= np.max(y)
    y1 = (y[1:, :] - y[:-1, :])**2
    y[:] = 0
    y[:-1, :] = y1
    y[1:, :] += y1
    y[1:-1, :] /= 2
    y = np.mean(y, axis=1)
    N = int(SAMPLE_RATE / SEG_LEN / 20)
    n = y.shape[0]
    y2 = y
    for i in range(2 * N + 1):
      if i != N:
        y2[N:-N] += y[i:n - 2 * N + i] / (2 * N + 1)
    return y2


############################################### CLI ################################################


@analysis_method_cli
class CLI:
  COMMAND = 'spectrogram'
  DESCRIPTION = 'Audio spectrogram analysis'
  ARG_THRESHOLD = 'threshold'
  ARG_DUR_MULTI = 'dur_multi'
  ARG_PAD = 'padding'
  DEFAULT_ARGS = {ARG_THRESHOLD: 1.5, ARG_DUR_MULTI: 0, ARG_PAD: 0}

  @staticmethod
  def prepare_method(args, logger) -> 'SpectrogramAnalysis':
    return SpectrogramAnalysis(args[CLI.ARG_THRESHOLD], args[CLI.ARG_DUR_MULTI], logger=logger)

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument('-t',
                        f'--{CLI.ARG_THRESHOLD}',
                        help='Threshold value: greater value = less aggresive cuts',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_THRESHOLD])
    parser.add_argument('-m',
                        f'--{CLI.ARG_DUR_MULTI}',
                        help='Duration multiplier of segments selected for removal',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_DUR_MULTI])
    parser.add_argument('-p',
                        f'--{CLI.ARG_PAD}',
                        help='Padding in seconds at the edges of removed segments',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_PAD])
    parser.set_defaults(**{ARG_PREPARE_METHOD: CLI.prepare_method})
