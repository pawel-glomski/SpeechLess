import librosa
import numpy as np

from logging import Logger
from typing import Dict, List
from argparse import ArgumentParser

from .analysis import AnalysisMethod, AnalysisDomain, analysis_method_cli, ARG_PREPARE_METHOD_FN
from speechless.readers import StreamInfo, read_entire_audio
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.math import ranges_of_truth
from speechless.edit_context import TimelineChange

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
CONV_N = int(SAMPLE_RATE / HOP_LENGTH / 20)


class SpectrogramAnalysis(AnalysisMethod):

  def __init__(self, th_ratio: float, dur_multi: float, logger: Logger = NULL_LOGGER):
    """Spectrogram based analysis method. This method looks at features of 2 adjacent timesteps and
    removes segments, for which the difference is small, thus reducing redundance in the signal.
    This method will remove silence and prolongations of sounds, syllables, words, or phrases.

    Args:
        th_ratio (float): Threshold ratio: greater value = more aggresive cuts
        dur_multi (float): Duration multiplier of segments selected for removal
        logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER.
    """
    super().__init__('Spectrogram Analysis', [AnalysisDomain.AUDIO], logger)
    self.th_ratio = th_ratio
    self.dur_multi = dur_multi
    self.logger = logger

  def analyze(self, recording_path: str, _) -> List[TimelineChange]:
    sig, _ = read_entire_audio(recording_path, sample_rate=SAMPLE_RATE, logger=self.logger)
    scored_segments = SpectrogramAnalysis.optimize_signal(sig[0], self.th_ratio)
    segment_time = HOP_LENGTH / SAMPLE_RATE
    changes = ranges_of_truth(scored_segments != 1) * segment_time
    changes = np.concatenate([changes, np.ones((changes.shape[0], 1)) * self.dur_multi], axis=1)
    return TimelineChange.from_numpy(changes)

  @staticmethod
  def optimize_signal(sig: np.ndarray, th_ratio: float) -> np.ndarray:
    min_true_seq = 12
    min_false_seq = 6

    sig = sig[:int(len(sig) / HOP_LENGTH) * HOP_LENGTH].reshape((-1, HOP_LENGTH))
    indicator = SpectrogramAnalysis.calc_indicator(sig.reshape(-1))
    labels = indicator >= (th_ratio * (np.median(indicator) + np.mean(indicator)) / 2)

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
  def calc_indicator(sig: np.ndarray) -> np.ndarray:
    y = librosa.power_to_db(
        librosa.feature.melspectrogram(sig, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH).T)
    y -= np.mean(y)
    y[y < 0] = 0
    y /= np.max(y)
    y[:-1, :] = (y[1:, :] - y[:-1, :])**2
    y[-1, :] = y[-2, :]
    y = np.mean(y, axis=1)

    y[CONV_N:-CONV_N] = np.convolve(y, np.ones(2 * CONV_N + 1), mode='valid') / (2 * CONV_N + 1)
    return y


############################################### CLI ################################################


@analysis_method_cli
class CLI:
  COMMAND = 'spectrogram'
  DESCRIPTION = """
    Looks at spectrogram features of 2 adjacent timesteps and removes segments, for which the
    difference is small, thus reducing redundance in the signal. This method will remove silence and
    prolongations of sounds, syllables, words, or phrases.""".replace('\n', '')
  ARG_TH_RATIO = 'th_ratio'
  ARG_DUR_MULTI = 'dur_multi'
  DEFAULT_ARGS = {ARG_TH_RATIO: 0.6667, ARG_DUR_MULTI: 0}

  @staticmethod
  def prepare_method(args, logger) -> 'SpectrogramAnalysis':
    return SpectrogramAnalysis(args.get(CLI.ARG_TH_RATIO, CLI.DEFAULT_ARGS[CLI.ARG_TH_RATIO]),
                               args.get(CLI.ARG_DUR_MULTI, CLI.DEFAULT_ARGS[CLI.ARG_DUR_MULTI]),
                               logger=logger)

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument('-tr',
                        f'--{CLI.ARG_TH_RATIO}',
                        help='Threshold ratio: greater value = more aggresive cuts',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_TH_RATIO])
    parser.add_argument('-m',
                        f'--{CLI.ARG_DUR_MULTI}',
                        help='Duration multiplier of segments selected for removal',
                        type=float,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_DUR_MULTI])
    parser.set_defaults(**{ARG_PREPARE_METHOD_FN: CLI.prepare_method})
