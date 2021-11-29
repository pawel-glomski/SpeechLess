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

N_FFT = 2048


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
    sig, stream_info = read_entire_audio(recording_path, logger=self.logger)
    scored_segments = SpectrogramAnalysis.optimize_signal(sig[0], stream_info, self.th_ratio)
    segment_size = stream_info[StreamInfo.FRAME_SIZE] / stream_info[StreamInfo.SAMPLE_RATE]
    changes = ranges_of_truth(scored_segments != 1) * segment_size
    changes = np.concatenate([changes, np.ones((changes.shape[0], 1)) * self.dur_multi], axis=1)
    return TimelineChange.from_numpy(changes)

  @staticmethod
  def optimize_signal(sig: np.ndarray, stream_info: Dict[StreamInfo, object], th_ratio: float) \
    -> np.ndarray:
    SEG_LEN = stream_info[StreamInfo.FRAME_SIZE]

    min_true_seq = 6
    min_false_seq = 4

    sig = sig[:int(len(sig) / SEG_LEN) * SEG_LEN].reshape((-1, SEG_LEN))
    indicator = SpectrogramAnalysis.calc_indicator(sig.reshape(-1), stream_info)
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
