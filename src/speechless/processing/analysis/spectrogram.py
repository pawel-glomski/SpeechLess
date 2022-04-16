import numpy as np

from typing import List
from logging import Logger
from argparse import ArgumentParser
from librosa.core import stft
from scipy.signal import convolve2d as conv2d
from scipy.stats import entropy as kl_div
from scipy.special import softmax

from .analysis import AnalysisMethod, ARG_PREPARE_ANALYSIS_METHOD_FN
from speechless.readers import read_entire_audio
from speechless.edit_context import TimelineChange
from speechless.utils.logging import NULL_LOGGER
from speechless.utils.math import ranges_of_truth, kernel_2d_from_window

SR = 16000
NFFT = 256
HOP = NFFT // 1
SPEC_POWER = 0.2
SPEC_NORM_MAX = 9

MIN_AMPLITUDE = 3
SOUND_POINT = 15  # number of freq with amplitude > 0 (after transformations)

REDUNDANCY_THRESHOLD = 0.025
MIN_SILENCE_LEN = int(0.1 * SR / HOP + 0.5)
MIN_SOUND_LEN = MIN_SILENCE_LEN


class SpectrogramAnalysis(AnalysisMethod):

  def __init__(self,
               th_ratio: float,
               dur_multi: float,
               silence_len: int,
               logger: Logger = NULL_LOGGER):
    """Spectrogram based analysis method. This method looks at features of 2 adjacent timesteps and
    removes segments, for which the difference is small, thus reducing redundance in the signal.
    This method will remove silence and prolongations of sounds, syllables, words, or phrases.

    Args:
        th_ratio (float): Threshold ratio: the greater the value, the more aggressive the cuts.
        dur_multi (float): Duration multiplier of segments selected for removal.
        silence_len (int): Desired length of silence as a number of segments.
        logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER.
    """
    super().__init__('Spectrogram Analysis', logger)
    self.th_ratio = th_ratio
    self.dur_multi = dur_multi
    self.silence_len = silence_len
    self.logger = logger

  def analyze(self, recording_path: str, _) -> List[TimelineChange]:
    signal, _ = read_entire_audio(recording_path, sample_rate=SR, mono=True, logger=self.logger)
    spec = SpectrogramAnalysis.make_spectrogram(signal[0])
    is_sound = SpectrogramAnalysis.find_sound_and_silence(spec)
    redundancy = SpectrogramAnalysis.redundancy(spec)
    cls_segments = self.classify(is_sound, redundancy)

    changes = ranges_of_truth(cls_segments == False) * HOP / SR
    changes = np.concatenate([changes, np.ones((changes.shape[0], 1)) * self.dur_multi], axis=1)
    return TimelineChange.from_numpy(changes)

  def classify(self, is_sound: np.ndarray, redundancy: np.ndarray) -> np.ndarray:
    cls = redundancy >= REDUNDANCY_THRESHOLD * self.th_ratio
    cls = cls & is_sound

    # filter out very small silence segments
    ranges = ranges_of_truth(cls == False)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= MIN_SILENCE_LEN]
    cls[:] = True
    for r in ranges:
      cls[r[0]:r[1]] = False

    # cut out very small sound segments
    ranges = ranges_of_truth(cls)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= MIN_SOUND_LEN]
    cls[:] = False
    for r in ranges:
      cls[r[0]:r[1]] = True

    # keep the silence
    silence_ranges = ranges_of_truth(is_sound == False)
    redundancy_ranges = ranges_of_truth(cls == False)
    red_idx = 0
    for silence_range in silence_ranges:
      sil_beg, sil_end = silence_range
      while red_idx < len(redundancy_ranges):
        red_beg, red_end = redundancy_ranges[red_idx]
        common_beg = max(sil_beg, red_beg)
        common_end = min(sil_end, red_end)
        if common_end - common_beg > 0:
          if common_end - common_beg > self.silence_len:
            cls[common_end - self.silence_len:common_end] = True
          red_idx += 1
        elif red_beg < sil_beg:
          red_idx += 1
        else:
          break

    return cls

  @staticmethod
  def make_spectrogram(signal: np.ndarray) -> np.ndarray:
    spec = np.abs(stft(signal, NFFT, HOP))
    spec = conv2d(spec**SPEC_POWER, kernel_2d_from_window((5, 5), np.hanning), 'same')
    spec = spec * (SPEC_NORM_MAX / spec.max())
    return spec

  @staticmethod
  def find_sound_and_silence(spec: np.ndarray) -> np.ndarray:
    is_sound = (spec > MIN_AMPLITUDE).sum(axis=0) > SOUND_POINT

    # filter out very small silence segments
    ranges = ranges_of_truth(is_sound == False)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= MIN_SILENCE_LEN]
    is_sound[:] = True
    for r in ranges:
      is_sound[r[0]:r[1]] = False

    # cut out very small sound segments
    ranges = ranges_of_truth(is_sound)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) >= MIN_SOUND_LEN]
    is_sound[:] = False
    for r in ranges:
      is_sound[r[0]:r[1]] = True

    return is_sound

  @staticmethod
  def redundancy(spec: np.ndarray) -> np.ndarray:
    spec = softmax(spec, axis=0)
    redundancy = np.zeros(spec.shape[1])
    redundancy[1:] = kl_div(spec[:, :-1], spec[:, 1:], axis=0)
    redundancy = np.convolve(redundancy, np.hanning(7), 'same')
    return redundancy


############################################### CLI ################################################


class CLI:
  COMMAND = 'spectrogram'
  DESCRIPTION = """
    Looks at 2 adjacent timesteps of a spectrogram and removes segments, for which the difference
    in features is small, thus reducing redundance in the signal. This method will remove silence
    and prolongations of sounds, syllables, words, or phrases.""".replace('\n', ' ')
  ARG_TH_RATIO = 'th_ratio'
  ARG_DUR_MULTI = 'dur_multi'
  ARG_SILENCE_LEN = 'silence_len'
  DEFAULT_ARGS = {ARG_TH_RATIO: 1.0, ARG_DUR_MULTI: 0, ARG_SILENCE_LEN: MIN_SILENCE_LEN}

  @staticmethod
  def prepare_method(args, logger) -> 'SpectrogramAnalysis':
    return SpectrogramAnalysis(args.get(CLI.ARG_TH_RATIO, CLI.DEFAULT_ARGS[CLI.ARG_TH_RATIO]),
                               args.get(CLI.ARG_DUR_MULTI, CLI.DEFAULT_ARGS[CLI.ARG_DUR_MULTI]),
                               args.get(CLI.ARG_SILENCE_LEN, CLI.DEFAULT_ARGS[CLI.ARG_SILENCE_LEN]),
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
    parser.add_argument('-s',
                        f'--{CLI.ARG_SILENCE_LEN}',
                        help='Desired length of silence as a number of segments',
                        type=int,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_SILENCE_LEN])
    parser.set_defaults(**{ARG_PREPARE_ANALYSIS_METHOD_FN: CLI.prepare_method})
