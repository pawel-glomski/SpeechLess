import deepspeech
import numpy
import os
import wget

from typing import List

from speechless.utils.storage import make_cache_dir_rel
from speechless.processing.tokenization import EditToken

SCORER_URL = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer'
MODEL_URL = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm'

DEEPSPEECH_CACHE_DIR = make_cache_dir_rel('deepspeech')
SCORER_FILE = str(DEEPSPEECH_CACHE_DIR / os.path.basename(SCORER_URL))
MODEL_FILE = str(DEEPSPEECH_CACHE_DIR / os.path.basename(MODEL_URL))


def get_deepspeech_resources():
  for file, url in [(SCORER_FILE, SCORER_URL), (MODEL_FILE, MODEL_URL)]:
    if not os.path.exists(file):
      wget.download(url, file)


def transcript_count_words(transcript: deepspeech.CandidateTranscript) -> dict:
  """Count occurrences of individual words in transcript

  This function counts occurrences of each separate words. It returns
  a dictionary, in wich keys correspond to words and values
  to the number of occurrences.

  Args:
      transcript (deepspeech.CandidateTranscript): Transcript for the counting.

  Returns:
      dict: A dictionary, containing numbers of occurrences.
  """
  tokens = transcript.tokens
  words = dict()
  for token in tokens:
    word = token.text
    print(word)
    if word not in words:
      words.update({word: 1})
    else:
      value = words.get(word)
      words.update({word: value + 1})
  return words


def transcript_to_string(transcript: deepspeech.CandidateTranscript) -> str:
  """Convert transcript to string

  Args:
      transcript (deepspeech.CandidateTranscript): Transcript to convert.

  Returns:
      str: Concatenated text from all the tokens from the transcript.
  """
  tokens = transcript.tokens
  s = ''
  for token in tokens:
    s += token.text
  return s


def string_count_words(string: str) -> dict:
  """Count occurrences of individual words in a given string

  This function counts occurrences of each separate words. It returns
  a dictionary, in wich keys correspond to words and values
  to the number of occurrences.

  Args:
      string (str): String for the counting.

  Returns:
      dict: A dictionary, containing numbers of occurrences.
  """
  word_list = string.split()
  words = {}
  for word in word_list:
    if word not in words:
      words.update({word: 1})
    else:
      value = words.get(word)
      words.update({word: value + 1})
  return words


def transcript_to_edit_tokens(transcript: deepspeech.CandidateTranscript) -> List[EditToken]:
  """Create a list of EditTokens from transcript

  Args:
      transcript (transcript: deepspeech.CandidateTranscript): Transcript to convert.

  Returns:
      List[EditToken]: List with tokens
  """
  tokens = []
  start_i = -1
  for i, token in enumerate(transcript.tokens[:-1]):
    if start_i == -1:
      start_i = i
    if token.text == ' ' and start_i != i:
      tokens.append(
          EditToken(''.join([t.text for t in transcript.tokens[start_i:i]]),
                    transcript.tokens[start_i].start_time, token.start_time))
      start_i = -1
  tokens.append(
      EditToken(
          ''.join([t.text for t in transcript.tokens[start_i:]]),
          transcript.tokens[start_i].start_time, transcript.tokens[-1].start_time +
          (transcript.tokens[-1].start_time - transcript.tokens[-2].start_time)))
  return tokens


def speech_to_text(audio: numpy.array) -> List[EditToken]:
  """Perform a speech to text transcription

  Args:
      audio (numpy.array): A 16-bit, mono raw audio signal.

  Returns:
      List[EditToken]: A transcript containing recognized words and their timestamps.
  """
  get_deepspeech_resources()
  model = deepspeech.Model(MODEL_FILE)
  model.enableExternalScorer(SCORER_FILE)
  return transcript_to_edit_tokens(model.sttWithMetadata(audio).transcripts[0])


def remove_characters(s: str, characters: str) -> str:
  """Remove given characters from string

  Args:
      s (str): String to remove characters from.
      chracters (str): Character set containing characters to remove.

  Returns:
      str: Copy of a given string, with specified characters removed.
  """
  for c in characters:
    s = s.replace(c, '')
  return s


def load_and_adjust_script(file: str) -> str:
  """Load text from file and adjust it for comparison with transcript

  Args:
      file (str): Path to file.

  Returns:
      str: Adjusted text.
  """
  content = ''
  with open(file, encoding='UTF-8') as f:
    content = f.read()
    content = content.lower()
    content = content.replace('\n', ' ')
    content = remove_characters(content, ',.?!:"*()')
    content = content.replace('  ', ' ')
  return content


def test(transcript: deepspeech.CandidateTranscript, compare_to: str) -> float:
  """Test transcription accuracy by comparing it with another text

  Args:
      transcipt (deepspeech.CandidateTranscript): Transcript to test.
      compare_to (str): Path to text file to use for comparison.

  Returns:
      float: Value from range <0, 1>, where 1 represents complete similarity.
  """
  # dictionary1 = transcript_count_words(transcript)
  text = transcript_to_string(transcript)
  dictionary1 = string_count_words(text)
  dictionary2 = string_count_words(load_and_adjust_script(compare_to))
  result = 1.0

  length = len(dictionary2)
  for key, value in dictionary1.items():
    value2 = dictionary2.get(key, 0)
    if value != value2:
      result = result * \
          (1.0 - (abs(value - value2)/max(value, value2)/length))
  return result
