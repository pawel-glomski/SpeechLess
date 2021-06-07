from pathlib import Path
import deepspeech
import numpy

MODEL_PATH = (Path.cwd()/("../models" if Path.cwd().name == "src" else "models"
                          )/"deepspeech-0.9.3-models.pbmm").resolve()
SCORER_PATH = (Path.cwd()/("../models" if Path.cwd().name == "src" else "models"
                           )/"deepspeech-0.9.3-models.scorer").resolve()


def transcriptCountWords(transcript: deepspeech.CandidateTranscript) -> dict:
    """Count occurrences of individual words in :class:`deepspeech.CandidateTranscript`

    This function counts occurrences of each separate words. It returns
    a dictionary, in wich keys correspond to words and values
    to the number of occurrences.

    :param deepspeech.CandidateTranscript transcript: Transcript for the counting.
    :return: A dictionary, containing numbers of occurrences.
    :rtype: :class:`dict`
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


def transcriptToString(transcript: deepspeech.CandidateTranscript) -> str:
    """Convert transcript to string

    :param deepspeech.CandidateTranscript transcript: Transcript to convert.
    :return: Concatenated text from all the tokens from the transcript.
    :rtype: :class:`str`
    """
    tokens = transcript.tokens
    s = ""
    for token in tokens:
        s += token.text
    return s


def stringCountWords(str: str) -> dict:
    """Count occurrences of individual words in a given string

    This function counts occurrences of each separate words. It returns
    a dictionary, in wich keys correspond to words and values
    to the number of occurrences.

    :param str str: String for the counting.
    :return: A dictionary, containing numbers of occurrences.
    :rtype: :class:`dict`
    """
    list = str.split()
    words = dict()
    for word in list:
        if word not in words:
            words.update({word: 1})
        else:
            value = words.get(word)
            words.update({word: value + 1})
    return words


def speechToText(audio: numpy.array) -> deepspeech.CandidateTranscript:
    """ Perform a speech to text transcription

    :param numpy.array audio: A 16-bit, mono raw audio signal.
    :return: A transcript object containing recognized words and their timestamps.
    :rtype: :class:`deepspeech.CandidateTranscript`
    """
    model = deepspeech.Model(str(MODEL_PATH))
    model.enableExternalScorer(str(SCORER_PATH))
    return model.sttWithMetadata(audio).transcripts[0]


def removeCharacters(s: str, characters: str) -> str:
    """ Remove given characters from string

    :param str s: String to remove characters from.
    :param str chracters: Character set containing characters to remove.
    :return: Copy of a given string, with specified characters removed.
    :rtype: :class:`str`
    """
    for c in characters:
        s = s.replace(c, "")
    return s


def loadAndAdjustScript(file: str) -> str:
    """ Load text from file and adjust it for comparison with transcript

    :param str file: Path to file.
    :return: Adjusted text.
    :rtype: :class:`str`
    """
    content = ""
    with open(file) as f:
        content = f.read()
        content = content.lower()
        content = content.replace("\n", " ")
        content = removeCharacters(content, ',.?!:"*()')
        content = content.replace("  ", " ")
    return content


def test(transcript: deepspeech.CandidateTranscript, compareTo: str) -> float:
    """ Test transcription accuracy by comparing it with another text

    :param deepspeech.CandidateTranscript transcipt: Transcript to test.
    :param str compareTo: Path to text file to use for comparison.
    :return: Value from range <0, 1>, where 1 represents complete similarity.
    :rtype: :class:`float`
    """
    # dictionary1 = transcriptCountWords(transcript)
    text = transcriptToString(transcript)
    dictionary1 = stringCountWords(text)
    dictionary2 = stringCountWords(loadAndAdjustScript(compareTo))
    result = 1.0

    length = len(dictionary2)
    for key, value in dictionary1.items():
        value2 = dictionary2.get(key, 0)
        if value != value2:
            result = result * \
                (1.0 - (abs(value - value2)/max(value, value2)/length))
    return result
