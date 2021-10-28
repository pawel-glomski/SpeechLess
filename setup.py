import sys

from setuptools import setup

SUPPORTED_LANGUAGES = ['en']


def download_spacy(_: str):
  import spacy  # pylint: disable=import-outside-toplevel
  from speechless.processing.tokenization import SPACY_MODEL  # pylint: disable=import-outside-toplevel
  spacy.cli.download(SPACY_MODEL)


if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] in SUPPORTED_LANGUAGES:
    download_spacy(sys.argv[1])
  else:
    setup()
