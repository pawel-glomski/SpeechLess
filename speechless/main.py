import logging
import argparse

from speechless import downloader, editor

SUBMODULES = ['editor', 'classifier', 'recogniser', 'downloader', 'trainer']


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='submodule')
    downloader.setupArgParser(subparsers.add_parser('downloader', help=downloader.DESCRIPTION))
    editor.setupArgParser(subparsers.add_parser('editor', help=editor.DESCRIPTION))
    # args = parser.parse_args()
    args = parser.parse_args()

    # TODO: set verbosity level
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler()],
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    args.run(args, logging.getLogger())


if __name__ == '__main__':
    main()
