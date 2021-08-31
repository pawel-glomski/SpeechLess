import logging
import argparse

import speechless

SUBMODULES = [speechless.editor, speechless.downloader]


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  subparsers = parser.add_subparsers(title='submodule')
  for submodule in SUBMODULES:
    submodule.setup_arg_parser(subparsers.add_parser(submodule.NAME, help=submodule.DESCRIPTION))
  args = parser.parse_args()

  if len(args.__dict__) > 0 and hasattr(args, 'run'):
    # TODO: set verbosity level
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler()],
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args.run(args, logging.getLogger())
  else:
    parser.print_help()


if __name__ == '__main__':
  main()
