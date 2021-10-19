import logging
import argparse
from pathlib import Path
from importlib import import_module

from speechless.utils.cli import SUBCOMMANDS

# import modules: speechless.*
for module_file in Path(__file__).parent.glob('*'):
  if module_file.is_file() and str(module_file).endswith('.py') and '__' not in str(module_file):
    module = f'speechless.{module_file.stem}'
    if module != __name__:
      import_module(module)


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  subcommands = parser.add_subparsers()
  for subcommand in SUBCOMMANDS:
    subcommand.setup_arg_parser(
        subcommands.add_parser(subcommand.COMMAND,
                               help=subcommand.DESCRIPTION,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter))
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
