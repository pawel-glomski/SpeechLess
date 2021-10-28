import argparse

from typing import Type

SUBCOMMANDS = []
FORMATTER_CLASS = argparse.ArgumentDefaultsHelpFormatter


def cli_subcommand(subcommand_class: Type) -> Type:
  """Registers a CLI subcommand. A subcommand, to be properly registered, must be defined in a
  module located at the root, so it can be automatically imported in the main.py

  Args:
      subcommand_class (Type): A class (acting as a namespace) with proper static functions and \
        attributes defined

  Returns:
      Type: The provided (unchanged) class
  """
  assert hasattr(subcommand_class, 'setup_arg_parser')

  SUBCOMMANDS.append(subcommand_class)
  return subcommand_class
