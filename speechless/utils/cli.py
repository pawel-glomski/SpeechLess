import argparse

FORMATTER_CLASS = argparse.ArgumentDefaultsHelpFormatter
SUBCOMMANDS = []


def cli_subcommand(subcommand):
  assert hasattr(subcommand, 'setup_arg_parser')

  SUBCOMMANDS.append(subcommand)
  return subcommand
