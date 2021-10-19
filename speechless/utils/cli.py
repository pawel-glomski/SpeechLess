import argparse

FORMATTER_CLASS = argparse.ArgumentDefaultsHelpFormatter
SUBCOMMANDS = []


def cli_subcommand(submodule):
  SUBCOMMANDS.append(submodule)
  return submodule
