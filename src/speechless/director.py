import json

from logging import Logger
from pathlib import Path
from argparse import ArgumentParser

from speechless.editor import Editor

from .utils.cli import cli_subcommand, FORMATTER_CLASS
from .processing.analysis import ANALYSIS_METHODS, ARG_PREPARE_METHOD


@cli_subcommand
class CLI:
  COMMAND = 'direct'
  DESCRIPTION = 'Directs and produces summarized recordings'
  ARG_SRC = 'src'
  ARG_DST = 'dst'
  ARG_SUBS = 'subs'
  ARG_CONFIG = 'config'
  ARG_RECURSIVE = 'recursive'
  ARG_NO_EDIT = 'no_edit'
  ARG_METHOD = 'method'
  DEFAULT_ARGS = {ARG_SUBS: '', ARG_CONFIG: '', ARG_RECURSIVE: False, ARG_NO_EDIT: False}

  @staticmethod
  def setup_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Sets up a CLI argument parser for this submodule

    Returns:
        ArgumentParser: Configured parser
    """
    parser.add_argument(CLI.ARG_SRC,
                        help='Path to the recording to summarize. If its a path to a directory, ' \
                          'every recording in the directory will be processed. Subtitle files ' \
                          'should then have the same name as their corresponding recordings',
                        type=str,
                        action='store')
    parser.add_argument(CLI.ARG_DST,
                        help='Directory for the summarized recordings',
                        type=str,
                        action='store')
    parser.add_argument('-s',
                        f'--{CLI.ARG_SUBS}',
                        help=f'Subtitle file. This is ignored when `{CLI.ARG_SRC}` is a directory',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_SUBS])
    parser.add_argument('-c',
                        f'--{CLI.ARG_CONFIG}',
                        help='Configuration file defining the format of the output recordings',
                        type=str,
                        action='store',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_CONFIG])
    parser.add_argument('-r',
                        f'--{CLI.ARG_RECURSIVE}',
                        help='Look recursively for recordings to summarize ' \
                          '(keeps folder structure)',
                        action='store_true',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_RECURSIVE])
    parser.add_argument('-n',
                        f'--{CLI.ARG_NO_EDIT}',
                        help='Instead of editing the recordings, only produces editing cfg files',
                        action='store_true',
                        default=CLI.DEFAULT_ARGS[CLI.ARG_NO_EDIT])

    analysis_methods = parser.add_subparsers(title='Analysis methods',
                                             dest=CLI.ARG_METHOD,
                                             required=True)
    for method in ANALYSIS_METHODS:
      method.setup_arg_parser(
          analysis_methods.add_parser(method.COMMAND,
                                      help=method.DESCRIPTION,
                                      formatter_class=FORMATTER_CLASS))
    parser.set_defaults(run=CLI.run_submodule)
    return parser

  @staticmethod
  def run_submodule(args: object, logger: Logger) -> None:
    """Runs this submodule

    Args:
        args (object): Arguments of this submodule (defined in setup_arg_parser)
        logger (Logger): Logger for messages
    """
    args = args.__dict__
    src = Path(args[CLI.ARG_SRC]).resolve()
    dst = Path(args[CLI.ARG_DST]).resolve()
    subs = Path(args[CLI.ARG_SUBS]).resolve()
    cfg = Path(args[CLI.ARG_CONFIG]).resolve()
    no_edit = args[CLI.ARG_NO_EDIT]
    assert dst.is_dir()

    method = args[ARG_PREPARE_METHOD](args, logger)

    if src.is_file():
      dir_files = list(dst.glob('*'))
      dst_path = dst / src.name
      idx = 1
      while dst_path in dir_files:
        dst_path = dst / (src.stem + f' ({idx})' + src.suffix)
        idx += 1

      tl_changes = method.analyze(str(src), str(subs) if subs.is_file() else None)
      if cfg.is_file():
        with open(cfg, encoding='UTF-8') as cfg_file:
          editor = Editor.from_json(json.load(cfg_file), logger)
      else:
        editor = Editor(logger=logger)

      if no_edit:
        editor.export_json(dst / (dst_path.stem + '.json'), tl_changes)
      else:
        editor.edit(str(src), tl_changes, str(dst_path))
    elif src.is_dir():
      raise NotImplementedError()

    for file_path in Path(src).glob('*'):
      if not file_path.is_file():
        continue
