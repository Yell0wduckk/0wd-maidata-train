"""Editable configuration and command-line entry point."""

import argparse
from pprint import pprint
import sys

from .model import build_model

CONFIG = {}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maidata",
        description="Maidata model scaffold; training is not implemented.",
    )
    parser.add_argument("--show-config", action="store_true", help="show the configuration in entry.py")
    args = parser.parse_args(argv)
    if args.show_config:
        pprint(CONFIG, sort_dicts=False)
        return 0
    try:
        build_model(CONFIG)
    except NotImplementedError as error:
        print(error, file=sys.stderr)
        return 1
    return 0
