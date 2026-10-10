"""Compatibility command builder: one explicitly selected training stage.

Cache preparation and generation remain explicit tools. This module does not
construct A/cache/B queues, run diagnostics, or create acceptance credentials.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from maimai_baseline.experiment import CONFIG
from maimai_baseline.training.paths import PATH_KEYS, resolve_training_paths

SOURCE = Path(__file__).resolve().parents[3]


def flags(**values):
    return [item for key, value in values.items() if value is not None
            for item in ('--' + key.replace('_', '-'), str(value))]


def input_paths(args):
    copied = SimpleNamespace(**vars(args))
    resolve_training_paths(copied, getattr(args, 'stage', 'A'))
    return {key: getattr(copied, key) for key in PATH_KEYS if getattr(copied, key, None) is not None}, getattr(copied, 'dataset_manifest', None)


def a_flags(args, output):
    values = {key: getattr(args, key, None) for key in (*PATH_KEYS, 'dataset_root', 'dataset_manifest')}
    values.update(experiment_config=getattr(args, 'experiment_config', CONFIG), output=output,
                  device=getattr(args, 'device', 'cuda'), resume=getattr(args, 'resume', None))
    return flags(**values)


def commands(args):
    stage = getattr(args, 'stage', 'A')
    values = a_flags(args, args.output)
    return [(stage, [sys.executable, '-u', '-m', 'maimai_baseline', 'train', '--stage', stage, *values])]


def execute(args, *, command_builder=None, completion=None):
    steps = (command_builder or commands)(args)
    # Retained only for callers that print/build one command. Do not silently
    # resurrect old multi-stage execution through this compatibility adapter.
    if len(steps) != 1:
        raise ValueError('run one explicit stage; prepare caches and start other stages separately')
    name, command = steps[0]
    code = subprocess.call(command, cwd=SOURCE)
    if code:
        raise SystemExit(code)
    return dict(stage=name, exit_code=code)


def parser(selection=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('A', 'B', 'sft'), default='A')
    parser.add_argument('--experiment-config', type=Path, default=CONFIG)
    for key in dict.fromkeys((*PATH_KEYS, 'dataset_root', 'dataset_manifest')):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=key == 'epoch_plan_cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda', choices=('auto', 'cuda', 'cpu'))
    parser.add_argument('--execute', action='store_true', help='execute the printed single-stage command')
    return parser


def main(selection=None):
    args = parser().parse_args()
    print(json.dumps(execute(args) if args.execute else commands(args), indent=2))
