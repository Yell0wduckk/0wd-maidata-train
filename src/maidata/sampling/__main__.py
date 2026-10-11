"""Build fixed-selection window and occurrence reference files."""

import argparse
from pathlib import Path

from maidata.entry import CONFIG
from .plan import CONFIG_KEYS, build_plan, write_plan


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m maidata.sampling', description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chart-cache', type=Path, help='override the cache root recorded by selection')
    parser.add_argument('--split', type=Path, help='use an existing encoder/draft split; never resample songs')
    parser.add_argument('--partition', choices=('all', 'train', 'validation'), default='all',
                        help='validation keeps base windows only; train/validation require --split')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--epoch', type=int, default=1)
    for key in CONFIG_KEYS:
        parser.add_argument('--' + key.replace('_', '-'), type=float if key.endswith('multiplier') else int,
                            default=CONFIG[key], help=f'default: entry.CONFIG[{key!r}]')
    args = parser.parse_args(argv)
    config = {key: getattr(args, key) for key in CONFIG_KEYS}
    try:
        plan = build_plan(args.selection, config=config, seed=args.seed, epoch=args.epoch,
                          chart_cache=args.chart_cache, split=args.split, partition=args.partition)
        paths = write_plan(plan, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    totals = plan['totals']
    print(f"{totals['charts']} charts, {totals['base_windows']} base windows, "
          f"{totals['complex_windows']} complex windows, {totals['occurrences']} occurrences, "
          f"{totals['audio_groups']} audio groups")
    print(paths['plan'].resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
