"""Independent shared-Encoder joint-training CLI; help/config need no Torch."""
import argparse
import json
from pathlib import Path

from maidata.entry import CONFIG
from .recipe import resolve


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train fresh E/D/R/P/M jointly with shared audio and live Z.')
    parser.add_argument('--config', type=Path, help='JSON overrides for the unique entry.CONFIG')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='complete-epoch joint checkpoint')
    parser.add_argument('--upgrade-z-normalization', action='store_true',
                        help='Explicitly enable parameter-free Z adapter normalization on a legacy joint resume')
    parser.add_argument('--upgrade-single-plan16', action='store_true',
                        help='Explicit centered-overlap to window-single Plan16 resume; preserve all training state')
    parser.add_argument('--show-config', action='store_true')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--cuda-check', action='store_true', help='one original batch correctness audit only')
    checks.add_argument('--cost-check', action='store_true', help='original batch cost comparison, no training')
    parser.add_argument('--check-step', type=int, help='zero-based first-epoch update; default largest group proxy')
    args = parser.parse_args(argv)
    config = dict(CONFIG)
    if args.config:
        values = json.loads(args.config.read_text(encoding='utf-8-sig'))
        unknown = set(values) - set(config)
        if unknown:
            parser.error('unknown configuration keys: ' + str(sorted(unknown)))
        config.update(values)
    if args.output:
        config['joint_train_output'] = str(args.output)
    if args.resume and (args.cuda_check or args.cost_check):
        parser.error('diagnostics use an independent scratch model, not a resume')
    if args.upgrade_z_normalization and not args.resume:
        parser.error('--upgrade-z-normalization requires --resume')
    if args.upgrade_single_plan16 and not args.resume:
        parser.error('--upgrade-single-plan16 requires --resume')
    if args.upgrade_single_plan16 and args.upgrade_z_normalization:
        parser.error('single Plan16 migration cannot also change Z normalization')
    if args.check_step is not None and args.check_step < 0:
        parser.error('--check-step must be nonnegative')
    config = resolve(config)
    if args.show_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    from .runner import run
    run(config, resume=args.resume, cuda_check=args.cuda_check,
        cost_check=args.cost_check, check_step=args.check_step,
        upgrade_z_normalization=args.upgrade_z_normalization,
        upgrade_single_plan16=args.upgrade_single_plan16)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
