"""Offline mirror-lr repair for any prepared pilot/full dataset; never edits inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full_dataset_tools.lr_repair import audit, build, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'build', 'verify'))
    parser.add_argument('--dataset-manifest', type=Path, required=True,
                        help='Source dataset.json, or repaired dataset.json for verify')
    parser.add_argument('--output', type=Path, help='New dataset directory, required for build')
    parser.add_argument('--reuse', choices=('hardlink', 'copy'), default='hardlink',
                        help='Reuse immutable payloads; hardlink requires the same filesystem')
    parser.add_argument('--reuse-dataset', type=Path,
                        help='Verified repaired pilot dataset whose overlapping mirror-lr chunks are reused')
    args = parser.parse_args(argv)
    if args.action == 'build' and args.output is None:
        parser.error('build requires --output')
    if args.action != 'build' and (args.output is not None or args.reuse_dataset is not None):
        parser.error('--output and --reuse-dataset only apply to build')
    if args.action == 'audit':
        result = audit(args.dataset_manifest)
    elif args.action == 'build':
        result = build(args.dataset_manifest, args.output, reuse=args.reuse,
                       reuse_manifest=args.reuse_dataset)
    else:
        result = verify(args.dataset_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
