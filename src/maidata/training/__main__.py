"""Explicit configuration overlay; ordinary component entry stays independent."""
import argparse
import json
from pathlib import Path

from maidata.entry import CONFIG


def load_config(path):
    config = dict(CONFIG)
    if path:
        overrides = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
        config.update(overrides)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent whole-song Encoder/Draft/R/P pretraining')
    parser.add_argument('--config', type=Path, help='JSON overrides for entry.CONFIG, including asset/output paths')
    parser.add_argument('--resume', type=Path, help='strict completed-epoch checkpoint')
    parser.add_argument('--cuda-check', action='store_true', help='one unchanged logical batch on a disposable model')
    parser.add_argument('--show-config', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.show_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    from .runner import run
    run(config, resume=args.resume, cuda_check=args.cuda_check)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
