"""Incrementally recompile selected old windows from exact original-v2 sources.

Only the old small manifest is read. Rounded v1/v2 tensor labels are never
decoded or used to infer ticks. Fresh export uses the same compile_window.
"""
import argparse
import json
from pathlib import Path

from .export_windows import export


def main():
    from maidata.entry import CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-id', action='append')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    config = dict(CONFIG, **json.loads(args.config.read_text(encoding='utf-8-sig')))
    config['decoder_train_window_cache'] = str(args.output)
    result = export(config, workers=args.workers, window_ids=args.window_id,
                    source_windows=args.source_windows)
    print(json.dumps(dict(schema=result['schema'], windows=len(result['windows']),
                          unsupported_windows=result['unsupported_windows'])), flush=True)


if __name__ == '__main__':
    main()
