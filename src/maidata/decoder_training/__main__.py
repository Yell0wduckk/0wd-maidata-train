"""CLI imports Torch only when executing, after help/config processing."""
import argparse
import json
from pathlib import Path
from maidata.entry import CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage II: freeze E/D/P, train Decoder and its adapters.')
    parser.add_argument('--config',type=Path,help='JSON overrides for entry.CONFIG')
    parser.add_argument('--resume',type=Path,help='complete-epoch Decoder checkpoint (not stage-I initialization)')
    parser.add_argument('--show-config',action='store_true')
    parser.add_argument('--cuda-check',action='store_true',help='one actual configured update; no training checkpoint')
    parser.add_argument('--check-step',type=int,default=0,help='zero-based first-epoch logical batch for CUDA check')
    args = parser.parse_args(argv)
    config = dict(CONFIG)
    if args.config:
        values = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if set(values)-set(config):
            parser.error('unknown configuration keys: '+str(sorted(set(values)-set(config))))
        config.update(values)
    if args.show_config:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        return 0
    from .runner import run
    run(config,resume=args.resume,cuda_check=args.cuda_check,check_step=args.check_step)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
