"""Build grouped frozen features without changing the model runtime ABI."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.cli import parser
from global_local_cache import prepare


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--cache-token-budget", type=int, default=4096)
    p.add_argument("--cache-workers", type=int, default=12)
    p.add_argument("--cache-cpu-threads", type=int, default=1)
    options, rest = p.parse_known_args(argv)
    args = parser().parse_args(rest)
    if args.operation not in ("style-cache", "style-p0-cache"):
        raise ValueError("this producer only builds chart features or P0 caches")
    return args, options


def main(argv=None):
    args, options = parse_args(argv)
    producer = prepare
    if args.operation == "style-p0-cache":
        from style_p0_cache import prepare as producer
    value = producer(args, token_budget=options.cache_token_budget, workers=options.cache_workers,
                    cpu_threads=options.cache_cpu_threads)
    print(json.dumps(value), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
