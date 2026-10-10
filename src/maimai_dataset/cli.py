from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

from maimai_dataset.encoding import run_timing_audit
from maimai_dataset.chart_cache import (
    build_chart_vector_cache,
    verify_chart_vector_cache,
)
from maimai_dataset.manifest import DatasetError

from .codec import build_profile, run_semantic_audit
from .geometry import build_geometry_sidecar


PROFILE = "v2-test1-all-difficulties"


def _profile(value: str) -> str:
    if value != PROFILE:
        raise argparse.ArgumentTypeError(
            f"{value!r} is archived and is not part of the canonical baseline; "
            f"expected {PROFILE!r}"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maimai-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("build-profile")
    profile.add_argument("--dataset", required=True)
    profile.add_argument("--profile", type=_profile, default=PROFILE)
    profile.add_argument("--output", required=True)
    profile.add_argument("--debug", action="store_true")
    semantic = subparsers.add_parser("semantic-audit")
    semantic.add_argument("--dataset", required=True)
    semantic.add_argument("--profile-root", required=True)
    semantic.add_argument("--output", required=True)
    semantic.add_argument("--debug", action="store_true")
    timing = subparsers.add_parser("timing-audit")
    timing.add_argument("--dataset", required=True)
    timing.add_argument("--profile", type=_profile, default=PROFILE)
    timing.add_argument("--output", required=True)
    timing.add_argument("--debug", action="store_true")
    geometry = subparsers.add_parser("build-interpolation-geometry")
    geometry.add_argument("--profile-root", required=True)
    geometry.add_argument("--chart-cache")
    geometry.add_argument("--feature-root")
    geometry.add_argument("--output", required=True)
    geometry.add_argument("--debug", action="store_true")
    chart_cache = subparsers.add_parser(
        "build-chart-cache",
        help="materialize pilot chart vectors without copying audio, AIO or geometry",
    )
    chart_cache.add_argument("--profile-root", required=True)
    chart_cache.add_argument("--suite", required=True)
    chart_cache.add_argument("--output", required=True)
    chart_cache.add_argument("--debug", action="store_true")
    verify_cache = subparsers.add_parser(
        "verify-chart-cache",
        help="compare cached chart tensors with current dynamic materialization",
    )
    verify_cache.add_argument("--profile-root", required=True)
    verify_cache.add_argument("--cache", required=True)
    verify_cache.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-profile":
            result = build_profile(
                dataset=Path(args.dataset).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
            )
        elif args.command == "semantic-audit":
            result = run_semantic_audit(
                dataset=Path(args.dataset).expanduser().resolve(),
                profile_root=Path(args.profile_root).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
            )
        elif args.command == "timing-audit":
            result = run_timing_audit(
                dataset=Path(args.dataset).expanduser().resolve(),
                profile=args.profile,
                output=Path(args.output).expanduser().resolve(),
            )
        elif args.command == "build-interpolation-geometry":
            result = build_geometry_sidecar(
                profile_root=Path(args.profile_root).expanduser().resolve(),
                chart_cache_root=(
                    None
                    if args.chart_cache is None
                    else Path(args.chart_cache).expanduser().resolve()
                ),
                feature_root=(
                    None
                    if args.feature_root is None
                    else Path(args.feature_root).expanduser().resolve()
                ),
                output=Path(args.output).expanduser().resolve(),
            )
        elif args.command == "build-chart-cache":
            result = build_chart_vector_cache(
                profile_root=Path(args.profile_root).expanduser().resolve(),
                suite_path=Path(args.suite).expanduser().resolve(),
                output=Path(args.output).expanduser().resolve(),
            )
        else:
            result = verify_chart_vector_cache(
                profile_root=Path(args.profile_root).expanduser().resolve(),
                cache_root=Path(args.cache).expanduser().resolve(),
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (DatasetError, FileNotFoundError, ValueError) as error:
        if args.debug:
            traceback.print_exc()
        else:
            print(str(error), file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
