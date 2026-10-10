"""Audit/build/verify the explicit full train-only dataset.

Run from the maidata-baseline worktree with ``PYTHONPATH=src``.  This command
never contacts a remote host and never starts training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maimai_baseline.data.full_dataset import FullDatasetError, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "build", "verify"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training-dataset/full-dataset/config.json"),
        help="full dataset preparation specification",
    )
    parser.add_argument(
        "--no-payload-hash",
        action="store_true",
        help="skip large source payload hashes for a diagnostic audit; not an acceptance audit",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="build profile/chart-cache only; audio remains explicitly not-run",
    )
    args = parser.parse_args()
    try:
        result = run(
            args.config,
            args.stage,
            hash_payloads=not args.no_payload_hash,
            audio=not args.no_audio,
        )
    except FullDatasetError as error:
        parser.error(str(error))
    # Reports contain per-record hashes and the selected audio catalog; keep
    # the CLI output useful for a human while leaving the complete evidence in
    # training-dataset/full-dataset/reports/.
    summary = {
        "stage": args.stage,
        "schema": result.get("schema"),
        "status": result.get("status"),
        "passed": result.get("passed"),
        "ready": result.get("ready"),
        "selection_sha256": result.get("selection_sha256"),
        "failure_count": len(result.get("failures", [])),
        "blocker_count": len(result.get("blockers", [])),
        "reports": {
            "source_audit": "reports/source-audit.json",
            "build": "reports/build.json",
            "verify": "reports/verify.json",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("passed", result.get("status") in {"built", "partial_blocked"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
