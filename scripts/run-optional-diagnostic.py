"""Run one intermediate diagnostic without making its failure block training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a diagnostic command is required after --")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        exit_code = subprocess.call(command)
        error = None
    except BaseException as exc:  # Launch failures are also non-blocking for this stage.
        exit_code = None
        error = f"{type(exc).__name__}: {exc}"
    report = {
        "schema": "maimai.optional_intermediate_diagnostic.v1",
        "passed": exit_code == 0 and error is None,
        "continued": True,
        "started_at": started_at,
        "seconds": time.perf_counter() - started,
        "exit_code": exit_code,
        "error": error,
        "argv": command,
    }
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
