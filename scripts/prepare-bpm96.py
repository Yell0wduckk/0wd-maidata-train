"""Run from this worktree with PYTHONPATH=src. No implicit data selection."""
from maimai_baseline.data.prepare import main

if __name__ == "__main__":
    raise SystemExit(main())
