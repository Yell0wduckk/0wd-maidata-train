"""Train fresh decoder-only MoE A90 and GMM4 MoE AR B50; print commands unless --execute."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.training.query6_queue import main

if __name__ == "__main__":
    main()
