"""Train only the frozen-A90 AR B50 branch; inspect without --execute."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.training.ar_runner import main

if __name__ == '__main__':
    main()
