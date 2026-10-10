"""Build and read back only the new subset plan; no accepted payload scan."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.data.master_plan import build
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output','experiment-config'): p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
