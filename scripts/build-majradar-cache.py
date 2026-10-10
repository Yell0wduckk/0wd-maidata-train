"""Build explicitly selected MajRadar targets; never invoked by training startup."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.majradar import build_cache, repair_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("profile", "subset-config", "output", "miacode-exe"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("majradar-project", "host-project", "host-dll", "build-root"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--repair-record-ids", nargs="+", help="Explicit records in an incomplete producer output; keep initial evidence")
    args = vars(parser.parse_args())
    record_ids = args.pop("repair_record_ids")
    return repair_cache(**args, record_ids=record_ids) if record_ids else build_cache(**args)


if __name__ == "__main__":
    main()
