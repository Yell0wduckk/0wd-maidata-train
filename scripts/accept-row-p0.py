"""One-time acceptance of this run's newly produced P0 files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.style_audio_cache import AudioConditionCache
from maimai_baseline.data.style_cache import StyleLatentCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-cache", type=Path, required=True)
    parser.add_argument("--p0-cache", type=Path, required=True)
    args = parser.parse_args()
    cache = AudioConditionCache(args.p0_cache, StyleLatentCache(args.style_cache),
                                verify_payloads=True)
    print(json.dumps({"accepted_p0_identity": cache.manifest["identity"]}))


if __name__ == "__main__":
    main()
