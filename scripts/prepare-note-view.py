"""Prepare canonical short-note views without audio rebuild or training."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.note_view import audit, build, verify, write

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "build", "verify"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--bpm-audio-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-record-id", action="append", default=[])
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.output / "manifest.json", profile=args.profile,
                        bpm_manifest=args.bpm_audio_manifest)
    else:
        if args.profile is None or args.bpm_audio_manifest is None:
            parser.error("--profile and --bpm-audio-manifest are required")
        if args.command == "audit":
            result = audit(args.profile, args.bpm_audio_manifest,
                           excluded_record_ids=args.exclude_record_id)
            write(args.output, result)
        else:
            result = build(args.profile, args.bpm_audio_manifest, args.output,
                           excluded_record_ids=args.exclude_record_id)
    print(json.dumps({k: v for k, v in result.items() if k != "charts"}, ensure_ascii=False))
    return 1 if result.get("errors") else 0

if __name__ == "__main__":
    raise SystemExit(main())
