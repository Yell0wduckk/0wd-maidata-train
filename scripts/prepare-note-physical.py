"""Build a fresh short-note Physical cache after the explicit audio/GT audit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.note_view import verify, load_metadata
from full_dataset_tools.bitplane import build_physical_bitplane_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("note-view-manifest", "profile", "chart-cache", "spatial-cache", "path-catalog", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    metadata = load_metadata(args.note_view_manifest, profile=args.profile)
    verify(args.note_view_manifest, profile=args.profile,
           bpm_manifest=args.profile.parent / "audio/bpm-audio-manifest.json")
    from maimai_dataset.manifest import load_jsonl
    def record_id(row):
        # The pilot cache predates the record_id field and encodes it as the
        # song:difficulty prefix of segment_id. Keep the old cache read-only.
        if "record_id" in row:
            return str(row["record_id"])
        segment_id = str(row["segment_id"])
        if ":" not in segment_id:
            raise ValueError("chart-cache segment_id lacks record identity")
        return segment_id.rsplit(":", 1)[0]

    selected = {record_id(row) for row in load_jsonl(args.chart_cache / "segments.jsonl")
                if row["role"] in ("train", "validation")}
    if not selected.issubset(metadata["charts"]):
        raise ValueError("note view does not cover the selected Physical charts")
    result = build_physical_bitplane_cache(profile_root=args.profile,
        chart_cache_root=args.chart_cache, spatial_cache_root=args.spatial_cache,
        catalog_root=args.path_catalog, output=args.output, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
