"""Repack accepted whole-song GT and BPM metadata into the new full-train input asset."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.song_chart_compact import build, CompactSongChartCache
from maimai_baseline.source_identity import baseline_source_identity
from maimai_dataset.integrity import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--old-song-chart-cache", type=Path)
    parser.add_argument("--bpm-audio-manifest", type=Path)
    parser.add_argument("--profile-segments", type=Path)
    parser.add_argument("--song-audio-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        if any(x is None for x in (args.old_song_chart_cache, args.bpm_audio_manifest,
                                   args.profile_segments, args.song_audio_cache)):
            parser.error("build requires old GT, BPM/profile timing metadata and retained song audio")
        manifest = build(args.old_song_chart_cache, args.output,
                         source_identity={'source': baseline_source_identity(),
                                          'script_sha256': file_sha256(Path(__file__))},
                         bpm_manifest=args.bpm_audio_manifest,
                         profile_segments=args.profile_segments,
                         song_audio_root=args.song_audio_cache)
        result = {"operation": "build", "charts": len(manifest["charts"]),
                  "timing_maps": len(manifest["timing"]["tables"]), "identity": manifest["identity"]}
    else:
        cache = CompactSongChartCache(args.output, verify=True)
        # Validate structural indexes and all field shapes once for this new asset.
        for key in cache.lookup:
            cache.get(key)
        result = {"operation": "verify", "charts": len(cache.lookup),
                  "timing_maps": len(cache.time_tables.arrays), "identity": cache.manifest["identity"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
