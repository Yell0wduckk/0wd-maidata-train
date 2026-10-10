from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from maimai_dataset.chart_cache import ChartVectorStore
from maimai_dataset.dataset import CompactSegmentDataset
from maimai_dataset.geometry import build_segment_geometry
from maimai_dataset.vocabulary import CompactVocabulary


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--chart-cache", required=True)
    parser.add_argument("--geometry", required=True)
    args = parser.parse_args()
    profile = Path(args.profile).resolve()
    cache = Path(args.chart_cache).resolve()
    geometry = Path(args.geometry).resolve()
    summary = json.loads((profile / "summary.json").read_text("utf-8"))
    vocabulary_path = profile / "vocabulary.json"
    vocabulary = CompactVocabulary.from_dict(
        json.loads(vocabulary_path.read_text("utf-8"))
    )
    store = ChartVectorStore.load(
        cache,
        profile_fingerprint=summary["profile_fingerprint"],
        vocabulary_sha256=hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
    )
    dataset_kwargs = {
        "split": None,
        "song_ids": store.song_ids("validation"),
        "chart_cache_store": store,
        "geometry_root": geometry,
        "geometry_suite_sha256": store.manifest["selection"]["selection_sha256"],
    }
    try:
        dataset = CompactSegmentDataset(
            profile, vocabulary, defer_audio=True, **dataset_kwargs
        )
    except TypeError as error:
        if "defer_audio" not in str(error):
            raise
        dataset = CompactSegmentDataset(profile, vocabulary, **dataset_kwargs)
    mismatches: list[dict[str, object]] = []
    checked = 0
    if hasattr(dataset, "audio_group_keys"):
        items = (dataset.audio_item(key) for key in dataset.audio_group_keys)
        expected_items = len(dataset.audio_group_keys)
    else:
        items = (dataset[index] for index in range(len(dataset)))
        expected_items = len(dataset)
    for item_index, sidecar_item in enumerate(items):
        start_tick = int(
            round(
                float(
                    sidecar_item.get(
                        "audio_start_tick", sidecar_item.get("start_tick", 0.0)
                    )
                )
            )
        )
        dynamic = build_segment_geometry(
            frame_beats=sidecar_item["frame_beats"].numpy(),
            frame_valid=sidecar_item["audio_valid"].numpy(),
            target_start_tick=start_tick,
            target_end_tick=start_tick + 8 * 96,
        )
        sidecar = sidecar_item["interpolation_geometry"]
        for name, dynamic_value in dynamic.items():
            sidecar_value = sidecar[name].numpy()
            if not np.array_equal(dynamic_value, sidecar_value):
                mismatches.append(
                    {
                        "item_index": item_index,
                        "field": name,
                        "dynamic_sha256": _digest(dynamic_value),
                        "sidecar_sha256": _digest(sidecar_value),
                        "different_values": int(
                            np.count_nonzero(dynamic_value != sidecar_value)
                        ),
                    }
                )
                break
        checked += 1
    print(
        json.dumps(
            {
                "segments": len(dataset),
                "items": expected_items,
                "checked_groups": checked,
                "mismatch_count": len(mismatches),
                "first_mismatches": mismatches[:8],
            },
            sort_keys=True,
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
