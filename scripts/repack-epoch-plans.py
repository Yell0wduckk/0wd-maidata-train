"""Audit, build, or verify paired batch plans without rebuilding input caches."""
import argparse
from pathlib import Path
import json
import numpy as np

from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, sha256_file
from maimai_baseline.data.dataset import BaselineDataset
from maimai_baseline.data.repack import build_plans, flatten, verify_plan
from maimai_baseline.contract import VIEW_NAMES
from maimai_dataset.chart_cache import ChartVectorStore
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_dataset.manifest import canonical_json_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "build", "verify"))
    for name in ("dataset", "chart-cache", "spatial-view-cache", "bpm-audio-manifest", "source-plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="paired output root (3072/ and 4096/)")
    args = parser.parse_args()
    summary = read(args.dataset / "summary.json")
    vocab = CompactVocabulary.from_dict(read(args.dataset / "vocabulary.json"))
    kwargs = dict(profile_fingerprint=summary["profile_fingerprint"],
                  vocabulary_sha256=sha256_file(args.dataset / "vocabulary.json"), verify_payloads=False)
    chart = ChartVectorStore.load(args.chart_cache, **kwargs)
    spatial = MultiViewRouteStore.load(args.spatial_view_cache, **kwargs,
                                      selection_sha256=chart.manifest["selection"]["selection_sha256"])
    dataset = BaselineDataset(args.dataset, vocab, split=None, song_ids=chart.song_ids("train"),
                              chart_cache_store=chart, spatial_store=spatial, defer_audio=True)
    source = EpochPlanStore(args.source_plan)
    if source.manifest["segment_count"] != len(dataset):
        raise ValueError("source plan segment count mismatch")
    order_hash = canonical_json_sha256([str(s["segment_id"]) for s in dataset.segments])
    if source.manifest.get("segment_order_sha256") != order_hash:
        raise ValueError("source plan segment order mismatch")
    bpm = read(args.bpm_audio_manifest)
    # Same zero-gain window grouping as BaselineDataset.configure_bpm_audio.
    keys = [canonical_json_sha256({"variant": bpm["segments"][s["segment_id"]],
                                  "start_tick": int(s["target_start_tick"]), "gain_db": 0.0})
            for s in dataset.segments]
    notes = {}
    for epoch in range(source.available_epochs):
        for ref in flatten(source.load_epoch(epoch)):
            key = (ref.canonical_index, ref.view_id)
            if key in notes:
                continue
            segment = dataset.segments[ref.canonical_index]
            index = spatial.segment_indices[segment["segment_id"]]
            begin, end = map(int, spatial.offsets[index:index + 2])
            events = spatial.arrays[VIEW_NAMES[ref.view_id]]["target_ids.event"][begin:end]
            rows = int(np.count_nonzero(events != -100))
            if rows != dataset.target_lengths[ref.canonical_index]:
                raise ValueError("profile/spatial target row count mismatch")
            notes[key] = int(np.count_nonzero(events == vocab.event_id("NOTE")))
    identities = {name: sha256_file(path) for name, path in (
        ("profile_segments", args.dataset / "segments.jsonl"),
        ("chart_manifest", args.chart_cache / "manifest.json"),
        ("spatial_manifest", args.spatial_view_cache / "manifest.json"),
        ("bpm_manifest", args.bpm_audio_manifest))}
    common = dict(lengths=dataset.target_lengths, notes=notes, audio_keys=keys)
    if args.action == "audit":
        report = {"passed": True, "epochs": source.available_epochs, "segments": len(dataset),
                  "source_plan_identity": source.manifest["manifest_identity_sha256"], "inputs": identities}
    elif args.action == "build":
        report = build_plans(source, args.output, input_identity=identities, **common)
    else:
        report = []
        for budget in (3072, 4096):
            target = EpochPlanStore(args.output / str(budget))
            if target.manifest["repack_inputs"] != identities:
                raise ValueError("repack input identity mismatch")
            report.append(verify_plan(source, target, **common))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
