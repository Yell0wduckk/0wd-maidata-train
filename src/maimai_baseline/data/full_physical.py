"""Verify a completed full chart-side dataset through current runtime readers."""
from pathlib import Path
import numpy as np
from maimai_baseline.data.training_dataset import read, rows, sha, inside
from maimai_baseline.data.curriculum import MultiViewRouteStore
from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_dataset.window import WINDOW
from maimai_dataset.timing import TimingMap


def verify_full_physical(root: Path, *, full_hash: bool = True):
    root = Path(root)
    summary = read(root / "profile/summary.json")
    vocab_sha = sha(root / "profile/vocabulary.json")
    chart_sha = sha(root / "chart-cache/manifest.json")
    spatial_sha = sha(root / "spatial-views/manifest.json")
    spatial = MultiViewRouteStore.load(root / "spatial-views", profile_fingerprint=summary["profile_fingerprint"],
                vocabulary_sha256=vocab_sha, selection_sha256=read(root / "chart-cache/manifest.json")["selection"]["selection_sha256"], verify_payloads=full_hash)
    if spatial.manifest.get("canonical_cache_manifest_sha256") != chart_sha or spatial.manifest.get("profile_manifest_sha256") != sha(root / "profile/manifest.jsonl"):
        raise ValueError("full spatial canonical/profile binding mismatch")
    maps = PhysicalSourceRowStore(root / "physical/source-rows", profile_fingerprint=summary["profile_fingerprint"],
                vocabulary_sha256=vocab_sha, chart_cache_manifest_sha256=chart_sha,
                spatial_cache_manifest_sha256=spatial_sha, verify_payloads=full_hash)
    vocab = CompactVocabulary.from_dict(read(root / "profile/vocabulary.json"))
    catalog = CompactPathCatalog.load(root / "physical/path-catalog", vocab)
    planes = PhysicalBitplaneStore(root / "physical/bitplanes", profile_fingerprint=summary["profile_fingerprint"],
                chart_cache_manifest_sha256=chart_sha, spatial_cache_manifest_sha256=spatial_sha, catalog_identity=catalog.identity)
    planes.validate(full_hash=full_hash, scan_contents=full_hash)
    for manifest in (spatial.manifest, planes.manifest, read(root / "physical/source-rows/manifest.json")):
        if manifest.get("window") != WINDOW:
            raise ValueError("full Physical/spatial window mismatch")
    records = rows(root / "profile/manifest.jsonl")
    segments = rows(root / "profile/segments.jsonl")
    expected = {r["record_id"] for r in records}
    selected = read(root / "selection.json")
    if expected != set(selected["record_ids"]) or selected["validation_song_ids"]:
        raise ValueError("full Physical selection mismatch")
    if expected != set(planes.record_indices) or expected != set(maps.indices["identity"]):
        raise ValueError("full Physical record coverage mismatch")
    segment_ids = {s["segment_id"] for s in segments}
    if segment_ids != set(spatial.segment_indices) or segment_ids != set(planes.segment_records):
        raise ValueError("full Physical segment coverage mismatch")
    if any(r.get("role") != "train" for r in planes.chart_records):
        raise ValueError("Physical index contains non-training records")
    first = {s["record_id"]: s for s in segments}
    row_counts = {}
    for r in records:
        rid = r["record_id"]; index = planes.record_indices[rid]
        path = inside(root / "profile", r["encoded_path"])
        if sha(path) != planes.chart_records[index]["encoded_sha256"]:
            raise ValueError(f"Physical encoded hash mismatch: {rid}")
        row_counts[rid] = len(read(path)["rows"])
        if int(planes.offsets["source_offsets"][index + 1] - planes.offsets["source_offsets"][index]) != row_counts[rid]:
            raise ValueError(f"Physical row count mismatch: {rid}")
        timing = TimingMap.from_dict(first[rid]["timing"])
        a, b = map(int, planes.offsets["timing_offsets"][index:index + 2])
        for key, values in {"timing_ticks": [float(p.beat) * 96 for p in timing.points],
                            "timing_seconds": [p.second for p in timing.points], "timing_bpms": [p.bpm for p in timing.points]}.items():
            if not np.array_equal(planes.arrays[key][a:b], np.asarray(values, dtype=np.float64)):
                raise ValueError(f"Physical timing mismatch: {rid}/{key}")
    maps.validate_records(row_counts)
    return {"passed": True, "status": "passed", "chart_count": len(expected), "segment_count": len(segment_ids),
            "full_hash": full_hash, "spatial_manifest_sha256": spatial_sha,
            "bitplane_manifest_sha256": sha(root / "physical/bitplanes/manifest.json"),
            "source_rows_manifest_sha256": sha(root / "physical/source-rows/manifest.json")}
