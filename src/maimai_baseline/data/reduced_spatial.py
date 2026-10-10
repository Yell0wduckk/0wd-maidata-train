"""Derive paired spatial plans by replacing view IDs without reordering batches."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np

from maimai_baseline.contract import (
    DIFFICULTY_POOLS, SEED, VIEW_NAMES, spatial_sampling_identity,
)
from maimai_baseline.data.curriculum import (
    ChartSlot, EpochPlanStore, VirtualSegmentRef, assign_spatial_views,
    sha256_file, validate_spatial_sampling,
)
from maimai_baseline.data.repack import flatten, reference_hash, summarize
from maimai_baseline.training.recipe import RECIPES
from maimai_dataset.manifest import canonical_json_sha256 as identity, DatasetError


def reassigned(batches, segments, epoch):
    """One allocation per unique (song, difficulty, occurrence), shared by windows."""
    def slot(ref):
        if not 0 <= ref.canonical_index < len(segments):
            raise DatasetError("parent plan segment index out of range")
        s = segments[ref.canonical_index]
        return ChartSlot(str(s["song_id"]), int(s["source_difficulty_id"]), ref.occurrence)
    slots = {slot(ref) for batch in batches for ref in batch}
    assigned = assign_spatial_views(sorted(slots, key=lambda s: (s.song_id, s.difficulty_id, s.occurrence)),
                                    seed=SEED, epoch=epoch)
    result = tuple(tuple(VirtualSegmentRef(r.canonical_index, VIEW_NAMES.index(assigned[slot(r)]),
                                           r.occurrence) for r in b) for b in batches)
    counts = defaultdict(Counter)
    for item, view in assigned.items():
        counts[DIFFICULTY_POOLS[item.difficulty_id]][view] += 1
    return result, {pool: dict(count) for pool, count in counts.items()}


def validate_parent(parent, source, inputs):
    m = parent.manifest
    if (m.get("schema") != "maimai.repacked_epoch_plan.v1"
            or m.get("token_budget") != 4096 or m.get("recipe") != RECIPES["4096"].identity()
            or m.get("seed") != SEED or parent.available_epochs != 75
            or m.get("repack_inputs") != inputs
            or m.get("segment_order_sha256") != source.manifest["segment_order_sha256"]
            or m.get("segment_count") != len(source.segments)):
        raise DatasetError("parent 4096 plan/input contract mismatch")
    if "spatial_sampling" in m:
        raise DatasetError("parent must be the original control plan")


def verify_plans(parent, target, *, segments, lengths, notes, audio_keys):
    validate_spatial_sampling(target.manifest)
    if (target.available_epochs != parent.available_epochs
            or target.manifest.get("repack_inputs") != parent.manifest.get("repack_inputs")
            or target.manifest.get("recipe") != RECIPES["4096"].identity()
            or target.manifest.get("token_budget") != 4096):
        raise DatasetError("derived plan parent/input identity mismatch")
    reports = []
    for index in range(parent.available_epochs):
        original = parent.load_epoch(index)
        actual = target.load_epoch(index)
        expected, counts = reassigned(original, segments, index + 1)
        if actual != expected:
            raise DatasetError(f"derived spatial references/boundaries mismatch at epoch {index + 1}")
        stats = summarize(actual, lengths, notes, audio_keys, 4096)
        audit = target.audit(index)
        if any(audit.get(k) != v for k, v in stats.items()) or audit["view_counts_by_pool"] != counts:
            raise DatasetError("derived spatial audit mismatch")
        # Only view IDs may change. Compare all view-independent training quantities.
        old = summarize(original, lengths, notes, audio_keys, 4096)
        if any(stats[k] != v for k, v in old.items() if k != "reference_sha256"):
            raise DatasetError("spatial derivation changed training progress or audio grouping")
        if audit.get("parent_plan_sha256") != parent.audit(index)["plan_sha256"]:
            raise DatasetError("derived epoch parent identity mismatch")
        reports.append({"epoch": index + 1, "batch_count": len(actual),
                        "target_rows": stats["target_rows"], "view_counts_by_pool": counts})
    return {"passed": True, "epochs": len(reports),
            "optimizer_steps": sum(row["batch_count"] for row in reports),
            "paired_references_and_boundaries": True,
            "parent_plan_identity": parent.manifest["manifest_identity_sha256"],
            "manifest_identity_sha256": target.manifest["manifest_identity_sha256"],
            "spatial_sampling": spatial_sampling_identity(), "epoch_audits": reports}


def build_plans(parent, output, *, segments, lengths, notes, audio_keys):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / parent.mode).mkdir()
    audits = []
    for index in range(parent.available_epochs):
        batches, counts = reassigned(parent.load_epoch(index), segments, index + 1)
        refs = flatten(batches)
        offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
        relative = f"{parent.mode}/epoch-{index + 1:03d}.npz"
        np.savez(output / relative,
                 canonical_indices=np.array([r.canonical_index for r in refs], dtype=np.int64),
                 view_ids=np.array([r.view_id for r in refs], dtype=np.int64),
                 occurrences=np.array([r.occurrence for r in refs], dtype=np.int64),
                 batch_offsets=offsets)
        stats = summarize(batches, lengths, notes, audio_keys, 4096)
        audit = {**parent.audit(index), **stats, "view_counts_by_pool": counts,
                 "parent_plan_sha256": parent.audit(index)["plan_sha256"],
                 "source_reference_sha256": reference_hash(refs),
                 "source_plan_sha256": reference_hash(refs),
                 "file_sha256": sha256_file(output / relative),
                 "plan_sha256": identity({"references": stats["reference_sha256"], "offsets": offsets.tolist()})}
        audits.append(audit)
    (output / "audit.jsonl").write_text("".join(json.dumps(a, sort_keys=True) + "\n" for a in audits), encoding="utf-8")
    manifest = dict(parent.manifest)
    manifest.pop("manifest_identity_sha256")
    manifest.update(spatial_sampling=spatial_sampling_identity(),
                    parent_plan_identity=parent.manifest["manifest_identity_sha256"],
                    source_plan_identity=identity({"parent": parent.manifest["manifest_identity_sha256"],
                                                   "spatial_sampling": spatial_sampling_identity()}),
                    derivation="replace-view-ids-preserve-batches-v1",
                    modes={parent.mode: {"epoch_plan_aggregate_sha256": identity([a["plan_sha256"] for a in audits])}},
                    audit={"path": "audit.jsonl", "rows": len(audits), "sha256": sha256_file(output / "audit.jsonl")})
    manifest["manifest_identity_sha256"] = identity(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = verify_plans(parent, EpochPlanStore(output), segments=segments,
                          lengths=lengths, notes=notes, audio_keys=audio_keys)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
