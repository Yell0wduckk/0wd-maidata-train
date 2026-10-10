"""Offline, order-preserving reblocking of existing epoch plans."""
from itertools import groupby
from pathlib import Path
import json
import numpy as np

from maimai_baseline.contract import PLAN_AUDIT_SCHEMA
from maimai_baseline.data.curriculum import EpochPlanStore, sha256_file
from maimai_baseline.training.recipe import RECIPES
from maimai_dataset.manifest import canonical_json_sha256, DatasetError


SCHEMA = "maimai.repacked_epoch_plan.v1"


def flatten(batches):
    return tuple(ref for batch in batches for ref in batch)


def reference_hash(refs):
    return canonical_json_sha256([[r.canonical_index, r.view_id, r.occurrence] for r in refs])


def pack(refs, *, token_budget, lengths, audio_keys):
    """Keep contiguous shared-audio groups intact, including oversized groups."""
    batches, batch, used = [], [], 0
    for _, members in groupby(refs, key=lambda r: audio_keys[r.canonical_index]):
        group = list(members)
        cost = sum(max(1, lengths[r.canonical_index]) for r in group)
        if batch and used + cost > token_budget:
            batches.append(tuple(batch))
            batch, used = [], 0
        batch.extend(group)
        used += cost
    if batch:
        batches.append(tuple(batch))
    return tuple(batches)


def summarize(batches, lengths, notes, audio_keys, budget):
    rows = [sum(max(1, lengths[r.canonical_index]) for r in b) for b in batches]
    note_rows = [sum(notes[(r.canonical_index, r.view_id)] for r in b) for b in batches]
    audio = [len({audio_keys[r.canonical_index] for r in b}) for b in batches]
    return dict(batch_count=len(batches), batch_target_rows=rows, target_rows=sum(rows),
                batch_note_rows=note_rows, note_rows=sum(note_rows),
                batch_audio_windows=audio, audio_window_evaluations=sum(audio),
                unique_audio_windows=len({audio_keys[r.canonical_index] for r in flatten(batches)}),
                oversized_batches=[i for i, count in enumerate(rows) if count > budget],
                reference_sha256=reference_hash(flatten(batches)))


def build_plans(source, output, *, lengths, notes, audio_keys, input_identity, budgets=(3072, 4096)):
    """Build both experiments from one source; all input payloads remain read-only."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    reports = []
    output.mkdir(parents=True)
    for budget in budgets:
        root = output / str(budget)
        (root / source.mode).mkdir(parents=True)
        audits = []
        for epoch in range(source.available_epochs):
            refs = flatten(source.load_epoch(epoch))
            if not refs or any(not 0 <= r.canonical_index < len(lengths) for r in refs):
                raise DatasetError("source plan reference bounds mismatch")
            batches = pack(refs, token_budget=budget, lengths=lengths, audio_keys=audio_keys)
            if flatten(batches) != refs:
                raise AssertionError("repacking changed reference order")
            stats = summarize(batches, lengths, notes, audio_keys, budget)
            relative = f"{source.mode}/epoch-{epoch + 1:03d}.npz"
            offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
            np.savez(root / relative,
                     canonical_indices=np.array([r.canonical_index for r in refs], dtype=np.int64),
                     view_ids=np.array([r.view_id for r in refs], dtype=np.int64),
                     occurrences=np.array([r.occurrence for r in refs], dtype=np.int64), batch_offsets=offsets)
            audit = {**source.audit(epoch), **stats, "schema": PLAN_AUDIT_SCHEMA,
                     "epoch": epoch + 1, "mode": source.mode, "file": relative,
                     "source_reference_sha256": reference_hash(refs),
                     "source_plan_sha256": source.audit(epoch)["plan_sha256"],
                     "file_sha256": sha256_file(root / relative),
                     "plan_sha256": canonical_json_sha256({"references": stats["reference_sha256"], "offsets": offsets.tolist()})}
            audit.pop("fixed_batch_count", None)
            audit["natural_batch_count"] = len(batches)
            audits.append(audit)
        (root / "audit.jsonl").write_text("".join(json.dumps(a, sort_keys=True) + "\n" for a in audits), encoding="utf-8")
        manifest = {k: v for k, v in source.manifest.items() if k not in
                    ("manifest_identity_sha256", "fixed_batch_count", "modes", "audit", "paired_first_ten_epoch_plan_sha256")}
        manifest.update(schema=SCHEMA, token_budget=budget, recipe=RECIPES[str(budget)].identity(),
                        source_plan_identity=source.manifest["manifest_identity_sha256"],
                        repack_inputs=input_identity,
                        modes={source.mode: {"epoch_plan_aggregate_sha256": canonical_json_sha256([a["plan_sha256"] for a in audits])}},
                        audit={"path": "audit.jsonl", "rows": len(audits), "sha256": sha256_file(root / "audit.jsonl")})
        manifest["manifest_identity_sha256"] = canonical_json_sha256(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reports.append(verify_plan(source, EpochPlanStore(root), lengths=lengths, notes=notes, audio_keys=audio_keys))
    (output / "verification.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return reports


def verify_plan(source, target, *, lengths, notes, audio_keys):
    if target.available_epochs != source.available_epochs:
        raise DatasetError("repacked source identity mismatch")
    budget = target.manifest["token_budget"]
    if target.manifest.get("recipe") != RECIPES[str(budget)].identity():
        raise DatasetError("repacked recipe mismatch")
    for epoch in range(source.available_epochs):
        refs = flatten(source.load_epoch(epoch))
        batches = target.load_epoch(epoch)
        expected = pack(refs, token_budget=budget, lengths=lengths, audio_keys=audio_keys)
        if batches != expected:
            raise DatasetError("repacked references or boundaries mismatch")
        stats = summarize(batches, lengths, notes, audio_keys, budget)
        if any(target.audit(epoch).get(k) != v for k, v in stats.items()):
            raise DatasetError("repacked audit statistics mismatch")
    return {"passed": True, "token_budget": budget, "epochs": target.available_epochs,
            "optimizer_steps": sum(target.batch_count(e) for e in range(target.available_epochs)),
            "manifest_identity_sha256": target.manifest["manifest_identity_sha256"]}
