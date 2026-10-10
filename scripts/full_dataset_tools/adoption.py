"""Explicit, backed-up adoption of an approved frozen pilot chart.

Copies existing chart vectors; no audio extraction, geometric projection,
spatial transform or bitplane projection is performed here. BPM receipts are
rebound only after identical timing/window/audio references are established.
"""
import json
from pathlib import Path
import shutil
import numpy as np

from maimai_baseline.data.training_dataset import read, rows, sha, identity, inside
from maimai_dataset.window import WINDOW
from .pipeline import dump, meta, profile_records, segment_groups, require_new, SEGMENT_KEYS


def replace_values(value, mapping):
    if isinstance(value, dict):
        return {k: replace_values(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_values(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values), encoding="utf-8")


def adopt(spec, pilot, policy_path):
    root, pilot = spec.runtime, Path(pilot).resolve()
    policy = read(policy_path)
    if policy.get("schema") != "maimai.approved_pilot_adoption.v1":
        raise ValueError("explicit approved pilot adoption policy required")
    selected = policy["records"]
    if (root / "provenance/pilot-adoption.json").exists():
        raise ValueError("pilot adoption already published; do not replace its receipt")
    if any(r not in spec.selected_record_ids for r in selected):
        raise ValueError("adoption escaped full selection")
    for directory in ("spatial-views", "physical/source-rows", "physical/bitplanes", "epoch-plans"):
        if (root / directory).exists():
            raise ValueError("adopt frozen source before building dependent full caches")
    pm = read(pilot / "dataset.json")
    if not pm.get("ready") or pm.get("window") != WINDOW:
        raise ValueError("pilot is not verified for the current window")
    full, old = profile_records(root), profile_records(pilot)
    fg, pg = segment_groups(root), segment_groups(pilot)
    proof = {}
    for rid, hashes in selected.items():
        fp = inside(root / "profile", full[rid]["encoded_path"])
        pp = inside(pilot / "profile", old[rid]["encoded_path"])
        if sha(fp) != hashes["full_encoded_sha256"] or sha(pp) != hashes["pilot_encoded_sha256"]:
            raise ValueError(f"approved adoption source changed: {rid}")
        # The new chart can change rows, but audio windows/timing must not move.
        keys = [k for k in SEGMENT_KEYS if k not in ("context_row_start", "target_row_start", "target_row_end")]
        a = [{k: s.get(k) for k in keys} for s in fg[rid]]
        b = [{k: s.get(k) for k in keys} for s in pg[rid]]
        if a != b or full[rid]["audio"] != old[rid]["audio"]:
            raise ValueError(f"adoption changes audio/window coordinates: {rid}")
        proof[rid] = {**hashes, "audio_window_identity": identity(a), "pilot_record": old[rid],
                      "before_rows": len(read(fp)["rows"]), "after_rows": len(read(pp)["rows"])}
    for name in ("vocabulary.json",):
        a, b = read(root / "profile" / name), read(pilot / "profile" / name)
        if {k: v for k, v in a.items() if k != "base_dataset_fingerprint"} != {k: v for k, v in b.items() if k != "base_dataset_fingerprint"}:
            raise ValueError("adoption vocabulary IDs differ")
    from maimai_dataset.chart_cache import ChartVectorStore
    for base in (root, pilot):
        checked = ChartVectorStore.load(base / "chart-cache",
            profile_fingerprint=read(base / "profile/summary.json")["profile_fingerprint"],
            vocabulary_sha256=sha(base / "profile/vocabulary.json"), verify_payloads=True)
        del checked
    stage = spec.root / "preparation/pilot-adoption-stage"
    backup = spec.root / "provenance/before-pilot-adoption"
    require_new(stage); require_new(backup)
    # Profile copy remains independent; no source files or ready pilot touched.
    shutil.copytree(root / "profile", stage / "profile")
    new_records = []
    for record in rows(root / "profile/manifest.jsonl"):
        rid = record["record_id"]
        if rid in selected:
            source = inside(pilot / "profile", old[rid]["encoded_path"])
            destination = inside(stage / "profile", record["encoded_path"])
            shutil.copyfile(source, destination)
            record = {**old[rid], "encoded_path": record["encoded_path"], "split": "train", "source_split": record.get("source_split", record.get("split")),
                      "pilot_adoption": {"policy_sha256": sha(policy_path), "pilot_encoded_sha256": sha(source)}}
        new_records.append(record)
    new_segments = []
    pilot_segments = {s["segment_id"]: s for group in pg.values() for s in group}
    for segment in rows(root / "profile/segments.jsonl"):
        if segment["record_id"] in selected:
            # Preserve current full audio reference and role; replace only
            # chart-window row metadata from the verified pilot segmentation.
            prior = pilot_segments[segment["segment_id"]]
            for key in ("context_row_start", "target_row_start", "target_row_end", "target_rows", "context_first_tick", "target_first_tick"):
                if key in prior:
                    segment[key] = prior[key]
        new_segments.append(segment)
    jsonl(stage / "profile/manifest.jsonl", new_records)
    jsonl(stage / "profile/segments.jsonl", new_segments)
    summary = read(root / "profile/summary.json")
    summary["training_rows"] = sum(len(read(inside(stage / "profile", r["encoded_path"]))["rows"]) for r in new_records)
    summary["target_row_split_counts"] = {"train": sum(s["target_row_end"] - s["target_row_start"] for s in new_segments)}
    summary["profile_manifest_sha256"] = sha(stage / "profile/manifest.jsonl")
    summary["segments_sha256"] = sha(stage / "profile/segments.jsonl")
    summary["pilot_adoption"] = {"policy_sha256": sha(policy_path), "proof": proof}
    summary["profile_fingerprint"] = identity({"source": read(root / "profile/summary.json")["profile_fingerprint"],
            "adoption": summary["pilot_adoption"], "manifest": sha(stage / "profile/manifest.jsonl"), "segments": sha(stage / "profile/segments.jsonl")})
    dump(stage / "profile/summary.json", summary)
    profile = read(root / "profile/profile.json")
    profile["summary_sha256"] = sha(stage / "profile/summary.json")
    dump(stage / "profile/profile.json", profile)
    # Repack existing raw canonical arrays, choosing pilot only for adopted IDs.
    chart = stage / "chart-cache"; require_new(chart)
    cm, pcm = read(root / "chart-cache/manifest.json"), read(pilot / "chart-cache/manifest.json")
    old_index = rows(root / "chart-cache/segments.jsonl")
    pilot_index = {s["segment_id"]: s for s in rows(pilot / "chart-cache/segments.jsonl")}
    segments_by_id = {s["segment_id"]: s for s in new_segments}
    refs, offsets, new_index = [], [0], []
    for cached in old_index:
        use_pilot = segments_by_id[cached["segment_id"]]["record_id"] in selected
        source = pilot_index[cached["segment_id"]] if use_pilot else cached
        refs.append((use_pilot, source["offset"], source["length"]))
        new_index.append({**cached, "offset": offsets[-1], "length": source["length"]})
        offsets.append(offsets[-1] + source["length"])
    arrays_meta = {}
    for name, old_meta in cm["arrays"].items():
        destination = chart / old_meta["path"]; destination.parent.mkdir(parents=True, exist_ok=True)
        a = np.load(root / "chart-cache" / old_meta["path"], mmap_mode="r", allow_pickle=False)
        b = np.load(pilot / "chart-cache" / pcm["arrays"][name]["path"], mmap_mode="r", allow_pickle=False)
        output = np.lib.format.open_memmap(destination, mode="w+", dtype=a.dtype, shape=(offsets[-1],))
        for i, (use_pilot, begin, length) in enumerate(refs):
            output[offsets[i]:offsets[i + 1]] = (b if use_pilot else a)[begin:begin + length]
        output.flush(); arrays_meta[name] = meta(chart, destination, output)
        output._mmap.close(); a._mmap.close(); b._mmap.close()
    np.save(chart / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    jsonl(chart / "segments.jsonl", new_index)
    shutil.copyfile(root / "chart-cache/audio-references.jsonl", chart / "audio-references.jsonl")
    cm.update(profile_fingerprint=summary["profile_fingerprint"], profile_manifest_sha256=sha(stage / "profile/manifest.jsonl"),
              row_count=offsets[-1], arrays=arrays_meta, offsets=meta(chart, chart / "offsets.npy", np.asarray(offsets, dtype=np.int64)),
              segment_index=meta(chart, chart / "segments.jsonl"), pilot_adoption=summary["pilot_adoption"])
    cm["selection"]["profile_fingerprint"] = summary["profile_fingerprint"]
    dump(chart / "manifest.json", cm)
    checked = ChartVectorStore.load(chart, profile_fingerprint=summary["profile_fingerprint"],
        vocabulary_sha256=sha(stage / "profile/vocabulary.json"), verify_payloads=True)
    del checked
    coverage_path = Path("provenance/canonical-coverage.json")
    coverage = read(root / coverage_path)
    for rid in selected:
        doc = read(inside(stage / "profile", full[rid]["encoded_path"]))
        coverage[rid].update(encoded_sha256=selected[rid]["pilot_encoded_sha256"], rows=len(doc["rows"]),
            notes=sum(r.get("event") == "NOTE" for r in doc["rows"]), target_row_count=len(doc["rows"]),
            pilot_adoption_policy_sha256=sha(policy_path))
    dump(stage / coverage_path, coverage)
    # Preserve every audio/geometry byte. Rebind metadata/receipts with evidence
    # that the exact timing, end tick, audio source and window coordinates agree.
    hashes = {sha(root / "profile/manifest.jsonl"): sha(stage / "profile/manifest.jsonl"),
              sha(root / "profile/segments.jsonl"): sha(stage / "profile/segments.jsonl"),
              sha(root / "chart-cache/manifest.json"): sha(chart / "manifest.json"),
              read(root / "profile/summary.json")["profile_fingerprint"]: summary["profile_fingerprint"]}
    audio = read(root / "audio/bpm-audio-manifest.json")
    audit = read(root / "audio/audit.json")
    new_material = replace_values(audit["inputs"], hashes)
    hashes[audio["preparation_identity"]] = identity(new_material)
    for relative in ("audio/audit.json", "geometry/manifest.json"):
        value = replace_values(read(root / relative), hashes)
        dump(stage / relative, value)
        hashes[sha(root / relative)] = sha(stage / relative)
    dump(stage / "audio/bpm-audio-manifest.json", replace_values(audio, hashes))
    hashes[sha(root / "audio/bpm-audio-manifest.json")] = sha(stage / "audio/bpm-audio-manifest.json")
    receipts = []
    for directory in ("audio", "geometry"):
        for path in (root / directory).rglob("*.complete.json"):
            relative = path.relative_to(root)
            value = replace_values(read(path), hashes)
            dump(stage / relative, value); receipts.append(relative)
    evidence = {"schema": "maimai.pilot_adoption_receipt.v1", "policy_sha256": sha(policy_path),
                "proof": proof, "hash_rebindings": hashes, "audio_geometry_payloads": "unchanged",
                "canonical_vectors": "copied from existing full or verified pilot; no projection"}
    dump(stage / "adoption.json", evidence)
    # Publication is resumable by restoring the named backup; ready is false
    # before the first move, and no recursive deletion is used on any source.
    targets = [Path("profile"), Path("chart-cache"), coverage_path, Path("audio/audit.json"), Path("audio/bpm-audio-manifest.json"), Path("geometry/manifest.json"), *receipts]
    dataset = read(root / "dataset.json"); dataset["ready"] = False; dump(root / "dataset.json", dataset)
    moved = []
    try:
        for relative in targets:
            original, saved = inside(root, relative), inside(backup, relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            original.replace(saved); moved.append(relative)
            (stage / relative).replace(original)
        (stage / "adoption.json").replace(root / "provenance/pilot-adoption.json")
    except BaseException:
        for relative in reversed(moved):
            original, saved = inside(root, relative), inside(backup, relative)
            if original.exists():
                rollback = stage / relative
                rollback.parent.mkdir(parents=True, exist_ok=True)
                original.replace(rollback)
            saved.replace(original)
        raise
    return {"passed": True, "adopted_record_ids": list(selected), "backup": str(backup), "ready": False}
