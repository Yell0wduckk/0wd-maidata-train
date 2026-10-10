"""Incremental chart-side preparation with verified pilot exclusion and merge.

No historical packages, remote actions, training or source-manifest rewrites.
The canonical v2 plane producer is in bitplane.py. New spatial rows keep the
canonical NOTE order; coordinates follow MiaCode's four involutions, and row
maps make this explicit. Reused pilot vectors keep their exact original order.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import zlib

import numpy as np
from maimai_baseline.contract import VIEW_NAMES, VIEW_CACHE_SCHEMA
from maimai_baseline.data.full_dataset import FullDatasetSpec, contained
from maimai_baseline.data.training_dataset import read, rows, sha, identity
from maimai_dataset.window import WINDOW
from maimai_dataset.encoding import restore_path, factorize_path
from maimai_baseline.physical.spatial import LANE_PERMUTATIONS, TOUCH_DE_PERMUTATIONS


SCHEMA = "maimai.full_incremental_physical_plan.v1"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def meta(root, path, array=None):
    value = {"path": path.relative_to(root).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
    if array is not None:
        value.update(dtype=array.dtype.name, shape=list(array.shape))
    return value


def profile_records(root):
    values = rows(root / "profile/manifest.jsonl")
    result = {r["record_id"]: r for r in values}
    if len(result) != len(values):
        raise ValueError("duplicate profile record IDs")
    return result


def segment_groups(root):
    result = defaultdict(list)
    for row in rows(root / "profile/segments.jsonl"):
        result[row["record_id"]].append(row)
    return result


def semantic_document(document):
    # Source-file labels can differ after the IR rebuild. Keep full rows and
    # moments in the proof; never pretend the encoded file hashes are equal.
    return {k: document[k] for k in ("record_id", "rows", "moment_chunks")}


SEGMENT_KEYS = ("segment_id", "target_start_tick", "target_end_tick", "context_start_tick",
                "context_row_start", "target_row_start", "target_row_end", "song_end_tick", "timing")


def reusable_proof(full, pilot, rid, fg, pg):
    fd, pd = read(full), read(pilot)
    if semantic_document(fd) != semantic_document(pd):
        raise ValueError(f"pilot semantic conflict (not silently rebuilt): {rid}")
    a = [{k: s.get(k) for k in SEGMENT_KEYS} for s in fg]
    b = [{k: s.get(k) for k in SEGMENT_KEYS} for s in pg]
    if a != b:
        raise ValueError(f"pilot window/timing conflict (not silently rebuilt): {rid}")
    return {"full_encoded_sha256": sha(full), "pilot_encoded_sha256": sha(pilot),
            "semantic_sha256": identity(semantic_document(fd)), "segments_sha256": identity(a),
            "row_count": len(fd["rows"])}


def partition(full_ids, pilot_ids):
    full, pilot = set(full_ids), set(pilot_ids)
    return sorted(full & pilot), sorted(full - pilot), sorted(pilot - full)


def tool_identity():
    files = sorted(Path(__file__).parent.glob("*.py"))
    from maimai_baseline.source_identity import baseline_source_identity
    return {**{p.name: sha(p) for p in files},
            "entrypoint": sha(Path(__file__).parents[1] / "prepare-full-physical.py"),
            "baseline_source": baseline_source_identity()["source_sha256"]}


def audit(spec, pilot, *, verify_payloads=True):
    from maimai_baseline.data.curriculum import MultiViewRouteStore
    from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
    from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
    from maimai_baseline.physical.catalog import CompactPathCatalog
    from maimai_dataset.vocabulary import CompactVocabulary
    pilot = Path(pilot).resolve()
    full = spec.runtime
    if pilot == full or pilot.is_relative_to(full):
        raise ValueError("pilot and full runtime must be independent roots")
    pm = read(pilot / "dataset.json")
    if not pm.get("ready") or pm["window"] != WINDOW:
        raise ValueError("pilot is not ready for this window")
    full_records, pilot_records = profile_records(full), profile_records(pilot)
    if set(full_records) != set(spec.selected_record_ids):
        raise ValueError("runtime profile does not match frozen full record_ids")
    reuse, build, excluded = partition(full_records, pilot_records)
    selection_path = spec.root / "incremental-selection.json"
    dump(selection_path, {"schema": "maimai.full_incremental_selection.v1", "selection_sha256": spec.selection_sha256,
        "reuse_record_ids": reuse, "build_record_ids": build, "excluded_pilot_record_ids": excluded,
        "policy": "build is full minus the entire existing pilot intersection; conflicts block and never enter build",
        "counts": {"reuse": len(reuse), "build": len(build), "excluded_pilot": len(excluded), "full": len(full_records)}})
    fg, pg = segment_groups(full), segment_groups(pilot)
    proof, conflicts = {}, []
    for rid in reuse:
        try:
            proof[rid] = reusable_proof(contained(full / "profile", full_records[rid]["encoded_path"]),
                                        contained(pilot / "profile", pilot_records[rid]["encoded_path"]), rid, fg[rid], pg[rid])
        except ValueError as error:
            conflicts.append({"record_id": rid, "reason": str(error)})
    reuse = sorted(proof)
    full_vocab = read(full / "profile/vocabulary.json")
    pilot_vocab = read(pilot / "profile/vocabulary.json")
    if {k: v for k, v in full_vocab.items() if k != "base_dataset_fingerprint"} != {k: v for k, v in pilot_vocab.items() if k != "base_dataset_fingerprint"}:
        raise ValueError("full/pilot vocabulary mismatch")
    spatial = read(pilot / "spatial-views/manifest.json")
    summary = read(pilot / "profile/summary.json")
    checks = dict(profile_fingerprint=summary["profile_fingerprint"],
                  vocabulary_sha256=sha(pilot / "profile/vocabulary.json"),
                  canonical_cache_manifest_sha256=sha(pilot / "chart-cache/manifest.json"))
    MultiViewRouteStore.load(pilot / "spatial-views", profile_fingerprint=checks["profile_fingerprint"], vocabulary_sha256=checks["vocabulary_sha256"], selection_sha256=spatial["selection_sha256"], verify_payloads=verify_payloads)
    source = PhysicalSourceRowStore(pilot / "physical/source-rows", profile_fingerprint=summary["profile_fingerprint"],
                                   vocabulary_sha256=checks["vocabulary_sha256"],
                                   chart_cache_manifest_sha256=checks["canonical_cache_manifest_sha256"],
                                   spatial_cache_manifest_sha256=sha(pilot / "spatial-views/manifest.json"), verify_payloads=verify_payloads)
    source.validate_records({rid: p["row_count"] for rid, p in proof.items()})
    planes = PhysicalBitplaneStore(pilot / "physical/bitplanes", profile_fingerprint=summary["profile_fingerprint"],
                                  chart_cache_manifest_sha256=checks["canonical_cache_manifest_sha256"],
                                  spatial_cache_manifest_sha256=sha(pilot / "spatial-views/manifest.json"),
                                  catalog_identity=CompactPathCatalog.load(pilot / "physical/path-catalog", CompactVocabulary.from_dict(pilot_vocab)).identity)
    if verify_payloads:
        planes.validate(full_hash=True, scan_contents=True)
        from maimai_baseline.model.route import load_route_artifacts
        load_route_artifacts(pilot / "route-artifact", verify_payloads=True)
    for rid, p in proof.items():
        if planes.chart_records[planes.record_indices[rid]]["encoded_sha256"] != p["pilot_encoded_sha256"]:
            raise ValueError(f"pilot plane/encoded source mismatch: {rid}")
    # Freeze full source bytes too: later build must not consume changed rows.
    encoded_hashes = {rid: sha(contained(full / "profile", row["encoded_path"])) for rid, row in full_records.items()}
    inputs = {str(p): sha(p) for p in [spec.config_path, spec.root / "selection.json", selection_path,
              full / "profile/manifest.jsonl", full / "profile/segments.jsonl", full / "profile/summary.json",
              full / "chart-cache/manifest.json", full / "profile/vocabulary.json", pilot / "profile/vocabulary.json", pilot / "dataset.json", pilot / "profile/manifest.jsonl",
              pilot / "profile/segments.jsonl", pilot / "spatial-views/manifest.json",
              pilot / "chart-cache/manifest.json",
              pilot / "physical/bitplanes/manifest.json", pilot / "physical/source-rows/manifest.json"]}
    path_files = [pilot / "physical/path-catalog" / n for n in ("paths.json", "runtime-path-semantics.json", "runtime-path-rules.json")]
    inputs.update({str(p): sha(p) for p in path_files})
    inputs.update({str(pilot / "route-artifact" / n): sha(pilot / "route-artifact" / n)
                   for n in ("manifest.json", "dual_route_embeddings_256d.npz")})
    result = {"schema": SCHEMA, "passed": verify_payloads and not conflicts, "conflicts": conflicts, "pilot_root": str(pilot), "runtime_root": str(full),
              "selection_sha256": spec.selection_sha256, "window": WINDOW, "tools": tool_identity(), "inputs": inputs,
              "reuse_record_ids": reuse, "build_record_ids": build, "excluded_pilot_record_ids": excluded,
              "reuse_proof": proof, "full_encoded_sha256": encoded_hashes,
              "counts": {"reuse": len(reuse), "build": len(build), "conflicts": len(conflicts), "excluded_pilot": len(excluded), "full": len(full_records)},
              "pilot_chain": {"spatial": spatial.get("reuse_provenance"), "bitplane": planes.manifest.get("reuse_provenance"),
                              "source_rows": read(pilot / "physical/source-rows/manifest.json").get("reuse_provenance")}}
    result["identity"] = identity(result)
    dump(spec.reports / "incremental-physical-plan.json", result)
    return result


def load_plan(spec):
    plan = read(spec.reports / "incremental-physical-plan.json")
    if plan.get("schema") != SCHEMA or not plan.get("passed") or plan["selection_sha256"] != spec.selection_sha256:
        raise ValueError("a successful incremental audit is required")
    if Path(plan["runtime_root"]).resolve() != spec.runtime.resolve():
        raise ValueError("incremental plan output root mismatch")
    if identity({k: v for k, v in plan.items() if k != "identity"}) != plan["identity"]:
        raise ValueError("incremental plan identity mismatch")
    if tool_identity() != plan["tools"]:
        raise ValueError("offline producer changed after audit; rerun audit")
    for name, digest in plan["inputs"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"input changed after audit: {name}")
    for rid, row in profile_records(spec.runtime).items():
        if sha(contained(spec.runtime / "profile", row["encoded_path"])) != plan["full_encoded_sha256"][rid]:
            raise ValueError(f"full encoded changed after audit: {rid}")
    pilot_records = profile_records(Path(plan["pilot_root"]))
    for rid, proof in plan["reuse_proof"].items():
        if sha(contained(Path(plan["pilot_root"]) / "profile", pilot_records[rid]["encoded_path"])) != proof["pilot_encoded_sha256"]:
            raise ValueError(f"pilot encoded changed after audit: {rid}")
    if set(plan["reuse_record_ids"]) & set(plan["build_record_ids"]) or set(plan["reuse_record_ids"] + plan["build_record_ids"]) != set(spec.selected_record_ids):
        raise ValueError("incremental coverage is not an exact disjoint partition")
    return plan


def transform_pad(pad, view):
    if pad == "C":
        return pad
    if not re.fullmatch(r"[ABDEK][1-8]", pad):
        raise ValueError(f"invalid spatial pad: {pad}")
    permutation = TOUCH_DE_PERMUTATIONS if pad[0] in "DE" else LANE_PERMUTATIONS
    return pad[0] + str(permutation[view][int(pad[1]) - 1] + 1)


def transform_path(path, view):
    # Literal rules from ChartBatchTransform.Transform.cpp, with absolute arc
    # orientation dependent on the starting lane (rotation also changes < >).
    match = re.fullmatch(r"([1-8])(pp|qq|[-^<>vpqswzV])([1-8]{1,2})", path)
    if not match:
        raise ValueError(f"unsupported single route: {path}")
    start, op, tail = int(match[1]), match[2], match[3]
    mapped = LANE_PERMUTATIONS[view][start - 1] + 1
    if op in "<>":
        clockwise = (op == ">") == (start in (1, 2, 7, 8))
        if view in (1, 2):
            clockwise = not clockwise
        op = ">" if clockwise == (mapped in (1, 2, 7, 8)) else "<"
    elif view in (1, 2):
        op = {"p": "q", "q": "p", "pp": "qq", "qq": "pp", "s": "z", "z": "s"}.get(op, op)
    return str(mapped) + op + "".join(str(LANE_PERMUTATIONS[view][int(c) - 1] + 1) for c in tail)


def transform_rows(canonical, view):
    result, start = [], None
    for original in canonical:
        row = dict(original)
        if row["event"] == "NOTE":
            start = row.get("start_pad")
        if "shape" in row:
            path = restore_path(start_pad=start, shape=row["shape"], end_offset=row["end_offset"], via_offset=row.get("via_offset"))
            factored = factorize_path(transform_path(path, view))
            row.pop("via_offset", None)
            row.update(factored)
        for key in ("pad", "start_pad"):
            if key in row:
                row[key] = transform_pad(row[key], view)
        result.append(row)
    return result


def pack_mapping(pack, values, rid):
    raw = np.asarray(values, dtype="<i4").tobytes()
    compressed = zlib.compress(raw, 6)
    offset = pack.tell()
    pack.write(compressed)
    return dict(record_id=rid, offset=offset, length=len(compressed), uncompressed_length=len(raw),
                row_count=len(values), record_sha256=hashlib.sha256(raw).hexdigest())


def require_new(path):
    if path.exists():
        raise ValueError(f"output exists; keep it intact and select a fresh output after review: {path}")
    path.mkdir(parents=True)


def spatial_build(spec, plan):
    from maimai_dataset.vocabulary import CompactVocabulary
    from maimai_dataset.chart_cache import materialize_chart_vectors, _tensor_at
    from maimai_baseline.data.route import Route71Codec
    from maimai_baseline.data.curriculum import MultiViewRouteStore
    from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
    root, pilot = spec.runtime, Path(plan["pilot_root"])
    final = root / "spatial-views"
    source_final = root / "physical/source-rows"
    if final.exists() or source_final.exists():
        raise ValueError("spatial/source-row output already exists; refusing replacement")
    stage = spec.root / "preparation/spatial-stage"
    output, source_out = stage / "spatial", stage / "source-rows"
    require_new(output); require_new(source_out)
    pm = read(pilot / "spatial-views/manifest.json")
    reused = MultiViewRouteStore.load(pilot / "spatial-views", profile_fingerprint=pm["profile_fingerprint"],
                                    vocabulary_sha256=pm["vocabulary_sha256"], selection_sha256=pm["selection_sha256"], verify_payloads=True)
    sr = PhysicalSourceRowStore(pilot / "physical/source-rows", profile_fingerprint=pm["profile_fingerprint"],
                               vocabulary_sha256=pm["vocabulary_sha256"], chart_cache_manifest_sha256=pm["canonical_cache_manifest_sha256"],
                               spatial_cache_manifest_sha256=sha(pilot / "spatial-views/manifest.json"))
    offsets = np.load(root / "chart-cache/offsets.npy", allow_pickle=False)
    shutil.copyfile(root / "chart-cache/offsets.npy", output / "offsets.npy")
    shutil.copyfile(root / "chart-cache/segments.jsonl", output / "segments.jsonl")
    cached = sorted(rows(root / "chart-cache/segments.jsonl"), key=lambda s: s["cache_index"])
    segments = {s["segment_id"]: s for s in rows(root / "profile/segments.jsonl")}
    groups = defaultdict(list)
    for i, c in enumerate(cached):
        s = segments[c["segment_id"]]
        groups[s["record_id"]].append((i, s))
    records = profile_records(root)
    vocab = CompactVocabulary.from_dict(read(root / "profile/vocabulary.json")); codec = Route71Codec(vocab)
    reuse_ids = set(plan["reuse_record_ids"])
    views, maps = {}, {}
    temp_profile = stage / "profile"
    require_new(temp_profile)
    for view in pm['views']:
        view_id = VIEW_NAMES.index(view)
        arrays = {}
        for name, old in pm["views"][view]["arrays"].items():
            path = output / "arrays" / view / (name + ".npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=old["dtype"], shape=(int(offsets[-1]),))
        mapping_path = source_out / "views" / view / "source_rows.pack"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        indices = []
        with mapping_path.open("wb") as stream:
            for number, (rid, refs) in enumerate(groups.items()):
                doc = read(contained(root / "profile", records[rid]["encoded_path"]))
                if rid in reuse_ids:
                    mapping = sr.rows(view, rid).numpy()
                else:
                    # Canonical order is deliberately retained. No parser sort
                    # or speculative marker pairing; NOTE/branch identity is exact.
                    doc["rows"] = transform_rows(doc["rows"], view_id)
                    mapping = np.arange(len(doc["rows"]), dtype=np.int32)
                    dump(contained(temp_profile, records[rid]["encoded_path"]), doc)
                indices.append(pack_mapping(stream, mapping, rid))
                for i, seg in refs:
                    item = reused.record(seg["segment_id"], view) if rid in reuse_ids else codec.adapt_item(materialize_chart_vectors(profile_root=temp_profile, segment=seg, vocabulary=vocab))
                    begin, end = map(int, offsets[i:i + 2])
                    for name, array in arrays.items():
                        values = _tensor_at(item, name).numpy()
                        if values.shape != (end - begin,):
                            raise ValueError(f"spatial row length mismatch: {view}/{rid}/{name}")
                        array[begin:end] = values
                if number % 100 == 0:
                    print(f"{view}: {number + 1}/{len(groups)} charts", flush=True)
        index_path = mapping_path.with_name("source_rows.jsonl")
        index_path.write_text("".join(json.dumps(x) + "\n" for x in indices), encoding="utf-8")
        maps[view] = dict(source_row_index=meta(source_out, index_path), source_row_pack=meta(source_out, mapping_path))
        metadata = {}
        for name, array in arrays.items():
            array.flush()
            metadata[name] = meta(output, output / "arrays" / view / (name + ".npy"), array)
            array._mmap.close()
        views[view] = {"operation": pm["views"][view]["operation"], "arrays": metadata}
    binding = {"plan_identity": plan["identity"], "pilot_manifest_sha256": sha(pilot / "spatial-views/manifest.json"),
               "reuse_records": len(reuse_ids), "new_records": len(plan["build_record_ids"]),
               "new_row_order": "canonical-note-and-branch-order-preserved-v1"}
    manifest = dict(schema=VIEW_CACHE_SCHEMA, window=WINDOW,
                    profile_fingerprint=read(root / "profile/summary.json")["profile_fingerprint"],
                    profile_manifest_sha256=sha(root / "profile/manifest.jsonl"), vocabulary_sha256=sha(root / "profile/vocabulary.json"),
                    canonical_cache_manifest_sha256=sha(root / "chart-cache/manifest.json"),
                    selection_sha256=read(root / "chart-cache/manifest.json")["selection"]["selection_sha256"],
                    segment_count=len(cached), row_count=int(offsets[-1]), segment_index=meta(output, output / "segments.jsonl"),
                    offsets=meta(output, output / "offsets.npy", offsets), views=views,
                    route_informative_mass_by_class=pm["route_informative_mass_by_class"], reuse_provenance=binding)
    manifest["manifest_identity_sha256"] = identity(manifest)
    dump(output / "manifest.json", manifest)
    sm = dict(schema="maimai.physical_state.multiview.v2", window=WINDOW,
              source_row_mapping="view-position-to-canonical-int32-v1", chart_count=len(records), views=maps,
              profile_fingerprint=manifest["profile_fingerprint"], vocabulary_sha256=manifest["vocabulary_sha256"],
              canonical_cache_manifest_sha256=manifest["canonical_cache_manifest_sha256"],
              spatial_cache_manifest_sha256=sha(output / "manifest.json"), reuse_provenance=binding)
    sm["state_fingerprint"] = identity(sm); dump(source_out / "manifest.json", sm)
    MultiViewRouteStore.load(output, profile_fingerprint=manifest["profile_fingerprint"], vocabulary_sha256=manifest["vocabulary_sha256"],
                            selection_sha256=manifest["selection_sha256"])
    source_final.parent.mkdir(parents=True, exist_ok=True)
    output.replace(final)
    try:
        source_out.replace(source_final)
    except BaseException:
        final.replace(output)
        raise
    return {"passed": True, "plan_identity": plan["identity"], **binding}


def copy_resources(root, pilot):
    for directory, names in [("physical/path-catalog", ("paths.json", "runtime-path-semantics.json", "runtime-path-rules.json")),
                             ("route-artifact", ("manifest.json", "dual_route_embeddings_256d.npz"))]:
        for name in names:
            source, destination = pilot / directory / name, root / directory / name
            if destination.exists():
                if sha(destination) != sha(source):
                    raise ValueError(f"shared resource conflict: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha(destination) != sha(source):
                raise ValueError(f"shared resource copy failed: {destination}")


def bitplane_build(spec, plan, workers):
    from .bitplane import build_physical_bitplane_cache
    pilot, root = Path(plan["pilot_root"]), spec.runtime
    copy_resources(root, pilot)
    result = build_physical_bitplane_cache(profile_root=root / "profile", chart_cache_root=root / "chart-cache",
                 spatial_cache_root=root / "spatial-views", catalog_root=root / "physical/path-catalog",
                 output=root / "physical/bitplanes", workers=workers,
                 reuse={rid: str(pilot / "physical/bitplanes") for rid in plan["reuse_record_ids"]})
    return {"passed": True, "plan_identity": plan["identity"], "chart_count": result["chart_count"], "counts": plan["counts"]}
