"""Offline BPM audio audit, build, and verification."""

from __future__ import annotations
from maimai_dataset.window import WINDOW
from fractions import Fraction
from importlib.metadata import version
from maimai_baseline.data.audio import TARGET_P50, extract_at_centers, prepare_waveform, validate_stats
from maimai_baseline.data.store import (
    BpmFeatureStore,
    GEOMETRY_SCHEMA,
    SCHEMA,
    completed,
    contained,
    publish,
    read_json,
    read_jsonl,
    window_arrays,
)
from maimai_baseline.time.grid import (
    INTEGRATION,
    SAMPLING,
    audio_grid,
    identity,
    quantization_audit,
    timing_identity,
)
from maimai_dataset.audio import _imports, resample_linear_float32, sha256_file
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.timing import TimingMap
from pathlib import Path
import argparse
import json
import math
import numpy as np
import sys
import torch
import uuid


def environment():
    return {"interpreter": sys.executable, "python": sys.version,
            "dependencies": {name: version(name) for name in
                             ("numpy", "torch", "torchcodec", "av", "pyloudnorm", "soundfile", "scipy")}}


def tool_identity():
    from maimai_baseline.source_identity import baseline_source_identity
    return {"runtime": baseline_source_identity(), "entry_sha256": sha256_file(Path(__file__).parents[3] / "scripts/prepare-bpm96.py")}


def pilot_source_records(feature_manifest, audio_manifest, *, song_ids, selection_sha256,
                         chart_manifest_sha256, stats_sha256):
    """Read the accepted pilot's exact source selection; do not decode/rebuild features.

    Identical MP3 bytes can occur at several paths. FLOAT WAV files can also have
    different PEAK timestamps despite identical PCM. Neither permits guessing a
    source: the accepted feature record pins both the original path and WAV hash.
    """
    feature_manifest, audio_manifest = Path(feature_manifest), Path(audio_manifest)
    if feature_manifest.parent.resolve() != audio_manifest.parent.resolve():
        raise ValueError("pilot feature/audio manifests must belong to the same normalization output")
    root = feature_manifest.parent
    summary_path, normalization_path = root / "feature-summary.json", root / "normalization-summary.json"
    summary, normalization = read_json(summary_path), read_json(normalization_path)
    expected = {"feature_manifest_sha256": sha256_file(feature_manifest),
                "selection_sha256": selection_sha256,
                "selection_manifest_sha256": chart_manifest_sha256, "stats_sha256": stats_sha256,
                "song_count": len(song_ids)}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("accepted pilot feature summary/selection/statistics identity mismatch")
    if (normalization.get("target_p50_lufs") != TARGET_P50 or
            normalization.get("audio_manifest_sha256") != sha256_file(audio_manifest)):
        raise ValueError("accepted normalization summary/audio manifest identity mismatch")
    if sha256_file(root / "pilot/feature-stats.json") != stats_sha256:
        raise ValueError("accepted pilot statistics bytes mismatch")
    features, audio = read_jsonl(feature_manifest), read_jsonl(audio_manifest)
    by_song = {str(r["song_id"]): r for r in features}
    by_wav = {r["output_relative_path"]: r for r in audio}
    if len(by_song) != len(features) or set(by_song) != set(song_ids):
        raise ValueError("accepted pilot source selection must exactly cover train/validation songs")
    if len(by_wav) != len(audio) or normalization.get("count") != len(audio):
        raise ValueError("duplicate/incomplete accepted normalized audio records")
    selected = {}
    for song, f in by_song.items():
        a = by_wav.get(f["wav_relative_path"])
        if (a is None or a["relative_source_path"] != f["resolved_source_path"] or
                a["source_sha256"] != f["resolved_source_sha256"] or
                a["output_sha256"] != f["wav_sha256"] or f["stats_sha256"] != stats_sha256):
            raise ValueError(f"accepted pilot exact source/WAV identity mismatch: {song}")
        selected[song] = a
    provenance = {"pilot_feature_manifest_sha256": expected["feature_manifest_sha256"],
                  "pilot_feature_summary_sha256": sha256_file(summary_path),
                  "normalization_summary_sha256": sha256_file(normalization_path)}
    return selected, provenance


def source_for_record(record, selected):
    song = str(record["song_id"])
    a = selected.get(song)
    if a is None or a["source_sha256"] != record["audio"]["file_sha256"]:
        raise ValueError(f"accepted pilot source/profile SHA mismatch: {record['record_id']}")
    return a


def encoded_moments(document, record):
    """Original rational positions retained by the frozen chart-row codec."""
    if (document.get("schema") != "maimai.compact_training_rows.v2_test1" or
            document.get("row_codec") != "v2-test1-derived-each-v1" or
            document.get("record_id") != record["record_id"] or
            document.get("source_ir_sha256") != record["ir_sha256"]):
        raise ValueError("encoded chart schema/record/original IR provenance mismatch")
    rows, chunks = document["rows"], document["moment_chunks"]
    moments, end, current_tick = [], 0, 0
    for i, chunk in enumerate(chunks):
        begin, stop = chunk["row_start"], chunk["row_end"]
        if (not isinstance(begin, int) or not isinstance(stop, int) or
                begin != end or not begin < stop <= len(rows) or
                rows[begin].get("event") != "MOMENT" or
                any(r.get("event") == "MOMENT" for r in rows[begin + 1:stop])):
            raise ValueError(f"encoded moment row coverage mismatch: {i}")
        num, den = chunk["beat_num"], chunk["beat_den"]
        if not isinstance(num, int) or not isinstance(den, int) or den <= 0:
            raise ValueError(f"invalid original rational beat: {i}")
        tick = Fraction(num, den) * 96
        current_tick += rows[begin]["delta_tick"]
        if tick != current_tick or tick < 0 or (moments and tick <= moments[-1][1]):
            raise ValueError(f"encoded rational beat/teacher-forced tick mismatch: {i}")
        moments.append((i, tick))
        end = stop
    if end != len(rows) or not moments:
        raise ValueError("encoded moment chunks do not cover all chart rows")
    return moments


def compare_chart_vectors(expected, actual, names):
    from maimai_dataset.chart_cache import _tensor_at
    mismatches = {}
    for name in names:
        a, b = _tensor_at(expected, name), _tensor_at(actual, name)
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatches[name] = {"expected_shape": list(a.shape), "cached_shape": list(b.shape),
                                "expected_dtype": str(a.dtype), "cached_dtype": str(b.dtype)}
        elif not torch.equal(a, b):
            mismatches[name] = {"row_indices": torch.nonzero(a != b).flatten().tolist()}
    return mismatches


def inputs(args):
    profile = args.profile.resolve()
    chart = read_json(args.chart_cache / "manifest.json")
    if chart["profile_fingerprint"] != read_json(profile / "summary.json")["profile_fingerprint"]:
        raise ValueError("pilot chart-cache/profile fingerprint mismatch")
    if chart["vocabulary_sha256"] != sha256_file(profile / "vocabulary.json"):
        raise ValueError("pilot vocabulary identity mismatch")
    selection = chart["selection"]
    raw_selection = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if canonical_json_sha256(raw_selection) != selection["selection_sha256"]:
        raise ValueError("pilot selection SHA mismatch")
    train, validation = [list(map(str, selection[f"{role}_song_ids"])) for role in ("train", "validation")]
    if (len(train), len(validation)) != (256, 16) or len(set(train + validation)) != 272:
        raise ValueError("pilot must contain exactly 256 distinct train and 16 distinct validation songs")
    if sha256_file(profile / "manifest.jsonl") != chart["profile_manifest_sha256"]:
        raise ValueError("pilot chart-cache/profile manifest SHA mismatch")
    stats = read_json(args.stats)
    validate_stats(stats)
    profile_info = read_json(profile / "profile.json")
    if sha256_file((profile / profile_info["base_feature_stats"]).resolve()) != sha256_file(args.stats):
        raise ValueError("supplied statistics are not the original profile bytes")
    index_path = contained(args.chart_cache, chart["segment_index"]["path"])
    if sha256_file(index_path) != chart["segment_index"]["sha256"]:
        raise ValueError("chart cache segment-index SHA mismatch")
    index = read_jsonl(index_path)
    indexed = {str(r["segment_id"]): r for r in index}
    selected_ids = {str(r["segment_id"]) for r in index}
    all_segments = read_jsonl(profile / "segments.jsonl")
    segments = [s for s in all_segments if str(s["segment_id"]) in selected_ids]
    if len(segments) != len(selected_ids) or len(index) != chart["segment_count"] or len(index) != len(selected_ids):
        raise ValueError("pilot chart-cache/profile segment coverage mismatch")
    from maimai_dataset.chart_cache import audio_reference
    for s in segments:
        if audio_reference(s)[0] != indexed[str(s["segment_id"])]["audio_key"]:
            raise ValueError(f"pilot cached source/timing identity mismatch: {s['segment_id']}")
    songs = set(train + validation)
    if {str(s["song_id"]) for s in segments} != songs:
        raise ValueError("missing pilot songs in selected chart windows")
    records = {str(r["record_id"]): r for r in read_jsonl(profile / "manifest.jsonl")
               if str(r["record_id"]) in {str(s["record_id"]) for s in segments}}
    rows_root = getattr(args, "chart_rows_root", None)
    rows_identity = {}
    if rows_root is not None:
        if (read_json(rows_root / "summary.json")["profile_fingerprint"] != chart["profile_fingerprint"] or
                sha256_file(rows_root / "vocabulary.json") != chart["vocabulary_sha256"]):
            raise ValueError("chart rows backup/profile identity mismatch")
        # Do not substitute the backup's segment selection or timing for the pilot.
        backup = {str(r["record_id"]): r for r in read_jsonl(rows_root / "manifest.jsonl")}
        for rid, r in records.items():
            if rid not in backup or any(backup[rid].get(k) != r.get(k)
                    for k in ("ir_sha256", "encoded_path", "maidata_sha256", "chart_end_tick")):
                raise ValueError(f"chart rows backup/original profile provenance mismatch: {rid}")
        rows_identity = {"chart_rows_manifest_sha256": sha256_file(rows_root / "manifest.jsonl"),
                         "chart_rows_summary_sha256": sha256_file(rows_root / "summary.json")}
    stats_sha = sha256_file(args.stats)
    selected, source_provenance = pilot_source_records(args.pilot_feature_manifest, args.audio_manifest,
        song_ids=songs, selection_sha256=selection["selection_sha256"],
        chart_manifest_sha256=sha256_file(args.chart_cache / "manifest.json"), stats_sha256=stats_sha)
    variants, mapping, errors, bad_records = {}, {}, [], set()
    for s in segments:
        rid, sid = str(s["record_id"]), str(s["segment_id"])
        record = records[rid]
        source_hash = record["audio"]["file_sha256"]
        if rid in bad_records:
            continue
        try:
            a = source_for_record(record, selected)
        except ValueError as error:
            errors.append({"record_id": rid, "kind": "source-mapping", "error": str(error), "source_sha256": source_hash})
            bad_records.add(rid)
            continue
        original = contained(args.workspace_root, a["relative_source_path"])
        source = original if args.audio_policy == "legacy" else contained(args.audio_manifest.parent, a["output_relative_path"])
        expected_hash = source_hash if args.audio_policy == "legacy" else a["output_sha256"]
        t = TimingMap.from_dict(s["timing"])
        material = {"original_sha256": source_hash, "timing": timing_identity(t),
                    "source_sha256": expected_hash, "base_policy": args.audio_policy,
                    "stats_sha256": stats_sha, "sampling": SAMPLING}
        key = identity(material)
        mapping[sid] = key
        if key not in variants:
            variants[key] = {"identity": material, "source_path": str(source), "source_sha256": expected_hash,
                "original_path": str(original), "timing": t.to_dict(), "song_ids": [],
                "old_frames": int(record["audio"]["feature_frames"]),
                "num_samples": max(1, round(a["samples_per_channel"] * 24000 / a["source_sample_rate"])),
                "audio": {"source_p50_lufs": a["source_p50_lufs"], "normalization_gain_db": 0 if args.audio_policy == "legacy" else a["gain_db"]}}
            try:
                if sha256_file(original) != source_hash or sha256_file(source) != expected_hash:
                    raise ValueError("source audio SHA mismatch")
                if args.audio_policy != "legacy":
                    p50 = a["output_momentary"]["momentary_p50_lufs"]
                    if a["target_p50_lufs"] != TARGET_P50 or not math.isfinite(p50) or abs(p50 - TARGET_P50) > .1:
                        raise ValueError("normalization target/acceptance mismatch")
            except Exception as error:
                errors.append({"variant": key, "kind": "source-identity", "error": str(error)})
        if str(s["song_id"]) not in variants[key]["song_ids"]:
            variants[key]["song_ids"].append(str(s["song_id"]))
    material = {"profile_manifest_sha256": sha256_file(profile / "manifest.jsonl"),
        "segments_sha256": sha256_file(profile / "segments.jsonl"),
        "chart_cache_manifest_sha256": sha256_file(args.chart_cache / "manifest.json"),
        "selection_sha256": selection["selection_sha256"], "stats_sha256": stats_sha,
        "audio_manifest_sha256": sha256_file(args.audio_manifest), "base_policy": args.audio_policy,
        **source_provenance,
        "chart_evidence_mode": "encoded-rows" if rows_root is not None else "original-ir", **rows_identity,
        "sampling": SAMPLING, "integration": INTEGRATION,
        "tools": tool_identity(), "environment": environment()}
    return {"material": material, "identity": identity(material), "profile": profile,
            "chart": chart, "segments": segments, "records": records, "variants": variants,
            "mapping": mapping, "errors": errors, "stats": stats, "song_ids": train + validation}


def offset_analysis(path, timing, duration):
    """Read only the first eight beats and FFT/search context, never a full track."""
    first = timing.beat_to_second(Fraction(0))
    end = timing.beat_to_second(Fraction(8))
    start, stop = max(0., first - .060), min(duration, end + .060)
    if stop <= start or duration <= 0:
        return {"status": "missing-evidence", "reason": "first eight beats outside audio"}
    decoder = _imports()[3](str(path))
    samples = decoder.get_samples_played_in_range(start_seconds=start, stop_seconds=stop)
    wave = samples.data.float().mean(0, keepdim=True)
    rate = int(samples.sample_rate)
    actual_start = float(samples.pts_seconds)
    wave = resample_linear_float32(wave, max(1, round(wave.shape[1] * 24000 / rate)))
    if wave.shape[1] < 1024:
        return {"status": "missing-evidence", "reason": "insufficient FFT context"}
    center_seconds = actual_start + np.arange(512, wave.shape[1] - 512, 24) / 24000
    coords = {"samples": np.arange(512, wave.shape[1] - 512, 24), "valid": np.ones(len(center_seconds), dtype=bool)}
    mel = extract_at_centers(wave, coords)
    flux = np.maximum(0, np.diff(mel, axis=1)).mean(0)
    flux_times = center_seconds[1:]
    offsets = np.arange(-30, 31)
    blocks, missing = [], []
    for begin in (0, 4):
        probes = np.asarray([timing.beat_to_second(Fraction(b)) for b in range(begin, begin + 4)])
        # One common evidence set across all offsets prevents edge bias.
        evidence = (probes - .030 >= flux_times[0]) & (probes + .030 <= flux_times[-1])
        missing.append([int(begin + i) for i in np.flatnonzero(~evidence)])
        blocks.append([float(np.interp(probes[evidence] + delta / 1000, flux_times, flux).mean())
                       if evidence.any() else None for delta in offsets])
    combined = [(a + b) / 2 if a is not None and b is not None else None for a, b in zip(*blocks)]
    def summary(curve):
        if curve[30] is None or max(curve) <= 0:
            return {"curve": curve, "best_offset_ms": None, "score_change": None, "evidence": "missing-or-flat"}
        i = min(range(61), key=lambda i: (-curve[i], abs(int(offsets[i])), int(offsets[i])))
        return {"curve": curve, "best_offset_ms": int(offsets[i]), "score_change": curve[i] - curve[30]}
    return {"status": "suggestion-only", "applied": False, "read_seconds": [start, stop],
            "step_ms": 1, "offsets_ms": offsets.tolist(), "first_four": summary(blocks[0]),
            "second_four": summary(blocks[1]), "joint": summary(combined), "missing_quarter_beats": missing,
            "evidence_definition": "positive log-Mel spectral change at quarter-beat probes; not onset labels"}


def audit(args, data):
    conflicts, timing_rows, offsets = list(data["errors"]), [], {}
    ir_hashes, chart_evidence = {}, {}
    rows_root = getattr(args, "chart_rows_root", None)
    store = vocabulary = None
    if rows_root is not None:
        from maimai_dataset.chart_cache import ChartVectorStore
        from maimai_dataset.vocabulary import CompactVocabulary
        store = ChartVectorStore.load(args.chart_cache, profile_fingerprint=data["chart"]["profile_fingerprint"],
                                     vocabulary_sha256=data["chart"]["vocabulary_sha256"])
        vocabulary = CompactVocabulary.from_dict(read_json(data["profile"] / "vocabulary.json"))
    by_record = {}
    for s in data["segments"]:
        by_record.setdefault(str(s["record_id"]), []).append(s)
    for rid, segments in by_record.items():
        r = data["records"][rid]
        evidence = {"expected_original_ir_sha256" if rows_root is not None else "expected_canonical_sha256": r["ir_sha256"]}
        try:
            path = contained(rows_root, r["encoded_path"]) if rows_root is not None else contained(args.chart_ir_root or data["profile"], r["ir_path"])
            evidence["path"] = str(path)
            digest = sha256_file(path)
            document = read_json(path)
            canonical_digest = canonical_json_sha256(document)
            evidence.update(file_sha256=digest, actual_canonical_sha256=canonical_digest)
            if rows_root is not None:
                evidence["declared_original_ir_sha256"] = document.get("source_ir_sha256")
            timing = TimingMap.from_dict(segments[0]["timing"])
            if rows_root is not None:
                moments = encoded_moments(document, r)
                from maimai_dataset.chart_cache import materialize_chart_vectors, _encoded
                _encoded.cache_clear()  # A re-run must not reuse a prior file snapshot.
                for s in segments:
                    try:
                        expected = materialize_chart_vectors(profile_root=rows_root, segment=s, vocabulary=vocabulary)
                        differences = compare_chart_vectors(expected, store.record(s["segment_id"]), store.arrays)
                        if differences:
                            conflicts.append({"record_id": rid, "segment_id": s["segment_id"],
                                              "kind": "chart-cache-evidence", "differences": differences})
                    except Exception as error:
                        conflicts.append({"record_id": rid, "segment_id": s["segment_id"],
                                          "kind": "chart-cache-evidence", "error": str(error)})
                if sha256_file(path) != digest:
                    raise ValueError("encoded chart changed during cache comparison")
            else:
                if canonical_digest != r["ir_sha256"]:
                    raise ValueError("original Chart IR canonical JSON SHA mismatch (profile identity; not file-byte SHA)")
                from maimai_chart_ir.models import validate_chart_ir
                validate_chart_ir(document)
                moments = [(i, Fraction(m["position_beats"]["num"], m["position_beats"]["den"]) * 96)
                           for i, m in enumerate(document["moments"])]
                if timing_identity(TimingMap.from_document(document)) != timing_identity(timing):
                    raise ValueError("original Chart IR and cached timing disagree")
                ir_hashes[rid] = {"path": str(path), "sha256": digest, "canonical_sha256": canonical_digest}
            chart_evidence[rid] = {"kind": "encoded-rows" if rows_root is not None else "original-ir",
                "path": str(path), "sha256": digest, "canonical_sha256": canonical_digest,
                "original_ir_sha256": r["ir_sha256"], "checked_segments": len(segments),
                "rational_positions": "moment_chunks" if rows_root is not None else "position_beats"}
            whole = quantization_audit(timing, moments, start_tick=0, end_tick=int(segments[0]["song_end_tick"]) + 1)
            timing_rows.append({"record_id": rid, "scope": "song", **whole,
                "source_evidence": "encoded-moment-chunks" if rows_root is not None else "original-ir-moments",
                **({"source_row_ranges": [{"moment_index": i, "row_start": c["row_start"], "row_end": c["row_end"]}
                    for i, c in enumerate(document["moment_chunks"])]} if rows_root is not None else {})})
            conflicts.extend({"record_id": rid, **c} for c in whole["conflicts"])
            for s in segments:
                start, end = int(s["target_start_tick"]), min(int(s["target_end_tick"]), int(s["song_end_tick"]) + 1)
                report = quantization_audit(timing, [(i, t) for i, t in moments if start <= t < end], start_tick=start, end_tick=end)
                conflicts.extend({"segment_id": s["segment_id"], **c} for c in report["conflicts"])
        except Exception as error:
            conflicts.append({"record_id": rid, "kind": "encoded-chart-evidence" if rows_root is not None else "original-chart-evidence", "error": str(error), **evidence})
    for key, v in data["variants"].items():
        t = TimingMap.from_dict(v["timing"])
        v["new_frames"] = len(audio_grid(t, v["num_samples"])["grid_k"])
        try:
            offsets[key] = offset_analysis(Path(v["source_path"]), t, v["num_samples"] / 24000)
        except Exception as error:
            offsets[key] = {"status": "missing-evidence", "error": str(error)}
    report = {"schema": "maimai.bpm96_audit.v1", "complete": True, "passed": not conflicts,
        "sampling": SAMPLING, "integration": INTEGRATION,
        "preparation_identity": data["identity"], "inputs": data["material"], "ir_hashes": ir_hashes,
        "chart_evidence": chart_evidence,
        "song_count": len(data["song_ids"]), "chart_count": len(by_record), "segment_count": len(data["segments"]),
        "resolved_segment_count": len(data["mapping"]), "checked_chart_count": len(chart_evidence),
        "coverage_complete": len(data["mapping"]) == len(data["segments"]) and len(chart_evidence) == len(by_record),
        "variants": data["variants"], "quantization": timing_rows, "conflicts": conflicts, "offset_analysis": offsets}
    publish(args.output / "audit.json", report, data["material"])
    print(json.dumps({"stage": "audit", "passed": report["passed"], "conflicts": len(conflicts)}, ensure_ascii=False))
    return not conflicts


def build(args, data):
    if completed(args.output / "audit.json", data["material"]) is None:
        raise ValueError("build requires a completed audit")
    report = read_json(args.output / "audit.json")
    if (report.get("complete") is not True or report.get("passed") is not True or report.get("conflicts")
            or report.get("sampling") != SAMPLING or report.get("integration") != INTEGRATION
            or report.get("preparation_identity") != data["identity"] or data["errors"]):
        raise ValueError("build requires matching completed audit without blocking conflicts")
    if set(report["chart_evidence"]) != set(data["records"]):
        raise ValueError("audit original chart evidence coverage is incomplete")
    for rid, r in report["chart_evidence"].items():
        if (sha256_file(Path(r["path"])) != r["sha256"] or
                canonical_json_sha256(read_json(Path(r["path"]))) != r["canonical_sha256"] or
                r["original_ir_sha256"] != data["records"][rid]["ir_sha256"]):
            raise ValueError("Chart evidence changed after audit")
    variants, geometry = {}, {}
    if sha256_file(args.stats) != data["material"]["stats_sha256"]:
        raise ValueError("statistics changed after audit")
    publish(args.output / "feature-stats.json", args.stats.read_bytes(), data["material"])
    for key, v in data["variants"].items():
        t = TimingMap.from_dict(v["timing"])
        coords = audio_grid(t, v["num_samples"])
        artifact_identity = {"preparation": data["identity"], "variant": v["identity"]}
        fpath, cpath = f"features/{key}.npy", f"coordinates/{key}.npz"
        info = {"num_samples": v["num_samples"], "sample_rate": 24000,
                "duration_seconds": v["num_samples"] / 24000, **v["audio"]}
        if completed(args.output / fpath, artifact_identity) is None:
            wave, decoded = prepare_waveform(v["source_path"], policy=args.audio_policy,
                                              already_normalized=args.audio_policy != "legacy")
            if decoded["source_sha256"] != v["source_sha256"]:
                raise ValueError(f"source changed after audit: {key}")
            if wave.shape[1] != v["num_samples"]:
                raise ValueError(f"decoded sample count differs from audited source: {key}")
            features = extract_at_centers(wave, coords, data["stats"])
        else:
            features = np.load(args.output / fpath, mmap_mode="r", allow_pickle=False)
        f = publish(args.output / fpath, features, artifact_identity)
        c = publish(args.output / cpath, coords, artifact_identity)
        variants[key] = {**v, "feature_path": fpath, "feature_sha256": f["sha256"],
                         "coordinates_path": cpath, "coordinates_sha256": c["sha256"],
                         "dtype": "float32", "shape": list(features.shape), "audio": {**info, **v["audio"]}}
        windows = {}
        for s in data["segments"]:
            if data["mapping"][str(s["segment_id"])] != key:
                continue
            start = int(s["target_start_tick"])
            if start not in windows:
                values = window_arrays(features, coords, t, start, v["num_samples"])[4]
                name = f"{key}-{start}.npz"
                receipt = publish(args.output / "geometry" / name, values, {**artifact_identity, "start_tick": start})
                windows[start] = {"path": name, "sha256": receipt["sha256"]}
            geometry[str(s["segment_id"])] = windows[start]
    geom = {"schema": GEOMETRY_SCHEMA, "profile_fingerprint": data["chart"]["profile_fingerprint"],
            "suite_sha256": data["material"]["selection_sha256"], "segment_count": len(geometry), "segments": geometry}
    g = publish(args.output / "geometry/manifest.json", geom, data["material"])
    manifest = {"schema": SCHEMA, "complete": True, "window": WINDOW, "geometry_root": "geometry", "sampling": SAMPLING, "integration": INTEGRATION,
        "base_policy": args.audio_policy, "stats_path": "feature-stats.json", "stats_sha256": data["material"]["stats_sha256"],
        "selection_sha256": data["material"]["selection_sha256"], "song_ids": data["song_ids"],
        "profile_manifest_sha256": data["material"]["profile_manifest_sha256"], "segments_sha256": data["material"]["segments_sha256"],
        "preparation_identity": data["identity"], "normalization_manifest_sha256": data["material"]["audio_manifest_sha256"],
        "geometry_manifest_sha256": g["sha256"], "variants": variants, "segments": data["mapping"],
        "audit_sha256": sha256_file(args.output / "audit.json")}
    publish(args.output / "bpm-audio-manifest.json", manifest, data["material"])
    return True


def verify(args, data):
    failures = list(data["errors"])
    # Source byte verification belongs to data preparation, not training or
    # prebuilt-feature inference. Keep it here even for direct verify() calls.
    for key, variant in data["variants"].items():
        for field, expected in (("original_path", variant["identity"]["original_sha256"]),
                                ("source_path", variant["source_sha256"])):
            try:
                if sha256_file(Path(variant[field])) != expected:
                    raise ValueError("source audio SHA mismatch")
            except Exception as error:
                failures.append({"variant": key, "path": variant[field],
                                 "kind": "source-identity", "error": str(error)})
    manifest = args.output / "bpm-audio-manifest.json"
    files = []
    # Enumerate all missing/corrupt independent variants, even after interruption.
    for key in data["variants"]:
        for name in (f"features/{key}.npy", f"coordinates/{key}.npz"):
            path = args.output / name
            try:
                receipt = read_json(path.with_name(path.name + ".complete.json"))
                digest = sha256_file(path)
                if digest != receipt["sha256"] or receipt["identity"]["preparation"] != data["identity"]:
                    raise ValueError("hash or preparation identity mismatch")
                files.append({"path": name, "sha256": digest, "passed": True})
            except Exception as error:
                failures.append({"path": name, "kind": "file-verification", "error": str(error)})
                files.append({"path": name, "passed": False})
    checked_geometry = set()
    for s in data["segments"]:
        sid = str(s["segment_id"])
        if sid not in data["mapping"]:
            failures.append({"segment_id": sid, "kind": "missing-variant"})
            continue
        name = f"geometry/{data['mapping'][sid]}-{int(s['target_start_tick'])}.npz"
        if name in checked_geometry:
            continue
        checked_geometry.add(name)
        path = args.output / name
        try:
            receipt = read_json(path.with_name(path.name + ".complete.json"))
            digest = sha256_file(path)
            if digest != receipt["sha256"] or receipt["identity"]["preparation"] != data["identity"]:
                raise ValueError("geometry hash or preparation identity mismatch")
            files.append({"path": name, "sha256": digest, "passed": True})
        except Exception as error:
            failures.append({"segment_id": sid, "path": name, "kind": "geometry-file", "error": str(error)})
            files.append({"path": name, "passed": False})
    try:
        m = read_json(manifest)
        gm = read_json(args.output / "geometry/manifest.json")
        for sid, row in gm["segments"].items():
            try:
                if sha256_file(contained(args.output / "geometry", row["path"])) != row["sha256"]:
                    raise ValueError("geometry file hash mismatch")
            except Exception as error:
                failures.append({"segment_id": sid, "path": row["path"], "kind": "geometry-file", "error": str(error)})
        if m["preparation_identity"] != data["identity"] or m["audit_sha256"] != sha256_file(args.output / "audit.json"):
            raise ValueError("BPM verification audit/input identity mismatch")
        BpmFeatureStore(manifest, profile=data["profile"], stats=data["stats"], song_ids=set(data["song_ids"]),
            selection_sha256=data["material"]["selection_sha256"], geometry_root=args.output / "geometry",
            expected_segment_ids=set(data["mapping"]), verify_payloads=True)
    except Exception as error:
        failures.append({"kind": "artifact-verification", "error": str(error)})
    result = {"schema": "maimai.bpm96_verification.v1", "complete": not failures, "passed": not failures,
              "preparation_identity": data["identity"], "failures": failures,
              "files": files,
              "expected_songs": len(data["song_ids"]), "expected_variants": len(data["variants"]), "expected_segments": len(data["segments"]),
              "offsets_applied": False, "gpu_acceptance": "not-run", "environment": environment()}
    # A failed verification is diagnostic and can be replaced on a later retry.
    args.output.mkdir(parents=True, exist_ok=True)
    def diagnostic(name, value):
        path = args.output / name
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(value, encoding="utf8")
        temporary.replace(path)
    diagnostic("verification.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    text = f"# BPM96 验收报告\n\n整体通过：{not failures}。期望 {len(data['song_ids'])} 首、{len(data['variants'])} 份 timing 特征、{len(data['segments'])} 个谱面窗口。\n\n起始偏移未修改；GPU 与加速验收未执行。\n\n失败明细：\n\n```json\n{json.dumps(failures, ensure_ascii=False, indent=2)}\n```\n"
    diagnostic("验收报告.md", text)
    return not failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="BPM96 pilot preparation; no training or inference")
    parser.add_argument("stage", choices=("audit", "build", "verify"))
    for name in ("profile", "chart-cache", "audio-manifest", "pilot-feature-manifest", "workspace-root", "stats", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--chart-ir-root", type=Path, help="root containing original ir_path files; canonical JSON SHA must match profile")
    evidence.add_argument("--chart-rows-root", type=Path, help="original encoded profile backup; rational chunks and every cached chart vector are verified")
    parser.add_argument("--audio-policy", choices=("legacy", "normalized-p50"), default="normalized-p50")
    args = parser.parse_args(argv)
    data = inputs(args)
    return 0 if globals()[args.stage](args, data) else 1
