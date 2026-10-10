"""Explicit audit/build/verify for the v10 chart view; audio arrays remain shared."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from maimai_dataset.modifiers import CONTRACT as MODIFIERS, normalize_row, materialize_row, row_bits, row_masks
from maimai_dataset.integrity import file_sha256
from maimai_dataset.timing import TimingMap
from maimai_baseline.time.audio_end import CONTRACT as AUDIO_END, audio_end, duration_upper
from maimai_baseline.time.duration import LINEAR_CONTRACT

SCHEMA = "maimai.note_view.v1"
PILOT_CANONICAL_SOURCE_SHA256 = "2609cc287612676698f4b0d643bfd17433d5c907b51d5620d3514d65ae76710a"
FULL_CANONICAL_SOURCE_SHA256 = "b7f929242f2da3b0bad66ab2a1e150af6c554565b56e3e4309f71a7efa23469a"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("note view path escapes root")
    return path

def inspect_chart(document, timing, num_samples):
    boundary = audio_end(timing, int(num_samples))
    canonical, physical, errors = [], [], []
    counts = Counter()
    tick = 0
    for index, original in enumerate(document["rows"]):
        row = normalize_row(original)
        if row.get("event") == "MOMENT":
            tick += int(row["delta_tick"])
            if tick > boundary.onset_tick:
                errors.append({"row": index, "reason": "onset-after-audio", "tick": tick})
        restored = materialize_row(row)
        for field in ("hold_duration_tick", "branch_duration_tick"):
            if field in restored:
                d = int(restored[field])
                maximum = duration_upper(field, tick, timing=timing, end_seconds=boundary.seconds)
                if d < 1 or d > maximum:
                    errors.append({"row": index, "reason": "duration-outside-audio", "field": field,
                                   "duration": d, "maximum": maximum, "tick": tick})
        bits = row_bits(row)
        counts.update({k: v for k, v in bits.items() if v})
        canonical.append(row)
        physical.append(restored)
    return dict(canonical_rows=canonical, physical_rows=physical, bits=[row_bits(r) for r in canonical],
                masks=[row_masks(r) for r in canonical], source_row_ids=list(range(len(canonical))),
                counts=dict(counts), errors=errors, audio_num_samples=int(num_samples),
                audio_end_seconds=boundary.seconds, audio_onset_tick=boundary.onset_tick,
                audio_tail_tick=boundary.tail_tick)

def audit(profile, bpm_manifest, *, excluded_record_ids=()):
    profile, bpm_manifest = Path(profile).resolve(), Path(bpm_manifest).resolve()
    audio = read(bpm_manifest)
    segments = jsonl(profile / "segments.jsonl")
    records = {r["record_id"]: r for r in jsonl(profile / "manifest.jsonl")}
    chosen = {}
    for segment in segments:
        sid = segment["segment_id"]
        if sid not in audio["segments"]:
            continue
        key = audio["segments"][sid]
        if segment["record_id"] in chosen and chosen[segment["record_id"]]["variant"] != key:
            raise ValueError("a chart has conflicting audio variants")
        chosen[segment["record_id"]] = dict(segment=segment, variant=key)
    excluded = set(excluded_record_ids)
    if len(excluded) != len(tuple(excluded_record_ids)):
        raise ValueError("duplicate excluded record id")
    if excluded - set(chosen):
        raise ValueError(f"excluded charts absent from note view: {sorted(excluded - set(chosen))}")
    charts, errors, excluded_errors, counts = {}, [], [], Counter()
    for rid, selected in chosen.items():
        path = contained(profile, records[rid]["encoded_path"])
        document = read(path)
        t = TimingMap.from_dict(selected["segment"]["timing"])
        info = inspect_chart(document, t, audio["variants"][selected["variant"]]["num_samples"])
        destination = excluded_errors if rid in excluded else errors
        destination.extend(dict(record_id=rid, **e) for e in info["errors"])
        counts.update(info["counts"])
        charts[rid] = dict(source=str(path), source_sha256=file_sha256(path),
                           variant=selected["variant"], info=info)
    if not charts:
        raise ValueError("note view selection is empty")
    return dict(schema=SCHEMA, modifiers=MODIFIERS, duration=LINEAR_CONTRACT, audio_end=AUDIO_END,
        profile=str(profile), profile_manifest_sha256=file_sha256(profile / "manifest.jsonl"),
        segments_sha256=file_sha256(profile / "segments.jsonl"),
        bpm_manifest=str(bpm_manifest), bpm_manifest_sha256=file_sha256(bpm_manifest),
        charts=charts, counts=dict(counts), errors=errors,
        excluded_record_ids=sorted(excluded), excluded_errors=excluded_errors, ready=False)

def build(profile, bpm_manifest, output, *, excluded_record_ids=()):
    report = audit(profile, bpm_manifest, excluded_record_ids=excluded_record_ids)
    if report["errors"]:
        raise ValueError("GT exceeds audio boundary: " + json.dumps(report["errors"][:20], ensure_ascii=False))
    profile_root = Path(profile).resolve()
    for chart in report["charts"].values():
        chart["source"] = Path(chart["source"]).relative_to(profile_root).as_posix()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("note view output already exists")
    output.mkdir(parents=True)
    write(output / "manifest.json", report)
    for number, (rid, chart) in enumerate(report["charts"].items()):
        relative = f"charts/{number:08d}.json"
        payload = dict(record_id=rid, source_sha256=chart["source_sha256"], **chart.pop("info"))
        write(output / relative, payload)
        chart.update(path=relative, sha256=file_sha256(output / relative),
                     audio_num_samples=payload["audio_num_samples"], rows=len(payload["canonical_rows"]))
    write(output / "manifest.json", report)
    verify(output / "manifest.json", require_ready=False)
    report["ready"] = True
    write(output / "manifest.json", report)
    return report

def verify(manifest, *, profile=None, bpm_manifest=None, require_ready=True):
    manifest = Path(manifest).resolve()
    value = load_metadata(manifest, profile=profile, bpm_manifest=bpm_manifest,
                          require_ready=require_ready)
    profile_root = Path(profile or value["profile"]).resolve()
    for chart in value["charts"].values():
        source = Path(chart["source"])
        if not source.is_absolute():
            source = contained(profile_root, chart["source"])
        if file_sha256(source) != chart["source_sha256"]:
            raise ValueError("note view source changed")
        path = contained(manifest.parent, chart["path"])
        if file_sha256(path) != chart["sha256"]:
            raise ValueError("note view payload changed")
        p = read(path)
        raw = read(source)["rows"]
        if p["canonical_rows"] != [normalize_row(r) for r in raw] or p["physical_rows"] != [materialize_row(normalize_row(r)) for r in raw]:
            raise ValueError("note view semantic mismatch")
        if p["bits"] != [row_bits(r) for r in p["canonical_rows"]] or len(raw) != chart["rows"]:
            raise ValueError("note view bits/source-row mismatch")
        if p["masks"] != [row_masks(r) for r in p["canonical_rows"]] or p["source_row_ids"] != list(range(len(raw))):
            raise ValueError("note view mask/source-row mismatch")
    return {"schema": SCHEMA, "verified_charts": len(value["charts"]), "ready": value["ready"]}

def load_metadata(manifest, *, profile=None, bpm_manifest=None, require_ready=True,
                  canonical_source=False):
    manifest = Path(manifest).resolve()
    value = read(manifest)
    if value.get("schema") != SCHEMA or value.get("modifiers") != MODIFIERS:
        raise ValueError("note view row/modifier layout mismatch")
    if require_ready and not value.get("ready"):
        raise ValueError("note view is not ready")
    return value
