from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .encoding import (
    ALL_DIFFICULTIES_PROFILE_NAME,
    CHART_TICKS_PER_BEAT,
    CHART_TICKS_PER_WHOLE,
    PREFIX_BEATS,
    TARGET_BEATS,
    AUDIO_CONTEXT_BEATS,
    TIME_FIELDS,
    _segment_rows,
    chart_tick_semantic_projection,
    restore_compound_row,
)
from .manifest import DatasetError, canonical_json_sha256, load_jsonl, write_jsonl
from .timing import TimingMap


PROFILE_NAME = "v2-test1-all-difficulties"
PROFILE_SCHEMA = "maimai.dataset_profile.v2_test1"
PROFILE_RECORD_SCHEMA = "maimai.dataset_profile_record.v2_test1"
TRAINING_ROWS_SCHEMA = "maimai.compact_training_rows.v2_test1"
SEGMENT_SCHEMA = "maimai.compact_segment.v2_test1"
VOCABULARY_SCHEMA = "maimai.compact_row_vocabulary.v2_test1"
SEMANTIC_AUDIT_SCHEMA = "maimai.compact_semantic_audit.v2_test1"
ROW_CODEC = "v2-test1-derived-each-v1"

EVENTS = (
    "SEGMENT_START",
    "TARGET_START",
    "SEGMENT_END",
    "MOMENT",
    "NOTE",
    "SLIDE_BRANCH",
)


def _flag_parts(value: Any) -> list[str]:
    return sorted({part for part in str(value or "").split("+") if part})


def _without_each(value: Any) -> str:
    return "+".join(part for part in _flag_parts(value) if part != "each")


def _with_each(value: Any) -> str:
    return "+".join(sorted({*_flag_parts(value), "each"}))


def _canonical_note_each(note: Mapping[str, Any], *, each: bool) -> dict[str, Any]:
    result = dict(note)
    kind = str(result.get("kind", ""))
    result["flags"] = _without_each(result.get("flags", ""))
    if kind in {"slide", "wifi"}:
        result["head_flags"] = _without_each(result.get("head_flags", ""))
        if each:
            result["head_flags"] = _with_each(result["head_flags"])
    elif each:
        result["flags"] = _with_each(result["flags"])
    return result


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compress_moment(source: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not source or source[0].get("event") != "MOMENT":
        raise DatasetError("v2-test1 moment chunk must start with MOMENT")
    if len(source) < 4 or source[1].get("event") != "GROUP":
        raise DatasetError("v2-test1 moment chunk must contain one GROUP after MOMENT")
    if sum(row.get("event") == "GROUP" for row in source) != 1:
        raise DatasetError("structure:multiple_groups_per_moment")
    if source[-1].get("event") != "MOMENT_END":
        raise DatasetError("v2-test1 moment chunk must end with MOMENT_END")

    result = [dict(source[0])]
    index = 2
    note_count = 0
    while index < len(source) - 1:
        note = source[index]
        if note.get("event") != "NOTE":
            raise DatasetError(
                f"v2-test1 expected NOTE, got {note.get('event')!r}"
            )
        compressed_note = dict(note)
        compressed_note["flags"] = _without_each(compressed_note.get("flags", ""))
        if str(compressed_note.get("kind", "")) in {"slide", "wifi"}:
            compressed_note["head_flags"] = _without_each(
                compressed_note.get("head_flags", "")
            )
        result.append(compressed_note)
        note_count += 1
        kind = str(note.get("kind", ""))
        index += 1
        while index < len(source) - 1 and source[index].get("event") == "SLIDE_BRANCH":
            if kind not in {"slide", "wifi"}:
                raise DatasetError("v2-test1 non-Slide NOTE cannot own SLIDE_BRANCH")
            result.append(dict(source[index]))
            index += 1
        if index >= len(source) - 1 or source[index].get("event") != "NOTE_END":
            raise DatasetError("v2-test1 source NOTE is missing NOTE_END")
        index += 1
    if note_count == 0:
        raise DatasetError("v2-test1 moment cannot be empty")
    return result


def compress_compact_stream(
    rows: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove deterministic structure rows while retaining moment boundaries."""
    compressed: list[dict[str, Any]] = []
    compressed_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        source = rows[int(chunk["row_start"]) : int(chunk["row_end"])]
        start = len(compressed)
        compressed.extend(_compress_moment(source))
        compressed_chunks.append(
            {
                "beat_num": int(chunk["beat_num"]),
                "beat_den": int(chunk["beat_den"]),
                "row_start": start,
                "row_end": len(compressed),
            }
        )
    return compressed, compressed_chunks


def _expand_moment(source: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(source) < 2 or source[0].get("event") != "MOMENT":
        raise DatasetError("v2-test1 compressed moment must start with MOMENT")
    moment = dict(source[0])
    if "each" in moment:
        raise DatasetError("v2-test1 MOMENT must not contain derived each")
    note_count = sum(row.get("event") == "NOTE" for row in source[1:])
    if note_count == 0:
        raise DatasetError("v2-test1 compressed moment cannot be empty")
    each = note_count >= 2
    result: list[dict[str, Any]] = [moment, {"event": "GROUP", "each": each}]
    index = 1
    parsed_notes = 0
    while index < len(source):
        note = source[index]
        if note.get("event") != "NOTE":
            raise DatasetError(
                f"v2-test1 compressed moment expected NOTE, got {note.get('event')!r}"
            )
        result.append(_canonical_note_each(note, each=each))
        parsed_notes += 1
        kind = str(note.get("kind", ""))
        index += 1
        while index < len(source) and source[index].get("event") == "SLIDE_BRANCH":
            if kind not in {"slide", "wifi"}:
                raise DatasetError("v2-test1 non-Slide NOTE cannot own SLIDE_BRANCH")
            result.append(dict(source[index]))
            index += 1
        result.append({"event": "NOTE_END"})
    if parsed_notes != note_count:
        raise DatasetError("v2-test1 compressed moment note count changed")
    result.append({"event": "MOMENT_END"})
    return result


def expand_compact_stream(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand a complete v2-test1 chart stream into compact-v2 structure rows."""
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        if rows[start].get("event") != "MOMENT":
            raise DatasetError(
                f"v2-test1 chart expected MOMENT, got {rows[start].get('event')!r}"
            )
        end = start + 1
        while end < len(rows) and rows[end].get("event") != "MOMENT":
            if rows[end].get("event") == "SEGMENT_END":
                break
            end += 1
        result.extend(_expand_moment(rows[start:end]))
        start = end
        if start < len(rows) and rows[start].get("event") == "SEGMENT_END":
            if start != len(rows) - 1:
                raise DatasetError("v2-test1 SEGMENT_END must be final")
            break
    return result


def restore_v1_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [restore_compound_row(row) for row in expand_compact_stream(rows)]


def _build_vocabulary(streams: Iterable[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    def key(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    values: dict[str, set[str]] = {"event": {key(event) for event in EVENTS}}
    applicability: dict[str, set[str]] = {event: {"event"} for event in EVENTS}
    event_constraints: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    kind_constraints: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    time_values: dict[str, set[int]] = {field: set() for field in TIME_FIELDS}
    for rows in streams:
        for row in rows:
            event = str(row["event"])
            for field, value in row.items():
                if field in TIME_FIELDS:
                    time_values[field].add(int(value))
                    continue
                values.setdefault(field, set()).add(key(value))
                applicability.setdefault(event, {"event"}).add(field)
                event_constraints[event][field].add(key(value))
                if event == "NOTE" and isinstance(row.get("kind"), str):
                    kind_constraints[str(row["kind"])][field].add(key(value))
    return {
        "schema": VOCABULARY_SCHEMA,
        "row_codec": ROW_CODEC,
        "fields": {
            field: ["<NA>", "<OOV>", *sorted(field_values)]
            for field, field_values in sorted(
                values.items(), key=lambda item: (item[0] != "event", item[0])
            )
        },
        "applicability": {
            event: sorted(field_names)
            for event, field_names in sorted(applicability.items())
        },
        "event_value_constraints": {
            event: {
                field: sorted(field_values)
                for field, field_values in sorted(by_field.items())
            }
            for event, by_field in sorted(event_constraints.items())
        },
        "note_kind_value_constraints": {
            kind: {
                field: sorted(field_values)
                for field, field_values in sorted(by_field.items())
            }
            for kind, by_field in sorted(kind_constraints.items())
        },
        "time_values": {
            field: sorted(field_values) for field, field_values in time_values.items()
        },
    }


def build_profile(*, dataset: Path, output: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = output.resolve()
    if output == dataset or dataset not in output.parents:
        raise DatasetError("v2-test1 profile output must be a descendant of the base dataset")
    source_root = dataset / "profiles" / ALL_DIFFICULTIES_PROFILE_NAME
    if not (source_root / "summary.json").is_file():
        raise DatasetError(f"compact-v2 all-difficulties source profile is missing: {source_root}")
    source_summary = json.loads(
        (source_root / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("schema") != "maimai.dataset_profile.compact_v2":
        raise DatasetError("v2-test1 source profile has an unsupported schema")

    output.mkdir(parents=True, exist_ok=True)
    (output / ".maimai_dataset_profile").write_text(
        PROFILE_SCHEMA + "\n", encoding="utf-8"
    )
    source_records = load_jsonl(source_root / "manifest.jsonl")
    profile_manifest: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    training_streams: list[list[dict[str, Any]]] = []
    row_counts: Counter[str] = Counter()
    roundtrips = 0

    for index, source_record in enumerate(source_records, start=1):
        record = {
            **source_record,
            "schema": PROFILE_RECORD_SCHEMA,
            "profile": PROFILE_NAME,
        }
        if source_record.get("status") != "eligible":
            record.pop("encoded_path", None)
            profile_manifest.append(record)
            continue
        source_encoded_path = source_root / str(source_record["encoded_path"])
        source_encoded = json.loads(source_encoded_path.read_text(encoding="utf-8"))
        compressed, chunks = compress_compact_stream(
            source_encoded["rows"], source_encoded["moment_chunks"]
        )
        recompressed: list[dict[str, Any]] = []
        for chunk in chunks:
            compact_chunk = compressed[
                int(chunk["row_start"]) : int(chunk["row_end"])
            ]
            expanded_chunk = expand_compact_stream(compact_chunk)
            compressed_chunk, _ = compress_compact_stream(
                expanded_chunk,
                [
                    {
                        "beat_num": int(chunk["beat_num"]),
                        "beat_den": int(chunk["beat_den"]),
                        "row_start": 0,
                        "row_end": len(expanded_chunk),
                    }
                ],
            )
            recompressed.extend(compressed_chunk)
        if recompressed != compressed:
            raise DatasetError(
                f"v2-test1 row codec is not idempotent for {source_record['record_id']}"
            )
        roundtrips += 1
        source_difficulty = int(source_record["source_difficulty_id"])
        relative_encoded = (
            Path("encoded") / str(source_record["song_id"]) / f"{source_difficulty}.json"
        )
        _json(
            output / relative_encoded,
            {
                "schema": TRAINING_ROWS_SCHEMA,
                "row_codec": ROW_CODEC,
                "record_id": source_record["record_id"],
                "rows": compressed,
                "moment_chunks": chunks,
                "source_compact_v2_sha256": _sha256(source_encoded_path),
                "source_ir_sha256": source_record["ir_sha256"],
            },
        )
        document = json.loads(
            (dataset / str(source_record["ir_path"])).read_text(encoding="utf-8")
        )
        chart_segments = _segment_rows(
            record=record,
            rows=compressed,
            chunks=chunks,
            document=document,
            encoded_path=relative_encoded.as_posix(),
        )
        for segment in chart_segments:
            segment["schema"] = SEGMENT_SCHEMA
        segments.extend(chart_segments)
        if source_record["split"] == "train":
            training_streams.append(compressed)
        record["encoded_path"] = relative_encoded.as_posix()
        profile_manifest.append(record)
        row_counts[str(source_record["split"])] += len(compressed)
        if index % 250 == 0:
            print(f"{PROFILE_NAME} {index}/{len(source_records)}", flush=True)

    vocabulary = _build_vocabulary(training_streams)
    vocabulary["base_dataset_fingerprint"] = source_summary[
        "base_dataset_fingerprint"
    ]
    write_jsonl(output / "manifest.jsonl", profile_manifest)
    write_jsonl(output / "segments.jsonl", segments)
    _json(output / "vocabulary.json", vocabulary)

    segment_counts = Counter(str(row["split"]) for row in segments)
    target_row_counts: Counter[str] = Counter()
    source_difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for segment in segments:
        split = str(segment["split"])
        target_row_counts[split] += int(segment["target_rows"])
        source_difficulty_counts[split][str(segment["source_difficulty_id"])] += 1
    fingerprint = canonical_json_sha256(
        {
            "profile": PROFILE_NAME,
            "manifest": profile_manifest,
            "vocabulary": vocabulary,
            "segments": segments,
        }
    )
    summary = {
        "schema": PROFILE_SCHEMA,
        "row_codec": ROW_CODEC,
        "each_encoding": "derived_from_note_count",
        "profile": PROFILE_NAME,
        "source_profile": ALL_DIFFICULTIES_PROFILE_NAME,
        "source_profile_fingerprint": source_summary["profile_fingerprint"],
        "base_dataset": "../..",
        "base_dataset_fingerprint": vocabulary["base_dataset_fingerprint"],
        "profile_fingerprint": fingerprint,
        "chart_ticks_per_whole": CHART_TICKS_PER_WHOLE,
        "chart_ticks_per_quarter_beat": CHART_TICKS_PER_BEAT,
        "target_beats": TARGET_BEATS,
        "prefix_beats": PREFIX_BEATS,
        "audio_context_beats": AUDIO_CONTEXT_BEATS,
        "input_charts": source_summary["input_charts"],
        "eligible_charts": source_summary["eligible_charts"],
        "isolated_charts": source_summary["isolated_charts"],
        "eligible_split_counts": source_summary["eligible_split_counts"],
        "eligible_source_difficulty_counts": source_summary[
            "eligible_source_difficulty_counts"
        ],
        "isolation_reason_counts": source_summary["isolation_reason_counts"],
        "segment_split_counts": dict(sorted(segment_counts.items())),
        "target_row_split_counts": dict(sorted(target_row_counts.items())),
        "segment_source_difficulty_counts": {
            split: dict(sorted(values.items()))
            for split, values in sorted(source_difficulty_counts.items())
        },
        "training_rows": row_counts["train"],
        "row_roundtrip_charts": roundtrips,
        "test_used_for_vocabulary": False,
        "test_used_for_feature_stats": False,
    }
    expected_variants = (
        {
            "input_charts": 4754,
            "eligible_charts": 4625,
            "isolated_charts": 129,
            "train_segments": 158465,
            "train_target_rows": 2649360,
            "train_rows": 2490895,
            "roundtrips": 4625,
        },
        {
            "input_charts": 4754,
            "eligible_charts": 4625,
            "isolated_charts": 129,
            "train_segments": 158465,
            "train_target_rows": 2649355,
            "train_rows": 2490890,
            "roundtrips": 4625,
        },
    )
    actual = {
        "input_charts": summary["input_charts"],
        "eligible_charts": summary["eligible_charts"],
        "isolated_charts": summary["isolated_charts"],
        "train_segments": segment_counts["train"],
        "train_target_rows": target_row_counts["train"],
        "train_rows": row_counts["train"],
        "roundtrips": roundtrips,
    }
    if actual not in expected_variants:
        raise DatasetError(
            "v2-test1 profile acceptance counts changed: "
            f"expected one of {expected_variants}, actual={actual}"
        )
    _json(output / "summary.json", summary)
    _json(
        output / "profile.json",
        {
            "schema": PROFILE_SCHEMA,
            "row_codec": ROW_CODEC,
            "profile": PROFILE_NAME,
            "base_dataset": "../..",
            "source_profile": f"../{ALL_DIFFICULTIES_PROFILE_NAME}",
            "base_feature_stats": "../../feature-stats.json",
            "base_manifest": "../../manifest.jsonl",
            "summary_sha256": _sha256(output / "summary.json"),
            "vocabulary_sha256": _sha256(output / "vocabulary.json"),
        },
    )
    (output / "report.md").write_text(
        "\n".join(
            [
                "# v2-test1 all-difficulties profile",
                "",
                f"- Profile fingerprint: `{fingerprint}`",
                f"- Eligible charts: {summary['eligible_charts']}",
                f"- Train windows: {segment_counts['train']}",
                f"- Train target rows: {target_row_counts['train']}",
                "- Removed target rows versus compact-v2: 57.43%.",
                "- Vocabulary source: train only.",
                "- Each is absent from targets and deterministically derived from NOTE count.",
                f"- Row codec: `{ROW_CODEC}`.",
                "- Test used for vocabulary/statistics: no.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def run_semantic_audit(
    *, dataset: Path, profile_root: Path, output: Path
) -> dict[str, Any]:
    from maimai_dataset.materialize import (
        materialize_chart_ir,
        normalize_each_semantics,
    )

    dataset = dataset.resolve()
    profile_root = profile_root.resolve()
    summary = json.loads((profile_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != PROFILE_SCHEMA:
        raise DatasetError("semantic audit requires a v2-test1 profile")
    records = [
        row
        for row in load_jsonl(profile_root / "manifest.jsonl")
        if row.get("status") == "eligible"
    ]
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        try:
            encoded = json.loads(
                (profile_root / str(record["encoded_path"])).read_text(encoding="utf-8")
            )
            v1_rows = restore_v1_rows(encoded["rows"])
            document = json.loads(
                (dataset / str(record["ir_path"])).read_text(encoding="utf-8")
            )
            rematerialized = materialize_chart_ir(
                v1_rows,
                timing=TimingMap.from_document(document),
                difficulty=str(document.get("difficulty") or "MAS"),
            )
            if chart_tick_semantic_projection(
                normalize_each_semantics(rematerialized)
            ) != chart_tick_semantic_projection(normalize_each_semantics(document)):
                raise DatasetError("chart-tick semantic projection changed")
        except Exception as error:
            failures.append(
                {"record_id": str(record["record_id"]), "error": str(error)}
            )
        if index % 250 == 0 or index == len(records):
            print(f"v2-test1 semantic audit {index}/{len(records)}", flush=True)
    receipt = {
        "schema": SEMANTIC_AUDIT_SCHEMA,
        "row_codec": ROW_CODEC,
        "profile_fingerprint": summary["profile_fingerprint"],
        "eligible_charts": len(records),
        "passed_charts": len(records) - len(failures),
        "failed_charts": len(failures),
        "physical_180_tps_projection_ignored": True,
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "semantic-audit.json", receipt)
    if failures:
        raise DatasetError(
            f"v2-test1 semantic audit failed for {len(failures)} charts"
        )
    return receipt


__all__ = [
    "EVENTS",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "ROW_CODEC",
    "VOCABULARY_SCHEMA",
    "build_profile",
    "compress_compact_stream",
    "expand_compact_stream",
    "restore_v1_rows",
    "run_semantic_audit",
]
