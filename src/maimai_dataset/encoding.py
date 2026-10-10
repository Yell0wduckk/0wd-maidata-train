from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from maimai_chart_ir.models import model_projection, validate_chart_ir

from .audio import HOP_LENGTH, TARGET_SAMPLE_RATE
from .manifest import DatasetError, load_jsonl, write_jsonl
from .timing import TimingMap


PROFILE_NAME = "compact-v2"
ALL_DIFFICULTIES_PROFILE_NAME = "compact-v2-all-difficulties"
SUPPORTED_PROFILES = frozenset({PROFILE_NAME, ALL_DIFFICULTIES_PROFILE_NAME})
PROFILE_SCHEMA = "maimai.dataset_profile.compact_v2"
AUDIT_SCHEMA = "maimai.timing_audit.compact_v2"
TRAINING_ROWS_SCHEMA = "maimai.compact_training_rows.v2"
SEGMENT_SCHEMA = "maimai.compact_segment.v2"
from .window import CHART_TICKS_PER_BEAT, TARGET_BEATS, HISTORY_BEATS, AUDIO_BEATS_BEFORE, AUDIO_BEATS_AFTER, AUDIO_CONTEXT_BEATS, training_window_count
CHART_TICKS_PER_WHOLE = 4 * CHART_TICKS_PER_BEAT
PREFIX_BEATS = HISTORY_BEATS

PRE_FESTIVAL_VERSIONS = frozenset(
    {
        "maimai",
        "maimai PLUS",
        "maimai GreeN",
        "maimai GreeN PLUS",
        "maimai ORANGE",
        "maimai ORANGE PLUS",
        "maimai PiNK",
        "maimai PiNK PLUS",
        "maimai MURASAKi",
        "maimai MURASAKi PLUS",
        "maimai MiLK",
        "maimai MiLK PLUS",
        "maimai FiNALE",
        "maimai DX",
        "maimai DX PLUS",
        "maimai DX Splash",
        "maimai DX Splash PLUS",
        "maimai DX UNiVERSE",
        "maimai DX UNiVERSE PLUS",
    }
)

TIME_FIELDS = ("delta_tick", "hold_duration_tick", "branch_duration_tick")
_SPECIAL_ID = re.compile(r"^\d{6}$")
_PATH = re.compile(r"^([1-8])(pp|qq|[-^<>vpqswz])([1-8])$")
_V_PATH = re.compile(r"^([1-8])V([1-8])([1-8])$")


def profile_source_difficulties(profile: str) -> tuple[int, ...]:
    if profile == PROFILE_NAME:
        return (5,)
    if profile == ALL_DIFFICULTIES_PROFILE_NAME:
        return (2, 3, 4, 5, 6)
    raise DatasetError(f"unsupported dataset profile: {profile!r}")


def condition_difficulty_id(source_difficulty_id: int) -> int:
    source = int(source_difficulty_id)
    if source not in {2, 3, 4, 5, 6}:
        raise DatasetError(f"unsupported source difficulty: {source}")
    return 5 if source == 6 else source


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
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def chart_tick_semantic_projection(document: Mapping[str, Any]) -> Any:
    """Model semantics with all 180 TPS/seconds projections removed."""
    ir_time_projection_keys = {
        "tick",
        "start_tick",
        "end_tick",
        "body_start_tick",
        "resolved_ticks",
        "second",
        "seconds",
    }

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip(child)
                for key, child in value.items()
                if key not in ir_time_projection_keys
            }
        if isinstance(value, list):
            return [strip(child) for child in value]
        return value

    return strip(model_projection(document))


def _fraction(value: Mapping[str, Any], *, prefix: str = "beat") -> Fraction:
    return Fraction(int(value[f"{prefix}_num"]), int(value[f"{prefix}_den"]))


def beat_fraction_to_chart_tick(value: Fraction, *, field: str) -> int:
    tick = value * CHART_TICKS_PER_BEAT
    if tick.denominator != 1:
        raise DatasetError(f"{field} is not an integer chart tick: {value} beat")
    return int(tick)


def beat_value_to_chart_tick(value: Mapping[str, Any], *, field: str) -> int:
    if value.get("basis") != "beat":
        raise DatasetError(f"{field} uses absolute {value.get('basis')!r} timing")
    return beat_fraction_to_chart_tick(_fraction(value), field=field)


def chart_tick_to_compound_time(tick: int) -> str:
    value = Fraction(int(tick), CHART_TICKS_PER_BEAT)
    return f"beat:{value.numerator}/{value.denominator}"


def factorize_path(path: str, *, start_pad: str | None = None) -> dict[str, Any]:
    match_v = _V_PATH.fullmatch(str(path))
    if match_v:
        start, via, end = (int(match_v.group(index)) for index in (1, 2, 3))
        shape = "V"
        result: dict[str, Any] = {
            "shape": shape,
            "end_offset": (end - start) % 8,
            "via_offset": (via - start) % 8,
        }
    else:
        match = _PATH.fullmatch(str(path))
        if not match:
            raise DatasetError(f"unsupported v2 Slide path: {path!r}")
        start, shape, end = int(match.group(1)), match.group(2), int(match.group(3))
        result = {"shape": shape, "end_offset": (end - start) % 8}
    if start_pad is not None:
        expected = f"A{start}"
        if start_pad != expected:
            raise DatasetError(
                f"Slide path {path!r} starts at {expected}, not row start_pad {start_pad!r}"
            )
    return result


def restore_path(*, start_pad: str, shape: str, end_offset: int, via_offset: int | None = None) -> str:
    if not re.fullmatch(r"A[1-8]", start_pad):
        raise DatasetError(f"invalid Slide start pad: {start_pad!r}")
    start = int(start_pad[1])
    end = ((start - 1 + int(end_offset)) % 8) + 1
    if shape == "V":
        if via_offset is None:
            raise DatasetError("V Slide is missing via_offset")
        via = ((start - 1 + int(via_offset)) % 8) + 1
        return f"{start}V{via}{end}"
    if via_offset is not None:
        raise DatasetError(f"non-V Slide {shape!r} cannot carry via_offset")
    candidate = f"{start}{shape}{end}"
    if not _PATH.fullmatch(candidate):
        raise DatasetError(f"invalid restored Slide path: {candidate!r}")
    return candidate


def audit_document(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable whole-chart v2 isolation codes.

    This intentionally reads beat-valued semantic fields and never consumes the
    IR's 180 TPS physical ticks.
    """
    validate_chart_ir(document)
    reasons: set[str] = set()
    for moment in document["moments"]:
        try:
            beat_fraction_to_chart_tick(
                Fraction(
                    int(moment["position_beats"]["num"]),
                    int(moment["position_beats"]["den"]),
                ),
                field="moment.position_beats",
            )
        except DatasetError:
            reasons.add("timing:non_integer_chart_tick")
        for group in moment["groups"]:
            for note in group["notes"]:
                kind = str(note["kind"])
                if kind in {"hold", "touch_hold"}:
                    duration = note["duration"]
                    if duration.get("basis") != "beat":
                        reasons.add("timing:absolute_hold_duration")
                    else:
                        try:
                            beat_value_to_chart_tick(duration, field="hold.duration")
                        except DatasetError:
                            reasons.add("timing:non_integer_chart_tick")
                if kind not in {"slide", "wifi"}:
                    continue
                wait = note["wait"]
                if wait.get("basis") != "beat":
                    reasons.add("timing:absolute_slide_wait")
                else:
                    try:
                        wait_tick = beat_value_to_chart_tick(wait, field="slide.wait")
                        if wait_tick != CHART_TICKS_PER_BEAT:
                            reasons.add("slide:non_default_wait")
                    except DatasetError:
                        reasons.add("timing:non_integer_chart_tick")
                for branch in note["branches"]:
                    duration = branch["duration"]
                    if duration.get("basis") != "beat":
                        reasons.add("timing:absolute_slide_duration")
                    else:
                        try:
                            tick = beat_value_to_chart_tick(
                                duration, field="slide.branch.duration"
                            )
                            if tick < 1:
                                reasons.add("slide:non_positive_duration")
                        except DatasetError:
                            reasons.add("timing:non_integer_chart_tick")
                    if len(branch["segments"]) != 1:
                        reasons.add("slide:multi_segment")
                    for segment in branch["segments"]:
                        try:
                            factorize_path(
                                str(segment["path"]), start_pad=str(note["start_pad"])
                            )
                        except DatasetError:
                            reasons.add("slide:unsupported_path")
    return tuple(sorted(reasons))


def restore_compound_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Invert the v2 field factorization to a semantically equivalent v1 row."""
    from maimai_dataset.modifiers import materialize_row
    row = materialize_row(row)
    event = str(row["event"])
    if event == "TARGET_START":
        raise DatasetError("TARGET_START is a segment-only boundary")
    result = dict(row)
    if event == "MOMENT":
        result["delta"] = chart_tick_to_compound_time(int(result.pop("delta_tick")))
    if event == "NOTE" and result.get("kind") in {"hold", "touch_hold"}:
        result["duration"] = chart_tick_to_compound_time(
            int(result.pop("hold_duration_tick"))
        )
    is_slide = (
        event == "SLIDE_BRANCH"
        or (event == "NOTE" and result.get("kind") in {"slide", "wifi"})
    )
    if is_slide:
        duration = chart_tick_to_compound_time(int(result.pop("branch_duration_tick")))
        shape = str(result.pop("shape"))
        end_offset = int(result.pop("end_offset"))
        via_offset_raw = result.pop("via_offset", None)
        path = restore_path(
            start_pad=str(result["start_pad"]),
            shape=shape,
            end_offset=end_offset,
            via_offset=None if via_offset_raw is None else int(via_offset_raw),
        )
        end = int(path[-1])
        result.update(
            {
                "branch_duration": duration,
                "branch_timing": "total",
                "path": path,
                "end_pad": f"A{end}",
                "segment_duration": duration,
            }
        )
        if event == "NOTE":
            result["wait"] = "beat:1/1"
        else:
            result.pop("start_pad", None)
    return {key: result[key] for key in sorted(result, key=lambda key: (key != "event", key))}


def _load_input_records(dataset: Path, *, profile: str) -> list[dict[str, Any]]:
    source_difficulties = set(profile_source_difficulties(profile))
    records = [
        row
        for row in load_jsonl(dataset / "manifest.jsonl")
        if row.get("difficulty_id") in source_difficulties
        and row.get("status") == "eligible"
        and row.get("version") in PRE_FESTIVAL_VERSIONS
    ]
    records.sort(
        key=lambda row: (
            len(str(row["song_id"])),
            str(row["song_id"]),
            int(row["difficulty_id"]),
        )
    )
    leaks = [row["song_id"] for row in records if _SPECIAL_ID.fullmatch(str(row["song_id"]))]
    if leaks:
        raise DatasetError(f"six-digit IDs leaked into compact-v2: {leaks[:5]}")
    return records


def _audit_rows(
    dataset: Path, *, profile: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _load_input_records(dataset, profile=profile)
    audited: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    id_version_ranges: dict[str, list[int]] = defaultdict(list)
    for record in records:
        ir_path = dataset / str(record["ir_path"])
        document = json.loads(ir_path.read_text(encoding="utf-8"))
        reasons = audit_document(document)
        status = "isolated" if reasons else "eligible"
        reason_counts.update(reasons)
        if status == "eligible":
            split_counts[str(record["split"])] += 1
        song_id = str(record["song_id"])
        if song_id.isdigit():
            id_version_ranges[str(record["version"])].append(int(song_id))
        audited.append(
            {
                "schema": "maimai.timing_audit_record.compact_v2",
                "record_id": record["record_id"],
                "song_id": song_id,
                "source_difficulty_id": int(record["difficulty_id"]),
                "condition_difficulty_id": condition_difficulty_id(
                    int(record["difficulty_id"])
                ),
                "split": record["split"],
                "version": record["version"],
                "status": status,
                "isolation_reasons": list(reasons),
                "ir_path": record["ir_path"],
                "ir_sha256": record["ir_sha256"],
            }
        )
    summary = {
        "schema": AUDIT_SCHEMA,
        "profile": profile,
        "chart_ticks_per_whole": CHART_TICKS_PER_WHOLE,
        "chart_ticks_per_quarter_beat": CHART_TICKS_PER_BEAT,
        "input_charts": len(records),
        "eligible_charts": sum(row["status"] == "eligible" for row in audited),
        "isolated_charts": sum(row["status"] == "isolated" for row in audited),
        "eligible_split_counts": dict(sorted(split_counts.items())),
        "isolation_reason_counts": dict(sorted(reason_counts.items())),
        "six_digit_id_leaks": 0,
        "eligible_source_difficulty_counts": dict(
            sorted(
                Counter(
                    str(row["source_difficulty_id"])
                    for row in audited
                    if row["status"] == "eligible"
                ).items()
            )
        ),
        "regular_id_version_ranges": {
            version: {"min": min(values), "max": max(values), "count": len(values)}
            for version, values in sorted(id_version_ranges.items())
            if values
        },
    }
    expected = (
        {
            "input_charts": 1153,
            "eligible_charts": 1108,
            "isolated_charts": 45,
            "eligible_split_counts": {"test": 94, "train": 886, "validation": 128},
            "eligible_source_difficulty_counts": {"5": 1108},
        }
        if profile == PROFILE_NAME
        else {
            "input_charts": 4754,
            "eligible_charts": 4625,
            "isolated_charts": 129,
            "eligible_split_counts": {"test": 405, "train": 3685, "validation": 535},
            "eligible_source_difficulty_counts": {
                "2": 1143,
                "3": 1131,
                "4": 1121,
                "5": 1108,
                "6": 122,
            },
        }
    )
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise DatasetError(f"{profile} acceptance counts changed: {mismatches}")
    return audited, summary


def run_timing_audit(*, dataset: Path, profile: str, output: Path) -> dict[str, Any]:
    profile_source_difficulties(profile)
    dataset = dataset.resolve()
    if not (dataset / "manifest.jsonl").is_file():
        raise DatasetError(f"dataset manifest is missing: {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    audited, summary = _audit_rows(dataset, profile=profile)
    write_jsonl(output / "records.jsonl", audited)
    _json(output / "audit.json", summary)
    lines = [
        f"# {profile} timing audit",
        "",
        f"- Input pre-FESTiVAL eligible MASTER charts: {summary['input_charts']}",
        f"- Eligible: {summary['eligible_charts']}",
        f"- Isolated: {summary['isolated_charts']}",
        f"- Split: `{summary['eligible_split_counts']}`",
        f"- Reasons: `{summary['isolation_reason_counts']}`",
        f"- Six-digit ID leaks: {summary['six_digit_id_leaks']}",
        "- Time base: 384 chart tick per whole note; 96 per quarter beat.",
        "- 180 TPS physical ticks were not used for model timing.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _chart_end_tick(document: Mapping[str, Any]) -> int:
    from maimai_dataset.event_time import duration_end, seconds, written_tick_seconds
    timing = TimingMap.from_document(document)
    end = 0
    for moment in document["moments"]:
        start = beat_fraction_to_chart_tick(
            Fraction(
                int(moment["position_beats"]["num"]),
                int(moment["position_beats"]["den"]),
            ),
            field="moment.position_beats",
        )
        end = max(end, start)
        for group in moment["groups"]:
            for note in group["notes"]:
                if note["kind"] in {"hold", "touch_hold"}:
                    end = max(
                        end,
                        math.ceil(timing.second_to_beat(duration_end(timing, start,
                            beat_value_to_chart_tick(note["duration"], field="hold.duration"))) * 96 - 1e-9),
                    )
                elif note["kind"] in {"slide", "wifi"}:
                    wait = beat_value_to_chart_tick(note["wait"], field="slide.wait")
                    for branch in note["branches"]:
                        end = max(
                            end,
                            math.ceil(timing.second_to_beat(seconds(timing, start + wait)
                                + beat_value_to_chart_tick(branch["duration"], field="slide.branch.duration")
                                * written_tick_seconds(timing, start)) * 96 - 1e-9),
                        )
    return end


def _segment_rows(
    *,
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any],
    encoded_path: str,
) -> list[dict[str, Any]]:
    source_difficulty = int(record["difficulty_id"])
    condition_difficulty = condition_difficulty_id(source_difficulty)
    timing = TimingMap.from_document(document)
    chunk_values = [
        {
            **dict(chunk),
            "beat": Fraction(int(chunk["beat_num"]), int(chunk["beat_den"])),
        }
        for chunk in chunks
    ]
    last_beat = max(chunk["beat"] for chunk in chunk_values)
    song_end_tick = _chart_end_tick(document)
    window_count = training_window_count(
        beat_fraction_to_chart_tick(last_beat, field="moment.position_beats"), song_end_tick)
    segments: list[dict[str, Any]] = []
    for window_index in range(window_count):
        start = Fraction(window_index * TARGET_BEATS)
        end = start + TARGET_BEATS
        context_start = max(Fraction(0), start - PREFIX_BEATS)
        context = [chunk for chunk in chunk_values if context_start <= chunk["beat"] < start]
        target = [chunk for chunk in chunk_values if start <= chunk["beat"] < end]
        context_row_start = int(context[0]["row_start"]) if context else (
            int(target[0]["row_start"]) if target else len(rows)
        )
        target_row_start = int(target[0]["row_start"]) if target else (
            next(
                (
                    int(chunk["row_start"])
                    for chunk in chunk_values
                    if chunk["beat"] >= end
                ),
                len(rows),
            )
        )
        target_row_end = int(target[-1]["row_end"]) if target else target_row_start
        audio_start_second = timing.beat_to_second(start - AUDIO_BEATS_BEFORE)
        audio_end_second = timing.beat_to_second(end + AUDIO_BEATS_AFTER)
        frame_start = math.floor(audio_start_second * TARGET_SAMPLE_RATE / HOP_LENGTH)
        frame_end = math.ceil(audio_end_second * TARGET_SAMPLE_RATE / HOP_LENGTH)
        segments.append(
            {
                "schema": SEGMENT_SCHEMA,
                "segment_id": f"{record['record_id']}:{window_index:04d}",
                "record_id": record["record_id"],
                "song_id": record["song_id"],
                "split": record["split"],
                # ``difficulty_id`` remains the model-conditioning value for
                # compatibility with existing compact-v2 loaders.
                "difficulty_id": condition_difficulty,
                "source_difficulty_id": source_difficulty,
                "condition_difficulty_id": condition_difficulty,
                "level_value": record.get("level_value"),
                "cabinet": record["cabinet"],
                "encoded_path": encoded_path,
                "feature_path": record["feature_path"],
                "target_start_tick": beat_fraction_to_chart_tick(start, field="target.start"),
                "target_end_tick": beat_fraction_to_chart_tick(end, field="target.end"),
                "context_start_tick": beat_fraction_to_chart_tick(
                    context_start, field="context.start"
                ),
                "song_end_tick": song_end_tick,
                "context_row_start": context_row_start,
                "target_row_start": target_row_start,
                "target_row_end": target_row_end,
                "context_first_tick": (
                    beat_fraction_to_chart_tick(context[0]["beat"], field="context.first")
                    if context
                    else None
                ),
                "target_first_tick": (
                    beat_fraction_to_chart_tick(target[0]["beat"], field="target.first")
                    if target
                    else None
                ),
                "target_rows": target_row_end - target_row_start + 1,
                "audio_frame_start": frame_start,
                "audio_frame_end": frame_end,
                "timing": timing.to_dict(end_beat=Fraction(song_end_tick, CHART_TICKS_PER_BEAT)),
            }
        )
    return segments


__all__ = [
    "ALL_DIFFICULTIES_PROFILE_NAME",
    "AUDIO_CONTEXT_BEATS",
    "CHART_TICKS_PER_BEAT",
    "CHART_TICKS_PER_WHOLE",
    "PREFIX_BEATS",
    "TARGET_BEATS",
    "TIME_FIELDS",
    "chart_tick_semantic_projection",
    "condition_difficulty_id",
    "factorize_path",
    "restore_compound_row",
    "restore_path",
    "run_timing_audit",
]
