from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import re
from typing import Any, Iterable, Mapping


SCHEMA = "maimai.chart_ir.v1"
NOTE_KINDS = {"tap", "hold", "touch", "touch_hold", "slide", "wifi"}
CONTROL_KINDS = {"bpm", "meter", "hs"}
HEAD_KINDS = {"star", "tap", "none_fade", "none_immediate"}
_PATH_RE = re.compile(r"[1-8](?:[-^v<>szw][1-8]|V[1-8][1-8]|p{1,2}[1-8]|q{1,2}[1-8])")


class ChartIRError(ValueError):
    """Raised when chart IR or a codec stream violates its grammar."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChartIRError(message)


def _integer(value: Any, path: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{path} must be an integer")
    return value


def _number(value: Any, path: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be numeric")
    return float(value)


def rational(value: Any, path: str) -> Fraction:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    num = _integer(value.get("num"), f"{path}.num")
    den = _integer(value.get("den"), f"{path}.den")
    _require(den > 0, f"{path}.den must be positive")
    return Fraction(num, den)


def validate_duration(value: Any, path: str, *, positive: bool = False) -> None:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    basis = value.get("basis")
    _require(basis in {"beat", "tick"}, f"{path}.basis is invalid")
    ticks = _integer(value.get("resolved_ticks"), f"{path}.resolved_ticks")
    _require(ticks >= (1 if positive else 0), f"{path}.resolved_ticks is too small")
    if "seconds" in value:
        seconds = _number(value["seconds"], f"{path}.seconds")
        _require(seconds >= 0, f"{path}.seconds must be non-negative")
    if basis == "beat":
        numerator = _integer(value.get("beat_num"), f"{path}.beat_num")
        denominator = _integer(value.get("beat_den"), f"{path}.beat_den")
        _require(denominator > 0, f"{path}.beat_den must be positive")
        _require(numerator >= (1 if positive else 0), f"{path}.beat_num is too small")


def _validate_note(note: Any, path: str) -> None:
    _require(isinstance(note, Mapping), f"{path} must be an object")
    kind = note.get("kind")
    _require(kind in NOTE_KINDS, f"{path}.kind is invalid")
    start_tick = _integer(note.get("start_tick"), f"{path}.start_tick")
    flags = note.get("flags", [])
    _require(isinstance(flags, list) and all(isinstance(item, str) for item in flags), f"{path}.flags is invalid")

    if kind in {"tap", "touch"}:
        _require(isinstance(note.get("pad"), str) and note["pad"], f"{path}.pad is required")
        return
    if kind in {"hold", "touch_hold"}:
        _require(isinstance(note.get("pad"), str) and note["pad"], f"{path}.pad is required")
        validate_duration(note.get("duration"), f"{path}.duration")
        end_tick = _integer(note.get("end_tick"), f"{path}.end_tick")
        _require(end_tick >= start_tick, f"{path}.end_tick precedes start_tick")
        return

    _require(isinstance(note.get("start_pad"), str) and note["start_pad"], f"{path}.start_pad is required")
    body_tick = _integer(note.get("body_start_tick"), f"{path}.body_start_tick")
    end_tick = _integer(note.get("end_tick"), f"{path}.end_tick")
    _require(body_tick >= start_tick and end_tick >= body_tick, f"{path} slide ticks are out of order")
    validate_duration(note.get("wait"), f"{path}.wait")
    head = note.get("head")
    _require(isinstance(head, Mapping) and head.get("kind") in HEAD_KINDS, f"{path}.head is invalid")
    head_flags = head.get("flags", [])
    _require(
        isinstance(head_flags, list) and all(isinstance(item, str) for item in head_flags),
        f"{path}.head.flags is invalid",
    )
    branches = note.get("branches")
    _require(isinstance(branches, list) and branches, f"{path}.branches must not be empty")
    for branch_index, branch in enumerate(branches):
        branch_path = f"{path}.branches[{branch_index}]"
        _require(isinstance(branch, Mapping), f"{branch_path} must be an object")
        _require(branch.get("timing_mode") in {"total", "per_segment"}, f"{branch_path}.timing_mode is invalid")
        branch_flags = branch.get("flags", [])
        _require(
            isinstance(branch_flags, list) and all(isinstance(item, str) for item in branch_flags),
            f"{branch_path}.flags is invalid",
        )
        validate_duration(branch.get("duration"), f"{branch_path}.duration", positive=True)
        segments = branch.get("segments")
        _require(isinstance(segments, list) and segments, f"{branch_path}.segments must not be empty")
        previous_end = note["start_pad"]
        for segment_index, segment in enumerate(segments):
            segment_path = f"{branch_path}.segments[{segment_index}]"
            _require(isinstance(segment, Mapping), f"{segment_path} must be an object")
            path_key = segment.get("path")
            _require(isinstance(path_key, str) and _PATH_RE.fullmatch(path_key) is not None,
                     f"{segment_path}.path is invalid: {path_key!r}")
            _require(path_key[0] == previous_end[-1], f"{segment_path}.path is not continuous")
            end_pad = segment.get("end_pad")
            _require(isinstance(end_pad, str) and end_pad.endswith(path_key[-1]), f"{segment_path}.end_pad disagrees with path")
            validate_duration(segment.get("duration"), f"{segment_path}.duration")
            previous_end = end_pad


def validate_chart_ir(document: Any) -> dict[str, Any]:
    _require(isinstance(document, Mapping), "chart IR must be an object")
    _require(document.get("schema") == SCHEMA, f"unsupported schema: {document.get('schema')!r}")
    _require(document.get("tick_rate") == 180, "tick_rate must be 180")
    _number(document.get("first_seconds"), "first_seconds")
    default_meter = document.get("default_meter")
    _require(isinstance(default_meter, Mapping), "default_meter must be an object")
    _require(_integer(default_meter.get("numerator"), "default_meter.numerator") > 0,
             "default_meter.numerator must be positive")
    _require(_integer(default_meter.get("denominator"), "default_meter.denominator") > 0,
             "default_meter.denominator must be positive")

    controls = document.get("controls")
    _require(isinstance(controls, list), "controls must be an array")
    for index, control in enumerate(controls):
        path = f"controls[{index}]"
        _require(isinstance(control, Mapping), f"{path} must be an object")
        _require(control.get("kind") in CONTROL_KINDS, f"{path}.kind is invalid")
        rational(control.get("position_beats"), f"{path}.position_beats")
        _integer(control.get("tick"), f"{path}.tick")
        if control["kind"] == "bpm":
            _require(_number(control.get("bpm"), f"{path}.bpm") > 0, f"{path}.bpm must be positive")
        elif control["kind"] == "meter":
            _require(_integer(control.get("numerator"), f"{path}.numerator") > 0,
                     f"{path}.numerator must be positive")
            _require(_integer(control.get("denominator"), f"{path}.denominator") > 0,
                     f"{path}.denominator must be positive")
        else:
            _number(control.get("multiplier"), f"{path}.multiplier")

    moments = document.get("moments")
    _require(isinstance(moments, list), "moments must be an array")
    previous_position: Fraction | None = None
    for moment_index, moment in enumerate(moments):
        path = f"moments[{moment_index}]"
        _require(isinstance(moment, Mapping), f"{path} must be an object")
        position = rational(moment.get("position_beats"), f"{path}.position_beats")
        _require(previous_position is None or position > previous_position, "moment positions must be strictly increasing")
        previous_position = position
        _integer(moment.get("tick"), f"{path}.tick")
        _require(_number(moment.get("bpm"), f"{path}.bpm") > 0, f"{path}.bpm must be positive")
        _number(moment.get("hs", 1.0), f"{path}.hs")
        meter = moment.get("meter")
        _require(isinstance(meter, Mapping), f"{path}.meter must be an object")
        _require(_integer(meter.get("numerator"), f"{path}.meter.numerator") > 0, "meter numerator must be positive")
        _require(_integer(meter.get("denominator"), f"{path}.meter.denominator") > 0, "meter denominator must be positive")
        groups = moment.get("groups")
        _require(isinstance(groups, list) and groups, f"{path}.groups must not be empty")
        for group_index, group in enumerate(groups):
            group_path = f"{path}.groups[{group_index}]"
            _require(isinstance(group, Mapping) and isinstance(group.get("each"), bool), f"{group_path}.each is invalid")
            notes = group.get("notes")
            _require(isinstance(notes, list) and notes, f"{group_path}.notes must not be empty")
            for note_index, note in enumerate(notes):
                _validate_note(note, f"{group_path}.notes[{note_index}]")

    if "render_end_beat" in document:
        render_end_beat = rational(document["render_end_beat"], "render_end_beat")
        _require(render_end_beat > 0, "render_end_beat must be positive")
        latest_position = previous_position
        for control in controls:
            control_position = rational(control["position_beats"], "controls[].position_beats")
            latest_position = control_position if latest_position is None else max(latest_position, control_position)
        _require(
            latest_position is None or render_end_beat > latest_position,
            "render_end_beat must be after every moment and control",
        )

    eligibility = document.get("eligibility")
    _require(isinstance(eligibility, Mapping), "eligibility must be an object")
    _require(isinstance(eligibility.get("training_eligible"), bool), "eligibility.training_eligible is invalid")
    reasons = eligibility.get("isolation_reasons")
    _require(isinstance(reasons, list) and all(isinstance(item, str) for item in reasons),
             "eligibility.isolation_reasons is invalid")
    return dict(document)


_IGNORED_KEYS = {
    "id", "source", "source_signature", "second", "diagnostics", "difficulty", "derived",
    "render_end_beat",
}

_UNORDERED_SEMANTIC_LISTS = {"branches", "flags", "groups", "isolation_reasons", "notes"}


def _duration_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the authoritative, serializable meaning of a duration.

    Beat durations are defined by their reduced rational value. Their resolved
    tick count and seconds are parser projections. Tick durations, on the other
    hand, are defined by the exact resolved tick count.
    """

    basis = value.get("basis")
    if basis == "beat":
        beat = Fraction(int(value["beat_num"]), int(value["beat_den"]))
        return {
            "basis": "beat",
            "beat_num": beat.numerator,
            "beat_den": beat.denominator,
        }
    if basis == "tick":
        return {
            "basis": "tick",
            "resolved_ticks": int(value["resolved_ticks"]),
        }
    raise ChartIRError(f"invalid duration basis in semantic projection: {basis!r}")


def _codec_strip(value: Any) -> Any:
    """Preserve the historical codec payload while removing non-model metadata."""

    if isinstance(value, list):
        return [_codec_strip(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _IGNORED_KEYS:
                continue
            if key == "seconds":
                if value.get("basis") != "tick":
                    continue
                result[key] = float(format(float(item), ".12g"))
                continue
            result[key] = _codec_strip(item)
        return result
    return value


def _semantic_strip(value: Any) -> Any:
    if isinstance(value, list):
        return [_semantic_strip(item) for item in value]
    if isinstance(value, Mapping):
        if value.get("basis") in {"beat", "tick"} and "resolved_ticks" in value:
            return _duration_projection(value)
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _IGNORED_KEYS or key == "seconds":
                continue
            result[key] = _semantic_strip(item)
        return result
    return value


def _semantic_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _branch_projection(branch: Mapping[str, Any]) -> dict[str, Any]:
    result = _semantic_strip(branch)
    timing_mode = branch.get("timing_mode")
    if timing_mode == "total":
        # Total-mode simai serializes one duration for the whole branch. Segment
        # durations are geometric/parser projections and are not independent.
        for segment in result.get("segments", []):
            segment.pop("duration", None)
    elif timing_mode == "per_segment":
        # Per-segment simai serializes every segment duration. The branch total
        # is derived from those authoritative durations.
        result.pop("duration", None)
    else:
        raise ChartIRError(f"invalid branch timing_mode in semantic projection: {timing_mode!r}")
    return result


def _note_projection(note: Mapping[str, Any]) -> dict[str, Any]:
    result = _semantic_strip(note)
    if note.get("kind") in {"slide", "wifi"}:
        result["branches"] = [_branch_projection(branch) for branch in note["branches"]]
    return result


def _group_projection(group: Mapping[str, Any]) -> dict[str, Any]:
    result = _semantic_strip(group)
    result["notes"] = [_note_projection(note) for note in group["notes"]]
    return result


def _canonical_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    result = _semantic_strip(document)
    for moment_index, moment in enumerate(document["moments"]):
        projected_moment = result["moments"][moment_index]
        projected_moment["groups"] = [_group_projection(group) for group in moment["groups"]]
        for group in projected_moment["groups"]:
            for note in group["notes"]:
                if note["kind"] in {"slide", "wifi"}:
                    note["branches"].sort(key=_semantic_key)
            group["notes"].sort(key=_semantic_key)
        projected_moment["groups"].sort(key=_semantic_key)
    return result


def canonicalize_chart_ir(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_chart_ir(document)
    result = deepcopy(dict(document))
    for moment in result["moments"]:
        for group in moment["groups"]:
            for note in group["notes"]:
                note["flags"] = sorted(set(note.get("flags", [])))
                if note["kind"] in {"slide", "wifi"}:
                    note["head"]["flags"] = sorted(set(note["head"].get("flags", [])))
                    for branch in note["branches"]:
                        branch["flags"] = sorted(set(branch.get("flags", [])))
                    note["branches"].sort(key=lambda item: _semantic_key(_codec_strip(item)))
            group["notes"].sort(key=lambda item: _semantic_key(_codec_strip(item)))
        moment["groups"].sort(key=lambda item: _semantic_key(_codec_strip(item)))
    result["controls"].sort(
        key=lambda item: (
            Fraction(item["position_beats"]["num"], item["position_beats"]["den"]),
        )
    )
    result["eligibility"]["isolation_reasons"] = sorted(set(result["eligibility"]["isolation_reasons"]))
    return result


def model_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_chart_ir(document)
    return _canonical_projection(canonical)


def _append_difference(
    differences: list[dict[str, Any]],
    difference: dict[str, Any],
    *,
    count: list[int],
    maximum: int,
) -> None:
    count[0] += 1
    if len(differences) < maximum:
        differences.append(difference)


def _semantic_differences(
    left: Any,
    right: Any,
    *,
    path: str,
    differences: list[dict[str, Any]],
    count: list[int],
    maximum: int,
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            _append_difference(
                differences,
                {"path": f"{path}.{key}", "kind": "missing_from_right", "left": left[key], "right": None},
                count=count,
                maximum=maximum,
            )
        for key in sorted(right_keys - left_keys):
            _append_difference(
                differences,
                {"path": f"{path}.{key}", "kind": "missing_from_left", "left": None, "right": right[key]},
                count=count,
                maximum=maximum,
            )
        for key in sorted(left_keys & right_keys):
            _semantic_differences(
                left[key],
                right[key],
                path=f"{path}.{key}",
                differences=differences,
                count=count,
                maximum=maximum,
            )
        return

    if isinstance(left, list) and isinstance(right, list):
        list_name = path.rsplit(".", 1)[-1]
        if list_name in _UNORDERED_SEMANTIC_LISTS:
            left_items: dict[str, tuple[Any, int]] = {}
            right_items: dict[str, tuple[Any, int]] = {}
            for item in left:
                key = _semantic_key(item)
                previous = left_items.get(key)
                left_items[key] = (item, 1 if previous is None else previous[1] + 1)
            for item in right:
                key = _semantic_key(item)
                previous = right_items.get(key)
                right_items[key] = (item, 1 if previous is None else previous[1] + 1)
            for key in sorted(set(left_items) | set(right_items)):
                left_item, left_count = left_items.get(key, (None, 0))
                right_item, right_count = right_items.get(key, (None, 0))
                if left_count == right_count:
                    continue
                _append_difference(
                    differences,
                    {
                        "path": path,
                        "kind": "unordered_item_count",
                        "left": left_item,
                        "right": right_item,
                        "left_count": left_count,
                        "right_count": right_count,
                    },
                    count=count,
                    maximum=maximum,
                )
            return
        if len(left) != len(right):
            _append_difference(
                differences,
                {"path": path, "kind": "length", "left": len(left), "right": len(right)},
                count=count,
                maximum=maximum,
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _semantic_differences(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                differences=differences,
                count=count,
                maximum=maximum,
            )
        return

    if left != right:
        _append_difference(
            differences,
            {"path": path, "kind": "value", "left": left, "right": right},
            count=count,
            maximum=maximum,
        )


def _audit_records(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            audit: dict[str, Any] = {}
            for key in sorted(_IGNORED_KEYS):
                if key in value:
                    audit[key] = deepcopy(value[key])
            if "seconds" in value:
                audit["seconds"] = value["seconds"]
            if value.get("basis") == "beat" and "resolved_ticks" in value:
                audit["resolved_ticks"] = value["resolved_ticks"]
            if audit:
                records[path] = audit
            for key, item in value.items():
                if key in _IGNORED_KEYS or key == "seconds":
                    continue
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(document, "$")
    return records


def _audit_differences(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    maximum: int,
) -> tuple[list[dict[str, Any]], int]:
    left_records = _audit_records(left)
    right_records = _audit_records(right)
    differences: list[dict[str, Any]] = []
    count = [0]
    for path in sorted(set(left_records) | set(right_records)):
        left_value = left_records.get(path)
        right_value = right_records.get(path)
        if left_value == right_value:
            continue
        _append_difference(
            differences,
            {"path": path, "kind": "audit_value", "left": left_value, "right": right_value},
            count=count,
            maximum=maximum,
        )
    return differences, count[0]


def _normalizations_applied(document: Mapping[str, Any]) -> set[str]:
    normalizations: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "seconds" in value:
                normalizations.add("audit.seconds")
            for key in _IGNORED_KEYS:
                if key in value:
                    normalizations.add(f"audit.{key}")
            if value.get("basis") == "beat" and "resolved_ticks" in value:
                normalizations.add("duration.beat.resolved_ticks")
            if value.get("timing_mode") == "total":
                normalizations.add("slide.total.segment_duration")
            elif value.get("timing_mode") == "per_segment":
                normalizations.add("slide.per_segment.branch_duration")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return normalizations


def compare_chart_ir_semantics(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    max_differences: int = 100,
) -> dict[str, Any]:
    """Compare two chart IR documents using only authoritative model semantics."""

    _require(max_differences >= 0, "max_differences must be non-negative")
    left_canonical = canonicalize_chart_ir(left)
    right_canonical = canonicalize_chart_ir(right)
    left_projection = _canonical_projection(left_canonical)
    right_projection = _canonical_projection(right_canonical)
    semantic_differences: list[dict[str, Any]] = []
    semantic_count = [0]
    _semantic_differences(
        left_projection,
        right_projection,
        path="$",
        differences=semantic_differences,
        count=semantic_count,
        maximum=max_differences,
    )
    audit_differences, audit_count = _audit_differences(
        left_canonical,
        right_canonical,
        maximum=max_differences,
    )
    return {
        "schema": "maimai.chart_ir.semantic_comparison.v1",
        "equal": semantic_count[0] == 0,
        "semantic_mismatch_count": semantic_count[0],
        "semantic_mismatches": semantic_differences,
        "audit_difference_count": audit_count,
        "audit_differences": audit_differences,
        "normalizations_applied": sorted(
            _normalizations_applied(left_canonical) | _normalizations_applied(right_canonical)
        ),
        "truncated": {
            "semantic_mismatches": semantic_count[0] > len(semantic_differences),
            "audit_differences": audit_count > len(audit_differences),
        },
    }


def iter_notes(document: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for moment in document.get("moments", []):
        for group in moment.get("groups", []):
            yield from group.get("notes", [])
