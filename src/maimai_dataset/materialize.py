from __future__ import annotations

from fractions import Fraction
import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from maimai_chart_ir.models import ChartIRError, canonicalize_chart_ir, validate_chart_ir
from maimai_dataset.timing import TimingMap


def normalize_slide_duration_ticks(document: Mapping[str, Any]) -> dict[str, Any]:
    """Give positive beat-based slide durations at least one 180 Hz IR tick.

    A 1/96-beat model duration can have both endpoints in the same IR tick.
    Keep its exact beat value, but give the discrete branch/segment projection
    a nonzero extent. Apply this also to MiaCode's reparse of the same beats.
    Invalid zero/negative semantic durations still fail normal IR validation.
    """
    result = copy.deepcopy(dict(document))
    for moment in result.get("moments", []):
        for group in moment.get("groups", []):
            for note in group.get("notes", []):
                if note.get("kind") not in {"slide", "wifi"}:
                    continue
                for branch in note.get("branches", []):
                    branch_ticks = branch["duration"]["resolved_ticks"]
                    durations = [branch["duration"], *(s["duration"] for s in branch["segments"])]
                    for duration in durations:
                        if (duration.get("basis") == "beat"
                                and duration.get("resolved_ticks") == 0
                                and duration.get("beat_num", 0) > 0
                                and duration.get("beat_den", 0) > 0):
                            duration["resolved_ticks"] = 1
                    if branch["duration"]["resolved_ticks"] != branch_ticks:
                        note["end_tick"] = max(note["end_tick"], note["body_start_tick"] + 1)
    return result


def normalize_each_semantics(document: Mapping[str, Any]) -> dict[str, Any]:
    """Make redundant group/note Each state match canonical simai semantics."""
    result = copy.deepcopy(dict(document))
    for moment in result.get("moments", []):
        for group in moment.get("groups", []):
            notes = group.get("notes", [])
            each = len(notes) >= 2
            group["each"] = each
            for note in notes:
                flags = {str(flag) for flag in note.get("flags", [])}
                is_slide = note.get("kind") in {"slide", "wifi"}
                if each and not is_slide:
                    flags.add("each")
                else:
                    flags.discard("each")
                note["flags"] = sorted(flags)
                if is_slide:
                    head = note.get("head")
                    if isinstance(head, dict):
                        head_flags = {str(flag) for flag in head.get("flags", [])}
                        if each:
                            head_flags.add("each")
                        else:
                            head_flags.discard("each")
                        head["flags"] = sorted(head_flags)
    return result


def _flags(value: Any) -> list[str]:
    return sorted(part for part in str(value or "").split("+") if part)


def _literal_key(row: Mapping[str, Any]) -> str:
    basis = row.get("basis")
    if basis == "beat":
        return f"beat:{int(row['num'])}/{int(row['den'])}"
    if basis == "tick":
        return f"tick:{int(row['ticks'])}"
    if basis == "decimal":
        return f"decimal:{row['value']}"
    raise ChartIRError(f"invalid generated literal basis: {basis!r}")


def logical_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    result: list[tuple[dict[str, Any], dict[str, str]]] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        if row.get("event") in {"SEGMENT_START", "SEGMENT_END"}:
            index += 1
            continue
        if row.get("event") == "VALUE_LITERAL":
            raise ChartIRError("orphan generated VALUE_LITERAL")
        index += 1
        literals: dict[str, str] = {}
        while index < len(rows) and rows[index].get("event") == "VALUE_LITERAL":
            literal = rows[index]
            slot = literal.get("slot")
            if not isinstance(slot, str) or slot in literals:
                raise ChartIRError("duplicate generated literal slot")
            literals[slot] = _literal_key(literal)
            index += 1
        result.append((row, literals))
    return result


def _atom(row: Mapping[str, Any], literals: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if value == "@literal":
        if field not in literals:
            raise ChartIRError(f"generated row is missing literal {field}")
        return literals[field]
    if not isinstance(value, str):
        raise ChartIRError(f"generated row is missing field {field}")
    return value


def _beat_atom(value: str) -> Fraction:
    basis, payload = value.split(":", 1)
    if basis != "beat":
        raise ChartIRError(f"expected beat atom, got {value!r}")
    numerator, denominator = payload.split("/", 1)
    return Fraction(int(numerator), int(denominator))


def _tick_for_second(second: float) -> int:
    return math.ceil(second * 180.0 - 1e-6)


def _duration(
    key: str,
    *,
    timing: TimingMap,
    start_beat: Fraction,
    start_tick: int,
    written_beat: Fraction | None = None,
) -> tuple[dict[str, Any], Fraction | None, int]:
    basis, payload = key.split(":", 1)
    if basis == "beat":
        numerator, denominator = payload.split("/", 1)
        beats = Fraction(int(numerator), int(denominator))
        if written_beat is None:
            # Waiting follows the global beat clock.
            end_second = timing.beat_to_second(start_beat + beats)
        else:
            end_second = timing.beat_to_second(start_beat) + float(beats) * 60 / timing.bpm_at_beat(written_beat)
        end_tick = _tick_for_second(end_second)
        resolved = max(0, end_tick - start_tick)
        return {
            "basis": "beat",
            "beat_num": beats.numerator,
            "beat_den": beats.denominator,
            "resolved_ticks": resolved,
        }, beats, resolved
    if basis == "tick":
        ticks = max(0, int(payload))
        return {"basis": "tick", "resolved_ticks": ticks}, None, ticks
    raise ChartIRError(f"invalid duration atom: {key!r}")


def _control_objects(timing: TimingMap) -> list[dict[str, Any]]:
    controls = []
    for index, point in enumerate(timing.points):
        controls.append({
            "id": f"generated-control-{index:06d}",
            "kind": "bpm",
            "position_beats": {"num": point.beat.numerator, "den": point.beat.denominator},
            "second": point.second,
            "tick": _tick_for_second(point.second),
            "bpm": point.bpm,
        })
    offset = len(controls)
    for index, (beat, numerator, denominator) in enumerate(timing.meter_points):
        second = timing.beat_to_second(beat)
        controls.append({
            "id": f"generated-control-{offset + index:06d}",
            "kind": "meter",
            "position_beats": {"num": beat.numerator, "den": beat.denominator},
            "second": second,
            "tick": _tick_for_second(second),
            "numerator": numerator,
            "denominator": denominator,
        })
    offset = len(controls)
    for index, (beat, multiplier) in enumerate(timing.hs_points):
        second = timing.beat_to_second(beat)
        controls.append({
            "id": f"generated-control-{offset + index:06d}",
            "kind": "hs",
            "position_beats": {"num": beat.numerator, "den": beat.denominator},
            "second": second,
            "tick": _tick_for_second(second),
            "multiplier": multiplier,
        })
    controls.sort(key=lambda item: (
        Fraction(item["position_beats"]["num"], item["position_beats"]["den"]),
        item["id"],
    ))
    return controls


def _parse_note(
    logical: Sequence[tuple[dict[str, Any], dict[str, str]]],
    index: int,
    *,
    timing: TimingMap,
    beat: Fraction,
    start_tick: int,
    ordinal: int,
) -> tuple[dict[str, Any], int]:
    row, literals = logical[index]
    if row.get("event") != "NOTE":
        raise ChartIRError("generated group expected NOTE")
    kind = str(row.get("kind", ""))
    flags = _flags(row.get("flags"))
    note: dict[str, Any] = {
        "id": f"generated-note-{ordinal:08d}",
        "kind": kind,
        "start_tick": start_tick,
        "flags": flags,
    }
    if kind in {"tap", "touch"}:
        note["pad"] = str(row.get("pad", ""))
    elif kind in {"hold", "touch_hold"}:
        note["pad"] = str(row.get("pad", ""))
        duration, _, ticks = _duration(
            _atom(row, literals, "duration"), timing=timing, start_beat=beat, start_tick=start_tick, written_beat=beat
        )
        note["duration"] = duration
        note["end_tick"] = start_tick + ticks
    elif kind in {"slide", "wifi"}:
        note["start_pad"] = str(row.get("start_pad", ""))
        note["head"] = {"kind": str(row.get("head_kind", "star")), "flags": _flags(row.get("head_flags"))}
        wait, wait_beats, wait_ticks = _duration(
            _atom(row, literals, "wait"), timing=timing, start_beat=beat, start_tick=start_tick
        )
        note["wait"] = wait
        body_beat = beat + (wait_beats or Fraction(0))
        body_tick = start_tick + wait_ticks
        note["body_start_tick"] = body_tick

        def branch_from(parent: Mapping[str, Any], parent_literals: Mapping[str, str]) -> dict[str, Any]:
            duration, _, branch_ticks = _duration(
                _atom(parent, parent_literals, "branch_duration"),
                timing=timing,
                start_beat=body_beat,
                start_tick=body_tick,
                written_beat=beat,
            )
            segment_duration, _, _ = _duration(
                _atom(parent, parent_literals, "segment_duration"),
                timing=timing,
                start_beat=body_beat,
                start_tick=body_tick,
                written_beat=beat,
            )
            return {
                "timing_mode": str(parent.get("branch_timing", "total")),
                "flags": _flags(parent.get("branch_flags")),
                "duration": duration,
                "segments": [{
                    "path": str(parent.get("path", "")),
                    "end_pad": str(parent.get("end_pad", "")),
                    "duration": segment_duration,
                }],
                "_ticks": branch_ticks,
            }

        branches = [branch_from(row, literals)]
        index += 1
        while index < len(logical):
            child, child_literals = logical[index]
            event = child.get("event")
            if event == "NOTE_END":
                break
            if event == "SLIDE_BRANCH":
                branches.append(branch_from(child, child_literals))
            elif event == "SLIDE_SEGMENT":
                branch_index = int(child.get("branch_index", len(branches) - 1))
                if not 0 <= branch_index < len(branches):
                    raise ChartIRError("generated slide segment has an invalid branch_index")
                duration, _, _ = _duration(
                    _atom(child, child_literals, "segment_duration"),
                    timing=timing,
                    start_beat=body_beat,
                    start_tick=body_tick,
                    written_beat=beat,
                )
                branches[branch_index]["segments"].append({
                    "path": str(child.get("path", "")),
                    "end_pad": str(child.get("end_pad", "")),
                    "duration": duration,
                })
            else:
                raise ChartIRError(f"unexpected generated slide child: {event!r}")
            index += 1
        end_tick = body_tick
        for branch in branches:
            end_tick = max(end_tick, body_tick + int(branch.pop("_ticks")))
        note["branches"] = branches
        note["end_tick"] = end_tick
        if index >= len(logical) or logical[index][0].get("event") != "NOTE_END":
            raise ChartIRError("generated slide is missing NOTE_END")
        return note, index + 1
    else:
        raise ChartIRError(f"unsupported generated note kind: {kind!r}")

    index += 1
    if index >= len(logical) or logical[index][0].get("event") != "NOTE_END":
        raise ChartIRError("generated note is missing NOTE_END")
    return note, index + 1


def materialize_chart_ir(
    rows: Sequence[Mapping[str, Any]],
    *,
    timing: TimingMap,
    difficulty: str,
    render_end_beat: Fraction | None = None,
) -> dict[str, Any]:
    logical = logical_rows(rows)
    moments: list[dict[str, Any]] = []
    current_beat = Fraction(0)
    index = 0
    ordinal = 0
    while index < len(logical):
        row, literals = logical[index]
        if row.get("event") != "MOMENT":
            raise ChartIRError(f"expected generated MOMENT, got {row.get('event')!r}")
        current_beat += _beat_atom(_atom(row, literals, "delta"))
        second = timing.beat_to_second(current_beat)
        start_tick = _tick_for_second(second)
        groups: list[dict[str, Any]] = []
        index += 1
        while index < len(logical) and logical[index][0].get("event") != "MOMENT_END":
            group_row, _ = logical[index]
            if group_row.get("event") != "GROUP":
                raise ChartIRError("generated moment expected GROUP")
            group = {"each": bool(group_row.get("each", False)), "notes": []}
            index += 1
            while index < len(logical) and logical[index][0].get("event") == "NOTE":
                note, index = _parse_note(
                    logical, index, timing=timing, beat=current_beat, start_tick=start_tick, ordinal=ordinal
                )
                ordinal += 1
                group["notes"].append(note)
            if not group["notes"]:
                raise ChartIRError("generated group contains no notes")
            groups.append(group)
        if index >= len(logical) or logical[index][0].get("event") != "MOMENT_END":
            raise ChartIRError("generated moment is missing MOMENT_END")
        index += 1
        if not groups:
            raise ChartIRError("generated moment contains no groups")
        meter_numerator, meter_denominator = timing.meter_at_beat(current_beat)
        moments.append({
            "position_beats": {"num": current_beat.numerator, "den": current_beat.denominator},
            "second": second,
            "tick": start_tick,
            "bpm": timing.bpm_at_beat(current_beat),
            "meter": {"numerator": meter_numerator, "denominator": meter_denominator},
            "hs": timing.hs_at_beat(current_beat),
            "groups": groups,
        })
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document = {
        "schema": "maimai.chart_ir.v1",
        "source_signature": hashlib.sha256(payload).hexdigest(),
        "tick_rate": 180,
        "difficulty": difficulty,
        "first_seconds": timing.first_seconds,
        "default_meter": {"numerator": timing.meter_numerator, "denominator": timing.meter_denominator},
        "controls": _control_objects(timing),
        "moments": moments,
        "eligibility": {"training_eligible": True, "isolation_reasons": []},
        "diagnostics": {"warnings": []},
    }
    if render_end_beat is not None:
        document["render_end_beat"] = {
            "num": render_end_beat.numerator,
            "den": render_end_beat.denominator,
        }
    document = normalize_slide_duration_ticks(document)
    validate_chart_ir(document)
    return canonicalize_chart_ir(document)
