"""Comparison of generated chart artifacts."""

from __future__ import annotations
from collections import Counter
from maimai_baseline.time.coordinates import half_up_nonnegative
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from maimai_dataset.timing import TimingMap
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import statistics


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _moments(
    rows: Sequence[Mapping[str, Any]], timing: TimingMap
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous_tick = 0
    current_tick = 0
    for row in rows:
        if row.get("event") != "MOMENT":
            continue
        delta_tick = int(row["delta_tick"])
        current_tick += delta_tick
        start_second = timing.beat_to_second(previous_tick / CHART_TICKS_PER_BEAT)
        end_second = timing.beat_to_second(current_tick / CHART_TICKS_PER_BEAT)
        delta_ms = (end_second - start_second) * 1000.0
        result.append(
            {
                "index": len(result),
                "tick": current_tick,
                "second": float(end_second),
                "delta_tick": delta_tick,
                "delta_ms": float(delta_ms),
                "delta_10ms": half_up_nonnegative(max(0.0, delta_ms) / 10.0),
                "anchor_bpm": float(
                    timing.bpm_at_beat(previous_tick / CHART_TICKS_PER_BEAT)
                ),
            }
        )
        previous_tick = current_tick
    return result


def _longest_run(values: Sequence[int], selected: int | None) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value == selected else 0
        longest = max(longest, current)
    return longest


def _mode(values: Sequence[int]) -> tuple[int | None, int, float]:
    if not values:
        return None, 0, 0.0
    counts = Counter(values)
    value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return value, count, count / len(values)


def _quantile(values: Sequence[float], position: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = position * (len(ordered) - 1)
    left = int(math.floor(index))
    right = int(math.ceil(index))
    weight = index - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _distribution_distance_ms(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("nan")
    return statistics.fmean(
        abs(_quantile(left, index / 100.0) - _quantile(right, index / 100.0))
        for index in range(101)
    )


def _nearest_reference_error_ms(
    moments: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> dict[str, float | int | None]:
    reference_seconds = [float(moment["second"]) for moment in reference]
    if not moments or not reference_seconds:
        return {"count": 0, "median": None, "mean": None}
    errors = [
        min(abs(float(moment["second"]) - value) for value in reference_seconds)
        * 1000.0
        for moment in moments
    ]
    return {
        "count": len(errors),
        "median": float(statistics.median(errors)),
        "mean": float(statistics.fmean(errors)),
    }


def summarize_generation(
    directory: Path,
    *,
    timing: TimingMap,
    reference_moments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    rows = _json(directory / "model.rows.json")
    generation = _json(directory / "generation.json")
    moments = _moments(rows, timing)
    tick_values = [int(moment["delta_tick"]) for moment in moments]
    time_values = [int(moment["delta_10ms"]) for moment in moments]
    positive_ms = [float(moment["delta_ms"]) for moment in moments if moment["delta_ms"] > 0]
    dominant_tick, tick_count, tick_share = _mode(tick_values)
    dominant_10ms, time_count, time_share = _mode(time_values)
    bpm_groups: dict[str, list[float]] = {}
    for moment in moments:
        key = f"{float(moment['anchor_bpm']):.6g}"
        bpm_groups.setdefault(key, []).append(float(moment["delta_ms"]))
    traces = _trace(directory / "prediction-trace.jsonl")
    moment_traces = [value for trace in traces
                     if isinstance((value := trace.get("time", {}).get("delta_ms")), Mapping)]
    later_nonpositive = sum(
        int(moment["delta_tick"]) <= 0 for moment in moments[1:]
    )
    end_tick = (generation.get("audio_preparation") or {}).get("audio_onset_tick")
    out_of_bounds = sum(
        int(moment["tick"]) < 0
        or (end_tick is not None and int(moment["tick"]) > end_tick)
        for moment in moments
    )
    materialization_path = directory / "materialization.json"
    materialization = _json(materialization_path) if materialization_path.is_file() else None
    return {
        "directory": str(directory),
        "architecture": generation.get("architecture"),
        "checkpoint_epoch": generation.get("checkpoint_epoch"),
        "moments": len(moments),
        "intervals": {
            "dominant_tick": dominant_tick,
            "dominant_tick_count": tick_count,
            "dominant_tick_share": tick_share,
            "dominant_tick_longest_consecutive_run": _longest_run(
                tick_values, dominant_tick
            ),
            "dominant_10ms": dominant_10ms,
            "dominant_10ms_count": time_count,
            "dominant_10ms_share": time_share,
            "dominant_10ms_longest_consecutive_run": _longest_run(
                time_values, dominant_10ms
            ),
            "actual_ms": {
                "median": None if not positive_ms else statistics.median(positive_ms),
                "mean": None if not positive_ms else statistics.fmean(positive_ms),
                "p90": None if not positive_ms else _quantile(positive_ms, 0.9),
            },
            "projected_tick_histogram": dict(
                sorted(Counter(tick_values).items())
            ),
            "quantized_10ms_histogram": dict(
                sorted(Counter(time_values).items())
            ),
            "by_anchor_bpm": {
                bpm: {
                    "count": len(values),
                    "median_ms": statistics.median(values),
                    "mean_ms": statistics.fmean(values),
                }
                for bpm, values in sorted(bpm_groups.items(), key=lambda item: float(item[0]))
            },
        },
        "time_candidates": {"traced_moments": len(moment_traces)},
        "validity": {
            "later_nonpositive_moments": later_nonpositive,
            "out_of_bounds_moments": out_of_bounds,
            "truncated_windows": sum(
                window.get("status") != "complete"
                for window in generation.get("windows", [])
            ),
            "roundtrip": (
                None
                if materialization is None
                else materialization.get("roundtrip")
            ),
        },
        "nearest_reference_onset_error_ms": _nearest_reference_error_ms(
            moments, reference_moments
        ),
        "_interval_ms_values": [float(moment["delta_ms"]) for moment in moments],
    }


def compare_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = _json(manifest_path)
    cases: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        timing_path = (manifest_path.parent / str(case["timing"])).resolve()
        timing = TimingMap.from_dict(_json(timing_path))
        reference: list[dict[str, Any]] = []
        if case.get("reference_rows"):
            reference = _moments(
                _json((manifest_path.parent / str(case["reference_rows"])).resolve()),
                timing,
            )
        left = summarize_generation(
            (manifest_path.parent / str(case["left"])).resolve(),
            timing=timing,
            reference_moments=reference,
        )
        right = summarize_generation(
            (manifest_path.parent / str(case["right"])).resolve(),
            timing=timing,
            reference_moments=reference,
        )
        distance = _distribution_distance_ms(
            left.pop("_interval_ms_values"),
            right.pop("_interval_ms_values"),
        )
        cases.append(
            {
                "song_id": str(case["song_id"]),
                "difficulty": case.get("difficulty"),
                "seed": int(case.get("seed", 20260811)),
                "temperature": float(case.get("temperature", 0.0)),
                "left": left,
                "right": right,
                "interval_distribution_distance_ms": distance,
            }
        )
    return {
        "schema": "maimai.baseline_comparison.v1",
        "selection_policy": "caller-selected-runs",
        "raw_nll_compared": False,
        "cases": cases,
    }
