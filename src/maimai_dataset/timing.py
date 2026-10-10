from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import bisect
import math
from typing import Any, Mapping, Sequence

from maimai_chart_ir.models import ChartIRError, rational, validate_chart_ir


TIMING_SCHEMA = "maimai.timing.v1"


def _fraction(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["num"]), int(value["den"]))


@dataclass(frozen=True)
class BpmPoint:
    beat: Fraction
    bpm: float
    second: float


class TimingMap:
    """Piecewise-constant BPM map with deterministic beat/second conversion."""

    def __init__(
        self,
        *,
        first_seconds: float,
        bpm_points: Sequence[tuple[Fraction, float]],
        meter_points: Sequence[tuple[Fraction, int, int]] = (),
        hs_points: Sequence[tuple[Fraction, float]] = (),
        meter_numerator: int = 4,
        meter_denominator: int = 4,
        end_beat: Fraction | None = None,
    ) -> None:
        if not math.isfinite(first_seconds):
            raise ChartIRError("timing first_seconds must be finite")
        normalized: dict[Fraction, float] = {}
        for beat, bpm in bpm_points:
            if not math.isfinite(float(bpm)) or float(bpm) <= 0:
                raise ChartIRError("timing BPM must be positive")
            normalized[Fraction(beat)] = float(bpm)
        if not normalized:
            normalized[Fraction(0)] = 120.0
        if Fraction(0) not in normalized:
            first_key = min(normalized)
            normalized[Fraction(0)] = normalized[first_key]
        ordered = sorted(normalized.items())
        points: list[BpmPoint] = []
        current_second = float(first_seconds)
        previous_beat = Fraction(0)
        previous_bpm = normalized[Fraction(0)]
        for beat, bpm in ordered:
            if beat < 0:
                continue
            if beat == 0:
                previous_bpm = bpm
                if points:
                    points[0] = BpmPoint(beat, bpm, current_second)
                else:
                    points.append(BpmPoint(beat, bpm, current_second))
                continue
            current_second += float(beat - previous_beat) * 60.0 / previous_bpm
            points.append(BpmPoint(beat, bpm, current_second))
            previous_beat = beat
            previous_bpm = bpm
        self.first_seconds = float(first_seconds)
        self.points = tuple(points)
        self._beats = tuple(point.beat for point in points)
        self._seconds = tuple(point.second for point in points)
        self.meter_numerator = int(meter_numerator)
        self.meter_denominator = int(meter_denominator)
        self.meter_points = tuple(sorted(
            (Fraction(beat), int(numerator), int(denominator))
            for beat, numerator, denominator in meter_points
        ))
        self.hs_points = tuple(sorted(
            (Fraction(beat), float(multiplier)) for beat, multiplier in hs_points
        ))
        self.end_beat = Fraction(end_beat) if end_beat is not None else None

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "TimingMap":
        validate_chart_ir(document)
        bpm_points: list[tuple[Fraction, float]] = []
        meter_points: list[tuple[Fraction, int, int]] = []
        hs_points: list[tuple[Fraction, float]] = []
        for control in document["controls"]:
            if control["kind"] == "bpm":
                bpm_points.append((_fraction(control["position_beats"]), float(control["bpm"])))
            elif control["kind"] == "meter":
                meter_points.append((
                    _fraction(control["position_beats"]),
                    int(control["numerator"]),
                    int(control["denominator"]),
                ))
            elif control["kind"] == "hs":
                hs_points.append((_fraction(control["position_beats"]), float(control["multiplier"])))
        if not bpm_points:
            initial_bpm = float(document["moments"][0]["bpm"]) if document["moments"] else 120.0
            bpm_points.append((Fraction(0), initial_bpm))
        end_beat = max(
            (_fraction(moment["position_beats"]) for moment in document["moments"]),
            default=Fraction(0),
        )
        meter = document.get("default_meter", {"numerator": 4, "denominator": 4})
        return cls(
            first_seconds=float(document["first_seconds"]),
            bpm_points=bpm_points,
            meter_points=meter_points,
            hs_points=hs_points,
            meter_numerator=int(meter["numerator"]),
            meter_denominator=int(meter["denominator"]),
            end_beat=end_beat,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimingMap":
        if value.get("schema") != TIMING_SCHEMA:
            raise ChartIRError(f"unsupported timing schema: {value.get('schema')!r}")
        bpm_points = []
        for index, point in enumerate(value.get("bpm_points", [])):
            if not isinstance(point, Mapping):
                raise ChartIRError(f"bpm_points[{index}] must be an object")
            bpm_points.append((rational(point.get("position_beats"), f"bpm_points[{index}].position_beats"), float(point["bpm"])))
        meter_points = [
            (
                rational(point["position_beats"], f"meter_points[{index}].position_beats"),
                int(point["numerator"]),
                int(point["denominator"]),
            )
            for index, point in enumerate(value.get("meter_points", []))
        ]
        hs_points = [
            (
                rational(point["position_beats"], f"hs_points[{index}].position_beats"),
                float(point["multiplier"]),
            )
            for index, point in enumerate(value.get("hs_points", []))
        ]
        meter = value.get("meter", {"numerator": 4, "denominator": 4})
        end_value = value.get("end_beat")
        end_beat = rational(end_value, "end_beat") if end_value is not None else None
        return cls(
            first_seconds=float(value["first_seconds"]),
            bpm_points=bpm_points,
            meter_points=meter_points,
            hs_points=hs_points,
            meter_numerator=int(meter["numerator"]),
            meter_denominator=int(meter["denominator"]),
            end_beat=end_beat,
        )

    def to_dict(self, *, end_beat: Fraction | None = None) -> dict[str, Any]:
        final_end = self.end_beat if end_beat is None else Fraction(end_beat)
        return {
            "schema": TIMING_SCHEMA,
            "first_seconds": self.first_seconds,
            "meter": {"numerator": self.meter_numerator, "denominator": self.meter_denominator},
            "bpm_points": [
                {
                    "position_beats": {"num": point.beat.numerator, "den": point.beat.denominator},
                    "bpm": point.bpm,
                    "second": point.second,
                }
                for point in self.points
            ],
            "meter_points": [
                {
                    "position_beats": {"num": beat.numerator, "den": beat.denominator},
                    "numerator": numerator,
                    "denominator": denominator,
                }
                for beat, numerator, denominator in self.meter_points
            ],
            "hs_points": [
                {
                    "position_beats": {"num": beat.numerator, "den": beat.denominator},
                    "multiplier": multiplier,
                }
                for beat, multiplier in self.hs_points
            ],
            "end_beat": None if final_end is None else {"num": final_end.numerator, "den": final_end.denominator},
        }

    def _beat_index(self, beat: Fraction) -> int:
        return max(0, bisect.bisect_right(self._beats, beat) - 1)

    def _second_index(self, second: float) -> int:
        return max(0, bisect.bisect_right(self._seconds, second) - 1)

    def bpm_at_beat(self, beat: Fraction | float) -> float:
        value = Fraction(beat)
        if value < 0:
            return self.points[0].bpm
        return self.points[self._beat_index(value)].bpm

    def meter_at_beat(self, beat: Fraction | float) -> tuple[int, int]:
        value = Fraction(beat)
        result = (self.meter_numerator, self.meter_denominator)
        for position, numerator, denominator in self.meter_points:
            if position > value:
                break
            result = (numerator, denominator)
        return result

    def hs_at_beat(self, beat: Fraction | float) -> float:
        value = Fraction(beat)
        result = 1.0
        for position, multiplier in self.hs_points:
            if position > value:
                break
            result = multiplier
        return result

    def beat_to_second(self, beat: Fraction | float) -> float:
        value = Fraction(beat)
        if value < 0:
            return self.first_seconds + float(value) * 60.0 / self.points[0].bpm
        point = self.points[self._beat_index(value)]
        return point.second + float(value - point.beat) * 60.0 / point.bpm

    def second_to_beat(self, second: float) -> float:
        value = float(second)
        if value < self.first_seconds:
            return (value - self.first_seconds) * self.points[0].bpm / 60.0
        point = self.points[self._second_index(value)]
        return float(point.beat) + (value - point.second) * point.bpm / 60.0
