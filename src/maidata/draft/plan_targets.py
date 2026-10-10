"""Simai music_structure_v2 statistics migrated to public NOTE/timing semantics.

No external simai imports; durations use the full TimingMap, including BPM changes.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .source_timing import read_source_end_beat


@dataclass(frozen=True)
class PlanTarget:
    continuous: tuple[float, ...]
    category: int


def statistics(notes, duration, span_ticks, start_tick):
    count = len(notes)
    groups = Counter(n.tick for n in notes)
    chord = sum(n > 1 for n in groups.values()) / max(len(groups), 1)
    slide = sum(n.kind == 'slide' for n in notes) / max(count, 1)
    touch = sum(n.kind in ('touch', 'touch_hold') for n in notes) / max(count, 1)
    hold = sum(n.kind in ('hold', 'touch_hold') for n in notes) / max(count, 1)
    buttons = [n for n in notes if n.button is not None]
    pairs = list(zip(buttons, buttons[1:]))
    jacks = sum(a.button == b.button and b.tick - a.tick <= 96 for a, b in pairs) / max(len(pairs), 1)
    jumps = sum(min(abs(a.button - b.button), 8 - abs(a.button - b.button)) >= 3
                for a, b in pairs) / max(len(pairs), 1)
    density = count / max(duration, 1e-6)
    workload = density * (1 + .35 * chord + .2 * slide + .1 * hold + .15 * jacks + .15 * jumps)
    last = max(((n.tick - start_tick) / max(span_ticks, 1e-6) for n in notes), default=0.)
    return (min(1., math.log1p(count) / math.log1p(64)), min(1., density / 16),
            chord, slide, min(1., last), touch, hold), workload


class MusicPlanTargets:
    def __init__(self, chart, *, validate_input=True):
        self.validate_input = validate_input
        end = chart.source_end_beat
        if end is None:
            end = read_source_end_beat(chart.source_path, chart.difficulty, validate_input=validate_input)
        if validate_input and (end <= 0):
            raise ValueError('source end beat must be positive')
        self.chart = chart
        self.source_end_beat = end
        self.notes = sorted(chart.notes, key=lambda n: n.tick)  # Stable within each onset.
        self.ticks = [n.tick for n in self.notes]
        duration = chart.timing.beat_to_second(end) - chart.timing.beat_to_second(0)
        _, self.reference = statistics(self.notes, duration, float(end * 96), 0)

    def window(self, start_tick, end_tick):
        if self.validate_input and end_tick <= start_tick:
            raise ValueError('Planner target interval must have positive duration')
        notes = self.notes[bisect_left(self.ticks, start_tick):bisect_left(self.ticks, end_tick)]
        duration = self.chart.timing.second(end_tick) - self.chart.timing.second(start_tick)
        values, workload = statistics(notes, duration, end_tick - start_tick, start_tick)
        intensity = min(1., max(0., .5 * workload / max(self.reference, 1e-6)))
        grouped = defaultdict(list)
        for note in notes:
            grouped[note.tick].append(note)
        classes = Counter()
        for group in grouped.values():
            kinds = {n.kind for n in group}
            category = (3 if kinds & {'touch', 'touch_hold'} else 2 if 'slide' in kinds
                        else 1 if len(group) > 1 else 0)
            classes[category] += 1
        category = max(classes, key=lambda k: (classes[k], k)) if classes else 0
        return PlanTarget((*values, intensity), category)
