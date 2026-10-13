"""Inference-only post-sampling repairs for Slide path-entry collisions.

The raw Physical oracle and training labels retain their original meaning.
Only path_guard is deferred; launch_guard and other categories stay masked.
"""
from dataclasses import dataclass, replace
import math

from maidata.chart.types import Note
from maidata.physical.actions import materialize
from maidata.physical.geometry import pad_bit
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.rules import EPS


@dataclass(frozen=True)
class TailRepair:
    original: Note
    gaps_ms: tuple[float, ...]
    replacement: Note | None = None
    minimum_tick: int | None = None

    def record(self, tick):
        from maidata.chart import token_to_dict
        return dict(strategy='ex' if self.replacement is not None else 'moment',
                    tick=tick, original=token_to_dict(self.original), gaps_ms=list(self.gaps_ms))


class GenerationPhysicalState(InferencePhysicalState):
    """Do not eliminate path collisions before their NOTE has been sampled."""

    def _action(self, completion, *, touch_repeat=True, deferred=False):
        decision = super()._action(completion, touch_repeat=touch_repeat, deferred=deferred)
        if 'tap_on_slide' not in decision.physical:
            return decision
        c = materialize(completion, self.timing, self.geometry)
        launches_here = any(a.track and a.pad == c.pad and abs(a.launch-c.press_start) <= EPS
                            for a in self.actions)
        launch_guard = any(a.track and a.pad == c.pad and not launches_here
                          and a.launch-EPS <= c.press_start <= a.guard_end+EPS for a in self.actions)
        if launch_guard:
            return decision
        return replace(decision, physical=decision.physical - {'tap_on_slide'})

    def tail_repair(self, note):
        if note.kind in ('touch', 'touch_hold'):
            return None
        tick = self.grammar.state.cursor
        now, bit = self.timing.second(tick), pad_bit(note.pad)
        launches_here = any(a.track and a.pad == note.pad and abs(a.launch-now) <= EPS for a in self.actions)
        entries = [entered for a in self.actions if a.track for motion in a.motions
                   for entered, mask in motion.entries
                   if entered < now-EPS and now <= entered+.2+EPS and mask & bit
                   and not (launches_here and a.pad == note.pad and abs(entered-a.launch) <= EPS)]
        if not entries:
            return None
        gaps = tuple((now-entered)*1000 for entered in entries)
        if any(gap < 150-EPS*1000 for gap in gaps):
            safe = max(entered+.15 for entered in entries)
            minimum = max(tick+1, math.ceil(self.timing.second_to_beat(safe)*96 - 1e-9))
            return TailRepair(note, gaps, minimum_tick=minimum)
        if not note.ex:
            return TailRepair(note, gaps, replacement=replace(note, ex=1))
        return None


def generation_rules(rules):
    """Upgrade standalone raw callers while preserving their complete prefix."""
    if isinstance(rules, GenerationPhysicalState):
        return rules
    adapted = GenerationPhysicalState(rules.timing, rules.geometry, audio_end_seconds=rules.audio_end_seconds)
    adapted.duration_end_seconds = getattr(rules, 'duration_end_seconds', None)
    adapted.restore(rules.snapshot())
    return adapted
