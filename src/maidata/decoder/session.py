"""Optional runtime snapshot binding. No search policy or full-song scheduler."""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeSnapshot:
    cache: object
    routed: object
    completed: object
    pending: dict
    rules: object


class DecodeSession:
    """Caller-owned pending selections, network K/V and rule state roll back together.

    rules is a PhysicalState or AttributionState implementing snapshot/restore.
    It retains full action/handoff/last-onset state outside cropped token history.
    Immutable model objects keep tensor references; no copying or detaching K/V.
    """
    def __init__(self, rules):
        self.rules = rules
        self.cache = self.routed = self.completed = None
        self.pending = {}

    def snapshot(self):
        return DecodeSnapshot(self.cache, self.routed, self.completed, deepcopy(self.pending), self.rules.snapshot())

    def restore(self, snapshot):
        if not isinstance(snapshot, DecodeSnapshot):
            raise TypeError('expected DecodeSnapshot')
        self.rules.restore(snapshot.rules)
        self.cache, self.routed, self.completed = snapshot.cache, snapshot.routed, snapshot.completed
        self.pending = deepcopy(snapshot.pending)

    def reset_attention(self):
        """Start fresh self-attention at a new window; never clear Physical/history."""
        if self.pending:
            raise ValueError('cannot switch window during incomplete content')
        grammar = getattr(self.rules, 'physical', self.rules).grammar
        if grammar.state.phase != 'start':
            grammar.finish()
        self.cache = self.routed = self.completed = None
