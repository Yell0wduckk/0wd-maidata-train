"""Rule-record to probability support adapter. No network, geometry or cache ownership."""
from dataclasses import dataclass
import math
import torch
from maidata.chart.types import Moment
from maidata.chart.windows import TimeTarget
from maidata.time import duration_cells


def categorical_support(records, names, *, device):
    basic, penalty, raw = [], [], []
    for name in names:
        record = records.get(name) if isinstance(records, dict) else records[name]
        ok = record is not None and not record['basic']
        basic.append(ok)
        penalty.append([ok and bool(record['penalty'] & (1 << c)) for c in range(4)])
        raw.append([ok and bool(record['physical'] & (1 << c)) for c in range(4)])
    return (torch.tensor(basic, dtype=torch.bool, device=device),
            torch.tensor(penalty, dtype=torch.bool, device=device),
            torch.tensor(raw, dtype=torch.bool, device=device))


@dataclass(frozen=True)
class TimeCandidates:
    lower: torch.Tensor
    upper: torch.Tensor
    valid: torch.Tensor
    penalty: torch.Tensor  # [cells,4]; one decision/representative per cell
    raw: torch.Tensor
    labels: tuple  # exact ticks (M), 4-tick bucket IDs (durations), None = tail
    ticks: tuple
    basic_tick_intervals: tuple | None = None  # exact duration GT support, before bucket rounding

    def target_mask(self, target, timing, cursor):
        if isinstance(target, TimeTarget):
            label = None if target.censored else target.absolute_tick
        elif isinstance(target, Moment):
            label = timing.tick_after(cursor, target.delta_ms)
        else:
            if self.basic_tick_intervals is None or not any(
                    lo <= target.ticks <= hi for lo, hi in self.basic_tick_intervals):
                raise ValueError('duration target outside basic legal tick support')
            label = (target.ticks + 2) // 4
        mask = self.valid & self.valid.new_tensor([v == label for v in self.labels])
        if not mask.any():
            raise ValueError('time target has no basic legal probability mass')
        return mask

    @property
    def choices(self):
        return len({label for label, ok in zip(self.labels, self.valid.tolist()) if ok})


def time_candidates(stage, timing, *, kind, window_end, cap_tick, device):
    """High precision endpoints; U remains nominal, never clamped to audio_end.

    A duration bucket is one generated choice at the field layer's representative
    tick. Its whole cell inherits that tick's support, including nonmonotonic
    Physical intervals; no extra sub-bucket choices or guessed safe_upper.
    """
    cursor = stage['cursor_tick']
    rows = []
    basic_tick_intervals = None
    clean = dict(basic=[], penalty=0, physical=0)
    if kind == 'MOMENT':
        if not cap_tick >= window_end > cursor:
            raise ValueError('MOMENT requires cursor < window_end <= U')
        support = stage['moment']
        intervals, at = support['intervals'], 0
        ticks = list(range(cursor + 1, math.ceil(cap_tick)))
        for i, tick in enumerate(ticks):
            while at < len(intervals) and intervals[at][1] < tick:
                at += 1
            record = clean if tick >= window_end else (
                intervals[at][2] if at < len(intervals) and intervals[at][0] <= tick
                else dict(basic=[support['outside_basic']], penalty=0, physical=0))
            lo = timing.delta_ms(cursor, tick - .5)
            hi = timing.delta_ms(cursor, cap_tick if i == len(ticks)-1 else tick + .5)
            rows.append((lo, hi, tick, tick, record))
        rows.append((timing.delta_ms(cursor, cap_tick), math.inf, None, cap_tick, clean))
    else:
        support = stage['duration']
        lo, hi = support['domain']
        if hi < lo:
            raise ValueError('empty basic duration domain')
        intervals, at = support['intervals'], 0
        basic_tick_intervals = tuple((max(lo, a), min(hi, b)) for a, b, record in intervals
                                     if not record['basic'] and max(lo, a) <= min(hi, b))
        for cell in duration_cells(kind, lo, hi, timing=timing, onset_tick=cursor):
            while at < len(intervals) and intervals[at][1] < cell.tick:
                at += 1
            record = (intervals[at][2] if at < len(intervals) and intervals[at][0] <= cell.tick
                      else dict(basic=[support['outside_basic']], penalty=0, physical=0))
            rows.append((cell.lower, cell.upper, cell.bucket, cell.tick, record))
    if not rows:
        raise ValueError('empty time support')
    a, b, labels, ticks, records = zip(*rows)
    basic, penalty, raw = categorical_support(list(records), range(len(records)), device=device)
    return TimeCandidates(torch.tensor(a, dtype=torch.float64, device=device),
        torch.tensor(b, dtype=torch.float64, device=device), basic, penalty, raw, tuple(labels), tuple(ticks),
        basic_tick_intervals)


def window_stages(example, attribution):
    """Explicit online oracle for current-window supports, with exact snapshot restore.

    Caller supplies real Physical state BEFORE the window (including older live
    actions). It is never reconstructed from the cropped/dropout token history.
    Accepted cache stages inside the window may also be supplied directly to loss;
    boundary/terminal supports must reflect the current cursor and window.
    """
    saved = attribution.snapshot()
    try:
        attribution.inject_boundary(example.inputs[example.boundary_index], start_tick=example.start_tick,
            history_present=not example.history_dropped, window_end=example.end_tick)
        stages = [None] * example.boundary_index
        for target in example.targets[example.boundary_index:]:
            if isinstance(target, TimeTarget):
                stages.append(attribution.terminal_stage())
                attribution.close_window(target)
            else:
                stages.append(attribution.stage_and_accept(target))
        return tuple(stages)
    finally:
        attribution.restore(saved)
