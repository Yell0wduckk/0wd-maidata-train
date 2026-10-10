"""Offline loss-equivalent partitions of time support; no model or rule queries."""
from dataclasses import dataclass
import math
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.chart.types import Moment
from maidata.chart.windows import TimeTarget


@dataclass(frozen=True)
class TimeIntervals:
    lower: torch.Tensor
    upper: torch.Tensor
    valid: torch.Tensor
    penalty: torch.Tensor
    target: torch.Tensor
    choices: int  # Original allowed labels, not the number of merged regions.


def _record_runs(support, lower, upper):
    """Clip accepted integer support runs and retain uncovered basic holes."""
    if upper < lower:
        return
    at = lower
    outside = dict(basic=[support['outside_basic']], penalty=0)
    for a, b, record in support['intervals']:
        if b < at:
            continue
        if a > upper:
            break
        if a > at:
            yield at, a-1, outside
        stop = min(b, upper)
        yield max(at, a), stop, record
        at = stop+1
    if at <= upper:
        yield at, upper, outside


def time_intervals(stage, timing, target, *, kind, window_end, cap_tick):
    """Produce compact loss tensors directly from Physical integer intervals.

    Only rule boundaries and the GT label split the support. Duration runs map
    through the existing clamped four-tick representative, not the full cell's
    ticks. MOMENT endpoints use the complete BPM map; duration uses written BPM.
    Work and memory scale with support runs, never the candidate tick count.
    """
    cursor = stage['cursor_tick']
    if kind == 'MOMENT':
        if not cap_tick >= window_end > cursor:
            raise ValueError('MOMENT requires cursor < window_end <= U')
        if isinstance(target, TimeTarget):
            label = None if target.censored else target.absolute_tick
        elif isinstance(target, Moment):
            label = timing.tick_after(cursor, target.delta_ms)
        else:
            raise ValueError('MOMENT requires a moment target')
        first, last = cursor+1, math.ceil(cap_tick)-1
        closing = math.ceil(window_end)
        runs = list(_record_runs(stage['moment'], first, min(last, closing-1)))
        if last >= closing:
            runs.append((closing, last, dict(basic=[], penalty=0)))

        def bounds(a, b):
            return (timing.delta_ms(cursor, a-.5),
                    timing.delta_ms(cursor, cap_tick if b == last else b+.5))
    else:
        support = stage['duration']
        lower, upper = support['domain']
        minimum = {'HOLD_DURATION': 2, 'SLIDE_DURATION': 1}.get(kind)
        if (minimum is None or type(lower) is not int or type(upper) is not int
                or lower < minimum or upper < lower):
            raise ValueError('invalid duration support')
        if not any(max(lower, a) <= target.ticks <= min(upper, b) and not record['basic']
                   for a, b, record in support['intervals']):
            raise ValueError('duration target outside basic legal tick support')
        label = (target.ticks+2)//4
        first, last = (lower+2)//4, (upper+2)//4
        runs = []
        for a, b, record in _record_runs(support, lower, upper):
            # Invert min(upper, max(lower, bucket*4)) over an integer run.
            start = first if a == lower else max(first, (a+3)//4)
            stop = last if b == upper else min(last, b//4)
            if start <= stop:
                runs.append((start, stop, record))
        scale = 1000*timing.written_tick_seconds(cursor) if kind == 'HOLD_DURATION' else 1.

        def bounds(a, b):
            return ((lower-.5 if a == first else a*4-2)*scale,
                    (upper+.5 if b == last else b*4+2)*scale)

    regions, choices = [], 0

    def append(lo, hi, penalty, is_target):
        if regions and regions[-1][1] == lo and regions[-1][2:] == [penalty, is_target]:
            regions[-1][1] = hi
        else:
            regions.append([lo, hi, penalty, is_target])

    for a, b, record in runs:
        if record['basic']:
            continue
        choices += b-a+1
        if label is not None and a <= label <= b and label == int(label):
            label = int(label)
            parts = ((a, label-1, False), (label, label, True), (label+1, b, False))
        else:
            parts = ((a, b, False),)
        for start, stop, is_target in parts:
            if start <= stop:
                append(*bounds(start, stop), record['penalty'], is_target)
    if kind == 'MOMENT':
        choices += 1
        append(timing.delta_ms(cursor, cap_tick), math.inf, 0, label is None)
    if not any(row[3] for row in regions):
        raise ValueError('time target has no basic legal probability mass')
    return TimeIntervals(
        torch.tensor([r[0] for r in regions], dtype=torch.float64),
        torch.tensor([r[1] for r in regions], dtype=torch.float64),
        torch.ones(len(regions), dtype=torch.bool),
        torch.tensor([[bool(r[2] & (1 << c)) for c in range(4)] for r in regions], dtype=torch.bool),
        torch.tensor([r[3] for r in regions], dtype=torch.bool), choices)


def compact_time_group(group):
    """Merge touching cells only when GT membership and all penalties agree.

    Logistic interval mass is additive. This preserves the basic normalizer,
    GT mass and each Physical complement, including disconnected supports and
    the censored infinite tail. Original decision counts/scales remain intact.
    These intervals are loss integration regions, not generation candidates.
    """
    rows = {key: [] for key in ('lower', 'upper', 'target', 'penalty', 'allowed')}
    for row, valid in enumerate(group['allowed']):
        selected = valid.nonzero().flatten()
        lower, upper = (group[key][row, selected] for key in ('lower', 'upper'))
        target, penalty = (group[key][row, selected] for key in ('target', 'penalty'))
        breaks = ((lower[1:] != upper[:-1]) | (target[1:] != target[:-1])
                  | (penalty[1:] != penalty[:-1]).any(-1))
        starts = torch.cat((selected.new_zeros(1), breaks.nonzero().flatten()+1))
        stops = torch.cat((starts[1:]-1, selected.new_full((1,), len(selected)-1)))
        rows['lower'].append(lower[starts])
        rows['upper'].append(upper[stops])
        rows['target'].append(target[starts])
        rows['penalty'].append(penalty[starts])
        rows['allowed'].append(torch.ones(len(starts), dtype=torch.bool))
    return dict(group, **{key: pad_sequence(values, batch_first=True,
        padding_value=1 if key == 'upper' else 0) for key, values in rows.items()})


def compact_labels(labels):
    return dict(labels, groups={key: compact_time_group(group) if 'lower' in group else group
                               for key, group in labels['groups'].items()})
