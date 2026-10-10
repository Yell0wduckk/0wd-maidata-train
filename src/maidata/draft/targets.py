"""Canonical R row labels and tensor collation, separate from all networks."""
from dataclasses import dataclass

from maidata.chart.types import Note, Route, HoldDuration, SlideDuration, APPLICABLE
from .schema import ROW_TYPES, KINDS, BUTTON_PADS, TOUCH_PADS, FIELD_SIZES, TIME_FIELDS


@dataclass(frozen=True)
class TargetRow:
    fields: dict
    times: dict


@dataclass(frozen=True)
class WindowTarget:
    rows: tuple[TargetRow, ...]
    moment_counts: tuple[int, ...]


def reconstruction_target(chart, *, start_tick, target_beats, validate_input=True):
    if validate_input and (type(start_tick) is not int or type(target_beats) is not int or target_beats < 1):
        raise ValueError('R targets require an integer start tick and positive beat length')
    rows, counts = [], [0] * target_beats
    for group in chart.groups:
        if not start_tick <= group.tick < start_tick + 96 * target_beats:
            continue
        offset = group.tick - start_tick
        counts[offset // 96] += 1
        # Even a real onset at the externally injected boundary needs an R MOMENT.
        rows.append(TargetRow({'type': ROW_TYPES.index('MOMENT')}, {'moment_position_tick': offset}))
        for token in group.tokens:
            fields, times = {'type': ROW_TYPES.index(token.type)}, {}
            if isinstance(token, Note):
                fields['kind'] = KINDS.index(token.kind)
                if token.kind in ('touch', 'touch_hold'):
                    fields['touch_pad'] = TOUCH_PADS.index(token.pad)
                else:
                    fields['button_pad'] = BUTTON_PADS.index(token.pad)
                fields.update({name: int(getattr(token, name)) for name in APPLICABLE[token.kind]})
            elif isinstance(token, Route):
                fields['route_id'] = token.route_id
                if token.track_break is not None:
                    fields['track_break'] = int(token.track_break)
            elif isinstance(token, HoldDuration):
                times['hold_duration_tick'] = token.ticks
            elif isinstance(token, SlideDuration):
                times['slide_duration_tick'] = token.ticks
            rows.append(TargetRow(fields, times))
    if validate_input and (any(n > 96 for n in counts)):
        raise ValueError('more than 96 distinct MOMENTs in a beat')
    rows.append(TargetRow({'type': ROW_TYPES.index('SEGMENT_END')}, {}))
    return WindowTarget(tuple(rows), tuple(counts))


def required_query_rows(target_lengths):
    """Lengths must already include SEGMENT_END; call once over selected metadata."""
    lengths = tuple(target_lengths)
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('positive new-field row lengths including SEGMENT_END required')
    return max(lengths)


def collate_reconstruction(targets, *, query_rows, target_beats, device, validate_input=True):
    import torch
    from .types import ReconstructionTargets
    if validate_input and (type(query_rows) is not int or query_rows < 1 or not targets):
        raise ValueError('positive fixed query_rows and a nonempty batch are required')
    shape = (len(targets), query_rows)
    fields = {name: torch.full(shape, -100, dtype=torch.long, device=device) for name in FIELD_SIZES}
    times = {name: torch.zeros(shape, dtype=torch.float32, device=device) for name in TIME_FIELDS}
    time_valid = {name: torch.zeros(shape, dtype=torch.bool, device=device) for name in TIME_FIELDS}
    counts = torch.zeros((len(targets), target_beats), dtype=torch.long, device=device)
    valid = torch.tensor([t is not None for t in targets], dtype=torch.bool, device=device)
    for w, target in enumerate(targets):
        if target is None:
            continue
        if validate_input and (len(target.rows) > query_rows):
            raise ValueError(f'R query_rows={query_rows} cannot hold {len(target.rows)} rows including SEGMENT_END')
        if validate_input and (len(target.moment_counts) != target_beats):
            raise ValueError('R count length does not match target_beats')
        counts[w] = torch.tensor(target.moment_counts, device=device)
        for i, row in enumerate(target.rows):
            for name, value in row.fields.items():
                fields[name][w, i] = value
            for name, value in row.times.items():
                times[name][w, i] = value
                time_valid[name][w, i] = True
    return ReconstructionTargets(fields, times, time_valid, counts, valid)


def collate_planner(targets, *, device, validate_input=True):
    import torch
    from .types import PlannerTargets
    if validate_input and (not targets):
        raise ValueError('empty Planner target batch')
    return PlannerTargets(
        torch.tensor([t.continuous if t is not None else (0.,) * 8 for t in targets], device=device, dtype=torch.float32),
        torch.tensor([t.category if t is not None else 0 for t in targets], device=device, dtype=torch.long),
        torch.tensor([t is not None for t in targets], device=device, dtype=torch.bool))
