"""Independent prefix-conditioned reconstruction queries from exact chart ticks."""
from dataclasses import dataclass

from maidata.chart.types import Note, Route, HoldDuration, SlideDuration, APPLICABLE
from maidata.time.discrete import compile_time_target, duration_cap, TIME_CLASSES
from .schema import KINDS, FIELD_SIZES, TIME_FIELDS, QUERY_ROLES


@dataclass(frozen=True)
class FieldQuery:
    role: str
    beat: float
    origin_tick: int
    elapsed_ticks: int = 0
    ordinal: int = 0
    track: int = 0
    note_kind: int = -1
    route_prefix: tuple = ()
    last_real_tick: int | None = None
    boundary: bool = False


@dataclass(frozen=True)
class TargetRow:
    fields: dict
    times: dict
    query: FieldQuery
    target_classes: tuple = ()
    first_time_step: bool = False
    source_token_index: int | None = None
    basic_classes: tuple = ()
    censored: bool = False
    closes_window: bool = False
    closed_by_continue: bool = False
    closed_lower_bound_tick: int | None = None


@dataclass(frozen=True)
class WindowTarget:
    rows: tuple[TargetRow, ...]
    moment_counts: tuple[int, ...]
    deferred_durations: tuple = ()

    @property
    def query_cost(self):
        return len(self.rows) + len(self.moment_counts)


def reconstruction_target(chart, *, start_tick, target_beats, audio_after_beats=None,
                          audio_end_seconds=None, validate_input=True):
    """Labels arrange queries; each query sees only its own known prefix.

    Out-of-cap GTs remain exact deferred records. Collation refuses those
    windows so they cannot silently become shortened or partial examples.
    """
    if audio_after_beats is None:
        from maidata.entry import CONFIG
        audio_after_beats = CONFIG['audio_after_beats']
    if validate_input and (type(start_tick) is not int or type(target_beats) is not int or target_beats < 1):
        raise ValueError('R targets require an integer start tick and positive beat length')
    end_tick = start_tick + 96 * target_beats
    upper_tick = end_tick + 96 * audio_after_beats
    rows, counts, deferred = [], [0] * target_beats, []
    earlier = [g.tick for g in chart.groups if g.tick < start_tick]
    last_real = earlier[-1] if earlier else None
    cursor, cursor_boundary = start_tick, True

    def time_rows(role, origin, *, ticks=None, lower=None, ordinal=0, track=0,
                  note_kind=-1, routes=(), source=None, cap=None):
        path = compile_time_target(role, ticks=ticks, censored_lower=lower,
            origin_tick=origin, timing=chart.timing,
            window_end_tick=end_tick if role == 'MOMENT' else None)
        name = TIME_FIELDS[('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION').index(role)]
        for i, step in enumerate(path.steps):
            query = FieldQuery(role, step.query_beat, origin, step.elapsed_ticks,
                               ordinal, track, note_kind, tuple(routes), last_real,
                               role == 'MOMENT' and cursor_boundary)
            minimum = 2 if role == 'HOLD_DURATION' else 1
            basic = tuple(c for c in range(TIME_CLASSES) if role == 'MOMENT' or
                (minimum <= step.elapsed_ticks+c+1 <= cap if c < 384 else cap > step.elapsed_ticks+384))
            rows.append(TargetRow({}, {name: ticks if ticks is not None else lower},
                query, step.target_classes, i == 0, source, basic,
                path.censored, path.closes_window, path.closed_by_continue, path.closed_lower_bound_tick))

    for group in chart.groups:
        if not start_tick <= group.tick < end_tick:
            continue
        counts[(group.tick - start_tick) // 96] += 1
        if group.tick != start_tick:
            time_rows('MOMENT', cursor, ticks=group.tick-cursor, source=group.moment_source_index)
        cursor, last_real, cursor_boundary = group.tick, group.tick, False
        ordinal, track, routes, kind = -1, 0, [], -1
        for index, token in enumerate(group.tokens):
            source = group.source_indices[index] if group.source_indices else None
            if isinstance(token, Note):
                ordinal += 1
                track, routes, kind = 0, [], KINDS.index(token.kind)
                fields = {'kind': kind, **{n: int(getattr(token, n)) for n in APPLICABLE[token.kind]}}
                query = FieldQuery('NOTE', cursor/96, cursor, ordinal=ordinal, last_real_tick=last_real)
                rows.append(TargetRow(fields, {}, query, source_token_index=source))
            elif isinstance(token, Route):
                fields = {'route_id': token.route_id}
                if token.track_break is not None:
                    fields['track_break'] = int(token.track_break)
                query = FieldQuery('ROUTE', cursor/96, cursor, ordinal=ordinal,
                    track=track, note_kind=kind, route_prefix=tuple(routes), last_real_tick=last_real)
                rows.append(TargetRow(fields, {}, query, source_token_index=source))
                routes.append(token.route_id)
            elif isinstance(token, (HoldDuration, SlideDuration)):
                role = token.type
                cap = duration_cap(role, origin_tick=cursor, timing=chart.timing,
                    local_end_tick=upper_tick, audio_end_seconds=float('inf') if audio_end_seconds is None else audio_end_seconds)
                if token.ticks > cap:
                    deferred.append(dict(source_token_index=source, role=role, origin_tick=cursor,
                        exact_ticks=token.ticks, local_cap=cap, route_prefix=tuple(routes)))
                else:
                    time_rows(role, cursor, ticks=token.ticks, ordinal=ordinal,
                              track=track, note_kind=kind, routes=routes, source=source, cap=cap)
                if isinstance(token, SlideDuration):
                    track += 1
                    routes = []
    following = next((g for g in chart.groups if g.tick >= end_tick), None)
    if following is not None and following.tick < upper_tick:
        time_rows('MOMENT', cursor, ticks=following.tick-cursor, source=following.moment_source_index)
    else:
        time_rows('MOMENT', cursor, lower=max(1, upper_tick-cursor))
    if validate_input and any(n > 96 for n in counts):
        raise ValueError('more than 96 distinct MOMENTs in a beat')
    return WindowTarget(tuple(rows), tuple(counts), tuple(deferred))


def required_query_rows(target_lengths):
    """Inspect query cost; this is no longer a network capacity or GT input."""
    lengths = tuple(target_lengths)
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('positive query counts required')
    return max(lengths)


def collate_reconstruction(targets, *, target_beats, device, query_rows=None, validate_input=True):
    import torch
    from .types import ReconstructionQueries, ReconstructionTargets
    if not targets:
        raise ValueError('a nonempty reconstruction batch is required')
    for w, target in enumerate(targets):
        if target is not None and target.deferred_durations:
            raise ValueError(f'GT duration beyond local audio window is deferred: batch window {w}: '
                             f'{target.deferred_durations!r}')
    width = max(1, max((len(t.rows) for t in targets if t is not None), default=0))
    shape = (len(targets), width)
    fields = {n: torch.full(shape, -100, dtype=torch.long, device=device) for n in FIELD_SIZES}
    times = {n: torch.zeros(shape, device=device) for n in TIME_FIELDS}
    time_valid = {n: torch.zeros(shape, dtype=torch.bool, device=device) for n in TIME_FIELDS}
    classes = {n: torch.zeros((*shape, TIME_CLASSES), dtype=torch.bool, device=device) for n in TIME_FIELDS}
    basic = {n: torch.zeros((*shape, TIME_CLASSES), dtype=torch.bool, device=device) for n in TIME_FIELDS}
    counts = torch.zeros((len(targets), target_beats), dtype=torch.long, device=device)
    valid = torch.tensor([t is not None for t in targets], device=device, dtype=torch.bool)
    max_chain = max(1, max((len(r.query.route_prefix) for t in targets if t for r in t.rows), default=0))
    integer = lambda: torch.zeros(shape, dtype=torch.long, device=device)
    real = lambda: torch.zeros(shape, device=device)
    q = ReconstructionQueries(integer(), real(), real(), real(), integer(), integer(),
        torch.full(shape, -1, dtype=torch.long, device=device),
        torch.zeros((*shape, max_chain), dtype=torch.long, device=device),
        torch.zeros((*shape, max_chain), dtype=torch.bool, device=device),
        torch.zeros(shape, dtype=torch.bool, device=device), real(),
        torch.zeros(shape, dtype=torch.bool, device=device), torch.zeros(shape, dtype=torch.bool, device=device))
    for w, target in enumerate(targets):
        if target is None:
            continue
        counts[w] = torch.tensor(target.moment_counts, device=device)
        for i, row in enumerate(target.rows):
            query = row.query
            for name, value in row.fields.items():
                fields[name][w, i] = value
            for name, value in row.times.items():
                times[name][w, i] = value
                time_valid[name][w, i] = row.first_time_step
                classes[name][w, i, list(row.target_classes)] = True
                basic[name][w, i, list(row.basic_classes)] = True
            q.roles[w, i] = QUERY_ROLES.index(query.role)
            q.beats[w, i], q.origins[w, i], q.elapsed[w, i] = query.beat, query.origin_tick/96, query.elapsed_ticks/96
            q.ordinals[w, i], q.tracks[w, i], q.note_kinds[w, i] = query.ordinal, query.track, query.note_kind
            q.valid[w, i] = True
            q.history_gaps[w, i] = 0 if query.last_real_tick is None else (query.origin_tick-query.last_real_tick)/96
            q.boundary[w, i], q.last_real_valid[w, i] = query.boundary, query.last_real_tick is not None
            if query.route_prefix:
                n = len(query.route_prefix)
                q.route_ids[w, i, :n] = torch.tensor(query.route_prefix, device=device)
                q.route_valid[w, i, :n] = True
    return ReconstructionTargets(fields, times, time_valid, counts, valid, q, classes, basic)


def collate_planner(targets, *, device, validate_input=True):
    import torch
    from .types import PlannerTargets
    if validate_input and not targets:
        raise ValueError('empty Planner target batch')
    return PlannerTargets(
        torch.tensor([t.continuous if t is not None else (0.,) * 8 for t in targets], device=device, dtype=torch.float32),
        torch.tensor([t.category if t is not None else 0 for t in targets], device=device, dtype=torch.long),
        torch.tensor([t is not None for t in targets], device=device, dtype=torch.bool))
