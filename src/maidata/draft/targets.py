"""Independent owning-beat queries from exact source onsets; duration is not supervised."""
from dataclasses import dataclass, replace

from maidata.chart.types import Note, Route, SlideDuration, APPLICABLE
from .schema import KINDS, FIELD_SIZES, QUERY_ROLES


@dataclass(frozen=True)
class FieldQuery:
    role: str
    beat: float
    origin_tick: int
    ordinal: int = 0  # MOMENT ordinal within the owning beat, not total count.
    note_ordinal: int = 0
    track: int = 0
    segment: int = 0
    note_kind: int = -1
    route_prefix: tuple = ()
    current_route_id: int = -1
    last_real_tick: int | None = None
    owner_z_beat: float | None = None


@dataclass(frozen=True)
class TargetRow:
    fields: dict
    query: FieldQuery
    moment_tick: int | None = None
    source_token_index: int | None = None


@dataclass(frozen=True)
class WindowTarget:
    rows: tuple[TargetRow, ...]
    moment_counts: tuple[int, ...]
    owner_z_beats: tuple[float, ...] | None = None

    @property
    def query_cost(self):
        return len(self.rows) + len(self.moment_counts)


def reconstruction_target(chart, *, start_tick, target_beats, audio_after_beats=None,
                          audio_end_seconds=None, validate_input=True):
    """Arrange valid queries; unused audio bounds remain accepted for shared callers.

    Draft has no duration, terminal, CONT or out-of-window targets.
    """
    if validate_input and (type(start_tick) is not int or type(target_beats) is not int or target_beats < 1):
        raise ValueError('R targets require an integer start tick and positive beat length')
    end_tick = start_tick + 96 * target_beats
    rows, counts = [], [0] * target_beats
    earlier = [g.tick for g in chart.groups if g.tick < start_tick]
    last_real = earlier[-1] if earlier else None
    for group in chart.groups:
        if not start_tick <= group.tick < end_tick:
            continue
        owner = group.tick // 96
        window_beat = (group.tick-start_tick) // 96
        ordinal = counts[window_beat]
        counts[window_beat] += 1
        base = dict(beat=owner+.5, ordinal=ordinal)
        # Current u never enters this query. Previous onset may precede the window.
        rows.append(TargetRow({}, FieldQuery('MOMENT', origin_tick=owner*96,
            last_real_tick=last_real, **base), group.tick-owner*96, group.moment_source_index))
        note_ordinal, track, routes, kind = -1, 0, [], -1
        for index, token in enumerate(group.tokens):
            source = group.source_indices[index] if group.source_indices else None
            if isinstance(token, Note):
                note_ordinal += 1
                track, routes, kind = 0, [], KINDS.index(token.kind)
                common = dict(origin_tick=group.tick, note_ordinal=note_ordinal, **base)
                rows.append(TargetRow({'kind': kind}, FieldQuery('NOTE_KIND', **common),
                                      source_token_index=source))
                flags = {name: int(getattr(token, name)) for name in APPLICABLE[token.kind]}
                if flags:
                    rows.append(TargetRow(flags, FieldQuery('NOTE_FLAGS', note_kind=kind, **common),
                                          source_token_index=source))
            elif isinstance(token, Route):
                common = dict(origin_tick=group.tick, note_ordinal=note_ordinal,
                    track=track, segment=len(routes), note_kind=kind, route_prefix=tuple(routes), **base)
                rows.append(TargetRow({'route_id': token.route_id}, FieldQuery('ROUTE_ID', **common),
                                      source_token_index=source))
                if token.track_break is not None:
                    rows.append(TargetRow({'track_break': int(token.track_break)},
                        FieldQuery('TRACK_BREAK', current_route_id=token.route_id, **common),
                        source_token_index=source))
                routes.append(token.route_id)
            elif isinstance(token, SlideDuration):
                # Keep the source separator even though its numeric value is absent.
                track += 1
                routes = []
        last_real = group.tick
    if validate_input and any(n > 96 for n in counts):
        raise ValueError('more than 96 distinct MOMENTs in a beat')
    return WindowTarget(tuple(rows), tuple(counts))


def overlap_reconstruction_target(chart, *, start_tick, target_beats, coverage_end_beat,
                                  first_beat=0, span_beats=4, validate_input=True):
    """Assign each integer owner once, retaining each distinct owner/real-query pair.

    A parent owns [start,start+target_beats); the final parent additionally owns
    the coverage endpoint. Labels come from the whole chart, including across
    parent boundaries, while each owner's responsibility is [owner-2,owner+2).
    """
    if span_beats != 4 or start_tick % 96:
        raise ValueError('overlapping integer reconstruction requires four complete beats')
    start = start_tick//96
    last_parent = start+target_beats == coverage_end_beat
    stop = start+target_beats+int(last_parent)
    owners = tuple(float(t) for t in range(start, stop))
    lower, upper = start-2, stop+1
    source = reconstruction_target(chart, start_tick=lower*96, target_beats=upper-lower,
                                   validate_input=validate_input)
    rows = {owner: [] for owner in owners}
    for row in source.rows:
        beat = int(row.query.beat//1)
        if not first_beat <= beat < coverage_end_beat:
            continue
        for owner in range(max(start, beat-1), min(stop, beat+3)):
            rows[float(owner)].append(replace(row, query=replace(row.query, owner_z_beat=float(owner))))
    counts = source.moment_counts[2:2+target_beats]
    return WindowTarget(tuple(row for owner in owners for row in rows[owner]), tuple(counts), owners)


def required_query_rows(target_lengths):
    """Inspect real query cost; an empty window legitimately has zero queries."""
    lengths = tuple(target_lengths)
    if not lengths or any(type(n) is not int or n < 0 for n in lengths):
        raise ValueError('nonnegative query counts required')
    return max(lengths)


def collate_reconstruction(targets, *, target_beats, device, query_rows=None, validate_input=True):
    import torch
    from .types import ReconstructionQueries, ReconstructionTargets
    if not targets:
        raise ValueError('a nonempty reconstruction batch is required')
    # One inert tensor slot makes an all-empty batch shape stable, never supervised.
    width = max(1, max((len(t.rows) for t in targets if t is not None), default=0))
    shape = (len(targets), width)
    fields = {n: torch.full(shape, -100, dtype=torch.long, device=device) for n in FIELD_SIZES}
    moment = torch.full(shape, -100, dtype=torch.long, device=device)
    counts = torch.zeros((len(targets), target_beats), dtype=torch.long, device=device)
    valid = torch.tensor([t is not None for t in targets], device=device, dtype=torch.bool)
    max_chain = max(1, max((len(r.query.route_prefix) for t in targets if t for r in t.rows), default=0))
    integer = lambda: torch.zeros(shape, dtype=torch.long, device=device)
    real = lambda: torch.zeros(shape, device=device)
    boolean = lambda: torch.zeros(shape, dtype=torch.bool, device=device)
    q = ReconstructionQueries(integer(), real(), real(), integer(), integer(),
        torch.full(shape, -1, dtype=torch.long, device=device),
        torch.zeros((*shape, max_chain), dtype=torch.long, device=device),
        torch.zeros((*shape, max_chain), dtype=torch.bool, device=device),
        boolean(), real(), boolean(), integer(), integer(),
        torch.full(shape, -1, dtype=torch.long, device=device))
    overlapping = any(t is not None and t.owner_z_beats is not None for t in targets)
    if overlapping:
        q.owner_z_beats = real()
    for w, target in enumerate(targets):
        if target is None:
            continue
        counts[w] = torch.tensor(target.moment_counts, device=device)
        for i, row in enumerate(target.rows):
            query = row.query
            for name, value in row.fields.items():
                fields[name][w, i] = value
            if row.moment_tick is not None:
                moment[w, i] = row.moment_tick
            q.roles[w, i] = QUERY_ROLES.index(query.role)
            q.beats[w, i], q.origins[w, i] = query.beat, query.origin_tick/96
            q.ordinals[w, i], q.note_ordinals[w, i] = query.ordinal, query.note_ordinal
            q.tracks[w, i], q.segments[w, i] = query.track, query.segment
            q.note_kinds[w, i], q.current_route_ids[w, i] = query.note_kind, query.current_route_id
            q.valid[w, i] = True
            if overlapping:
                if query.owner_z_beat is None:
                    raise ValueError('overlap reconstruction rows require an explicit Z owner')
                q.owner_z_beats[w, i] = query.owner_z_beat
            # Signed previous-onset position relative to this beat's start.
            q.history_gaps[w, i] = 0 if query.last_real_tick is None else (query.last_real_tick-query.origin_tick)/96
            q.last_real_valid[w, i] = query.last_real_tick is not None
            if query.route_prefix:
                n = len(query.route_prefix)
                q.route_ids[w, i, :n] = torch.tensor(query.route_prefix, device=device)
                q.route_valid[w, i, :n] = True
    return ReconstructionTargets(fields, moment, counts, valid, q)


def planner_target(planner, start_tick, end_tick, *, span_beats=16, layout='legacy',
                   last_parent=False, stride_beats=4):
    """Keep the original scalar target or recompute each exact four-beat target."""
    if type(span_beats) is not int or span_beats not in (4, 16):
        raise ValueError('Planner span must be 4 or 16 beats')
    if layout == 'centered-overlap':
        if span_beats != 16 or stride_beats != 4 or end_tick-start_tick != 16*96:
            raise ValueError('centered-overlap Planner requires span16, stride4 and a sixteen-beat parent')
        return [planner.window(center-8*96, center+8*96) if index < 4 or last_parent else None
                for index, center in enumerate(range(start_tick, end_tick+1, stride_beats*96))]
    if layout != 'legacy':
        raise ValueError('unsupported Planner layout')
    if span_beats == 16:
        return planner.window(start_tick, end_tick)
    if end_tick <= start_tick or (end_tick - start_tick) % (4 * 96):
        raise ValueError('Four-beat Planner target interval must contain whole four-beat segments')
    return [planner.window(start, start + 4 * 96) for start in range(start_tick, end_tick, 4 * 96)]


def collate_planner(targets, *, device, validate_input=True):
    import torch
    from .types import PlannerTargets
    if validate_input and not targets:
        raise ValueError('empty Planner target batch')
    segmented = [t for t in targets if isinstance(t, (list, tuple))]
    if segmented:
        width = len(segmented[0])
        if width < 1 or any(t is not None and (not isinstance(t, (list, tuple)) or len(t) != width) for t in targets):
            raise ValueError('Planner target segments must have a common positive length')
        rows = [list(t) if t is not None else [None] * width for t in targets]
        return PlannerTargets(
            torch.tensor([[t.continuous if t is not None else (0.,) * 8 for t in row] for row in rows],
                         device=device, dtype=torch.float32),
            torch.tensor([[t.category if t is not None else 0 for t in row] for row in rows],
                         device=device, dtype=torch.long),
            torch.tensor([[t is not None for t in row] for row in rows], device=device, dtype=torch.bool))
    return PlannerTargets(
        torch.tensor([t.continuous if t is not None else (0.,) * 8 for t in targets], device=device, dtype=torch.float32),
        torch.tensor([t.category if t is not None else 0 for t in targets], device=device, dtype=torch.long),
        torch.tensor([t is not None for t in targets], device=device, dtype=torch.bool))
