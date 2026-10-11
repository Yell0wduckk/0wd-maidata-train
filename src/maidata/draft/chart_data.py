"""Read accepted original chart records into real onset groups, without Physical."""
from dataclasses import dataclass
from fractions import Fraction

from maidata.chart.codec import token_from_dict
from maidata.chart.types import Moment, Note
from maidata.time import TimingMap


@dataclass(frozen=True)
class OnsetGroup:
    tick: int
    tokens: tuple
    source_indices: tuple = ()
    moment_source_index: int | None = None


@dataclass(frozen=True)
class PlanNote:
    tick: int
    kind: str
    button: int | None


@dataclass(frozen=True)
class ChartData:
    groups: tuple[OnsetGroup, ...]
    notes: tuple[PlanNote, ...]
    timing: TimingMap
    source_end_beat: Fraction | None
    source_path: str | None = None
    difficulty: str | None = None


def from_record(record, *, source_end_beat=None, validate_input=True):
    """No Decoder window boundary/terminal is reused as a real R event."""
    if validate_input and (record['schema'] != 'maidata.full-chart-original.v2' or record['view'] != 'original'):
        raise ValueError('Draft labels require the current original chart representation')
    timing = TimingMap.from_ir(record['metadata'], validate_input=validate_input)
    if source_end_beat is None:
        end = record['metadata'].get('render_end_beat')
        if end is not None:
            source_end_beat = Fraction(end['num'], end['den'])
    if source_end_beat is not None:
        source_end_beat = Fraction(source_end_beat)
        if validate_input and (source_end_beat <= 0):
            raise ValueError('source end beat must be positive')
    if validate_input and (len(record['tokens']) != len(record['token_map'])):
        raise ValueError('chart token/ownership lengths differ')
    groups, notes, seen_owners = [], [], set()
    owner, onset, pending_moment = None, None, None
    for i, (raw, mapping) in enumerate(zip(record['tokens'], record['token_map'])):
        token = token_from_dict(raw, validate_input=validate_input)
        if validate_input and (mapping['token_index'] != i or type(mapping['tick']) is not int):
            raise ValueError('invalid token index/tick mapping')
        if isinstance(token, Moment):
            pending_moment = i
            continue
        tick = mapping['tick']
        if isinstance(token, Note):
            owner, onset = mapping['owner'], tick
            if validate_input and (type(owner) is not int or owner in seen_owners):
                raise ValueError('NOTE ownership must be unique')
            if validate_input:
                seen_owners.add(owner)
            if not groups or tick != groups[-1][0]:
                if validate_input and (groups and tick < groups[-1][0]):
                    raise ValueError('onset groups must be ordered')
                groups.append((tick, [], [], pending_moment))
            kind = ('hold' if token.short_hold else 'touch_hold' if token.short_touch_hold else token.kind)
            button = None if token.kind in ('touch', 'touch_hold') else int(token.pad[1:])
            notes.append(PlanNote(tick, kind, button))
        if validate_input and (owner is None or mapping['owner'] != owner or tick != onset):
            raise ValueError('ROUTE/duration must retain its NOTE owner and onset')
        groups[-1][1].append(token)
        groups[-1][2].append(i)
    source = record.get('source', {})
    return ChartData(tuple(OnsetGroup(tick, tuple(tokens), tuple(indices), moment) for tick, tokens, indices, moment in groups),
                     tuple(notes), timing, source_end_beat,
                     source.get('source_path'), source.get('difficulty'))
