"""Chart IR/structured-token conversion, without Physical filtering."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math

from maidata.time import TimingMap, duration_end, split_duration
from .types import (ChartError, Moment, Note, Route, HoldDuration,
                    SlideDuration, APPLICABLE)
from .grammar import Grammar
from .routes import route_for, chain_paths, mirror_pad, mirror_route, path_for


@dataclass(frozen=True)
class EncodedChart:
    tokens: tuple
    metadata: dict
    start_tick: int = 0

    @property
    def timing(self):
        return TimingMap.from_ir(self.metadata)


def _chart_tick(value, *, position=False):
    if not position and value.get('basis') != 'beat':
        raise ChartError('absolute duration is not represented by written-BPM ticks')
    n, d = ('num', 'den') if position else ('beat_num', 'beat_den')
    try:
        if type(value[n]) is not int or type(value[d]) is not int or value[d] <= 0:
            raise ValueError()
        tick = Fraction(value[n], value[d])*96
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise ChartError('invalid rational beat value') from e
    if tick.denominator != 1:
        raise ChartError('beat value is not on the 96-tick chart grid')
    if not position and tick < 0:
        raise ChartError('negative duration')
    return int(tick)


def _flags(values, allowed):
    values = set(values)
    unknown = values-set(allowed)-{'each'}
    if unknown:
        raise ChartError(f'unsupported flags: {sorted(unknown)}')
    return {('break_' if n == 'break' else n): int(n in values) for n in allowed}


def encode_chart(ir):
    if ir.get('schema') != 'maimai.chart_ir.v1' or ir.get('tick_rate') != 180:
        raise ChartError('expected chart_ir.v1; its ticks are 180 Hz projections, not beat ticks')
    timing = TimingMap.from_ir(ir)
    onset_ticks = [_chart_tick(m['position_beats'], position=True) for m in ir['moments']]
    if any(a >= b for a, b in zip(onset_ticks, onset_ticks[1:])):
        raise ChartError('IR moments must be strictly increasing')
    cursor = min(0, onset_ticks[0]) if onset_ticks else 0
    start_tick = cursor
    tokens = [Moment(0)]
    for m, tick in zip(ir['moments'], onset_ticks):
        if tick != cursor:
            tokens.append(Moment(timing.delta_ms(cursor, tick)))
        cursor = tick
        notes = [n for g in m['groups'] for n in g['notes']]
        if not notes:
            raise ChartError('empty real MOMENT')
        for n in notes:
            kind = n['kind']
            if kind not in (*APPLICABLE, 'wifi'):
                raise ChartError(f'unknown kind: {kind}')
            if kind in ('slide', 'wifi'):
                if n['head']['kind'] != 'star':
                    raise ChartError('only the default star head is represented')
                _flags(n.get('flags', ()), ())
                flags = _flags(n['head'].get('flags', ()), ('ex', 'break'))
                if _chart_tick(n['wait']) != 96:
                    raise ChartError('only the default one-global-beat wait is represented')
                tokens.append(Note.make('slide', n['start_pad'], **flags))
                if not n['branches']:
                    raise ChartError('slide has no tracks')
                for branch in n['branches']:
                    if branch['timing_mode'] != 'total':
                        raise ChartError('per-segment timing is outside the official total-duration representation')
                    track_break = _flags(branch.get('flags', ()), ('break',))['break_']
                    if not branch['segments']:
                        raise ChartError('empty track')
                    start = n['start_pad']
                    for i, segment in enumerate(branch['segments']):
                        path = segment['path']
                        route = route_for(path)
                        if path_for(start, route) != path or segment['end_pad'] != 'A'+path[-1]:
                            raise ChartError('disconnected route or inconsistent end_pad')
                        tokens.append(Route(route, track_break if i == 0 else None))
                        start = segment['end_pad']
                    tokens.append(SlideDuration(_chart_tick(branch['duration'])))
            else:
                allowed = ('ex', 'break') if kind in ('tap', 'hold') else ('firework',)
                flags = _flags(n.get('flags', ()), allowed)
                duration = _chart_tick(n['duration']) if kind in ('hold', 'touch_hold') else None
                if duration is not None and duration <= 1:
                    flag = 'short_hold' if kind == 'hold' else 'short_touch_hold'
                    kind = 'tap' if kind == 'hold' else 'touch'
                    flags[flag] = 1
                tokens.append(Note.make(kind, n['pad'], **flags))
                if duration is not None and duration >= 2:
                    tokens.append(HoldDuration(duration))
    g = Grammar(timing)
    g.inject_boundary(tokens[0], start_tick=start_tick)
    for t in tokens[1:]:
        g.accept(t)
    g.finish()
    return EncodedChart(tuple(tokens), deepcopy({k: v for k, v in ir.items() if k != 'moments'}), start_tick)


def _ir_tick(second):
    return math.floor(second*180+.5)


def _duration(ticks, seconds, minimum=0):
    beats = Fraction(ticks, 96)
    return dict(basis='beat', beat_num=beats.numerator, beat_den=beats.denominator,
                resolved_ticks=max(minimum, _ir_tick(seconds)), seconds=seconds)


def _position(tick):
    b = Fraction(tick, 96)
    return dict(num=b.numerator, den=b.denominator)


def decode_chart(encoded, geometry):
    timing, result = encoded.timing, deepcopy(encoded.metadata)
    g, moments, owners = Grammar(timing), {}, {}
    g.inject_boundary(encoded.tokens[0], start_tick=encoded.start_tick)
    for token in encoded.tokens[1:]:
        c = g.accept(token)
        if c is None:
            continue
        second = timing.second(c.tick)
        if c.tick not in moments:
            meter = result.get('default_meter', {'numerator': 4, 'denominator': 4})
            for control in result['controls']:
                if control['kind'] == 'meter' and Fraction(control['position_beats']['num'], control['position_beats']['den']) <= Fraction(c.tick, 96):
                    meter = {'numerator': control['numerator'], 'denominator': control['denominator']}
            moments[c.tick] = dict(id=f'moment-{c.moment}', position_beats=_position(c.tick),
                                   second=second, tick=_ir_tick(second), bpm=timing.bpm_at_beat(c.tick/96),
                                   meter=deepcopy(meter), groups=[dict(each=False, notes=[])])
        if c.owner not in owners:
            n = c.note
            kind = 'hold' if n.short_hold else 'touch_hold' if n.short_touch_hold else 'wifi' if c.track and c.track.routes == (70,) else n.kind
            flags = [name for name, v in (('ex', n.ex), ('break', n.break_), ('firework', n.firework)) if v]
            note = dict(id=f'note-{c.owner}', kind=kind, start_tick=_ir_tick(second), flags=flags)
            if c.track is None:
                note['pad'] = n.pad
                d = 1 if n.short_hold or n.short_touch_hold else c.duration_tick
                if d is not None:
                    end = duration_end(timing, c.tick, d)
                    note.update(duration=_duration(d, end-second), end_tick=_ir_tick(end))
            else:
                launch = timing.second(c.tick+96)
                note.update(start_pad=n.pad, flags=[], head=dict(kind='star', flags=flags),
                            wait=_duration(96, launch-second), body_start_tick=_ir_tick(launch),
                            end_tick=_ir_tick(launch), branches=[])
            owners[c.owner] = note
            moments[c.tick]['groups'][0]['notes'].append(note)
        if c.track is not None:
            note = owners[c.owner]
            launch = timing.second(c.tick+96)
            end = duration_end(timing, c.tick, c.track.duration_tick, slide=True)
            paths = chain_paths(c.note.pad, c.track.routes)
            bounds = split_duration(launch, end, [geometry.paths[p].length for p in paths])
            segments = [dict(path=p, end_pad='A'+p[-1],
                             duration=dict(basis='tick', resolved_ticks=max(0, _ir_tick(b)-_ir_tick(a)), seconds=b-a))
                        for p, (a, b) in zip(paths, bounds)]
            branch = dict(timing_mode='total', flags=['break'] if c.track.track_break else [],
                          duration=_duration(c.track.duration_tick, end-launch, 1), segments=segments)
            note['branches'].append(branch)
            note['end_tick'] = max(note['end_tick'], _ir_tick(end), note['body_start_tick']+1)
    g.finish()
    result['moments'] = list(moments.values())
    for moment in result['moments']:
        group = moment['groups'][0]
        group['each'] = len(group['notes']) >= 2
        if group['each']:
            for note in group['notes']:
                target = note['head'] if 'head' in note else note
                target['flags'] = sorted([*target['flags'], 'each'])
    return result


def token_to_dict(token):
    values = asdict(token)
    if 'break_' in values:
        values['break'] = values.pop('break_')
    return {'type': token.type, **values}


def token_from_dict(record):
    values = dict(record)
    kind = values.pop('type', None)
    classes = {c.type: c for c in (Moment, Note, Route, HoldDuration, SlideDuration)}
    if kind not in classes:
        raise ChartError(f'unknown token type: {kind}')
    if 'break' in values:
        values['break_'] = values.pop('break')
    try:
        return classes[kind](**values)
    except TypeError as e:
        raise ChartError(f'invalid {kind} fields: {e}') from e


def mirror_tokens(tokens):
    result, start, owner_start = [], None, None
    for t in tokens:
        if isinstance(t, Note):
            owner_start = start = t.pad
            result.append(replace(t, pad=mirror_pad(t.pad)))
        elif isinstance(t, Route):
            if t.track_break is not None:
                start = owner_start
            if start is None:
                raise ChartError('orphan ROUTE')
            result.append(replace(t, route_id=mirror_route(start, t.route_id)))
            start = 'A'+path_for(start, t.route_id)[-1]
        else:
            result.append(t)
    return tuple(result)
