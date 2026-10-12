"""Pure Python simai spelling, shared by IR export and committed inference groups."""
from fractions import Fraction
from copy import copy
import math
import re

from .simai import bounded_render_document, compact_spacing, four_beat_lines, persistent_subdivisions


def position(value):
    return Fraction(value['num'], value['den'])


def decimal(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('simai number must be finite')
    return format(value, '.17g')


def pad_text(pad):
    return pad[1:] if re.fullmatch(r'A[1-8]', pad) else pad


def flags_text(flags, *, firework=False, tap=False):
    flags = set(flags)
    text = ''.join(symbol for name, symbol in [('break', 'b'), ('ex', 'x')]
                   if name in flags)
    if firework and 'firework' in flags:
        text += 'f'
    if tap:
        text += '$$' if 'star_material_double' in flags else '$' if 'star_material' in flags else ''
    return text + ('m' if 'mine' in flags else '')


def beat_signature(beats):
    whole = beats / 4
    return f'{whole.denominator}:{whole.numerator}'


def duration_signature(duration, *, slide=False):
    if duration['basis'] == 'beat':
        beats = Fraction(duration['beat_num'], duration['beat_den'])
        if beats < 0 or slide and not beats:
            raise ValueError('invalid beat duration')
        return beat_signature(beats)
    if duration['basis'] == 'tick':
        ticks = duration['resolved_ticks']
        if ticks < 0 or slide and ticks == 0:
            raise ValueError('invalid tick duration')
        return '#' + decimal(duration.get('seconds', ticks / 180))
    raise ValueError('unknown duration basis')


def slide_signature(wait, duration):
    if (wait['basis'] == 'beat' and Fraction(wait['beat_num'], wait['beat_den']) == 1
            and duration['basis'] == 'beat'):
        return duration_signature(duration, slide=True)
    wt, dt = wait['resolved_ticks'], duration['resolved_ticks']
    if wt < 0 or dt <= 0:
        raise ValueError('invalid explicit slide wait/duration')
    return decimal(wait.get('seconds', wt / 180)) + '##' + decimal(duration.get('seconds', dt / 180))


def render_note(note):
    kind = note['kind']
    if kind in ('tap', 'touch', 'hold', 'touch_hold'):
        touch = kind in ('touch', 'touch_hold')
        text = (note['pad'] if touch else pad_text(note['pad']))
        text += flags_text(note.get('flags', ()), firework=touch, tap=kind == 'tap')
        if kind in ('hold', 'touch_hold'):
            text += 'h[' + duration_signature(note['duration']) + ']'
        return text
    if kind not in ('slide', 'wifi'):
        raise ValueError(f'unsupported note kind: {kind}')
    start = pad_text(note['start_pad'])
    head = note['head']
    heads = {'star': '', 'tap': '@', 'none_fade': '?', 'none_immediate': '!'}
    text = start + flags_text(head.get('flags', ())) + heads[head['kind']]
    branches = []
    for branch in note['branches']:
        body, expected = '', start
        per_segment = branch['timing_mode'] == 'per_segment'
        for i, segment in enumerate(branch['segments']):
            path = segment['path']
            if (not re.fullmatch(r'[1-8](?:[-^v<>szw][1-8]|V[1-8][1-8]|p{1,2}[1-8]|q{1,2}[1-8])', path)
                    or path[0] != expected or pad_text(segment['end_pad']) != path[-1]):
                raise ValueError(f'invalid slide path: {path}')
            body += path[1:]
            expected = path[-1]
            if per_segment:
                wait = note['wait'] if i == 0 else dict(basis='tick', resolved_ticks=0)
                duration = segment['duration']
                signature = (slide_signature(wait, duration) if duration['basis'] == 'tick'
                             else duration_signature(duration, slide=True))
                body += '[' + signature + ']'
        if not branch['segments']:
            raise ValueError('slide branch has no segment')
        body += ''.join(symbol for name, symbol in [('break', 'b'), ('mine', 'm')]
                        if name in branch.get('flags', ()))
        if not per_segment:
            body += '[' + slide_signature(note['wait'], branch['duration']) + ']'
        branches.append(body)
    if not branches:
        raise ValueError('slide has no branch')
    return text + '*'.join(branches)


def render_group(group):
    """Spell completed generated fields directly; no Chart IR or geometry projection."""
    from maidata.chart.types import Note, Route, HoldDuration, SlideDuration
    from maidata.chart.routes import path_for
    notes, text, start, path_start, routes, track_break = [], '', None, None, '', 0
    branches = []

    def finish_note():
        if text:
            notes.append(text + '*'.join(branches))

    for token in group.content:
        if isinstance(token, Note):
            finish_note()
            branches, routes = [], ''
            start = token.pad
            flags = [name for name, value in [('break', token.break_), ('ex', token.ex),
                                             ('firework', token.firework)] if value]
            touch = token.kind in ('touch', 'touch_hold')
            text = (start if touch else pad_text(start)) + flags_text(flags, firework=touch)
            if token.short_hold or token.short_touch_hold:
                text += 'h[' + beat_signature(Fraction(1, 96)) + ']'
        elif isinstance(token, HoldDuration):
            text += 'h[' + beat_signature(Fraction(token.ticks, 96)) + ']'
        elif isinstance(token, Route):
            if token.track_break is not None:
                path_start, routes, track_break = start, '', token.track_break
            path = path_for(path_start, token.route_id)
            routes += path[1:]
            path_start = 'A' + path[-1]
        elif isinstance(token, SlideDuration):
            branches.append(routes + ('b' if track_break else '') + '['
                            + beat_signature(Fraction(token.ticks, 96)) + ']')
        else:
            raise ValueError(f'unexpected onset content: {token}')
    finish_note()
    return '/'.join(notes)


def control_text(control):
    kind = control['kind']
    if kind == 'bpm':
        return '(' + decimal(control['bpm']) + ')'
    if kind == 'meter':
        return f"|| {control['numerator']} / {control['denominator']}\n"
    if kind == 'hs':
        return '<HS*' + decimal(control['multiplier']) + '>'
    raise ValueError(f'unsupported control kind: {kind}')


class SimaiWriter:
    """Delay only the last position until its following delta is known."""
    def __init__(self, controls=(), end=None):
        self.controls = sorted(((position(c['position_beats']), control_text(c)) for c in controls),
                               key=lambda item: item[0])
        self.control_index = 0
        self.end = end
        self.pending = (Fraction(0), '')
        self.parts = []
        self.moments = 0

    def _add(self, beat, body):
        previous, text = self.pending
        if beat < previous or beat < 0 or (beat * 96).denominator != 1:
            raise ValueError('simai positions must increase on the nonnegative 1/96-beat grid')
        if beat == previous:
            self.pending = beat, text + body
            return
        delta = beat - previous
        self.parts.append(f'{{{4 * delta.denominator}}}' + text + ',' * delta.numerator)
        self.pending = beat, body

    def add(self, beat, body):
        while self.control_index < len(self.controls) and self.controls[self.control_index][0] <= beat:
            self._add(*self.controls[self.control_index])
            self.control_index += 1
        self._add(beat, body)
        self.moments += 1

    def add_groups(self, groups):
        for group in groups:
            self.add(Fraction(group.tick, 96), render_group(group))

    def preview(self, groups=()):
        """Render a temporary tail without consuming committed text or controls."""
        writer = copy(self)
        writer.parts = self.parts.copy()
        writer.add_groups(groups)
        writer.end = writer.pending[0] + Fraction(1, 96)
        writer.controls = [c for c in writer.controls if c[0] < writer.end]
        return writer.finish()

    def finish(self):
        for control in self.controls[self.control_index:]:
            self._add(*control)
        end = self.end if self.end is not None else self.pending[0] + 1
        if end <= self.pending[0] or end <= 0:
            raise ValueError('render_end_beat must be positive and after all controls and moments')
        self._add(end, 'E')
        raw = ''.join(self.parts) + 'E'
        return persistent_subdivisions(four_beat_lines(compact_spacing(raw)))


def render_document(document):
    if document.get('schema') != 'maimai.chart_ir.v1':
        raise ValueError('unsupported chart IR schema')
    document = bounded_render_document(document)
    end = position(document['render_end_beat']) if 'render_end_beat' in document else None
    writer = SimaiWriter(document.get('controls', ()), end)
    previous = None
    for moment in document['moments']:
        beat = position(moment['position_beats'])
        if previous is not None and beat <= previous:
            raise ValueError('IR moments must be strictly increasing')
        previous = beat
        groups = ['/'.join(render_note(n) for n in g['notes']) for g in moment['groups'] if g['notes']]
        writer.add(beat, '`'.join(groups))
    return writer.finish()
