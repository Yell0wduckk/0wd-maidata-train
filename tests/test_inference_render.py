"""Independent native parsing is a development test, never an inference stage."""
from fractions import Fraction
import json
import subprocess

import pytest

from test_chart_fields import geometry, complex_tokens, metadata
from test_inference_materialize import native_miacode, check_roundtrip
from maidata.chart import EncodedChart, Moment, Note, Route, SlideDuration, decode_chart
from maidata.decoder.windows import real_onsets
from maidata.inference.__main__ import ir_metadata
from maidata.inference.materialize import render_origin
from maidata.inference.render import SimaiWriter, render_document, render_span, position
from maidata.inference.simai import four_beat_lines, persistent_subdivisions
from maidata.time import TimingMap


def native_dump(executable, chart, first):
    result = subprocess.run([str(executable), 'dump', '--file', '-', '--first', str(first), '--output', '-'],
                            input=chart.encode('utf-8'), capture_output=True, check=True)
    return json.loads(result.stdout.decode('utf-8-sig'))


@pytest.mark.parametrize('ticks,expected', [([0,24,48], '{16}(120)1,1,1,'),
                                          ([0,72], '{16}(120)1,,,1,')])
def test_persistent_delta_examples(ticks, expected, geometry):
    timing = TimingMap([(0,120)])
    tokens = [Moment(0)]
    for i, tick in enumerate(ticks):
        if i:
            tokens.append(Moment(timing.delta_ms(ticks[i-1], tick)))
        tokens.append(Note.make('tap','A1'))
    ir = decode_chart(EncodedChart(tuple(tokens), metadata(timing)), geometry)
    end = Fraction(ticks[-1]+24,96)
    ir['render_end_beat'] = dict(num=end.numerator,den=end.denominator)
    assert render_document(ir) == expected+'E'


def test_brace_dedup_keeps_duration_text_and_line_start_subdivisions():
    assert persistent_subdivisions('{16}1[16:1],\n{16}(180)1,{8}2,{8}3,') == \
           '{16}1[16:1],\n{16}(180)1,{8}2,3,'


@pytest.mark.parametrize('start,end,expected', [
    (0, 384, '{4},,,,\n'),
    (384, 768, '{4},,,,\n'),
    (336, 432, '{8},\n{8},'),
    (1488, 1920, '{8},\n{8},{4},,,{96}' + ',' * 12 + '\n'),
])
def test_fragment_bar_end_is_a_newline_without_a_dangling_declaration(start, end, expected):
    text = render_span((), start, end)
    assert text == expected
    assert four_beat_lines(text, start_tick=start) == text


def test_consecutive_continuations_compose_into_four_beat_lines():
    from maidata.decoder.windows import Onset
    spans = [(0, 96), (96, 336), (336, 432), (432, 768), (768, 864)]
    text = ''.join(render_span((Onset(a, 0., (Note.make('tap', 'A1'),)),), a, b)
                   for a, b in spans)
    # A model span may start halfway through a line; its internal and trailing
    # newlines remain anchored to the whole chart, not to each request.
    tick = 0
    import re
    for token in re.findall(r'\{\d+\}|[,\n]', text):
        if token.startswith('{'):
            step = 384 // int(token[1:-1])
        elif token == ',':
            tick += step
        else:
            assert tick in (384, 768)
    assert text.count('\n') == 2 and tick == 864


@pytest.mark.parametrize('source,expected', [
    ('{4}1,2,3,4,5,E', '{4}1,2,3,4,\n{4}5,E'),
    ('{8}1,,,,,,,{4}2,{8}3,E', '{8}1,,,,,,,2,\n{8},3,E'),
    ('{8}1,,,,,,,{4}2,3,E', '{8}1,,,,,,,2,\n{8},{4}3,E'),
    ('{8}1,,,,,,,{4}2,{4}{8}3,E', '{8}1,,,,,,,2,\n{8},3,E'),
    ('{4}1,2,3,4,E', '{4}1,2,3,4,\n{4}E'),
    ('{4}(120)|| 3 / 4\n<HS*1.25>1,2,E',
     '{4}(120)|| 3 / 4\n{4}<HS*1.25>1,2,E'),
])
def test_four_beat_layout_has_explicit_line_starts_without_stale_restores(source, expected):
    chart = persistent_subdivisions(four_beat_lines(source))
    assert chart == expected
    assert four_beat_lines(chart) == chart


@pytest.mark.parametrize('offset', [0, 1, 24, 48, 95, 96, 144, 192, 240, 288, 336, 360, 383])
@pytest.mark.parametrize('division', [n for n in range(1, 385) if 384 % n == 0])
def test_interval_splitting_preserves_time_and_exact_bar_boundaries(offset, division):
    # Each note has one original interval; split commas must add no onset.
    import re
    source = f'{{{division}}}1,2,3,E'
    chart = persistent_subdivisions(four_beat_lines(source, start_tick=offset))
    tick, active, notes = offset, None, []
    for token in re.findall(r'\{\d+\}|[123E,\n]', chart):
        if token.startswith('{'):
            active = int(token[1:-1])
        elif token == ',':
            tick += 384 // active
        elif token == '\n':
            assert tick % 384 == 0
        else:
            notes.append((token, tick))
    step = 384 // division
    assert notes == [(note, offset+i*step) for i, note in enumerate('123E')]
    assert all(re.match(r'^\{\d+\}', line) for line in chart.splitlines())
    assert not re.search(r'\{\d+\}\{\d+\}', chart)


def test_cross_bar_layout_preserves_native_note_and_control_semantics(native_miacode):
    source = ('{8}(120)1h[4:1],,,,,,,{4}<HS*1.25>2-4[8:1]*-6[8:1],'
              '{8}(180)3,4,E')
    chart = persistent_subdivisions(four_beat_lines(source))
    check_roundtrip(native_dump(native_miacode, source, .125), native_dump(native_miacode, chart, .125))
    assert '\n{8},' in chart
    assert '{4}{8}' not in chart


@pytest.mark.parametrize('case',['complex','empty','negative','wait_across_bpm','long_spacing'])
def test_python_render_native_semantics(case, geometry, native_miacode):
    timing = TimingMap([(0,120),(Fraction(3,2),240)],.125)
    tokens, start = complex_tokens(timing), 0
    if case == 'empty':
        tokens = (Moment(0),)
    elif case == 'negative':
        tokens, start = (Moment(0),Note.make('tap','A1'),
                        Moment(timing.delta_ms(-12,0)),Note.make('touch','C')), -12
    elif case == 'wait_across_bpm':
        tokens = (Moment(0),Moment(timing.delta_ms(0,96)),Note.make('slide','A1'),Route(0,1),SlideDuration(97))
    elif case == 'long_spacing':
        tokens = (Moment(0),Note.make('tap','A1'),Moment(timing.delta_ms(0,1973)),Note.make('tap','A2'))
    ir = render_origin(decode_chart(EncodedChart(tokens,ir_metadata(timing,20.),start),geometry))
    chart = render_document(ir)
    check_roundtrip(ir, native_dump(native_miacode, chart, ir['first_seconds']))


@pytest.mark.parametrize('first', [.02,0.,-1.25])
def test_incremental_groups_match_ir_render(first, geometry, native_miacode):
    timing = TimingMap([(0,120),(Fraction(3,2),240)], first)
    encoded = EncodedChart(complex_tokens(timing),ir_metadata(timing,20.))
    ir = decode_chart(encoded,geometry)
    writer = SimaiWriter(ir['controls'],position(ir['render_end_beat']))
    groups = list(real_onsets(encoded.tokens,timing,sequence_start_tick=0))
    for group in groups:
        writer.add_groups([group])
    chart = writer.finish()
    assert chart == render_document(ir)
    assert writer.moments == len(groups)
    check_roundtrip(ir,native_dump(native_miacode,chart,first))


def test_all_route_variants_starts_and_track_breaks(geometry,native_miacode):
    timing = TimingMap([(0,137)])
    tokens = [Moment(0)]
    for i in range(71*8):
        if i:
            tokens.append(Moment(timing.delta_ms((i-1)*96,i*96)))
        tokens.extend((Note.make('slide',f'A{i%8+1}',ex=i%2,break_=(i//2)%2),
                       Route(i//8,i%2),SlideDuration(1+i%385)))
    encoded = EncodedChart(tuple(tokens),ir_metadata(timing,300.))
    ir = decode_chart(encoded,geometry)
    writer = SimaiWriter(ir['controls'],position(ir['render_end_beat']))
    writer.add_groups(real_onsets(encoded.tokens,timing,sequence_start_tick=0))
    chart = writer.finish()
    assert chart == render_document(ir)
    check_roundtrip(ir,native_dump(native_miacode,chart,0.))


@pytest.mark.parametrize('case', ['controls', 'tick_duration', 'explicit_wait', 'per_segment'])
def test_retained_ir_spelling_matches_native_export(case,geometry,native_miacode):
    timing = TimingMap([(0,120)])
    ir = decode_chart(EncodedChart(complex_tokens(timing),ir_metadata(timing,10.)),geometry)
    notes = ir['moments'][0]['groups'][0]['notes']
    if case == 'controls':
        ir['controls'].extend([dict(kind='meter',numerator=3,denominator=4,position_beats=dict(num=1,den=1)),
                               dict(kind='hs',multiplier=1.25,position_beats=dict(num=1,den=1))])
    elif case == 'tick_duration':
        notes[2]['duration'] = dict(basis='tick',resolved_ticks=18,seconds=.1)
    elif case == 'explicit_wait':
        notes[4]['wait'] = dict(basis='tick',resolved_ticks=18,seconds=.1)
    else:
        notes[4]['branches'][0]['timing_mode'] = 'per_segment'
    native = subprocess.run([str(native_miacode),'render','--input','-','--output','-'],
        input=json.dumps(ir).encode('utf-8'),capture_output=True,check=True).stdout.decode('utf-8-sig')
    def semantic(value):
        if isinstance(value,dict):
            return {k:semantic(v) for k,v in value.items() if k not in ('id','source','source_signature')}
        if isinstance(value,list):
            return [semantic(v) for v in value]
        return value
    assert semantic(native_dump(native_miacode,render_document(ir),.0)) == \
           semantic(native_dump(native_miacode,native,.0))


@pytest.mark.parametrize('position_value', [dict(num=1,den=97),dict(num=-1,den=1)])
def test_reject_unrepresentable_positions(position_value,geometry):
    timing = TimingMap([(0,120)])
    ir = decode_chart(EncodedChart((Moment(0),Note.make('tap','A1')),metadata(timing)),geometry)
    ir['moments'][0]['position_beats'] = position_value
    with pytest.raises(ValueError):
        render_document(ir)
