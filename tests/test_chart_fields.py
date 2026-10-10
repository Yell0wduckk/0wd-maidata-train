from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import pytest

from maidata.chart import *
from maidata.chart.routes import catalog, path_for, route_for, mirror_pad, mirror_route
from maidata.chart.types import APPLICABLE, Completion, Track
from maidata.time import TimingMap, duration_end
from maidata.physical import Geometry, PathGeometry


@pytest.fixture
def timing():
    return TimingMap([(0, 120), (Fraction(3, 2), 240)], .125)


@pytest.fixture
def geometry():
    return Geometry({p: PathGeometry(p, float(i % 3+1), ((0., 1 << (int(p[0])-1)),))
                     for i, v in enumerate(catalog()) for p in v['paths_by_start']})


def metadata(timing):
    return dict(schema='maimai.chart_ir.v1', tick_rate=180, first_seconds=timing.first_seconds,
                default_meter=dict(numerator=4, denominator=4),
                controls=[dict(kind='bpm', bpm=bpm, position_beats=dict(num=b.numerator, den=b.denominator),
                               tick=round(timing.beat_to_second(b)*180)) for b, bpm in timing.points],
                eligibility=dict(training_eligible=True, isolation_reasons=[]))


def complex_tokens(timing):
    return (Moment(0), Note.make('tap', 'A1', ex=1, break_=1),
            Note.make('touch', 'D1', firework=1, short_touch_hold=1),
            Note.make('hold', 'A2', break_=1), HoldDuration(2),
            Note.make('touch_hold', 'C'), HoldDuration(196),
            Note.make('slide', 'A3', ex=1), Route(0, 1), Route(2), SlideDuration(97),
            Route(4, 0), Route(0), SlideDuration(121),
            Moment(timing.delta_ms(0, 192)), Note.make('slide', 'A8'), Route(70, 1), SlideDuration(1),
            Note.make('tap', 'A4', short_hold=1))


@pytest.mark.parametrize('kind,pad', [('tap','A1'),('hold','A8'),('touch','C'),('touch_hold','E8'),('slide','A3')])
def test_note_applicability_and_serialization(kind, pad):
    n = Note.make(kind, pad)
    assert token_from_dict(token_to_dict(n)) == n
    for field in APPLICABLE[kind]:
        assert getattr(n, field) == 0
        with pytest.raises(ChartError):
            replace(n, **{field: None})
    with pytest.raises(ChartError):
        Note.make(kind, 'K1')


def test_invalid_fields():
    for record in [dict(type='OOV'), dict(type='NOTE', kind='tap', pad='A1'),
                   dict(type='ROUTE', route_id=71), dict(type='MOMENT', delta_ms=float('nan')),
                   dict(type='HOLD_DURATION', ticks=1), dict(type='SLIDE_DURATION', ticks=0)]:
        with pytest.raises(ChartError):
            token_from_dict(record)
    assert Route(0, 0).route_id == 0
    with pytest.raises(ChartError):
        Note.make('touch', 'C', ex=0)


def test_all_routes_and_mirror():
    for v in catalog():
        for lane in range(1, 9):
            pad, r = f'A{lane}', v['variant_id']
            assert route_for(path_for(pad, r)) == r
            mirrored = mirror_route(pad, r)
            assert mirror_route(mirror_pad(pad), mirrored) == r
    assert [mirror_pad(x) for x in ('A1','D1','E2','C')] == ['A8','D1','E8','C']


def test_ir_round_trip_with_chains_branches_and_modifiers(timing, geometry):
    tokens = complex_tokens(timing)
    encoded = EncodedChart(tokens, metadata(timing))
    ir = decode_chart(encoded, geometry)
    # Independent retained IR validator; production code does not import it.
    from maimai_chart_ir.models import validate_chart_ir
    validate_chart_ir(ir)
    assert encode_chart(ir).tokens == tokens
    assert mirror_tokens(mirror_tokens(tokens)) == tokens
    slide = ir['moments'][0]['groups'][0]['notes'][4]
    assert slide['head']['flags'] == ['each', 'ex']
    assert [b['flags'] for b in slide['branches']] == [['break'], []]
    assert [s['path'][0] for s in slide['branches'][1]['segments']] == ['3', '1']
    assert ir['moments'][0]['groups'][0]['each']
    assert ir['moments'][0]['tick'] != 0  # first_seconds, not chart tick
    assert ir['moments'][1]['position_beats'] == dict(num=2, den=1)


@pytest.mark.parametrize('ticks', [0, 1, 2])
def test_short_normalization(timing, geometry, ticks):
    ir = decode_chart(EncodedChart((Moment(0), Note.make('hold','A1'), HoldDuration(2)), metadata(timing)), geometry)
    d = ir['moments'][0]['groups'][0]['notes'][0]['duration']
    b = Fraction(ticks,96)
    d.update(beat_num=b.numerator, beat_den=b.denominator)
    out = encode_chart(ir)
    if ticks <= 1:
        assert out.tokens[-1] == Note.make('tap','A1',short_hold=1)
        rebuilt = decode_chart(out, geometry)['moments'][0]['groups'][0]['notes'][0]['duration']
        assert Fraction(rebuilt['beat_num'],rebuilt['beat_den']) == Fraction(1,96)
    else:
        assert out.tokens[-1] == HoldDuration(2)


@pytest.mark.parametrize('change', ['timing','wait','head','path','grid'])
def test_unsupported_ir_is_not_silently_changed(timing, geometry, change):
    ir = decode_chart(EncodedChart(complex_tokens(timing), metadata(timing)), geometry)
    slide = ir['moments'][0]['groups'][0]['notes'][4]
    if change == 'timing': slide['branches'][0]['timing_mode'] = 'per_segment'
    if change == 'wait': slide['wait']['beat_num'] = 2
    if change == 'head': slide['head']['kind'] = 'none_fade'
    if change == 'path': slide['branches'][0]['segments'][1]['path'] = '1-3'
    if change == 'grid': ir['moments'][0]['position_beats'] = dict(num=1,den=7)
    before = deepcopy(ir)
    with pytest.raises(ChartError): encode_chart(ir)
    assert ir == before


def test_grammar_atomic_routes_and_ownership(timing):
    g = Grammar(timing)
    g.inject_boundary(Moment(0), start_tick=0)
    for t in ( Note.make('slide','A1'), Route(0,1)):
        g.accept(t)
    saved = g.snapshot()
    with pytest.raises(ChartError): g.accept(Route(2,0))
    assert saved == g.snapshot()
    with pytest.raises(ChartError): g.accept(Moment(1))
    g.accept(Route(2))
    first = g.accept(SlideDuration(96))
    g.accept(Route(4,0))
    second = g.accept(SlideDuration(192))
    assert (first.owner, second.owner, first.track_index, second.track_index) == (0,0,0,1)
    assert first.track.track_break == 1 and second.track.track_break == 0
    g.restore(saved)
    assert g.accept(SlideDuration(48)).track.routes == (0,)


def test_wifi_and_unbounded_data_chains(timing):
    g = Grammar(timing)
    g.inject_boundary(Moment(0), start_tick=0)
    for t in (Note.make('slide','A1'),Route(70,0)):
        g.accept(t)
    with pytest.raises(ChartError): g.accept(Route(0))
    g.accept(SlideDuration(96))
    with pytest.raises(ChartError): g.accept(Route(0,0))
    g.accept(Note.make('slide','A3'))
    # Field syntax does not impose the Physical two-track limit or a chain cap.
    for branch in range(4):
        for i in range(40): g.accept(Route(0,0 if i == 0 else None))
        assert len(g.accept(SlideDuration(96)).track.routes) == 40


def test_timing_written_bpm_and_global_wait(timing):
    assert timing.second(192) == pytest.approx(1.)
    assert timing.tick_after(96,timing.delta_ms(96,192)) == 192
    assert duration_end(timing,96,192) == pytest.approx(1.625)
    assert duration_end(timing,96,192,slide=True) == pytest.approx(2.)


def test_windows_exact_and_censored_terminal(timing):
    tokens = (Moment(0),Note.make('hold','A1'),HoldDuration(1000),
              Moment(timing.delta_ms(0,192)),Note.make('tap','A2'),
              Moment(timing.delta_ms(192,384)),Note.make('tap','A3'))
    w = window_sequence(tokens,timing,start_tick=0,end_tick=192,audio_after_beats=2,audio_end_seconds=10)
    assert w.tokens[-1] == HoldDuration(1000)  # crosses window without truncation
    assert w.terminal == TimeTarget(timing.delta_ms(0,192),192,False,True)
    w = window_sequence(tokens,timing,start_tick=193,end_tick=288,audio_after_beats=1,audio_end_seconds=10)
    assert w.tokens == (Moment(timing.delta_ms(192,193)),)
    assert w.terminal.censored  # exact onset at U is part of the unobserved tail
    assert w.terminal.absolute_tick == pytest.approx(384)
    w = window_sequence(tokens,timing,start_tick=480,end_tick=600,audio_after_beats=4,audio_end_seconds=timing.second(610.5))
    assert w.terminal.censored and w.terminal.absolute_tick == pytest.approx(984)


def test_terminal_does_not_replace_real_cursor(timing):
    g = Grammar(timing)
    g.inject_boundary(Moment(0), start_tick=0, window_end=96)
    g.accept(Note.make('tap','A1'))
    g.accept(Moment(timing.delta_ms(0,192)))
    assert g.state.phase == 'closed' and g.state.last_real_tick == 0
    assert g.state.cursor == 0
    g.inject_boundary(Moment(timing.delta_ms(0,96)), start_tick=96, window_end=192)
    assert g.state.last_real_tick == 0
    with pytest.raises(ChartError): g.accept(Moment(0))
    g.accept(Note.make('tap','A2'))
    assert g.state.last_real_tick == 96


def test_censored_terminal_closes_without_fabricating_a_grid_onset(timing):
    g = Grammar(timing)
    g.inject_boundary(Moment(0), start_tick=0, window_end=96)
    target=TimeTarget(timing.delta_ms(0,80.5),80.5,True,True)
    g.close_window(target)
    assert g.state.last_real_tick is None and g.state.cursor == 0
    assert g.state.token_count == 1 and g.state.phase == 'closed'


def test_data_duration_cells_match_retained_math(timing):
    import torch
    from maidata.time import duration_cells, duration_target_bucket
    from maimai_baseline.time.duration import candidate_ticks, tick_bounds
    for kind, field, lower in [('HOLD_DURATION','hold_duration_tick',2),('SLIDE_DURATION','branch_duration_tick',1)]:
        for upper in (lower,3,5,10,97):
            cells=duration_cells(kind,lower,upper,timing=timing,onset_tick=96)
            buckets,ticks=candidate_ticks(field=field,lower=lower,upper=upper,device='cpu')
            q=(buckets*4-2).clamp(lower,upper)
            bounds=tick_bounds(field,torch.full_like(q,96),q,q,torch.full_like(q,lower),torch.full_like(q,upper),
                               tick_seconds=torch.full(q.shape,timing.written_tick_seconds(96),dtype=torch.float64))
            assert [c.tick for c in cells]==ticks.tolist()
            for c,b in zip(cells,bounds):
                assert (c.lower,c.upper)==pytest.approx(tuple(b[:2].tolist()))
            assert duration_target_bucket(lower,lower,upper)==cells[0].bucket


def test_data_modules_import_without_torch_or_old_packages():
    import os, subprocess, sys
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'src'
    script="import maidata.chart, maidata.time, maidata.physical, sys; assert not any(x == 'torch' or x.startswith('maimai_') for x in sys.modules)"
    result=subprocess.run([sys.executable,'-S','-c',script],env={**os.environ,'PYTHONPATH':str(source)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
