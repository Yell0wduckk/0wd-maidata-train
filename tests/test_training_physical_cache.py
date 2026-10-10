from dataclasses import replace
from fractions import Fraction
import random

import pytest

from maidata.chart import Grammar, Moment, Note, Route, HoldDuration, SlideDuration, ChartError, EncodedChart, decode_chart
from maidata.chart.routes import catalog, route_for
from maidata.chart.types import Completion, Track
from maidata.physical import PhysicalState, Geometry, PathGeometry
from maidata.physical.actions import materialize
from maidata.physical.support import duration_support, moment_support, interval_decision
from maidata.chart_cache import produce
from maidata.chart_cache.training import build_supports, save_supports, load_supports, time_decision, duration_candidate_cells
from maidata.time import TimingMap


@pytest.fixture
def geometry():
    return Geometry({p:PathGeometry(p,2. if p == '5-3' else 1.,((0.,1 << (int(p[0])-1)),(.5,1 << (int(p[-1])-1))))
                     for v in catalog() for p in v['paths_by_start']})


def test_boundary_role_empty_windows_dropout_and_real_registration():
    timing=TimingMap([(0,120)])
    g=Grammar(timing)
    g.inject_boundary(Moment(0),start_tick=0,window_end=96)
    assert g.state.last_real_tick is None and g.state.moment == -1 and g.state.owner == -1
    with pytest.raises(ChartError): g.accept(Moment(0))
    g.accept(Moment(timing.delta_ms(0,24)))
    assert g.state.last_real_tick is None
    with pytest.raises(ChartError): g.accept(Moment(1))
    g.accept(Note.make('tap','A1'))
    assert g.state.last_real_tick == 24 and g.state.moment == 0
    g.accept(Moment(timing.delta_ms(24,100)))
    for a in (96,192,288):
        g.inject_boundary(Moment(timing.delta_ms(24,a)),start_tick=a,window_end=a+96)
        assert g.state.last_real_tick == 24 and g.state.owner == 0
        g.accept(Moment(timing.delta_ms(a,a+96)))
    g.inject_boundary(Moment(0),start_tick=384,history_present=False)
    assert g.state.cursor == 384 and g.state.last_real_tick == 24
    g.accept(Note.make('tap','A2'))
    assert g.state.last_real_tick == 384 and g.state.moment == 1


def test_unified_wifi_and_route_masks(geometry):
    s=PhysicalState(TimingMap([(0,60)]),geometry,audio_end_seconds=20)
    s.inject_boundary(Moment(0),start_tick=0)
    assert s.query_note_kinds()['slide'].allowed
    s.accept(Note.make('slide','A1'))
    assert s.query(Route(70,0)).allowed and s.query(Route(0,0)).allowed
    s.accept(Route(70,0));s.accept(SlideDuration(96))
    assert s.actions[-1].wifi and s.grammar.allowed_types() == ('NOTE','MOMENT')
    with pytest.raises(ChartError): Note.make('wifi','A1')
    s.accept(Note.make('slide','A3'));s.accept(Route(0,0))
    assert not s.query(Route(70)).allowed
    s.accept_source(SlideDuration(96))
    assert not s.query(Route(70,0)).allowed


def test_nonmonotone_partition_matches_every_tick(geometry):
    timing=TimingMap([(0,60)])
    old=materialize(Completion(0,0,0,Note.make('slide','A5'),track=Track((route_for('5-3'),route_for('3-5')),0,288),track_index=0),timing,geometry)
    s=PhysicalState(timing,geometry,audio_end_seconds=20)
    s.inject_boundary(Moment(0),start_tick=96);s.actions=(old,)
    s.accept(Note.make('slide','A1'));s.accept(Route(route_for('1-3'),0));s.accept(Route(route_for('3-5')))
    saved=s.snapshot();support=duration_support(s)
    for n in range(1,1001):
        assert interval_decision(support,n) == s.query(SlideDuration(n))
    assert [r[:2] for r in support['intervals'] if not r[2]['physical']] == [[1,191],[385,1728]]
    assert s.snapshot() == saved


def test_supports_match_oracle_across_bpm_and_handoffs(geometry):
    timing=TimingMap([(0,100),(Fraction(5,2),180),(6,75)],.08)
    rng=random.Random(14)
    for iteration in range(8):
        s=PhysicalState(timing,geometry,audio_end_seconds=8.2)
        s.inject_boundary(Moment(0),start_tick=180)
        history=[]
        for owner in range(4):
            tick=rng.randrange(0,180)
            if owner%2:
                c=Completion(owner,owner,tick,Note.make('slide',f'A{owner+1}'),track=Track((rng.randrange(71),),0,rng.randrange(2,400)),track_index=0)
            else:
                c=Completion(owner,owner,tick,Note.make('hold',f'A{owner+1}'),duration_tick=rng.randrange(2,400))
            history.append(materialize(c,timing,geometry))
        s.actions=tuple(history)
        support=moment_support(s)
        for n in range(181,800):
            assert interval_decision(support,n) == s.query(Moment(timing.delta_ms(180,n)))
        s.accept_source(Note.make('slide','A7') if iteration%2 else Note.make('hold','A7'))
        if iteration%2:
            s.accept_source(Route(0,0));s.accept_source(Route(2))
        constructor=SlideDuration if iteration%2 else HoldDuration
        support=duration_support(s)
        for n in range(2,1000):
            assert interval_decision(support,n) == s.query(constructor(n))


def test_training_readback_causal_prefix_and_window_tail(tmp_path,geometry,monkeypatch):
    monkeypatch.setattr(produce,'GEOMETRY',geometry)
    timing=TimingMap([(0,120)])
    metadata=dict(schema='maimai.chart_ir.v1',tick_rate=180,first_seconds=0,
                  controls=[dict(kind='bpm',bpm=120,position_beats=dict(num=0,den=1))])
    tokens=(Moment(0),Note.make('hold','A1'),HoldDuration(300),Moment(1000),Note.make('tap','A2'))
    ir=decode_chart(EncodedChart(tokens,metadata),geometry)
    row=dict(chart_id='test:1',timing=dict(bpm=[[0,1,120]],first_seconds=0),audio_end_seconds=5)
    record=produce.build_record(ir,row);a=build_supports(record,geometry)
    changed=decode_chart(EncodedChart((*tokens[:-1],Note.make('tap','A8')),metadata),geometry)
    b=build_supports(produce.build_record(changed,row),geometry)
    assert a['stages'][:3] == b['stages'][:3]
    path=tmp_path/'physical.json.gz';save_supports(path,a)
    assert load_supports(path) == a
    duration=duration_candidate_cells(a['stages'][1],timing)
    assert duration[0][0].lower == pytest.approx(1.5*timing.written_tick_seconds(0)*1000)
    assert time_decision(a['stages'][2],96,window_end=96).allowed
    assert time_decision(a['stages'][2],144.25,censored=True).allowed
    # An exact outside query and a censored empty-audio tail never require NOTE.
    assert time_decision(a['stages'][-1],1536,window_end=1152).allowed
