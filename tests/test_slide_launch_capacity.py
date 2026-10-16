"""Mandatory future occupancy, first responsibility and combined migrations."""
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.chart.routes import route_for
from maidata.physical import PhysicalState
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.attribution import AttributionState
from maidata.physical.conflicts import action_conflicts
from maidata.physical.support import interval_decision
from maidata.time import TimingMap
from maidata.inference.sampling import Sampler
from maidata.chart_cache.training import build_supports, SLIDE_LAUNCH
from maidata.chart_cache.migrate_route_coverage import migrate_supports
from test_physical_inference import geometry
from test_decoder_training import fields


def source_tokens(timing, onset=48):
    # {8}1-3[8:1]*-4-2-8-6-8-2-4-6-8<3[2:1],1-6-2[8:3],
    paths = ('1-4','4-2','2-8','8-6','6-8','8-2','2-4','4-6','6-8','8<3')
    return [Note.make('slide','A1'), Route(route_for('1-3'),0), SlideDuration(48),
        *[Route(route_for(p),0 if i==0 else None) for i,p in enumerate(paths)],
        SlideDuration(192), Moment(timing.delta_ms(0,onset))]


def prefix(timing=None, onset=48):
    timing = timing or TimingMap([(0,180)])
    s = InferencePhysicalState(timing,geometry(),audio_end_seconds=20)
    s.inject_boundary(Moment(0),start_tick=0,window_end=1536)
    for token in source_tokens(timing,onset): s.accept_source(token)
    return s


@pytest.mark.parametrize('timing', [TimingMap([(0,180)]), TimingMap([(0,120)]),
    TimingMap([(0,120),(Fraction(5,4),240)])])
def test_exact_user_fragment_rejected_before_route_and_again_at_duration(timing):
    s = prefix(timing)
    note = Note.make('slide','A1')
    before = s.snapshot()
    assert s.query(note).physical == {'multitouch'}
    training = AttributionState(s)
    early = training.query(note)
    assert early.raw.decision() == s.query(note)
    assert s.snapshot() == before
    assert s.query(Note.make('tap','A1')).allowed  # Waiting is not held.
    stage = training.stage_and_accept(note)
    charged = [w for rows in stage['attribution'].values() for w in rows]
    assert charged and all(c[0]=='multitouch' for c in charged)
    for i,p in enumerate(('1-6','6-2')):
        training.stage_and_accept(Route(route_for(p),0 if i==0 else None))
    assert s.query(SlideDuration(144)).physical == {'multitouch'}
    assert interval_decision(s.time_support('SLIDE_DURATION'),144).physical == {'multitouch'}
    complete = training.query(SlideDuration(144))
    assert early.raw.conflicts <= complete.raw.conflicts
    assert not (early.raw.conflicts & complete.penalty.conflicts)
    for duration in (1,2,48,144,385,768):
        assert 'multitouch' in s.query(SlideDuration(duration)).physical


@pytest.mark.parametrize('onset,blocked',[(69,True),(70,False),(71,False)])
def test_release_boundary_and_exact_incoming_handoff(onset,blocked):
    s=prefix(onset=onset)
    assert ('multitouch' in s.query(Note.make('slide','A1')).physical) == blocked
    # Previous 1-3 ends before this launch: its hand can continue at A3.
    assert 'multitouch' not in s.query(Note.make('slide','A3')).physical


def test_wifi_and_parallel_branch_extra_resource_is_at_route():
    t=TimingMap([(0,120)])
    s=InferencePhysicalState(t,geometry(),audio_end_seconds=20)
    s.inject_boundary(Moment(0),start_tick=0)
    for token in (Note.make('hold','A8'),HoldDuration(384),Note.make('slide','A1')):
        s.accept_source(token)
    a=AttributionState(s)
    assert 'multitouch' not in a.query(Route(route_for('1-3'),0)).raw.decision().physical
    assert 'multitouch' in a.query(Route(70,0)).raw.decision().physical
    a.stage_and_accept(Route(route_for('1-3'),0))
    a.stage_and_accept(SlideDuration(96))
    assert 'multitouch' in a.query(Route(route_for('1-5'),0)).raw.decision().physical


def test_all_positions_crowded_kind_gradient_and_no_future_labels():
    t=TimingMap([(0,120)])
    s=PhysicalState(t,geometry(),audio_end_seconds=20)
    s.inject_boundary(Moment(0),start_tick=0)
    for token in (Note.make('slide','A1'),Route(route_for('1-3'),0),SlideDuration(384),
                  Route(route_for('1-5'),0),SlideDuration(384)):
        s.accept_source(token)
    a=AttributionState(s)
    with patch('maidata.physical.attribution.duration_support',side_effect=AssertionError('future search')):
        records={k:v.record() for k,v in a.query_kinds().items()}
    assert records['slide']['penalty'] & 1
    with patch('maidata.physical.attribution.slide_launch_capacity',return_value=(frozenset(),frozenset())):
        assert not a.query_kinds()['slide'].record()['penalty']
    from maidata.decoder.supports import categorical_support
    from maidata.decoder.losses import physical_categories
    from maidata.chart.types import KINDS
    allowed,penalty,_=categorical_support(records,KINDS,device='cpu')
    logits=torch.zeros(len(KINDS),requires_grad=True)
    physical_categories(logits.masked_fill(~allowed,-torch.inf).log_softmax(-1),penalty).sum().backward()
    assert logits.grad[KINDS.index('slide')] > 0


@pytest.mark.parametrize('hard',[False,True])
def test_sampler_and_cont_keep_multitouch_mandatory(hard):
    from maidata.physical.state import Decision
    sampler=Sampler(physical_hard_mask=hard)
    assert not sampler.allowed(Decision(physical=frozenset({'multitouch'})))
    assert sampler.allowed(Decision(physical=frozenset({'slide_too_fast'}))) == (not hard)
    stage=dict(cursor_tick=0,duration=dict(domain=[1,800],intervals=[
        [1,384,dict(basic=[],physical=4)], [385,800,dict(basic=[],physical=1)]],outside_basic='end'))
    tree=sampler.support_tree(stage,TimingMap([(0,120)]),kind='SLIDE_DURATION',window_end=1536,cap_tick=1536)
    assert not tree.hard[0][384]  # No mandatory-legal leaf behind CONT.
    assert tree.hard[0][0] == (not hard)


@pytest.mark.parametrize('prior_coverage',[False,True])
def test_combined_migration_equals_fresh_and_preserves_source(monkeypatch,prior_coverage):
    t=TimingMap([(0,120)])
    tokens=[Moment(0),*source_tokens(t),Note.make('slide','A1'),Route(route_for('1-6'),0),
        Route(route_for('6-2')),SlideDuration(144),Moment(2000),Note.make('tap','A8')]
    disabled=(frozenset(),frozenset())
    with patch('maidata.physical.attribution.slide_launch_capacity',return_value=disabled):
        if prior_coverage:
            record,old,geo=fields(monkeypatch,tokens,audio_end=20,bpm=120)
        else:
            with patch('maidata.physical.attribution.first_route_coverage',return_value=frozenset()):
                record,old,geo=fields(monkeypatch,tokens,audio_end=20,bpm=120)
    old.pop('slide_launch_attribution')
    original=deepcopy(old)
    new,audit=migrate_supports(record,old,geo)
    assert new == build_supports(record,geo)
    assert old == original and audit['changed_target_indices']
    assert new['slide_launch_attribution']==SLIDE_LAUNCH
    with patch('maidata.physical.attribution.duration_support',side_effect=AssertionError('unaffected duration search')):
        again,audit2=migrate_supports(record,new,geo)
    assert again==new and not audit2['changed_target_indices']
    assert not audit2['replayed_stages']


def test_new_launch_window_penalties_migrate_without_changing_gt(monkeypatch):
    from maidata.decoder_training.support_cache import ChartSupports
    from maidata.decoder_training.export_windows import compile_window
    from maidata.decoder_training.migrate_physical import replace_physical, equal
    from maidata.decoder.windows import real_onsets
    t=TimingMap([(0,120)])
    tokens=[Moment(0),*source_tokens(t),Note.make('slide','A1'),Route(route_for('1-6'),0),
        Route(route_for('6-2')),SlideDuration(144),Moment(2000),Note.make('tap','A8')]
    disabled=(frozenset(),frozenset())
    with patch('maidata.physical.attribution.slide_launch_capacity',return_value=disabled):
        record,old,geo=fields(monkeypatch,tokens,audio_end=20,bpm=120)
    old.pop('slide_launch_attribution')
    new,_=migrate_supports(record,old,geo)
    cfg=dict(target_beats=2,audio_after_beats=2,chart_history_beats=4,decoder_history_dropout=.15)
    charts=[ChartSupports(record,support,geo) for support in (old,new)]
    onsets=real_onsets(charts[0].tokens,t,sequence_start_tick=0)
    changed_any=False
    for start in (0,48,192,1536):
        row=dict(chart_id='0:5',window_id=f'0:5:{start}',start_tick=start,end_tick=start+192)
        with patch('maidata.physical.attribution.slide_launch_capacity',return_value=disabled):
            before=compile_window(charts[0],row,cfg,20,onsets)
        fresh=compile_window(charts[1],row,cfg,20,onsets)
        after,changed=replace_physical(before,fresh)
        changed_any |= changed
        assert equal(after['labels'],fresh['labels'])
        assert equal(after['tokens'],before['tokens']) and equal(after['next_types'],before['next_types'])
        assert after['metadata']['slide_launch_attribution']==SLIDE_LAUNCH
    assert changed_any
