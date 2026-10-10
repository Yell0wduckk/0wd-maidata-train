from dataclasses import replace

import pytest

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration, ChartError
from maidata.chart.routes import catalog, route_for
from maidata.chart.types import Completion, Track
from maidata.physical import PhysicalState, AttributionState, Geometry, PathGeometry, PhysicalError
from maidata.physical.actions import materialize
from maidata.physical.attribution import Evaluation
from maidata.physical.conflicts import action_conflicts, capacity_conflicts
from maidata.physical.rules import hand_segments
from maidata.chart_cache.training import candidate_decision, type_decision, time_decision
from maidata.time import TimingMap


@pytest.fixture
def context():
    geometry = Geometry({p: PathGeometry(p, 1., ((0., 1 << (int(p[0])-1)),))
                         for v in catalog() for p in v['paths_by_start']})
    timing = TimingMap([(0, 60)])
    state = PhysicalState(timing, geometry, audio_end_seconds=8)
    state.inject_boundary(Moment(0), start_tick=0)
    return AttributionState(state)


def track(context, owner, path, *, tick=0, duration=96):
    p = context.physical
    return materialize(Completion(owner, 0, tick, Note.make('slide', 'A'+path[0]),
                       track=Track((route_for(path),), 0, duration), track_index=0), p.timing, p.geometry)


def add_track(context, route=0):
    context.stage_and_accept(Route(route, 0))
    return context.stage_and_accept(SlideDuration(96))


def test_hold_overlap_is_charged_once_but_hard_rejection_stays(context):
    context.stage_and_accept(Note.make('tap', 'A1'))
    note = context.stage_and_accept(Note.make('hold', 'A1'))
    assert not type_decision(note, 'NOTE').physical
    assert not candidate_decision(note['kind']['hold']).physical
    assert candidate_decision(note['pad']['A1']).physical == {'overlap'}
    saved = context.snapshot()
    duration = context.stage_and_accept(HoldDuration(96))
    assert time_decision(duration, 96).allowed
    assert time_decision(duration, 96, penalty=False).physical == {'overlap'}
    assert all(v['penalty'] == 0 for _, _, v in duration['duration']['intervals'])
    context.restore(saved)
    with pytest.raises(PhysicalError):
        context.physical.accept(HoldDuration(96))
    assert context.snapshot() == saved


def test_router_third_branch_and_wifi_route_have_distinct_first_stages(context):
    context.stage_and_accept(Note.make('slide', 'A1'))
    add_track(context, route_for('1-3'))
    before = context.snapshot()
    wifi = context.stage_and_accept(Route(70, 0))
    assert not type_decision(wifi, 'ROUTE').physical
    assert candidate_decision(wifi['route'][70]).physical == {'multitouch'}
    duration = context.stage_and_accept(SlideDuration(96))
    assert not type_decision(duration, 'SLIDE_DURATION').physical
    assert not time_decision(duration, 96).physical
    assert time_decision(duration, 96, penalty=False).physical == {'multitouch'}
    context.restore(before)
    add_track(context, route_for('1-7'))
    third = context.stage_and_accept(Route(route_for('1-5'), 0))
    assert type_decision(third, 'ROUTE').physical == {'multitouch'}
    assert not candidate_decision(third['route'][route_for('1-5')]).physical
    assert all(v['physical'] & 1 for v in third['route'])


def test_wifi_first_then_branch_is_router_fault_and_gt_is_retained(context):
    context.stage_and_accept(Note.make('slide', 'A1'))
    add_track(context, 70)
    saved = context.snapshot()
    branch = context.stage_and_accept(Route(0, 0))
    assert type_decision(branch, 'ROUTE').physical == {'multitouch'}
    assert not candidate_decision(branch['route'][0]).physical
    context.stage_and_accept(SlideDuration(96))
    assert len(context.physical.actions) == 2
    context.restore(saved)
    assert not context.physical.query_types()['ROUTE'].allowed
    assert not context.physical.query(Route(0, 0)).allowed


def test_later_duration_new_multitouch_not_erased_by_category(context):
    p = context.physical
    a = materialize(Completion(10, 0, 0, Note.make('tap', 'A1')), p.timing, p.geometry)
    b = materialize(Completion(11, 0, 0, Note.make('tap', 'A2')), p.timing, p.geometry)
    # Two short presses now, and two already-scheduled tracks later. The later
    # conflict has different participants despite the same category.
    x = replace(track(context, 12, '3-5'), head=False, press_end=0.)
    y = replace(track(context, 13, '6-8'), head=False, press_end=0.)
    p.actions = (a, b, x, y)
    note = context.stage_and_accept(Note.make('hold', 'A4'))
    assert type_decision(note, 'NOTE').physical == {'multitouch'}
    duration = context.stage_and_accept(HoldDuration(144))
    assert not type_decision(duration, 'HOLD_DURATION').physical
    assert not time_decision(duration, 2).physical
    assert time_decision(duration, 144).physical == {'multitouch'}
    assert duration['attribution']['duration']


def test_existing_gt_overload_does_not_penalize_extra_touch_in_same_pool(context):
    p = context.physical
    p.actions = tuple(materialize(Completion(i+10, 0, 0, Note.make(kind, pad)), p.timing, p.geometry)
                      for i, (kind, pad) in enumerate((('tap','A1'), ('tap','A2'), ('touch','B1'))))
    value = context.query(Note.make('touch', 'B2'))
    assert value.raw.decision().physical == {'multitouch'}
    assert not value.penalty.decision().physical


def test_chain_can_extend_before_choosing_duration_type(context):
    p = context.physical
    p.geometry.paths = {k: replace(v, exempt=True) for k,v in p.geometry.paths.items()}
    p.actions = (track(context, 10, '1-3'),)
    context.stage_and_accept(Note.make('slide', 'A1'))
    first = context.stage_and_accept(Route(route_for('1-3'), 0))
    assert not candidate_decision(first['route'][route_for('1-3')]).physical
    saved = context.snapshot()
    closed = context.stage_and_accept(SlideDuration(96))
    assert type_decision(closed, 'SLIDE_DURATION').physical == {'slide_too_fast'}
    assert not time_decision(closed, 96).physical
    context.restore(saved)
    extended = context.stage_and_accept(Route(route_for('3-5')))
    assert not type_decision(extended, 'ROUTE').physical
    duration = context.stage_and_accept(SlideDuration(96))
    assert not type_decision(duration, 'SLIDE_DURATION').physical
    assert not time_decision(duration, 96).physical


def test_queries_and_snapshot_restore_include_responsibility(context):
    context.stage_and_accept(Note.make('tap', 'A1'))
    before = context.snapshot()
    context.query_types()
    context.query_kinds()
    assert context.snapshot() == before
    context.stage_and_accept(Note.make('hold', 'A1'))
    charged = context.snapshot()
    assert charged.charged
    with pytest.raises(ChartError):
        context.stage_and_accept(Route(0, 0))
    assert context.snapshot() == charged
    context.stage_and_accept(HoldDuration(96))
    context.restore(charged)
    assert context.snapshot() == charged
    context.restore(before)
    assert not context.charged


def test_terminal_has_no_note_constraint_and_touch_repeat_stays_at_pad(context):
    from maidata.chart.types import TOUCH_PADS
    p = context.physical
    p.actions = tuple(materialize(Completion(i+10,0,0,Note.make('touch',pad)),p.timing,p.geometry)
                      for i,pad in enumerate(TOUCH_PADS))
    p.grammar.state = replace(p.grammar.state, cursor=24)
    types = context.query_types()
    assert not types['NOTE'].raw.decision().physical
    assert not context.query_kinds()['touch'].raw.decision().physical
    assert all(v.raw.decision().physical == {'overlap'} for v in context.query_positions('touch').values())
    assert context.query(Moment(1000),window_end=96) == Evaluation()
    assert not types['MOMENT'].raw.decision().physical


def test_capacity_witnesses_match_matching_rule(context):
    p = context.physical
    import itertools
    candidates = [track(context, 10, '1-3'), track(context, 11, '3w7'),
                  materialize(Completion(12,0,96,Note.make('tap','A1')),p.timing,p.geometry),
                  materialize(Completion(13,0,96,Note.make('touch','A3')),p.timing,p.geometry)]
    for n in range(1, len(candidates)+1):
        for actions in itertools.combinations(candidates,n):
            expected = any(d > 2 for _,_,d in hand_segments(actions))
            assert bool(capacity_conflicts(actions,[(-1.,20.)])) == expected


def test_projection_does_not_charge_optional_same_category_conflict():
    from maidata.physical.attribution import Evidence, project
    from maidata.physical.conflicts import Conflict
    always = Conflict('multitouch','capacity',(('press',0,0,'A1'),('track',1,0,0)))
    later = Conflict('multitouch','capacity',(('press',0,0,'A1'),('track',2,0,0)))
    short = Evidence(conflicts=frozenset({always}))
    long = Evidence(conflicts=frozenset({always,later}))
    value = project([Evaluation(short,short),Evaluation(long,long)])
    assert value.raw.conflicts == {always,later}
    assert value.penalty.conflicts == {always}


def test_capacity_subset_cannot_match_a_removed_track_of_same_owner(context):
    from maidata.physical.conflicts import press_key, track_key
    p = context.physical
    normal = track(context, 10, '1-3')
    wifi = replace(track(context, 10, '3w7'), track_index=1)
    press = materialize(Completion(11,0,96,Note.make('tap','A1')),p.timing,p.geometry)
    conflicts = capacity_conflicts((normal,wifi,press),[(1.,1.01)])
    # The A1 press can merge with the normal track, but not with the A3 WiFi
    # alone just because that removed normal track had the same owner.
    assert any(set(c.participants) == {press_key(press),track_key(wifi)} for c in conflicts)
