"""Local inference cache and category-only interval differential checks."""
import random
from unittest.mock import patch

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.chart.routes import catalog
from maidata.chart.types import Completion, Track
from maidata.physical import Geometry, PhysicalState
from maidata.physical.actions import materialize
from maidata.physical.attribution import AttributionState
from maidata.physical.conflicts import action_conflicts
from maidata.physical.geometry import PathGeometry
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.inference_context import InferenceIntervalContext
from maidata.time import TimingMap


def geometry():
    paths = {path for row in catalog() for path in row['paths_by_start']}
    return Geometry({p: PathGeometry(p, 1., ((0., 1 << (int(p[0])-1)),
                                            (.5, 1 << (int(p[-1])-1)))) for p in paths})


def state(cls=InferencePhysicalState):
    result = cls(TimingMap([(0, 120)]), geometry(), audio_end_seconds=30.)
    result.inject_boundary(Moment(0), start_tick=0, window_end=1536)
    return result


def raw(value):
    return value.raw.decision() if hasattr(value, 'raw') else value


def test_kind_short_circuit_and_current_prefix_reuse():
    s = state()
    with patch.object(s, '_query', wraps=s._query) as query:
        assert s.query_types()['NOTE'].allowed
        assert query.call_count == 1
        assert all(d.allowed for d in s.query_kinds().values())
        assert query.call_count == 5
        s.query_positions('tap')
        assert query.call_count == 12
        s.query_kinds()
        s.query_types()
        s.query_positions('tap')
        s.restore(s.snapshot())
        s.query_types()
        assert query.call_count == 12


def test_cosmetic_flags_share_raw_query_but_short_is_distinct():
    s = state()
    with patch.object(s, '_query', wraps=s._query) as query:
        a = s.query(Note.make('tap', 'A1'))
        assert s.query(Note.make('tap', 'A1', ex=1, break_=1)) is a
        assert query.call_count == 1
        assert s.query(Note.make('tap', 'A1', short_hold=1)).allowed
        assert query.call_count == 2
        s.accept_source(Note.make('slide', 'A1'))
        a = s.query(Route(0, 0))
        assert s.query(Route(0, 1)) is a


def test_duration_exists_does_not_partition_when_shortest_is_legal():
    s = state()
    s.accept_source(Note.make('hold', 'A1'))
    with patch('maidata.physical.inference.duration_support', wraps=__import__(
            'maidata.physical.support', fromlist=['duration_support']).duration_support) as support:
        assert s.query_types()['HOLD_DURATION'].allowed
        assert support.call_count == 0
        a = s.time_support('HOLD_DURATION')
        assert s.time_support('HOLD_DURATION') is a
        assert support.call_count == 1


def test_snapshot_mutations_expire_supports_and_preserve_actual_flags():
    s = state()
    prefix = s.snapshot()
    a = s.query_types()
    note = Note.make('tap', 'A1', ex=1, break_=1)
    s.accept(note)
    assert s.grammar.state.note == note
    assert s.query(Note.make('tap', 'A1')).physical == frozenset({'overlap'})
    s.restore(prefix)
    assert s.query_types() == a
    assert s.query(Note.make('tap', 'A1')).allowed
    assert len(s._queries) <= 2


def test_interval_panels_survive_pending_fields_until_actions_change():
    s = state()
    s.query_kinds()
    context = s._context
    s.accept_source(Note.make('slide', 'A1'))
    assert s._context is context
    s.accept_source(Route(0, 0))
    assert s._context is context
    s.query(SlideDuration(96))
    assert s._context is context
    s.accept_source(SlideDuration(96))
    assert s._context is None
    s.query_types()
    assert s._context is not context


def test_raw_supports_match_attribution_on_each_generated_prefix():
    s = state()
    for token in (Note.make('tap', 'A1'), Moment(250.), Note.make('hold', 'A2'),
                  HoldDuration(48), Moment(250.), Note.make('slide', 'A3'),
                  Route(0, 1), SlideDuration(96)):
        reference = AttributionState(s)
        assert s.query_types() == {key: raw(value) for key, value in reference.query_types().items()}
        if 'NOTE' in s.grammar.allowed_types():
            assert s.query_kinds() == {key: raw(value) for key, value in reference.query_kinds().items()}
            for kind in ('tap', 'hold', 'touch', 'touch_hold', 'slide'):
                assert s.query_positions(kind) == {key: raw(value) for key, value in reference.query_positions(kind).items()}
        assert s.query(token) == raw(reference.query(token))
        s.accept_source(token)


def test_interval_context_categories_match_complete_witness_engine():
    timing, geom, rng = TimingMap([(0, 120), (4, 180)]), geometry(), random.Random(20261010)

    def candidate(owner):
        tick, pad = rng.randrange(0, 600), f'A{rng.randrange(1, 9)}'
        kind = rng.choice(('tap', 'hold', 'touch', 'touch_hold', 'slide'))
        note = Note.make(kind, pad)
        duration = rng.randrange(2, 450) if kind in ('hold', 'touch_hold') else None
        track = Track((rng.randrange(71),), 0, rng.randrange(1, 450)) if kind == 'slide' else None
        if kind == 'slide' and track.routes != (70,) and rng.random() < .5:
            track = Track((*track.routes, rng.randrange(70)), 0, track.duration_tick)
        return materialize(Completion(owner, owner, tick, note, duration_tick=duration,
                                      track=track, track_index=0 if track else None), timing, geom)

    # Includes overlapping prefixes, touches, WiFi, varying chain times and
    # handoffs to later launches. No serialized assets or model are read.
    for _ in range(350):
        history = tuple(candidate(i) for i in range(rng.randrange(7)))
        context = InferenceIntervalContext(history)
        c = candidate(20)
        for repeat in (False, True):
            expected = frozenset(x.category for x in action_conflicts(history, c, touch_repeat=repeat))
            assert context.reasons(c, touch_repeat=repeat) == expected
            assert context.reasons(c, touch_repeat=repeat) == expected


def test_same_owner_branches_remain_distinct_capacity_resources():
    timing, geom = TimingMap([(0, 120)]), geometry()
    note = Note.make('slide', 'A1')
    history = tuple(materialize(Completion(0, 0, 0, note, track=Track((i,), 0, 192), track_index=i),
                                timing, geom) for i in (0, 1))
    candidate = materialize(Completion(1, 0, 96, Note.make('tap', 'A8')), timing, geom)
    expected = frozenset(x.category for x in action_conflicts(history, candidate))
    assert 'multitouch' in expected
    assert InferenceIntervalContext(history).reasons(candidate) == expected


def test_branch_launch_matching_shares_only_one_credit_per_owner():
    timing, geom = TimingMap([(0, 120)]), geometry()
    note = Note.make('slide', 'A1')
    branches = tuple(materialize(Completion(0, 0, 0, note, track=Track((i,), 0, 192), track_index=i),
                                 timing, geom) for i in (0, 1))
    press = materialize(Completion(1, 1, 96, Note.make('tap', 'A1')), timing, geom)
    candidate = materialize(Completion(2, 1, 96, Note.make('touch', 'A1')), timing, geom)
    history = (*branches, press)
    expected = frozenset(x.category for x in action_conflicts(history, candidate))
    assert 'multitouch' in expected
    context = InferenceIntervalContext(history)
    assert context.overloaded(candidate)
    assert context.reasons(candidate) == expected
