"""Semantic regressions for the inference-only Physical fast path."""
from dataclasses import replace
from fractions import Fraction
from random import Random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.chart.routes import catalog, route_for
from maidata.chart.types import Completion, Track, TOUCH_PADS
from maidata.physical import Geometry, PathGeometry, PhysicalState, PhysicalError
from maidata.physical.actions import materialize
from maidata.physical.attribution import AttributionState
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.support import interval_decision
from maidata.time import TimingMap
from maidata.decoder.supports import time_candidates
from maidata.inference.time_support import TimeCandidateCache


@pytest.fixture
def timing():
    return TimingMap([(0, 60)])


@pytest.fixture
def geometry():
    return Geometry({p: PathGeometry(p, 2. if p == '5-3' else 1.,
        ((0., 1 << (int(p[0])-1)), (.5, 1 << (int(p[-1])-1))))
        for row in catalog() for p in row['paths_by_start']})


def state(timing, geometry, *, tick=0, duration=20., window_end=None):
    result = InferencePhysicalState(timing, geometry, audio_end_seconds=duration)
    result.inject_boundary(Moment(0), start_tick=tick, window_end=window_end)
    return result


def press(timing, geometry, pad, *, tick=0, kind='tap', owner=10, duration=None):
    return materialize(Completion(owner, 0, tick, Note.make(kind, pad), duration_tick=duration),
                       timing, geometry)


def track(timing, geometry, paths, *, tick=0, duration=96, owner=10):
    return materialize(Completion(owner, 0, tick, Note.make('slide', 'A'+paths[0][0]),
        track=Track(tuple(route_for(p) for p in paths), 0, duration), track_index=0), timing, geometry)


def test_touch_repetition_is_pad_only_and_query_options_do_not_share_masks(timing, geometry):
    s = state(timing, geometry, tick=24)
    s.actions = tuple(press(timing, geometry, p, kind='touch', owner=i+10)
                      for i, p in enumerate(TOUCH_PADS))
    saved = s.snapshot()
    assert s.query_types()['NOTE'].allowed
    assert s.query_kinds()['touch'].allowed
    assert s.query_kinds()['touch_hold'].allowed
    token = Note.make('touch', 'C')
    assert s.query(token, touch_repeat=False).allowed
    assert s.query(token).physical == frozenset({'overlap'})
    assert s.query(token, touch_repeat=False).allowed
    assert all(not d.allowed for d in s.query_positions('touch').values())
    assert all(d.allowed for d in s.query_positions('touch', touch_repeat=False).values())
    assert all(not d.allowed for d in s.query_positions('touch').values())
    assert s.snapshot() == saved


@pytest.mark.parametrize('kind,pad,flag', [('tap', 'A1', 'short_hold'),
                                         ('touch', 'C', 'short_touch_hold')])
@pytest.mark.parametrize('short_first', [False, True])
def test_short_hold_audio_limit_is_not_erased_by_flag_cache(timing, geometry, kind, pad, flag, short_first):
    s = state(timing, geometry, duration=.005)
    ordinary, short = Note.make(kind, pad), Note.make(kind, pad, **{flag: 1})
    saved = s.snapshot()
    for token in ((short, ordinary, short) if short_first else (ordinary, short, ordinary)):
        decision = s.query(token)
        assert decision.allowed == (token == ordinary)
        if token == short:
            assert decision.basic_errors == ('duration_outside_audio',)
            assert not decision.physical
    assert s.query_kinds()[kind].allowed
    assert s.query_positions(kind)[pad].allowed
    assert s.snapshot() == saved


def test_nonmonotone_duration_islands_survive_type_projection_and_cache(timing, geometry):
    s = state(timing, geometry, tick=96)
    s.actions = (track(timing, geometry, ('5-3', '3-5'), duration=288),)
    s.accept(Note.make('slide', 'A1'))
    s.accept(Route(route_for('1-3'), 0))
    s.accept(Route(route_for('3-5')))
    saved = s.snapshot()
    assert s.query_types()['SLIDE_DURATION'].allowed
    support = s.time_support('SLIDE_DURATION')
    assert [row[:2] for row in support['intervals'] if not row[2]['basic'] and not row[2]['physical']] == [
        [1, 191], [385, 1728]]
    # Check both islands and the equality cases around the forbidden hole.
    for tick in (1, 190, 191, 192, 193, 288, 383, 384, 385, 386, 1728, 1729):
        canonical = PhysicalState(timing, geometry, audio_end_seconds=20)
        canonical.restore(saved)
        assert interval_decision(support, tick) == canonical.query(SlideDuration(tick))
        assert s.query(SlideDuration(tick)) == canonical.query(SlideDuration(tick))
    assert s.time_support('SLIDE_DURATION') == support
    assert s.snapshot() == saved


def test_duration_type_is_legal_when_only_later_intervals_have_completions(timing, geometry):
    geometry.paths['5-3'] = replace(geometry.paths['5-3'], length=1.)
    s = state(timing, geometry, tick=96)
    s.actions = (track(timing, geometry, ('5-3', '3-5'), duration=192),)
    s.accept(Note.make('slide', 'A1'))
    s.accept(Route(route_for('1-3'), 0))
    s.accept(Route(route_for('3-5')))
    # Rejecting the minimum duration cannot exclude the entire token type.
    assert not s.query(SlideDuration(1)).allowed
    assert not s.query(SlideDuration(192)).allowed
    assert s.query(SlideDuration(193)).allowed
    assert s.query_types()['SLIDE_DURATION'].allowed
    support = s.time_support('SLIDE_DURATION')
    assert [row[:2] for row in support['intervals'] if not row[2]['physical']] == [[193, 1728]]


def test_route_prefix_can_be_repaired_by_extension_then_rolled_back(timing, geometry):
    geometry.paths = {p: replace(v, exempt=True) for p, v in geometry.paths.items()}
    s = state(timing, geometry, tick=96)
    s.actions = (track(timing, geometry, ('1-3',)),)
    s.accept(Note.make('slide', 'A1'))
    first = Route(route_for('1-3'), 0)
    assert s.query(first).allowed and s.query(first).deferred
    s.accept(first)
    pending = s.snapshot()
    # Closing now repeats the original path; a longer route prefix changes it.
    assert not s.query_types()['SLIDE_DURATION'].allowed
    extension = Route(route_for('3-5'))
    assert s.query(extension).allowed and s.query(extension).deferred
    s.accept(extension)
    assert s.query_types()['SLIDE_DURATION'].allowed
    assert s.query(SlideDuration(96)).allowed
    s.restore(pending)
    assert not s.query_types()['SLIDE_DURATION'].allowed
    assert s.query(extension).allowed
    assert s.snapshot() == pending


def test_first_route_coverage_is_rejected_before_duration_or_extension(timing, geometry):
    s = state(timing, geometry, tick=192)
    s.actions = (track(timing, geometry, ('1-3',)),)
    s.accept(Note.make('slide', 'A1'))
    saved = s.snapshot()
    route = Route(route_for('1-3'), 0)
    # Old motion ends exactly when the new head appears. Every duration and
    # later extension keeps this first segment's appearance and launch fixed.
    with patch('maidata.physical.inference.duration_support', side_effect=AssertionError('duration query')):
        decision = s.query(route)
        assert decision.physical == frozenset({'slide_too_fast'})
        assert not decision.deferred
        with pytest.raises(PhysicalError):
            s.accept(route)
    assert s.snapshot() == saved
    # Training now charges the same immutable witness at the ROUTE decision.
    training = PhysicalState(timing, geometry, audio_end_seconds=20.)
    training.restore(saved)
    oracle = AttributionState(training)
    assert not oracle.query(route).raw.deferred
    assert not oracle.query(route).raw.decision().allowed
    training.accept_source(route)
    assert 'slide_too_fast' in oracle.query(SlideDuration(96)).raw.decision().physical
    # A restored legacy/deferred prefix cannot append its way out of coverage.
    s.accept_source(route)
    assert s.query(Route(route_for('3-5'))).physical == frozenset({'slide_too_fast'})


@pytest.mark.parametrize('exempt', ['old', 'new', None])
def test_route_coverage_preserves_exemptions_and_snapshot_cache(timing, geometry, exempt):
    s = state(timing, geometry, tick=192)
    old = track(timing, geometry, ('1-3',))
    if exempt == 'old':
        old = replace(old, motions=tuple(replace(m, exempt=True) for m in old.motions))
    elif exempt == 'new':
        geometry.paths['1-3'] = replace(geometry.paths['1-3'], exempt=True)
    s.actions = (old,)
    s.accept(Note.make('slide', 'A1'))
    saved = s.snapshot()
    route = Route(route_for('1-3'), 0)
    decision = s.query(route)
    assert decision.allowed == (exempt is not None)
    s.actions = ()
    assert s.query(route).allowed
    s.restore(saved)
    assert s.query(route) == decision


def test_route_coverage_allows_a_finished_path_before_new_appearance(timing, geometry):
    s = state(timing, geometry, tick=193)
    s.actions = (track(timing, geometry, ('1-3',)),)
    s.accept(Note.make('slide', 'A1'))
    assert s.query(Route(route_for('1-3'), 0)).allowed


def test_chained_segments_do_not_consume_branch_slots_and_wifi_stays_exclusive(timing, geometry):
    s = state(timing, geometry)
    s.accept(Note.make('slide', 'A1'))
    s.accept(Route(route_for('1-3'), 0))
    s.accept(Route(route_for('3-5')))
    assert s.grammar.state.track_count == 0
    s.accept(SlideDuration(192))
    first = s.snapshot()
    assert len(s.actions) == 1 and len(s.actions[0].motions) == 2
    assert s.query_types()['ROUTE'].allowed
    assert not s.query(Route(70, 0)).allowed
    s.accept(Route(route_for('1-7'), 0))
    s.accept(SlideDuration(96))
    assert not s.query_types()['ROUTE'].allowed
    assert not s.query(Route(route_for('1-5'), 0)).allowed
    s.restore(first)
    assert s.query_types()['ROUTE'].allowed
    assert s.query(Route(route_for('1-5'), 0)).allowed
    s.restore(state(timing, geometry).snapshot())
    s.accept(Note.make('slide', 'A1'))
    s.accept(Route(70, 0))
    assert not s.query(Route(0)).allowed  # WiFi cannot be a chained segment.
    s.accept(SlideDuration(96))
    assert not s.query_types()['ROUTE'].allowed
    assert not s.query(Route(0, 0)).allowed


def test_snapshot_restore_and_replaced_action_tuple_invalidate_prefix_masks(timing, geometry):
    s = state(timing, geometry)
    empty = s.snapshot()
    assert s.query(Note.make('tap', 'A4')).allowed
    assert s.query_types()['NOTE'].allowed
    s.actions = (press(timing, geometry, 'A1', kind='hold', duration=96),
                 press(timing, geometry, 'A2', kind='hold', owner=11, duration=96))
    blocked = s.snapshot()
    assert not s.query(Note.make('tap', 'A4')).allowed
    assert not s.query_types()['NOTE'].allowed
    s.restore(empty)
    assert s.query(Note.make('tap', 'A4')).allowed
    assert s.query_types()['NOTE'].allowed
    s.accept(Note.make('tap', 'A4'))
    assert not s.query(Note.make('tap', 'A4')).allowed
    s.restore(blocked)
    assert not s.query(Note.make('tap', 'A4')).allowed
    s.restore(empty)
    assert s.query(Note.make('tap', 'A4')).allowed
    assert s.snapshot() == empty


def test_terminal_does_not_require_note_and_window_end_is_in_query_cache_key(timing, geometry):
    s = state(timing, geometry, window_end=96)
    s.actions = (press(timing, geometry, 'A1', kind='hold', duration=480),
                 press(timing, geometry, 'A2', kind='hold', owner=11, duration=480))
    saved = s.snapshot()
    assert not s.query_types()['NOTE'].allowed
    assert s.query_types()['MOMENT'].allowed
    token = Moment(timing.delta_ms(0, 96))
    assert s.query(token).allowed  # default end: terminal.
    assert not s.query(token, window_end=192).allowed  # override: real onset.
    assert s.query(token).allowed
    s.accept(token)
    s.inject_boundary(Moment(0), start_tick=96, window_end=192)
    assert s.query_types()['MOMENT'].allowed
    assert not s.query_types()['NOTE'].allowed
    s.restore(saved)
    assert s.query(token).allowed
    assert not s.query(token, window_end=192).allowed
    assert s.snapshot() == saved


def test_fast_path_matches_raw_attribution_for_real_candidates(timing, geometry):
    s = state(timing, geometry, tick=96, window_end=192)
    s.actions = (track(timing, geometry, ('1-3',)),
                 press(timing, geometry, 'B2', tick=80, kind='touch', owner=11))
    saved = s.snapshot()
    reference = PhysicalState(timing, geometry, audio_end_seconds=20)
    reference.restore(saved)
    oracle = AttributionState(reference)
    for kind, expected in oracle.query_types().items():
        assert s.query_types()[kind].allowed == expected.raw.decision().allowed
    for kind, expected in oracle.query_kinds().items():
        assert s.query_kinds()[kind].allowed == expected.raw.decision().allowed
        for pad, decision in oracle.query_positions(kind).items():
            assert s.query_positions(kind)[pad].allowed == decision.raw.decision().allowed
    for token in (Note.make('touch', 'B2'), Note.make('tap', 'A1'),
                  Note.make('tap', 'A3', short_hold=1), Note.make('hold', 'A4'),
                  Note.make('slide', 'A5'), Moment(timing.delta_ms(96, 120)),
                  Moment(timing.delta_ms(96, 192))):
        assert s.query(token).allowed == oracle.query(token).raw.decision().allowed
    assert s.snapshot() == saved


def test_random_time_candidates_preserve_masks_labels_and_clipped_duration_support():
    rng = Random(20261010)
    timing = TimingMap([(0, 131), (Fraction(1, 7), 177),
                        (Fraction(2053, 96), 83), (100, 201)], first_seconds=-.17)
    cache = TimeCandidateCache()
    for kind in ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION'):
        for case in range(24):
            cursor = rng.choice((-2050, -1025, -2, 0, 1023, 2052, 9600)) + rng.randrange(4)
            end = cursor + rng.randrange(1, 110)
            cap = end + rng.randrange(130) + rng.choice((0., .25, .5, .75))
            lower = cursor - 10 if kind == 'MOMENT' else rng.randrange(2, 70)
            upper = int(cap) + 10 if kind == 'MOMENT' else lower + rng.randrange(1, 270)
            intervals, at = [], lower - 5
            while at <= upper + 5:
                stop = min(upper + 5, at + rng.randrange(20))
                intervals.append([at, stop, dict(basic=['outside'] if rng.randrange(5) == 0 else [],
                    physical=rng.randrange(16), penalty=rng.randrange(16))])
                at = stop + 1 + rng.randrange(4)
            support = dict(domain=[lower, upper], outside_basic='outside', intervals=intervals)
            stage = dict(cursor_tick=cursor,
                         **{'moment' if kind == 'MOMENT' else 'duration': support})
            args = dict(kind=kind, window_end=end, cap_tick=cap, device='cpu')
            expected = time_candidates(stage, timing, **args)
            actual = cache.candidates(stage, timing, **args)
            for field in ('lower', 'upper', 'valid', 'raw', 'penalty'):
                torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                           rtol=0, atol=0, msg=f'{kind} case {case}: {field}')
            assert (actual.labels, actual.ticks, actual.basic_tick_intervals, actual.choices) == (
                expected.labels, expected.ticks, expected.basic_tick_intervals, expected.choices)
            if kind != 'MOMENT':
                target_type = HoldDuration if kind == 'HOLD_DURATION' else SlideDuration
                # Basic support refers to exact target ticks, even when the
                # bucket representative lies in a different Physical interval.
                for tick in (lower, upper, rng.randint(lower, upper)):
                    target = target_type(tick)
                    try:
                        mask = expected.target_mask(target, timing, cursor)
                    except ValueError:
                        with pytest.raises(ValueError):
                            actual.target_mask(target, timing, cursor)
                    else:
                        assert torch.equal(actual.target_mask(target, timing, cursor), mask)


def test_time_coordinate_lru_revisits_evicted_negative_and_boundary_blocks():
    timing = TimingMap([(0, 131), (Fraction(1, 7), 177)], first_seconds=-.17)
    cache = TimeCandidateCache()
    # Small blocks exercise the same eviction arithmetic without constructing
    # a long-song fixture merely to reach the real 16-block capacity.
    cache.BLOCK_SIZE, cache.MAX_BLOCKS = 8, 2
    requests = ([0, 7, 8, 15], [0], [16], [-1, -8], [8, 15, 16, 23])
    expected_blocks = ((0, 1), (1, 0), (0, 2), (2, -1), (1, 2))
    expected_calls = (16, 0, 8, 8, 16)
    for half_ticks, blocks, calls in zip(requests, expected_blocks, expected_calls):
        expected = np.asarray([timing.second(float(i) / 2) for i in half_ticks])
        with patch.object(timing, 'second', wraps=timing.second) as coordinates:
            actual = cache.seconds(timing, half_ticks)
            assert coordinates.call_count == calls
        np.testing.assert_array_equal(actual, expected)
        assert tuple(cache.blocks) == blocks
