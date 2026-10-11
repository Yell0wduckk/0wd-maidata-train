"""Duration supervision stops querying at the local audio boundary, not its GT value."""
from copy import deepcopy
import math

import pytest
import torch

from maidata.chart.types import HoldDuration, SlideDuration
from maidata.time import TimingMap, duration_end
from maidata.time.discrete import (
    CONTINUE, TIME_CLASSES, compile_support_tree, compile_time_target,
    duration_query_limit,
)
from maidata.time.distribution import (
    complete_field_log_probabilities, physical_categories, time_path_nll,
)


def _record(penalty=0):
    return dict(basic=(), physical=penalty, penalty=penalty)


def _stage(kind, *, upper=2000, intervals=None, origin=0):
    lower = 2 if kind == 'HOLD_DURATION' else 1
    return dict(cursor_tick=origin, duration=dict(
        domain=[lower, upper],
        intervals=intervals or [(lower, upper, _record())],
        outside_basic='duration_outside_audio'))


def _target(kind, ticks):
    return (HoldDuration if kind == 'HOLD_DURATION' else SlideDuration)(ticks)


@pytest.mark.parametrize('kind', ['HOLD_DURATION', 'SLIDE_DURATION'])
@pytest.mark.parametrize('ticks, expected', [
    (384, [(383,)]),
    (385, [(CONTINUE,), (0,)]),
    (700, [(CONTINUE,), (315,)]),
    (768, [(CONTINUE,), (383,)]),
    (769, [(CONTINUE,), (CONTINUE,)]),
    (1900, [(CONTINUE,), (CONTINUE,)]),
])
def test_local_boundary_retains_final_short_or_crossing_continue(kind, ticks, expected):
    timing = TimingMap([(0, 120)])
    path = compile_time_target(kind, ticks=ticks, origin_tick=0, timing=timing,
                               duration_query_end_ticks=500)
    assert [step.target_classes for step in path.steps] == expected
    assert [step.elapsed_ticks for step in path.steps] == [384 * i for i in range(len(expected))]
    assert path.exact_ticks == ticks
    # Keeping an exact source field does not manufacture a censored MOMENT label.
    assert not path.censored


@pytest.mark.parametrize('kind', ['HOLD_DURATION', 'SLIDE_DURATION'])
@pytest.mark.parametrize('boundary, expected_limit, expected_steps', [
    (384., 384, [(CONTINUE,)]),
    (384.25, 385, [(CONTINUE,), (0,)]),
])
def test_fractional_audio_tail_does_not_remove_last_in_window_query(
        kind, boundary, expected_limit, expected_steps):
    timing = TimingMap([(0, 120)], .125)
    end_seconds = duration_end(timing, 0, boundary, slide=kind == 'SLIDE_DURATION')
    limit = duration_query_limit(kind, timing=timing, local_end_tick=2000,
                                 audio_end_seconds=end_seconds, origin_tick=0)
    assert limit == expected_limit
    path = compile_time_target(kind, ticks=385, origin_tick=0, timing=timing,
                               duration_query_end_ticks=limit)
    assert [step.target_classes for step in path.steps] == expected_steps
    assert path.exact_ticks == 385


@pytest.mark.parametrize('kind', ['HOLD_DURATION', 'SLIDE_DURATION'])
def test_written_bpm_and_slide_launch_determine_boundary_queries(kind):
    timing = TimingMap([(0, 120), (2, 240), (6, 60)], .125)
    origin = 96
    end_seconds = duration_end(timing, origin, 384.25, slide=kind == 'SLIDE_DURATION')
    limit = duration_query_limit(kind, timing=timing, local_end_tick=8 * 96,
                                 audio_end_seconds=end_seconds, onset_tick=origin)
    assert limit == 385
    path = compile_time_target(kind, ticks=385, origin_tick=origin, timing=timing,
                               duration_query_end_ticks=limit)
    assert len(path.steps) == 2
    last = path.steps[-1]
    expected_seconds = duration_end(timing, origin, 384, slide=kind == 'SLIDE_DURATION')
    assert timing.beat_to_second(last.query_beat) == pytest.approx(expected_seconds)
    assert expected_seconds < end_seconds
    assert last.query_beat != pytest.approx((origin + 384) / 96 + (kind == 'SLIDE_DURATION'))
    assert last.target_classes == (0,)


@pytest.mark.parametrize('kind', ['HOLD_DURATION', 'SLIDE_DURATION'])
def test_training_tree_retains_full_source_support_and_inference_keeps_strict_cap(kind):
    timing = TimingMap([(0, 120)])
    source = _stage(kind)
    before = deepcopy(source)
    # Slide motion starts one beat after NOTE, so both relative limits are 500.
    cap = 500 + (96 if kind == 'SLIDE_DURATION' else 0)
    trained = compile_support_tree(source, timing, kind=kind, window_end=384,
                                   cap_tick=cap, target=_target(kind, 1900))
    assert source == before
    assert trained.elapsed_ticks == (0, 384)
    assert trained.target_path.exact_ticks == 1900
    assert trained.target_steps == (True, True)
    assert [step.target_classes for step in trained.target_path.steps] == [(CONTINUE,), (CONTINUE,)]
    assert trained.continuation_closes == (False, True)
    assert trained.basic[-1][383]  # Complete 768 remains a valid short candidate.
    assert trained.basic[-1][CONTINUE]
    inferred = compile_support_tree(source, timing, kind=kind, window_end=384, cap_tick=cap)
    assert inferred.target_path is None
    assert inferred.elapsed_ticks == (0, 384)
    assert not any(inferred.continuation_closes)
    assert inferred.basic[-1][115]  # 384 + 116 == 500 is inclusive.
    assert not any(inferred.basic[-1][116:])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_outside_short_keeps_physical_penalty_and_closed_continue_keeps_probability(dtype):
    timing = TimingMap([(0, 120)])
    source = _stage('HOLD_DURATION', intervals=[
        (2, 500, _record()), (501, 2000, _record(4)),
    ])
    tree = compile_support_tree(source, timing, kind='HOLD_DURATION', window_end=384,
                                cap_tick=500, target=HoldDuration(700))
    assert tree.elapsed_ticks == (0, 384)
    assert tree.target_path.exact_ticks == 700
    assert tree.target[-1][315]  # 384 + 316 == 700, outside the local window.
    assert tree.penalty[-1][315] == (False, False, True, False)
    assert not any(tree.penalty[-1][CONTINUE])
    allowed = torch.tensor(tree.basic)
    closes = torch.tensor(tree.continuation_closes)
    target = torch.tensor(tree.target)
    target_steps = torch.tensor(tree.target_steps)
    penalty = torch.tensor(tree.penalty)
    logits = torch.zeros(2, TIME_CLASSES, dtype=dtype, requires_grad=True)
    leaves = complete_field_log_probabilities(logits, allowed, closes)
    torch.testing.assert_close(leaves.exp().sum(), torch.tensor(1.))
    assert leaves[-1, CONTINUE].exp() > 0
    physical = physical_categories(leaves, penalty)
    # There are 268 forbidden exact leaves, 501..768. The unresolved CONT tail
    # is a separate zero-penalty leaf, not a fabricated duration completion.
    expected = -math.log1p(-268 / (384 * 385))
    torch.testing.assert_close(physical, torch.tensor([0., 0., expected, 0.]), atol=1e-6, rtol=1e-4)
    nll = time_path_nll(logits, allowed, target, target_steps)
    torch.testing.assert_close(nll, torch.tensor(math.log(384) + math.log(385)))
    (nll + physical.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 315] < 0


def test_short_gt_does_not_hide_reachable_closed_tail_from_physical_distribution():
    timing = TimingMap([(0, 120)])
    tree = compile_support_tree(_stage('HOLD_DURATION'), timing, kind='HOLD_DURATION',
                                window_end=384, cap_tick=500, target=HoldDuration(2))
    assert tree.elapsed_ticks == (0, 384)
    assert tree.target_steps == (True, False)
    assert tree.continuation_closes == (False, True)
    logits = torch.randn(2, TIME_CLASSES, requires_grad=True)
    allowed = torch.tensor(tree.basic)
    leaves = complete_field_log_probabilities(logits, allowed, torch.tensor(tree.continuation_closes))
    torch.testing.assert_close(leaves.exp().sum(), torch.tensor(1.))
    nll = time_path_nll(logits, allowed, torch.tensor(tree.target), torch.tensor(tree.target_steps))
    nll.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[1]) == 0
