"""Targeted field-query tests; no source asset scans, CUDA or training launch."""
from fractions import Fraction
import math

import pytest
import torch

from maidata.chart.types import HoldDuration, Moment, SlideDuration
from maidata.chart.windows import TimeTarget
from maidata.time import TimingMap, duration_end
from maidata.time.discrete import (CONTINUE, TIME_CLASSES, TimeQuery,
    compile_support_tree, compile_time_target,
    decode_ticks, duration_cap, encode_ticks, local_z_indices, substep_masks)
from maidata.time.distribution import (complete_field_log_probabilities,
    masked_log_probs, physical_categories, physical_mean, time_path_nll)


def record(*, basic=(), physical=0, penalty=None):
    return dict(basic=basic, physical=physical, penalty=physical if penalty is None else penalty)


def stage(kind, lower, upper, intervals=None, origin=0):
    return dict(cursor_tick=origin, **{kind: dict(domain=[lower, upper],
        intervals=intervals or [(lower, upper, record())], outside_basic='outside')})


@pytest.mark.parametrize('ticks,path', [
    (1, (0,)), (2, (1,)), (383, (382,)), (384, (383,)),
    (385, (CONTINUE, 0)), (768, (CONTINUE, 383)),
    (769, (CONTINUE, CONTINUE, 0))])
def test_unique_encoding(ticks, path):
    assert encode_ticks(ticks) == path
    assert decode_ticks(path) == ticks


@pytest.mark.parametrize('path', [(), (CONTINUE,), (0, 1), (0, CONTINUE), (385,), (-1,)])
def test_incomplete_or_noncanonical_paths_rejected(path):
    with pytest.raises(ValueError):
        decode_ticks(path)


def test_hold_minimum_applies_to_total_not_remainder():
    timing = TimingMap([(0, 120)])
    with pytest.raises(ValueError, match='minimum'):
        compile_time_target('HOLD_DURATION', ticks=1, origin_tick=0, timing=timing)
    path = compile_time_target('HOLD_DURATION', ticks=385, origin_tick=0, timing=timing)
    assert [step.target_classes for step in path.steps] == [(CONTINUE,), (0,)]
    assert TimeQuery('HOLD_DURATION', 0, 384).advance(0).complete_ticks == 385


def test_moment_first_crossing_closes_without_fabricated_onset():
    timing = TimingMap([(0, 120)])
    query = TimeQuery('MOMENT', 123)
    advance = query.advance(CONTINUE, window_end_tick=500)
    assert advance.query.elapsed_ticks == 384
    assert advance.complete_ticks is None
    assert advance.lower_bound_tick == 508
    assert advance.closes_window and advance.closed_by_continue
    exact = compile_time_target('MOMENT', ticks=900, origin_tick=123,
                                timing=timing, window_end_tick=500)
    assert len(exact.steps) == 1
    assert exact.exact_ticks == 900
    assert exact.closed_by_continue
    assert exact.closed_lower_bound_tick == 508
    at_boundary = compile_time_target('MOMENT', ticks=377, origin_tick=123,
                                      timing=timing, window_end_tick=500)
    assert at_boundary.closes_window and not at_boundary.closed_by_continue


def test_censor_sets_and_cross_window_truncation():
    timing = TimingMap([(0, 120)])
    tail = compile_time_target('MOMENT', censored_lower=400, origin_tick=0,
                               timing=timing, window_end_tick=390)
    assert [s.target_classes for s in tail.steps] == [(CONTINUE,), tuple(range(15, 385))]
    assert tail.censored and tail.closes_window
    assert tail.censored_lower == 400
    early_close = compile_time_target('MOMENT', censored_lower=1000, origin_tick=20,
                                      timing=timing, window_end_tick=400)
    assert len(early_close.steps) == 1 and early_close.closed_by_continue
    assert not early_close.exact_ticks


def test_written_bpm_query_and_local_cap_across_tempo_changes():
    timing = TimingMap([(0, 120), (2, 240), (6, 60)], .125)
    origin = 96
    for kind in ('HOLD_DURATION', 'SLIDE_DURATION'):
        query = TimeQuery(kind, origin, 384)
        expected_seconds = duration_end(timing, origin, 384, slide=kind == 'SLIDE_DURATION')
        assert timing.beat_to_second(query.query_beat(timing)) == pytest.approx(expected_seconds)
        # The written-BPM tick duration crosses two global tempo segments.
        assert query.query_beat(timing) != (origin + 384) / 96 + (kind == 'SLIDE_DURATION')
        cap = duration_cap(kind, timing=timing, origin_tick=origin, local_end_tick=8*96,
                           audio_end_seconds=timing.second(7*96))
        assert duration_end(timing, origin, cap, slide=kind == 'SLIDE_DURATION') <= timing.second(7*96) + 1e-10
        assert duration_end(timing, origin, cap+1, slide=kind == 'SLIDE_DURATION') > timing.second(7*96)
    assert TimeQuery('SLIDE_DURATION', origin).query_beat(timing) == pytest.approx(2.)
    assert local_z_indices(3.75) == (2, 3, 4)
    assert local_z_indices(3.75, moment=True) == (3, 4, 5, 6, 7)


def test_joint_cont_support_not_categorywise_and_and_cap():
    # The tail has completions, each violating a different class. There is no
    # jointly hard-legal completion despite category-wise existence.
    intervals = [(2, 384, record()), (385, 500, record(physical=1)),
                 (501, 600, record(physical=2))]
    masks = substep_masks('HOLD_DURATION', intervals, origin_tick=0, upper_tick=600)
    assert masks.basic[CONTINUE] and not masks.hard[CONTINUE]
    assert not masks.basic[0] and masks.basic[1]
    truncated = substep_masks('HOLD_DURATION', intervals, origin_tick=0, upper_tick=384)
    assert not truncated.basic[CONTINUE]
    remainder_one = substep_masks('HOLD_DURATION', intervals, origin_tick=0,
                                  elapsed_ticks=384, upper_tick=385)
    assert remainder_one.basic[0]
    assert not remainder_one.basic[CONTINUE]


def tree_tensors(tree):
    return {key: torch.tensor(getattr(tree, key), dtype=torch.bool) for key in
            ('basic', 'penalty', 'continuation_closes', 'target', 'target_steps')}


def test_all_depths_and_full_leaf_mass_including_window_close():
    timing = TimingMap([(0, 120)])
    tree = compile_support_tree(stage('moment', 1, 3000), timing, kind='MOMENT',
                               window_end=1000, cap_tick=1400, target=Moment(timing.delta_ms(0, 1)))
    assert len(tree.supports) == 3  # GT uses only one depth; probability does not.
    assert tree.target_steps == (True, False, False)
    tensors = tree_tensors(tree)
    logits = torch.randn(3, TIME_CLASSES, requires_grad=True)
    leaves = complete_field_log_probabilities(logits, tensors['basic'], tensors['continuation_closes'])
    torch.testing.assert_close(leaves.exp().sum(), torch.tensor(1.))
    assert torch.isneginf(leaves[:2, CONTINUE]).all()
    assert torch.isfinite(leaves[2, CONTINUE])
    # Closing shorts and closing CONT carry no first-responsibility mass.
    assert not tensors['penalty'][2, 231:].any()  # 768+232 reaches window end
    loss = time_path_nll(logits, tensors['basic'], tensors['target'], tensors['target_steps'])
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[1:]) == 0


def test_nonmonotone_support_full_value_physical_and_field_count():
    timing = TimingMap([(0, 120)])
    support = stage('duration', 2, 800, [(2, 200, record(physical=1)),
        (201, 384, record()), (385, 600, record(physical=2)),
        (601, 799, record(physical=4)), (800, 800, record())])
    tree = compile_support_tree(support, timing, kind='HOLD_DURATION', window_end=1536,
                               cap_tick=1920, target=HoldDuration(2))
    assert len(tree.supports) == 3
    tensors = tree_tensors(tree)
    logits = torch.zeros(3, TIME_CLASSES, requires_grad=True)
    leaves = complete_field_log_probabilities(logits, tensors['basic'], tensors['continuation_closes'])
    torch.testing.assert_close(leaves.exp().sum(), torch.tensor(1.))
    categories = physical_categories(leaves, tensors['penalty'])
    mass = (leaves.exp()[..., None] * tensors['penalty']).sum((0, 1))
    torch.testing.assert_close(categories, -torch.log1p(-mass))
    # This remains a single original-field numerator, despite three queries.
    assert categories.shape == (4,)
    objective = physical_mean(categories) + time_path_nll(logits, tensors['basic'], tensors['target'], tensors['target_steps'])
    objective.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[2].abs().sum() > 0


def test_censored_set_nll_and_batched_padded_tree():
    timing = TimingMap([(0, 120)])
    tree = compile_support_tree(stage('moment', 1, 1000), timing, kind='MOMENT',
        window_end=300, cap_tick=350, target=TimeTarget(timing.delta_ms(0, 350), 350, True, True))
    tensors = tree_tensors(tree)
    logits = torch.zeros(2, 2, TIME_CLASSES, requires_grad=True)
    basic = torch.zeros_like(logits, dtype=torch.bool)
    basic[:, 0] = tensors['basic']
    closes = torch.tensor([[True, False], [True, False]])
    target = torch.zeros_like(basic)
    target[:, 0] = tensors['target']
    active = torch.tensor([[True, False], [True, False]])
    leaves = complete_field_log_probabilities(logits, basic, closes)
    torch.testing.assert_close(leaves.exp().sum((-1, -2)), torch.ones(2))
    nll = time_path_nll(logits, basic, target, active)
    expected = -math.log((384 - 350 + 1 + 1) / 385)
    torch.testing.assert_close(nll, torch.full((2,), expected))
    (nll.sum() + physical_categories(leaves, torch.zeros((*logits.shape, 4), dtype=torch.bool)).sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_all_forbidden_physical_has_finite_floor_and_gradient():
    logits = torch.randn(1, 385, requires_grad=True)
    basic = torch.ones_like(logits, dtype=torch.bool)
    closes = torch.ones(1, dtype=torch.bool)
    leaves = complete_field_log_probabilities(logits, basic, closes)
    penalty = torch.ones(1, 385, 4, dtype=torch.bool)
    result = physical_categories(leaves, penalty)
    torch.testing.assert_close(result, torch.full((4,), -math.log(torch.finfo(torch.float32).eps)))
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_truncated_cont_chain_is_rejected():
    logits = torch.zeros(1, 385)
    with pytest.raises(ValueError, match='tail probability'):
        complete_field_log_probabilities(logits, torch.ones_like(logits, dtype=torch.bool),
                                         torch.zeros(1, dtype=torch.bool))


def test_local_duration_query_prefix_and_inference_no_gt():
    timing = TimingMap([(0, 120), (2, 240)])
    support = stage('duration', 1, 10000, origin=96)
    trained = compile_support_tree(support, timing, kind='SLIDE_DURATION', window_end=384,
                                   cap_tick=480, target=SlideDuration(1000))
    assert trained.target_path.exact_ticks == 1000
    assert trained.target_path.closed_by_continue
    assert trained.elapsed_ticks == (0,)
    tree = compile_support_tree(support, timing, kind='SLIDE_DURATION', window_end=384, cap_tick=480)
    assert tree.target_path is None and not any(tree.target_steps)
    assert not any(any(row) for row in tree.target)
