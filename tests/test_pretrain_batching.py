"""Configured batch boundaries retain complete groups and every supervision row."""
import pytest

from maidata.entry import CONFIG
from maidata.training.batching import optimizer_batches, group_token_cost


def test_default_two_whole_groups_without_accumulation():
    groups = [dict(audio_variant=str(i), chart_ids=[f'{i}:5', f'{i}:6'],
                   windows=list(range(i + 1))) for i in range(5)]
    steps = optimizer_batches(groups, CONFIG, token_costs=[100]*5)
    assert [[len(batch) for batch in step] for step in steps] == [[2], [2], [1]]
    flattened = [group for step in steps for batch in step for group in batch]
    assert all(actual is expected for actual, expected in zip(flattened, groups))
    assert len(flattened) == len(groups)
    assert len(optimizer_batches(range(288), CONFIG, token_costs=[100]*288)) == 144
    assert len(optimizer_batches(range(32), CONFIG, token_costs=[100]*32)) == 16


def test_nondefault_sizes_and_incomplete_accumulation_keep_order():
    config = dict(CONFIG, pretrain_audio_groups_per_batch=3,
                  pretrain_gradient_accumulation_steps=2, pretrain_token_budget=None)
    steps = optimizer_batches(range(8), config)
    assert steps == (((0, 1, 2), (3, 4, 5)), ((6, 7),))
    assert optimizer_batches([], config) == ()


@pytest.mark.parametrize('key', ['pretrain_audio_groups_per_batch', 'pretrain_gradient_accumulation_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_batch_dimensions(key, value):
    with pytest.raises(ValueError, match='positive integers'):
        optimizer_batches(range(3), dict(CONFIG, **{key: value}))


def test_budget_and_group_limit_jointly_preserve_order_and_accumulation():
    c = dict(CONFIG, pretrain_token_budget=100, pretrain_gradient_accumulation_steps=2)
    assert optimizer_batches(range(5), c, token_costs=[60,60,40,80,20]) == (((0,), (1,2)), ((3,4),))
    with pytest.raises(ValueError, match='above token budget'):
        optimizer_batches(['oversize'], c, token_costs=[101])
    with pytest.raises(ValueError, match='requires one cost'):
        optimizer_batches([0], c)


def test_cost_counts_overlap_and_empty_tail_queries_not_gt_rows():
    g = dict(encoder_starts_beats=[0,16], actual_audio_end_beat=31.2,first_beat=0,occurrences=6)
    assert group_token_cost(g, CONFIG) == 2*24*24+32+6*207
    c=dict(CONFIG,pretrain_audio_token_weight=2,pretrain_row_token_weight=.5)
    assert group_token_cost(g,c)==2*(2*24*24+32)+.5*6*207
