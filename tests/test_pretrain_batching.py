"""Configured batch boundaries retain complete groups and every supervision row."""
import pytest

from maidata.entry import CONFIG
from maidata.training.batching import optimizer_batches


def test_default_two_whole_groups_without_accumulation():
    groups = [dict(audio_variant=str(i), chart_ids=[f'{i}:5', f'{i}:6'],
                   windows=list(range(i + 1))) for i in range(5)]
    steps = optimizer_batches(groups, CONFIG)
    assert [[len(batch) for batch in step] for step in steps] == [[2], [2], [1]]
    flattened = [group for step in steps for batch in step for group in batch]
    assert all(actual is expected for actual, expected in zip(flattened, groups))
    assert len(flattened) == len(groups)
    assert len(optimizer_batches(range(288), CONFIG)) == 144
    assert len(optimizer_batches(range(32), CONFIG)) == 16


def test_nondefault_sizes_and_incomplete_accumulation_keep_order():
    config = dict(CONFIG, pretrain_audio_groups_per_batch=3,
                  pretrain_gradient_accumulation_steps=2)
    steps = optimizer_batches(range(8), config)
    assert steps == (((0, 1, 2), (3, 4, 5)), ((6, 7),))
    assert optimizer_batches([], config) == ()


@pytest.mark.parametrize('key', ['pretrain_audio_groups_per_batch', 'pretrain_gradient_accumulation_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_batch_dimensions(key, value):
    with pytest.raises(ValueError, match='positive integers'):
        optimizer_batches(range(3), dict(CONFIG, **{key: value}))
