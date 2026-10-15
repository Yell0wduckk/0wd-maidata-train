"""Training plan gate removes packed keys while retaining Z and auxiliary labels."""
from dataclasses import replace

import pytest
import torch

from maidata.entry import CONFIG
from maidata.decoder_training import checkpoint, recipe
from maidata.decoder_training.data import collate_audio
from maidata.decoder_training.network import DecoderTrainer
from maidata.joint_training import recipe as joint_recipe
from maidata.joint_training.data import collate_audio as collate_joint
from test_decoder_training import config, encoder_config
from test_decoder_plan_source import comparison_items
from test_joint_data import joint_item


def test_default_and_checkpoint_recipe(config):
    assert CONFIG['decoder_train_plan_input'] is True
    for capture in (checkpoint.recipe, __import__(
            'maidata.joint_training.checkpoint', fromlist=['recipe']).recipe):
        assert capture(dict(config, decoder_train_plan_input=False))['decoder_train_plan_input'] is False


@pytest.mark.parametrize('validator', [recipe.validate, joint_recipe.validate])
@pytest.mark.parametrize('bad', [None, 0, 1, 'false'])
def test_gate_requires_boolean(config, validator, bad):
    with pytest.raises(ValueError, match='decoder_train_plan_input must be a boolean'):
        validator(dict(config, decoder_train_plan_input=bad))


@pytest.mark.parametrize('source', ['gt', 'predicted'])
def test_stage_two_gate_masks_packed_plan_keys_and_preserves_sampling(config, monkeypatch, source):
    c = dict(config, decoder_train_plan_source=source)
    items = comparison_items(c, monkeypatch)
    enabled = collate_audio(items, config=c)
    disabled = collate_audio(items, config=dict(c, decoder_train_plan_input=False))
    assert enabled.conditions.plan_valid.all()
    assert not disabled.conditions.plan_valid.any()
    assert disabled.conditions.plan.count_nonzero() == 0
    for name in ('audio', 'decoder_z'):
        for field in ('values', 'beats', 'valid'):
            torch.testing.assert_close(getattr(getattr(enabled.conditions, name), field),
                                       getattr(getattr(disabled.conditions, name), field))
    assert [w['identity'] for w in enabled.windows] == [w['identity'] for w in disabled.windows]
    width = disabled.conditions.decoder_z.valid.shape[1] + 1
    for on, off in zip(enabled.chunks, disabled.chunks):
        assert on.labels.counts == off.labels.counts
        keys = off.audio_layout['condition_rows']['key_indices']
        assert not ((keys % width) == width - 1).any()


def test_joint_gate_preserves_planner_supervision(config, monkeypatch):
    items = [joint_item(0, config, monkeypatch)]
    enabled = collate_joint(items, config=config)
    disabled = collate_joint(items, config=dict(config, decoder_train_plan_input=False))
    assert enabled.conditions.plan_valid.all() and not disabled.conditions.plan_valid.any()
    assert disabled.conditions.decoder_z.valid.any()
    for name in vars(enabled.planner):
        value, actual = getattr(enabled.planner, name), getattr(disabled.planner, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, actual)
    assert enabled.window_ids == disabled.window_ids


def test_disabled_plan_cannot_change_decoder_output_or_receive_gradient(config, monkeypatch):
    c = dict(config, decoder_train_plan_input=False)
    batch = collate_audio(comparison_items(c, monkeypatch), config=c)
    model = DecoderTrainer(c).eval()
    conditions = model.conditions(batch)
    conditions.plan.requires_grad_()
    conditions.decoder_z.values.requires_grad_()
    chunk = batch.chunks[0]
    audio_kv = model.decoder.prepare_audio(conditions.audio)
    memory = model.memory(conditions, audio_kv, chunk.indices, chunk.audio_layout)
    original = model.decoder(chunk.tokens, chunk.next_types, memory).hidden
    changed = replace(conditions, plan=torch.full_like(conditions.plan, 123456.))
    memory = model.memory(changed, audio_kv, chunk.indices, chunk.audio_layout)
    actual = model.decoder(chunk.tokens, chunk.next_types, memory).hidden
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    original.square().sum().backward()
    assert conditions.plan.grad is None or not conditions.plan.grad.any()
    assert conditions.decoder_z.values.grad.abs().sum() > 0
