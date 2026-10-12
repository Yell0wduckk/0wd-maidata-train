"""Inference compatibility follows actual component weights, not training tags."""
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config
from test_inference_joint import joint_config, joint_checkpoint
from maidata.inference.checkpoint import JointCheckpoint, load_models
from maidata.model import build_model, build_draft, build_planner, build_decoder


def separate_checkpoints(config, tmp_path, *, tags=True, mode='joint'):
    producer = torch.nn.ModuleDict(dict(encoder=build_model(config), draft=build_draft(config),
                                       planner=build_planner(config)))
    consumer = torch.nn.ModuleDict(dict(encoder=build_model(config), decoder=build_decoder(config)))
    recipe = dict(config, decoder_train_audio_mode=mode,
                  decoder_train_p0_cache='unused-training-only-cache',
                  pretrain_chart_cache='unused-training-only-input')
    payloads = [dict(model_state=m.state_dict(), recipe=dict(recipe)) for m in (producer, consumer)]
    if tags:
        for p in payloads:
            p.update(schema='another-training.v999', architecture='another-training-recipe')
    paths = [tmp_path/'producer.ckpt', tmp_path/'consumer.ckpt']
    for p, path in zip(payloads, paths):
        torch.save(p, path)
    return paths, (producer, consumer), payloads


@pytest.mark.parametrize('tags', [True, False])
@pytest.mark.parametrize('mode', ['joint', 'cached', 'other-optimization-recipe'])
def test_separate_loaders_ignore_training_tags_and_mode(config, tmp_path, tags, mode):
    paths, original, _ = separate_checkpoints(config, tmp_path, tags=tags, mode=mode)
    with patch('maidata.decoder_training.producer.teacher_configuration', side_effect=AssertionError('training loader called')):
        loaded = load_models(*paths, device='cpu')
    assert loaded.encoder is not loaded.decoder_encoder
    for actual, expected in ((loaded.encoder, original[0]['encoder']),
                             (loaded.draft, original[0]['draft']), (loaded.planner, original[0]['planner']),
                             (loaded.decoder_encoder, original[1]['encoder']), (loaded.decoder, original[1]['decoder'])):
        assert not actual.training and not any(p.requires_grad for p in actual.parameters())
        assert all(torch.equal(value, actual.state_dict()[key]) for key, value in expected.state_dict().items())
    assert loaded.provenance['decoder']['schema'] == ('another-training.v999' if tags else None)


def test_decoder_encoder_is_loaded_even_without_version_tags_or_dp(config, tmp_path):
    paths, original, _ = separate_checkpoints(config, tmp_path, tags=False)
    loaded = load_models(None, paths[1], device='cpu', z_cross_attention=False, plan_source='none')
    assert loaded.encoder is None and loaded.draft is None and loaded.planner is None
    assert loaded.decoder_encoder is not None and not loaded.decoder.condition.z_cross_attention
    assert all(torch.equal(v, loaded.decoder_encoder.state_dict()[k]) for k, v in original[1]['encoder'].state_dict().items())


def test_older_recipe_keeps_unnormalized_encoder_math(config, tmp_path):
    paths, _, payloads = separate_checkpoints(dict(config, fpn_p0_normalization=False), tmp_path)
    for payload, path in zip(payloads, paths):
        payload['recipe'].pop('fpn_p0_normalization')
        torch.save(payload, path)
    loaded = load_models(*paths, device='cpu')
    assert loaded.config['fpn_p0_normalization'] is False
    assert loaded.teacher_config['fpn_p0_normalization'] is False


@pytest.mark.parametrize('component', ['encoder', 'draft', 'planner', 'decoder'])
@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape'])
def test_separate_checkpoints_keep_strict_weight_compatibility(config, tmp_path, component, damage):
    paths, _, payloads = separate_checkpoints(config, tmp_path)
    index = 0 if component in ('draft', 'planner') else 1
    state = payloads[index]['model_state']
    key = next(k for k, v in state.items() if k.startswith(component+'.') and v.ndim > 1)
    if damage == 'missing':
        state.pop(key)
    elif damage == 'unexpected':
        state[component+'.unknown.weight'] = torch.ones(1)
    else:
        state[key] = state[key].flatten()[:1]
    torch.save(payloads[index], paths[index])
    with pytest.raises(RuntimeError, match={'missing': 'Missing key', 'unexpected': 'Unexpected key', 'shape': 'size mismatch'}[damage]):
        load_models(*paths, device='cpu')


def test_matching_tags_do_not_hide_real_architecture_mismatch(joint_config, tmp_path):
    path, _, payload = joint_checkpoint(joint_config, tmp_path)
    payload['recipe']['decoder_layers'] += 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        JointCheckpoint(path).models(device='cpu')


def test_stages_load_only_consumed_components_and_still_check_them(joint_config, tmp_path):
    path, _, payload = joint_checkpoint(joint_config, tmp_path)
    payload.pop('schema'); payload.pop('architecture')
    payload['model_state']['decoder.extra.weight'] = torch.ones(1)
    torch.save(payload, path)
    checkpoint = JointCheckpoint(path)
    assert checkpoint.models(device='cpu', components=('encoder',)).encoder is not None
    with pytest.raises(RuntimeError, match='Unexpected key'):
        checkpoint.models(device='cpu', components=('decoder',))
