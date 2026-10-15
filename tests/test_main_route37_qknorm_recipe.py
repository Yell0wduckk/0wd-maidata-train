"""Prepared main recipe: expand capacity and normalize Q/K, preserve Route37."""
from contextlib import ExitStack
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens
from maidata.draft.types import BeatMemory
from maidata.draft.windows import select_windows
from maidata.entry import CONFIG
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.conditioning import prepare_joint_conditions
from maidata.joint_training.__main__ import main as training_cli
from maidata.joint_training import checkpoint as joint_checkpoint_state
from maidata.joint_training.network import JointTrainer
from maidata.joint_training.recipe import resolve
from maidata.time import TimingMap

from test_inference import config, producer_config
from test_inference_joint import joint_config, joint_checkpoint
from test_pretrain_runtime import item


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/'configs/joint-main320-z4-plan16-qknorm-route37-gt.json'
LARGE_RECIPE = ROOT/'configs/joint-main320-z4-plan16-qknorm-route37-large-gt.json'
# Small config-only snapshot from old-full-run.json; no external training assets.
REFERENCE = ROOT/'tests/fixtures/route37-original-config.json'
QK_FLAGS = ('qk_normalization', 'draft_qk_normalization',
            'planner_qk_normalization', 'decoder_qk_normalization')
ARCHITECTURE_CHANGES = dict(d_model=320, ffn_dim=1280, draft_dim=320, draft_latent_dim=320,
    draft_expert_ffn_dim=640, decoder_dim=320, decoder_layers=8, decoder_shared_ffn_dim=1280,
    decoder_expert_ffn_dim=640, decoder_note_dims=[80, 160, 20, 20, 20, 10, 10],
    decoder_route_dims=[300, 20])
OUTPUT_KEYS = ('joint_train_output', 'pretrain_output', 'decoder_train_output')


def configuration(recipe=RECIPE):
    # Normalize tuples like the JSON runtime recipe, without opening any payloads.
    overrides = json.loads(recipe.read_text(encoding='utf-8'))
    assert not set(overrides)-set(CONFIG)
    return json.loads(json.dumps(resolve(dict(CONFIG, **overrides))))


@pytest.mark.parametrize('recipe,groups,budget,learning_rate', [
    (RECIPE, 2, 45056, 4e-4), (LARGE_RECIPE, 3, 80000, 6e-4)])
def test_cli_recipe_preserves_route37_optimizer_batch_and_scheduler_contract(capsys, recipe, groups, budget, learning_rate):
    assert training_cli(['--config', str(recipe), '--show-config']) == 0
    c = json.loads(capsys.readouterr().out)
    assert c == configuration(recipe)
    assert (c['heads'], c['draft_heads'], c['decoder_heads']) == (8, 8, 8)
    assert c['decoder_router_layer'] == 2
    for key, expected in ARCHITECTURE_CHANGES.items():
        assert c[key] == expected, key
    assert all(c[key] is True for key in QK_FLAGS)
    assert (c['pretrain_audio_groups_per_batch'], c['pretrain_token_budget']) == (groups, budget)
    assert c['pretrain_gradient_accumulation_steps'] == c['decoder_train_gradient_accumulation_steps'] == 1
    assert c['decoder_train_chunk_token_budget'] == 32768 and c['decoder_train_row_budget'] == 8192
    for prefix in ('joint_train_', 'pretrain_', 'decoder_train_'):
        assert c[prefix+'learning_rate'] == learning_rate
        assert c[prefix+'optimizer'] == 'adamw'
        assert c[prefix+'weight_decay'] == .01 and c[prefix+'gradient_clip'] == 1.
        assert c[prefix+'warmup_steps'] == 200
        assert c[prefix+'stop_after_epochs'] == 90 and c[prefix+'scheduler_epochs'] == 150
        assert c[prefix+'precision'] == 'bf16' and c[prefix+'workers'] == 12
        assert c[prefix+'seed'] == 20261010
    assert c['joint_train_initialization'] == c['pretrain_initialization'] == 'scratch'
    assert c['joint_train_z_stride_beats'] == c['draft_z_stride_beats'] == 4
    assert c['planner_span_beats'] == 16
    assert c['joint_train_plan_source'] == c['decoder_train_plan_source'] == 'gt'
    assert CONFIG['decoder_train_plan_input'] is True and c['decoder_train_plan_input'] is True
    assert c['decoder_z_normalization'] and c['fpn_p0_normalization']
    assert c['reconstruction_route_vocabulary'] == 'axis15-37'
    assert c['reconstruction_route_loss'] == 'acceptance37'
    assert c['reconstruction_count_supervision'] is False and c['reconstruction_count_loss'] == 'mse'
    assert c['planner_continuous_weight'] == 1. and c['planner_category_weight'] == .25
    # This main merge contains neither integer/overlap geometry nor onset supervision.
    assert not {'planner_layout', 'planner_stride_beats', 'draft_z_anchor_mode',
                'reconstruction_span_beats', 'reconstruction_onset_presence_supervision'} & c.keys()


def test_only_authorized_values_differ_from_saved_full_route37_metadata():
    old = json.loads(REFERENCE.read_text(encoding='utf-8'))
    old.setdefault('planner_span_beats', 16)
    for key in QK_FLAGS:
        old.setdefault(key, False)
    current = configuration()
    expected = dict(old, **ARCHITECTURE_CHANGES, **dict.fromkeys(QK_FLAGS, True),
                    decoder_train_chunk_token_budget=32768)
    assert len({current[key] for key in OUTPUT_KEYS}) == 1
    assert current['joint_train_output'] != old['joint_train_output']
    for key in OUTPUT_KEYS:
        expected[key] = current[key]
    differences = {key: (expected.get(key), current.get(key)) for key in set(expected) | set(current)
                   if expected.get(key) != current.get(key) or (key in expected) != (key in current)}
    assert not differences
    changed = {key for key in old if old[key] != current[key] and key not in OUTPUT_KEYS}
    assert changed == set(ARCHITECTURE_CHANGES) | set(QK_FLAGS) | {'decoder_train_chunk_token_budget'}


def test_large_recipe_changes_only_groups_budget_learning_rate_and_output_aliases():
    small, large = configuration(), configuration(LARGE_RECIPE)
    allowed = dict(pretrain_audio_groups_per_batch=3, pretrain_token_budget=80000,
                   joint_train_learning_rate=6e-4, pretrain_learning_rate=6e-4,
                   decoder_train_learning_rate=6e-4)
    assert len({large[key] for key in OUTPUT_KEYS}) == 1
    assert large['joint_train_output'] != small['joint_train_output']
    allowed.update({key: large[key] for key in OUTPUT_KEYS})
    assert large == dict(small, **allowed)
    # Also check the serialized source, so stale aliases cannot be hidden by resolve().
    raw_small = json.loads(RECIPE.read_text(encoding='utf-8'))
    raw_large = json.loads(LARGE_RECIPE.read_text(encoding='utf-8'))
    assert raw_large == dict(raw_small, **allowed)
    changed = {key for key in raw_small if raw_small[key] != raw_large[key]}
    assert changed == set(allowed)


def test_actual_320_m8_model_has_no_parameter_growth_from_qk_normalization():
    torch.set_num_threads(1)
    c = configuration()
    enabled = JointTrainer(c)
    disabled = JointTrainer(dict(c, **dict.fromkeys(QK_FLAGS, False)))
    expected = dict(encoder=11115880, draft=21488672, reconstruction=815703,
                    planner=151692, decoder=22606325)
    assert {name: sum(parameter.numel() for parameter in getattr(enabled, name).parameters())
            for name in expected} == expected
    assert sum(parameter.numel() for parameter in enabled.parameters()) == 56_178_272
    assert {key: value.shape for key, value in enabled.state_dict().items()} == {
        key: value.shape for key, value in disabled.state_dict().items()}
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        flags = [module.qk_normalization for module in getattr(enabled, name).modules()
                 if hasattr(module, 'qk_normalization')]
        assert flags and all(flag is True for flag in flags), name
    assert not hasattr(enabled.reconstruction, 'onset_presence_head')
    assert enabled.planner.span_beats == 16 and enabled.planner.pool.width//enabled.planner.pool.heads == 32
    assert len(enabled.decoder.layers) == 8 and enabled.decoder.dim//enabled.decoder.layers[0].self_attn.heads == 40


def test_complete_old_joint_resume_missing_qk_options_restores_all_state(joint_config):
    torch.set_num_threads(1)
    old = resolve({key: value for key, value in joint_config.items() if key not in QK_FLAGS})
    model = JointTrainer(old)
    optimizer = torch.optim.AdamW(model.parameters(), lr=old['joint_train_learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .001)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(seed), _iterator=None)
               for seed in (17, 31)]
    payload = joint_checkpoint_state.capture(model, optimizer, scheduler, epoch=1, step=1,
        config=old, binding={}, source=dict(commit='old-joint-no-qk-fields'), steps=[1]*150,
        loaders=loaders, best=.4, metrics=dict(epoch=1))
    assert not set(QK_FLAGS) & payload['recipe'].keys()
    expected_random = random.random(), np.random.rand(3), torch.rand(3)
    current = dict(old, **dict.fromkeys(QK_FLAGS, False))
    restored = JointTrainer(current)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=old['joint_train_learning_rate'])
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda _: 1.)
    new_loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(99), _iterator=None)
                   for _ in loaders]
    assert joint_checkpoint_state.restore(payload, restored, new_optimizer, new_scheduler,
        config=current, loaders=new_loaders) == (1, 1, .4)
    assert random.random() == expected_random[0]
    np.testing.assert_equal(np.random.rand(3), expected_random[1])
    torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    previous_state, current_state = optimizer.state_dict(), new_optimizer.state_dict()
    assert previous_state['param_groups'] == current_state['param_groups']
    assert previous_state['state'].keys() == current_state['state'].keys()
    for index, values in previous_state['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(current_state['state'][index][key], value, rtol=0, atol=0)
    assert new_scheduler.state_dict() == scheduler.state_dict()
    for prior, replacement in zip(loaders, new_loaders):
        assert torch.equal(prior.generator.get_state(), replacement.generator.get_state())


@pytest.mark.parametrize('key', QK_FLAGS)
def test_joint_resume_rejects_each_qk_change_before_loading_weights(joint_config, key):
    old = resolve({name: value for name, value in joint_config.items() if name not in QK_FLAGS})
    current = {**old, **dict.fromkeys(QK_FLAGS, False), key: True}
    model = JointTrainer(current)
    payload = dict(schema=joint_checkpoint_state.SCHEMA, architecture=joint_checkpoint_state.ARCHITECTURE,
        completed_epochs=1, step=1, steps_by_epoch=[1], scheduler_state=dict(last_epoch=1),
        loader_rng=[], loader_started=[], recipe=joint_checkpoint_state.recipe(old))
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('Loaded before semantic check')):
        with pytest.raises(ValueError, match='architecture/representation'):
            joint_checkpoint_state.restore(payload, model, None, None, config=current, loaders=[])


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_new_norm_retains_four_z_planner_and_six_z_plus_unpositioned_plan(joint_config, tmp_path, precision):
    torch.set_num_threads(1)
    c = dict(joint_config, **dict.fromkeys(QK_FLAGS, True),
             decoder_layers=8, planner_span_beats=16, decoder_z_normalization=True)
    path, _, _ = joint_checkpoint(c, tmp_path)
    models = JointCheckpoint(path).models(device='cpu')
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16'))
        planner = stack.enter_context(patch.object(models.planner, 'forward', wraps=models.planner.forward))
        pool = stack.enter_context(patch.object(models.planner.pool, 'forward', wraps=models.planner.pool.forward))
        conditions = prepare_joint_conditions(models.encoder, models.draft, models.planner, models.config,
            item(0, c)['grid'], TimingMap([(0, 120)]), 1.2,
            difficulty=5, level=14.5, device='cpu')
        assert conditions.starts == (0, 16)
        assert conditions.z.beats.tolist() == [[2., 6., 10., 14., 18., 22., 26., 30.]]
        assert conditions.plans.shape == (2, 12) and conditions.plan_valid.tolist() == [True, True]
        assert planner.call_count == pool.call_count == 1
        assert planner.call_args.args[0].memory.beats.tolist() == [[2., 6., 10., 14.], [18., 22., 26., 30.]]
        assert pool.call_args.args[4].sum(-1).tolist() == [4, 4]
        # Retain the original P pooling-query positions rather than adopting integer-plan centers.
        assert pool.call_args.args[1].tolist() == [[0.], [16.]]
        for index, start in enumerate(conditions.starts):
            z = select_windows(conditions.z, torch.tensor([0]), torch.tensor([float(start)]),
                torch.tensor([True]), target_beats=c['target_beats'],
                before_beats=c['decoder_z_before_beats'], after_beats=c['decoder_z_after_beats'],
                z_stride_beats=c['draft_z_stride_beats']).memory
            assert z.beats.tolist() == [[start-2., start+2., start+6., start+10., start+14., start+18.]]
            assert z.valid.tolist() == ([[False, True, True, True, True, True]] if index == 0
                                        else [[True, True, True, True, True, False]])
            audio = BeatMemory(*(getattr(conditions.audio, key)[index:index+1]
                                for key in ('values', 'beats', 'valid')))
            plan, valid = conditions.plans[index:index+1], conditions.plan_valid[index:index+1]
            condition = models.decoder.condition(z, plan, valid)
            memory = models.decoder.prepare_memory(models.decoder.prepare_audio(audio),
                torch.zeros(1, dtype=torch.long), z, plan, valid)
            assert memory.plan_index == 6 and condition.values.shape[1] == 7
            assert condition.beats[:, -1].count_nonzero() == 0
            for layer, projected in zip(models.decoder.layers, memory.condition):
                attention = layer.condition_attn
                raw = attention.normalize_qk(attention.split(attention.key(
                    condition.values.masked_fill(~condition.valid[..., None], 0))))
                # Plan receives Q/K normalization but continues using an unrotated key.
                torch.testing.assert_close(projected.key[:, :, 6:], raw[:, :, 6:], rtol=0, atol=0)
            tokens = pack_tokens([[Moment(0), Note.make('tap', 'A1')]],
                                 [[float(start), start+.5]], device='cpu')
            output = models.decoder(tokens, torch.tensor([[1, 0]]), memory)
            assert torch.isfinite(output.hidden).all() and torch.isfinite(output.type_logits).all()
