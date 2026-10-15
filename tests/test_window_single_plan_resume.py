"""Explicit centered-to-single objective migration preserves complete training state."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from maidata.decoder_training import dp_cache
from maidata.decoder_training.cache_schema import DP_SCHEMA, Z_REPRESENTATION
from maidata.decoder_training.frozen_cache import FrozenCache
from maidata.decoder_training.gt_plan_cache import select_plan_conditions
from maidata.draft.planner import select_planner_windows
from maidata.inference.__main__ import main as infer
from maidata.inference.checkpoint import JointCheckpoint
from maidata.joint_training import checkpoint, recipe
from maidata.joint_training.__main__ import main as train
from maidata.joint_training.network import JointTrainer
from test_chart_physical import geometry
from test_inference import config, producer_config
from test_inference_cache import inputs, spies
from test_inference_joint import arguments
from test_integer_joint_runtime import integer_config
from test_joint_runtime import assert_state_equal
from test_plan4_runtime import plan4_config
from test_window_single_plan_runtime import single_config, single_geometry_config, single_head, z_memory


@pytest.fixture
def migration(integer_config, tmp_path):
    paths = {key: str(tmp_path/'accepted'/key) for key in recipe.PATH_KEYS if key != 'joint_train_output'}
    old = recipe.resolve(dict(integer_config, **paths))
    current = recipe.resolve(dict(old, planner_layout='window-single', planner_stride_beats=None,
        decoder_train_gt_plan_cache=str(tmp_path/'accepted/legacy-gt'),
        joint_train_output=str(tmp_path/'new-output')))
    model = JointTrainer(old)
    opt = recipe.optimizer(model.parameters(), old)
    steps = [1]*old['joint_train_scheduler_epochs']
    scheduler = recipe.scheduler(opt, steps, old)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)*.001
    opt.step()
    scheduler.step()
    opt.zero_grad(set_to_none=True)
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(seed), _iterator=object())
               for seed in (8, 19)]
    payload = checkpoint.capture(model, opt, scheduler, epoch=1, step=1, config=old,
        binding=recipe.provenance(old), source={'commit': 'centered-source'},
        steps=steps, loaders=loaders, best=.25, metrics={'epoch': 1})
    return old, current, payload, steps


def test_controlled_single_upgrade_preserves_all_states_and_new_checkpoint_inference(migration, tmp_path):
    old, current, payload, steps = migration
    original = deepcopy(payload)
    expected_random = random.random(), np.random.rand(3), torch.rand(3)
    model = JointTrainer(current)
    opt = recipe.optimizer(model.parameters(), current)
    scheduler = recipe.scheduler(opt, steps, current)
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(999), _iterator=object()) for _ in range(2)]
    assert checkpoint.restore(payload, model, opt, scheduler, config=current,
        binding=recipe.provenance(current), steps=steps, loaders=loaders,
        upgrade_single_plan16=True) == (1, 1, .25)
    inherited = checkpoint.capture(model, opt, scheduler, epoch=1, step=1, config=current,
        binding=recipe.provenance(current), source={'commit': 'single-source'}, steps=steps,
        loaders=loaders, best=.25, metrics={'epoch': 1})
    for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng', 'loader_rng',
                'loader_started', 'step', 'completed_epochs', 'best_validation_loss'):
        assert_state_equal(inherited[key], original[key])
    assert_state_equal(payload, original)  # The checkpoint's original recipe is never rewritten.
    assert random.random() == expected_random[0]
    np.testing.assert_equal(np.random.rand(3), expected_random[1])
    torch.testing.assert_close(torch.rand(3), expected_random[2], atol=0, rtol=0)
    path = tmp_path/'single-plan16.ckpt'
    torch.save(inherited, path)
    loaded = JointCheckpoint(path).models(device='cpu')
    assert loaded.config['planner_layout'] == 'window-single'
    assert loaded.config['draft_z_anchor_mode'] == 'integer' and loaded.config['draft_z_stride_beats'] == 1
    assert loaded.config['reconstruction_span_beats'] == 4
    assert loaded.planner.layout == loaded.decoder.planner_layout == 'window-single'
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        assert_state_equal(getattr(loaded, name).state_dict(), getattr(model, name).state_dict())
    # Ordinary continuation of the newly saved objective no longer needs the migration flag.
    checkpoint.restore(inherited, model, opt, scheduler, config=current, loaders=loaders)


@pytest.mark.parametrize('key,value', [
    ('joint_train_learning_rate', 9e-4), ('joint_train_workers', 1),
    ('joint_train_scheduler_epochs', 7), ('joint_train_seed', 17),
    ('decoder_train_chunk_token_budget', 9), ('joint_train_stop_after_epochs', 5),
    ('draft_z_stride_beats', 4), ('reconstruction_span_beats', 8),
    ('draft_z_anchor_mode', 'block-center'), ('planner_layout', 'legacy'),
    ('planner_stride_beats', 4), ('planner_span_beats', 4),
])
def test_single_upgrade_rejects_any_unapproved_recipe_change_before_weight_load(migration, key, value):
    _, current, payload, steps = migration
    changed = dict(current, **{key: value})
    model = JointTrainer(current)
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('weights loaded before gate')):
        with pytest.raises(ValueError, match='single Plan16 upgrade'):
            checkpoint.restore(payload, model, None, None, config=changed,
                binding=recipe.provenance(current), steps=steps,
                loaders=[None, None], upgrade_single_plan16=True)


@pytest.mark.parametrize('damage', ['steps', 'input_path', 'missing_binding', 'wrong_direction', 'both_flags', 'cursor'])
def test_single_upgrade_keeps_schedule_binding_direction_and_complete_epoch_gates(migration, damage):
    _, current, payload, steps = migration
    binding = recipe.provenance(current)
    if damage == 'steps':
        steps = [*steps[:-1], steps[-1]+1]
    elif damage == 'input_path':
        binding['paths']['pretrain_train_plan'] += '-changed'
    elif damage == 'missing_binding':
        binding = {}
    elif damage == 'wrong_direction':
        payload['recipe']['planner_layout'] = 'window-single'
    elif damage == 'cursor':
        payload['step'] += 1
    model = JointTrainer(current)
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('weights loaded before gate')):
        with pytest.raises(ValueError):
            checkpoint.restore(payload, model, None, None, config=current, binding=binding, steps=steps,
                loaders=[None, None], upgrade_single_plan16=True, upgrade_z_normalization=damage == 'both_flags')


def test_default_resume_rejects_objective_change_and_migration_keeps_strict_weights(migration):
    _, current, payload, steps = migration
    model = JointTrainer(current)
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('weights loaded before gate')):
        with pytest.raises(ValueError, match='architecture/representation'):
            checkpoint.restore(payload, model, None, None, config=current, loaders=[None, None])
    payload['model_state'].pop('planner.query')
    with pytest.raises(RuntimeError, match='Missing key'):
        checkpoint.restore(payload, model, None, None, config=current, binding=recipe.provenance(current),
            steps=steps, loaders=[None, None], upgrade_single_plan16=True)


def test_single_upgrade_cli_requires_resume_and_forwards_only_explicit_flag(tmp_path):
    for args in (['--upgrade-single-plan16', '--show-config'],
                 ['--resume', 'saved.ckpt', '--upgrade-single-plan16', '--upgrade-z-normalization', '--show-config']):
        with pytest.raises(SystemExit):
            train(args)
    override = tmp_path/'single.json'
    override.write_text(json.dumps(dict(planner_layout='window-single', planner_stride_beats=None,
        draft_z_anchor_mode='integer', draft_z_stride_beats=1, joint_train_z_stride_beats=1,
        reconstruction_span_beats=4)))
    with patch('maidata.joint_training.runner.run', return_value={}) as run:
        assert train(['--config', str(override), '--resume', 'saved.ckpt', '--upgrade-single-plan16']) == 0
    assert run.call_args.kwargs['upgrade_single_plan16'] is True
    assert run.call_args.kwargs['upgrade_z_normalization'] is False
    assert run.call_args.args[0]['planner_layout'] == 'window-single'


@pytest.mark.parametrize('z_enabled,source', [(True, 'predicted'), (False, 'predicted'),
                                            (True, 'none'), (False, 'none'), (False, 'gt')])
def test_single_plan_online_cold_hot_match_with_independent_switches(single_config, tmp_path, geometry,
                                                                    z_enabled, source):
    path, _ = inputs(single_config, tmp_path)
    args = [*arguments(path, tmp_path), '--grid', str(tmp_path/'grid.npz'), '--plan-source', source]
    if not z_enabled:
        args += ['--no-z-cross-attention']
    if source == 'gt':
        gt = tmp_path/'gt.json'
        gt.write_text(json.dumps(dict(plan_span_beats=16, windows=[dict(start_beat=start, valid=True,
            plan12=[0.]*8+[1., 0., 0., 0.]) for start in (0, 16)])))
        args += ['--gt-plan', str(gt)]
    reports, charts = [], []
    for mode in ('online', 'cold', 'hot'):
        forbid = ('encoder', 'draft', 'planner') if mode == 'hot' else (
            ('draft', 'planner') if not z_enabled and source != 'predicted' else ())
        with ExitStack() as stack:
            calls = spies(stack, forbid=forbid)
            stack.enter_context(patch('maidata.physical.Geometry.from_resources', return_value=geometry))
            infer([*args, '--output', str(tmp_path/mode),
                   *([] if mode == 'online' else ['--cache-dir', str(tmp_path/'cache')])])
            assert calls['decoder'].call_count == 1
        reports.append(json.loads((tmp_path/mode/'generation.json').read_text('utf-8')))
        charts.append((tmp_path/mode/'maidata.txt').read_bytes())
    assert charts[0] == charts[1] == charts[2]
    assert reports[0]['windows'] == reports[1]['windows'] == reports[2]['windows']
    assert all(report['complete'] for report in reports)
    assert all(row['plan_beats'] is None and len(row['plan12']) == 12
               and row['plan_valid'] == (source != 'none') for row in reports[0]['windows'])
    if z_enabled or source == 'predicted':
        assert reports[1]['cache']['draft']['hit'] is False
        assert reports[2]['cache']['draft']['hit'] is True
        saved = torch.load(next((tmp_path/'cache/draft').glob('*.pt')), weights_only=True)
        assert saved['z']['beats'].tolist() == [list(range(33))]
        assert saved['plans'].shape == (2, 12) and saved['plan_beats'] is None


def test_single_predicted_dp_export_roundtrip_and_layout_identity(tmp_path):
    c, z, model = single_geometry_config(), z_memory(0, 64), single_head()
    starts = torch.tensor([0., 16.])
    output = model(select_planner_windows(z, torch.tensor([0, 0]), starts, torch.ones(2, dtype=torch.bool), c))
    batch = dict(variant='song', chart_to_group={'A': 0}, conditions=[], groups=torch.tensor([0, 0]),
                 starts=starts, window_ids=['A:0', 'A:16'], chart_ids=['A', 'A'])
    path = tmp_path/'banks/song.pt'
    metadata = dp_cache.save_outputs(path, batch, z, output, 'producer', z_stride_beats=1,
        plan_span_beats=16, planner_layout='window-single', planner_stride_beats=None,
        z_anchor_mode='integer', reconstruction_span_beats=4)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION,
        z_width=4, z_stride_beats=1, z_anchor_mode='integer', planner_layout='window-single',
        plan_span_beats=16, plan_stride_beats=None,
        producer={'recipe': {'draft_z_stride_beats': 1, 'draft_z_anchor_mode': 'integer'}}, groups={'song': metadata})))
    cache = FrozenCache([], dict(c, decoder_train_dp_cache=str(tmp_path)), {})
    dp = cache.read('song')
    values, valid, centers = select_plan_conditions(dp['planner'], 'A:16', 'A', 16, c,
                                                   window_index=dp['window_index'])
    torch.testing.assert_close(values, output.plan_values[1], atol=0, rtol=0)
    assert valid and centers is None
    payload = torch.load(path, weights_only=True)
    payload['planner_layout'] = 'legacy'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='fixed Draft representation'):
        cache.read('song')
