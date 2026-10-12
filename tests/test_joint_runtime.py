"""Real five-module complete-epoch continuation and the independent joint CLI."""
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from maidata.joint_training import checkpoint, recipe, runner
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.decoder_training.ordering import history_dropped
from maidata.decoder_training.window_cache import select_window
from maidata.training.batching import optimizer_batches
from maidata.training.data import AudioBatchSampler
from test_decoder_training import config, encoder_config, distances
from test_joint_data import joint_item


def joint_config(config, output):
    return recipe.resolve(dict(config, joint_train_device='cpu', joint_train_workers=0,
        joint_train_precision='fp32', joint_train_z_stride_beats=1,
        joint_train_scheduler_epochs=6, joint_train_stop_after_epochs=6,
        joint_train_warmup_steps=1, joint_train_output=str(output),
        dropout=.1, draft_dropout=.1, planner_dropout=.1,
        reconstruction_dropout=.1, decoder_dropout=.1,
        decoder_train_chunk_token_budget=10000,
        **{key: 'accepted/' + key for key in recipe.PATH_KEYS if key != 'joint_train_output'}))


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


class EpochItems(Dataset):
    def __init__(self, items, config, training):
        self.items, self.config, self.training = items, config, training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        variant, epoch = index
        item = self.items[int(variant)]
        windows = []
        for window in item['m_windows']:
            identity, occurrence = window['identity']
            if not self.training and occurrence:
                continue
            row = dict(window_id=identity, occurrence=occurrence)
            source = dict(history=window['offset'], tokens=window['tokens'],
                next_types=window['next_types'], labels=window['labels'],
                metadata=dict(logical_lengths=[window['length']-window['offset'], window['offset']]))
            selected = select_window(source, history_dropped(self.config, self.training, epoch, row))
            windows.append(dict(window, **selected))
        return dict(item, m_windows=windows)


def runtime_dependencies(monkeypatch, config, items):
    class Plan:
        def __init__(self, root, config, *, training):
            self.training = training
            self.groups = {str(i): {} for i in range(len(items))}
            self.manifest = {'epoch': 1}

        def order(self, epoch):
            result = list(self.groups)
            return result if epoch % 2 else list(reversed(result))

        def batches(self, epoch, config):
            return optimizer_batches(self.order(epoch), config)

    def loader(plan, config):
        return DataLoader(EpochItems(items, config, plan.training), batch_sampler=AudioBatchSampler(plan, config),
            collate_fn=partial(collate_audio, config=config, packed=True), num_workers=0,
            generator=torch.Generator().manual_seed(731))

    monkeypatch.setattr(runner, 'JointPlan', Plan)
    monkeypatch.setattr(runner, 'make_loader', loader)
    monkeypatch.setattr(runner, 'load_route_distances', lambda path: distances())
    monkeypatch.setattr(checkpoint, 'source_identity', lambda: {'commit': 'source-before'})


def test_real_five_module_nonzero_dropout_complete_epoch_resume_is_exact(config, tmp_path, monkeypatch):
    c = joint_config(config, tmp_path / 'full')
    items = [joint_item(i, c, monkeypatch, remaster=i == 0, training=False) for i in range(2)]
    runtime_dependencies(monkeypatch, c, items)
    runner.run(c)
    interrupted = dict(c, joint_train_output=str(tmp_path / 'interrupted'))
    real_update, calls = runner.update, 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('inside epoch interruption')
        return real_update(*args, **kwargs)

    monkeypatch.setattr(runner, 'update', fail)
    with pytest.raises(RuntimeError, match='inside epoch'):
        runner.run(interrupted)
    path = tmp_path / 'interrupted/checkpoints/last.ckpt'
    saved = torch.load(path, weights_only=False)
    assert saved['completed_epochs'] == saved['step'] == 2
    assert sorted(value.name for value in path.parent.iterdir()) == ['last.ckpt']
    assert sorted({key.split('.')[0] for key in saved['model_state']}) == [
        'decoder', 'draft', 'encoder', 'planner', 'reconstruction']
    monkeypatch.setattr(runner, 'update', real_update)
    monkeypatch.setattr(checkpoint, 'source_identity', lambda: {'commit': 'source-after'})
    relocated = dict(interrupted, pretrain_audio_cache='relocated/accepted-audio',
                     decoder_train_gt_plan_cache='relocated/accepted-plan')
    runner.run(relocated, resume=path)
    full = torch.load(tmp_path / 'full/checkpoints/last.ckpt', weights_only=False)
    resumed = torch.load(path, weights_only=False)
    for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng',
                'loader_rng', 'loader_started', 'step', 'completed_epochs', 'best_validation_loss'):
        assert_state_equal(full[key], resumed[key])
    assert resumed['source'] == {'commit': 'source-after'}
    assert full['binding'] != resumed['binding']
    assert full['step'] == resumed['step'] == 6
    metrics = [json.loads(line) for line in (tmp_path / 'interrupted/metrics.jsonl').read_text().splitlines()]
    continuous = [json.loads(line) for line in (tmp_path / 'full/metrics.jsonl').read_text().splitlines()]
    assert [row['train'] for row in continuous] == [row['train'] for row in metrics]
    assert [row['epoch'] for row in metrics if 'validation' in row] == [5, 6]
    assert [row['epoch'] for row in metrics if 'diagnostics' in row.get('validation', {})] == [6]
    assert sorted(value.name for value in path.parent.iterdir()) == [
        'best.ckpt', 'epoch-005.ckpt', 'epoch-006.ckpt', 'last.ckpt']
    assert runner.run(relocated, resume=path) == {'completed_epochs': 6, 'step': 6}


def test_empty_validation_plan_skips_validation_and_best_checkpoint(config, tmp_path, monkeypatch):
    c = joint_config(config, tmp_path / 'no-validation')
    items = [joint_item(0, c, monkeypatch, training=False)]
    runtime_dependencies(monkeypatch, c, items)
    original_plan = runner.JointPlan

    def plan(root, config, *, training):
        result = original_plan(root, config, training=training)
        if not training:
            result.groups = {}
        return result

    monkeypatch.setattr(runner, 'JointPlan', plan)
    monkeypatch.setattr(runner, 'validate', lambda *args, **kwargs: pytest.fail('validation disabled'))
    runner.run(c)
    output = Path(c['joint_train_output'])
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in metrics] == list(range(1, 7))
    assert all('validation' not in row for row in metrics)
    for row in metrics:
        losses = row['train']['losses']
        assert losses['drafter'] == pytest.approx(sum(losses[key] for key in (
            'drafter_fields', 'drafter_moment', 'drafter_count', 'planner')))
        assert row['train']['loss'] == pytest.approx(losses['main_decoder'] + losses['drafter'])
    saved = torch.load(output / 'checkpoints/last.ckpt', weights_only=False)
    assert saved['completed_epochs'] == saved['step'] == 6
    assert saved['best_validation_loss'] is None
    assert sorted(p.name for p in (output / 'checkpoints').iterdir()) == [
        'epoch-005.ckpt', 'epoch-006.ckpt', 'last.ckpt']


def test_fifth_and_final_epoch_validate_once_and_failed_first_epoch_saves_nothing(config, tmp_path, monkeypatch):
    c = joint_config(config, tmp_path / 'coincident')
    c.update(joint_train_stop_after_epochs=5)
    items = [joint_item(0, c, monkeypatch, training=False)]
    runtime_dependencies(monkeypatch, c, items)
    calls = []
    original = runner.validate

    def validate(model, loader, config, geometry, *, final, restored_rng=None):
        calls.append((loader.batch_sampler.epoch, final))
        return original(model, loader, config, geometry, final=final, restored_rng=restored_rng)

    monkeypatch.setattr(runner, 'validate', validate)
    runner.run(c)
    assert calls == [(5, True)]
    saved = torch.load(tmp_path / 'coincident/checkpoints/last.ckpt', weights_only=False)
    assert saved['completed_epochs'] == 5 and 'diagnostics' in saved['metrics']['validation']
    monkeypatch.setattr(runner, 'update', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('first update')))
    with pytest.raises(RuntimeError, match='first update'):
        runner.run(dict(c, joint_train_output=str(tmp_path / 'failed')))
    assert not (tmp_path / 'failed/checkpoints').exists()


@pytest.mark.parametrize('schema', ['maidata.encoder-draft-training.v3', 'maidata.decoder-training.v4'])
def test_joint_restore_rejects_old_stage_schema_before_loading(config, tmp_path, schema):
    c = joint_config(config, tmp_path / 'unused')
    model = JointTrainer(c)
    opt = recipe.optimizer(model.parameters(), c)
    sched = recipe.scheduler(opt, [1]*6, c)
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('partial restore')):
        with pytest.raises(ValueError, match='complete E/D/R/P/M'):
            checkpoint.restore(dict(schema=schema), model, opt, sched, config=c, loaders=[])


def test_restore_rejects_real_z_stride_change_even_when_parameter_shapes_match(config, tmp_path):
    c = recipe.resolve(dict(joint_config(config, tmp_path / 'unused'),
                            target_beats=16, joint_train_z_stride_beats=4))
    model = JointTrainer(c)
    opt = recipe.optimizer(model.parameters(), c)
    sched = recipe.scheduler(opt, [1]*6, c)
    # A valid completed epoch cursor, without fabricating an optimizer update.
    opt.step()
    sched.step()
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(49), _iterator=None)
    saved = checkpoint.capture(model, opt, sched, epoch=1, step=1, config=c,
        binding={'paths': 'accepted'}, source={'commit': 'before'}, steps=[1]*6,
        loaders=[loader], best=None, metrics={})
    changed = recipe.resolve(dict(c, joint_train_z_stride_beats=8))
    other = JointTrainer(changed)
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in other.state_dict().items()}
    with patch.object(other, 'load_state_dict', side_effect=AssertionError('weights loaded before stride check')):
        with pytest.raises(ValueError, match='architecture/representation'):
            other_opt = recipe.optimizer(other.parameters(), changed)
            checkpoint.restore(saved, other, other_opt,
                recipe.scheduler(other_opt, [1]*6, changed),
                config=changed, loaders=[loader])


def test_provenance_records_paths_without_reading_or_hashing_old_assets(config, tmp_path):
    c = joint_config(config, tmp_path / 'output')
    with patch.object(Path, 'read_bytes', side_effect=AssertionError('old payload read')), \
         patch.object(Path, 'read_text', side_effect=AssertionError('old metadata read')), \
         patch.object(Path, 'glob', side_effect=AssertionError('payload scan')), \
         patch.object(Path, 'rglob', side_effect=AssertionError('recursive scan')):
        result = recipe.provenance(c)
    assert result['paths']['pretrain_audio_cache'].endswith('pretrain_audio_cache')
    assert 'joint_train_output' not in result['paths']
    assert not (tmp_path / 'output').exists()


def test_joint_cli_default_config_requires_only_standard_library(tmp_path):
    package = str(Path(__file__).resolve().parents[1] / 'src')
    script = ('import sys; sys.path.insert(0, ' + repr(package) + '); '
              'sys.modules["torch"] = None; sys.modules["numpy"] = None; '
              'from maidata.joint_training.__main__ import main; '
              'raise SystemExit(main(["--show-config"]))')
    process = subprocess.run([sys.executable, '-S', '-c', script], capture_output=True, text=True, check=True)
    c = json.loads(process.stdout)
    assert c['draft_z_stride_beats'] == c['joint_train_z_stride_beats'] == 4
    assert c['joint_train_initialization'] == 'scratch'
    assert c['joint_train_learning_rate'] == c['pretrain_learning_rate'] == c['decoder_train_learning_rate'] == 4e-4
    assert c['joint_train_stop_after_epochs'] == 90 and c['joint_train_scheduler_epochs'] == 150
    assert c['joint_train_workers'] == c['pretrain_workers'] == c['decoder_train_workers'] == 12
    assert c['joint_train_precision'] == 'bf16' and c['joint_train_plan_source'] == 'gt'
    assert [c['joint_train_' + key + '_weight'] for key in ('decoder', 'reconstruction', 'planner')] == [1., 1., 1.]
    assert [c['decoder_train_' + key + '_weight'] for key in ('physical', 'judge', 'trajectory')] == [.8, .5, .5]
    assert c['pretrain_audio_mode'] == c['decoder_train_audio_mode'] == 'joint'
    assert all(c[key] is None for key in ('pretrain_p0_cache', 'decoder_train_p0_cache',
        'decoder_train_local_p0_cache', 'decoder_train_dp_cache', 'decoder_train_checkpoint'))
