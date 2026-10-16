"""Independent end-to-end checks for Q/K normalization, Z2 and onset presence."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.draft.chart_data import from_record
from maidata.draft.targets import overlap_reconstruction_target
from maidata.entry import CONFIG
from maidata.inference.__main__ import main as infer
from maidata.inference.checkpoint import JointCheckpoint, load_joint_models
from maidata.inference.conditioning import (decoder_z_window, prepare_joint_conditions,
                                           with_plan_source)
from maidata.joint_training import checkpoint, update as updates
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.time import TimingMap
from maidata.training.runner import supports

from test_chart_physical import geometry
from test_decoder_training import fields, distances
from test_inference_cache import inputs, spies
from test_inference_joint import arguments, joint_checkpoint
from test_joint_gradients import optimizers
from joint_runtime_helpers import direct_update
from test_window_single_plan_runtime import (single_config, single_item, integer_config,
                                            shared_joint_config, config, producer_config)


QK_FLAGS = ('qk_normalization', 'draft_qk_normalization',
            'planner_qk_normalization', 'decoder_qk_normalization')


@pytest.fixture
def z2_config(single_config):
    return dict(single_config, **dict.fromkeys(QK_FLAGS, True),
        draft_z_stride_beats=2, joint_train_z_stride_beats=2,
        reconstruction_count_loss='mse', reconstruction_count_supervision=False,
        reconstruction_onset_presence_supervision=True, reconstruction_onset_presence_weight=1.0)


def z2_item(c, monkeypatch):
    item = single_item(c, monkeypatch)
    coverage = item['coverage']
    end = coverage.first_beat+coverage.draft_beats
    charts = {}
    for key in item['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, _, _ = fields(monkeypatch, tokens, key=key)
        charts[key] = from_record(record, source_end_beat=Fraction(16))
    windows = [(identity, key, start, overlap_reconstruction_target(charts[key],
        start_tick=start*96, target_beats=16, first_beat=coverage.first_beat,
        coverage_end_beat=end, span_beats=4, z_stride_beats=2,
        onset_presence_supervision=True), plan)
        for identity, key, start, _, plan in item['windows']]
    return dict(item, windows=windows)


def assert_qk_flags(models, enabled):
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        flags = [module.qk_normalization for module in getattr(models, name).modules()
                 if hasattr(module, 'qk_normalization')]
        assert flags and all(flag is enabled for flag in flags), name


@pytest.mark.parametrize('budget', [1, 10000])
def test_joint_update_matches_one_graph_oracle_and_keeps_common_supports(z2_config, monkeypatch, budget):
    c = dict(z2_config, decoder_train_chunk_token_budget=budget)
    item = z2_item(c, monkeypatch)
    batch = collate_audio([item], config=c, packed=True)
    counts = supports(batch)
    assert batch.draft_inputs.query_beats.tolist() == [list(range(0, 33, 2))]
    assert batch.layouts['planner']['beats'].shape == (2, 9)
    assert batch.conditions.decoder_z.beats.tolist() == [list(range(-2, 19, 2)), list(range(14, 35, 2))]
    assert batch.conditions.plan.shape == (2, 12) and batch.conditions.plan_valid.all()
    assert batch.conditions.plan_beats is None
    assert counts['onset_presence'] == 2*item['coverage'].draft_beats*len(item['conditions']) == 64
    assert counts['onset_presence_positive'] == 4 and counts['onset_presence_negative'] == 60
    assert counts['moment96'] == 4
    assert (len(batch.chunks) > 1) if budget == 1 else len(batch.chunks) == 1
    model = JointTrainer(c)
    assert_qk_flags(model, True)
    # Prepared selectors and the independent mask-aware reduction must agree.
    plain = replace(batch.unpack(), layouts={key: value for key, value in batch.layouts.items()
                                            if key != 'auxiliary'})
    with torch.no_grad():
        fast, _, _ = model.forward_shared(batch.to('cpu'))
        regular, _, _ = model.forward_shared(plain)
    assert fast.keys() == regular.keys()
    assert 'onset_presence' in fast and not {'count', 'empty', 'positive_count'} & fast.keys()
    for name in fast:
        assert fast[name].count == regular[name].count
        torch.testing.assert_close(fast[name].total, regular[name].total)
    reference = deepcopy(model)
    opt, sched = optimizers(model)
    ref_opt, ref_sched = optimizers(reference)
    traversals = dict(encoder=0, draft=0)
    def hook(name):
        def count(_): traversals[name] += 1
        return count
    handles = [next(getattr(model, name).parameters()).register_hook(hook(name)) for name in traversals]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder, 'forward', wraps=model.encoder.forward))
        draft = stack.enter_context(patch.object(model.draft, 'forward', wraps=model.draft.forward))
        actual, audit = updates.update(model, [batch], opt, sched, c, distances(), audit=True)
    for handle in handles:
        handle.remove()
    assert encode.call_count == draft.call_count == 1 and traversals == dict(encoder=1, draft=1)
    together = collate_audio([item], config=dict(c, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, together, ref_opt, ref_sched, c)
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    head = model.reconstruction.onset_presence_head
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in model.reconstruction.count_head.parameters())
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    report = actual.report(c)
    assert report['loss'] == pytest.approx(expected.report(c)['loss'], rel=3e-6)
    presence = report['auxiliary']['terms']['onset_presence']
    assert (presence['count'], presence['positive_count'], presence['negative_count']) == (64, 4, 60)
    assert report['losses']['drafter_onset_presence'] == pytest.approx(presence['mean'])
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=2e-6, rtol=2e-5, msg=name)
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name+' gradient')


def test_complete_new_checkpoint_restores_exact_state_and_automatic_inference(z2_config, tmp_path):
    c = z2_config
    model = JointTrainer(c)
    opt = torch.optim.AdamW(model.parameters(), lr=c['joint_train_learning_rate'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)*.001
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    payload = checkpoint.capture(model, opt, sched, epoch=1, step=1, config=c, binding={},
        source=dict(commit='z2-qknorm-onset-test'), steps=[1]*6, loaders=[], best=None, metrics={})
    path = tmp_path/'new.ckpt'
    torch.save(payload, path)
    expected_rng = torch.rand(4)
    restored = JointTrainer(c)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=c['joint_train_learning_rate'])
    restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_opt, lambda _: 1.)
    assert checkpoint.restore(torch.load(path, weights_only=False), restored, restored_opt,
                              restored_sched, loaders=[], config=c)[:2] == (1, 1)
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
    assert len(restored_opt.state) == len(opt.state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    loaded = load_joint_models(path, device='cpu')
    assert_qk_flags(loaded, True)
    assert loaded.config['draft_z_stride_beats'] == 2
    assert loaded.config['reconstruction_onset_presence_supervision'] is True
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        for key, value in getattr(model, name).state_dict().items():
            torch.testing.assert_close(getattr(loaded, name).state_dict()[key], value, rtol=0, atol=0)


def test_old_checkpoint_missing_new_options_retains_old_math_when_defaults_change(single_config, tmp_path, monkeypatch):
    old = {key: value for key, value in single_config.items() if key not in QK_FLAGS
           and key not in ('reconstruction_onset_presence_supervision', 'reconstruction_onset_presence_weight')}
    path, _, _ = joint_checkpoint(old, tmp_path)
    for key in (*QK_FLAGS, 'reconstruction_onset_presence_supervision'):
        monkeypatch.setitem(CONFIG, key, True)
    loaded = load_joint_models(path, device='cpu')
    assert_qk_flags(loaded, False)
    assert loaded.config['draft_z_stride_beats'] == 1
    assert loaded.config['reconstruction_onset_presence_supervision'] is False


@pytest.mark.parametrize('key', [*QK_FLAGS, 'reconstruction_onset_presence_supervision'])
def test_strict_resume_rejects_new_semantic_mismatch_before_loading_weights(z2_config, key):
    model = JointTrainer(z2_config)
    saved = dict(z2_config, **{key: False})
    payload = dict(schema=checkpoint.SCHEMA, architecture=checkpoint.ARCHITECTURE,
        completed_epochs=1, step=1, steps_by_epoch=[1], scheduler_state=dict(last_epoch=1),
        loader_rng=[], loader_started=[], recipe=checkpoint.recipe(saved))
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('Loaded before semantic check')):
        with pytest.raises(ValueError, match='architecture/representation'):
            checkpoint.restore(payload, model, None, None, loaders=[], config=z2_config)


def test_actual_320_parameter_delta_is_only_the_binary_readout():
    root = Path(__file__).resolve().parents[1]
    old = dict(CONFIG, **json.loads((root/'configs/joint-main320-integer-single-plan16-gt.json').read_text()))
    new = dict(CONFIG, **json.loads((root/'configs/joint-main320-z2-single-plan16-qknorm-onset-gt.json').read_text()))
    original, changed = JointTrainer(old), JointTrainer(new)
    old_count = sum(parameter.numel() for parameter in original.parameters())
    new_count = sum(parameter.numel() for parameter in changed.parameters())
    assert old_count == 56_178_272 and new_count == 56_178_593 and new_count-old_count == 321
    added = set(changed.state_dict())-set(original.state_dict())
    assert added == {'reconstruction.onset_presence_head.weight', 'reconstruction.onset_presence_head.bias'}


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_new_ordinary_cold_hot_inference_uses_identical_predictions(z2_config, tmp_path, geometry, precision):
    path, _ = inputs(z2_config, tmp_path, notes=True)
    args = [*arguments(path, tmp_path, precision), '--grid', str(tmp_path/'grid.npz'),
            '--max-tokens-per-moment', '4']
    charts, reports = [], []
    with patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        for name in ('ordinary', 'cold', 'hot'):
            extra = [] if name == 'ordinary' else ['--cache-dir', str(tmp_path/'cache')]
            with ExitStack() as stack:
                calls = spies(stack, forbid=('encoder', 'draft', 'planner') if name == 'hot' else ())
                infer([*args, *extra, '--output', str(tmp_path/name)])
            assert calls['decoder'].call_count == 1
            if name != 'hot':
                assert calls['encoder'].call_count == calls['draft'].call_count == calls['planner'].call_count == 1
            charts.append((tmp_path/name/'maidata.txt').read_bytes())
            reports.append(json.loads((tmp_path/name/'generation.json').read_text()))
    assert charts[0] == charts[1] == charts[2]
    assert all(report['complete'] and len(report['windows']) == 2 for report in reports)
    assert not reports[1]['cache']['audio']['hit'] and not reports[1]['cache']['draft']['hit']
    assert reports[2]['cache']['audio']['hit'] and reports[2]['cache']['draft']['hit']


def test_actual_predicted_gt_and_disabled_plan_keep_z_masks_independent(z2_config, tmp_path):
    path, grid = inputs(z2_config, tmp_path)
    models = JointCheckpoint(path).models(device='cpu')
    predicted = prepare_joint_conditions(models.encoder, models.draft, models.planner,
        models.config, grid, TimingMap([(0, 120)]), 1.2, difficulty=5, level=14.5, device='cpu')
    assert predicted.z.values.shape == (1, 17, z2_config['draft_latent_dim'])
    assert predicted.plan_beats is None and predicted.plan_valid.all()
    document = dict(plan_span_beats=16, windows=[dict(start_beat=start,
        plan12=[0.]*8+[1., 0., 0., 0.], valid=True) for start in predicted.starts])
    gt = with_plan_source(predicted, 'gt', document)
    absent = with_plan_source(predicted, 'none')
    for source in (predicted, gt, absent):
        z = decoder_z_window(source, 1, models.config, 'cpu')
        assert z.values.shape[1] == 11 and z.valid.tolist() == [[True]*10+[False]]
        plan = source.plans[1:2]
        plan_valid = source.plan_valid[1:2]
        condition = models.decoder.condition(z, plan, plan_valid)
        torch.testing.assert_close(condition.valid[:, :11], z.valid)
        assert condition.valid[:, 11].item() == (source is not absent)
        assert condition.beats[:, 11].item() == 0  # Plan remains unpositioned.
