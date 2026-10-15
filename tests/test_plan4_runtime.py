"""Small real Plan4 E/D/R/P/M updates, complete resume and cached generation."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.decoder.windows import build_window
from maidata.decoder_training.runner import chunk_loss
from maidata.decoder_training.supervision import KEYS, objective
from maidata.decoder_training.support_cache import ChartSupports
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import planner_target, reconstruction_target
from maidata.draft.windows import song_coverage
from maidata.inference.__main__ import main as infer
from maidata.inference.checkpoint import load_joint_models
from maidata.inference.conditioning import prepare_joint_conditions
from maidata.joint_training import checkpoint, recipe, update as updates
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.time import TimingMap
from maidata.training.runner import supports

from test_chart_physical import geometry
from test_decoder_training import fields, tensor_window, distances
from test_inference import config, producer_config
from test_inference_cache import spies
from test_inference_joint import arguments, joint_checkpoint
from test_inference_materialize import write_audio
from test_joint_gradients import optimizers
from test_pretrain_runtime import item as auxiliary_item


@pytest.fixture
def plan4_config(config, tmp_path):
    return recipe.resolve(dict(config, target_beats=16, target_stride_beats=16,
        audio_before_beats=4, audio_after_beats=4, decoder_z_before_beats=4,
        decoder_z_after_beats=4, draft_z_stride_beats=4, joint_train_z_stride_beats=4,
        planner_span_beats=4, joint_train_device='cpu', joint_train_workers=0,
        joint_train_precision='fp32', joint_train_scheduler_epochs=6,
        joint_train_stop_after_epochs=6, joint_train_output=str(tmp_path/'unused'),
        dropout=0., draft_dropout=0., planner_dropout=0., reconstruction_dropout=0.,
        decoder_dropout=0., decoder_history_dropout=0., decoder_train_chunk_token_budget=10000))


def plan4_item(c, monkeypatch):
    """Exact chart/Physical windows plus a new small four-segment GT bank."""
    base = auxiliary_item(0, c)
    coverage = song_coverage(TimingMap([(0, 120)]), 1.2, first_beat=0, target_beats=16)
    auxiliary, decoder, ids, chart_ids, plans, centers = [], [], [], [], [], []
    for key in base['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, physical, geo = fields(monkeypatch, tokens, key=key)
        chart = from_record(record, source_end_beat=Fraction(16))
        planner = MusicPlanTargets(chart)
        physical_chart = ChartSupports(record, physical, geo)
        for start in coverage.starts(include_tail=True):
            identity = f'{key}:{start}'
            target = planner_target(planner, start*96, (start+16)*96, span_beats=4)
            auxiliary.append((identity, key, start,
                reconstruction_target(chart, start_tick=start*96, target_beats=16), target))
            ids.append(identity)
            chart_ids.append(key)
            centers.append([start+2+4*i for i in range(4)])
            plans.append([[*t.continuous, *[float(i == t.category) for i in range(4)]] for t in target])
            for occurrence in range(2 if start == 0 else 1):
                window = build_window(physical_chart.tokens, physical_chart.timing, sequence_start_tick=0,
                    start_tick=start*96, end_tick=(start+16)*96, audio_after_beats=4,
                    audio_end_seconds=1.2, history_beats=4, history_dropout=0.,
                    training=True, rng=random.Random(0))
                row = dict(identity=(identity, occurrence), chart_id=key, start=start, example=window,
                    stages=physical_chart.window(window)[0], timing=physical_chart.timing, audio_end=1.2)
                decoder.append(dict(tensor_window(row), sequence_index=len(decoder)))
    bank = dict(plan_span_beats=4, window_ids=ids, chart_ids=chart_ids,
        window_index={key: i for i, key in enumerate(ids)}, center_beats=torch.tensor(centers).float(),
        plan_values=torch.tensor(plans).float(), valid=torch.ones(len(ids), 4, dtype=torch.bool))
    return dict(base, coverage=coverage, windows=auxiliary, m_windows=decoder, gt_plan=bank)


def direct_update(model, batch, opt, sched, c):
    """Oracle: retain every real M graph and backpropagate one scalar sum."""
    opt.zero_grad(set_to_none=True)
    raw = batch.to('cpu')
    counts = supports(raw)
    decoder_counts = {key: sum(chunk.labels.counts[key] for chunk in raw.chunks) for key in KEYS}
    terms, conditions, _ = model.forward_shared(raw)
    values = [updates.auxiliary_objective(terms, counts, c)]
    metrics = updates.JointMetrics()
    metrics.add_auxiliary(terms, counts)
    audio = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
    for chunk in raw.chunks:
        terms, _ = chunk_loss(model, conditions, audio, chunk.to('cpu'), c, distances())
        values.append(objective(terms, decoder_counts, c)*c['joint_train_decoder_weight'])
        metrics.add_decoder(terms)
    torch.stack(values).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), c['pretrain_gradient_clip'], error_if_nonfinite=True)
    opt.step()
    sched.step()
    return metrics


@pytest.mark.parametrize('budget', [1, 10000])
def test_joint_plan4_direct_and_chunk_bridge_match_with_one_shared_traversal(plan4_config, monkeypatch, budget):
    c = dict(plan4_config, decoder_train_chunk_token_budget=budget)
    item = plan4_item(c, monkeypatch)
    batch = collate_audio([item], config=c, packed=True)
    assert batch.planner.continuous.shape == (2, 4, 8)
    assert supports(batch)['planner_continuous'] == supports(batch)['planner_category'] == 8
    assert batch.conditions.plan.shape == (2, 6, 12)
    assert batch.conditions.plan_valid.tolist() == [[False, True, True, True, True, True],
                                                  [True, True, True, True, True, False]]
    model = JointTrainer(c)
    reference = deepcopy(model)
    opt, sched = optimizers(model)
    ref_opt, ref_sched = optimizers(reference)
    traversals = dict(encoder=0, draft=0)
    def count(name):
        def hook(gradient):
            traversals[name] += 1
        return hook
    handles = [next(getattr(model, name).parameters()).register_hook(count(name)) for name in traversals]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder, 'forward', wraps=model.encoder.forward))
        draft = stack.enter_context(patch.object(model.draft, 'forward', wraps=model.draft.forward))
        bridge = stack.enter_context(patch.object(updates, 'SharedJointGradients', wraps=updates.SharedJointGradients))
        actual, audit = updates.update(model, [batch], opt, sched, c, distances(), audit=True)
    for handle in handles:
        handle.remove()
    together = collate_audio([item], config=dict(c, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, together, ref_opt, ref_sched, c)
    assert encode.call_count == draft.call_count == 1 and traversals == dict(encoder=1, draft=1)
    assert bridge.call_count == int(budget == 1)
    assert audit['chunks'] == (3 if budget == 1 else 1)
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    assert actual.report(c)['loss'] == pytest.approx(expected.report(c)['loss'], rel=3e-6)
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=2e-6, rtol=2e-5, msg=name)
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name+' gradient')


def test_gt_plan4_preserves_m_to_z_d_e_and_p_has_only_its_auxiliary_path(plan4_config, monkeypatch):
    c = plan4_config
    batch = collate_audio([plan4_item(c, monkeypatch)], config=c)
    model = JointTrainer(c)
    terms, conditions, outputs = model.forward_shared(batch)
    assert outputs[1].plan_values.shape == (2, 4, 12) and outputs[1].valid.all()
    assert not conditions.plan.requires_grad and conditions.decoder_z.values.requires_grad
    parameters = tuple(model.parameters())
    owners = {id(p): name for name, module in model.named_children() for p in module.parameters()}
    def reached(value):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        return {owners[id(p)] for p, g in zip(parameters, gradients) if g is not None and g.abs().sum() > 0}
    counts = supports(batch)
    planner_terms = {key: value for key, value in terms.items() if key.startswith('planner_')}
    assert reached(updates.weighted_loss(planner_terms, c, denominators=counts)) == {'encoder', 'draft', 'planner'}
    audio = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
    chunk = batch.chunks[0]
    decoder_terms, _ = chunk_loss(model, conditions, audio, chunk, c, distances())
    assert reached(objective(decoder_terms, chunk.labels.counts, c)) == {'encoder', 'draft', 'decoder'}


@pytest.mark.parametrize('z_enabled,source', [(True, 'predicted'), (False, 'predicted'), (False, 'none')])
def test_real_plan4_generation_online_cold_hot_and_zoff_paths(plan4_config, tmp_path, geometry, z_enabled, source):
    c = plan4_config
    path, _, _ = joint_checkpoint(c, tmp_path)
    write_audio(tmp_path/'audio.wav')
    grid = auxiliary_item(0, c)['grid']
    np.savez(tmp_path/'grid.npz', **grid)
    args = [*arguments(path, tmp_path), '--grid', str(tmp_path/'grid.npz'), '--plan-source', source]
    if not z_enabled:
        args += ['--no-z-cross-attention']
    reports, charts = [], []
    with patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        for mode in ('online', 'cold', 'hot'):
            extra = [] if mode == 'online' else ['--cache-dir', str(tmp_path/'cache')]
            forbid = ('encoder', 'draft', 'planner') if mode == 'hot' else (
                ('draft', 'planner') if source == 'none' else ())
            with ExitStack() as stack:
                calls = spies(stack, forbid=forbid)
                if mode == 'hot':
                    stack.enter_context(patch('maidata.preparation.cache.load_grid', side_effect=AssertionError('hot grid read')))
                    stack.enter_context(patch('soundfile.info', side_effect=AssertionError('hot audio read')))
                infer([*args, *extra, '--output', str(tmp_path/mode)])
                assert calls['decoder'].call_count == 1
                if mode != 'hot':
                    assert calls['encoder'].call_count == 1
                    if source == 'predicted':
                        assert calls['draft'].call_count == calls['planner'].call_count == 1
            reports.append(json.loads((tmp_path/mode/'generation.json').read_text('utf-8')))
            charts.append((tmp_path/mode/'maidata.txt').read_bytes())
    assert charts[0] == charts[1] == charts[2]
    assert reports[0]['windows'] == reports[1]['windows'] == reports[2]['windows']
    assert all(report['complete'] for report in reports)
    expected = ([[False, True, True, True, True, True], [True, True, True, True, True, False]]
                if source == 'predicted' else [[False]*6, [False]*6])
    assert [row['plan_valid'] for row in reports[0]['windows']] == expected
    assert all(np.asarray(row['plan12']).shape == (6, 12) and row['plan_span_beats'] == 4
               for row in reports[0]['windows'])
    assert not reports[1]['cache']['audio']['hit'] and reports[2]['cache']['audio']['hit']
    if source == 'predicted':
        assert not reports[1]['cache']['draft']['hit'] and reports[2]['cache']['draft']['hit']
        cached = torch.load(next((tmp_path/'cache/draft').glob('*.pt')), weights_only=True)
        loaded = load_joint_models(path, device='cpu', z_cross_attention=z_enabled, plan_source=source)
        online = prepare_joint_conditions(loaded.encoder, loaded.draft, loaded.planner, loaded.config,
            grid, TimingMap([(0, 120)]), 1.2, difficulty=5, level=14.5, device='cpu')
        torch.testing.assert_close(cached['plans'], online.plans, atol=0., rtol=0.)
        assert torch.equal(cached['plan_valid'], online.plan_valid)


def test_plan_span_resume_rejects_same_shapes_but_missing_legacy_field_restores(plan4_config, tmp_path):
    legacy = dict(plan4_config, planner_span_beats=16)
    model = JointTrainer(legacy)
    opt, sched = optimizers(model)
    opt.step()
    sched.step()
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(69), _iterator=None)
    saved = checkpoint.capture(model, opt, sched, epoch=1, step=1, config=legacy,
        binding={}, source={'commit': 'fixture'}, steps=[1]*6, loaders=[loader], best=None, metrics={})
    changed = JointTrainer(plan4_config)
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in changed.state_dict().items()}
    changed_opt, changed_sched = optimizers(changed)
    with patch.object(changed, 'load_state_dict', side_effect=AssertionError('loaded before semantic check')):
        with pytest.raises(ValueError, match='architecture/representation'):
            checkpoint.restore(saved, changed, changed_opt, changed_sched, config=plan4_config, loaders=[loader])
    saved['recipe'].pop('planner_span_beats')
    restored = JointTrainer(legacy)
    restored_opt, restored_sched = optimizers(restored)
    assert checkpoint.restore(saved, restored, restored_opt, restored_sched,
        config=legacy, loaders=[loader]) == (1, 1, None)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0., rtol=0.)
    path = tmp_path/'legacy-missing-span.ckpt'
    torch.save(saved, path)
    loaded = load_joint_models(path, device='cpu')
    assert loaded.planner.span_beats == loaded.decoder.plan_span_beats == 16
