"""Real E/D/R/P/M graph, chunk bridge and independent supervision oracles."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import random
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.decoder.types import Memory
from maidata.decoder.windows import build_window
from maidata.decoder_training.batching import window_chunks
from maidata.decoder_training.runner import chunk_loss
from maidata.decoder_training.supervision import KEYS, objective
from maidata.decoder_training.support_cache import ChartSupports
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target
from maidata.draft.windows import song_coverage
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.joint_training import update as updates
from maidata.training.runner import supports
from maidata.time import TimingMap
from test_decoder_training import config, encoder_config, fields, tensor_window, distances
from test_joint_data import joint_item
from test_pretrain_runtime import item as auxiliary_item


@pytest.fixture
def joint_config(config):
    return dict(config, draft_latent_dim=config['draft_dim'], draft_z_stride_beats=1,
        draft_dropout=0., reconstruction_dropout=0., planner_dropout=0., decoder_dropout=0., dropout=0.,
        joint_train_decoder_weight=1., joint_train_reconstruction_weight=1., joint_train_planner_weight=1.,
        pretrain_gradient_clip=1., pretrain_precision='fp32')


def optimizers(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    return optimizer, scheduler


def direct_update(model, batches, optimizer, scheduler, config):
    """Keep every real M graph and backpropagate their scalar sum once."""
    optimizer.zero_grad(set_to_none=True)
    rows = [supports(batch) for batch in batches]
    counts = {key: sum(row[key] for row in rows) for key in rows[0]}
    m_counts = {key: sum(chunk.labels.counts[key] for batch in batches for chunk in batch.chunks) for key in KEYS}
    values, metrics = [], updates.JointMetrics()
    for raw, counts_here in zip(batches, rows):
        batch = raw.to('cpu')
        terms, conditions, _ = model.forward_shared(batch)
        values.append(updates.auxiliary_objective(terms, counts, config))
        metrics.add_aux(terms, counts_here)
        audio = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
        for cpu_chunk in batch.chunks:
            terms, _ = chunk_loss(model, conditions, audio, cpu_chunk.to('cpu'), config, distances())
            values.append(objective(terms, m_counts, config)*config['joint_train_decoder_weight'])
            metrics.add_decoder(terms)
    torch.stack(values).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['pretrain_gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return metrics


def budget_for(windows, chunks):
    lengths = [window['length'] for window in windows]
    for budget in range(1, max(lengths)*len(windows)+1):
        if len(window_chunks(windows, budget)) == chunks:
            return budget
    raise AssertionError('fixture cannot produce requested chunk partition')


@pytest.mark.parametrize('chunk_count', [1, 2, 3])
def test_direct_graph_and_chunk_bridge_have_same_gradients_and_single_shared_traversal(joint_config, monkeypatch, chunk_count):
    item = joint_item(0, joint_config, monkeypatch)
    config = dict(joint_config, decoder_train_chunk_token_budget=budget_for(item['m_windows'], chunk_count))
    batch = collate_audio([item], config=config, packed=True)
    model = JointTrainer(config)
    reference = deepcopy(model)
    opt, scheduler = optimizers(model)
    reference_opt, reference_scheduler = optimizers(reference)
    backward_calls = dict(encoder=0, draft=0)
    def count(name):
        def hook(gradient):
            backward_calls[name] += 1
        return hook
    handles = [next(getattr(model, name).parameters()).register_hook(count(name)) for name in backward_calls]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder, 'forward', wraps=model.encoder.forward))
        local = stack.enter_context(patch.object(model.encoder, 'encode_windows', wraps=model.encoder.encode_windows))
        draft = stack.enter_context(patch.object(model.draft, 'forward', wraps=model.draft.forward))
        project = stack.enter_context(patch.object(model.decoder, 'prepare_audio', wraps=model.decoder.prepare_audio))
        bridge = stack.enter_context(patch.object(updates, 'SharedJointGradients', wraps=updates.SharedJointGradients))
        actual, audit = updates.update(model, [batch], opt, scheduler, config, distances(), audit=True)
    for handle in handles:
        handle.remove()
    reference_batch = collate_audio([item], config=dict(config, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, [reference_batch], reference_opt, reference_scheduler, config)
    assert encode.call_count == local.call_count == draft.call_count == project.call_count == 1
    assert bridge.call_count == int(chunk_count > 1)
    assert backward_calls == dict(encoder=1, draft=1)
    assert audit['chunks'] == chunk_count
    assert set(audit['gradients']) == {'encoder', 'draft', 'reconstruction', 'planner', 'decoder'}
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    assert actual.report(config)['loss'] == pytest.approx(expected.report(config)['loss'], rel=2e-6)
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=2e-6, rtol=2e-5, msg=name)
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name+' gradient')


@pytest.mark.parametrize('accumulated', [False, True])
def test_sparse_physical_batches_and_decoder_chunks_share_logical_denominators(joint_config, monkeypatch, accumulated):
    config = dict(joint_config, decoder_train_chunk_token_budget=1)
    items = [joint_item(0, config, monkeypatch, remaster=True), joint_item(1, config, monkeypatch)]
    batches = [collate_audio([item], config=config, packed=True) for item in items] if accumulated else [collate_audio(items, config=config, packed=True)]
    model, reference = JointTrainer(config), None
    reference = deepcopy(model)
    actual, _ = updates.update(model, batches, *optimizers(model), config, distances())
    expected = direct_update(reference, batches, *optimizers(reference), config)
    assert actual.report(config)['loss'] == pytest.approx(expected.report(config)['loss'], rel=2e-6)
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name)


def test_each_loss_reaches_its_producer_and_gt_plan_stays_constant(joint_config, monkeypatch):
    batch = collate_audio([joint_item(0, joint_config, monkeypatch)], config=joint_config)
    model = JointTrainer(joint_config)
    terms, conditions, _ = model.forward_shared(batch)
    assert not conditions.plan.requires_grad
    assert conditions.decoder_z.values.requires_grad
    parameters = tuple(model.parameters())
    owners = {id(parameter): name for name, module in model.named_children() for parameter in module.parameters()}
    def reached(value):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        return {owners[id(parameter)] for parameter, gradient in zip(parameters, gradients)
                if gradient is not None and bool(gradient.abs().sum() > 0)}
    counts = supports(batch)
    reconstruction = {key: value for key, value in terms.items() if not key.startswith('planner_')}
    planner = {key: value for key, value in terms.items() if key.startswith('planner_')}
    assert reached(updates.weighted_loss(reconstruction, joint_config, denominators=counts)) == {'encoder', 'draft', 'reconstruction'}
    assert reached(updates.weighted_loss(planner, joint_config, denominators=counts)) == {'encoder', 'draft', 'planner'}
    audio = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
    chunk = batch.chunks[0]
    m_terms, _ = chunk_loss(model, conditions, audio, chunk, joint_config, distances())
    assert reached(objective(m_terms, chunk.labels.counts, joint_config)) == {'encoder', 'draft', 'decoder'}
    detached = replace(conditions, decoder_z=replace(conditions.decoder_z, values=conditions.decoder_z.values.detach()))
    m_terms, _ = chunk_loss(model, detached, audio, chunk, joint_config, distances())
    assert reached(objective(m_terms, chunk.labels.counts, joint_config)) == {'encoder', 'decoder'}


def aligned_item(index, config, monkeypatch):
    """Z4 and aligned targets, using exact source/Physical tensor windows."""
    base = auxiliary_item(index, config)
    coverage = song_coverage(TimingMap([(0, 120)]), 1.2, first_beat=0, target_beats=config['target_beats'])
    auxiliary, decoder, ids, plans = [], [], [], []
    for key in base['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, physical, geometry = fields(monkeypatch, tokens, key=key)
        chart = from_record(record, source_end_beat=Fraction(config['target_beats']))
        planner = MusicPlanTargets(chart)
        supports_chart = ChartSupports(record, physical, geometry)
        for start in coverage.starts(include_tail=True):
            identity = f'{key}:{start}'
            target = planner.window(start*96, (start+config['target_beats'])*96)
            auxiliary.append((identity, key, start, reconstruction_target(chart, start_tick=start*96,
                target_beats=config['target_beats']), target))
            ids.append(identity)
            plans.append(torch.tensor((*target.continuous, *[float(i == target.category) for i in range(4)])))
            for occurrence in range(2 if start == 0 else 1):
                window = build_window(supports_chart.tokens, supports_chart.timing, sequence_start_tick=0,
                    start_tick=start*96, end_tick=(start+config['target_beats'])*96,
                    audio_after_beats=config['audio_after_beats'], audio_end_seconds=1.2,
                    history_beats=4, history_dropout=0., training=True, rng=random.Random(0))
                decoder.append(dict(tensor_window(dict(identity=(identity, occurrence), chart_id=key, start=start,
                    example=window, stages=supports_chart.window(window)[0], timing=supports_chart.timing, audio_end=1.2)),
                    sequence_index=len(decoder)))
    return dict(base, coverage=coverage, windows=auxiliary, m_windows=decoder,
        gt_plan=dict(window_ids=ids, window_index={key: i for i, key in enumerate(ids)},
                     plan_values=torch.stack(plans), valid=torch.ones(len(ids), dtype=torch.bool)))


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_z4_real_update_and_exact_original_24beat_local_p0(joint_config, monkeypatch, precision):
    config = dict(joint_config, target_beats=16, target_stride_beats=16, audio_before_beats=4, audio_after_beats=4,
                  draft_z_stride_beats=4, pretrain_precision=precision, decoder_train_chunk_token_budget=1)
    item = aligned_item(0, config, monkeypatch)
    batch = collate_audio([item], config=config)
    assert batch.audio_windows.mel.shape[1] == 24*config['samples_per_beat']
    model = JointTrainer(config)
    with torch.no_grad():
        _, conditions, (_, _, z, local) = model.forward_shared(batch)
        assert conditions.audio.values is local.values
        assert z.values.shape[1] == 8
        assert z.beats[0].tolist() == [2., 6., 10., 14., 18., 22., 26., 30.]
        # The M rows are the original local outputs, including left/right
        # context; no full-song-center crop is substituted for these tensors.
        assert conditions.audio.values.shape[1] == 24*config['fpn_rates'][0]
        for unique, local_index in enumerate(conditions.audio_mapping.tolist()):
            assert batch.audio_windows.target_start_beats[local_index] == batch.decoder_starts[unique]
            torch.testing.assert_close(conditions.audio.values[local_index], local.values[local_index], rtol=0, atol=0)
    metrics, audit = updates.update(model, [batch], *optimizers(model), config, distances(), audit=True)
    assert torch.isfinite(torch.tensor(metrics.report(config)['loss']))
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())


@pytest.mark.parametrize('mode,changed', [('detach_z', {'encoder','draft','reconstruction','planner','decoder'}),
    ('auxiliary', {'encoder','draft','reconstruction','planner'}), ('decoder', {'encoder','decoder'})])
def test_cost_comparison_modes_have_explicit_gradient_ownership(joint_config, monkeypatch, mode, changed):
    config = dict(joint_config, decoder_train_chunk_token_budget=1)
    batch = collate_audio([joint_item(0, config, monkeypatch)], config=config)
    model = JointTrainer(config)
    _, audit = updates.update(model, [batch], *optimizers(model), config, distances(), audit=True, mode=mode)
    assert {name for name, count in audit['changed_parameter_tensors'].items() if count} == changed


def test_fixed_decoder_cost_baseline_does_not_repeat_draft_or_auxiliary(joint_config, monkeypatch):
    config = dict(joint_config, decoder_train_chunk_token_budget=1)
    batch = collate_audio([joint_item(0, config, monkeypatch)], config=config)
    model = JointTrainer(config)
    with torch.no_grad():
        _, conditions, _ = model.forward_shared(batch)
        fixed = conditions.decoder_z
    with patch.object(model.draft, 'forward', side_effect=AssertionError('fixed baseline cannot execute D')), \
         patch.object(model.reconstruction, 'forward', side_effect=AssertionError('fixed baseline cannot execute R')), \
         patch.object(model.planner, 'forward', side_effect=AssertionError('fixed baseline cannot execute P')):
        _, audit = updates.update(model, [batch], *optimizers(model), config, distances(),
                                  audit=True, mode='decoder', fixed_z=[fixed])
    assert {name for name, count in audit['changed_parameter_tensors'].items() if count} == {'encoder', 'decoder'}
