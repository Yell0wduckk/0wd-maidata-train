"""Plan16-only public APIs and Q/K-normalized joint updates with the M plan gate."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import json
import random
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.decoder.attention import ConditionAdapter
from maidata.decoder.windows import build_window
from maidata.decoder_training.gt_plan_cache import GTPlanCache
from maidata.decoder_training.support_cache import ChartSupports
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.planner import PlannerHead
from maidata.draft.targets import reconstruction_target
from maidata.draft.windows import song_coverage
from maidata.inference.checkpoint import JointCheckpoint
from maidata.joint_training import recipe, update as updates
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.model import build_decoder, build_planner
from maidata.time import TimingMap

from test_decoder_training import fields, tensor_window, distances
from test_inference import config, producer_config
from test_inference_joint import joint_checkpoint
from test_joint_gradients import direct_update, optimizers
from test_pretrain_runtime import item as auxiliary_item


@pytest.fixture
def legacy_config(config, tmp_path):
    return recipe.resolve(dict(config, target_beats=16, target_stride_beats=16,
        audio_before_beats=4, audio_after_beats=4, decoder_z_before_beats=4,
        decoder_z_after_beats=4, draft_z_stride_beats=4, joint_train_z_stride_beats=4,
        planner_span_beats=16, joint_train_device='cpu', joint_train_workers=0,
        joint_train_precision='fp32', joint_train_scheduler_epochs=6,
        joint_train_stop_after_epochs=6, joint_train_output=str(tmp_path/'unused'),
        dropout=0., draft_dropout=0., planner_dropout=0., reconstruction_dropout=0.,
        decoder_dropout=0., decoder_history_dropout=0., decoder_train_chunk_token_budget=10000,
        reconstruction_count_loss='mse', reconstruction_count_supervision=False,
        qk_normalization=True, draft_qk_normalization=True,
        planner_qk_normalization=True, decoder_qk_normalization=True))


def legacy_item(c, monkeypatch):
    """Build real chart/Physical supervision directly, without unsupported-plan fixtures."""
    base = auxiliary_item(0, c)
    coverage = song_coverage(TimingMap([(0, 120)]), 1.2, first_beat=0, target_beats=16)
    auxiliary, decoder, ids, plans = [], [], [], []
    for key in base['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, physical, geometry = fields(monkeypatch, tokens, key=key)
        chart = from_record(record, source_end_beat=Fraction(16))
        planner, physical_chart = MusicPlanTargets(chart), ChartSupports(record, physical, geometry)
        for start in coverage.starts(include_tail=True):
            identity = f'{key}:{start}'
            target = planner.window(start*96, (start+16)*96)
            auxiliary.append((identity, key, start,
                reconstruction_target(chart, start_tick=start*96, target_beats=16), target))
            ids.append(identity)
            plans.append([*target.continuous, *[float(i == target.category) for i in range(4)]])
            for occurrence in range(2 if start == 0 else 1):
                window = build_window(physical_chart.tokens, physical_chart.timing, sequence_start_tick=0,
                    start_tick=start*96, end_tick=(start+16)*96, audio_after_beats=4,
                    audio_end_seconds=1.2, history_beats=4, history_dropout=0.,
                    training=True, rng=random.Random(0))
                row = dict(identity=(identity, occurrence), chart_id=key, start=start, example=window,
                    stages=physical_chart.window(window)[0], timing=physical_chart.timing, audio_end=1.2)
                decoder.append(dict(tensor_window(row), sequence_index=len(decoder)))
    bank = dict(plan_span_beats=16, window_ids=ids, window_index={key: i for i, key in enumerate(ids)},
                plan_values=torch.tensor(plans).float(), valid=torch.ones(len(ids), dtype=torch.bool))
    return dict(base, coverage=coverage, windows=auxiliary, m_windows=decoder, gt_plan=bank)


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('budget', [1, 10000])
def test_qk_plan16_joint_update_and_plan_gate_match_one_graph(enabled, budget, legacy_config, monkeypatch):
    c = dict(legacy_config, decoder_train_plan_input=enabled, decoder_train_chunk_token_budget=budget)
    item = legacy_item(c, monkeypatch)
    batch = collate_audio([item], config=c, packed=True)
    on = collate_audio([item], config=dict(c, decoder_train_plan_input=True), packed=True)
    assert len(batch.chunks) == (3 if budget == 1 else 1)
    assert batch.conditions.plan.shape == (2, 12)
    assert batch.conditions.plan_valid.all().item() is enabled
    if not enabled:
        assert batch.conditions.plan.count_nonzero() == 0
        for chunk in batch.chunks:
            assert not (chunk.audio_layout['condition_rows']['key_indices'] % 7 == 6).any()
    assert batch.window_ids == on.window_ids
    for name in ('values', 'beats', 'valid'):
        torch.testing.assert_close(getattr(batch.conditions.decoder_z, name), getattr(on.conditions.decoder_z, name))
    for name in vars(batch.planner):
        value = getattr(batch.planner, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(on.planner, name))
    model = JointTrainer(c)
    reference = deepcopy(model)
    optimizer, scheduler = optimizers(model)
    reference_optimizer, reference_scheduler = optimizers(reference)
    traversals = dict(encoder=0, draft=0)
    def hook(name):
        def count(_): traversals[name] += 1
        return count
    handles = [next(getattr(model, name).parameters()).register_hook(hook(name)) for name in traversals]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder, 'forward', wraps=model.encoder.forward))
        draft = stack.enter_context(patch.object(model.draft, 'forward', wraps=model.draft.forward))
        actual, audit = updates.update(model, [batch], optimizer, scheduler, c, distances(), audit=True)
    for handle in handles:
        handle.remove()
    assert encode.call_count == draft.call_count == 1 and traversals == dict(encoder=1, draft=1)
    full = collate_audio([item], config=dict(c, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, [full], reference_optimizer, reference_scheduler, c)
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    assert actual.report(c)['loss'] == pytest.approx(expected.report(c)['loss'], rel=3e-6)
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    plan_grad = sum(float(parameter.grad.abs().sum()) for parameter in model.decoder.condition.plan.parameters()
                    if parameter.grad is not None)
    assert plan_grad > 0 if enabled else plan_grad == 0
    assert all(parameter.grad is None for parameter in model.reconstruction.count_head.parameters())
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=2e-6, rtol=2e-5, msg=name)
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=5e-4, msg=name+' gradient')


@pytest.mark.parametrize('component', ['planner-factory', 'decoder-factory', 'planner-api', 'condition-api'])
def test_plan4_configuration_is_rejected_by_public_model_entrypoints(legacy_config, component):
    c = dict(legacy_config, planner_span_beats=4)
    with pytest.raises(ValueError, match='16'):
        if component == 'planner-factory':
            build_planner(c)
        elif component == 'decoder-factory':
            build_decoder(c)
        elif component == 'planner-api':
            PlannerHead(latent_dim=c['draft_latent_dim'], dim=c['planner_dim'], heads=c['planner_heads'],
                mlp_dim=c['planner_mlp_dim'], dropout=0., norm_eps=c['planner_norm_eps'],
                rope_base=c['rope_base'], rope_position_scale=c['rope_position_scale'], span_beats=4)
        else:
            ConditionAdapter(c['draft_latent_dim'], c['decoder_dim'], True, c['decoder_norm_eps'],
                             plan_span_beats=4)


def test_plan4_cache_schema_is_rejected_before_any_payload_read(legacy_config, tmp_path):
    root = tmp_path/'invalid-plan-cache'
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(schema='maidata.gt-plan4x12.v1', groups={})))
    with pytest.raises(ValueError, match='schema|Plan16|16'):
        GTPlanCache(root, legacy_config)


def test_plan4_checkpoint_recipe_is_rejected_by_inference_loader(legacy_config, tmp_path):
    path, _, payload = joint_checkpoint(legacy_config, tmp_path)
    payload['recipe'] = dict(payload['recipe'], planner_span_beats=4)
    torch.save(payload, path)
    with pytest.raises(ValueError, match='16'):
        JointCheckpoint(path).models(device='cpu')
