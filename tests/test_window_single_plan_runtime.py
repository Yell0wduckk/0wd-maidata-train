"""Integer Z/R geometry with one unpositioned, legacy-statistics Plan16 per M window."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
import math
from unittest.mock import patch

import pytest
import torch

from maidata.decoder.batching import audio_key_spans, audio_query_layout
from maidata.decoder_training import gt_plan_cache
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.planner import PlannerHead, select_planner_windows
from maidata.draft.targets import planner_target
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage
from maidata.inference.conditioning import (Conditions, decode_joint_draft, decoder_z_window,
    empty_plan_geometry, plan_centers_for_starts, with_plan_source)
from maidata.joint_training import update as updates
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.nn.position import apply_rope
from maidata.training.runner import supports

from test_centered_plan_runtime import config as geometry_config, memory as z_memory
from test_decoder_gt_plan_cache import record, write_chart
from test_decoder_integer_conditions import decoder, inputs
from test_decoder_training import distances
from test_integer_joint_runtime import integer_config, integer_item
from test_joint_gradients import optimizers
from joint_runtime_helpers import direct_update, shared_joint_config
from test_inference import config, producer_config


@pytest.fixture
def single_config(integer_config):
    return dict(integer_config, planner_layout='window-single', planner_stride_beats=None)


def single_geometry_config(**updates):
    return dict(geometry_config(), planner_layout='window-single', planner_stride_beats=None, **updates)


def single_head():
    return PlannerHead(latent_dim=4, dim=8, heads=2, mlp_dim=12, dropout=0., norm_eps=1e-5,
        rope_base=10000., rope_position_scale=1., span_beats=16,
        layout='window-single', stride_beats=None).eval()


def single_item(c, monkeypatch):
    item = integer_item(c, monkeypatch)
    # The middle centered target has center s+8 and exact statistics [s,s+16).
    rows = [(*row[:4], row[4][2]) for row in item['windows']]
    ids = [row[0] for row in rows]
    values = [[*row[4].continuous, *[float(i == row[4].category) for i in range(4)]] for row in rows]
    bank = dict(planner_layout='legacy', plan_span_beats=16, window_ids=ids,
        window_index={key: i for i, key in enumerate(ids)},
        plan_values=torch.tensor(values), valid=torch.ones(len(rows), dtype=torch.bool))
    return dict(item, windows=rows, gt_plan=bank)


def single_decoder_inputs(model):
    tokens, audio, z, plans, _, _, mapping = inputs(model)
    plan = plans[:, 2].detach().clone().requires_grad_()
    with torch.no_grad():
        plan[2].zero_()
        plan[2, 8] = 1.  # An existing, empty tail window is a valid Plan.
    valid = torch.tensor([True, False, True])
    return tokens, audio, z, plan, valid, mapping


def single_memory(model, tokens, audio, z, plan, valid, mapping, *, packed=False):
    layout, indices = None, None
    if packed:
        z_valid = z.valid if model.condition.z_cross_attention else torch.zeros_like(z.valid)
        layout = audio_query_layout(tokens.valid, mapping, len(audio.valid),
            spans=audio_key_spans(audio.valid),
            condition_valid=torch.cat((z_valid, valid[:, None]), -1))
        indices = audio.valid.flatten().nonzero().flatten()
    return model.prepare_memory(model.prepare_audio(audio, indices=indices), mapping, z, plan, valid,
        audio_layout=layout, plan_beats=None)


def test_single_pool_uses_closed_17_nodes_and_the_window_center_query():
    torch.manual_seed(190)
    z, model, c = z_memory(0, 64), single_head(), single_geometry_config()
    z.values.requires_grad_()
    starts = torch.tensor([16.])
    windows = select_planner_windows(z, torch.tensor([0]), starts, torch.tensor([True]), c)
    assert windows.memory.beats.tolist() == [list(range(16, 33))]
    with patch('maidata.draft.attention.F.scaled_dot_product_attention',
               wraps=torch.nn.functional.scaled_dot_product_attention) as attention:
        output = model(windows)
    assert attention.call_count == 1
    mask = attention.call_args.kwargs['attn_mask']
    assert mask.sum(-1).flatten().tolist() == [17]
    raw_query = model.pool.split(model.pool.query(model.query[None, None]))
    expected_query = apply_rope(raw_query, (starts+8)[:, None], model.pool.base, model.pool.scale)
    torch.testing.assert_close(attention.call_args.args[0], expected_query, rtol=0, atol=0)
    assert output.plan_values.shape == (1, 12) and output.valid.tolist() == [True]
    assert output.plan_beats is None
    output.plan_values.square().sum().backward()
    assert z.values.grad[0, 16].abs().sum() > 0
    assert z.values.grad[0, 32].abs().sum() > 0
    assert z.values.grad[0, :16].count_nonzero() == 0
    assert z.values.grad[0, 33:].count_nonzero() == 0


def test_single_plan_equals_centered_middle_slot_and_ignores_every_outside_node():
    torch.manual_seed(194)
    z, model = z_memory(0, 64), single_head()
    centered = PlannerHead(latent_dim=4, dim=8, heads=2, mlp_dim=12, dropout=0., norm_eps=1e-5,
        rope_base=10000., rope_position_scale=1., span_beats=16,
        layout='centered-overlap', stride_beats=4).eval()
    centered.load_state_dict(model.state_dict(), strict=True)
    groups, starts, valid = torch.tensor([0]), torch.tensor([16.]), torch.tensor([True])
    one = model(select_planner_windows(z, groups, starts, valid, single_geometry_config()))
    five = centered(select_planner_windows(z, groups, starts, valid, geometry_config()))
    assert five.plan_beats[:, 2].tolist() == [24.]
    assert torch.equal(one.valid, five.valid[:, 2])
    for name in ('continuous_logits', 'config_logits', 'plan_values'):
        torch.testing.assert_close(getattr(one, name), getattr(five, name)[:, 2], atol=1e-7, rtol=1e-6)
    outside = (z.beats < 16) | (z.beats > 32)
    altered = replace(z, values=z.values+outside[..., None]*torch.randn_like(z.values)*1000)
    changed = model(select_planner_windows(altered, groups, starts, valid, single_geometry_config()))
    assert torch.equal(changed.valid, one.valid)
    for name in ('continuous_logits', 'config_logits', 'plan_values'):
        torch.testing.assert_close(getattr(changed, name), getattr(one, name), atol=0, rtol=0)


def test_single_gt_consumer_reuses_original_plan16_values_without_rewriting_the_asset(tmp_path):
    chart_root, output = tmp_path/'charts', tmp_path/'old-gt'
    chart = record()
    relative = write_chart(chart_root, chart)
    rows = [dict(chart_id='42:5', window_id=f'42:5:{s}', start_tick=s*96,
                 end_tick=(s+16)*96) for s in (0, 16)]
    gt_plan_cache.initialize(dict(decoder_train_chart_cache=str(chart_root),
        decoder_train_source_root=str(tmp_path/'source')), output)
    _, meta = gt_plan_cache.export_variant(('song', rows, {'42:5': relative}))
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(dict(schema=gt_plan_cache.SCHEMA, planner_layout='legacy',
        plan_span_beats=16, groups={'song': meta})))
    asset_before = {p: p.read_bytes() for p in (manifest, output/meta['path'])}
    c = single_geometry_config()
    with patch('maidata.chart_cache.cache.load_chart', side_effect=AssertionError('No GT regeneration')):
        reader = gt_plan_cache.GTPlanCache(output, c)
        loaded = reader.read('song')
    assert reader.manifest['schema'] == 'maidata.gt-plan12.v1'
    assert reader.manifest['planner_layout'] == loaded['planner_layout'] == 'legacy'
    planner = MusicPlanTargets(from_record(chart))
    for i, start in enumerate((0, 16)):
        values, valid, beats = gt_plan_cache.select_plan_conditions(loaded, f'42:5:{start}', '42:5', start, c)
        target = planner_target(planner, start*96, (start+16)*96, layout='window-single')
        expected = torch.tensor([*target.continuous, *[float(k == target.category) for k in range(4)]])
        torch.testing.assert_close(values, loaded['plan_values'][i], rtol=0, atol=0)
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        assert valid and beats is None and values.shape == (12,)
    assert loaded['plan_values'][1].tolist() == [0.]*8+[1., 0., 0., 0.]
    assert all(p.read_bytes() == contents for p, contents in asset_before.items())


def test_inference_has_single_plan_no_positions_and_preserves_empty_tail_and_z23():
    c, coverage = single_geometry_config(), SongCoverage(0, 1, 16, 8.)
    starts = coverage.starts(include_tail=True)
    empty_z, empty_plan, empty_valid = empty_plan_geometry(starts, coverage, c, 'cpu')
    assert empty_plan.shape == (2, 12) and empty_valid.shape == (2,)
    assert plan_centers_for_starts(starts, c, 'cpu') is None
    z, audio = z_memory(), z_memory(0, 1)
    local = Conditions(audio, empty_z, empty_plan, empty_valid, starts, coverage, 'window', None)
    predicted = decode_joint_draft(lambda *_: z, single_head(), c, local, audio,
        difficulty=5, level=14., device='cpu')
    assert predicted.plans.shape == (2, 12) and predicted.plan_valid.tolist() == [True, True]
    assert predicted.plan_beats is None
    selected = [decoder_z_window(predicted, i, c, 'cpu') for i in range(2)]
    assert [x.beats.tolist() for x in selected] == [[list(range(-3, 20))], [list(range(13, 36))]]
    assert [x.valid.sum().item() for x in selected] == [20, 20]
    assert selected[1].valid[0, 19]  # Global endpoint32 survives in the empty tail.
    disabled = decoder_z_window(predicted, 0, dict(c, decoder_z_cross_attention=False), 'cpu')
    assert disabled.values.shape == (1, 23, 4) and not disabled.valid.any()
    none = with_plan_source(predicted, 'none')
    assert none.plans.shape == (2, 12) and not none.plan_valid.any() and none.plan_beats is None


def test_single_plan_attention_matches_legacy_unrotated_score_in_one_softmax():
    torch.manual_seed(191)
    model = decoder(layout='window-single')
    tokens, audio, z, plan, valid, mapping = single_decoder_inputs(model)
    condition = model.condition(z, plan, valid, None)
    assert condition.values.shape == (3, 24, 24)
    assert torch.equal(condition.valid, torch.cat((z.valid, valid[:, None]), -1))
    memory = single_memory(model, tokens, audio, z, plan, valid, mapping)
    assert memory.plan_index == 23
    module, kv = model.layers[0].condition_attn, memory.condition[0]
    raw_keys = module.split(module.key(condition.values.masked_fill(~condition.valid[..., None], 0)))
    expected_z = apply_rope(raw_keys[:, :, :23], z.beats.masked_fill(~z.valid, 0), module.base, module.scale)
    torch.testing.assert_close(kv.key[:, :, :23], expected_z, rtol=0, atol=0)
    torch.testing.assert_close(kv.key[:, :, 23:], raw_keys[:, :, 23:], rtol=0, atol=0)
    x, beats = torch.randn(3, 2, 24), torch.tensor([[0., 1.], [16., 17.], [32., 33.]])
    query_valid = torch.ones((3, 2), dtype=torch.bool)
    raw = module.split(module.query(x))
    q = apply_rope(raw, beats, module.base, module.scale)
    scores = torch.cat((q@kv.key[:, :, :23].transpose(-1, -2),
                        raw@kv.key[:, :, 23:].transpose(-1, -2)), -1)/math.sqrt(q.shape[-1])
    probability = scores.masked_fill(~kv.valid[:, None, None], -torch.inf).softmax(-1)
    expected = module.output((probability@kv.value).transpose(1, 2).reshape(3, 2, 24))
    actual = module(x, beats, query_valid, kv, plan_index=memory.plan_index)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    old = decoder(layout='legacy')
    old.load_state_dict(model.state_dict(), strict=True)
    old_memory = single_memory(old, tokens, audio, z, plan, valid, mapping)
    types = torch.ones_like(tokens.types)
    torch.testing.assert_close(model(tokens, types, memory).hidden,
        old(tokens, types, old_memory).hidden, atol=0, rtol=0)


@pytest.mark.parametrize('z_enabled', [True, False])
def test_single_plan_dense_packed_masks_outputs_and_gradients(z_enabled):
    torch.manual_seed(192)
    model = decoder(layout='window-single', z_enabled=z_enabled)
    tokens, audio, z, plan, valid, mapping = single_decoder_inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    outputs = [model(tokens, types, single_memory(model, tokens, audio, z, plan, valid, mapping,
        packed=packed)) for packed in (False, True)]
    flattened = [torch.cat((out.hidden, out.type_logits), -1) for out in outputs]
    torch.testing.assert_close(*flattened, atol=3e-6, rtol=3e-5)
    parameters = (*model.parameters(), audio.values, z.values, plan)
    weights = torch.randn_like(flattened[0])
    gradients = [torch.autograd.grad((out*weights).sum(), parameters, retain_graph=True,
        allow_unused=True) for out in flattened]
    for left, right in zip(*gradients):
        if left is None or right is None:
            assert left is right or (left if left is not None else right).count_nonzero() == 0
        else:
            torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)
    audio_grad, z_grad, plan_grad = gradients[0][-3:]
    assert plan_grad[valid].abs().sum() > 0 and plan_grad[~valid].count_nonzero() == 0
    if z_enabled:
        assert z_grad[z.valid].abs().sum() > 0 and z_grad[~z.valid].count_nonzero() == 0
    else:
        assert z_grad is None
    assert audio_grad[audio.valid].abs().sum() > 0
    poisoned = model(tokens, types, single_memory(model, tokens, audio,
        replace(z, values=z.values.masked_fill(~z.valid[..., None], float('nan')),
            beats=z.beats.masked_fill(~z.valid, float('nan'))),
        plan.masked_fill(~valid[:, None], float('nan')), valid, mapping))
    torch.testing.assert_close(poisoned.hidden, outputs[0].hidden)


@pytest.mark.parametrize('packed_prefill', [True, False])
def test_single_plan_incremental_and_absolute_shift_match_dense(packed_prefill):
    torch.manual_seed(193)
    model = decoder(layout='window-single')
    tokens, audio, z, plan, valid, mapping = single_decoder_inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    full = model(tokens, types, single_memory(model, tokens, audio, z, plan, valid, mapping))
    prefix = 2
    memory = single_memory(model, tokens.slice(0, prefix), audio, z, plan, valid, mapping,
        packed=packed_prefill)
    prefill = model(tokens.slice(0, prefix), types[:, :prefix], memory)
    cache, hidden = prefill.cache, [prefill.hidden]
    for i in range(prefix, tokens.valid.shape[1]):
        out = model.complete(model.begin(tokens.slice(i, i+1), memory, cache=cache), types[:, i:i+1])
        cache = out.cache
        hidden.append(out.hidden)
    torch.testing.assert_close(torch.cat(hidden, 1), full.hidden, atol=3e-6, rtol=3e-5)
    shift = 13.25
    shifted = replace(tokens, beats=tokens.beats+shift, slide_beats=tokens.slide_beats+shift)
    shifted_out = model(shifted, types, single_memory(model, shifted,
        replace(audio, beats=audio.beats+shift), replace(z, beats=z.beats+shift), plan, valid, mapping))
    torch.testing.assert_close(shifted_out.hidden, full.hidden, atol=4e-6, rtol=4e-5)


@pytest.mark.parametrize('budget', [1, 10000])
def test_single_plan_joint_chunk_update_keeps_integer_r_and_one_shared_e_d(single_config, monkeypatch, budget):
    c = dict(single_config, decoder_train_chunk_token_budget=budget)
    item = single_item(c, monkeypatch)
    batch = collate_audio([item], config=c, packed=True)
    assert batch.draft_inputs.query_beats.tolist() == [list(range(33))]
    assert batch.conditions.decoder_z.beats.tolist() == [list(range(-3, 20)), list(range(13, 36))]
    assert batch.conditions.plan.shape == (2, 12) and batch.conditions.plan_valid.all()
    assert batch.conditions.plan_beats is None
    assert supports(batch)['planner_continuous'] == supports(batch)['planner_category'] == 2
    assert supports(batch)['moment96'] == 7
    model = JointTrainer(c)
    reference = deepcopy(model)
    opt, sched = optimizers(model)
    ref_opt, ref_sched = optimizers(reference)
    traversals = dict(encoder=0, draft=0)
    def count(name):
        def hook(_): traversals[name] += 1
        return hook
    handles = [next(getattr(model, name).parameters()).register_hook(count(name)) for name in traversals]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder, 'forward', wraps=model.encoder.forward))
        draft = stack.enter_context(patch.object(model.draft, 'forward', wraps=model.draft.forward))
        actual, audit = updates.update(model, [batch], opt, sched, c, distances(), audit=True)
    for handle in handles: handle.remove()
    together = collate_audio([item], config=dict(c, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, together, ref_opt, ref_sched, c)
    assert encode.call_count == draft.call_count == 1 and traversals == dict(encoder=1, draft=1)
    assert all(value > 0 for value in audit['gradients'].values())
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
