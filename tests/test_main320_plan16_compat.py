"""Expanded M8 checkpoints keep legacy Z4 and unpositioned single Plan16 inference."""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens
from maidata.entry import CONFIG
from maidata.inference.__main__ import main as infer
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.conditioning import decoder_z_window, prepare_joint_conditions
from maidata.joint_training import recipe
from maidata.joint_training.network import JointTrainer
from maidata.nn.position import apply_rope
from maidata.time import TimingMap
from test_chart_physical import geometry
from test_inference import config, producer_config
from test_inference_cache import inputs, spies
from test_inference_joint import arguments, joint_config, joint_checkpoint
from test_pretrain_runtime import item


OVERRIDE = Path(__file__).resolve().parents[1]/'configs/joint-main320-z4-plan16-gt.json'
LEGACY_DEFAULTS = dict(planner_layout='legacy', planner_stride_beats=None,
                       draft_z_anchor_mode='block-center', reconstruction_span_beats=None)


@pytest.fixture
def legacy_m8(joint_config):
    return dict(joint_config, **LEGACY_DEFAULTS, planner_span_beats=16,
                decoder_layers=8, decoder_router_layer=2)


def test_actual_main320_checkpoint_strict_load_preserves_missing_legacy_defaults(tmp_path):
    override = json.loads(OVERRIDE.read_text('utf-8'))
    c = recipe.resolve(dict(CONFIG, **override))
    assert c['joint_train_scheduler_epochs'] == c['joint_train_stop_after_epochs'] == 150
    assert c['joint_train_workers'] == 12 and c['joint_train_precision'] == 'bf16'
    assert (c['pretrain_audio_groups_per_batch'], c['pretrain_token_budget'],
            c['decoder_train_chunk_token_budget'], c['joint_train_learning_rate']) == (3, 80000, 32768, 6e-4)
    assert all(c[key] == value for key, value in LEGACY_DEFAULTS.items())
    original = JointTrainer(c)
    assert sum(p.numel() for p in original.parameters()) == 56178272
    # Historical training code records none of the new overlap-layout fields.
    saved_recipe = {key: value for key, value in c.items()
                    if key not in (*LEGACY_DEFAULTS, 'planner_span_beats')}
    path = tmp_path/'main320-legacy-plan16.ckpt'
    torch.save(dict(recipe=saved_recipe, model_state=original.state_dict()), path)
    checkpoint = JointCheckpoint(path)
    loaded = checkpoint.models(device='cpu')
    assert loaded.encoder is loaded.decoder_encoder
    assert loaded.config['draft_z_stride_beats'] == 4
    assert all(loaded.config[key] == value for key, value in LEGACY_DEFAULTS.items())
    assert loaded.planner.span_beats == loaded.decoder.plan_span_beats == 16
    assert loaded.planner.layout == loaded.decoder.planner_layout == 'legacy'
    assert loaded.decoder.dim == 320 and len(loaded.decoder.layers) == 8
    assert loaded.decoder.layers[0].condition_attn.dim//loaded.decoder.layers[0].condition_attn.heads == 40
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        for key, value in getattr(original, name).state_dict().items():
            assert torch.equal(value, getattr(loaded, name).state_dict()[key]), name+'.'+key
    del loaded
    checkpoint.payload['model_state'].pop('decoder.type_head.bias')
    with pytest.raises(RuntimeError, match='Missing key'):
        checkpoint.models(device='cpu', components=('decoder',))


def test_legacy_m8_online_cold_hot_match_and_keep_single_plan(legacy_m8, tmp_path, geometry):
    path, _ = inputs(legacy_m8, tmp_path)
    reports, charts = [], []
    for mode in ('online', 'cold', 'hot'):
        with ExitStack() as stack:
            calls = spies(stack, forbid=('encoder', 'draft', 'planner') if mode == 'hot' else ())
            stack.enter_context(patch('maidata.physical.Geometry.from_resources', return_value=geometry))
            infer([*arguments(path, tmp_path), '--grid', str(tmp_path/'grid.npz'),
                   '--output', str(tmp_path/mode),
                   *([] if mode == 'online' else ['--cache-dir', str(tmp_path/'cache')])])
            assert calls['decoder'].call_count == 1
        reports.append(json.loads((tmp_path/mode/'generation.json').read_text('utf-8')))
        charts.append((tmp_path/mode/'maidata.txt').read_bytes())
    assert charts[0] == charts[1] == charts[2]
    assert reports[0]['windows'] == reports[1]['windows'] == reports[2]['windows']
    assert all(report['complete'] for report in reports)
    assert all(len(row['plan12']) == 12 and row['plan_beats'] is None
               and row['plan_span_beats'] == 16 for row in reports[0]['windows'])
    assert reports[1]['cache']['draft']['hit'] is False
    assert reports[2]['cache']['draft']['hit'] is True
    audio = torch.load(next((tmp_path/'cache/audio').glob('*.pt')), weights_only=True)
    draft = torch.load(next((tmp_path/'cache/draft').glob('*.pt')), weights_only=True)
    assert draft['z']['beats'].tolist() == [[2., 6., 10., 14., 18., 22., 26., 30.]]
    assert draft['plans'].shape == (2, 12) and draft['plan_beats'] is None
    assert audio['starts'] == [0, 16]


@pytest.mark.parametrize('packed_prefill', [False, True])
def test_legacy_plan16_unrotated_scores_and_incremental_match(legacy_m8, tmp_path, packed_prefill):
    torch.manual_seed(174)
    path, _, _ = joint_checkpoint(legacy_m8, tmp_path)
    models = JointCheckpoint(path).models(device='cpu')
    conditions = prepare_joint_conditions(models.encoder, models.draft, models.planner,
        models.config, item(0, legacy_m8)['grid'], TimingMap([(0, 120)]), 1.2,
        difficulty=5, level=14.5, device='cpu')
    z = decoder_z_window(conditions, 0, models.config, 'cpu')
    assert z.beats.tolist() == [[-2., 2., 6., 10., 14., 18.]]
    assert z.valid.tolist() == [[False, True, True, True, True, True]]
    model = models.decoder
    plan, valid = conditions.plans[:1], conditions.plan_valid[:1]
    mapping = torch.zeros(1, dtype=torch.long)
    audio = model.prepare_audio(conditions.audio)
    memory = model.prepare_memory(audio, mapping, z, plan, valid)
    assert memory.plan_index == 6
    assert memory.condition[0].key.shape[2] == 7
    condition = model.condition(z, plan, valid)
    layer = model.layers[0].condition_attn
    raw_keys = layer.split(layer.key(condition.values.masked_fill(~condition.valid[..., None], 0)))
    torch.testing.assert_close(memory.condition[0].key[:, :, 6:], raw_keys[:, :, 6:], atol=0, rtol=0)
    # Even a nonzero placeholder cannot introduce Plan-key rotary position.
    arbitrary = condition.beats.clone()
    arbitrary[:, -1] = 13.25
    projected = layer.project(replace(condition, beats=arbitrary), plan_index=6)
    torch.testing.assert_close(projected.key[:, :, 6:], raw_keys[:, :, 6:], atol=0, rtol=0)
    query_beats = torch.tensor([[1.25, 3.5]])
    query_valid = torch.ones(1, 2, dtype=torch.bool)
    raw_query = torch.randn(1, layer.heads, 2, layer.dim//layer.heads)
    rotary_query = apply_rope(raw_query, query_beats, layer.base, layer.scale)
    z_scores = rotary_query @ projected.key[:, :, :6].transpose(-1, -2)
    plan_scores = raw_query @ projected.key[:, :, 6:].transpose(-1, -2)
    scores = torch.cat((z_scores, plan_scores), -1)/math.sqrt(raw_query.shape[-1])
    weights = scores.masked_fill(~projected.valid[:, None, None], -torch.inf).softmax(-1)
    reference = layer.output((weights @ projected.value).transpose(1, 2).reshape(1, 2, layer.dim))
    actual = layer.attend(raw_query, query_beats, query_valid, projected, plan_index=6)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)

    tokens = pack_tokens([[Moment(0), Note.make('tap', 'A1'), Moment(500), Note.make('tap', 'A2')]],
                         [[0., 0., 1., 1.]], device='cpu')
    types = torch.tensor([[1, 0, 1, 0]])
    full = model(tokens, types, memory)
    prefix = 2
    if packed_prefill:
        from maidata.decoder.batching import audio_query_layout, audio_key_spans
        condition_valid = torch.cat((z.valid, valid[:, None]), -1)
        layout = audio_query_layout(tokens.valid[:, :prefix], mapping, len(conditions.audio.valid),
            spans=audio_key_spans(conditions.audio.valid), condition_valid=condition_valid)
        memory = model.prepare_memory(audio, mapping, z, plan, valid, audio_layout=layout)
        assert memory.plan_index == 6
    first = model(tokens.slice(0, prefix), types[:, :prefix], memory)
    hidden, logits, cache = [first.hidden], [first.type_logits], first.cache
    for index in range(prefix, tokens.valid.shape[1]):
        output = model.complete(model.begin(tokens.slice(index, index+1), memory, cache=cache),
                                types[:, index:index+1])
        hidden.append(output.hidden)
        logits.append(output.type_logits)
        cache = output.cache
    torch.testing.assert_close(torch.cat(hidden, 1), full.hidden, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(torch.cat(logits, 1), full.type_logits, atol=3e-6, rtol=3e-5)
