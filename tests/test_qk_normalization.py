"""Q/K scale, projection/cache ownership and dense/packed/incremental contracts."""
from contextlib import nullcontext
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.audio.blocks import SharedConformerBlock
from maidata.audio.types import ScaleMemory
from maidata.decoder.attention import Attention
from maidata.decoder.batching import audio_query_layout, audio_key_spans
from maidata.decoder.packed_attention import condition_attention, self_attention
from maidata.decoder.row_layout import window_layout
from maidata.decoder.types import Memory
from maidata.draft.attention import Attention as DraftAttention
from maidata.nn.position import apply_rope
from maidata.nn.qk_norm import QK_NORM_EPS, normalize_qk
from test_decoder_batched_attention import assert_gradients
from test_decoder_network import config, inputs
from maidata.model import build_decoder, build_model, build_draft, QK_KEYS, ENCODER_KEYS


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_per_head_rms_uses_fp32_preserves_dtype_and_zero_gradient(dtype):
    x = (torch.randn(2, 3, 5, 40)*1000).to(dtype).requires_grad_()
    x.data[0, 0, 0].zero_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = normalize_qk(x, True)
    expected = (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+QK_NORM_EPS)).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype and actual[0, 0, 0].count_nonzero() == 0
    torch.testing.assert_close(actual[1].float().square().mean(-1), torch.ones(3, 5),
                               rtol=.01 if dtype == torch.bfloat16 else 1e-6, atol=1e-6)
    actual.float().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[0, 0, 0].abs().sum() > 0


def test_disabled_returns_same_tensor_and_norm_adds_no_parameters_or_rng():
    x = torch.randn(2, 3, 40)
    assert normalize_qk(x, False) is x
    for factory in (
        lambda enabled: Attention(80, 80, 2, 0., 10000., 1., qk_normalization=enabled),
        lambda enabled: DraftAttention(80, 80, 80, 2, 0., rotary=True,
                                        base=10000., scale=1., qk_normalization=enabled),
        lambda enabled: SharedConformerBlock(80, 2, 128, 3, 1., 0., 10000., 1.,
                                             qk_normalization=enabled),
    ):
        torch.manual_seed(49)
        old = factory(False)
        state = torch.get_rng_state()
        torch.manual_seed(49)
        new = factory(True)
        assert torch.equal(torch.get_rng_state(), state)
        assert old.state_dict().keys() == new.state_dict().keys()
        for name, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[name]), name
        with pytest.raises(ValueError, match='boolean'):
            factory(1)


@pytest.mark.parametrize('kind', ['decoder', 'draft'])
def test_projection_normalizes_each_head_before_rope_and_preserves_v(kind):
    torch.manual_seed(3)
    values = torch.randn(2, 5, 80, requires_grad=True)
    beats = torch.arange(5.)[None].expand(2, -1)*1.3
    valid = torch.tensor([[True, False, True, True, True], [False]*5])
    module = (Attention(80, 80, 2, 0., 10000., 1., qk_normalization=True) if kind == 'decoder'
              else DraftAttention(80, 80, 80, 2, 0., rotary=True, base=10000., scale=1.,
                                  qk_normalization=True))
    masked = values.masked_fill(~valid[..., None], 0)
    raw = normalize_qk(module.split(module.key(masked)), True)
    expected = apply_rope(raw, beats.masked_fill(~valid, 0), 10000., 1.)
    if kind == 'decoder':
        expected = torch.cat((expected[:, :, :4], raw[:, :, 4:]), 2)
        kv = module.project(Memory(values, beats, valid), plan_index=4)
        key, value = kv.key, kv.value
        indices = valid.flatten().nonzero().flatten()
        compact = module.project(Memory(values, beats, valid), plan_index=4, indices=indices)
        torch.testing.assert_close(compact.key.transpose(1, 2)[valid], key.transpose(1, 2)[valid])
        torch.testing.assert_close(compact.value.transpose(1, 2)[valid], value.transpose(1, 2)[valid])
    else:
        key, value = module.project_memory(values, beats, valid)
    torch.testing.assert_close(key, expected)
    torch.testing.assert_close(value, module.split(module.value(masked)), rtol=0, atol=0)
    torch.testing.assert_close(module.project_query(masked), normalize_qk(module.split(module.query(masked)), True))
    assert torch.isfinite(key).all()


def test_encoder_local_slot_and_cached_song_keys_share_norm_and_keep_masks():
    block = SharedConformerBlock(80, 2, 128, 3, 1., 0., 10000., 1., qk_normalization=True)
    memory = ScaleMemory(torch.randn(2, 5, 80), torch.randn(2, 5),
                         torch.tensor([[True, False, True, True, True], [False]*5]))
    normalized = block.attention_norm(memory.values.masked_fill(~memory.valid[..., None], 0))
    normalized = normalized.masked_fill(~memory.valid[..., None], 0)
    key, value = block.project_song(memory)
    raw = block._heads(block.k(normalized).masked_fill(~memory.valid[..., None], 0))
    expected = apply_rope(normalize_qk(raw, True), memory.beats.masked_fill(~memory.valid, 0), 10000., 1.)
    torch.testing.assert_close(key, expected)
    torch.testing.assert_close(value, block._heads(block.v(normalized).masked_fill(~memory.valid[..., None], 0)))
    assert key.transpose(1, 2)[~memory.valid].count_nonzero() == 0
    local, slots = block(memory, memory, (key, value), memory.valid, update_slots=True)
    assert torch.isfinite(local.values).all() and torch.isfinite(slots.values).all()
    assert local.values[~memory.valid].count_nonzero() == slots.values[~memory.valid].count_nonzero() == 0
    (local.values*torch.randn_like(local.values)).sum().backward()
    assert torch.isfinite(block.q.weight.grad).all() and block.q.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('plan_index', [None, 4])
def test_normalized_condition_dense_packed_single_softmax_and_gradients(bf16, plan_index):
    torch.manual_seed(48)
    module = Attention(80, 80, 2, 0., 10000., 1., qk_normalization=True)
    with torch.no_grad():
        for projection in (module.query, module.key):
            projection.weight.mul_(20)
            projection.bias.mul_(20)
    x = torch.randn(3, 4, 80, requires_grad=True)
    values = torch.randn(3, 5, 80, requires_grad=True)
    valid = torch.ones(3, 4, dtype=torch.bool)
    key_valid = torch.tensor([[True, False, True, True, True], [False]*4+[True], [False]*5])
    beats = torch.randn(3, 4)*100
    memory = Memory(values, torch.randn(3, 5)*100, key_valid)
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        kv = module.project(memory, plan_index=plan_index)
        expected = module(x, beats, valid, kv, plan_index=plan_index)
        actual = condition_attention(module, x.flatten(0, 1), beats.flatten(), valid, kv,
                                     plan_index, window_layout(valid, key_valid)).reshape_as(expected)
        # Independent FP32 score oracle: raw Plan and rotary Z compete once.
        raw = module.project_query(x)
        query = apply_rope(raw, beats, module.base, module.scale)
        with torch.autocast('cpu', enabled=False):
            if plan_index is None:
                score = query.float() @ kv.key.float().transpose(-1, -2)
            else:
                score = torch.cat((query.float() @ kv.key[:, :, :plan_index].float().transpose(-1, -2),
                                   raw.float() @ kv.key[:, :, plan_index:].float().transpose(-1, -2)), -1)
            score = score/math.sqrt(40)
            allowed = key_valid[:, None, None]
            score = score.masked_fill(~allowed, -torch.inf).masked_fill(~allowed.any(-1, keepdim=True), 0)
            weights = score.softmax(-1).masked_fill(~allowed, 0)
            attended = (weights @ kv.value.float()).to(kv.value.dtype)
        oracle = module.output(attended.transpose(1, 2).reshape_as(x))
        oracle = oracle.masked_fill(~key_valid.any(-1)[:, None, None], 0)
    torch.testing.assert_close(expected, oracle, rtol=.03 if bf16 else 2e-5, atol=.004 if bf16 else 2e-6)
    assert_gradients(actual, expected, (*module.parameters(), x, values), bf16=bf16)
    assert actual[2].count_nonzero() == 0
    # One-key/empty queries cannot backpropagate through Q/K.
    gradients = torch.autograd.grad(actual[1:].sum(), (*module.query.parameters(), *module.key.parameters()),
                                    allow_unused=True)
    assert all(g is None or g.count_nonzero() == 0 for g in gradients)


@pytest.mark.parametrize('bf16', [False, True])
def test_normalized_self_dense_packed_and_incremental_cache_agree(bf16):
    module = Attention(80, 80, 2, 0., 10000., 1., qk_normalization=True)
    x = torch.randn(2, 5, 80, requires_grad=True)
    valid = torch.tensor([[True]*5, [True]*3+[False]*2])
    beats = torch.arange(5.).expand(2, -1)*.3
    rows = window_layout(valid)
    indices = rows['indices']
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        dense, kv = module.self_attention(x, beats, valid, None)
        packed, packed_kv = self_attention(module, x.flatten(0, 1).index_select(0, indices),
            beats.flatten().index_select(0, indices), valid, rows)
        pieces, previous = [], None
        for i in range(5):
            piece, current = module.self_attention(x[:, i:i+1], beats[:, i:i+1], valid[:, i:i+1], previous)
            if previous is not None:
                assert torch.equal(current.key[:, :, :i], previous.key)
            pieces.append(piece)
            previous = current
        incremental = torch.cat(pieces, 1)
    assert_gradients(packed, dense.flatten(0, 1).index_select(0, indices),
                     (*module.parameters(), x), bf16=bf16)
    torch.testing.assert_close(packed_kv.key.transpose(1, 2)[valid], kv.key.transpose(1, 2)[valid])
    torch.testing.assert_close(incremental, dense, rtol=.03 if bf16 else 2e-5, atol=.004 if bf16 else 2e-6)
    torch.testing.assert_close(previous.key, kv.key)


@pytest.mark.parametrize('bf16', [False, True])
def test_all_decoder_layers_packed_audio_and_conditions_match_dense_gradients(bf16):
    c = dict(config(), decoder_qk_normalization=True)
    model = build_decoder(c).eval()
    assert all(a.qk_normalization for layer in model.layers
               for a in (layer.self_attn, layer.audio_attn, layer.condition_attn))
    tokens, audio, z, plan = inputs(model, gradients=True)
    mapping, plan_valid = torch.tensor([0, 0]), torch.tensor([True, True])
    layout = audio_query_layout(tokens.valid, mapping, 1, spans=audio_key_spans(audio.valid))
    kinds = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        keys = model.prepare_audio(audio)
        dense = model.prepare_memory(keys, mapping, z, plan, plan_valid)
        packed = model.prepare_memory(keys, mapping, z, plan, plan_valid, audio_layout=layout)
        expected, actual = model(tokens, kinds, dense), model(tokens, kinds, packed)
    assert_gradients(torch.cat((actual.hidden, actual.type_logits), -1),
        torch.cat((expected.hidden, expected.type_logits), -1),
        (*model.parameters(), audio.values, z.values, plan), bf16=bf16)


def test_encoder_factory_enables_every_block_and_preserves_song_graph_reuse(encoder_config, audio_inputs):
    model = build_model(dict(encoder_config, qk_normalization=True)).eval()
    assert all(block.qk_normalization for stage in model.stages for block in stage)
    song, windows = audio_inputs
    memory = model.prepare_song(song)
    keys = tuple(None if kv is None else kv[0].detach().clone() for kv in memory.block_kv)
    output = model.encode_windows(memory, windows)
    model.encode_windows(memory, windows)
    for before, after in zip(keys, memory.block_kv):
        if before is not None:
            assert torch.equal(before, after[0])
    (output.P0.values*torch.randn_like(output.P0.values)).sum().backward()
    assert model.song_tokenizer.project.weight.grad.abs().sum() > 0


def _checkpoint_config():
    return dict(config(), heads=3, ffn_dim=28, stage_depths=(1, 1, 1), mel_bins=7,
        samples_per_beat=5, stem_channels=4, frequency_dim=6, frequency_queries=2,
        draft_dim=24, draft_latent_dim=24, draft_heads=3, draft_layers=2,
        draft_experts=2, draft_expert_ffn_dim=29, planner_dim=16, planner_heads=2,
        planner_mlp_dim=23, dropout=0., planner_dropout=0., reconstruction_ffn_dim=48,
        reconstruction_dropout=0., joint_train_scheduler_epochs=2, joint_train_stop_after_epochs=2,
        joint_train_warmup_steps=1, joint_train_z_stride_beats=4,
        **dict.fromkeys(QK_KEYS, False))


@pytest.mark.parametrize('enabled', [False, True])
def test_saved_norm_recipe_controls_all_inference_components(enabled, tmp_path, monkeypatch):
    from maidata.entry import CONFIG
    from maidata.inference.checkpoint import JointCheckpoint
    from test_inference_joint import joint_checkpoint
    c = dict(_checkpoint_config(), **dict.fromkeys(QK_KEYS, enabled))
    path, _, payload = joint_checkpoint(c, tmp_path)
    if not enabled:
        for key in QK_KEYS:
            payload['recipe'].pop(key)
            # A future active default cannot silently alter historical math.
            monkeypatch.setitem(CONFIG, key, True)
        torch.save(payload, path)
    loaded = JointCheckpoint(path).models(device='cpu')
    assert all(loaded.config[key] is enabled for key in QK_KEYS)
    assert all(block.qk_normalization is enabled for stage in loaded.encoder.stages for block in stage)
    assert all(a.qk_normalization is enabled for block in loaded.draft.blocks
               for a in (block.self_attention, block.cross_attention))
    assert loaded.planner.pool.qk_normalization is enabled
    assert all(a.qk_normalization is enabled for layer in loaded.decoder.layers
               for a in (layer.self_attn, layer.audio_attn, layer.condition_attn))


@pytest.mark.parametrize('key', QK_KEYS)
def test_joint_resume_rejects_parameter_free_norm_change_before_loading_weights(key):
    from maidata.joint_training import checkpoint, recipe
    from maidata.joint_training.network import JointTrainer
    c = recipe.resolve(_checkpoint_config())
    model = JointTrainer(c)
    optimizer = recipe.optimizer(model.parameters(), c)
    scheduler = recipe.scheduler(optimizer, [1, 1], c)
    optimizer.step(); scheduler.step()
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(3), _iterator=None)
    payload = checkpoint.capture(model, optimizer, scheduler, epoch=1, step=1, config=c,
        binding={}, source={}, steps=[1, 1], loaders=[loader], best=None, metrics={})
    # Simulate the historical recipe, whose Q/K options were implicit false.
    for option in QK_KEYS:
        payload['recipe'].pop(option)
    changed = dict(c, **{key: True})
    with patch.object(model, 'load_state_dict', side_effect=AssertionError('weights loaded before norm check')):
        with pytest.raises(ValueError, match='architecture/representation'):
            checkpoint.restore(payload, model, optimizer, scheduler, config=changed, loaders=[loader])


def test_disabled_norm_keeps_historical_audio_and_draft_cache_identity():
    from maidata.inference.cache import encoder_recipe, draft_identity
    legacy = {key: value for key, value in _checkpoint_config().items() if key not in QK_KEYS}
    disabled = dict(legacy, **dict.fromkeys(QK_KEYS, False))
    assert encoder_recipe(legacy) == encoder_recipe(disabled) == {
        key: legacy[key] for key in ENCODER_KEYS if key != 'qk_normalization'}
    assert encoder_recipe(dict(disabled, qk_normalization=True)) != encoder_recipe(legacy)
    def identity(c):
        return draft_identity(SimpleNamespace(config=c, identity={'selected': 'same-checkpoint'}),
            'same-audio', difficulty=5, level=14., device='cpu', precision='fp32')
    assert identity(legacy) == identity(disabled)
    for key in ('draft_qk_normalization', 'planner_qk_normalization'):
        assert identity(dict(disabled, **{key: True})) != identity(legacy)


@pytest.mark.parametrize('layout,span', [('legacy', 16), ('window-single', 16),
                                      ('centered-overlap', 16)])
def test_every_planner_query_path_uses_normalized_projection(layout, span):
    from maidata.draft.planner import PlannerHead
    from maidata.draft.types import BeatMemory, WindowLatents
    planner = PlannerHead(latent_dim=24, dim=64, heads=2, mlp_dim=80, dropout=0.,
        norm_eps=1e-5, rope_base=10000., rope_position_scale=1., span_beats=span,
        layout=layout, qk_normalization=True)
    values = torch.randn(1, 33, 24, requires_grad=True)
    beats, valid = torch.arange(33.)[None], torch.ones(1, 33, dtype=torch.bool)
    windows = WindowLatents(BeatMemory(values, beats, valid), torch.zeros(1), 16, torch.ones(1, dtype=torch.bool))
    with patch.object(planner.pool, 'project_query', wraps=planner.pool.project_query) as projection:
        result = planner(windows)
        assert projection.call_count == 1
    assert torch.isfinite(result.plan_values).all()
    (result.continuous_logits*torch.randn_like(result.continuous_logits)).sum().backward()
    assert planner.pool.key.weight.grad.abs().sum() > 0
    assert torch.isfinite(values.grad).all()


def test_planner_rejects_discarded_plan4_with_qk_normalization():
    from maidata.draft.planner import PlannerHead
    with pytest.raises(ValueError, match='16'):
        PlannerHead(latent_dim=24, dim=64, heads=2, mlp_dim=80, dropout=0.,
            norm_eps=1e-5, rope_base=10000., rope_position_scale=1.,
            span_beats=4, layout='legacy', qk_normalization=True)
