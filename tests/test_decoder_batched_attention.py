"""Independent pre-optimization oracles and bounded backward-graph regressions."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from maidata.decoder.attention import Attention
from maidata.decoder.batching import audio_query_layout, audio_key_spans
from maidata.decoder.types import KV, Memory
from maidata.decoder.packed_attention import condition_attention
from maidata.decoder.row_layout import window_layout
from maidata.decoder_training import recipe
from maidata.nn.position import apply_rope
from maidata.model import build_decoder
from test_decoder_network import config, inputs


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def old_attention(module, x, beats, valid, kv, *, plan_index=None, offset=None):
    raw = module.split(module.query(x.masked_fill(~valid[..., None], 0)))
    q = apply_rope(raw, beats.masked_fill(~valid, 0), module.base, module.scale)
    if kv.query_layout is not None:
        layout = kv.query_layout
        flat = q.transpose(1, 2).flatten(0, 1)
        output = torch.zeros_like(flat)
        for song, selected in enumerate(layout['indices']):
            if selected.numel() == 0:
                continue
            lo, hi = layout['spans'][song]
            query = flat.index_select(0, selected).transpose(0, 1)[None]
            value = F.scaled_dot_product_attention(query, kv.key[song:song+1, :, lo:hi],
                kv.value[song:song+1, :, lo:hi], attn_mask=kv.valid[song:song+1, None, None, lo:hi])
            output = output.index_copy(0, selected, value[0].transpose(0, 1))
        y = output.reshape(*valid.shape, module.dim)
        active = valid & kv.valid.any(-1)[layout['mapping']][:, None]
    else:
        allowed = valid[:, None, :, None] & kv.valid[:, None, None, :]
        if offset is not None:
            allowed = allowed & (torch.arange(kv.key.shape[2])[None, :] <=
                                 offset+torch.arange(x.shape[1])[:, None])[None, None]
        if plan_index is None:
            y = F.scaled_dot_product_attention(q, kv.key, kv.value, attn_mask=allowed)
        else:
            scores = torch.cat((q.float() @ kv.key[:, :, :plan_index].float().transpose(-1, -2),
                                raw.float() @ kv.key[:, :, plan_index:].float().transpose(-1, -2)), -1)
            scores = scores/math.sqrt(q.shape[-1])
            active = allowed.any(-1, keepdim=True)
            scores = scores.masked_fill(~allowed, -torch.inf).masked_fill(~active, 0)
            weights = scores.softmax(-1).masked_fill(~allowed, 0).to(kv.value.dtype)
            y = weights @ kv.value
        y = y.transpose(1, 2).reshape(*valid.shape, module.dim)
        active = valid & kv.valid.any(-1, keepdim=True)
    return module.output(y).masked_fill(~active[..., None], 0)


def assert_gradients(a, b, parameters, *, bf16=False):
    torch.testing.assert_close(a, b, atol=4e-3 if bf16 else 2e-6, rtol=.025 if bf16 else 2e-5)
    # A fixed linear functional exercises signs without magnifying rounding by
    # feeding each implementation's rounded output back as its gradient.
    torch.manual_seed(22)
    weight = torch.randn_like(a)
    gradients = [torch.autograd.grad((x*weight).sum(), parameters, retain_graph=True, allow_unused=True) for x in (a, b)]
    for x, y in zip(*gradients):
        if x is None or y is None:
            assert x is y or (x if x is not None else y).count_nonzero() == 0
        else:
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            torch.testing.assert_close(x, y, atol=.025 if bf16 else 2e-5, rtol=.04 if bf16 else 3e-4)


@pytest.mark.parametrize('bf16', [False, True])
def test_buckets_match_per_song_oracle_repeats_holes_empty_songs_and_gradients(bf16):
    torch.manual_seed(32)
    module = Attention(24, 18, 3, 0., 10000., 1.)
    mapping = torch.tensor([4, 1, 4, 0, 2, 1, 3, 5, 6, 7, 8, 9])
    valid = torch.arange(9)[None] < torch.tensor([9, 3, 2, 8, 1, 1, 0, 5, 5, 5, 5, 5])[:, None]
    key_valid = torch.arange(21)[None] < torch.tensor([15, 8, 0, 11, 19, 9, 9, 9, 9, 9])[:, None]
    key_valid[:, 0] = False
    key_valid[4, 6:9] = False  # An internal hole must never be filled.
    audio = Memory(torch.randn(10, 21, 18, requires_grad=True),
                   torch.arange(21).float()[None].expand(10, -1), key_valid)
    x = torch.randn(12, 9, 24, requires_grad=True)
    beats = torch.arange(9).float()[None].expand(12, -1)*.3-5
    layout = audio_query_layout(valid, mapping, 10, spans=audio_key_spans(key_valid))
    assert len(layout['groups']) < sum(bool(i.numel()) for i in layout['indices'])
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        original = module.project(audio)
        kv = replace(original, query_layout=layout)
        expected = old_attention(module, x, beats, valid, kv)
        actual = module(x, beats, valid, kv)
    assert_gradients(actual, expected, (*module.parameters(), x, audio.values), bf16=bf16)
    assert actual[~valid].count_nonzero() == 0
    assert actual[mapping == 2].count_nonzero() == 0


@pytest.mark.parametrize('z_valid,plan_valid', [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize('bf16', [False, True])
def test_joint_sdpa_matches_original_single_softmax_and_gradients(z_valid, plan_valid, bf16):
    torch.manual_seed(48)
    module = Attention(24, 24, 3, 0., 10000., 1.)
    x = torch.randn(2, 7, 24, requires_grad=True)
    valid = torch.tensor([[True]*7, [True]*3+[False]*4])
    values = torch.randn(2, 5, 24, requires_grad=True)
    memory = Memory(values, torch.arange(5).float()[None].expand(2, -1),
                    torch.tensor([[z_valid]*4+[plan_valid]]*2))
    beats = torch.randn(2, 7)*100
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        kv = module.project(memory, plan_index=4)
        expected = old_attention(module, x, beats, valid, kv, plan_index=4)
        actual = module(x, beats, valid, kv, plan_index=4)
    assert_gradients(actual, expected, (*module.parameters(), x, values), bf16=bf16)


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('dropout', [0., .35])
def test_packed_condition_mask_changes_invalidate_multiple_cache_and_preserve_exact_single(inplace, dropout):
    torch.manual_seed(81)
    module = Attention(24, 24, 3, dropout, 10000., 1.)
    valid = torch.ones(3, 4, dtype=torch.bool)
    key_valid = torch.ones(3, 5, dtype=torch.bool)
    layout = window_layout(valid, key_valid)
    x = torch.randn(3, 4, 24, requires_grad=True)
    beats = torch.randn(3, 4)
    values = torch.randn(3, 5, 24, requires_grad=True)
    kv = module.project(Memory(values, torch.randn(3, 5), key_valid), plan_index=4)
    # Populate the original all-multiple decision before changing the same
    # public layout's runtime bank to one, zero and two valid condition keys.
    condition_attention(module, x.flatten(0, 1), beats.flatten(), valid, kv, 4, layout)
    changed = torch.tensor([[False, False, False, False, True], [False]*5,
                            [False, True, False, False, True]])
    if inplace:
        kv.valid.copy_(changed)
    else:
        kv = replace(kv, valid=changed)
    rng = torch.get_rng_state()
    expected = module(x, beats, valid, kv, plan_index=4).flatten(0, 1)
    torch.set_rng_state(rng)
    actual = condition_attention(module, x.flatten(0, 1), beats.flatten(), valid, kv, 4, layout)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual[4:8].count_nonzero() == 0
    # One-key and empty rows must be exactly independent of Q/K even after a
    # formerly wider bucket is reused. Dropout still applies once per head/query.
    parameters = (*module.query.parameters(), *module.key.parameters())
    gradients = torch.autograd.grad(actual[:8].square().sum(), parameters,
                                    retain_graph=True, allow_unused=True)
    assert all(gradient is None or gradient.count_nonzero() == 0 for gradient in gradients)
    assert_gradients(actual, expected, (*module.parameters(), x, values))


def test_packed_condition_stable_mask_reuses_decision_without_field_synchronization():
    module = Attention(24, 24, 3, 0., 10000., 1.)
    valid = torch.ones(2, 3, dtype=torch.bool)
    memory = Memory(torch.randn(2, 5, 24), torch.randn(2, 5), torch.ones(2, 5, dtype=torch.bool))
    kv = module.project(memory, plan_index=4)
    layout = window_layout(valid, memory.valid)
    x, beats = torch.randn(6, 24), torch.randn(6)
    expected = condition_attention(module, x, beats, valid, kv, 4, layout)
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('repeated condition mask synchronization')):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            actual = condition_attention(module, x, beats, valid, kv, 4, layout)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert 'aten::nonzero' not in {event.key for event in profiler.key_averages()}


def test_packed_condition_inference_tensor_without_version_keeps_single_key_semantics():
    module = Attention(24, 24, 3, 0., 10000., 1.)
    valid = torch.ones(2, 3, dtype=torch.bool)
    layout = window_layout(valid, torch.ones(2, 5, dtype=torch.bool))
    with torch.inference_mode():
        memory = Memory(torch.randn(2, 5, 24), torch.randn(2, 5),
                        torch.tensor([[False]*4+[True], [False]*5]))
        kv = module.project(memory, plan_index=4)
        x, beats = torch.randn(2, 3, 24), torch.randn(2, 3)
        actual = condition_attention(module, x.flatten(0, 1), beats.flatten(), valid, kv, 4, layout)
        expected = module(x, beats, valid, kv, plan_index=4).flatten(0, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_self_qkv_fusion_matches_original_parameters_causal_mask_and_gradients():
    torch.manual_seed(87)
    module = Attention(24, 24, 3, 0., 10000., 1.)
    x = torch.randn(2, 7, 24, requires_grad=True)
    valid = torch.tensor([[True]*7, [True]*3+[False]*4])
    beats = torch.randn(2, 7)
    original = module.project(Memory(x, beats, valid))
    expected = old_attention(module, x, beats, valid, original, offset=0)
    actual, kv = module.self_attention(x, beats, valid, None)
    assert_gradients(actual, expected, (*module.parameters(), x))
    torch.testing.assert_close(kv.key, original.key)
    torch.testing.assert_close(kv.value, original.value)


@pytest.mark.parametrize('empty', [False, True])
def test_valid_audio_projection_and_ffn_match_dense_model_skip_all_padding(empty):
    torch.manual_seed(73)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan = inputs(model, gradients=True)
    audio = replace(audio, valid=torch.tensor([[not empty]*6+[False]*5]))
    audio = replace(audio, values=audio.values.masked_fill(~audio.valid[..., None], float('nan')),
                    beats=audio.beats.masked_fill(~audio.valid, float('nan')))
    mapping = torch.tensor([0, 0])
    layout = audio_query_layout(tokens.valid, mapping, 1, spans=audio_key_spans(audio.valid))
    args = (mapping, z, plan, torch.tensor([True, True]))
    baseline = model.prepare_memory(model.prepare_audio(audio), *args)
    calls = []
    hook = model.layers[0].audio_attn.key.register_forward_pre_hook(lambda m, a: calls.append(a[0].numel()//model.audio_dim))
    keys = model.prepare_audio(audio, indices=audio.valid.flatten().nonzero().flatten())
    hook.remove()
    assert calls == [int(audio.valid.sum())]
    memory = model.prepare_memory(keys, *args, audio_layout=layout)
    kinds = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    expected = model(tokens, kinds, baseline)
    calls, handles = [], []
    for layer in model.layers:
        first = layer.ffn.experts[0][0] if layer.typed else layer.ffn[0]
        handles.append(first.register_forward_pre_hook(lambda m, a: calls.append(len(a[0]))))
    actual = model(tokens, kinds, memory)
    for handle in handles:
        handle.remove()
    assert calls == [int(tokens.valid.sum())]*len(model.layers)
    parameters = (*model.parameters(), audio.values, z.values, plan)
    assert_gradients(torch.cat((actual.hidden, actual.type_logits), -1),
                     torch.cat((expected.hidden, expected.type_logits), -1), parameters)


@pytest.mark.parametrize('songs', [2, 16])
def test_backward_collects_full_bank_once_instead_of_once_per_song(songs):
    module = Attention(24, 24, 3, 0., 10000., 1.)
    q = torch.randn(songs, 3, 5, 8, requires_grad=True)
    key = torch.randn(songs, 3, 17, 8, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    valid = torch.ones(songs, 5, dtype=torch.bool)
    key_valid = torch.ones(songs, 17, dtype=torch.bool)
    layout = audio_query_layout(valid, torch.arange(songs), songs, spans=audio_key_spans(key_valid))
    with patch.object(F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as sdpa:
        result = module.mapped(q, valid, KV(key, value, key_valid, layout))
    assert sdpa.call_count == 1
    seen, todo = set(), [result.grad_fn]
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        todo.extend(n for n, _ in node.next_functions)
    assert not any(type(n).__name__ == 'SliceBackward0' and n._saved_dim == 0 and
                   tuple(n._saved_self_sym_sizes) == tuple(key.shape) for n in seen)
    assert sum(type(n).__name__ == 'IndexCopyBackward0' for n in seen) == 1
    collect = [n for n in seen if type(n).__name__ == 'IndexSelectBackward0'
               and tuple(n._saved_self_sym_sizes) == tuple(key.shape)]
    assert len(collect) == 2  # K and V, independent of the song count.
    shapes = []
    for node in collect:
        node.register_hook(lambda gi, go: shapes.append(tuple(gi[0].shape)))
    result.square().sum().backward()
    assert shapes == [tuple(key.shape)]*2


def test_empty_query_bucket_preserves_zero_output_and_gradient():
    module = Attention(24, 24, 3, 0., 10000., 1.)
    x = torch.randn(2, 3, 24, requires_grad=True)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    memory = Memory(torch.randn(1, 5, 24), torch.arange(5)[None].float(), torch.ones(1, 5, dtype=torch.bool))
    layout = audio_query_layout(valid, torch.zeros(2, dtype=torch.long), 1, spans=audio_key_spans(memory.valid))
    key = replace(module.project(memory), query_layout=layout)
    result = module(x, torch.zeros(2, 3), valid, key)
    assert result.count_nonzero() == 0
    result.sum().backward()
    assert x.grad.count_nonzero() == 0


def test_optimizer_backend_change_preserves_checkpoint_moments_lr_and_step():
    p = torch.nn.Parameter(torch.randn(3))
    old = torch.optim.AdamW([p], lr=.002)
    p.square().sum().backward(); old.step()
    state = deepcopy(old.state_dict())
    new = torch.optim.AdamW([p], lr=.04, fused=False)
    recipe.restore_optimizer(new, state)
    assert new.param_groups[0]['lr'] == .002 and new.param_groups[0]['fused'] is False
    for key, value in old.state[p].items():
        torch.testing.assert_close(new.state[p][key], value, rtol=0, atol=0)
    assert state['param_groups'][0]['fused'] is None  # Input checkpoint untouched.
    captured = []
    recipe.restore_optimizer(SimpleNamespace(param_groups=[dict(fused=True)], load_state_dict=captured.append), state)
    assert captured[0]['param_groups'][0]['fused'] is True
    assert captured[0]['param_groups'][0]['foreach'] is False
    assert captured[0]['state'] is state['state']
