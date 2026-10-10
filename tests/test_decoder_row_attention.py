"""Packed projections/attention versus dense masks, gradients and cache semantics."""
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F

from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens, audio_query_layout, audio_key_spans
from maidata.decoder.types import Memory
from maidata.model import build_decoder
from test_decoder_network import config
from test_decoder_batched_attention import assert_gradients


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def fixture(model, *, empty=False):
    lengths = [33, 21, 7, 2, 1, 11]
    rows = [[Moment(0)] + [Note.make('tap', 'A1')]*(n-1) for n in lengths]
    tokens = pack_tokens(rows, [[-4+i*.25 for i in range(n)] for n in lengths], device='cpu')
    valid = tokens.valid.clone()
    valid[0, [1, 6, 12]] = False
    valid[4] = False
    if empty:
        valid.zero_()
    tokens = replace(tokens, valid=valid, beats=tokens.beats.masked_fill(~valid, float('nan')))
    def memory(batch, length, dim, lengths):
        valid = torch.arange(length)[None] < torch.tensor(lengths)[:, None]
        valid[:, 1] = False
        return Memory(torch.randn(batch, length, dim, requires_grad=True),
            torch.arange(length).float()[None].expand(batch, -1)-7,
            valid)
    audio = memory(3, 31, model.audio_dim, [31, 19, 0])
    z = memory(6, 24, model.condition.latent_dim, [24, 6, 0, 12, 3, 0])
    plan = torch.randn(6, 12, requires_grad=True)
    # Exercise Z-only, plan-only, neither, full and internally holed key sets.
    plan_valid = torch.tensor([True, False, True, False, True, False])
    mapping = torch.tensor([0, 1, 0, 1, 2, 2])
    return tokens, audio, z, plan, plan_valid, mapping


def prepare(model, tokens, audio, z, plan, plan_valid, mapping, *, packed):
    layout = None
    indices = None
    if packed:
        indices = audio.valid.flatten().nonzero().flatten()
        layout = audio_query_layout(tokens.valid, mapping, len(audio.valid), spans=audio_key_spans(audio.valid),
            condition_valid=torch.cat((z.valid, plan_valid[:, None]), -1))
    return model.prepare_memory(model.prepare_audio(audio, indices=indices), mapping, z, plan, plan_valid,
                                audio_layout=layout)


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_packed_model_matches_dense_values_gradients_holes_and_empty_windows(bf16, empty):
    torch.manual_seed(421)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan, plan_valid, mapping = fixture(model, empty=empty)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        dense = prepare(model, tokens, audio, z, plan, plan_valid, mapping, packed=False)
        packed = prepare(model, tokens, audio, z, plan, plan_valid, mapping, packed=True)
        expected = model(tokens, types, dense)
        actual = model(tokens, types, packed)
    a = torch.cat((actual.hidden, actual.type_logits), -1)
    b = torch.cat((expected.hidden, expected.type_logits), -1)
    parameters = (*model.parameters(), audio.values, z.values, plan)
    if not bf16:
        assert_gradients(a, b, parameters)
    else:
        torch.testing.assert_close(a, b, atol=.004, rtol=.025)
        torch.manual_seed(22)
        weight = torch.randn_like(a)
        gradients = [torch.autograd.grad((x*weight).sum(), parameters, retain_graph=True, allow_unused=True)
                     for x in (a, b)]
        # Different GEMM row counts change BF16 accumulation rounding, especially
        # near cancellation. Check the full gradient vector at BF16 precision;
        # the FP32 oracle above checks every parameter element individually.
        eps = torch.finfo(torch.bfloat16).eps
        differences, reference = [], []
        for x, y in zip(*gradients):
            if x is None or y is None:
                assert x is y or (x if x is not None else y).count_nonzero() == 0
            else:
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                differences.append((x-y).flatten())
                reference.append(y.flatten())
        assert torch.cat(differences).norm() <= eps*torch.cat(reference).norm()+1e-6
    assert actual.hidden[~tokens.valid].count_nonzero() == 0
    for a, b in zip(actual.cache.layers, expected.cache.layers):
        assert torch.equal(a.valid, b.valid)
        for x, y in ((a.key, b.key), (a.value, b.value)):
            torch.testing.assert_close(x.transpose(1, 2)[a.valid], y.transpose(1, 2)[b.valid],
                atol=.008 if bf16 else 2e-6, rtol=.025 if bf16 else 2e-5)


def test_all_attention_projections_and_norms_skip_padding_and_sdpa_work_is_bounded():
    torch.manual_seed(222)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan, plan_valid, mapping = fixture(model)
    rows = int(tokens.valid.sum())
    seen, handles, sdpa_calls, qkv_rows = {}, [], [], []
    for layer_id, layer in enumerate(model.layers):
        names = ('self_attn.output', 'audio_attn.query', 'audio_attn.output', 'condition_attn.query',
                 'condition_attn.key', 'condition_attn.value', 'condition_attn.output',
                 'norms.0', 'norms.1', 'norms.2', 'norms.3')
        for name, module in layer.named_modules():
            if name in names:
                handles.append(module.register_forward_pre_hook(lambda m,a,n=(layer_id,name):
                    seen.setdefault(n, []).append(a[0].numel()//a[0].shape[-1])))
    original_sdpa, original_linear = F.scaled_dot_product_attention, F.linear
    def sdpa(q,k,v,**kwargs):
        sdpa_calls.append((tuple(q.shape),tuple(k.shape),kwargs.get('is_causal',False)))
        return original_sdpa(q,k,v,**kwargs)
    def linear(x,w,b=None):
        if tuple(w.shape) == (3*model.dim, model.dim):
            qkv_rows.append(x.numel()//model.dim)
        return original_linear(x,w,b)
    with torch.no_grad(), patch.object(F, 'scaled_dot_product_attention', sdpa), patch.object(F, 'linear', linear):
        memory = prepare(model, tokens, audio, z, plan, plan_valid, mapping, packed=True)
        model(tokens, torch.ones_like(tokens.types), memory)
    for handle in handles:
        handle.remove()
    key_rows = int(z.valid.sum()+plan_valid.sum())
    for (layer_id, name), counts in seen.items():
        assert counts == [key_rows if name in ('condition_attn.key','condition_attn.value') else rows]
    assert qkv_rows == [rows]*len(model.layers)
    self_work = sum(q[0]*q[2]*k[2] for q,k,causal in sdpa_calls if causal)
    condition_work = sum(q[0]*q[2]*k[2] for q,k,causal in sdpa_calls
                         if not causal and q[-1]==2*model.dim//model.layers[0].self_attn.heads)
    assert self_work < len(model.layers)*tokens.valid.shape[0]*tokens.valid.shape[1]**2
    assert condition_work < len(model.layers)*tokens.valid.numel()*(z.valid.shape[1]+1)


def test_packed_prefill_can_continue_and_restore_dense_immutable_cache():
    torch.manual_seed(635)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan, plan_valid, mapping = fixture(model)
    prefix = tokens.slice(0, 5)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    memories = [prepare(model, prefix, audio, z, plan, plan_valid, mapping, packed=p) for p in (False, True)]
    results = []
    for memory in memories:
        output = model(prefix, types[:, :5], memory)
        snapshot = output.cache
        routed = model.begin(tokens.slice(5, 6), memory, cache=snapshot)
        a = model.complete(routed, types[:, 5:6])
        model.complete(routed, (types[:, 5:6]+1)%5)
        b = model.complete(model.begin(tokens.slice(5, 6), memory, cache=snapshot), types[:, 5:6])
        torch.testing.assert_close(a.hidden, b.hidden, rtol=0, atol=0)
        output = model.complete(model.begin(tokens.slice(6, 7), memory, cache=b.cache), types[:, 6:7])
        results.append(output)
    torch.testing.assert_close(results[0].hidden, results[1].hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(results[0].type_logits, results[1].type_logits, rtol=2e-5, atol=2e-6)


def test_bucketed_causal_attention_has_no_future_or_cross_window_dependency():
    torch.manual_seed(817)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan, plan_valid, mapping = fixture(model)
    mem = prepare(model, tokens, audio, z, plan, plan_valid, mapping, packed=True)
    types = torch.ones_like(tokens.types)
    expected = model(tokens, types, mem)
    pads = tokens.pad.clone()
    pads[0, 15:] = 4
    pads[1] = 6
    changed = model(replace(tokens, pad=pads), types, mem)
    torch.testing.assert_close(expected.hidden[0, :15], changed.hidden[0, :15], rtol=0, atol=0)
    torch.testing.assert_close(expected.hidden[2:], changed.hidden[2:], rtol=0, atol=0)
