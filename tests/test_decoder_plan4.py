"""Aligned Plan4 tokens: independent masks, positional attention and live gradients."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from maidata.chart import Moment, Note
from maidata.decoder.attention import ConditionAdapter
from maidata.decoder.batching import pack_tokens, audio_query_layout, audio_key_spans
from maidata.decoder.network import Decoder
from maidata.decoder.types import Memory
from maidata.nn.position import apply_rope


def decoder(*, span=4, z_enabled=True):
    options = dict(dim=24, audio_dim=18, latent_dim=10, layers=4, router_layer=1,
        heads=3, shared_ffn_dim=31, expert_ffn_dim=17, experts=3, readout_ffn_dim=19,
        dropout=0., norm_eps=1e-5, rope_base=10000., rope_position_scale=1.,
        note_dims=(4, 8, 2, 2, 2, 3, 3), route_dims=(20, 4), time_base=10000.,
        time_scale=1., condition_gate_init=.4, z_cross_attention=z_enabled,
        z_normalization=True)
    if span is not None:
        options['plan_span_beats'] = span
    return Decoder(**options).eval()


def inputs(model):
    lengths = (7, 5, 3, 2)
    tokens = pack_tokens([[Moment(0)] + [Note.make('tap', 'A1')]*(n-1) for n in lengths],
        [[16*w + i*.5 for i in range(n)] for w, n in enumerate(lengths)], device='cpu')
    valid = tokens.valid.clone()
    valid[0, 2] = False
    tokens = replace(tokens, valid=valid)
    audio_valid = torch.arange(9)[None] < torch.tensor([9, 6, 0])[:, None]
    audio = Memory(torch.randn(3, 9, model.audio_dim, requires_grad=True),
        torch.arange(9).float()[None].expand(3, -1), audio_valid)
    z_valid = torch.tensor([[0,1,1,1,1,1], [1,1,1,1,1,0],
                            [0,0,0,0,0,0], [0,0,0,0,0,0]], dtype=torch.bool)
    centers = torch.arange(4).float()[:, None]*16 + torch.tensor([-2.,2.,6.,10.,14.,18.])
    z = Memory(torch.randn(4, 6, model.condition.latent_dim, requires_grad=True), centers, z_valid)
    plan = torch.randn(4, 6, 12, requires_grad=True)
    plan_valid = torch.tensor([[0,1,0,1,1,1], [1,1,1,1,1,0],
                               [0,1,0,0,0,0], [0,0,0,0,0,0]], dtype=torch.bool)
    # A real empty segment is a valid Plan, including category=single.
    with torch.no_grad():
        plan[2, 1].zero_()
        plan[2, 1, 8] = 1.
    return tokens, audio, z, plan, plan_valid, torch.tensor([0, 1, 0, 2])


def prepare(model, tokens, audio, z, plan, plan_valid, mapping, *, packed=False, compact=True):
    layout = None
    indices = None
    if packed:
        condition_valid = torch.cat((z.valid if model.condition.z_cross_attention else torch.zeros_like(z.valid),
                                     plan_valid), -1)
        layout = audio_query_layout(tokens.valid, mapping, len(audio.valid), spans=audio_key_spans(audio.valid),
            condition_valid=condition_valid if compact else None)
        indices = audio.valid.flatten().nonzero().flatten()
    return model.prepare_memory(model.prepare_audio(audio, indices=indices), mapping, z, plan, plan_valid,
                                audio_layout=layout)


def test_six_plus_six_centers_independent_masks_and_all_keys_use_rope():
    torch.manual_seed(73)
    model = decoder()
    tokens, audio, z, plan, plan_valid, mapping = inputs(model)
    condition = model.condition(z, plan, plan_valid)
    assert condition.values.shape == (4, 12, 24)
    torch.testing.assert_close(condition.beats, torch.cat((z.beats, z.beats), -1))
    assert torch.equal(condition.valid[:, :6], z.valid)
    assert torch.equal(condition.valid[:, 6:], plan_valid)
    assert condition.valid[0, 2] and not condition.valid[0, 8]  # Missing Plan preserves Z.
    assert not condition.valid[2, 1] and condition.valid[2, 7]  # Valid empty Plan preserves itself.
    memory = prepare(model, tokens, audio, z, plan, plan_valid, mapping)
    assert memory.plan_index is None
    torch.testing.assert_close(memory.condition_beats, condition.beats)
    for layer, kv in zip(model.layers, memory.condition):
        attention = layer.condition_attn
        masked = condition.values.masked_fill(~condition.valid[..., None], 0)
        raw = attention.split(attention.key(masked))
        expected = apply_rope(raw, condition.beats.masked_fill(~condition.valid, 0),
                              attention.base, attention.scale)
        torch.testing.assert_close(kv.key, expected, atol=0., rtol=0.)
        assert kv.key.shape == (4, 3, 12, 8)


@pytest.mark.parametrize('span', [0, 8, True, 4.])
def test_rejects_unknown_plan_span(span):
    with pytest.raises(ValueError, match='plan_span_beats'):
        decoder(span=span)


@pytest.mark.parametrize('wrong_shape', ['legacy', 'four', 'mask', 'mask_dtype'])
@pytest.mark.parametrize('validate_inputs', [False, True])
def test_requires_slot_aligned_plan_values_and_masks(wrong_shape, validate_inputs):
    model = decoder()
    model.condition.validate_inputs = validate_inputs
    _, _, z, plan, valid, _ = inputs(model)
    if wrong_shape == 'legacy':
        plan = plan[:, 0]
    elif wrong_shape == 'four':
        plan, valid = plan[:, 1:5], valid[:, 1:5]
    elif wrong_shape == 'mask':
        valid = valid[:, 0]
    else:
        valid = valid.long()
    with pytest.raises(ValueError, match='aligned with Z slots'):
        model.condition(z, plan, valid)


def test_legacy16_default_keeps_original_single_unpositioned_plan_math_and_state():
    torch.manual_seed(74)
    default = decoder(span=None)
    explicit = decoder(span=16)
    explicit.load_state_dict(default.state_dict(), strict=True)
    _, _, z, plan, _, _ = inputs(default)
    plan, valid = plan[:, 1], torch.tensor([True, False, True, True])
    condition = default.condition(z, plan, valid)
    adapter = default.condition
    a = adapter.z_norm(adapter.z(z.values.masked_fill(~z.valid[..., None], 0)) + adapter.source.weight[0])
    b = adapter.plan_norm(adapter.plan(plan.masked_fill(~valid[:, None], 0)) + adapter.source.weight[1])
    torch.testing.assert_close(condition.values, torch.cat((a, b[:, None]), 1), atol=0., rtol=0.)
    assert condition.values.shape == (4, 7, 24) and condition.beats[:, -1].count_nonzero() == 0
    tokens, audio, _, _, _, mapping = inputs(default)
    memories = [prepare(m, tokens, audio, z, plan, valid, mapping) for m in (default, explicit)]
    assert all(memory.plan_index == 6 for memory in memories)
    layer = default.layers[0].condition_attn
    raw = layer.split(layer.key(condition.values.masked_fill(~condition.valid[..., None], 0)))
    torch.testing.assert_close(memories[0].condition[0].key[:, :, 6:], raw[:, :, 6:], atol=0., rtol=0.)
    types = torch.ones_like(tokens.types)
    left, right = [m(tokens, types, memory) for m, memory in zip((default, explicit), memories)]
    torch.testing.assert_close(left.hidden, right.hidden, atol=0., rtol=0.)
    torch.testing.assert_close(left.type_logits, right.type_logits, atol=0., rtol=0.)
    assert default.state_dict().keys() == decoder().state_dict().keys()


def test_masked_values_are_inert_but_empty_plan_is_not_missing():
    torch.manual_seed(75)
    model = decoder()
    tokens, audio, z, plan, valid, mapping = inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape) % 5
    original = model(tokens, types, prepare(model, tokens, audio, z, plan, valid, mapping))
    poisoned_audio = replace(audio, values=audio.values.masked_fill(~audio.valid[..., None], float('nan')),
                             beats=audio.beats.masked_fill(~audio.valid, float('nan')))
    poisoned_z = replace(z, values=z.values.masked_fill(~z.valid[..., None], float('nan')),
        beats=z.beats.masked_fill(~(z.valid | valid), float('nan')))
    poisoned_plan = plan.masked_fill(~valid[..., None], float('nan'))
    poisoned = model(tokens, types,
        prepare(model, tokens, poisoned_audio, poisoned_z, poisoned_plan, valid, mapping))
    torch.testing.assert_close(poisoned.hidden, original.hidden)
    missing = valid.clone()
    missing[2, 1] = False
    no_plan = model(tokens, types, prepare(model, tokens, audio, z, plan, missing, mapping))
    assert not torch.allclose(original.hidden[2, tokens.valid[2]], no_plan.hidden[2, tokens.valid[2]])
    weights = torch.randn_like(original.hidden)
    (original.hidden*weights).sum().backward()
    for value, mask in ((audio.values, audio.valid), (z.values, z.valid), (plan, valid)):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad[mask].abs().sum() > 0
        assert value.grad[~mask].count_nonzero() == 0
    for value in (model.condition.z.weight, model.condition.plan[0].weight, model.condition.source.weight):
        assert value.grad is not None and value.grad.abs().sum() > 0


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('empty_queries', [False, True])
def test_packed_matches_dense_and_separate_rows_with_gradients(compact, empty_queries):
    torch.manual_seed(76)
    model = decoder()
    tokens, audio, z, plan, valid, mapping = inputs(model)
    if empty_queries:
        tokens = replace(tokens, valid=torch.zeros_like(tokens.valid))
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape) % 5
    dense = model(tokens, types, prepare(model, tokens, audio, z, plan, valid, mapping))
    packed = model(tokens, types,
        prepare(model, tokens, audio, z, plan, valid, mapping, packed=True, compact=compact))
    expected, actual = [torch.cat((out.hidden, out.type_logits), -1) for out in (dense, packed)]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for row in range(len(tokens.valid)):
        row_tokens = replace(tokens, **{name: value[row:row+1] for name, value in vars(tokens).items()})
        row_z = Memory(z.values[row:row+1], z.beats[row:row+1], z.valid[row:row+1])
        memory = prepare(model, row_tokens, audio, row_z, plan[row:row+1], valid[row:row+1], mapping[row:row+1])
        row_output = model(row_tokens, types[row:row+1], memory)
        torch.testing.assert_close(row_output.hidden, dense.hidden[row:row+1], atol=3e-6, rtol=3e-5)
    torch.manual_seed(77)
    weight = torch.randn_like(expected)
    parameters = (*model.parameters(), audio.values, z.values, plan)
    gradients = [torch.autograd.grad((out*weight).sum(), parameters, retain_graph=True, allow_unused=True)
                 for out in (expected, actual)]
    for left, right in zip(*gradients):
        if left is None or right is None:
            assert left is right or (left if left is not None else right).count_nonzero() == 0
        else:
            torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)
    assert packed.hidden[~tokens.valid].count_nonzero() == 0


@pytest.mark.parametrize('packed_prefill', [False, True])
def test_incremental_after_prefill_matches_parallel_and_absolute_shift(packed_prefill):
    torch.manual_seed(78)
    model = decoder()
    tokens, audio, z, plan, valid, mapping = inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape) % 5
    full = model(tokens, types, prepare(model, tokens, audio, z, plan, valid, mapping))
    prefix = 3
    memory = prepare(model, tokens.slice(0, prefix), audio, z, plan, valid, mapping, packed=packed_prefill)
    prefill = model(tokens.slice(0, prefix), types[:, :prefix], memory)
    cache, hidden, logits = prefill.cache, [prefill.hidden], [prefill.type_logits]
    for index in range(prefix, tokens.valid.shape[1]):
        output = model.complete(model.begin(tokens.slice(index, index+1), memory, cache=cache), types[:, index:index+1])
        hidden.append(output.hidden)
        logits.append(output.type_logits)
        cache = output.cache
    torch.testing.assert_close(torch.cat(hidden, 1), full.hidden, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(torch.cat(logits, 1), full.type_logits, atol=3e-6, rtol=3e-5)
    offset = 17.25  # No absolute zero or modulo-four anchor is required by M.
    shifted_tokens = replace(tokens, beats=tokens.beats+offset, slide_beats=tokens.slide_beats+offset)
    shifted_audio, shifted_z = replace(audio, beats=audio.beats+offset), replace(z, beats=z.beats+offset)
    shifted = model(shifted_tokens, types,
        prepare(model, shifted_tokens, shifted_audio, shifted_z, plan, valid, mapping))
    torch.testing.assert_close(shifted.hidden, full.hidden, atol=4e-6, rtol=4e-5)


def test_plan4_stays_positional_when_z_attention_is_disabled():
    torch.manual_seed(79)
    model = decoder(z_enabled=False)
    tokens, audio, z, plan, valid, mapping = inputs(model)
    condition = model.condition(z, plan, valid)
    assert not condition.valid[:, :6].any()
    assert torch.equal(condition.valid[:, 6:], valid)
    torch.testing.assert_close(condition.beats[:, 6:], z.beats)
    memory = prepare(model, tokens, audio, z, plan, valid, mapping, packed=True)
    out = model(tokens, torch.ones_like(tokens.types), memory)
    (out.hidden*torch.randn_like(out.hidden)).sum().backward()
    assert z.values.grad is None and model.condition.z.weight.grad is None
    assert plan.grad is not None and plan.grad[valid].abs().sum() > 0


def test_cpu_bfloat16_packed_dense_forward_and_live_gradients():
    torch.manual_seed(80)
    model = decoder()
    tokens, audio, z, plan, valid, mapping = inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape) % 5
    with torch.autocast('cpu', dtype=torch.bfloat16):
        dense = model(tokens, types, prepare(model, tokens, audio, z, plan, valid, mapping))
        packed = model(tokens, types, prepare(model, tokens, audio, z, plan, valid, mapping, packed=True))
    eps = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(packed.hidden, dense.hidden, atol=2*eps, rtol=4*eps)
    torch.testing.assert_close(packed.type_logits, dense.type_logits, atol=2*eps, rtol=4*eps)
    ((packed.hidden*torch.randn_like(packed.hidden)).sum() + packed.type_logits.square().sum()).backward()
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters())
    for value, mask in ((audio.values, audio.valid), (z.values, z.valid), (plan, valid)):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad[mask].abs().sum() > 0 and value.grad[~mask].count_nonzero() == 0
