"""Architecture, cache equivalence, source/target isolation and gradient contracts."""
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from maidata.entry import CONFIG
from maidata.model import build_decoder
from maidata.chart.types import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.decoder.batching import pack_tokens
from maidata.decoder.types import Memory


def config():
    return dict(CONFIG, d_model=18, draft_latent_dim=10, decoder_dim=24, decoder_layers=4,
        decoder_router_layer=1, decoder_heads=3, decoder_shared_ffn_dim=31,
        decoder_expert_ffn_dim=17, decoder_readout_ffn_dim=19, decoder_experts=3,
        decoder_note_dims=(4, 8, 2, 2, 2, 3, 3), decoder_route_dims=(20, 4), decoder_dropout=0.)


def inputs(model, *, gradients=False):
    rows = [(Moment(0), Note.make('hold', 'A1'), HoldDuration(4), Note.make('slide', 'A2'),
             Route(0, 0), Route(4), SlideDuration(96), Moment(500), Note.make('touch', 'C')),
            (Moment(1000), Note.make('tap', 'A3', ex=1, break_=1, short_hold=1))]
    tokens = pack_tokens(rows, [(0, 0, 0, 0, 0, 0, 0, 1, 1), (2, 2)], device='cpu')
    def memory(batch, length, width):
        x = torch.randn(batch, length, width, requires_grad=gradients)
        return Memory(x, torch.arange(length).float()[None].expand(batch, -1),
                      torch.ones(batch, length, dtype=torch.bool))
    audio, z = memory(1, 11, model.audio_dim), memory(2, 6, model.condition.latent_dim)
    plan = torch.randn(2, 12, requires_grad=gradients)
    return tokens, audio, z, plan


def prepare(model, audio, z, plan):
    return model.prepare_memory(model.prepare_audio(audio), torch.tensor([0, 0]), z, plan, torch.tensor([True, True]))


def test_default_and_configurable_structure():
    default = build_decoder(CONFIG)
    assert len(default.layers) == 6 and default.router_layer == 2
    assert len(default.layers[2].ffn.experts) == 4 and len(default.layers[2].ffn.gates) == 5
    assert default.embedding.route.weight.shape == (71, 240)
    assert default.heads.route.weight.shape == (71, 256)
    assert default.heads.times['MOMENT'].out_features == 15
    assert default.target_type.weight.data_ptr() != default.embedding.input_type.weight.data_ptr()
    assert default.layers[2].ffn.experts[0][0].weight.data_ptr() != default.layers[3].ffn.experts[0][0].weight.data_ptr()
    other = build_decoder(config())
    assert other.layers[0].audio_attn.key.in_features == 18
    assert other.condition.z.in_features == 10
    assert other.layers[1].ffn.experts[0][0].out_features == 17
    for override in ({'decoder_router_layer': 4}, {'decoder_heads': 8}, {'decoder_note_dims': (4,)*7}):
        with pytest.raises(ValueError):
            build_decoder(dict(config(), **override))


def test_parallel_incremental_causal_and_type_backtrack():
    torch.manual_seed(55)
    model = build_decoder(config()).eval()
    tokens, audio, z, plan = inputs(model)
    memory = prepare(model, audio, z, plan)
    kinds = torch.arange(tokens.types.shape[1])[None].expand(2, -1) % 5
    full = model(tokens, kinds, memory)
    parts, logits, cache = [], [], None
    for i in range(tokens.types.shape[1]):
        routed = model.begin(tokens.slice(i, i+1), memory, cache=cache)
        out = model.complete(routed, kinds[:, i:i+1])
        parts.append(out.hidden)
        logits.append(out.type_logits)
        if i == 4:
            alternative = model.complete(routed, (kinds[:, i:i+1] + 1) % 5)
            torch.testing.assert_close(alternative.cache.layers[0].key, out.cache.layers[0].key)
            assert not torch.allclose(alternative.cache.layers[1].key, out.cache.layers[1].key)
            torch.testing.assert_close(model.complete(routed, kinds[:, i:i+1]).hidden, out.hidden)
        cache = out.cache
    torch.testing.assert_close(torch.cat(parts, 1), full.hidden, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.cat(logits, 1), full.type_logits, atol=2e-6, rtol=2e-5)
    other = model(tokens, (kinds+1) % 5, memory)
    torch.testing.assert_close(other.type_logits, full.type_logits)
    changed = replace(tokens, time=tokens.time.clone())
    changed.time[:, -1] += 10
    changed = replace(changed, pad=changed.pad.clone())
    changed.pad[0, -1] = 0
    result = model(changed, kinds, memory)
    torch.testing.assert_close(result.hidden[:, :-1], full.hidden[:, :-1])
    with pytest.raises(ValueError, match='fresh'):
        model.begin(tokens.slice(0, 1), prepare(model, audio, z, plan), cache=cache)


def test_joint_plan_has_no_position_and_padding_is_inert():
    model = build_decoder(config()).eval()
    tokens, audio, z, plan = inputs(model)
    types = torch.ones_like(tokens.types)
    output = model(tokens, types, prepare(model, audio, z, plan))
    shifted = model(replace(tokens, beats=tokens.beats+17), types,
        prepare(model, replace(audio, beats=audio.beats+17), replace(z, beats=z.beats+17), plan))
    torch.testing.assert_close(output.hidden, shifted.hidden, atol=2e-6, rtol=2e-5)
    z = replace(z, valid=torch.zeros_like(z.valid), values=torch.full_like(z.values, float('nan')),
                beats=torch.full_like(z.beats, float('nan')))
    audio = replace(audio, valid=torch.zeros_like(audio.valid), values=torch.full_like(audio.values, float('nan')))
    memory = prepare(model, audio, z, plan)
    result = model(tokens, types, memory)
    assert torch.isfinite(result.hidden).all()
    assert memory.condition[0].valid[:, -1].all()  # plan survives empty Z
    alternative = model(tokens, types, prepare(model, audio, z, plan + 2))
    assert not torch.allclose(result.hidden, alternative.hidden, atol=1e-7, rtol=1e-7)
    assert result.hidden[~tokens.valid].count_nonzero() == 0


def test_gradients_projections_and_input_na():
    model = build_decoder(config())
    tokens, audio, z, plan = inputs(model, gradients=True)
    with patch.object(model.condition, 'forward', wraps=model.condition.forward) as adapt:
        memory = prepare(model, audio, z, plan)
        assert adapt.call_count == 1
        out = model(tokens, torch.ones_like(tokens.types), memory)
    out.hidden.square().mean().backward()
    for x in (audio.values, z.values, plan):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for layer in model.layers[model.router_layer:]:
        for expert in layer.ffn.experts:
            assert expert[0].weight.grad.abs().sum() > 0
        assert layer.ffn.gates[1].weight.grad.abs().sum() > 0
        assert all(layer.ffn.gates[i].weight.grad is None or layer.ffn.gates[i].weight.grad.count_nonzero() == 0
                   for i in (0, 2, 3, 4))
    embedded = model.embedding(tokens)
    torch.testing.assert_close(embedded[0, 5, -4:] - model.embedding.input_type.weight[2, -4:], torch.zeros(4))
    assert not torch.equal(embedded[0, 4, -4:], embedded[0, 5, -4:])  # valid break=0 is learned


@pytest.mark.parametrize('z_valid,plan_valid', [(False,False), (True,False), (False,True)])
def test_independent_condition_masks_and_invalid_value_gradients(z_valid, plan_valid):
    model = build_decoder(config()).eval()
    tokens, audio, z, plan = inputs(model,gradients=True)
    audio = replace(audio,valid=torch.tensor([[True]*7+[False]*4]))
    z = replace(z,valid=torch.full_like(z.valid,z_valid))
    validity = torch.full((2,),plan_valid,dtype=torch.bool)
    kinds = torch.ones_like(tokens.types)
    def run(audio,z,plan):
        memory = model.prepare_memory(model.prepare_audio(audio),torch.tensor([0,0]),z,plan,validity)
        return model(tokens,kinds,memory)
    original = run(audio,z,plan)
    poisoned = run(replace(audio,values=audio.values.masked_fill(~audio.valid[...,None],float('nan')),
                         beats=audio.beats.masked_fill(~audio.valid,float('nan'))),
        replace(z,values=z.values.masked_fill(~z.valid[...,None],float('nan')),
                beats=z.beats.masked_fill(~z.valid,float('nan'))),
        plan.masked_fill(~validity[:,None],float('nan')))
    torch.testing.assert_close(original.hidden,poisoned.hidden)
    original.hidden.square().mean().backward()
    assert audio.values.grad[~audio.valid].count_nonzero() == 0
    assert z.values.grad[~z.valid].count_nonzero() == 0
    assert plan.grad[~validity].count_nonzero() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_content_conditions_do_not_recompute_network_and_are_independent():
    model = build_decoder(config())
    h = torch.randn(3, 24, requires_grad=True)
    kind, pad = torch.tensor([0, 2, 4]), torch.tensor([2, 32, 1])
    logits = model.heads.pad_logits(h, kind)
    assert torch.isneginf(logits[[0, 2], 8:]).all() and torch.isfinite(logits[1]).all()
    flags = model.heads.flag_logits(h, kind, pad)
    assert flags.shape == (3, 5, 2)
    flags.sum().backward()
    assert model.heads.kind_condition.weight.grad.abs().sum() > 0
    assert model.embedding.kind.weight.grad is None and model.target_type.weight.grad is None


@pytest.mark.parametrize('mode', ['autocast', 'bfloat16'])
def test_cpu_bfloat16_forward_backward(mode):
    from contextlib import nullcontext
    model = build_decoder(config())
    tokens, audio, z, plan = inputs(model)
    if mode == 'bfloat16':
        model = model.bfloat16()
        audio, z = replace(audio, values=audio.values.bfloat16()), replace(z, values=z.values.bfloat16())
        plan = plan.bfloat16()
    context = torch.autocast('cpu',dtype=torch.bfloat16) if mode=='autocast' else nullcontext()
    with context:
        out = model(tokens,torch.ones_like(tokens.types),prepare(model,audio,z,plan))
        kind, pad = torch.tensor([0,2]), torch.tensor([0,32])
        lp = model.heads.pad_logits(out.hidden[:,0],kind)
        flags = model.heads.flag_logits(out.hidden[:,0],kind,pad)
        loss = lp[torch.isfinite(lp)].float().square().mean()+flags.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_forward_backward_and_changed_time_components():
    model = build_decoder(CONFIG)
    tokens,audio,z,plan = inputs(model)
    out = model(tokens,torch.ones_like(tokens.types),prepare(model,audio,z,plan))
    out.type_logits.square().mean().backward()
    assert torch.isfinite(out.hidden).all() and model.embedding.kind.weight.grad.abs().sum()>0
    c = config()
    c['decoder_time_components'] = 3
    c['decoder_time_distributions'] = {name:dict(v,initial_centers=v['initial_centers'][:3],
        initial_scales=v['initial_scales'][:3]) for name,v in c['decoder_time_distributions'].items()}
    other = build_decoder(c)
    assert other.heads.time_parameters(torch.randn(2,24),'MOMENT').shape == (2,3,3)
