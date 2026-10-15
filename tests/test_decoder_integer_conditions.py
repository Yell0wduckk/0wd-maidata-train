"""Overlapping Plan16 uses independent centers and the common rotary softmax."""
from dataclasses import replace
import math

import pytest
import torch

from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens, audio_query_layout, audio_key_spans
from maidata.decoder.network import Decoder
from maidata.decoder.types import Memory
from maidata.nn.position import apply_rope


def decoder(*, z_enabled=True, layout='centered-overlap'):
    return Decoder(dim=24, audio_dim=18, latent_dim=10, layers=4, router_layer=1,
        heads=3, shared_ffn_dim=31, expert_ffn_dim=17, experts=3, readout_ffn_dim=19,
        dropout=0., norm_eps=1e-5, rope_base=10000., rope_position_scale=1.,
        note_dims=(4,8,2,2,2,3,3), route_dims=(20,4), time_base=10000., time_scale=1.,
        condition_gate_init=.4, z_cross_attention=z_enabled, z_normalization=True,
        plan_span_beats=16, planner_layout=layout).eval()


def inputs(model):
    starts = torch.tensor([0.,16.,32.])
    tokens = pack_tokens([[Moment(0)] + [Note.make('tap','A1')]*n for n in (4,2,0)],
        [[s+i*.5 for i in range(n+1)] for s,n in zip(starts,(4,2,0))], device='cpu')
    tokens.valid[0,2] = False
    audio_valid = torch.tensor([[1,1,1,1,1,1,1],[1,1,1,1,0,0,0]], dtype=torch.bool)
    audio = Memory(torch.randn(2,7,18,requires_grad=True), torch.arange(7.)[None].expand(2,-1),audio_valid)
    z_beats = starts[:,None] + torch.arange(-3.,20.)
    z_valid = (z_beats >= 0) & (z_beats <= 48)
    z_valid[1,8] = False
    z = Memory(torch.randn(3,23,10,requires_grad=True),z_beats,z_valid)
    plan_beats = starts[:,None] + torch.arange(0.,17.,4.)
    plan_valid = torch.tensor([[1,1,0,1,1],[1,1,1,1,1],[1,0,1,1,1]],dtype=torch.bool)
    plan = torch.randn(3,5,12,requires_grad=True)
    with torch.no_grad():
        plan[2,4].zero_()
        plan[2,4,8] = 1.  # A valid empty tail segment, rather than missing input.
    return tokens,audio,z,plan,plan_valid,plan_beats,torch.tensor([0,1,0])


def prepare(model,tokens,audio,z,plan,valid,centers,mapping,*,packed=False,compact=True):
    layout,indices = None,None
    if packed:
        z_valid = z.valid if model.condition.z_cross_attention else torch.zeros_like(z.valid)
        layout = audio_query_layout(tokens.valid,mapping,len(audio.valid),spans=audio_key_spans(audio.valid),
            condition_valid=torch.cat((z_valid,valid),-1) if compact else None)
        indices = audio.valid.flatten().nonzero().flatten()
    return model.prepare_memory(model.prepare_audio(audio,indices=indices),mapping,z,plan,valid,
        audio_layout=layout,plan_beats=centers)


def test_23_z_plus_five_independent_plan_centers_use_one_rotary_softmax():
    torch.manual_seed(83)
    model = decoder()
    tokens,audio,z,plan,valid,centers,mapping = inputs(model)
    condition = model.condition(z,plan,valid,centers)
    assert condition.values.shape == (3,28,24)
    assert torch.equal(condition.beats,torch.cat((z.beats,centers),-1))
    assert torch.equal(condition.valid,torch.cat((z.valid,valid),-1))
    memory = prepare(model,tokens,audio,z,plan,valid,centers,mapping)
    assert memory.plan_index is None
    module = model.layers[0].condition_attn
    kv = memory.condition[0]
    raw = module.split(module.key(condition.values.masked_fill(~condition.valid[...,None],0)))
    torch.testing.assert_close(kv.key,apply_rope(raw,condition.beats.masked_fill(~condition.valid,0),
        module.base,module.scale),atol=0,rtol=0)
    x = torch.randn(3,2,24)
    beats = torch.tensor([[0.,1.],[16.,17.],[32.,33.]])
    query_valid = torch.ones((3,2),dtype=torch.bool)
    q = apply_rope(module.split(module.query(x)),beats,module.base,module.scale)
    scores = (q @ kv.key.transpose(-1,-2))/math.sqrt(q.shape[-1])
    probability = scores.masked_fill(~kv.valid[:,None,None,:],-torch.inf).softmax(-1)
    expected = module.output((probability @ kv.value).transpose(1,2).reshape(3,2,24))
    torch.testing.assert_close(module(x,beats,query_valid,kv),expected,atol=3e-7,rtol=3e-6)
    # Different geometry does not add parameters or change checkpoint tensors.
    old = decoder(layout='legacy')
    old.load_state_dict(model.state_dict(),strict=True)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize('z_enabled',[False,True])
@pytest.mark.parametrize('compact',[False,True])
def test_packed_dense_independent_masks_and_gradients(z_enabled,compact):
    torch.manual_seed(84)
    model = decoder(z_enabled=z_enabled)
    tokens,audio,z,plan,valid,centers,mapping = inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    outputs = [model(tokens,types,prepare(model,tokens,audio,z,plan,valid,centers,mapping,
        packed=packed,compact=compact)) for packed in (False,True)]
    flattened = [torch.cat((out.hidden,out.type_logits),-1) for out in outputs]
    torch.testing.assert_close(*flattened,atol=3e-6,rtol=3e-5)
    weights = torch.randn_like(flattened[0])
    parameters = (*model.parameters(),audio.values,z.values,plan)
    gradients = [torch.autograd.grad((out*weights).sum(),parameters,retain_graph=True,allow_unused=True)
        for out in flattened]
    for left,right in zip(*gradients):
        if left is None or right is None:
            assert left is right or (left if left is not None else right).count_nonzero() == 0
        else:
            torch.testing.assert_close(left,right,atol=5e-6,rtol=5e-5)
    (*_,audio_grad,z_grad,plan_grad) = gradients[0]
    assert plan_grad[valid].abs().sum() > 0 and plan_grad[~valid].count_nonzero() == 0
    if z_enabled:
        assert z_grad[z.valid].abs().sum() > 0 and z_grad[~z.valid].count_nonzero() == 0
    else:
        assert z_grad is None
    assert audio_grad[audio.valid].abs().sum() > 0


@pytest.mark.parametrize('packed_prefill',[False,True])
def test_incremental_and_shift_match_full(packed_prefill):
    torch.manual_seed(85)
    model = decoder()
    tokens,audio,z,plan,valid,centers,mapping = inputs(model)
    types = torch.arange(tokens.valid.numel()).reshape(tokens.valid.shape)%5
    full = model(tokens,types,prepare(model,tokens,audio,z,plan,valid,centers,mapping))
    prefix = 2
    memory = prepare(model,tokens.slice(0,prefix),audio,z,plan,valid,centers,mapping,packed=packed_prefill)
    prefill = model(tokens.slice(0,prefix),types[:,:prefix],memory)
    cache,hidden = prefill.cache,[prefill.hidden]
    for i in range(prefix,tokens.valid.shape[1]):
        out = model.complete(model.begin(tokens.slice(i,i+1),memory,cache=cache),types[:,i:i+1])
        cache = out.cache
        hidden.append(out.hidden)
    torch.testing.assert_close(torch.cat(hidden,1),full.hidden,atol=3e-6,rtol=3e-5)
    shift = 13.25
    shifted = replace(tokens,beats=tokens.beats+shift,slide_beats=tokens.slide_beats+shift)
    shifted_out = model(shifted,types,prepare(model,shifted,replace(audio,beats=audio.beats+shift),
        replace(z,beats=z.beats+shift),plan,valid,centers+shift,mapping))
    torch.testing.assert_close(shifted_out.hidden,full.hidden,atol=4e-6,rtol=4e-5)


def test_independent_invalid_coordinates_are_inert_and_empty_plan_remains_valid():
    torch.manual_seed(86)
    model = decoder()
    tokens,audio,z,plan,valid,centers,mapping = inputs(model)
    types = torch.ones_like(tokens.types)
    original = model(tokens,types,prepare(model,tokens,audio,z,plan,valid,centers,mapping))
    poisoned = model(tokens,types,prepare(model,tokens,audio,
        replace(z,values=z.values.masked_fill(~z.valid[...,None],float('nan')),
            beats=z.beats.masked_fill(~z.valid,float('nan'))),
        plan.masked_fill(~valid[...,None],float('nan')),valid,
        centers.masked_fill(~valid,float('nan')),mapping))
    torch.testing.assert_close(poisoned.hidden,original.hidden)
    missing = valid.clone()
    missing[2,4] = False
    no_tail = model(tokens,types,prepare(model,tokens,audio,z,plan,missing,centers,mapping))
    assert not torch.allclose(no_tail.hidden[2,0],original.hidden[2,0])
    with pytest.raises(ValueError,match='independent plan_beats'):
        model.condition(z,plan,valid)


def test_plan_geometry_does_not_follow_z_coordinates_or_z_mask():
    torch.manual_seed(87)
    model = decoder(z_enabled=False)
    _,_,z,plan,valid,centers,_ = inputs(model)
    condition = model.condition(replace(z,beats=torch.full_like(z.beats,float('nan'))),plan,valid,centers)
    assert not condition.valid[:,:23].any()
    assert torch.equal(condition.beats[:,23:],centers)
    assert torch.equal(condition.valid[:,23:],valid)


def test_rejects_centered_plan4_or_unknown_layout():
    with pytest.raises(ValueError,match='planner_layout'):
        decoder(layout='unknown')
    from maidata.decoder.attention import ConditionAdapter
    with pytest.raises(ValueError,match='requires Plan16'):
        ConditionAdapter(10,24,True,1e-5,plan_span_beats=4,planner_layout='centered-overlap')


def test_bfloat16_packed_gradients_and_all_disabled_conditions():
    torch.manual_seed(88)
    model = decoder()
    tokens,audio,z,plan,valid,centers,mapping = inputs(model)
    types = torch.ones_like(tokens.types)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        outputs = [model(tokens,types,prepare(model,tokens,audio,z,plan,valid,centers,mapping,packed=packed))
            for packed in (False,True)]
    eps = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(outputs[0].hidden,outputs[1].hidden,atol=2*eps,rtol=4*eps)
    (outputs[1].hidden*torch.randn_like(outputs[1].hidden)).sum().backward()
    for value,mask in ((audio.values,audio.valid),(z.values,z.valid),(plan,valid)):
        assert torch.isfinite(value.grad).all() and value.grad[mask].abs().sum() > 0
        assert value.grad[~mask].count_nonzero() == 0
    empty_z = replace(z,valid=torch.zeros_like(z.valid))
    missing = torch.zeros_like(valid)
    for packed in (False,True):
        out = model(tokens,types,prepare(model,tokens,audio,empty_z,plan,missing,centers,mapping,packed=packed))
        assert torch.isfinite(out.hidden).all()
