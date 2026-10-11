"""Exact singleton attention and normalized plan memory regression coverage."""
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F

from maidata.decoder.condition_attention import attend
from maidata.decoder.attention import Attention, ConditionAdapter
from maidata.decoder.packed_attention import condition_attention
from maidata.decoder.row_layout import window_layout
from maidata.decoder.types import KV, Memory


CASES = [('cpu', torch.float32)]
if torch.cuda.is_available():
    CASES += [('cuda', torch.float32), ('cuda', torch.bfloat16)]


@pytest.mark.parametrize('device,dtype', CASES)
@pytest.mark.parametrize('scale', [1., 10000.])
@pytest.mark.parametrize('padded', [False, True])
def test_singleton_is_exact_value_and_has_no_qk_gradient(device, dtype, scale, padded):
    torch.manual_seed(81)
    width = 5 if padded else 1
    q = (torch.randn(2, 3, 7, 8, device=device, dtype=dtype)*scale).requires_grad_()
    k = (torch.randn(2, 3, width, 8, device=device, dtype=dtype)*scale).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    valid = torch.zeros(2, width, device=device, dtype=torch.bool)
    valid[:, -1] = True
    with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=AssertionError('singleton SDPA')):
        result = attend(q, k, v, valid, scale=8**-.5, dropout_p=0.)
    expected = v[:, :, -1:].expand(-1, -1, 7, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    upstream = torch.randn_like(result)
    got = torch.autograd.grad(result, (q, k, v), upstream, allow_unused=True)
    reference = torch.autograd.grad(expected, v, upstream)[0]
    assert got[0] is None and got[1] is None
    torch.testing.assert_close(got[2], reference, rtol=0, atol=0)


@pytest.mark.parametrize('device,dtype', CASES)
def test_mixed_zero_one_many_masks_do_not_softmax_singletons(device, dtype):
    torch.manual_seed(82)
    q = torch.randn(3, 2, 4, 8, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(3, 2, 5, 8, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    valid = torch.tensor([[0,0,0,0,0], [0,0,1,0,0], [1,0,1,1,0]], device=device, dtype=torch.bool)
    sdpa = F.scaled_dot_product_attention
    with patch('torch.nn.functional.scaled_dot_product_attention', wraps=sdpa) as call:
        result = attend(q, k, v, valid, scale=8**-.5, dropout_p=0.)
    assert call.call_count == 1 and call.call_args.args[0].shape[0] == 1
    torch.testing.assert_close(result[0], torch.zeros_like(result[0]), rtol=0, atol=0)
    torch.testing.assert_close(result[1], v[1, :, 2:3].expand(-1, 4, -1), rtol=0, atol=0)
    reference = sdpa(q[2:], k[2:], v[2:], attn_mask=valid[2:,None,None,:], scale=8**-.5)
    torch.testing.assert_close(result[2:], reference)
    result.float().square().sum().backward()
    for gradient in (q.grad, k.grad):
        assert torch.count_nonzero(gradient[:2]) == 0 and torch.count_nonzero(gradient[2:]) > 0
    assert torch.count_nonzero(v.grad[0]) == 0 and torch.count_nonzero(v.grad[1]) > 0


def test_singleton_dropout_is_per_query_head_and_empty_bank_is_zero():
    q, k = torch.randn(2, 3, 32, 8), torch.randn(2, 3, 1, 8)
    v = torch.ones_like(k, requires_grad=True)
    torch.manual_seed(83)
    result = attend(q, k, v, torch.ones(2,1,dtype=torch.bool), scale=8**-.5, dropout_p=.5)
    assert set(result.unique().tolist()) == {0., 2.}
    assert torch.equal(result, result[..., :1].expand_as(result))
    result.sum().backward()
    assert v.grad is not None and v.grad.abs().sum() > 0
    empty = attend(q, k[:,:,:0], v[:,:,:0], torch.zeros(2,0,dtype=torch.bool), scale=8**-.5, dropout_p=0.)
    assert empty.shape == q.shape and torch.count_nonzero(empty) == 0


@pytest.mark.parametrize('compact', [False, True])
def test_dense_packed_condition_outputs_and_gradients_with_mixed_masks(compact):
    torch.manual_seed(84)
    module = Attention(24, 24, 3, 0., 10000., 1.)
    valid = torch.tensor([[1,1,0,1], [1,1,1,1], [0,1,1,0]],dtype=torch.bool)
    mask = torch.tensor([[0,0,1], [1,0,1], [0,0,0]],dtype=torch.bool)
    x = torch.randn(3,4,24,requires_grad=True)
    k = torch.randn(3,3,3,8,requires_grad=True)
    v = torch.randn_like(k,requires_grad=True)
    beats = torch.arange(4).float()[None].expand(3,-1)
    layout = window_layout(valid, mask if compact else None)
    kv = KV(k,v,mask)
    dense = module(x,beats,valid,kv,plan_index=2)
    packed = condition_attention(module,x.flatten(0,1)[layout['indices']],beats.flatten()[layout['indices']],
        valid,kv,2,layout)
    torch.testing.assert_close(packed,dense.flatten(0,1)[layout['indices']],atol=2e-6,rtol=2e-5)
    parameters = (x,k,v,*module.query.parameters(),*module.output.parameters())
    a = torch.autograd.grad(dense.square().sum(),parameters,retain_graph=True)
    b = torch.autograd.grad(packed.square().sum(),parameters)
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,atol=3e-6,rtol=3e-5)
    assert torch.count_nonzero(a[1][0]) == 0


@pytest.mark.parametrize('enabled', [False, True])
def test_plan_norm_after_source_embedding_preserves_z_and_gradients(enabled):
    torch.manual_seed(85)
    adapter = ConditionAdapter(10,24,enabled,1e-4)
    plan = (torch.randn(2,12)*1000).requires_grad_()
    z = Memory(torch.randn(2,4,10),torch.zeros(2,4),torch.ones(2,4,dtype=torch.bool))
    valid = torch.tensor([True,False])
    result = adapter(z,plan,valid)
    raw = adapter.plan(plan.masked_fill(~valid[:,None],0))+adapter.source.weight[1]
    expected = F.layer_norm(raw,(24,),adapter.plan_norm.weight,adapter.plan_norm.bias,1e-4)
    torch.testing.assert_close(result.values[:,-1],expected)
    assert result.values[0,-1].square().mean().sqrt().item() == pytest.approx(1.,abs=1e-4)
    if enabled:
        torch.testing.assert_close(result.values[:,:-1],adapter.z(z.values)+adapter.source.weight[0])
    (result.values[0,-1]*torch.arange(24)).sum().backward()
    for value in (plan,adapter.plan[0].weight,adapter.source.weight,adapter.plan_norm.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert torch.count_nonzero(plan.grad[1]) == 0
