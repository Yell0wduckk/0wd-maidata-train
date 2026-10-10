"""Compare the restricted joint attention with an explicit dense reference."""

import math

import pytest
import torch

from maidata.audio.attention import apply_rope, global_joint_attention, local_joint_attention


def dense_attention(q, k, v, allowed):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
    weights = scores.softmax(-1).masked_fill(~allowed, 0)
    return weights @ v


@pytest.mark.parametrize("local", [True, False])
def test_joint_keys_match_dense_outputs_and_gradients(local):
    torch.manual_seed(22)
    b, h, n, s, g, d = 2, 2, 9, 3, 4, 6
    shapes = ([(b, h, n, d)] * 3 + [(b, h, s, d)] * 2 if local
              else [(b, h, s, d)] + [(b, h, g, d)] * 2 + [(b, h, n, d)] * 2)
    tensors = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes]
    reference_tensors = [value.detach().clone().requires_grad_() for value in tensors]
    l_valid = torch.tensor([[False, True, True, True, False, True, True, True, False], [False] * n])
    s_valid = torch.tensor([[True, False, True], [False] * s])
    g_valid = torch.tensor([[True, True, False, True], [False] * g])
    beats = (torch.arange(n).float() / 3 - 1.7).expand(b, -1)
    if local:
        result = local_joint_attention(*tensors, beats, l_valid, s_valid, 0.7, 0.0)
        q, kl, vl, ks, vs = reference_tensors
        allowed_l = ((beats[:, :, None] - beats[:, None, :]).abs() <= 0.7) & l_valid[:, None, :]
        allowed = torch.cat((allowed_l, s_valid[:, None, :].expand(-1, n, -1)), -1)
        allowed &= l_valid[:, :, None]
        reference = dense_attention(q, torch.cat((kl, ks), 2), torch.cat((vl, vs), 2), allowed[:, None])
    else:
        result = global_joint_attention(*tensors, s_valid, g_valid, l_valid, 0.0)
        q, kg, vg, kl, vl = reference_tensors
        allowed = s_valid[:, None, :, None] & torch.cat((g_valid, l_valid), -1)[:, None, None, :]
        reference = dense_attention(q, torch.cat((kg, kl), 2), torch.cat((vg, vl), 2), allowed)
    torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(result)
    (result * probe).sum().backward()
    (reference * probe).sum().backward()
    for actual, expected in zip(tensors, reference_tensors):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-9, atol=1e-10)
    assert torch.count_nonzero(result[1]) == 0


def test_local_attention_chunks_do_not_change_edges():
    torch.manual_seed(9)
    q, k, v = [torch.randn(1, 2, 133, 4) for _ in range(3)]
    ks, vs = [torch.randn(1, 2, 3, 4) for _ in range(2)]
    beats = torch.arange(133).float()[None] / 5 - 8.0
    valid = torch.ones(1, 133, dtype=torch.bool)
    valid[:, 20:26] = False
    s_valid = torch.tensor([[True, False, True]])
    actual = local_joint_attention(q, k, v, ks, vs, beats, valid, s_valid, 0.65, 0.0)
    local = (beats[:, :, None] - beats[:, None, :]).abs() <= 0.65
    allowed = torch.cat((local & valid[:, None], s_valid[:, None].expand(-1, 133, -1)), -1)
    allowed &= valid[:, :, None]
    expected = dense_attention(q, torch.cat((k, ks), 2), torch.cat((v, vs), 2), allowed[:, None])
    torch.testing.assert_close(actual, expected)


def test_rope_full_head_fractional_absolute_beats():
    value = torch.tensor([[[[1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.]]]])
    beats = torch.tensor([[-2.75, 1.5]])
    base, scale = 47.0, 0.8
    actual = apply_rope(value, beats, base, scale)
    expected = torch.empty_like(value)
    for t in range(2):
        for pair in range(3):
            phase = float(beats[0, t]) * scale * base ** (-2 * pair / 6)
            x, y = value[0, 0, t, pair * 2:pair * 2 + 2]
            expected[0, 0, t, pair * 2] = x * math.cos(phase) - y * math.sin(phase)
            expected[0, 0, t, pair * 2 + 1] = x * math.sin(phase) + y * math.cos(phase)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(apply_rope(value[..., :1, :], beats[:, :1], base, scale), actual[..., :1, :])
    assert not torch.allclose(actual[..., 0, :], actual[..., 1, :])
    assert apply_rope(value.bfloat16(), beats, base, scale).dtype == torch.bfloat16
