"""Independent tick384 classification heads and precise AMP readout."""
import pytest
import torch
from torch.nn import functional as F

from maidata.model import build_decoder
from test_decoder_network import config


KINDS = ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')


@pytest.mark.parametrize('kind', KINDS)
def test_time_head_has_fixed_vocabulary_and_updates_hidden_and_parameters(kind):
    torch.manual_seed(817)
    c = config()
    model = build_decoder(c)
    head = model.heads.times[kind]
    hidden = torch.randn(2, c['decoder_dim'], requires_grad=True)
    raw = model.heads.time_logits(hidden,kind)
    assert raw.shape == (2,385) and not torch.equal(raw[0],raw[1])
    assert len({id(h.weight) for h in model.heads.times.values()})==3
    def objective():
        return F.cross_entropy(model.heads.time_logits(hidden,kind),torch.tensor([0,384]))
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    objective().backward()
    assert head.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.weight.grad).all()
    optimizer.step()
    optimizer.zero_grad()
    hidden.grad = None
    objective().backward()
    assert hidden.grad.abs().sum() > 0
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize('kind', KINDS)
def test_bf16_autocast_preserves_fp32_time_output_and_gradients(kind):
    torch.manual_seed(140)
    model = build_decoder(config())
    head = model.heads.times[kind]
    # Trained heads are nonzero. A zero prior alone would hide AMP regressions.
    with torch.no_grad():
        head.weight.normal_(std=.2)
    hidden = torch.randn(2, 3, head.in_features, dtype=torch.bfloat16, requires_grad=True)
    expected = F.linear(hidden.float(), head.weight, head.bias)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model.heads.time_parameters(hidden, kind)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    params = (hidden, head.weight, head.bias)
    expected_grad = torch.autograd.grad(expected.square().sum(), params)
    actual_grad = torch.autograd.grad(actual.square().sum(), params)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_checkpoint_load_preserves_trained_time_weights():
    model = build_decoder(config())
    with torch.no_grad():
        for head in model.heads.times.values():
            head.weight.normal_()
    restored = build_decoder(config())
    restored.load_state_dict(model.state_dict(), strict=True)
    for kind in KINDS:
        torch.testing.assert_close(restored.heads.times[kind].weight, model.heads.times[kind].weight,
                                   rtol=0, atol=0)
