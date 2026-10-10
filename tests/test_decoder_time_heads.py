"""Time priors and AMP readout match the original Decoder behavior."""
import pytest
import torch
from torch.nn import functional as F

from maidata.decoder.probability import interval_log_probabilities
from maidata.model import build_decoder, decoder_time_distributions
from test_decoder_network import config


KINDS = ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')


@pytest.mark.parametrize('kind', KINDS)
def test_time_prior_is_independent_of_hidden_and_remains_trainable(kind):
    torch.manual_seed(817)
    c = config()
    model = build_decoder(c)
    head = model.heads.times[kind]
    spec = decoder_time_distributions(c)[kind]
    hidden = torch.randn(2, c['decoder_dim'], requires_grad=True)
    raw = model.heads.time_parameters(hidden, kind)
    assert torch.count_nonzero(head.weight) == 0
    torch.testing.assert_close(raw[0], raw[1], rtol=0, atol=0)
    torch.testing.assert_close(raw[..., 0], torch.zeros_like(raw[..., 0]), rtol=0, atol=0)
    torch.testing.assert_close(raw[0, :, 1] * spec.reference,
                               torch.tensor(spec.initial_centers), atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(F.softplus(raw[0, :, 2]) * spec.reference + spec.minimum_scale,
                               torch.tensor(spec.initial_scales), atol=1e-4, rtol=1e-6)
    lower = torch.tensor([[1., 30., 100.], [1., 30., 100.]], dtype=torch.float64)
    upper = torch.tensor([[30., 100., 500.], [30., 100., 500.]], dtype=torch.float64)
    def objective():
        lp = interval_log_probabilities(model.heads.time_parameters(hidden, kind), lower, upper,
                                         torch.ones_like(lower, dtype=torch.bool), spec)
        return -lp[:, 1].sum()
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
    expected = F.linear(hidden.float(), head.weight, head.bias).reshape(2, 3, 5, 3)
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
