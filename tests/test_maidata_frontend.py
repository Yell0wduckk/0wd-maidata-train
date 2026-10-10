"""Small CPU checks for the independent trainable audio frontends."""

import inspect

import pytest
import torch
from torch.nn import functional as F

from maidata.audio.frontend import (
    AudioFeatureFusion,
    BeatPatchTokenizer,
    SpectralTokenizer,
)


def test_frontend_constructors_require_explicit_parameters():
    for module in (SpectralTokenizer, AudioFeatureFusion, BeatPatchTokenizer):
        parameters = inspect.signature(module).parameters.values()
        assert all(parameter.default is inspect.Parameter.empty for parameter in parameters)


@pytest.mark.parametrize(
    "mel_bins,stem_channels,frequency_dim,queries,d_model",
    [(128, 32, 64, 4, 256), (17, 5, 7, 3, 20)],
)
def test_spectral_shape_and_dynamic_frequency_initialization(
    mel_bins, stem_channels, frequency_dim, queries, d_model
):
    tokenizer = SpectralTokenizer(mel_bins, stem_channels, frequency_dim, queries, d_model, 0.0)
    mel = torch.randn(2, 7, mel_bins)
    valid = torch.ones(2, 7, dtype=torch.bool)
    values = tokenizer(mel, valid)
    assert values.shape == (2, 7, d_model)
    assert torch.isfinite(values).all()
    pool = tokenizer.frequency_pool
    torch.testing.assert_close(pool.queries, torch.zeros(queries, frequency_dim))
    torch.testing.assert_close(pool.dynamic_query[-1].weight, torch.zeros_like(pool.dynamic_query[-1].weight))
    torch.testing.assert_close(pool.dynamic_query[-1].bias, torch.zeros_like(pool.dynamic_query[-1].bias))
    torch.testing.assert_close(pool.raw_centers.sigmoid(), (torch.arange(queries) + 0.5) / queries)
    initial_width = 1 / (2 * queries)
    torch.testing.assert_close(F.softplus(pool.raw_widths) + 1e-4, torch.full((queries,), initial_width))
    # Odd Mel counts exercise the actual stride-2 output width, rather than 64.
    stem = tokenizer.stem(mel.unsqueeze(1))
    features = torch.randn(2, 7, stem.shape[-1], frequency_dim)
    weights = pool.attention_weights(features)
    frequencies = (torch.arange(stem.shape[-1]) + 0.5) / stem.shape[-1]
    bias = -0.5 * ((frequencies[None] - pool.raw_centers.sigmoid()[:, None]) /
                   initial_width).square()
    torch.testing.assert_close(weights, bias.softmax(-1).expand(2, 7, -1, -1))


@pytest.mark.parametrize("queries", [5000, 5001])
def test_frequency_query_count_remains_configurable_at_width_floor(queries):
    # The requested initial width can reach or fall below the numerical floor.
    tokenizer = SpectralTokenizer(4, 1, 2, queries, 2, 0.0)
    assert torch.isfinite(tokenizer.frequency_pool.raw_widths).all()
    widths = F.softplus(tokenizer.frequency_pool.raw_widths) + 1e-4
    torch.testing.assert_close(widths, torch.full((queries,), 1e-4))
    output = tokenizer(torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))
    assert output.shape == (1, 2, 2)
    assert torch.isfinite(output).all()


def test_spectral_padding_is_ignored_by_convolution_and_normalization():
    torch.manual_seed(10)
    tokenizer = SpectralTokenizer(13, 8, 6, 2, 12, 0.0).eval()
    core = torch.randn(1, 5, 13)
    core_valid = torch.ones(1, 5, dtype=torch.bool)
    reference = tokenizer(core, core_valid)
    padded = F.pad(core, (0, 0, 2, 3))
    valid = F.pad(core_valid, (2, 3))
    padded = padded.masked_fill(~valid.unsqueeze(-1), float("nan"))
    output = tokenizer(padded, valid)
    torch.testing.assert_close(output[:, 2:7], reference, rtol=1e-5, atol=2e-6)
    assert torch.count_nonzero(output[~valid]) == 0
    empty = tokenizer(torch.full_like(padded, float("nan")), torch.zeros_like(valid))
    assert torch.isfinite(empty).all()
    assert torch.count_nonzero(empty) == 0
    # Zero-valued real audio keeps its supplied validity and learned biases.
    silent = tokenizer(torch.zeros_like(core), core_valid)
    assert torch.isfinite(silent).all()
    assert torch.count_nonzero(silent) > 0


@pytest.mark.parametrize("raw_width", [-45.0, -100.0])
def test_narrow_frequency_queries_keep_outputs_and_gradients_finite(raw_width):
    torch.manual_seed(16)
    tokenizer = SpectralTokenizer(128, 8, 8, 4, 16, 0.0)
    with torch.no_grad():
        tokenizer.frequency_pool.raw_widths.fill_(raw_width)
    mel = torch.randn(1, 5, 128, requires_grad=True)
    values = tokenizer(mel, torch.ones(1, 5, dtype=torch.bool))
    assert torch.isfinite(values).all()
    (values * torch.randn_like(values)).sum().backward()
    assert torch.isfinite(mel.grad).all()
    for name, parameter in tokenizer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_frontend_gradients_reach_mel_aux_and_dynamic_queries():
    torch.manual_seed(11)
    spectral = SpectralTokenizer(11, 4, 5, 3, 16, 0.0)
    fusion = AudioFeatureFusion(16)
    mel = torch.randn(2, 6, 11, requires_grad=True)
    aux = torch.randn(2, 6, 32, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, True, False], [False] * 6])
    values = fusion(spectral(mel, valid), aux, valid)
    values.square().mean().backward()
    assert torch.count_nonzero(mel.grad[valid]) > 0
    assert torch.count_nonzero(aux.grad[valid]) > 0
    assert torch.count_nonzero(mel.grad[~valid]) == 0
    assert torch.count_nonzero(aux.grad[~valid]) == 0
    for parameter in (spectral.stem.weight, spectral.feature_lift.weight,
                      spectral.frequency_pool.queries,
                      spectral.frequency_pool.dynamic_query[-1].weight,
                      spectral.frequency_pool.raw_centers,
                      spectral.frequency_pool.raw_widths, fusion.project.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_feature_fusion_is_one_biased_projection_and_masks_empty_audio():
    fusion = AudioFeatureFusion(10)
    spectral = torch.randn(1, 4, 10)
    aux = torch.randn(1, 4, 32)
    valid = torch.tensor([[True, False, True, False]])
    output = fusion(spectral, aux, valid)
    expected = F.linear(torch.cat((spectral, aux), -1), fusion.project.weight, fusion.project.bias)
    torch.testing.assert_close(output[valid], expected[valid])
    assert torch.count_nonzero(output[~valid]) == 0
    assert not any(isinstance(module, torch.nn.Dropout) for module in fusion.modules())


def test_beat_patch_preserves_partial_beats_and_masks_samples_before_projection():
    torch.manual_seed(12)
    tokenizer = BeatPatchTokenizer(7, 4, 14)
    mel = torch.randn(1, 3, 4, 7, requires_grad=True)
    valid = torch.tensor([[[True, False, True, False], [False] * 4, [True] * 4]])
    aux = torch.randn(1, 3, 32, requires_grad=True)
    output = tokenizer(mel, valid, aux)
    assert output.shape == (1, 3, 14)
    assert tokenizer.project.in_features == 4 * 7 + 4 + 32
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 0]) > 0
    assert torch.count_nonzero(output[:, 1]) == 0
    perturbed = mel.detach().masked_fill(~valid.unsqueeze(-1), float("nan"))
    dirty_aux = aux.detach().clone()
    dirty_aux[:, 1] = float("nan")
    torch.testing.assert_close(tokenizer(perturbed, valid, dirty_aux), output)
    # A weighted loss avoids the near-constant squared norm of LayerNorm.
    (output * torch.randn_like(output)).sum().backward()
    assert torch.count_nonzero(mel.grad[valid]) > 0
    assert torch.count_nonzero(mel.grad[~valid]) == 0
    assert torch.count_nonzero(aux.grad[:, 0]) > 0
    assert torch.count_nonzero(aux.grad[:, 1]) == 0
    assert torch.isfinite(tokenizer.project.weight.grad).all()
    assert not any(isinstance(module, torch.nn.Dropout) for module in tokenizer.modules())
    silent = tokenizer(torch.zeros_like(mel), torch.ones_like(valid), torch.zeros_like(aux))
    assert torch.isfinite(silent).all()
    assert torch.count_nonzero(silent) > 0
