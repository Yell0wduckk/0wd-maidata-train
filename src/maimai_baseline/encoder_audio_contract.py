"""Shared deterministic native-audio bottleneck and training perturbation."""
from __future__ import annotations

import torch
from torch.nn import functional as F

REPRESENTATION = 'layernorm-mu-eps1e-5'
LEGACY_REPRESENTATION = 'deterministic-mu'
DIMENSION = 128
MU_EPS = 1e-5
NOISE_PROBABILITY = .5
NOISE_STD = .05
NLL_COEFFICIENT = .25
VARIANCE_COEFFICIENT = 1.
VARIANCE_EPS = 1e-4
VARIANCE_FLOOR = 1.
CONTRACT = dict(representation=REPRESENTATION, dimension=DIMENSION, beats_per_token=4,
    audio_head=dict(layers=2, width=128, heads=4, ffn=512, components=4,
                    input='native-stitched-p0;absolute-block-centers;no-gt-prefix'),
    nll_coefficient=NLL_COEFFICIENT, nll_reduction='joint-128d-mixture/valid-coordinate-count',
    variance_coefficient=VARIANCE_COEFFICIENT, variance_floor=VARIANCE_FLOOR,
    variance_epsilon=VARIANCE_EPS, variance_reduction='mean-coordinate;unique-chart-view-block',
    noise_probability=NOISE_PROBABILITY, noise_std=NOISE_STD,
    noise_scope='one-bernoulli-per-window;valid-coordinates-only',
    moment_target='absolute-window-position-tick')


def normalize_mu(raw, valid):
    if raw.shape[-1] != DIMENSION or valid.shape != raw.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('mu normalization requires 128 coordinates and boolean validity')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        clean = raw.float().masked_fill(~valid[..., None], 0.)
        return F.layer_norm(clean, (DIMENSION,), eps=MU_EPS).masked_fill(~valid[..., None], 0.)


def perturb_mu(tokens, valid, *, training=True, generator=None, return_mask=False):
    """One shared draw per window, reused by every consumer in that forward."""
    if tokens.ndim != 3 or tokens.shape[1:] != (4, DIMENSION) or valid.shape != tokens.shape[:2]:
        raise ValueError('latent perturbation requires [windows,4,128] and [windows,4]')
    clean = tokens.masked_fill(~valid[..., None], 0.)
    selected = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    if training:
        selected = torch.rand((len(tokens),), device=tokens.device, generator=generator) < NOISE_PROBABILITY
        noise = torch.randn(tokens.shape, device=tokens.device, dtype=tokens.dtype, generator=generator)
        clean = clean + noise * NOISE_STD * selected[:, None, None] * valid[..., None]
    return (clean, selected) if return_mask else clean
