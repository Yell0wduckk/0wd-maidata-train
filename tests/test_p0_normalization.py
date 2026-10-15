"""One P0 producer normalization: masks, scale, graph and legacy recipes."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from maidata.audio.pyramid import LocalPyramid
from maidata.model import build_model, encoder_config as encoder_recipe_config
from test_maidata_geometry import stage_grid


@pytest.mark.parametrize('bf16', [False, True])
def test_p0_normalization_bounds_large_fusion_and_keeps_other_scales_and_graph(bf16):
    stages = [stage_grid(n, width=8) for n in (48, 24, 12)]
    for stage in stages:
        stage.values = (torch.randn_like(stage.values)*3).requires_grad_()
        stage.valid[:, :2] = False
        stage.values = stage.values.masked_fill(~stage.valid[..., None], 0).detach().requires_grad_()
    legacy = LocalPyramid(8, (5, 3, 2), 3, 1, p0_normalization=False)
    fixed = LocalPyramid(8, (5, 3, 2), 3, 1, p0_normalization=True)
    with torch.no_grad():
        for projection in legacy.projections:
            projection.weight.mul_(1000)
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        before = legacy(stages, torch.tensor([-2]))
        after = fixed(stages, torch.tensor([-2]))
    assert before.P0.values.abs().max() > 1000
    assert after.P0.values.abs().max() < 3
    expected = F.layer_norm(before.P0.values, (8,)).masked_fill(~before.P0.valid[..., None], 0)
    torch.testing.assert_close(after.P0.values, expected)
    for one, two in zip((before.P0, before.P1, before.P2), (after.P0, after.P1, after.P2)):
        assert torch.equal(one.beats, two.beats) and torch.equal(one.valid, two.valid)
    torch.testing.assert_close(before.P1.values, after.P1.values)
    torch.testing.assert_close(before.P2.values, after.P2.values)
    (after.P0.values.float()*torch.randn_like(after.P0.values).float()).sum().backward()
    assert all(torch.isfinite(s.values.grad).all() and s.values.grad.abs().sum() > 0 for s in stages)
    assert all(p.weight.grad is not None and torch.isfinite(p.weight.grad).all() and
               p.weight.grad.abs().sum() > 0 for p in fixed.projections)
    empty = fixed([replace(s, values=torch.full_like(s.values, float('nan')),
                           valid=torch.zeros_like(s.valid)) for s in stages], torch.tensor([-2]))
    assert torch.equal(empty.P0.values, torch.zeros_like(empty.P0.values))


def test_old_recipe_retains_identity_and_normalization_has_no_new_parameters(encoder_config):
    old = dict(encoder_config)
    old.pop('fpn_p0_normalization', None)
    assert encoder_recipe_config(old)['fpn_p0_normalization'] is False
    legacy, fixed = build_model(old), build_model(dict(old, fpn_p0_normalization=True))
    assert isinstance(legacy.pyramid.p0_norm, torch.nn.Identity)
    assert isinstance(fixed.pyramid.p0_norm, torch.nn.LayerNorm)
    assert fixed.pyramid.p0_norm.elementwise_affine is False
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.state_dict().keys() == fixed.state_dict().keys()
