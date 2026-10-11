"""Network boundaries, configurable dimensions, padding, reuse and gradients."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from maidata.entry import CONFIG
from maidata.model import build_draft, build_planner, build_reconstruction
from maidata.draft.types import BeatMemory, DraftInputs, WindowLatents
from maidata.draft.windows import select_windows
from maidata.draft.generator import DenseMoE
from maidata.nn.position import apply_rope
from maidata.draft.chart_data import ChartData, OnsetGroup
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.chart.types import Note
from maidata.time import TimingMap


@pytest.fixture
def config():
    return dict(CONFIG, d_model=18, draft_layers=2, draft_dim=24, draft_heads=3,
                draft_experts=3, draft_expert_ffn_dim=29, draft_latent_dim=24,
                draft_dropout=0., planner_dim=16, planner_heads=2, planner_mlp_dim=23,
                planner_dropout=0., reconstruction_ffn_dim=25, reconstruction_dropout=0.,
                target_beats=3)


def query_batch(windows, notes=1):
    timing = TimingMap([(0, 120)])
    targets = []
    for start, valid in zip(windows.target_start.tolist(), windows.window_valid.tolist()):
        chart = ChartData((OnsetGroup(int(start*96), (Note.make('tap', 'A1'),)*notes),),
                          (), timing, 1000)
        targets.append(reconstruction_target(chart, start_tick=int(start*96),
                       target_beats=windows.target_beats) if valid else None)
    return collate_reconstruction(targets, target_beats=windows.target_beats,
                                  device=windows.memory.values.device).queries


def inputs():
    torch.manual_seed(14)
    audio = BeatMemory(torch.randn(2, 7, 18, requires_grad=True),
        torch.tensor([[-2., -1.5, -.8, 0., 1., 1.5, 2.], [-1., 0., 1., 2., 3., 4., 5.]]),
        torch.tensor([[True] * 5 + [False] * 2, [True] * 7]))
    beats = (torch.arange(9).float() - 1.5).expand(3, -1).clone()
    valid = torch.ones(3, 9, dtype=torch.bool)
    valid[2, -2:] = False
    condition = DraftInputs(beats, valid, torch.tensor([0, 0, 1]), torch.tensor([3, 5, 7]),
                            torch.tensor([10.8, 14.6, 15.]), torch.tensor([5., 5., 7.]))
    return audio, condition


def test_widths_conditions_and_output_normalization(config):
    model = build_draft(config).eval()
    audio, condition = inputs()
    z = model(audio, condition)
    assert z.values.shape == (3, 9, 24) and z.values.dtype == torch.float32
    assert torch.count_nonzero(z.values[~z.valid]) == 0
    torch.testing.assert_close(z.values[z.valid].mean(-1), torch.zeros(z.valid.sum()), atol=1e-6, rtol=0)
    torch.testing.assert_close(z.values[z.valid].var(-1, unbiased=False), torch.ones(z.valid.sum()), atol=1e-3, rtol=0)
    assert not torch.allclose(z.values[0], z.values[1])  # Same audio, distinct difficulty/level.
    for field in ('level', 'span_beats', 'difficulty_id'):
        changed = getattr(condition, field).clone()
        changed[0] += 1
        result = model(audio, replace(condition, **{field: changed}))
        assert not torch.allclose(z.values[0], result.values[0])
        torch.testing.assert_close(z.values[1:], result.values[1:])
    altered = replace(audio, values=audio.values.detach().clone())
    altered.values[0, 0] += torch.arange(18)
    changed = model(altered, condition)
    assert not torch.allclose(z.values[0], changed.values[0])
    torch.testing.assert_close(z.values[2], changed.values[2])


def test_exported_z_is_exact_final_layernorm_without_projection(config):
    model = build_draft(config).eval()
    audio, condition = inputs()
    captured = []
    hook = model.blocks[-1].register_forward_hook(lambda _m, _args, output: captured.append(output))
    z = model(audio, condition)
    hook.remove()
    expected = torch.nn.functional.layer_norm(captured[0].float(), model.output_norm.normalized_shape,
        model.output_norm.weight.float(), model.output_norm.bias.float(), model.output_norm.eps)
    torch.testing.assert_close(z.values, expected.masked_fill(~z.valid[..., None], 0), rtol=0, atol=0)
    assert not hasattr(model, 'output') and not hasattr(model, 'latent_norm')
    (z.values * torch.randn_like(z.values)).sum().backward()
    assert model.output_norm.weight.grad.abs().sum() > 0
    assert model.blocks[-1].experts.experts[0][0].weight.grad.abs().sum() > 0


def test_project_once_shared_graph_matches_separate_conditions(config):
    model = build_draft(config).eval()
    reference = deepcopy(model)
    audio, condition = inputs()
    other = replace(audio, values=audio.values.detach().clone().requires_grad_())
    with patch.object(model.blocks[0].cross_attention, 'project_memory',
                      wraps=model.blocks[0].cross_attention.project_memory) as projection:
        projected = model.project_audio(audio)
        z = model.from_projected(projected, condition)
        repeated = model.from_projected(projected, condition)
        assert projection.call_count == 1
        torch.testing.assert_close(z.values, repeated.values)
    probe = torch.randn_like(z.values)
    (z.values * probe).sum().backward()
    for i in range(3):
        one = DraftInputs(*(getattr(condition, field)[i:i + 1] for field in condition.__dataclass_fields__ if field != "valid_indices"))
        result = reference(other, one)
        torch.testing.assert_close(result.values[0], z.values[i], atol=3e-6, rtol=3e-5)
        (result.values * probe[i:i + 1]).sum().backward()
    torch.testing.assert_close(audio.values.grad, other.values.grad, atol=5e-6, rtol=5e-5)
    assert audio.values.grad[audio.valid].abs().sum() > 0
    assert audio.values.grad[~audio.valid].count_nonzero() == 0
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)


def test_dense_experts_independent_and_all_used(config):
    model = build_draft(config)
    audio, condition = inputs()
    z = model(audio, condition)
    (z.values * torch.randn_like(z.values)).sum().backward()
    first, second = model.blocks
    assert first.experts.gate.weight is not second.experts.gate.weight
    for block in model.blocks:
        assert len(block.experts.experts) == 3
        assert block.experts.gate.weight.count_nonzero() == 0  # No optimizer has run.
        assert block.experts.gate.weight.grad.abs().sum() > 0
        assert all(expert[0].weight.grad.abs().sum() > 0 for expert in block.experts.experts)
    moe = DenseMoE(4, 3, 7, 0.)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(moe(x), torch.stack([expert(x) for expert in moe.experts]).mean(0))


def test_nan_padding_and_dummy_windows_are_zero(config):
    d, p, r = build_draft(config).eval(), build_planner(config).eval(), build_reconstruction(config).eval()
    audio, condition = inputs()
    clean = d(audio, condition)
    dirty = replace(audio, values=audio.values.masked_fill(~audio.valid[..., None], float('nan')),
                    beats=audio.beats.masked_fill(~audio.valid, float('nan')))
    z = d(dirty, replace(condition, query_beats=condition.query_beats.masked_fill(~condition.draft_valid, float('nan'))))
    torch.testing.assert_close(z.values, clean.values)
    windows = select_windows(clean, torch.tensor([0, 0]), torch.tensor([-2., 100.]),
                             torch.tensor([True, False]), target_beats=3, before_beats=2, after_beats=1)
    for module in (p, r):
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name.endswith('bias'):
                    parameter.fill_(.7)
    plan, reconstruction = p(windows), r(windows, query_batch(windows))
    assert plan.valid.tolist() == [True, False]
    assert plan.plan_values[1].count_nonzero() == 0
    assert reconstruction.empty_logits[1].count_nonzero() == 0
    assert reconstruction.positive_count_logits[1].count_nonzero() == 0
    assert all(t[1].count_nonzero() == 0 for t in reconstruction.field_logits.values())
    assert torch.isfinite(plan.plan_values).all()


def test_planner_uses_target_only_with_relative_beat_rope(config):
    planner = build_planner(config).eval()
    x = torch.randn(1, 7, 24, requires_grad=True)
    beats = torch.arange(7).float()[None] - 1.5
    window = WindowLatents(BeatMemory(x, beats, torch.ones(1, 7, dtype=torch.bool)),
                           torch.tensor([0.]), 3, torch.tensor([True]))
    output = planner(window)
    changed = replace(window, memory=replace(window.memory,
        values=x + ((beats < 0) | (beats >= 3))[..., None] * 100))
    torch.testing.assert_close(output.plan_values, planner(changed).plan_values)
    shifted = replace(window, target_start=window.target_start - 7,
                      memory=replace(window.memory, beats=beats - 7))
    torch.testing.assert_close(output.plan_values, planner(shifted).plan_values, atol=2e-6, rtol=2e-5)
    uneven = replace(window, memory=replace(window.memory, beats=beats + torch.tensor([[0., 0., .2, -.2, 0., 0., 0.]])))
    assert not torch.allclose(output.plan_values, planner(uneven).plan_values)
    (output.plan_values * torch.arange(12)).sum().backward()
    assert x.grad[:, 2:5].abs().sum() > 0 and x.grad[:, :2].count_nonzero() == 0
    torch.testing.assert_close(output.plan_values[:, 8:].sum(-1), torch.ones(1))


def test_reconstruction_has_no_row_interaction_and_relative_positions(config):
    short = build_reconstruction(config).eval()
    long = build_reconstruction(config).eval()
    long.load_state_dict(short.state_dict(), strict=True)
    windows = WindowLatents(BeatMemory(torch.randn(1, 6, 24), torch.arange(6)[None].float() - 1.5,
                                       torch.ones(1, 6, dtype=torch.bool)),
                            torch.tensor([0.]), 3, torch.tensor([True]))
    full_queries = query_batch(windows, notes=240)
    short_queries = replace(full_queries, **{name: getattr(full_queries, name)[:, :5]
                            for name in full_queries.__dataclass_fields__})
    a, b = short(windows, short_queries), long(windows, full_queries)
    for name in a.field_logits:
        torch.testing.assert_close(a.field_logits[name], b.field_logits[name][:, :5])
    torch.testing.assert_close(a.empty_logits, b.empty_logits)
    torch.testing.assert_close(a.positive_count_logits, b.positive_count_logits)
    shifted = replace(windows, target_start=windows.target_start + 12,
                      memory=replace(windows.memory, beats=windows.memory.beats + 12))
    shifted_queries = replace(short_queries, beats=short_queries.beats+12,
                              origins=short_queries.origins+12)
    torch.testing.assert_close(a.field_logits['kind'],
                               short(shifted, shifted_queries).field_logits['kind'])
    assert not any('attention' in name for name, _ in short.named_modules())


def test_rotary_reference_and_cpu_autocast(config):
    x, beats = torch.randn(2, 3, 7, 8), torch.randn(2, 7)
    actual = apply_rope(x, beats, 1234., .7)
    complex_x = torch.view_as_complex(x.reshape(2, 3, 7, 4, 2))
    phase = beats[:, None, :, None] * .7 * 1234 ** (-torch.arange(4).float() / 4)
    expected = torch.view_as_real(complex_x * torch.polar(torch.ones_like(phase), phase)).flatten(-2)
    torch.testing.assert_close(actual, expected)
    model = build_draft(config).eval()
    audio, condition = inputs()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        z = model(audio, condition)
        windows = select_windows(z, torch.tensor([0]), torch.tensor([-2.]), torch.tensor([True]),
                                 target_beats=3, before_beats=0, after_beats=0)
        plan = build_planner(config).eval()(windows)
        reconstruction = build_reconstruction(config).eval()(windows, query_batch(windows))
    assert z.values.dtype == torch.float32 and torch.isfinite(z.values).all()
    assert plan.plan_values.dtype == torch.float32
    assert torch.isfinite(reconstruction.empty_logits).all()
    assert torch.isfinite(reconstruction.positive_count_logits).all()


def test_empty_draft_batch_slot_cannot_revive_from_nan_padding(config):
    model = build_draft(config).eval()
    audio = BeatMemory(torch.full((1, 2, 18), float('nan')), torch.full((1, 2), float('nan')),
                       torch.zeros(1, 2, dtype=torch.bool))
    inputs = DraftInputs(torch.full((1, 3), float('nan')), torch.zeros(1, 3, dtype=torch.bool),
                         torch.tensor([0]), torch.tensor([5]), torch.tensor([14.6]), torch.tensor([1.]))
    z = model(audio, inputs)
    assert z.values.count_nonzero() == 0 and torch.isfinite(z.values).all()
    # The selector also handles arbitrary padded producer values without NaN * 0.
    z = replace(z, values=z.values.masked_fill(~z.valid[..., None], float('nan')))
    selected = select_windows(z, torch.tensor([0]), torch.tensor([0.]), torch.tensor([False]),
                              target_beats=3, before_beats=1, after_beats=1)
    assert selected.memory.values.count_nonzero() == 0 and torch.isfinite(selected.memory.values).all()


@pytest.mark.parametrize('overrides', [dict(draft_heads=5), dict(planner_heads=16),
                                       dict(draft_expert_ffn_dim=0), dict(draft_latent_dim=12)])
def test_configuration_rejects_mathematically_invalid_shapes(config, overrides):
    config.update(overrides)
    with pytest.raises(ValueError):
        build_draft(config)
        build_planner(config)
        build_reconstruction(config)
