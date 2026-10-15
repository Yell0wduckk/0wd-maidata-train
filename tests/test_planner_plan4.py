"""Plan4 preserves the old head, exact targets, local masks and segment reduction."""
from dataclasses import replace
from fractions import Fraction
from math import log1p
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from maidata.draft.chart_data import ChartData, PlanNote
from maidata.draft.losses import planner_loss, prepare_supervision
from maidata.draft.plan_targets import MusicPlanTargets, PlanTarget
from maidata.draft.planner import PlannerHead
from maidata.draft.targets import WindowTarget, collate_planner, collate_reconstruction, planner_target
from maidata.draft.types import BeatMemory, PlannerOutput, WindowLatents
from maidata.time import TimingMap
from maidata.training.data import AudioDataset


def head(span=16):
    return PlannerHead(latent_dim=12, dim=8, heads=2, mlp_dim=15, dropout=0.,
                       norm_eps=1e-5, rope_base=10000., rope_position_scale=1., span_beats=span)


def windows(target_beats=16, *, values=None):
    beats = torch.arange(1, target_beats + 4, 2).float()[None]
    values = torch.randn(1, beats.shape[1], 12, requires_grad=True) if values is None else values
    return WindowLatents(BeatMemory(values, beats, torch.ones_like(beats, dtype=torch.bool)),
                         torch.tensor([0.]), target_beats, torch.tensor([True]))


def old_forward(model, w):
    memory, starts = w.memory, w.target_start
    valid = (memory.valid & (memory.beats >= starts[:, None])
             & (memory.beats < starts[:, None] + w.target_beats) & w.window_valid[:, None])
    active = valid.any(-1) & w.window_valid
    query = model.query[None, None].expand(len(starts), 1, -1)
    kv = model.pool.project_memory(memory.values, memory.beats, valid)
    pooled = model.pool(query, starts[:, None], active[:, None], kv, valid)[:, 0]
    h = model.mlp(pooled)
    continuous = model.continuous(h).masked_fill(~active[:, None], 0)
    category = model.category(h).masked_fill(~active[:, None], 0)
    plan = torch.cat((continuous.float().sigmoid(), category.float().softmax(-1)), -1)
    return PlannerOutput(continuous, category, plan.masked_fill(~active[:, None], 0), active)


@pytest.mark.parametrize('target_beats', [3, 16, 24])
def test_legacy_single_pool_math_and_state_inventory_unchanged(target_beats):
    torch.manual_seed(19)
    model, segmented = head(), head(4)
    assert model.state_dict().keys() == segmented.state_dict().keys()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in segmented.parameters())
    segmented.load_state_dict(model.state_dict(), strict=True)
    w = windows(target_beats)
    actual, expected = model(w), old_forward(model, w)
    for name in actual.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert actual.plan_values.shape == (1, 12)


@pytest.mark.parametrize('target_beats,segments', [(16, 4), (24, 6)])
def test_plan4_shapes_centers_and_one_shared_kv_projection(target_beats, segments):
    model, w = head(4), windows(target_beats)
    with patch.object(model.pool, 'project_memory', wraps=model.pool.project_memory) as projected, \
         patch('maidata.draft.planner.apply_rope', wraps=__import__('maidata.nn.position', fromlist=['apply_rope']).apply_rope) as rotary:
        result = model(w)
    assert projected.call_count == 1
    torch.testing.assert_close(rotary.call_args.args[1], torch.arange(segments).float()[None] * 4 + 2)
    assert result.continuous_logits.shape == (1, segments, 8)
    assert result.config_logits.shape == (1, segments, 4)
    assert result.plan_values.shape == (1, segments, 12)
    assert result.valid.shape == (1, segments) and result.valid.all()
    torch.testing.assert_close(result.plan_values[..., 8:].sum(-1), torch.ones(1, segments))


def test_plan4_locality_and_gradients_are_confined_to_own_segment():
    torch.manual_seed(28)
    model, w = head(4), windows()
    original = model(w)
    changed_values = w.memory.values.detach().clone()
    changed_values[:, 2:4] += torch.arange(12) * 10
    changed = model(replace(w, memory=replace(w.memory, values=changed_values)))
    assert not torch.allclose(original.plan_values[:, 1], changed.plan_values[:, 1])
    torch.testing.assert_close(original.plan_values[:, [0, 2, 3]], changed.plan_values[:, [0, 2, 3]], rtol=0, atol=0)
    (original.plan_values[:, 1] * torch.arange(12)).sum().backward()
    assert w.memory.values.grad[:, 2:4].abs().sum() > 0
    assert w.memory.values.grad[:, [0, 1, 4, 5, 6, 7, 8, 9]].count_nonzero() == 0
    assert model.pool.value.weight.grad.abs().sum() > 0
    assert model.pool.query.bias.grad.abs().sum() > 0
    assert model.continuous.weight.grad.abs().sum() > 0
    assert model.category.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_plan4_matches_separate_segment_pools_in_values_and_gradients():
    torch.manual_seed(41)
    model, reference = head(4), head(4)
    reference.load_state_dict(model.state_dict())
    w = windows()
    separate = replace(w, memory=replace(w.memory, values=w.memory.values.detach().clone().requires_grad_()))
    result = model(w)
    predictions = []
    for segment in range(4):
        valid = separate.memory.valid & (separate.memory.beats >= 4 * segment) & (separate.memory.beats < 4 * (segment + 1))
        query = reference.query[None, None]
        kv = reference.pool.project_memory(separate.memory.values, separate.memory.beats, valid)
        pooled = reference.pool(query, torch.tensor([[4 * segment + 2.]]), torch.ones(1, 1, dtype=torch.bool), kv, valid)[:, 0]
        h = reference.mlp(pooled)
        predictions.append(torch.cat((reference.continuous(h).sigmoid(), reference.category(h).softmax(-1)), -1))
    expected = torch.stack(predictions, 1)
    torch.testing.assert_close(result.plan_values, expected, rtol=1e-6, atol=1e-7)
    probe = torch.randn_like(expected)
    (result.plan_values * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(w.memory.values.grad, separate.memory.values.grad, rtol=3e-5, atol=3e-7)
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-5, atol=3e-7, msg=name)


def test_plan4_missing_context_window_mask_and_nan_padding():
    model = head(4)
    beats = torch.tensor([[-2., 2., 6., 10., 14., 18.]]).expand(2, -1).clone()
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, [0, 3]] = False
    beats[0, [0, 3]] = float('nan')
    values = torch.randn(2, 6, 12).masked_fill(~valid[..., None], float('nan')).requires_grad_()
    w = WindowLatents(BeatMemory(values, beats, valid), torch.tensor([-4., -4.]), 24,
                     torch.tensor([True, False]))
    result = model(w)
    assert result.valid.tolist() == [[False, True, True, False, True, True], [False] * 6]
    for tensor in (result.continuous_logits, result.config_logits, result.plan_values):
        assert torch.isfinite(tensor).all()
        assert tensor[~result.valid].count_nonzero() == 0
    result.plan_values.sum().backward()
    assert values.grad[~valid].count_nonzero() == 0
    assert values.grad[1].count_nonzero() == 0


def chart():
    notes = (PlanNote(0, 'tap', 1), PlanNote(0, 'tap', 2), PlanNote(4 * 96, 'slide', 3),
             PlanNote(6 * 96, 'hold', 3), PlanNote(12 * 96, 'touch', None))
    return ChartData((), notes, TimingMap([(0, 120), (8, 240)]), Fraction(16))


def test_exact_plan4_targets_preserve_normalization_reference_empty_and_tail():
    planner = MusicPlanTargets(chart())
    target = planner_target(planner, 0, 16 * 96, span_beats=4)
    assert target == [planner.window(start, start + 4 * 96) for start in range(0, 16 * 96, 4 * 96)]
    assert target[0].continuous[0] == pytest.approx(log1p(2) / log1p(64))
    assert target[0].continuous[1] == pytest.approx(2 / 2 / 16)
    assert target[3].continuous[1] == pytest.approx(1 / 1 / 16)
    assert target[2].continuous == (0.,) * 8 and target[2].category == 0
    tail = planner_target(planner, 16 * 96, 32 * 96, span_beats=4)
    collated = collate_planner([target, tail, None], device='cpu')
    assert collated.continuous.shape == (3, 4, 8)
    assert collated.valid.tolist() == [[True] * 4, [True] * 4, [False] * 4]
    assert collated.continuous[1].count_nonzero() == 0
    assert planner_target(planner, 0, 16 * 96) == planner.window(0, 16 * 96)
    legacy = collate_planner([planner.window(0, 16 * 96), None], device='cpu')
    assert legacy.continuous.shape == (2, 8) and legacy.valid.tolist() == [True, False]


def test_segment_loss_prepared_denominator_mask_and_microbatch_additivity():
    torch.manual_seed(57)
    plan = PlanTarget((.2,) * 8, 2)
    target = collate_planner([[plan, plan, None, plan], [plan, None, plan, plan]], device='cpu')
    reconstruction = collate_reconstruction([WindowTarget((), (0,) * 16)] * 2, target_beats=16, device='cpu')
    prepared = prepare_supervision(reconstruction, target)
    assert prepared['planner_indices'].tolist() == [0, 1, 3, 4, 6, 7]
    assert prepared['planner_count'] == 6
    continuous = torch.randn(2, 4, 8, requires_grad=True)
    categories = torch.randn(2, 4, 4, requires_grad=True)
    valid = torch.tensor([[True, False, True, True], [True, True, True, True]])
    output = PlannerOutput(continuous, categories, torch.zeros(2, 4, 12), valid)
    regular, fast = planner_loss(output, target), planner_loss(output, target, prepared=prepared)
    mask = target.valid & output.valid
    expected_mse = (continuous[mask].sigmoid() - target.continuous[mask]).square().mean(-1).sum()
    expected_ce = F.cross_entropy(categories[mask], target.category[mask], reduction='sum')
    for name, expected in (('planner_continuous', expected_mse), ('planner_category', expected_ce)):
        assert regular[name].count == fast[name].count == 5
        torch.testing.assert_close(regular[name].total, expected)
        torch.testing.assert_close(regular[name].total, fast[name].total)
        parts = []
        for i in range(2):
            ot = PlannerOutput(continuous[i:i+1], categories[i:i+1], output.plan_values[i:i+1], valid[i:i+1])
            tt = replace(target, continuous=target.continuous[i:i+1], category=target.category[i:i+1], valid=target.valid[i:i+1])
            parts.append(planner_loss(ot, tt)[name])
        torch.testing.assert_close(sum(p.total for p in parts), regular[name].total)
        assert sum(p.count for p in parts) == regular[name].count
    regular_grad = torch.autograd.grad(sum(t.mean for t in regular.values()), (continuous, categories), retain_graph=True)
    fast_grad = torch.autograd.grad(sum(t.mean for t in fast.values()), (continuous, categories))
    for a, b in zip(regular_grad, fast_grad):
        torch.testing.assert_close(a, b)
        assert a[~mask].count_nonzero() == 0


def test_all_invalid_segment_loss_is_differentiably_zero():
    target = collate_planner([[None] * 4], device='cpu')
    continuous, categories = torch.randn(1, 4, 8, requires_grad=True), torch.randn(1, 4, 4, requires_grad=True)
    output = PlannerOutput(continuous, categories, torch.zeros(1, 4, 12), torch.zeros(1, 4, dtype=torch.bool))
    for prepared in (None, dict(planner_indices=torch.empty(0, dtype=torch.long))):
        terms = planner_loss(output, target, prepared=prepared)
        assert all(t.total == 0 and t.count == 0 for t in terms.values())
        gradients = torch.autograd.grad(sum(t.mean for t in terms.values()), (continuous, categories), retain_graph=True)
        assert all(g.count_nonzero() == 0 for g in gradients)


def test_audio_dataset_produces_four_targets_without_extra_chart_reads(monkeypatch):
    import maidata.training.data as data
    timing = dict(first_seconds=0., bpm=[[0, 1, 120.]])
    row = dict(window_id='chart:0', chart_id='chart', start_beat=0, start_tick=0, end_tick=16 * 96)
    plan = SimpleNamespace(groups={'variant':dict(chart_ids=['chart'], timing=timing, audio_end_seconds=8.)},
        charts={'chart':dict(chart_cache_path='unused')}, rows=lambda variant, epoch:[row])
    config = dict(pretrain_audio_mode='joint', pretrain_audio_cache='unused', pretrain_mel_root='unused',
        pretrain_chart_cache='unused', pretrain_source_root='unused', target_beats=16,
        audio_after_beats=4, planner_span_beats=4)
    loaded = []
    def load(*args, **kwargs):
        loaded.append(args[0])
        return dict(source=dict(song_id='1', difficulty='5'))
    monkeypatch.setattr(data, 'load_grid', lambda *a, **k:dict(first_beat=np.array(0)))
    monkeypatch.setattr(data, 'load_chart', load)
    monkeypatch.setattr(data, 'from_record', lambda *a, **k:chart())
    monkeypatch.setattr(data, 'read_condition', lambda *a, **k:object())
    monkeypatch.setattr(data, 'reconstruction_target', lambda *a, **k:object())
    item = AudioDataset(plan, config)[('variant', 1)]
    assert len(loaded) == 1
    assert item['windows'][0][4] == planner_target(MusicPlanTargets(chart()), 0, 16 * 96, span_beats=4)


@pytest.mark.parametrize('span', [0, 8, True])
def test_invalid_span_rejected(span):
    with pytest.raises(ValueError):
        head(span)


def test_plan4_rejects_partial_segment_and_mixed_collate_shapes():
    with pytest.raises(ValueError):
        head(4)(windows(6))
    with pytest.raises(ValueError):
        planner_target(MusicPlanTargets(chart()), 0, 6 * 96, span_beats=4)
    plan = PlanTarget((0.,) * 8, 0)
    with pytest.raises(ValueError):
        collate_planner([[plan] * 4, plan], device='cpu')
