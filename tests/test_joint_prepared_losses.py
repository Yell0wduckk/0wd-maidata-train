"""Prepared joint selectors and packed condition kernels match independent paths."""
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from test_draft_local_queries import chart
from test_pretrain_runtime import config
from maidata.chart.types import Note, Route, SlideDuration
from maidata.decoder.attention import Attention
from maidata.decoder.packed_attention import condition_attention
from maidata.decoder.row_layout import window_layout
from maidata.decoder.types import Memory
from maidata.draft.losses import prepare_supervision
from maidata.draft.layout import prepare_selection
from maidata.draft.plan_targets import PlanTarget
from maidata.draft.supervision import predict_auxiliary, auxiliary_losses
from maidata.draft.targets import reconstruction_target, collate_reconstruction, collate_planner
from maidata.draft.types import BeatMemory
from maidata.model import build_reconstruction, build_planner


def compare_gradients(actual, expected, parameters, bf16):
    first = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
    for a, b in zip(first, second):
        if a is None or b is None:
            assert a is b or (a if a is not None else b).count_nonzero() == 0
        else:
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, atol=.03 if bf16 else 2e-5,
                                       rtol=.05 if bf16 else 3e-4)


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('count_loss', ['ce', 'mse'])
def test_prepared_auxiliary_sparse_empty_padding_values_gradients_and_no_nonzero(config, empty, bf16, count_loss):
    c = dict(config, target_beats=16, draft_z_stride_beats=4, reconstruction_count_loss=count_loss,
             reconstruction_count_supervision=True)
    source = chart() if empty else chart([(0, [Note.make('tap', 'A1')]),
        (95, [Note.make('slide', 'A2'), Route(0, 1), Route(3), SlideDuration(97)]),
        (4*96, [Note.make('touch', 'C')])])
    target = collate_reconstruction([reconstruction_target(source, start_tick=0, target_beats=16),
        reconstruction_target(chart(), start_tick=0, target_beats=16), None],
        target_beats=16, device='cpu')
    planner_target = collate_planner([PlanTarget((.1,)*8, 2), PlanTarget((0.,)*8, 0), None], device='cpu')
    # Dummy windows and unused query padding can contain arbitrary sentinels.
    poison = {}
    for name in target.queries.__dataclass_fields__:
        value = getattr(target.queries, name).clone()
        if name != 'valid':
            invalid = ~target.queries.valid
            if value.ndim == 3:
                invalid = invalid[..., None].expand_as(value)
            value[invalid] = (float('nan') if value.is_floating_point()
                              else True if value.dtype == torch.bool else 999)
        poison[name] = value
    target = replace(target, queries=replace(target.queries, **poison))
    planner_target = replace(planner_target,
        continuous=planner_target.continuous.masked_fill(~planner_target.valid[:, None], float('nan')),
        category=planner_target.category.masked_fill(~planner_target.valid, 999))
    groups, starts, valid = torch.arange(3), torch.zeros(3), target.valid
    z_valid = torch.ones(3, 8, dtype=torch.bool)
    z_valid[-1] = False
    values = torch.randn(3, 8, c['draft_latent_dim']).masked_fill(~z_valid[..., None], float('nan')).requires_grad_()
    z = BeatMemory(values, torch.arange(8)[None].expand(3, -1).float()*4+2, z_valid)
    layouts = {name: prepare_selection(z.beats, z.valid, groups, starts, valid,
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=4)
        for name in ('reconstruction', 'planner')}
    prepared = prepare_supervision(target, planner_target)
    r, p = build_reconstruction(c), build_planner(c)
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        reference = predict_auxiliary(z, groups, starts, valid, r, p, c,
            layouts=layouts, queries=target.queries)
        reference_terms = auxiliary_losses(*reference, target, planner_target, c)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = predict_auxiliary(z, groups, starts, valid, r, p, c,
                layouts=dict(layouts, auxiliary=prepared), queries=target.queries)
            actual_terms = auxiliary_losses(*actual, target, planner_target, c, prepared=prepared)
    assert 'aten::nonzero' not in {event.key for event in profile.key_averages()}
    for name in actual_terms:
        a, b = actual_terms[name], reference_terms[name]
        torch.testing.assert_close(a.total, b.total, rtol=0, atol=0)
        torch.testing.assert_close(a.count, b.count, rtol=0, atol=0)
    assert actual_terms['count' if count_loss == 'mse' else 'empty'].count == 32
    if empty:
        assert actual_terms['moment96'].count == 0
        if count_loss == 'ce':
            assert actual_terms['positive_count'].count == 0
        assert all(term.count == 0 for name, term in actual_terms.items() if name.startswith('field/'))
    for a, b in zip(actual, reference):
        for name in a.__dataclass_fields__:
            one, two = getattr(a, name), getattr(b, name)
            if one is None or two is None:
                assert one is two
            elif isinstance(one, dict):
                for field in one:
                    torch.testing.assert_close(one[field], two[field], rtol=0, atol=0)
            else:
                torch.testing.assert_close(one, two, rtol=0, atol=0)
    compare_gradients(sum(term.mean for term in actual_terms.values()),
        sum(term.mean for term in reference_terms.values()), (*r.parameters(), *p.parameters(), values), bf16)


@pytest.mark.parametrize('bf16', [False, True])
def test_packed_condition_zero_one_multiple_masked_keys_and_no_nonzero(bf16):
    torch.set_num_threads(1)
    torch.manual_seed(72)
    module = Attention(24, 24, 3, 0., 10000., 1.)
    valid = torch.tensor([[True, True, True, True, True], [True, False, True, False, False],
                          [True, True, False, False, False], [True, True, True, False, True]])
    key_valid = torch.tensor([[False]*5, [False, True, False, False, False],
                              [True, False, True, False, False], [True, True, False, True, True]])
    x = torch.randn(4, 5, 24).masked_fill(~valid[..., None], float('nan')).requires_grad_()
    beats = (torch.arange(5)[None].expand(4, -1).float()-2).masked_fill(~valid, float('nan'))
    values = torch.randn(4, 5, 24).masked_fill(~key_valid[..., None], float('nan')).requires_grad_()
    memory = Memory(values, torch.arange(5)[None].expand(4, -1).float(), key_valid)
    layout = window_layout(valid, key_valid)
    flat_x = x.flatten(0, 1).index_select(0, layout['indices'])
    flat_beats = beats.flatten().index_select(0, layout['indices'])
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        kv = module.project(memory, plan_index=4)
        reference = module(x, beats, valid, kv, plan_index=4)
        expected = reference.flatten(0, 1).index_select(0, layout['indices'])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = condition_attention(module, flat_x, flat_beats, valid, kv, 4, layout)
    assert 'aten::nonzero' not in {event.key for event in profile.key_averages()}
    torch.testing.assert_close(actual, expected, atol=.004 if bf16 else 2e-6,
                               rtol=.03 if bf16 else 2e-5)
    weight = torch.randn_like(actual)
    compare_gradients((actual*weight).sum(), (expected*weight).sum(),
                      (*module.parameters(), x, values), bf16)
    assert actual[:5].count_nonzero() == 0
