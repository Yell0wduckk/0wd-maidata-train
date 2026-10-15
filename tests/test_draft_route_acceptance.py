"""Directed acceptance algebra, masked readouts and logical-update gradients."""
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from test_draft_route37 import example
from maidata.draft.losses import reconstruction_loss
from maidata.draft.route_acceptance import ACCEPTANCE37, acceptance_nll, route_loss_recipe
from maidata.draft.route_vocabulary import AXIS15_GROUPS, route_targets
from maidata.draft.supervision import auxiliary_losses
from maidata.draft.types import PlannerTargets
from maidata.entry import CONFIG
from maidata.model import build_planner
from maidata.training.pretrain import weighted_loss


def selectors(target):
    return dict(indices={
        **{'field/' + name: (target.valid[:, None] & target.queries.valid & (labels != -100))
           .flatten().nonzero().flatten() for name, labels in target.fields.items()},
        'moment96': (target.valid[:, None] & target.queries.valid & (target.moment_tick != -100))
        .flatten().nonzero().flatten()})


def test_fixed_table_preserves_directed_credit_and_wifi_row():
    a = torch.tensor(ACCEPTANCE37)
    assert a.shape == (len(AXIS15_GROUPS), len(AXIS15_GROUPS)) == (37, 37)
    assert torch.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    torch.testing.assert_close(a.diag(), torch.ones(37), rtol=0, atol=0)
    assert a[0, 1].item() == pytest.approx(.65)
    assert a[1, 0].item() == .5
    assert a[0].sum() == pytest.approx(3.5)  # No row normalization.
    torch.testing.assert_close(a[36], torch.eye(37)[36], rtol=0, atol=0)
    assert a[2, 36] == .5 and a[36, 2] == 0


def test_identity_acceptance_is_unsmoothed_nll():
    torch.manual_seed(715)
    logits = torch.randn(7, 37, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 15, 27, 28, 36])
    actual = acceptance_nll(logits, labels, matrix=torch.eye(37))
    expected = F.cross_entropy(logits, labels, reduction='none')
    torch.testing.assert_close(actual, expected)
    a, = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
    b, = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(a, b)


def test_partial_credit_matches_hand_calculation_and_direction():
    logits = torch.tensor([[.2, .8], [.2, .8]]).log()
    actual = acceptance_nll(logits, torch.tensor([0, 1]), matrix=((1., .5), (.3, 1.)))
    expected = torch.tensor([-math.log(.2 + .5*.8), -math.log(.3*.2 + .8)])
    torch.testing.assert_close(actual, expected)
    # Nearly all probability on the alternate: retain the absolute .65/.5 credit.
    concentrated = torch.full((2, 37), -1000.)
    concentrated[0, 1], concentrated[1, 0] = 0., 0.
    actual = acceptance_nll(concentrated, torch.tensor([0, 1]))
    torch.testing.assert_close(actual, torch.tensor([-math.log(.65), -math.log(.5)]))


def test_fully_accepted_row_does_not_force_probability_splitting():
    logits = torch.tensor([[8., -2., 1.]], requires_grad=True)
    loss = acceptance_nll(logits, torch.tensor([0]), matrix=torch.ones(3, 3)).sum()
    assert abs(float(loss.detach())) < 1e-6
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_extreme_logits_and_wifi_nll_remain_finite_in_fp32(dtype):
    logits = torch.full((2, 37), -20000., dtype=dtype)
    logits[0, 0], logits[1, 36] = 20000., 20000.
    logits.requires_grad_()
    labels = torch.tensor([36, 2])
    loss = acceptance_nll(logits, labels)
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all()
    torch.testing.assert_close(loss[0], F.cross_entropy(logits[:1].float(), labels[:1]))
    torch.testing.assert_close(loss[1], torch.tensor(-math.log(.5)))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
def test_real_route_readout_acceptance_other_fields_and_gradients(prepared, bf16):
    head, target, memory = example()
    raw = target.fields['route_id'].clone()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        output = head(memory, target.queries)
    select = selectors(target) if prepared else None
    terms = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                route_loss='acceptance37', prepared=select)
    hard = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                               route_loss='ce', prepared=select)
    active = target.valid[:, None] & target.queries.valid & output.query_valid & (raw != -100)
    labels = route_targets(raw, 'axis15-37')[active]
    logits = output.field_logits['route_id'][active].float()
    # Independent probability-space oracle is safe for these modest logits.
    expected = -(torch.tensor(ACCEPTANCE37)[labels] * logits.softmax(-1)).sum(-1).log().sum()
    torch.testing.assert_close(terms['field/route_id'].total, expected)
    assert terms['field/route_id'].count == 71
    assert torch.equal(raw, target.fields['route_id'])
    assert target.queries.current_route_ids.max() == 70
    for name in terms:
        if name != 'field/route_id':
            torch.testing.assert_close(terms[name].total, hard[name].total, rtol=0, atol=0)
            torch.testing.assert_close(terms[name].count, hard[name].count, rtol=0, atol=0)
    terms['field/route_id'].total.backward()
    for gradient in (head.field_heads['field_route_id'].weight.grad, memory.memory.values.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_route_smoothing_is_removed_only_for_acceptance_mode():
    head, target, memory = example()
    output = head(memory, target.queries)
    first = reconstruction_loss(output, target, label_smoothing=0., count_supervision=False,
                                route_loss='acceptance37')
    second = reconstruction_loss(output, target, label_smoothing=.7, count_supervision=False,
                                 route_loss='acceptance37')
    torch.testing.assert_close(first['field/route_id'].total, second['field/route_id'].total,
                               rtol=0, atol=0)
    assert not torch.isclose(first['field/kind'].total, second['field/kind'].total)


@pytest.mark.parametrize('prepared', [False, True])
def test_no_route_and_ignored_or_invalid_queries_have_zero_support(prepared):
    head, target, memory = example()
    output = head(memory, target.queries)
    masked = replace(target, fields={**target.fields, 'route_id':
                                    torch.full_like(target.fields['route_id'], -100)})
    terms = reconstruction_loss(output, masked, label_smoothing=.05, count_supervision=False,
                                route_loss='acceptance37', prepared=selectors(masked) if prepared else None)
    route = terms['field/route_id']
    assert route.count == 0 and route.total == 0 and route.total.requires_grad
    route.total.backward()
    assert head.field_heads['field_route_id'].weight.grad.count_nonzero() == 0
    if not prepared:
        invalid = replace(target, valid=torch.zeros_like(target.valid))
        terms = reconstruction_loss(output, invalid, label_smoothing=.05, count_supervision=False,
                                    route_loss='acceptance37')
        assert terms['field/route_id'].count == 0 and terms['field/route_id'].total == 0


def test_prepared_and_plain_route_values_gradients_and_no_nonzero():
    head, target, memory = example()
    output = head(memory, target.queries)
    selected = selectors(target)
    plain = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                route_loss='acceptance37')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        prepared = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                       route_loss='acceptance37', prepared=selected)
    assert 'aten::nonzero' not in {event.key for event in profile.key_averages()}
    parameters = (*head.parameters(), memory.memory.values)
    for name in plain:
        torch.testing.assert_close(plain[name].total, prepared[name].total, rtol=0, atol=0)
        torch.testing.assert_close(plain[name].count, prepared[name].count, rtol=0, atol=0)
    a = torch.autograd.grad(sum(t.mean for t in plain.values()), parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(sum(t.mean for t in prepared.values()), parameters, allow_unused=True)
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_logical_update_split_keeps_denominators_and_gradients():
    head, target, memory = example()
    output = head(memory, target.queries)
    whole_select = selectors(target)
    whole = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                route_loss='acceptance37', prepared=whole_select)
    chunks = []
    for part in (0, 1):
        select = dict(indices={name: rows[:len(rows)//3] if part == 0 else rows[len(rows)//3:]
                               for name, rows in whole_select['indices'].items()})
        chunks.append(reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                          route_loss='acceptance37', prepared=select))
    counts = {name: int(t.count) for name, t in whole.items()}
    expected = weighted_loss(whole, CONFIG)
    actual = sum(weighted_loss(chunk, CONFIG, denominators=counts) for chunk in chunks)
    torch.testing.assert_close(actual, expected)
    parameters = (*head.parameters(), memory.memory.values)
    a = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(expected, parameters, allow_unused=True)
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)


def test_shared_auxiliary_entry_uses_acceptance_and_missing_mode_keeps_ce():
    head, target, memory = example()
    config = dict(CONFIG, draft_latent_dim=8, planner_dim=8, planner_heads=2,
                  planner_mlp_dim=12, planner_dropout=0., reconstruction_count_supervision=False)
    planner = build_planner(config)(memory)
    plan_target = PlannerTargets(torch.zeros(1, 8), torch.zeros(1, dtype=torch.long),
                                torch.ones(1, dtype=torch.bool))
    output = head(memory, target.queries)
    actual = auxiliary_losses(output, planner, target, plan_target, config)
    expected = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                   route_loss='acceptance37')
    torch.testing.assert_close(actual['field/route_id'].total, expected['field/route_id'].total)
    legacy = {key: value for key, value in config.items() if key != 'reconstruction_route_loss'}
    actual = auxiliary_losses(output, planner, target, plan_target, legacy)
    expected = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False)
    torch.testing.assert_close(actual['field/route_id'].total, expected['field/route_id'].total)


def test_acceptance_requires_37_classes_and_unknown_mode_is_rejected():
    head, target, memory = example('route71')
    output = head(memory, target.queries)
    with pytest.raises(ValueError, match='acceptance37 requires'):
        reconstruction_loss(output, target, label_smoothing=.05, route_loss='acceptance37')
    with pytest.raises(ValueError, match='must be ce or acceptance37'):
        route_loss_recipe(dict(CONFIG, reconstruction_route_loss='unknown'))
