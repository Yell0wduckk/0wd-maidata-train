"""Loss normalization, original scheduler parity and absolute-stop continuation."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from maidata.draft.losses import LossTerm
from maidata.entry import CONFIG
from maidata.training import pretrain


def test_weighted_logical_batch_supports_and_gradients():
    totals = torch.tensor([2., 40., 6., 8., 10., 12., 14.], requires_grad=True)
    counts = torch.tensor([2, 10, 3, 2, 5, 3, 7])
    terms = {name: LossTerm(total, count)
             for name, total, count in zip(('field/kind', 'field/route_id', 'moment96',
                 'empty', 'positive_count', 'planner_continuous', 'planner_category'), totals, counts)}
    loss = pretrain.weighted_loss(terms, dict(CONFIG, reconstruction_count_supervision=True))
    assert loss.item() == pytest.approx(15.)
    loss.backward()
    torch.testing.assert_close(totals.grad, torch.tensor([.25, .05, 1/3, .5, .2, 1/3, 1/28]))
    config = dict(CONFIG, reconstruction_moment_weight=0, planner_category_weight=2,
                  reconstruction_count_supervision=True)
    assert pretrain.weighted_loss(terms, config).item() == pytest.approx(16.5)
    empty = {name: LossTerm(totals.sum() * 0, torch.tensor(0)) for name in terms}
    assert pretrain.weighted_loss(empty, config).item() == 0


def test_optimizer_uses_only_trainable_parameters_and_configured_clip():
    parameters = [torch.nn.Parameter(torch.tensor([float(i + 1)])) for i in range(4)]
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    config = dict(CONFIG, pretrain_learning_rate=3e-4, pretrain_weight_decay=.02,
                  pretrain_gradient_clip=.5)
    optimizer = pretrain.make_optimizer(iter([*parameters, frozen]), config)
    group, = optimizer.param_groups
    assert [id(p) for p in group['params']] == [id(p) for p in parameters]
    assert group['lr'] == 3e-4 and group['weight_decay'] == .02
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
    for p in parameters:
        p.grad = torch.ones_like(p)
    assert pretrain.clip_gradients(parameters, config).item() == pytest.approx(2.)
    assert torch.stack([p.grad.square().sum() for p in parameters]).sum().sqrt().item() == pytest.approx(.5)
    optimizer.step()
    assert all(p.item() < i + 1 for i, p in enumerate(parameters))
    assert frozen.item() == 1.


def original_scheduler():
    # Read only the copied original function, without importing legacy runtime modules.
    path = Path(__file__).resolve().parents[1] / 'src/maimai_baseline/training/chart_encoder_pretrain.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_scheduler')
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['make_scheduler']


def test_scheduler_matches_original_and_does_not_shorten_to_stop():
    # Nonconstant epoch lengths check that update totals, not epoch count, set the curve.
    counts = [8 + epoch % 3 for epoch in range(100)]
    optimizer = pretrain.make_optimizer([torch.nn.Parameter(torch.ones(1))], CONFIG)
    scheduler = pretrain.make_scheduler(optimizer, counts, CONFIG)
    reference_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=2e-4)
    reference = original_scheduler()(reference_optimizer, {'warmup_steps': 200}, counts)
    for step in (0, 1, 198, 199, 200, sum(counts[:50]), sum(counts), sum(counts) + 10):
        assert scheduler.lr_lambdas[0](step) == reference.lr_lambdas[0](step)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(2e-4 / 200)
    assert scheduler.lr_lambdas[0](sum(counts[:50])) > 0
    assert scheduler.lr_lambdas[0](sum(counts)) == 0
    full_config = dict(CONFIG, pretrain_stop_after_epochs=100)
    full_scheduler = pretrain.make_scheduler(reference_optimizer, counts, full_config)
    assert full_scheduler.lr_lambdas[0](sum(counts[:50])) == scheduler.lr_lambdas[0](sum(counts[:50]))
    assert list(pretrain.epoch_range(40, CONFIG)) == list(range(41, 51))
    assert list(pretrain.epoch_range(50, CONFIG)) == []
    assert len(pretrain.epoch_range(0, CONFIG)) == 50


def test_optimizer_scheduler_state_resume_keeps_next_update_identical():
    config = dict(CONFIG, pretrain_warmup_steps=3)
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = pretrain.make_optimizer([p], config)
    scheduler = pretrain.make_scheduler(optimizer, [2] * 100, config)
    for _ in range(7):
        p.grad = torch.tensor([.5, -.25])
        optimizer.step()
        scheduler.step()
    p2 = torch.nn.Parameter(p.detach().clone())
    optimizer2 = pretrain.make_optimizer([p2], config)
    scheduler2 = pretrain.make_scheduler(optimizer2, [2] * 100, config)
    optimizer2.load_state_dict(deepcopy(optimizer.state_dict()))
    scheduler2.load_state_dict(deepcopy(scheduler.state_dict()))
    for opt, sched, param in ((optimizer, scheduler, p), (optimizer2, scheduler2, p2)):
        param.grad = torch.tensor([-.3, .7])
        opt.step()
        sched.step()
    torch.testing.assert_close(p, p2, rtol=0, atol=0)
    assert scheduler.get_last_lr() == scheduler2.get_last_lr()
    assert scheduler2.last_epoch == 8


def test_scratch_seed_and_autocast_are_explicit(monkeypatch):
    config = dict(CONFIG, pretrain_seed=93)
    samples = []
    for _ in range(2):
        pretrain.seed_scratch(config)
        samples.append((random.random(), np.random.rand(), torch.nn.Linear(3, 2).weight.detach()))
    assert samples[0][:2] == samples[1][:2]
    torch.testing.assert_close(samples[0][2], samples[1][2], rtol=0, atol=0)
    calls = []
    from contextlib import nullcontext
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: calls.append((args, kwargs)) or nullcontext())
    with pretrain.autocast('cuda:0', CONFIG):
        pass
    assert calls == [(('cuda',), {'dtype': torch.bfloat16})]
    with pretrain.autocast('cpu', CONFIG), pretrain.autocast('cuda', dict(CONFIG, pretrain_precision='fp32')):
        pass
    assert calls[-1] == (('cpu',), {'dtype': torch.bfloat16})
    assert len(calls) == 2
