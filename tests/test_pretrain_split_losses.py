"""Independent field means, sparse logical updates and split count diagnostics."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from maidata.draft.losses import reconstruction_loss, LossTerm
from maidata.entry import CONFIG
from maidata.training.pretrain import weighted_loss
from maidata.training.runner import Metrics, Diagnostics


def example(*, empty=False):
    torch.manual_seed(34)
    n = 101
    kind = torch.full((1, n), -100, dtype=torch.long)
    route = kind.clone()
    moment = kind.clone()
    if not empty:
        kind[0, :100] = torch.arange(100) % 5
        route[0, 100] = 70
        moment[0, 0] = 95
    target = SimpleNamespace(fields={'kind': kind, 'route_id': route}, moment_tick=moment,
        moment_counts=torch.tensor([[0, 0 if empty else 2, 0]]), valid=torch.tensor([True]),
        queries=SimpleNamespace(valid=torch.ones(1, n, dtype=torch.bool)))
    logits = lambda size: torch.randn(1, n, size, requires_grad=True)
    output = SimpleNamespace(field_logits={'kind': logits(5), 'route_id': logits(71)},
        moment_logits=logits(96), empty_logits=torch.randn(1, 3, 2, requires_grad=True),
        positive_count_logits=torch.randn(1, 3, 96, requires_grad=True),
        query_valid=target.queries.valid, count_valid=torch.ones(1, 3, dtype=torch.bool))
    return output, target


def test_imbalanced_fields_use_independent_means_and_moment_unsmoothed():
    output, target = example()
    terms = reconstruction_loss(output, target, label_smoothing=.05)
    kind = F.cross_entropy(output.field_logits['kind'][0, :100], target.fields['kind'][0, :100], label_smoothing=.05)
    route = F.cross_entropy(output.field_logits['route_id'][0, 100:101], target.fields['route_id'][0, 100:101], label_smoothing=.05)
    moment = F.cross_entropy(output.moment_logits[0, :1], torch.tensor([95]))
    empty = F.cross_entropy(output.empty_logits[0], torch.tensor([0, 1, 0]), label_smoothing=.05)
    positive = F.cross_entropy(output.positive_count_logits[0, 1:2], torch.tensor([1]), label_smoothing=.05)
    expected = (kind + route)/2 + moment + empty + positive
    actual = weighted_loss(terms, CONFIG)
    torch.testing.assert_close(actual, expected)
    actual.backward(retain_graph=True)
    actual_grad = output.field_logits['route_id'].grad.clone()
    output.field_logits['route_id'].grad = None
    expected.backward()
    torch.testing.assert_close(actual_grad, output.field_logits['route_id'].grad)
    assert terms['field/kind'].count == 100 and terms['field/route_id'].count == 1
    assert terms['moment96'].count == 1 and terms['positive_count'].count == 1


def test_sparse_microbatches_share_logical_field_denominators_and_active_set():
    values = torch.tensor([9., 12., 21., 14.], requires_grad=True)
    zero = values.sum() * 0
    a = {'field/kind': LossTerm(values[0], torch.tensor(3)),
         'field/route_id': LossTerm(zero, torch.tensor(0))}
    b = {'field/kind': LossTerm(values[1], torch.tensor(4)),
         'field/route_id': LossTerm(values[2], torch.tensor(2))}
    c = {'field/kind': LossTerm(zero, torch.tensor(0)),
         'field/route_id': LossTerm(values[3], torch.tensor(2))}
    supports = {'field/kind': 7, 'field/route_id': 4}
    actual = sum(weighted_loss(part, CONFIG, denominators=supports) for part in (a, b, c))
    expected = ((values[0]+values[1])/7 + (values[2]+values[3])/4)/2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(values.grad, torch.tensor([1/14, 1/14, 1/8, 1/8]))


def test_real_ce_gradients_match_across_sparse_microbatch_partitions():
    torch.manual_seed(9)
    labels = torch.full((3, 6), -100, dtype=torch.long)
    kind, route, moment = (labels.clone() for _ in range(3))
    kind[0, :3], kind[1, 3] = torch.tensor([0, 1, 2]), 3
    route[1, 4], route[2, 5] = 70, 12
    moment[0, 3], moment[1, :2] = 0, torch.tensor([0, 95])
    target = SimpleNamespace(fields={'kind': kind, 'route_id': route}, moment_tick=moment,
        moment_counts=torch.tensor([[1, 0], [0, 2], [0, 0]]), valid=torch.ones(3, dtype=torch.bool),
        queries=SimpleNamespace(valid=torch.ones(3, 6, dtype=torch.bool)))
    reference = [torch.randn(*shape) for shape in
                 ((3, 6, 5), (3, 6, 71), (3, 6, 96), (3, 2, 2), (3, 2, 96))]

    def objective(partition):
        parameters = [value.clone().requires_grad_() for value in reference]
        output = SimpleNamespace(field_logits=dict(zip(('kind', 'route_id'), parameters[:2])),
            moment_logits=parameters[2], empty_logits=parameters[3], positive_count_logits=parameters[4],
            query_valid=target.queries.valid, count_valid=torch.ones(3, 2, dtype=torch.bool))
        full = reconstruction_loss(output, target, label_smoothing=.05)
        supports = {name: int(term.count) for name, term in full.items()}
        loss = parameters[0].sum() * 0
        metrics = Metrics()
        for rows in partition:
            local_output = SimpleNamespace(field_logits={k: v[rows] for k,v in output.field_logits.items()},
                moment_logits=output.moment_logits[rows], empty_logits=output.empty_logits[rows],
                positive_count_logits=output.positive_count_logits[rows], query_valid=output.query_valid[rows],
                count_valid=output.count_valid[rows])
            local_target = SimpleNamespace(fields={k: v[rows] for k,v in target.fields.items()},
                moment_tick=target.moment_tick[rows], moment_counts=target.moment_counts[rows], valid=target.valid[rows],
                queries=SimpleNamespace(valid=target.queries.valid[rows]))
            terms = reconstruction_loss(local_output, local_target, label_smoothing=.05)
            loss = loss + weighted_loss(terms, CONFIG, denominators=supports)
            metrics.add(terms)
        return loss, torch.autograd.grad(loss, parameters), metrics.report(CONFIG)

    expected, expected_gradients, expected_metrics = objective([slice(None)])
    for partition in ([slice(0, 1), slice(1, 3)], [slice(0, 1), slice(1, 2), slice(2, 3)]):
        actual, gradients, metrics = objective(partition)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        for a,b in zip(gradients, expected_gradients):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
        assert metrics['loss'] == pytest.approx(expected_metrics['loss'], rel=2e-6)
        for name in expected_metrics['terms']:
            assert metrics['terms'][name]['count'] == expected_metrics['terms'][name]['count']


def test_empty_batch_has_only_empty_supervision_and_no_fake_rows():
    output, target = example(empty=True)
    terms = reconstruction_loss(output, target, label_smoothing=.05)
    assert {name: int(t.count) for name,t in terms.items()} == {
        'field/kind': 0, 'field/route_id': 0, 'moment96': 0, 'empty': 3, 'positive_count': 0}
    expected = F.cross_entropy(output.empty_logits[0], torch.zeros(3, dtype=torch.long), label_smoothing=.05)
    torch.testing.assert_close(weighted_loss(terms, CONFIG), expected)
    weighted_loss(terms, CONFIG).backward()
    assert torch.count_nonzero(output.moment_logits.grad) == 0
    assert torch.count_nonzero(output.positive_count_logits.grad) == 0


def test_epoch_metrics_retain_independent_supports():
    metrics = Metrics()
    metrics.add({'field/kind': LossTerm(torch.tensor(300.), torch.tensor(100)),
                 'field/route_id': LossTerm(torch.tensor(1.), torch.tensor(1))})
    metrics.add({'field/kind': LossTerm(torch.tensor(3.), torch.tensor(1)),
                 'field/route_id': LossTerm(torch.tensor(4.), torch.tensor(1))})
    report = metrics.report(CONFIG)
    assert report['classification'] == {'mean': 2.75, 'active_fields': 2}
    assert report['terms']['field/kind']['count'] == 101
    assert report['terms']['field/route_id']['count'] == 2
    assert report['loss'] == 2.75


def test_final_diagnostics_distinguish_conditional_and_composed_count():
    output, target = example()
    with torch.no_grad():
        output.empty_logits.zero_()  # Always predicts empty, including the nonempty beat.
        output.positive_count_logits.zero_()
        output.positive_count_logits[..., 1] = 10
        output.moment_logits.zero_()
        output.moment_logits[..., 95] = 10
    diagnostics = Diagnostics()
    diagnostics.add(output, target)
    report = diagnostics.report()
    assert report['positive_count_gt_nonempty'] == {'correct': 1, 'count': 1, 'accuracy': 1.}
    assert report['composed_count'] == {'correct': 2, 'count': 3, 'accuracy': 2/3}
    assert report['moment96_conditional']['true_distribution'][95] == 1
    assert report['moment96_conditional']['predicted_distribution'][95] == 1
    assert report['fields']['route_id']['rare_class_rule'] == 'least-supported observed classes'
