"""Raw per-beat count MSE, common supports, diagnostics and live joint graph."""
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch

from maidata.draft.losses import reconstruction_loss, prepare_supervision
from maidata.entry import CONFIG
from maidata.training.runner import Diagnostics
from maidata.training.pretrain import weighted_loss
from maidata.joint_training.network import JointTrainer
from maidata.joint_training.data import collate_audio
from maidata.joint_training import update as updates
from test_joint_gradients import joint_config, optimizers, direct_update, budget_for
from test_decoder_training import config, distances
from test_joint_data import joint_item


@pytest.mark.parametrize('prepared', [False, True])
def test_raw_count_mse_zeros_padding_and_distance_gradient(prepared):
    values = torch.tensor([[1., 1., -1.], [float('nan')]*3], requires_grad=True)
    target = SimpleNamespace(fields={}, moment_tick=torch.full((2, 1), -100),
        moment_counts=torch.tensor([[0, 3, 0], [999, 999, 999]]),
        valid=torch.tensor([True, False]), queries=SimpleNamespace(valid=torch.zeros(2, 1, dtype=torch.bool)))
    output = SimpleNamespace(field_logits={}, moment_logits=torch.zeros(2, 1, 96, requires_grad=True),
        count_values=values, query_valid=target.queries.valid,
        count_valid=torch.tensor([[True]*3, [False]*3]))
    selector = prepare_supervision(target, SimpleNamespace(valid=target.valid)) if prepared else None
    terms = reconstruction_loss(output, target, label_smoothing=.05, prepared=selector)
    assert set(terms) == {'moment96', 'count'}
    assert terms['count'].count == 3
    assert terms['count'].mean.item() == 2.
    loss = weighted_loss(terms, CONFIG, denominators={'moment96': 0, 'count': 6})
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[1/3, -2/3, -1/3], [0., 0., 0.]]))
    diagnostics = Diagnostics()
    diagnostics.add(output, target)
    report = diagnostics.report()['count_regression']
    assert report['mse'] == 2. and report['mae'] == pytest.approx(4/3)
    assert report['rounded_accuracy'] == pytest.approx(1/3)
    assert report['true_distribution'] == {0: 2, 3: 1}
    assert report['predicted_distribution'] == {0: 1, 1: 2}


@pytest.mark.parametrize('chunks', [1, 3])
def test_mse_count_joint_update_matches_direct_graph_and_updates_five_modules(joint_config, monkeypatch, chunks):
    c = dict(joint_config, reconstruction_count_loss='mse')
    item = joint_item(0, c, monkeypatch)
    c['decoder_train_chunk_token_budget'] = budget_for(item['m_windows'], chunks)
    batch = collate_audio([item], config=c, packed=True)
    model = JointTrainer(c)
    assert hasattr(model.reconstruction, 'count_head')
    assert not hasattr(model.reconstruction, 'empty_head')
    reference = deepcopy(model)
    opt, sched = optimizers(model)
    ref_opt, ref_sched = optimizers(reference)
    actual, audit = updates.update(model, [batch], opt, sched, c, distances(), audit=True)
    direct_batch = collate_audio([item], config=dict(c, decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference, [direct_batch], ref_opt, ref_sched, c)
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=3e-4)
    report = actual.report(c)
    assert report['loss'] == pytest.approx(expected.report(c)['loss'], rel=1e-6)
    parts = report['losses']
    assert parts['drafter_count'] == report['auxiliary']['terms']['count']['mean']
    assert parts['drafter'] == pytest.approx(sum(parts[k] for k in
        ('drafter_fields', 'drafter_moment', 'drafter_count', 'planner')))
