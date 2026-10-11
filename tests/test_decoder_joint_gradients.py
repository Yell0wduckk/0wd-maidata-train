"""Independent direct-graph reference for fresh Stage-II Encoder/Decoder updates."""
from contextlib import ExitStack
from copy import deepcopy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maidata.decoder.types import Memory
from maidata.decoder_training import checkpoint, recipe, runner
from maidata.decoder_training.data import collate_audio
from maidata.decoder_training.network import DecoderTrainer
from maidata.decoder_training.supervision import KEYS, objective
from test_decoder_training import config, encoder_config, training_item, distances


def direct_update(model, batches, opt, sched, config):
    """No gradient bridge: retain every real graph and backpropagate once."""
    model.train()
    opt.zero_grad(set_to_none=True)
    counts = {k: sum(chunk.labels.counts[k] for b in batches for chunk in b.chunks)
              for k in KEYS}
    values, metrics = [], runner.Metrics()
    for raw in batches:
        batch = raw.to('cpu')
        conditions = model.conditions(batch)
        b = conditions.audio
        audio_kv = model.decoder.prepare_audio(Memory(b.values, b.beats, b.valid),
                                               indices=conditions.audio_indices)
        for cpu_chunk in batch.chunks:
            terms, _ = runner.chunk_loss(model, conditions, audio_kv, cpu_chunk.to('cpu'),
                                         config, distances())
            values.append(objective(terms, counts, config))
            metrics.add(terms)
    torch.stack(values).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['decoder_train_gradient_clip'])
    opt.step()
    sched.step()
    return metrics


def assert_trees_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_trees_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_trees_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize('accumulate', [False, True])
def test_single_chunk_uses_real_audio_graph_without_bridge_or_gradient_copies(config, monkeypatch, accumulate):
    c = dict(config, decoder_train_chunk_token_budget=10000,
             decoder_train_gradient_accumulation_steps=2 if accumulate else 1)
    rows = [training_item(0, c, monkeypatch, remaster=True),
            training_item(1, c, monkeypatch)]
    batches = ([collate_audio([row], config=c, packed=True) for row in rows] if accumulate else
               [collate_audio(rows, config=c, packed=True)])
    assert all(len(b.chunks) == 1 for b in batches)
    model = DecoderTrainer(c)
    reference = deepcopy(model)
    opts = [recipe.optimizer(m.parameters(), c) for m in (model, reference)]
    scheds = [recipe.scheduler(opt, [1]*6, c) for opt in opts]
    backward_calls = []
    hook = next(model.encoder.parameters()).register_hook(lambda grad: backward_calls.append(1))
    # The bridge owns all K/V-gradient cloning/conversion. A single chunk must
    # not construct it, including when several physical batches accumulate.
    with patch.object(runner, 'SharedAudioGradients', side_effect=AssertionError('single chunk needs no bridge')), \
         patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as encode, \
         patch.object(model.encoder, 'encode_windows', wraps=model.encoder.encode_windows) as local_encode, \
         patch.object(model.decoder, 'prepare_audio', wraps=model.decoder.prepare_audio) as project:
        actual, audit = runner.update(model, batches, opts[0], scheds[0], c, distances())
    hook.remove()
    assert encode.call_count == local_encode.call_count == project.call_count == len(batches)
    assert len(backward_calls) == len(batches)
    assert audit['chunks'] == len(batches)
    assert audit['padded_token_costs'] == [b.chunks[0].cost for b in batches]
    assert audit['batch_costs'] == [b.budget_cost for b in batches]
    expected = direct_update(reference, batches, opts[1], scheds[1], c)
    assert actual.counts == expected.counts
    assert actual.report(c)['loss'] == pytest.approx(expected.report(c)['loss'], rel=2e-6)
    for (name, a), (other_name, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(a, b, atol=2e-5, rtol=4e-4, msg=name)
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=4e-4, msg=name+' gradient')
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


@pytest.mark.parametrize('accumulate', [False, True])
def test_gradient_bridge_matches_direct_graph_and_updates_encoder_once_per_batch(config, monkeypatch, accumulate):
    rows = [training_item(0, config, monkeypatch, remaster=True),
            training_item(1, config, monkeypatch)]
    c = dict(config, decoder_train_chunk_token_budget=12,
             decoder_train_gradient_accumulation_steps=2 if accumulate else 1)
    batches = ([collate_audio([row], config=c, packed=True) for row in rows] if accumulate else
               [collate_audio(rows, config=c, packed=True)])
    unchunked = dict(c, decoder_train_chunk_token_budget=10000)
    reference_batches = ([collate_audio([row], config=unchunked, packed=True) for row in rows] if accumulate else
                         [collate_audio(rows, config=unchunked, packed=True)])
    assert all(len(b.chunks) > 1 for b in batches)
    assert all(len(b.chunks) == 1 for b in reference_batches)
    with ExitStack() as stack:
        for name in ('build_draft', 'build_planner', 'build_reconstruction'):
            stack.enter_context(patch('maidata.model.'+name,
                                      side_effect=AssertionError('Stage II cannot construct D/P/R')))
        model = DecoderTrainer(c)
        reference = deepcopy(model)
        assert set(dict(model.named_children())) == {'encoder', 'decoder'}
        opts = [recipe.optimizer(m.parameters(), c) for m in (model, reference)]
        scheds = [recipe.scheduler(opt, [1]*6, c) for opt in opts]
        backward_calls = []
        hook = next(model.encoder.parameters()).register_hook(lambda grad: backward_calls.append(1))
        with patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as encode, \
             patch.object(model.encoder, 'encode_windows', wraps=model.encoder.encode_windows) as local_encode, \
             patch.object(runner, 'SharedAudioGradients', wraps=runner.SharedAudioGradients) as bridge:
            actual, audit = runner.update(model, batches, opts[0], scheds[0], c, distances(), audit=True)
        hook.remove()
        assert encode.call_count == local_encode.call_count == bridge.call_count == len(batches)
        assert len(backward_calls) == len(batches)
        expected = direct_update(reference, reference_batches, opts[1], scheds[1], c)
    assert actual.counts == expected.counts
    assert actual.report(c)['loss'] == pytest.approx(expected.report(c)['loss'], rel=2e-6)
    assert set(audit['changed_parameter_tensors']) == {'encoder', 'decoder'}
    for name in ('encoder', 'decoder'):
        assert audit['gradients'][name] > 0
        assert audit['changed_parameter_tensors'][name] > 0
    for (name, a), (other_name, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(a, b, atol=2e-5, rtol=4e-4, msg=name)
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=4e-4, msg=name+' gradient')
    for raw in batches:
        conditions = raw.to('cpu').conditions
        assert not conditions.decoder_z.values.requires_grad
        assert not conditions.plan.requires_grad


def test_joint_checkpoint_restores_both_networks_optimizer_rng_and_rejects_m_only_v1(config, monkeypatch):
    c = dict(config, dropout=.12, decoder_dropout=.17, decoder_train_chunk_token_budget=12)
    batch = collate_audio([training_item(0, c, monkeypatch, remaster=True)], config=c, packed=True)
    model = DecoderTrainer(c)
    opt = recipe.optimizer(model.parameters(), c)
    sched = recipe.scheduler(opt, [1]*6, c)
    runner.update(model, [batch], opt, sched, c, distances())
    loaders = [DataLoader([0], generator=torch.Generator().manual_seed(31))]
    identity = dict(config=c, binding={}, source={}, producer={'teacher': 'fixed'},
                    steps=[1]*6, loaders=loaders)
    payload = deepcopy(checkpoint.capture(model, opt, sched, epoch=1, step=1,
        best=1., metrics={}, **identity))
    assert payload['schema'] == 'maidata.decoder-training.v4'
    assert payload['architecture']=='maidata.decoder-causal-cont-tick384.v2'
    assert any(k.startswith('encoder.') for k in payload['model_state'])
    assert any(k.startswith('decoder.') for k in payload['model_state'])
    runner.update(model, [batch], opt, sched, c, distances())
    expected_rng = checkpoint.rng_state()
    expected_draws = (random.random(), np.random.rand(), torch.rand(4))
    resumed = DecoderTrainer(c)
    resumed_opt = recipe.optimizer(resumed.parameters(), c)
    resumed_sched = recipe.scheduler(resumed_opt, [1]*6, c)
    assert checkpoint.restore(payload, resumed, resumed_opt, resumed_sched, **identity) == (1, 1, 1.)
    runner.update(resumed, [batch], resumed_opt, resumed_sched, c, distances())
    assert_trees_equal(resumed.state_dict(), model.state_dict())
    assert_trees_equal(resumed_opt.state_dict(), opt.state_dict())
    assert_trees_equal(resumed_sched.state_dict(), sched.state_dict())
    assert_trees_equal(checkpoint.rng_state(), expected_rng)
    actual_draws = (random.random(), np.random.rand(), torch.rand(4))
    assert_trees_equal(actual_draws, expected_draws)
    for schema in ('maidata.decoder-training.v1','maidata.decoder-training.v2','maidata.decoder-training.v3'):
        old = dict(payload,schema=schema,model_state=deepcopy(model.decoder.state_dict()))
        with pytest.raises(ValueError):
            checkpoint.restore(old,resumed,resumed_opt,resumed_sched,**identity)
    with pytest.raises(ValueError,match='causal CONT'):
        checkpoint.restore(dict(payload,architecture='wrong-head'),resumed,resumed_opt,resumed_sched,**identity)
    with pytest.raises(ValueError,match='causal CONT'):
        checkpoint.restore(dict(payload,architecture='maidata.decoder-causal-cont-tick384.v1'),resumed,resumed_opt,resumed_sched,**identity)
