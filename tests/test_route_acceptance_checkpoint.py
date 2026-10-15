"""Route supervision belongs to the saved recipe, not the model tensor inventory."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch

from maidata.draft import route_acceptance
from maidata.entry import CONFIG
from maidata.joint_training import checkpoint as joint_checkpoint
from maidata.model import build_reconstruction
from maidata.training import checkpoint as stage_checkpoint


STEPS = [1, 1]


@pytest.fixture(params=[stage_checkpoint, joint_checkpoint], ids=['stage', 'joint'])
def checkpoint(request, monkeypatch):
    # Exercise complete CPU state without initializing CUDA RNG.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    return request.param


def config(mode='acceptance37', vocabulary='axis15-37'):
    return dict(reconstruction_route_loss=mode,
                reconstruction_route_vocabulary=vocabulary,
                pretrain_audio_mode='joint', pretrain_stop_after_epochs=2,
                draft_z_stride_beats=1)


def state():
    model = torch.nn.Linear(2, 3)
    model.audio_producer = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(49), _iterator=None)]
    return model, optimizer, scheduler, loaders


def capture(checkpoint, values, c):
    model, optimizer, scheduler, loaders = values
    model(torch.tensor([[.2, .7]])).square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    epoch = dict(completed_epochs=1) if checkpoint is stage_checkpoint else dict(epoch=1)
    return deepcopy(checkpoint.capture(model, optimizer, scheduler,
        **epoch, step=1, config=c, binding={}, source={}, steps=STEPS,
        loaders=loaders, best=.4, metrics={}))


def restore(checkpoint, saved, values, c):
    model, optimizer, scheduler, loaders = values
    extra = dict(binding={}, source={}, steps=STEPS) if checkpoint is stage_checkpoint else {}
    return checkpoint.restore(saved, model, optimizer, scheduler,
                              config=c, loaders=loaders, **extra)


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for one, two in zip(actual, expected):
            assert_state_equal(one, two)
    else:
        assert actual == expected


def test_acceptance_recipe_saves_actual_matrix_and_owns_its_copy(checkpoint):
    c = config()
    # A stale config value must never replace the fixed source table in use.
    c['reconstruction_route_acceptance'] = 'an-old-table-path'
    saved = checkpoint.recipe(c)
    expected = [list(row) for row in route_acceptance.ACCEPTANCE37]
    assert saved['reconstruction_route_loss'] == 'acceptance37'
    assert saved['reconstruction_route_acceptance'] == expected
    assert len(expected) == 37 and all(len(row) == 37 for row in expected)
    saved['reconstruction_route_acceptance'][0][1] = -1.
    assert checkpoint.recipe(c)['reconstruction_route_acceptance'] == expected
    assert [list(row) for row in route_acceptance.ACCEPTANCE37] == expected
    assert 'reconstruction_route_acceptance' not in checkpoint.recipe(dict(c, reconstruction_route_loss='ce'))


@pytest.mark.parametrize('mode', ['ce', 'acceptance37'])
def test_same_route_recipe_restores_complete_state(checkpoint, mode):
    c = config(mode)
    saved = capture(checkpoint, state(), c)
    other = state()
    assert restore(checkpoint, saved, other, c) == (1, 1, .4)
    model, optimizer, scheduler, loaders = other
    assert_state_equal(model.state_dict(), saved['model_state'])
    assert_state_equal(optimizer.state_dict(), saved['optimizer_state'])
    assert_state_equal(scheduler.state_dict(), saved['scheduler_state'])
    torch.testing.assert_close(loaders[0].generator.get_state(), saved['loader_rng'][0], rtol=0, atol=0)
    assert_state_equal(checkpoint.rng_state(), saved['rng'])


@pytest.mark.parametrize('vocabulary', ['route71', 'axis15-37'])
def test_legacy_missing_route_loss_restores_only_as_ce(checkpoint, vocabulary):
    c = config('ce', vocabulary)
    saved = capture(checkpoint, state(), c)
    saved['recipe'].pop('reconstruction_route_loss')
    if vocabulary == 'route71':
        saved['recipe'].pop('reconstruction_route_vocabulary')
    original = deepcopy(saved['recipe'])
    assert restore(checkpoint, saved, state(), c) == (1, 1, .4)
    assert saved['recipe'] == original


@pytest.mark.parametrize('before,after', [('ce', 'acceptance37'), ('acceptance37', 'ce')])
def test_route_loss_change_rejected_before_loading_weights(checkpoint, before, after):
    saved = capture(checkpoint, state(), config(before))
    other = state()
    with patch.object(other[0], 'load_state_dict', side_effect=AssertionError('weights loaded')):
        with pytest.raises(ValueError, match='recipe|representation'):
            restore(checkpoint, saved, other, config(after))


def test_actual_table_change_rejected_before_loading_weights(checkpoint, monkeypatch):
    c = config()
    saved = capture(checkpoint, state(), c)
    changed = [list(row) for row in route_acceptance.ACCEPTANCE37]
    changed[0][1] = .25 if changed[0][1] != .25 else .5
    monkeypatch.setattr(route_acceptance, 'ACCEPTANCE37', tuple(tuple(row) for row in changed))
    assert checkpoint.recipe(c)['reconstruction_route_acceptance'] == changed
    other = state()
    with patch.object(other[0], 'load_state_dict', side_effect=AssertionError('weights loaded')):
        with pytest.raises(ValueError, match='recipe|representation'):
            restore(checkpoint, saved, other, c)


def test_missing_saved_acceptance_matrix_is_not_filled_from_current_source(checkpoint):
    c = config()
    saved = capture(checkpoint, state(), c)
    saved['recipe'].pop('reconstruction_route_acceptance')
    other = state()
    with patch.object(other[0], 'load_state_dict', side_effect=AssertionError('weights loaded')):
        with pytest.raises(ValueError, match='recipe|representation'):
            restore(checkpoint, saved, other, c)


def test_route_loss_does_not_add_model_parameters_or_buffers():
    c = dict(CONFIG, draft_dim=8, draft_latent_dim=8, target_beats=4,
             draft_z_stride_beats=1, reconstruction_ffn_dim=12,
             reconstruction_dropout=0., reconstruction_route_vocabulary='axis15-37')
    torch.manual_seed(64)
    ce = build_reconstruction(dict(c, reconstruction_route_loss='ce'))
    torch.manual_seed(64)
    acceptance = build_reconstruction(dict(c, reconstruction_route_loss='acceptance37'))
    assert_state_equal(ce.state_dict(), acceptance.state_dict())
    acceptance.load_state_dict(ce.state_dict(), strict=True)
    assert not any('acceptance' in key for key in acceptance.state_dict())
