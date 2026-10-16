"""Final training-epoch presence diagnostics reuse the real forward without validation."""
import json
from pathlib import Path

import pytest
import torch

from maidata.joint_training import recipe, runner
from maidata.joint_training.network import JointTrainer
from maidata.training.runner import Diagnostics
from test_joint_runtime import runtime_dependencies
from test_z2_qknorm_onset_runtime import (z2_config, z2_item, single_config, integer_config,
                                        shared_joint_config, config, producer_config)


@pytest.mark.parametrize('stop', [1, 2])
def test_empty_validation_final_training_diagnostics_need_no_extra_forward(z2_config, tmp_path, monkeypatch, stop):
    c = dict(z2_config, joint_train_output=str(tmp_path/'training'),
        joint_train_scheduler_epochs=2, joint_train_stop_after_epochs=stop,
        joint_train_warmup_steps=1,
        **{key: 'accepted/'+key for key in recipe.PATH_KEYS if key != 'joint_train_output'})
    item = z2_item(c, monkeypatch)
    runtime_dependencies(monkeypatch, c, [item])
    plan_factory = runner.JointPlan
    def plan(root, config, *, training):
        result = plan_factory(root, config, training=training)
        if not training:
            result.groups = {}
        return result
    monkeypatch.setattr(runner, 'JointPlan', plan)
    monkeypatch.setattr(runner, 'validate', lambda *args, **kwargs: pytest.fail('empty validation was executed'))
    models, diagnostics = [], []
    forward_counts = dict(encoder=0, draft=0)
    backward_counts = dict(encoder=0, draft=0)
    def model_factory(config):
        model = JointTrainer(config)
        models.append(model)
        for name in forward_counts:
            def forward(*_, name=name):
                forward_counts[name] += 1
            def backward(_, name=name):
                backward_counts[name] += 1
            module = getattr(model, name)
            module.register_forward_hook(forward)
            next(module.parameters()).register_hook(backward)
        return model
    def diagnostic_factory():
        value = Diagnostics()
        diagnostics.append(value)
        return value
    monkeypatch.setattr(runner, 'JointTrainer', model_factory)
    monkeypatch.setattr(runner, 'Diagnostics', diagnostic_factory)
    assert runner.run(c) == dict(completed_epochs=stop, step=stop)
    rows = [json.loads(line) for line in (Path(c['joint_train_output'])/'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == list(range(1, stop+1))
    assert all('validation' not in row for row in rows)
    assert [row['epoch'] for row in rows if 'diagnostics' in row['train']] == [stop]
    assert len(diagnostics) == 1
    final = rows[-1]['train']['diagnostics']
    assert final['population'] == 'final training epoch, teacher-conditioned reconstruction'
    assert final['numeric_count_supervision'] is False
    presence = final['auxiliary']['onset_presence']
    assert (presence['count'], presence['positive_count'], presence['negative_count']) == (64, 4, 60)
    assert sum(map(sum, presence['confusion'])) == 64
    terms = rows[-1]['train']['auxiliary']['terms']['onset_presence']
    assert (terms['count'], terms['positive_count'], terms['negative_count']) == (64, 4, 60)
    assert forward_counts == backward_counts == dict(encoder=stop, draft=stop)
    model = models[0]
    assert all(parameter.grad is None for parameter in model.reconstruction.count_head.parameters())
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.reconstruction.onset_presence_head.parameters())
    checkpoints = Path(c['joint_train_output'])/'checkpoints'
    assert sorted(path.name for path in checkpoints.iterdir()) == [f'epoch-{stop:03d}.ckpt', 'last.ckpt']
    payload = torch.load(checkpoints/'last.ckpt', weights_only=False)
    assert payload['completed_epochs'] == payload['step'] == stop
    assert json.loads(json.dumps(payload['metrics']['train']['diagnostics'])) == final
    assert payload['best_validation_loss'] is None
    assert payload['recipe']['reconstruction_onset_presence_supervision'] is True
    assert payload['recipe']['draft_z_stride_beats'] == 2
    assert 'reconstruction.onset_presence_head.weight' in payload['model_state']
