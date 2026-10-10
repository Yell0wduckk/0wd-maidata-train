"""Training Plan provenance never switches inference to reference-chart labels."""
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config
from test_inference_joint_encoder import checkpoints
from test_pretrain_runtime import item
from maidata.inference.checkpoint import load_models
from maidata.inference.conditioning import prepare_conditions
from maidata.time import TimingMap


@pytest.mark.parametrize('schema', ['maidata.decoder-training.v1', 'maidata.decoder-training.v2'])
def test_checkpoint_without_plan_source_defaults_to_predicted(config, tmp_path, schema):
    legacy_config = dict(config)
    legacy_config.pop('decoder_train_plan_source', None)
    paths, *_ = checkpoints(legacy_config, tmp_path, schema=schema)
    models = load_models(*paths, device='cpu')
    assert models.provenance['training_plan_source'] == 'predicted'
    assert models.provenance['inference_plan_source'] == 'predicted'


def test_gt_trained_checkpoint_still_predicts_inference_plan_without_gt_labels(config, tmp_path):
    c = dict(config, decoder_train_plan_source='gt')
    paths, *_ = checkpoints(c, tmp_path)
    models = load_models(*paths, device='cpu')
    assert models.provenance['training_plan_source'] == 'gt'
    assert models.provenance['inference_plan_source'] == 'predicted'
    with patch('maidata.draft.plan_targets.MusicPlanTargets',
               side_effect=AssertionError('inference must not read GT Plan labels')), \
         patch('maidata.draft.source_timing.read_source_end_beat',
               side_effect=AssertionError('inference must not read a reference chart')), \
         patch.object(models.planner, 'forward', wraps=models.planner.forward) as planner:
        conditions = prepare_conditions(models.encoder, models.draft, models.planner, models.config,
            item(0, c)['grid'], TimingMap([(0, 120)]), 1.2, difficulty=5, level=14.5,
            device='cpu', decoder_encoder=models.decoder_encoder)
    assert planner.call_count == 1
    with torch.inference_mode():
        prediction = models.planner(planner.call_args.args[0])
    assert torch.equal(conditions.plans, prediction.plan_values)
    assert torch.equal(conditions.plan_valid, prediction.valid)
