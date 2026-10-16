"""Real shared E/D/R/P/M updates for integer overlapping conditions."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import overlap_reconstruction_target, planner_target
from maidata.joint_training import checkpoint, update as updates
from maidata.joint_training.data import collate_audio
from maidata.joint_training.network import JointTrainer
from maidata.inference.checkpoint import load_joint_models
from maidata.training.runner import supports
from test_decoder_training import fields, distances
from test_joint_gradients import optimizers
from joint_runtime_helpers import shared_joint_config, shared_joint_item, direct_update
from test_inference import config, producer_config


@pytest.fixture
def integer_config(shared_joint_config):
    return dict(shared_joint_config, draft_z_stride_beats=1, joint_train_z_stride_beats=1,
        draft_z_anchor_mode='integer', reconstruction_span_beats=4,
        reconstruction_count_supervision=False, planner_span_beats=16,
        planner_stride_beats=4, planner_layout='centered-overlap')


def integer_item(c, monkeypatch):
    base = shared_joint_item(c, monkeypatch)
    end = base['coverage'].draft_beats
    auxiliary, ids, charts, centers, plans = [], [], [], [], []
    for key in base['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, _, _ = fields(monkeypatch, tokens, key=key)
        chart = from_record(record, source_end_beat=Fraction(16))
        planner = MusicPlanTargets(chart)
        for start in base['coverage'].starts(include_tail=True):
            auxiliary.append((f'{key}:{start}', key, start,
                overlap_reconstruction_target(chart, start_tick=start*96, target_beats=16,
                    coverage_end_beat=end),
                planner_target(planner, start*96, (start+16)*96, span_beats=16,
                    layout='centered-overlap', last_parent=start+16 == end)))
        for center in range(0, end+1, 4):
            t = planner.window((center-8)*96, (center+8)*96)
            ids.append(f'{key}@{center}'); charts.append(key); centers.append(center)
            plans.append([*t.continuous, *[float(i == t.category) for i in range(4)]])
    bank = dict(planner_layout='centered-overlap', plan_span_beats=16, plan_stride_beats=4,
        window_ids=ids, chart_ids=charts, center_beats=torch.tensor(centers).float(),
        plan_values=torch.tensor(plans).float(), valid=torch.ones(len(ids), dtype=torch.bool))
    return dict(base, windows=auxiliary, gt_plan=bank)


@pytest.mark.parametrize('budget', [1, 10000])
def test_integer_joint_direct_and_chunk_bridge_share_one_real_graph(integer_config, monkeypatch, budget):
    c = dict(integer_config, decoder_train_chunk_token_budget=budget)
    item = integer_item(c, monkeypatch)
    batch = collate_audio([item], config=c, packed=True)
    assert batch.draft_inputs.query_beats.tolist() == [list(range(33))]
    assert batch.conditions.decoder_z.beats.tolist() == [list(range(-3,20)), list(range(13,36))]
    assert batch.conditions.plan_beats.tolist() == [[0,4,8,12,16], [16,20,24,28,32]]
    assert batch.conditions.plan.shape == (2,5,12)
    assert batch.conditions.plan_valid.all()
    assert batch.conditions.decoder_z.valid.sum(-1).tolist() == [20,20]
    assert supports(batch)['planner_continuous'] == 9
    assert supports(batch)['moment96'] == 7
    model = JointTrainer(c)
    reference = deepcopy(model)
    opt, sched = optimizers(model)
    ref_opt, ref_sched = optimizers(reference)
    traversals = dict(encoder=0, draft=0)
    def hook(name):
        def count(gradient): traversals[name] += 1
        return count
    handles = [next(getattr(model,n).parameters()).register_hook(hook(n)) for n in traversals]
    with ExitStack() as stack:
        encode = stack.enter_context(patch.object(model.encoder,'forward',wraps=model.encoder.forward))
        draft = stack.enter_context(patch.object(model.draft,'forward',wraps=model.draft.forward))
        actual, audit = updates.update(model,[batch],opt,sched,c,distances(),audit=True)
    for handle in handles: handle.remove()
    together = collate_audio([item], config=dict(c,decoder_train_chunk_token_budget=10000), packed=True)
    expected = direct_update(reference,together,ref_opt,ref_sched,c)
    assert encode.call_count == draft.call_count == 1
    assert traversals == dict(encoder=1,draft=1)
    assert all(value > 0 for value in audit['gradients'].values())
    assert all(value > 0 for value in audit['changed_parameter_tensors'].values())
    assert actual.auxiliary.counts == expected.auxiliary.counts
    assert actual.decoder.counts == expected.decoder.counts
    assert actual.report(c)['loss'] == pytest.approx(expected.report(c)['loss'],rel=3e-6)
    for (name,p),(other_name,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5,msg=name)
        if p.grad is None or q.grad is None:
            assert p.grad is q.grad
        else:
            torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=5e-4,msg=name+' gradient')


def test_new_geometry_has_same_parameters_and_rejects_old_semantic_resume(integer_config):
    new = JointTrainer(integer_config)
    old_config = dict(integer_config,draft_z_anchor_mode='block-center',draft_z_stride_beats=4,
        joint_train_z_stride_beats=4,reconstruction_span_beats=None,planner_layout='legacy',
        planner_span_beats=16,planner_stride_beats=None)
    old = JointTrainer(old_config)
    assert {k:v.shape for k,v in new.state_dict().items()} == {k:v.shape for k,v in old.state_dict().items()}
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters())
    opt,sched = optimizers(new)
    payload = dict(schema=checkpoint.SCHEMA,architecture=checkpoint.ARCHITECTURE,
        completed_epochs=1,step=1,steps_by_epoch=[1],scheduler_state=dict(last_epoch=1),
        loader_rng=[],loader_started=[],recipe=checkpoint.recipe(old_config))
    with pytest.raises(ValueError,match='architecture/representation'):
        checkpoint.restore(payload,new,opt,sched,loaders=[],config=integer_config)


def test_complete_integer_checkpoint_restores_training_and_actual_inference(integer_config, tmp_path):
    c = integer_config
    model = JointTrainer(c)
    opt = torch.optim.AdamW(model.parameters(),lr=c['joint_train_learning_rate'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda step: 1.)
    for p in model.parameters(): p.grad = torch.ones_like(p)*.001
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    payload = checkpoint.capture(model,opt,sched,epoch=1,step=1,config=c,binding={},
        source={'commit':'integer-test'},steps=[1]*6,loaders=[],best=None,metrics={})
    path = tmp_path/'integer.ckpt'
    torch.save(payload,path)
    expected_rng = torch.rand(5)
    restored = JointTrainer(c)
    new_opt = torch.optim.AdamW(restored.parameters(),lr=c['joint_train_learning_rate'])
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt,lambda step: 1.)
    assert checkpoint.restore(torch.load(path,weights_only=False),restored,new_opt,new_sched,
                              loaders=[],config=c)[:2] == (1,1)
    torch.testing.assert_close(torch.rand(5),expected_rng,atol=0,rtol=0)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name],value,atol=0,rtol=0)
    assert len(new_opt.state) == len(opt.state)
    loaded = load_joint_models(path,device='cpu')
    assert loaded.config['planner_layout'] == 'centered-overlap'
    assert loaded.config['draft_z_anchor_mode'] == 'integer'
    assert loaded.config['reconstruction_span_beats'] == 4
    assert loaded.decoder.planner_layout == 'centered-overlap'
