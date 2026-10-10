"""Real small E/D/R/P update, whole-song collation and complete-epoch resume."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from maidata.entry import CONFIG
from maidata.chart.types import Note
from maidata.draft.chart_data import ChartData, OnsetGroup, PlanNote
from maidata.draft.conditions import ChartCondition
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target
from maidata.draft.windows import song_coverage
from maidata.time import TimingMap
from maidata.training.data import AudioPlan, AudioBatchSampler, collate_audio
from maidata.training.network import Pretrainer
from maidata.training.pretrain import seed_scratch, make_optimizer, make_scheduler
from maidata.training.runner import update
from maidata.training import runner, checkpoint


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return dict(CONFIG, mel_bins=7, samples_per_beat=5, d_model=18, heads=3, ffn_dim=28,
        stage_depths=(1, 1, 1), temporal_kernel_size=3, stem_channels=4, frequency_dim=6,
        frequency_queries=2, fpn_rates=(3, 2, 1), target_beats=3,
        audio_before_beats=2, audio_after_beats=1, dropout=0.,
        draft_layers=2, draft_dim=24, draft_heads=3, draft_experts=3, draft_expert_ffn_dim=29,
        draft_latent_dim=10, planner_dim=16, planner_heads=2, planner_mlp_dim=23, planner_dropout=0.,
        reconstruction_dim=12, reconstruction_heads=3, reconstruction_ffn_dim=25,
        reconstruction_dropout=0., reconstruction_query_rows=12,
        reconstruction_before_beats=2, reconstruction_after_beats=1,
        pretrain_device='cpu', pretrain_workers=0, pretrain_precision='fp32',
        pretrain_warmup_steps=2, pretrain_scheduler_epochs=8, pretrain_stop_after_epochs=6,
        pretrain_token_budget=None)


def item(index, config, remaster=False):
    timing = TimingMap([(0, 120)])
    coverage = song_coverage(timing, 1.2, first_beat=0, target_beats=3)
    times = np.arange(15).reshape(3, 5)/5
    valid = times < 2.4
    rng = np.random.default_rng(index)
    mel = rng.normal(size=(3,5,7)).astype(np.float32)
    aux = rng.normal(size=(3,5,32)).astype(np.float32)
    mel[~valid], aux[~valid] = 0, 0
    grid = dict(mel_patches=mel, sample_valid=valid, beat_centers=np.arange(3,dtype=np.float32)+.5,
        aux32_grid=aux, aux32_centers=aux.mean(1), sample_times_sec=times*.5,
        sample_indices=np.rint(times*.5*24000).astype(np.int64), grid_k=np.arange(15).reshape(3,5),
        first_beat=np.array(0))
    variant = str(index)
    conditions, windows = {}, []
    for difficulty in ([5,6] if remaster else [5]):
        key = f'{index}:{difficulty}'
        ticks = [0] if index % 2 else [0, 96]
        chart = ChartData(tuple(OnsetGroup(t,(Note.make('tap','A1'),)) for t in ticks),
                          tuple(PlanNote(t,'tap',1) for t in ticks), timing, Fraction(3))
        conditions[key] = ChartCondition(variant, difficulty, 14.+difficulty/10, str(14.+difficulty/10))
        planner = MusicPlanTargets(chart)
        for start in coverage.starts(include_tail=True):
            windows.append((f'{key}:{start}', key, start,
                reconstruction_target(chart, start_tick=start*96, target_beats=3),
                planner.window(start*96,(start+3)*96)))
    return dict(variant=variant,grid=grid,coverage=coverage,conditions=conditions,windows=windows)


def test_actual_update_shares_audio_and_preserves_all_tail_supervision(config):
    seed_scratch(config)
    batch = collate_audio([item(0,config,True), item(1,config)],config=config)
    assert len(batch.audio_variants)==2 and len(batch.draft_inputs.audio_mapping)==3
    assert batch.draft_inputs.audio_mapping.tolist()==[0,0,1]
    assert len(batch.window_ids)==6 and batch.draft_inputs.draft_valid.all()
    tail = batch.starts==3
    assert batch.reconstruction.moment_counts[tail].sum()==0
    assert batch.valid[tail].all() and batch.planner.valid[tail].all()
    model = Pretrainer(config)
    optimizer = make_optimizer(model.parameters(),config)
    scheduler = make_scheduler(optimizer,[1]*8,config)
    with patch.object(model.encoder,'prepare_song',wraps=model.encoder.prepare_song) as prepare, \
         patch.object(model.draft,'forward',wraps=model.draft.forward) as draft:
        metrics,audit=update(model,[batch],optimizer,scheduler,config,audit=True)
    assert prepare.call_count==draft.call_count==1
    assert all(v>0 for v in audit['gradients'].values())
    assert all(v>0 for v in audit['changed_parameter_tensors'].values())
    assert metrics.sums['count'][1]==18 and metrics.sums['planner_category'][1]==6
    assert scheduler.last_epoch==1


def test_accumulation_uses_global_supports_not_mean_of_means(config):
    seed_scratch(config)
    items=[item(0,config,True),item(1,config)]
    together=collate_audio(items,config=config)
    separately=[collate_audio([i],config=config) for i in items]
    one=Pretrainer(config)
    two=deepcopy(one)
    o1,o2=[make_optimizer(m.parameters(),config) for m in (one,two)]
    s1,s2=[make_scheduler(o,[1]*8,config) for o in (o1,o2)]
    a,_=update(one,[together],o1,s1,config)
    b,_=update(two,separately,o2,s2,config)
    assert a.report(config)['loss']==pytest.approx(b.report(config)['loss'],rel=2e-6)
    # Compare unscaled accumulated gradients before Adam's near-zero sign amplification.
    for p,q in zip(one.parameters(),two.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)


def make_plan(tmp_path,config):
    rows=[]
    for index in range(5):
        rows.extend(dict(window_id=f'{index}:{w}',audio_variant=str(index)) for w in range(index+1))
    p=dict(schema='maidata.window-sampling-plan.v1',seed=config['pretrain_seed'],epoch=1,
        policy=dict(views=['original'],consumer='encoder-draft-pretraining'),
        config=dict(target_beats=3,target_stride_beats=3,audio_before_beats=2,audio_after_beats=1,complex_rhythm_multiplier=1.),
        totals=dict(base_windows=len(rows)),files=dict(windows='windows.jsonl',occurrences='occurrences.jsonl'),charts=[])
    order=list(map(str,range(5)))
    random.Random(f'{p["seed"]}:1:groups').shuffle(order)
    p['audio_groups']=[dict(audio_variant=k,occurrences=int(k)+1,encoder_starts_beats=[0],
                            actual_audio_end_beat=2.4,first_beat=0) for k in order]
    occurrences=[]
    for key in order:
        group=[r for r in rows if r['audio_variant']==key]
        random.Random(f'{p["seed"]}:1:within:{key}').shuffle(group)
        occurrences.extend(dict(r,occurrence=0) for r in group)
    for name,value in [('plan.json',p),('windows.jsonl',rows),('occurrences.jsonl',occurrences)]:
        (tmp_path/name).write_text(json.dumps(value) if name.endswith('.json') else '\n'.join(map(json.dumps,value)),encoding='utf8')
    return p


def test_saved_plan_order_and_epoch_batch_boundaries(config,tmp_path):
    make_plan(tmp_path,config)
    plan=AudioPlan(tmp_path,config)
    sampler=AudioBatchSampler(plan,config)
    batches=list(sampler)
    assert list(map(len,batches))==[2,2,1]
    assert [k for b in batches for k,e in b]==list(plan.groups)
    sampler.epoch=2
    assert [k for b in sampler for k,e in b]==plan.order(2)
    assert all(e==2 for b in sampler for k,e in b)
    path=tmp_path/'occurrences.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    rows[0]['occurrence']=1
    path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='occurrences'):
        AudioPlan(tmp_path,config)


class SyntheticDataset(Dataset):
    def __init__(self,config): self.config=config
    def __len__(self): return 2
    def __getitem__(self,index): return item(int(index[0]),self.config)


def test_complete_epoch_resume_is_exact_and_never_saves_failed_epoch(config,tmp_path,monkeypatch):
    # The real runner, optimizer/scheduler, checkpoints, dropout and loader RNG.
    config=dict(config,dropout=.1,pretrain_train_plan='train',pretrain_validation_plan='validation')
    class Plan:
        def __init__(self,*args):
            self.groups={'0':{},'1':{}}
            self.windows=list(range(4))
            self.manifest={'epoch':1}
        def order(self,epoch): return ['1','0'] if epoch%2 else ['0','1']
        def batches(self,epoch,c):
            from maidata.training.batching import optimizer_batches
            return optimizer_batches(self.order(epoch),c)
    def loader(plan,c):
        from functools import partial
        return DataLoader(SyntheticDataset(c),batch_sampler=AudioBatchSampler(plan,c),
            collate_fn=partial(collate_audio,config=c),generator=torch.Generator().manual_seed(731))
    monkeypatch.setattr(runner,'AudioPlan',Plan)
    monkeypatch.setattr(runner,'make_loader',loader)
    monkeypatch.setattr(checkpoint,'data_binding',lambda *a: {'fixed':'inputs'})
    monkeypatch.setattr(checkpoint,'source_identity',lambda: {'fixed':'source'})
    full=dict(config,pretrain_output=str(tmp_path/'full'))
    runner.run(full)
    interrupted=dict(config,pretrain_output=str(tmp_path/'interrupted'))
    real_update=runner.update
    calls=0
    def fail(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3: raise RuntimeError('interruption')
        return real_update(*args,**kwargs)
    monkeypatch.setattr(runner,'update',fail)
    with pytest.raises(RuntimeError,match='interruption'): runner.run(interrupted)
    ckpt=Path(interrupted['pretrain_output'])/'checkpoints/last.ckpt'
    assert torch.load(ckpt,weights_only=False)['completed_epochs']==2
    assert sorted(p.name for p in ckpt.parent.iterdir())==['last.ckpt']
    monkeypatch.setattr(runner,'update',real_update)
    runner.run(interrupted,resume=ckpt)
    a=torch.load(Path(full['pretrain_output'])/'checkpoints/last.ckpt',weights_only=False)
    b=torch.load(ckpt,weights_only=False)
    for k in a['model_state']: torch.testing.assert_close(a['model_state'][k],b['model_state'][k],rtol=0,atol=0)
    assert a['step']==b['step']==6
    assert a['scheduler_state']==b['scheduler_state']
    for x,y in zip(a['loader_rng'],b['loader_rng']): assert torch.equal(x,y)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    metrics=[json.loads(l) for l in (Path(interrupted['pretrain_output'])/'metrics.jsonl').read_text().splitlines()]
    assert [m['epoch'] for m in metrics if 'validation' in m]==[5,6]
    assert [m['epoch'] for m in metrics if 'diagnostics' in m.get('validation',{})]==[6]
    assert sorted(p.name for p in ckpt.parent.iterdir())==['best.ckpt','epoch-005.ckpt','epoch-006.ckpt','last.ckpt']
    assert runner.run(interrupted,resume=ckpt)['step']==6
    with pytest.raises(ValueError,match='binding changed'):
        runner.run(dict(interrupted,pretrain_audio_groups_per_batch=1),resume=ckpt)


def test_cli_overlay_and_module_boundary(tmp_path):
    from maidata.training.__main__ import load_config
    p=tmp_path/'linux.json'
    p.write_text(json.dumps({'pretrain_source_root':'/srv/sources','pretrain_audio_groups_per_batch':3}))
    c=load_config(p)
    assert c['pretrain_source_root']=='/srv/sources' and c['pretrain_audio_groups_per_batch']==3
    assert c['pretrain_workers']==12 and CONFIG['pretrain_audio_groups_per_batch']==2
    p.write_text('{"token_budget":4096}')
    with pytest.raises(ValueError,match='unknown'): load_config(p)


def test_persistent_loader_rng_has_no_extra_resume_draw():
    from torch.utils.data import TensorDataset
    from maidata.training.data import epoch_iterator
    loaders=[]
    try:
        for _ in range(2):
            loaders.append(DataLoader(TensorDataset(torch.arange(4)),batch_size=2,num_workers=1,
                persistent_workers=True,generator=torch.Generator().manual_seed(82)))
        assert len(list(epoch_iterator(loaders[0])))==2
        saved=loaders[0].generator.get_state().clone()
        continuous=list(epoch_iterator(loaders[0]))
        loaders[1].generator.set_state(saved)
        resumed=list(epoch_iterator(loaders[1],saved))
        for a,b in zip(continuous,resumed): assert torch.equal(a[0],b[0])
        assert torch.equal(loaders[0].generator.get_state(),loaders[1].generator.get_state())
    finally:
        for loader in loaders:
            if loader._iterator is not None: loader._iterator._shutdown_workers()
