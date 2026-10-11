"""Stage-II fixed D/P, online audio, vectorized supervision and epoch resume."""
from copy import deepcopy
from dataclasses import replace, asdict
import json
import random
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

from maidata.chart import Moment,Note,Route,HoldDuration,SlideDuration,EncodedChart,decode_chart
from maidata.chart_cache import produce
from maidata.chart_cache.cache import save_chart
from maidata.chart_cache.training import build_supports,save_supports
from maidata.decoder.windows import build_window
from maidata.decoder.supports import window_stages
from maidata.decoder.batching import collate_windows
from maidata.decoder.supervision import decoder_loss
from maidata.decoder.losses import RouteGeometryLoss
from maidata.model import build_decoder,decoder_time_distributions
from maidata.time import TimingMap
from maidata.decoder_training.support_cache import ChartSupports
from maidata.decoder_training import supervision,checkpoint,recipe,runner
from maidata.decoder_training.data import collate_audio,DecoderDataset
from maidata.decoder_training.ordering import history_rng
from maidata.decoder_training.network import DecoderTrainer
from maidata.decoder_training.producer import FrozenProducer
from maidata.decoder_training.batching import window_chunks
from test_decoder_supervision import geometry
from test_decoder_network import config as decoder_config,inputs,prepare as memory
from test_pretrain_runtime import config as encoder_config,item as audio_item


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def config(encoder_config):
    c = dict(encoder_config)
    c.update({k:v for k,v in decoder_config().items() if k.startswith('decoder_')})
    c.update(target_stride_beats=3,decoder_z_before_beats=2,decoder_z_after_beats=1,
        decoder_train_device='cpu',decoder_train_workers=0,decoder_train_precision='fp32',
        decoder_train_learning_rate=1e-3,decoder_train_weight_decay=.01,decoder_train_gradient_clip=1.,
        decoder_train_warmup_steps=1,decoder_train_scheduler_epochs=6,decoder_train_stop_after_epochs=2,
        decoder_train_gradient_accumulation_steps=1,decoder_train_row_budget=8192,
        decoder_train_chunk_token_budget=24,decoder_train_physical_weight=.1,decoder_train_judge_weight=.2,
        decoder_train_trajectory_weight=.3,decoder_history_dropout=.15,
        decoder_train_plan_source='predicted')  # This fixture explicitly supplies a predicted-plan cache.
    return c


def fields(monkeypatch,tokens,*,audio_end=1.2,bpm=120,key='0:5'):
    geo = geometry()
    monkeypatch.setattr(produce,'GEOMETRY',geo)
    metadata = dict(schema='maimai.chart_ir.v1',tick_rate=180,first_seconds=0,
        controls=[dict(kind='bpm',bpm=bpm,position_beats=dict(num=0,den=1))])
    song,diff = key.split(':')
    row = dict(chart_id=key,song_id=song,difficulty=diff,audio_variant=song,path=f'charts/{song}/{diff}.json.gz',
        timing=dict(bpm=[[0,1,bpm]],first_seconds=0),audio_end_seconds=audio_end)
    record = produce.build_record(decode_chart(EncodedChart(tuple(tokens),metadata),geo),row)
    return record,build_supports(record,geo),geo


def example(chart,start,end,*,drop=False,audio_end=1.2):
    return build_window(chart.tokens,chart.timing,sequence_start_tick=chart.start_tick,start_tick=start,end_tick=end,
        audio_after_beats=1,audio_end_seconds=audio_end,history_beats=4,history_dropout=float(drop),
        training=True,rng=random.Random(0))


def test_cached_prefix_rejoins_oracle_with_old_live_actions_chains_and_dropout(monkeypatch):
    tokens = [Moment(0),Note.make('hold','A1'),HoldDuration(8*96),
        Note.make('slide','A2'),Route(0,1),Route(1),SlideDuration(7*96),
        Moment(6000),Note.make('tap','A1'),Note.make('hold','A1'),HoldDuration(96),
        Moment(1000),Note.make('slide','A3'),Route(0,0),SlideDuration(96),
        Route(1,0),Route(2),SlideDuration(96),Route(70,0),SlideDuration(96),
        Moment(3000),Note.make('touch','C')]
    record,supports,geo = fields(monkeypatch,tokens,audio_end=12.,bpm=60)
    chart = ChartSupports(record,supports,geo)
    total_reused = 0
    for start in (0,4*96,6*96,8*96,12*96):
        for drop in (False,True):
            w = example(chart,start,start+2*96,drop=drop,audio_end=12.)
            got,audit = chart.window(w)
            chart.state.restore(chart.before_window(start))
            want = window_stages(w,chart.state)
            assert got == want
            total_reused += audit['reused']
            if start == 6*96:
                assert chart.before_window(start).physical.actions
                assert w.boundary_index == 0  # outside four-beat token history
    assert total_reused > 5


def test_vectorized_heads_match_scalar_totals_counts_and_gradients(monkeypatch):
    tokens = [Moment(0),Note.make('tap','A1',short_hold=1),Note.make('hold','A2'),HoldDuration(2),
        Moment(250),Note.make('touch','C',short_touch_hold=1),Note.make('touch_hold','B1'),HoldDuration(4),
        Note.make('slide','A3'),Route(0,1),Route(1),SlideDuration(24),Route(70,0),SlideDuration(24)]
    record,supports,geo = fields(monkeypatch,tokens)
    chart = ChartSupports(record,supports,geo)
    windows = []
    for start in (0,96,288):
        w = example(chart,start,start+96)
        windows.append(dict(example=w,stages=chart.window(w)[0],timing=chart.timing,audio_end=1.2))
    model = build_decoder(decoder_config())
    model.eval()
    examples = [r['example'] for r in windows]
    tokens,types,_ = collate_windows(examples,device='cpu')
    _,audio,z,plan = inputs(model)
    # Test memory helper has two windows; explicitly expand one shared condition.
    z = type(z)(z.values[:1].expand(3,-1,-1),z.beats[:1].expand(3,-1),z.valid[:1].expand(3,-1))
    mem = model.prepare_memory(model.prepare_audio(audio),torch.zeros(3,dtype=torch.long),z,plan[:1].expand(3,-1),torch.ones(3,dtype=torch.bool))
    out = model(tokens,types,mem)
    distance = (torch.arange(71)[:,None]-torch.arange(71)[None,:]).abs().float()/70
    geo = RouteGeometryLoss(distance,distance.square())
    kwargs = dict(distributions=decoder_time_distributions(decoder_config()),geometry=geo)
    reference = decoder_loss(model.heads,out,examples,[r['stages'] for r in windows],
        [chart.timing]*3,[1.2]*3,**kwargs)
    prepared = supervision.prepare(windows)
    actual = supervision.loss(model.heads,out,prepared,**kwargs)
    for key in supervision.KEYS:
        a,b = getattr(actual,key),getattr(reference,key)
        assert a.count == b.count
        torch.testing.assert_close(a.total,b.total,atol=2e-5,rtol=2e-5)
    params = tuple(model.parameters())
    weights = dict(physical_weight=.1,judge_weight=.2,trajectory_weight=.3)
    a = torch.autograd.grad(actual.objective(**weights),params,retain_graph=True,allow_unused=True)
    b = torch.autograd.grad(reference.objective(**weights),params,retain_graph=True,allow_unused=True)
    for x,y in zip(a,b):
        if x is not None and y is not None:
            torch.testing.assert_close(x,y,atol=3e-5,rtol=2e-4)
        else:
            assert x is y or (x if x is not None else y).count_nonzero()==0
    flag_grads = torch.autograd.grad(actual.physical.total.sum(),tuple(model.heads.flags.parameters()),allow_unused=True)
    assert all(g is None for g in flag_grads)


def test_audio_grid_before_first_source_boundary_and_empty_chart(monkeypatch):
    for tokens in ([Moment(0),Note.make('tap','A1')],[Moment(0)]):
        record,supports,geo = fields(monkeypatch,tokens)
        chart = ChartSupports(record,supports,geo)
        for start,end in ((-96,96),(96,192),(288,384)):
            w = example(chart,start,end)
            got,_ = chart.window(w)
            chart.state.restore(chart.before_window(start))
            assert got==window_stages(w,chart.state)


def training_item(index,c,monkeypatch,*,remaster=False,epoch=1,training=True):
    item = audio_item(index,c,remaster)
    windows = []
    for key in item['conditions']:
        tokens = [Moment(0),Note.make('slide','A2'),Route(0,1),SlideDuration(24),
                  Moment(500),Note.make('tap','A1')]
        record,supports,geo = fields(monkeypatch,tokens,key=key)
        chart = ChartSupports(record,supports,geo)
        for start in (0,3):
            for occurrence in range(2 if start==0 else 1):
                identity = f'{key}:{start}'
                w = build_window(chart.tokens,chart.timing,sequence_start_tick=0,start_tick=start*96,end_tick=(start+3)*96,
                    audio_after_beats=1,audio_end_seconds=1.2,history_beats=4,history_dropout=c['decoder_history_dropout'],
                    training=training,rng=history_rng(c['decoder_train_seed'],epoch,identity,occurrence))
                windows.append(dict(identity=(identity,occurrence),chart_id=key,start=start,example=w,
                    stages=chart.window(w)[0],timing=chart.timing,audio_end=1.2))
    return freeze_item(dict(item,windows=[tensor_window(w) for w in windows]),c)


def tensor_window(row):
    from dataclasses import fields
    w = row['example']
    tokens, types, _ = collate_windows([w], device='cpu')
    labels = supervision.prepare([row])
    for group in labels.groups.values():
        group['indices'][:, 1] -= w.boundary_index
    labels.flags['indices'][:, 1] -= w.boundary_index
    return dict(row, tokens={f.name:getattr(tokens,f.name)[0] for f in fields(tokens)}, next_types=types[0],
                labels=dict(groups=labels.groups,counts=labels.counts,flags=labels.flags),
                length=len(w.inputs), offset=w.boundary_index, history_dropped=w.history_dropped)


def freeze_item(item,c):
    """Small independent online oracle, used only to create test cache tensors."""
    from maidata.decoder_training.p0_cache import collate,SCHEMA as P0_SCHEMA
    from maidata.decoder_training.dp_cache import SCHEMA as DP_SCHEMA, Z_REPRESENTATION
    from maidata.draft.conditions import make_draft_inputs
    from maidata.draft.windows import assemble_full_p0,select_windows
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(614)
        producer = FrozenProducer(c)
        audio = collate([(item['variant'],item['grid'],item['coverage'])],config=c).to('cpu')
        local = producer.encoder(audio.songs,audio.windows)
        bank = assemble_full_p0(local.P0,audio.windows.target_start_beats,audio.windows.window_to_song,
            audio.coverages,p0_rate=c['fpn_rates'][0],layout=audio.p0_layout)
        inputs,mapping = make_draft_inputs(list(item['conditions'].values()),[item['variant']],[item['coverage']],device='cpu')
        z = producer.draft(bank,inputs)
        owner = dict(zip(item['conditions'],mapping.tolist()))
        rows = list({w['identity'][0]:w for w in item['windows']}.values())
        groups = torch.tensor([owner[w['chart_id']] for w in rows])
        starts = torch.tensor([w['start'] for w in rows],dtype=torch.float32)
        valid = torch.ones(len(rows),dtype=torch.bool)
        predicted = producer.planner(select_windows(z,groups,starts,valid,target_beats=c['target_beats'],
            before_beats=0,after_beats=0))
    p0 = dict(schema=P0_SCHEMA,audio_variant=item['variant'],producer_sha256='fixed-producer',
        coverage=asdict(item['coverage']),**{k:getattr(bank,k)[0] for k in ('values','beats','valid')})
    dp = dict(schema=DP_SCHEMA,audio_variant=item['variant'],producer_sha256='fixed-producer',
        z_representation=Z_REPRESENTATION,z_width=c['draft_latent_dim'],
        chart_to_group=owner,conditions=[asdict(v) for v in item['conditions'].values()],
        z={k:getattr(z,k) for k in ('values','beats','valid')},
        planner=dict(window_ids=[w['identity'][0] for w in rows],chart_ids=[w['chart_id'] for w in rows],
            groups=groups,starts=starts,plan_values=predicted.plan_values,valid=predicted.valid))
    dp['window_index'] = {key:i for i,key in enumerate(dp['planner']['window_ids'])}
    dp['chart_conditions'] = dict(zip(owner,dp['conditions']))
    return dict(item,p0=p0,dp=dp)


def distances():
    d = (torch.arange(71)[:,None]-torch.arange(71)[None,:]).abs().float()/70
    return RouteGeometryLoss(d,d.square())


def test_joint_update_constructs_encoder_and_m_and_preserves_chunk_denominators(config,monkeypatch):
    rows = [training_item(0,config,monkeypatch,remaster=True),training_item(1,config,monkeypatch)]
    batches = [collate_audio([r],config=config) for r in rows]
    model = DecoderTrainer(config)
    other = deepcopy(model)
    one = dict(config,decoder_train_chunk_token_budget=10000)
    together = collate_audio(rows,config=one)
    # SGD isolates the loss-denominator/gradient oracle: regrouping E's FP32
    # kernels changes roundoff that Adam can amplify for nearly-zero gradients.
    opts = [torch.optim.SGD(m.parameters(),lr=config['decoder_train_learning_rate']) for m in (model,other)]
    scheds = [recipe.scheduler(o,[1]*6,config) for o in opts]
    with patch.object(model.decoder,'prepare_audio',wraps=model.decoder.prepare_audio) as project:
        a,audit = runner.update(model,[together],opts[0],scheds[0],one,distances(),audit=True)
    assert project.call_count==1
    b,_ = runner.update(other,batches,opts[1],scheds[1],config,distances(),audit=True)
    assert set(audit['changed_parameter_tensors'])=={'encoder','decoder'}
    assert set(dict(model.named_children()))=={'encoder','decoder'}
    assert all(not t.requires_grad for t in (together.conditions.audio.values,together.conditions.decoder_z.values,together.conditions.plan))
    assert model.decoder.condition.z.weight.grad.abs().sum()>0
    assert a.counts==b.counts
    assert a.report(config)['loss']==pytest.approx(b.report(config)['loss'],rel=1e-5)
    for (name,x),(other_name,y) in zip(model.named_parameters(),other.named_parameters()):
        assert name==other_name
        torch.testing.assert_close(x,y,atol=2e-5,rtol=3e-4,msg=name)
        if x.grad is None or y.grad is None:
            assert x.grad is y.grad
        else:
            torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=3e-4,msg=name+' gradient')
    assert not hasattr(model,'reconstruction')


def test_chunk_budget_never_crops_history_or_reorders_occurrences():
    rows = [dict(length=n) for n in (7,3,25,4)]
    chunks = window_chunks(rows,16)
    assert chunks==[[0,1],[2],[3]]
    assert sum(chunks,[])==list(range(4))
    values = [history_rng(99,2,'a',i).random() for i in range(6)]
    assert values==[history_rng(99,2,'a',i).random() for i in range(6)]
    assert len(set(values))==6
    assert values!=[history_rng(99,3,'a',i).random() for i in range(6)]


def write_plan(root,records,c,*,training):
    from maidata.sampling.plan import chart_windows,make_occurrences,write_plan as save_plan,CONFIG_KEYS
    windows = [w for r in records for w in chart_windows(r,config=c)]
    # This fixture deliberately includes additional occurrences like the real plan.
    for w in windows:
        w['complex'] = not w['extra_tail']
    multiplier = c['complex_rhythm_multiplier'] if training else 1
    occurrences = make_occurrences(windows,multiplier=multiplier,seed=c['decoder_train_seed'],epoch=1)
    groups = []
    for variant in dict.fromkeys(o['audio_variant'] for o in occurrences):
        local = [o for o in occurrences if o['audio_variant']==variant]
        source = next(r['source'] for r in records if r['source']['audio_variant']==variant)
        groups.append(dict(audio_variant=variant,audio_asset=variant,chart_ids=list(dict.fromkeys(o['chart_id'] for o in local)),
            encoder_starts_beats=[0],draft_end_beat=2*c['target_beats'],audio_end_seconds=1.2,timing=source['timing'],
            occurrences=len(local),occurrence_start=local[0]['sequence_index'],occurrence_stop=local[-1]['sequence_index']+1))
    p = dict(schema='maidata.window-sampling-plan.v1',view='original',seed=c['decoder_train_seed'],epoch=1,
        policy=dict(views=['original'],split='train' if training else 'validation'),
        config=dict({k:c[k] for k in CONFIG_KEYS},complex_rhythm_multiplier=multiplier),
        charts=[dict(chart_id=r['source']['chart_id'],audio_variant=r['source']['audio_variant'],
                     song_id=r['source']['song_id'],chart_cache_path=f"charts/{r['source']['song_id']}/{r['source']['difficulty']}.json.gz") for r in records],
        audio_groups=groups,windows=windows,occurrences=occurrences,totals=dict(base_windows=len(windows)),
        sources=dict(chart_recipe='fields-v2'))
    save_plan(p,root)


def test_real_dataset_reads_fields_supports_and_plan_without_rebuilding(config,tmp_path,monkeypatch):
    import numpy as np
    from maidata.decoder_training.plan import DecoderPlan
    tokens = [Moment(0),Note.make('tap','A1'),Moment(500),Note.make('hold','A2'),HoldDuration(2)]
    records,hashes = [],{}
    c = dict(config,decoder_train_chart_cache=str(tmp_path/'fields'),decoder_train_physical_cache=str(tmp_path/'physical'),
        decoder_train_audio_cache=str(tmp_path/'audio'),decoder_train_source_root=str(tmp_path/'source'))
    for i in (0,1):
        r,s,g = fields(monkeypatch,tokens,key=f'{i}:5')
        r['source']['level_text']='14.5'
        records.append(r)
        path = f'charts/{i}/5.json.gz'
        hashes[f'{i}:5'] = save_chart(tmp_path/'fields'/path,r)['sha256']
        s['source_sha256'] = hashes[f'{i}:5']
        save_supports(tmp_path/'physical'/path,s)
        (tmp_path/'audio/grids').mkdir(parents=True,exist_ok=True)
        np.savez(tmp_path/f'audio/grids/{i}.npz',**audio_item(i,c)['grid'])
    write_plan(tmp_path/'plan',records,c,training=True)
    plan = DecoderPlan(tmp_path/'plan',c,training=True)
    cached = {}
    for i in (0,1):
        item = audio_item(i,c)
        item['windows'] = [dict(identity=(r['window_id'],r['occurrence']),chart_id=r['chart_id'],start=r['start_beat'])
                           for r in plan.rows(str(i),2)]
        cached[str(i)] = freeze_item(item,c)
    class Cache:
        def read(self,variant):
            return cached[variant]['dp']
    from maidata.decoder_training.export_windows import compile_window
    from maidata.decoder.windows import real_onsets
    compiled = {}
    for record in records:
        key = record['source']['chart_id']
        from maidata.chart_cache.training import load_supports
        chart = ChartSupports(record,load_supports(tmp_path/'physical'/f'charts/{key[0]}/5.json.gz'),g)
        onsets = real_onsets(chart.tokens,chart.timing,sequence_start_tick=chart.start_tick)
        for row in plan.windows:
            if row['chart_id']==key:
                compiled[row['window_id']] = compile_window(chart,row,c,1.2,onsets)
    class Windows:
        def read(self,key): return compiled[key]
    dataset = DecoderDataset(plan,c,hashes,Cache(),Windows())
    with patch('maidata.chart_cache.training.build_supports',side_effect=AssertionError('no cache rebuild')):
        one = dataset[('0',2)]
        again = dataset[('0',2)]
    assert [w['identity'] for w in one['windows']]==[w['identity'] for w in again['windows']]
    assert [w['history_dropped'] for w in one['windows']]==[w['history_dropped'] for w in again['windows']]
    assert len(one['windows'])==3
    batch = collate_audio([one,dataset[('1',2)]],config=c)
    assert len(batch.windows)==6 and len(batch.audio_variants)==2
    plan.occurrences(3)
    assert sum(len(p) for step in plan.batches(3,c) for p in step)==6
    from collections import Counter
    from maidata.decoder_training.data import WindowBatchSampler
    from maidata.decoder_training.batching import batch_cost
    small = dict(c,decoder_train_row_budget=20)
    sampler = WindowBatchSampler(plan,small)
    sampler.epoch = 3
    seen = []
    for requests in sampler:
        assert len({r[0] for r in requests})==len(requests)
        parts = [dataset[r] for r in requests]
        batch = collate_audio(parts,config=small)
        seen.extend(w['identity'] for w in batch.windows)
        assert batch.budget_cost['row_tokens']<=20 or len(batch.windows)==1
        ordered = sorted([w for p in parts for w in p['windows']],key=lambda w:w['sequence_index'])
        lengths = []
        for w in ordered:
            target,history = plan._lengths[w['identity'][0]]
            length = target+(0 if w['history_dropped'] else history)
            assert length==w['length']
            lengths.append(length)
        context = c['audio_before_beats'] + c['target_beats'] + c['audio_after_beats']
        cost = batch_cost([len({w['start'] for w in p['windows']})*context*c['fpn_rates'][0]
                           for p in parts],lengths,small)
        assert cost==batch.budget_cost
    expected = [(r['window_id'],r['occurrence']) for r in plan.occurrences(3)]
    assert seen==expected
    original = [(r['window_id'],r['occurrence']) for r in plan._original_occurrences(3)]
    assert Counter(seen)==Counter(original) and seen!=original
    assert expected==[(r['window_id'],r['occurrence']) for r in plan.occurrences(3)]
    assert expected!=[(r['window_id'],r['occurrence']) for r in plan.occurrences(4)]
    cached['0']['dp']['chart_conditions']['0:5']['level_text'] = '14.50'
    dataset[('0',2)]  # Display formatting does not change the numeric D condition.
    cached['0']['dp']['chart_conditions']['0:5']['level'] = 15.
    dataset[('0',2)]  # Accepted producer metadata is not revalidated per batch.
    bad = deepcopy(plan.manifest)
    bad['config']['chart_history_beats']+=1
    (tmp_path/'plan/plan.json').write_text(json.dumps(bad))
    assert DecoderPlan(tmp_path/'plan',c,training=True).manifest == bad


class EpochDataset(torch.utils.data.Dataset):
    def __init__(self,items,config,plan):
        self.items,self.config,self.training = items,config,plan.training
        self.orders = {e:plan.occurrences(e) for e in range(1,config['decoder_train_scheduler_epochs']+1)}
    def __len__(self):
        return len(self.items)
    def __getitem__(self,index):
        variant,epoch,selection = index
        rows = self.orders[epoch]
        selected = {(rows[i]['window_id'],rows[i]['occurrence']):i for i in selection}
        item = dict(self.items[int(variant)])
        item['windows'] = []
        c = self.config
        tokens = (Moment(0),Note.make('slide','A2'),Route(0,1),SlideDuration(24),Moment(500),Note.make('tap','A1'))
        for original in self.items[int(variant)]['windows']:
            identity,occurrence = original['identity']
            if original['identity'] not in selected:
                continue
            old = original['example']
            w = build_window(tokens,original['timing'],sequence_start_tick=0,start_tick=old.start_tick,end_tick=old.end_tick,
                audio_after_beats=1,audio_end_seconds=1.2,history_beats=4,history_dropout=c['decoder_history_dropout'],
                training=self.training,rng=history_rng(c['decoder_train_seed'],epoch,identity,occurrence))
            item['windows'].append(tensor_window(dict(original,example=w,sequence_index=selected[original['identity']],
                stages=(None,)*w.boundary_index+original['stages'][old.boundary_index:])))
        return item


@pytest.mark.parametrize('workers', [0, 2])
def test_runner_exact_epoch_resume_dropout_validation_and_failure(config,tmp_path,monkeypatch,workers):
    from functools import partial
    from torch.utils.data import DataLoader
    from maidata.training.data import worker_init
    from maidata.decoder_training.data import WindowBatchSampler
    from maidata.decoder_training.batching import occurrence_batches
    from maidata.decoder_training.recipe import PATH_KEYS
    c = dict(config,decoder_dropout=.1,decoder_history_dropout=.5,decoder_train_stop_after_epochs=6,
             decoder_train_workers=workers,decoder_train_row_budget=24)
    c.update({k:str(tmp_path/k) for k in PATH_KEYS})
    c['decoder_train_init_checkpoint'] = None
    items = [training_item(i,c,monkeypatch) for i in range(2)]
    class Plan:
        def __init__(self,root,c,*,training):
            self.groups={'0':{},'1':{}}
            self.training=training
            self.manifest={'epoch':1}
        def batches(self,e,c):
            rows = self.occurrences(e)
            # Use complete freshly formed history/dropout input lengths.
            data = EpochDataset(items,c,self)
            lengths = [len(data[(r['audio_variant'],e,(r['sequence_index'],))]['windows'][0]['example'].inputs)
                       for r in rows]
            audio = {str(i):len(item['coverage'].starts(include_tail=False))*c['target_beats']*c['fpn_rates'][0]
                     for i,item in enumerate(items)}
            return occurrence_batches(rows,audio,lengths,c)
        def occurrences(self,e):
            rows = [dict(window_id=w['identity'][0],occurrence=w['identity'][1],audio_variant=str(i))
                    for i in range(2) for w in items[i]['windows'] if self.training or not w['identity'][1]]
            if self.training:
                random.Random(f'decoder-windows:{c["decoder_train_seed"]}:{e}').shuffle(rows)
            return [dict(row,sequence_index=i) for i,row in enumerate(rows)]
    def loader(p,c,hashes,cache,windows):
        return DataLoader(EpochDataset(items,c,p),batch_sampler=WindowBatchSampler(p,c),
            collate_fn=partial(collate_audio,config=c,packed=True),generator=torch.Generator().manual_seed(71),
            num_workers=workers,persistent_workers=workers > 0,worker_init_fn=worker_init)
    from types import SimpleNamespace
    monkeypatch.setattr(runner,'FrozenCache',lambda *a:SimpleNamespace(producer={'sha256':'fixed-producer'},binding={'p0':'p','dp':'d'}))
    monkeypatch.setattr(runner,'WindowCache',lambda *a:SimpleNamespace(binding='windows'))
    monkeypatch.setattr(runner,'DecoderPlan',Plan)
    monkeypatch.setattr(runner,'make_loader',loader)
    monkeypatch.setattr(checkpoint,'data_binding',lambda *a: (_ for _ in ()).throw(AssertionError('startup verification')))
    monkeypatch.setattr(checkpoint,'data_provenance',lambda *a:{'fixed':'inputs'})
    monkeypatch.setattr(checkpoint,'source_identity',lambda:{'fixed':'source'})
    monkeypatch.setattr(checkpoint,'cache_producer',lambda identity,*a:identity)
    monkeypatch.setattr(runner,'load_route_distances',lambda p:distances())
    full = dict(c,decoder_train_output=str(tmp_path/'full'))
    runner.run(full)
    interrupted = dict(c,decoder_train_output=str(tmp_path/'interrupted'))
    steps = [len(Plan(None,c,training=True).batches(e,c)) for e in range(1,7)]
    actual_update = runner.update
    calls = 0
    def fail(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls==sum(steps[:2])+1:
            raise RuntimeError('interruption')
        return actual_update(*args,**kwargs)
    monkeypatch.setattr(runner,'update',fail)
    with pytest.raises(RuntimeError,match='interruption'):
        runner.run(interrupted)
    path = tmp_path/'interrupted/checkpoints/last.ckpt'
    assert torch.load(path,weights_only=False)['completed_epochs']==2
    assert sorted(p.name for p in path.parent.iterdir())==['last.ckpt']
    monkeypatch.setattr(runner,'update',actual_update)
    runner.run(interrupted,resume=path)
    a = torch.load(tmp_path/'full/checkpoints/last.ckpt',weights_only=False)
    b = torch.load(path,weights_only=False)
    for k in a['model_state']:
        torch.testing.assert_close(a['model_state'][k],b['model_state'][k],rtol=0,atol=0)
    assert a['scheduler_state']==b['scheduler_state']
    assert a['optimizer_state']['param_groups']==b['optimizer_state']['param_groups']
    for key,state in a['optimizer_state']['state'].items():
        for name,value in state.items():
            other = b['optimizer_state']['state'][key][name]
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,other,rtol=0,atol=0)
            else:
                assert value==other
    assert a['step']==b['step']==sum(steps)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['rng']['python']==b['rng']['python']
    import numpy as np
    np.testing.assert_equal(a['rng']['numpy'],b['rng']['numpy'])
    for x,y in zip(a['loader_rng'],b['loader_rng']):
        assert torch.equal(x,y)
    metrics = [json.loads(s) for s in (tmp_path/'interrupted/metrics.jsonl').read_text().splitlines()]
    assert [m['epoch'] for m in metrics if 'validation' in m]==[5,6]
    assert [m['epoch'] for m in metrics if 'diagnostics' in m.get('validation',{})]==[6]
    continuous = [json.loads(s) for s in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert [m['train'] for m in continuous]==[m['train'] for m in metrics]
    assert metrics[-1]['validation']['diagnostics']['history_dropped']==0
    assert runner.run(interrupted,resume=path)['step']==sum(steps)
    assert runner.run(dict(interrupted,decoder_history_dropout=.2),resume=path)['step']==sum(steps)
    monkeypatch.setattr(runner,'FrozenCache',lambda *a:SimpleNamespace(producer={'sha256':'fixed-producer'},
        binding={'p0':'p','dp':'changed'}))
    monkeypatch.setattr(checkpoint,'source_identity',lambda:{'changed':'source'})
    assert runner.run(interrupted,resume=path)['step']==sum(steps)


@pytest.mark.parametrize('schema',['maidata.encoder-draft-training.v3'])
def test_producer_load_is_paired_strict_and_stage_two_resume_is_separate(config,tmp_path,monkeypatch,schema):
    from maidata.training.network import Pretrainer
    from maidata.training import checkpoint as pretrain
    upstream = Pretrainer(config)
    source = pretrain.source_identity()
    binding = dict(pretrain_chart_cache={'catalog.json':'fields'},pretrain_audio_cache={'catalog.json':'audio'})
    payload = dict(schema=schema,recipe=pretrain.recipe(config),source=source,binding=binding,
        completed_epochs=1,step=1,steps_by_epoch=[1]*6,scheduler_state={'last_epoch':1},model_state=upstream.state_dict())
    path = tmp_path/'upstream.ckpt'
    torch.save(payload,path)
    c = dict(config,decoder_train_checkpoint=str(path))
    model = FrozenProducer(c)
    identity = checkpoint.load_producer(model,c,binding)
    assert identity['sha256']==checkpoint.digest_file(path)
    with patch('torch.load',side_effect=AssertionError('training must not load producer weights')):
        assert checkpoint.cache_producer(identity,c,binding)==identity
    with patch.object(checkpoint,'digest_file',side_effect=AssertionError('training hash')):
        assert checkpoint.cache_producer(dict(identity,sha256='old'),c,binding)['sha256']=='old'
        assert checkpoint.cache_producer(identity,dict(c,draft_level_scale=8),binding)==identity
    for name in ('encoder','draft','planner'):
        for a,b in zip(getattr(model,name).parameters(),getattr(upstream,name).parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    with pytest.raises(ValueError,match='architecture'):
        checkpoint.load_producer(model,dict(c,draft_level_scale=8),binding)
    with pytest.raises(ValueError,match='normalization'):
        checkpoint.load_producer(model,c,dict(binding,pretrain_audio_cache={'catalog.json':'other'}))
    payload['schema']=checkpoint.SCHEMA
    torch.save(payload,path)
    with pytest.raises(ValueError,match='stage-I'):
        checkpoint.load_producer(model,c,binding)


def test_binding_checks_selected_producers_without_opening_old_payloads(config,tmp_path):
    from types import SimpleNamespace
    from maidata.decoder_training.recipe import PATH_KEYS
    c = dict(config,**{k:str(tmp_path/k) for k in PATH_KEYS})
    def save(path,data):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data),encoding='utf-8')
    rows = [dict(chart_id=f'{i}:5',song_id=str(i),path=f'charts/{i}/5.json.gz',audio_variant=str(i)) for i in range(2)]
    chart_root = Path(c['decoder_train_chart_cache'])
    save(chart_root/'catalog.json',dict(schema='maidata.full-chart-original.v2',view='original',views=['original'],
        producer_recipe='fields-v2',charts=rows))
    save(chart_root/'production-summary.json',dict(results=[dict(chart_id=r['chart_id'],status='success',sha256=f'hash-{i}')
                                                           for i,r in enumerate(rows)]))
    save(Path(c['decoder_train_audio_cache'])/'catalog.json',dict(variants={str(i):dict(asset=str(i),timing={},
        samples_per_beat=c['samples_per_beat']) for i in range(2)}))
    save(Path(c['decoder_train_audio_cache'])/'production-summary.json',{})
    physical_root = Path(c['decoder_train_physical_cache'])
    save(physical_root/'catalog.json',dict(schema='maidata.training-physical-original.v2',views=['original'],
        source_producer_recipe='fields-v2',producer_recipe='physical-v2',charts=rows))
    results = [dict(chart_id=r['chart_id'],path=r['path'],status='success',source_sha256=f'hash-{i}') for i,r in enumerate(rows)]
    save(physical_root/'production-summary.json',dict(recipe='physical-v2',results=results))
    for name in ('paths.json','runtime-path-semantics.json','runtime-path-rules.json'):
        save(Path(c['decoder_train_geometry'])/name,{})
    for name in ('slide_data','route_distances'):
        save(c['decoder_train_'+name],{})
    plans = []
    for i,r in enumerate(rows):
        root = tmp_path/f'plan-{i}'
        for name in ('plan.json','windows.jsonl','occurrences.jsonl'):
            save(root/name,{})
        plans.append(SimpleNamespace(root=root,manifest=dict(sources=dict(chart_recipe='fields-v2'),
            files=dict(windows='windows.jsonl',occurrences='occurrences.jsonl')),
            groups={str(i):dict(audio_asset=str(i),timing={})},charts={r['chart_id']:dict(r,chart_cache_path=r['path'])}))
    binding,hashes = checkpoint.data_binding(plans,c)
    assert hashes=={'0:5':'hash-0','1:5':'hash-1'}
    assert binding['physical'] and binding['geometry']
    assert not list(chart_root.glob('charts/*'))  # metadata-only binding, no payload scan
    results[0]['source_sha256']='wrong-producer'
    save(physical_root/'production-summary.json',dict(recipe='physical-v2',results=results))
    with pytest.raises(ValueError,match='producer binding'):
        checkpoint.data_binding(plans,c)


def test_gt_targets_do_not_enter_cached_conditions(config,monkeypatch):
    batch = collate_audio([training_item(0,config,monkeypatch)],config=config)
    model = DecoderTrainer(config).train()
    first = model.conditions(batch)
    # Current targets and GT histories never enter E or fixed D/P conditions.
    changed = replace(batch,windows=[])
    second = model.conditions(changed)
    torch.testing.assert_close(first.audio.values,second.audio.values,rtol=0,atol=0)
    torch.testing.assert_close(first.decoder_z.values,second.decoder_z.values,rtol=0,atol=0)
    torch.testing.assert_close(first.plan,second.plan,rtol=0,atol=0)
    assert not first.plan.requires_grad
    assert not first.decoder_z.values.requires_grad
    assert first.audio.values.requires_grad


def test_required_recipe_is_not_filled_from_pretraining_defaults(config):
    from maidata.entry import CONFIG
    assert CONFIG['decoder_train_workers']==12
    assert CONFIG['decoder_train_warmup_steps']==200
    assert CONFIG['decoder_train_gradient_accumulation_steps']==1
    assert CONFIG['decoder_train_row_budget']==8192
    assert 'decoder_train_audio_token_weight' not in CONFIG and 'decoder_train_chart_token_weight' not in CONFIG
    assert CONFIG['decoder_train_learning_rate'] is None and CONFIG['decoder_train_checkpoint'] is None
    with pytest.raises(ValueError,match='explicit Decoder training recipe'):
        recipe.validate(CONFIG)


def test_joint_training_does_not_require_export_only_paths(config,tmp_path):
    from maidata.decoder_training.recipe import PATH_KEYS
    c = dict(config,**{k:str(tmp_path/k) for k in PATH_KEYS})
    c.update(decoder_train_checkpoint=None,decoder_train_p0_cache=None,decoder_train_init_checkpoint=None)
    recipe.validate(c)
    assert 'decoder_train_checkpoint' not in checkpoint.data_provenance(c)['paths']
    assert 'decoder_train_p0_cache' not in checkpoint.data_provenance(c)['paths']


def test_row_budget_ignores_audio_preserves_padding_oversize_and_final_accumulation(config):
    from maidata.decoder_training.batching import occurrence_batches,batch_cost
    rows = [dict(audio_variant=v,window_id=str(i),occurrence=0) for i,v in enumerate(('a','a','b','c','a'))]
    c = dict(config,decoder_train_row_budget=32,decoder_train_gradient_accumulation_steps=2)
    steps = occurrence_batches(rows,dict(a=10000,b=14000,c=50000),[4,9,4,40,5],c)
    assert steps==[[rows[:3],[rows[3]]],[[rows[4]]]]
    assert batch_cost([10000],[4,9],c)['row_tokens']==18
    assert batch_cost([10000,14000],[4,4],c)['row_tokens']==8
    assert not batch_cost([50000],[3],c)['oversized']
    assert batch_cost([1],[40],c)['oversized']
    assert len(occurrence_batches(rows,dict(a=10,b=14,c=50),[4,9,4,40,5],c)[-1])==1


@pytest.mark.parametrize('dropout',[0.,.15,1.])
def test_length_index_matches_real_inputs_with_negative_start_chain_and_tail(config,monkeypatch,dropout):
    from maidata.decoder_training.lengths import chart_lengths
    tokens = [Moment(0),Note.make('slide','A2'),Route(0,1),SlideDuration(24),
              Moment(500),Note.make('hold','A1'),HoldDuration(2),Moment(500),Note.make('touch','C')]
    record,_,_ = fields(monkeypatch,tokens,audio_end=4)
    rows = [dict(window_id=str(start),start_tick=start,end_tick=start+288) for start in (-96,0,96,288,576)]
    lengths = chart_lengths(record,rows,4)
    timing = TimingMap.from_ir(record['metadata'])
    for row in rows:
        w = build_window(tokens,timing,sequence_start_tick=0,start_tick=row['start_tick'],end_tick=row['end_tick'],
            audio_after_beats=1,audio_end_seconds=4,history_beats=4,history_dropout=dropout,
            training=True,rng=history_rng(19,2,row['window_id'],0))
        target,history = lengths[row['window_id']]
        assert target+(0 if w.history_dropped else history)==len(w.inputs)
