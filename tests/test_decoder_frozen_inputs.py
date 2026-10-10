"""Offline bank contracts, mixed-window alignment and absence of producer execution."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.decoder_training.frozen_cache import FrozenCache
from maidata.decoder_training.cache_schema import P0_SCHEMA, DP_SCHEMA
from maidata.decoder_training.p0_cache import save_bank
from maidata.decoder_training.dp_cache import save_outputs
from maidata.decoder_training.data import collate_audio
from maidata.decoder_training.network import DecoderTrainer
from maidata.decoder_training import recipe, runner
from maidata.draft.types import BeatMemory
from maidata.training.checkpoint import digest_file
from test_decoder_training import config, encoder_config, training_item, distances


@pytest.fixture
def assets(config,tmp_path,monkeypatch):
    items = [training_item(i,config,monkeypatch,remaster=i==0) for i in range(2)]
    roots = {k:tmp_path/k for k in ('p0','dp')}
    binding = dict(pretrain_chart_cache={'catalog.json':'fields'},pretrain_audio_cache={'catalog.json':'audio'},
        plans=[{'plan.json':f'plan-{i}','windows.jsonl':f'windows-{i}'} for i in range(2)])
    producer = dict(sha256='fixed-producer')
    normalization = {k:binding[k] for k in ('pretrain_chart_cache','pretrain_audio_cache')}
    splits = dict(train=['0'],validation=['1'])
    p0 = dict(schema=P0_SCHEMA,complete=True,producer=producer,normalization=normalization,
              precision='fp32',p0_rate=config['fpn_rates'][0],splits=splits,groups={})
    dp = dict(schema=DP_SCHEMA,complete=True,producer=producer,normalization=normalization,
              precision='fp32',splits=splits,groups={},plans={})
    plans = []
    for i,item in enumerate(items):
        variant,coverage = item['variant'],item['coverage']
        bank = BeatMemory(*(item['p0'][k].unsqueeze(0) for k in ('values','beats','valid')))
        meta = dict(audio_asset='asset-'+variant,timing={},audio_end_seconds=1.2,
            chart_ids=list(item['dp']['chart_to_group']),encoder_starts_beats=list(coverage.starts(include_tail=False)),
            draft_end_beat=coverage.first_beat+coverage.draft_beats)
        row = save_bank(roots['p0']/f'banks/{variant}.pt',bank,0,coverage,variant=variant,
            producer_sha256='fixed-producer',p0_rate=config['fpn_rates'][0])
        p0['groups'][variant] = dict(row,path=f'banks/{variant}.pt',
            **{k:meta[k] for k in ('audio_asset','timing','audio_end_seconds')})
        cached, planner = item['dp'],item['dp']['planner']
        batch = dict(planner,variant=variant,conditions=cached['conditions'],chart_to_group=cached['chart_to_group'])
        z = BeatMemory(*(cached['z'][k] for k in ('values','beats','valid')))
        dp['groups'][variant] = save_outputs(roots['dp']/f'banks/{variant}.pt',batch,z,
            SimpleNamespace(plan_values=planner['plan_values'],valid=planner['valid']),'fixed-producer')
        windows = [dict(window_id=key,chart_id=chart,start_beat=float(start),audio_variant=variant)
                   for key,chart,start in zip(planner['window_ids'],planner['chart_ids'],planner['starts'])]
        plans.append(SimpleNamespace(groups={variant:meta},windows=windows,manifest={'files':{'windows':'windows.jsonl'}}))
        dp['plans'][('train','validation')[i]] = binding['plans'][i]
    (roots['p0']/'manifest.json').write_text(json.dumps(p0))
    dp['p0_manifest_sha256'] = digest_file(roots['p0']/'manifest.json')
    (roots['dp']/'manifest.json').write_text(json.dumps(dp))
    c = dict(config,decoder_train_p0_cache=str(roots['p0']),decoder_train_dp_cache=str(roots['dp']))
    return c,plans,binding,items,roots


def test_read_cache_and_select_interleaved_occurrences_without_building_edp(assets):
    c,plans,binding,items,roots = assets
    # Startup reads selected manifests without bank scans or input verification.
    with patch('torch.load',side_effect=AssertionError('startup cannot load banks')):
        cache = FrozenCache(plans,c,binding)
    selected = []
    for i,item in enumerate(items):
        p0,dp = cache.read(item['variant'])
        windows = [dict(w,sequence_index=2*j+i) for j,w in enumerate(item['windows'])]
        selected.append(dict(variant=item['variant'],p0=p0,dp=dp,windows=windows))
    packet = collate_audio(selected,config=c,packed=True)
    raw = packet.to('cpu')
    expected = sorted([w for item in selected for w in item['windows']],key=lambda w:w['sequence_index'])
    assert [w['identity'] for w in raw.windows]==[w['identity'] for w in expected]
    cond = raw.conditions
    for i,w in enumerate(expected):
        unique = cond.occurrence_mapping[i]
        song = cond.audio_mapping[unique]
        item = selected[song]
        dp = item['dp']
        row = dp['window_index'][w['identity'][0]]
        assert torch.equal(cond.plan[unique],dp['planner']['plan_values'][row])
        z = dp['z']
        group = dp['chart_to_group'][w['chart_id']]
        for j,beat in enumerate(cond.decoder_z.beats[unique]):
            matches = (z['beats'][group]==beat).nonzero().flatten()
            if len(matches):
                assert cond.decoder_z.valid[unique,j]
                assert torch.equal(cond.decoder_z.values[unique,j],z['values'][group,matches[0]])
            else:
                assert not cond.decoder_z.valid[unique,j]
                assert cond.decoder_z.values[unique,j].count_nonzero()==0
    assert len(cond.plan)<len(expected)  # repeated occurrences share cached conditions
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in ('maidata.model.build_model','maidata.model.build_draft','maidata.model.build_planner',
                     'maidata.preparation.cache.load_grid','maidata.decoder_training.checkpoint.load_producer'):
            stack.enter_context(patch(name,side_effect=AssertionError('online producer is forbidden')))
        model = DecoderTrainer(c)
        opt = recipe.optimizer(model.parameters(),c)
        sched = recipe.scheduler(opt,[1]*6,c)
        metrics,audit = runner.update(model,[packet],opt,sched,c,distances(),audit=True)
    assert set(dict(model.named_children()))=={'decoder'}
    assert audit['gradients']['decoder']>0
    assert model.decoder.condition.z.weight.grad.abs().sum()>0
    assert model.decoder.condition.plan[0].weight.grad.abs().sum()>0
    assert metrics.counts['main']>0
    assert not cond.audio.values.requires_grad and not cond.plan.requires_grad


@pytest.mark.parametrize('field',['complete','producer','normalization','p0_manifest_sha256','plans','splits'])
def test_training_does_not_revalidate_accepted_manifest(assets,field):
    c,plans,binding,_,roots = assets
    path = roots['dp']/'manifest.json'
    data = json.loads(path.read_text())
    data[field] = False if field=='complete' else {} if field in ('plans','splits','normalization','producer') else 'wrong'
    path.write_text(json.dumps(data))
    with patch('maidata.training.checkpoint.digest_file',side_effect=AssertionError('startup hash')):
        cache = FrozenCache(plans,c,binding)
        assert cache.read('0')[0]['values'].numel()>0


def test_training_reads_do_not_rescan_accepted_values(assets):
    c,plans,binding,_,roots = assets
    cache = FrozenCache(plans,c,binding)
    with patch('torch.isfinite',side_effect=AssertionError('training payload scan')):
        p0,dp=cache.read('0')
    assert p0['values'].numel()>0 and dp['window_index']


def test_bf16_frozen_values_can_feed_fp32_decoder_without_gradients(assets):
    c,_,_,items,_ = assets
    batch = collate_audio(items,config=c)
    source = batch.conditions
    batch.conditions = replace(source,audio=replace(source.audio,values=source.audio.values.bfloat16()),
        decoder_z=replace(source.decoder_z,values=source.decoder_z.values.bfloat16()),plan=source.plan.bfloat16())
    model = DecoderTrainer(c)
    opt = recipe.optimizer(model.parameters(),c)
    _,audit = runner.update(model,[batch],opt,recipe.scheduler(opt,[1]*6,c),c,distances(),audit=True)
    assert audit['gradients']['decoder']>0
    assert batch.conditions.audio.values.dtype==torch.bfloat16
