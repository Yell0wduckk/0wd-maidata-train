"""Offline window tensors match the independent live-rule oracle in values and gradients."""
from contextlib import ExitStack
from dataclasses import fields
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.decoder.windows import real_onsets
from maidata.decoder.batching import collate_windows
from maidata.decoder.supports import window_stages
from maidata.physical import PhysicalState
from maidata.physical.attribution import AttributionState
from maidata.decoder_training.export_windows import compile_window
from maidata.decoder_training.window_cache import (WindowCache, SCHEMA, contract, tensor_pack,
    tensor_unpack, select_window, collate_prepared)
from maidata.decoder_training import supervision, runner, recipe
from maidata.decoder_training.data import collate_audio
from maidata.decoder_training.network import DecoderTrainer
from test_decoder_training import config, encoder_config, fields as field_fixture, ChartSupports, example, training_item, distances


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_compiled_history_dropout_chain_and_terminal_match_oracle(config, monkeypatch, tmp_path):
    tokens = [Moment(0), Note.make('hold','A1'), HoldDuration(48),
              Note.make('slide','A2'), Route(0,1), Route(1), SlideDuration(24),
              Moment(6000), Note.make('tap','A1',ex=1), Moment(1000), Note.make('touch','C')]
    record, support, geo = field_fixture(monkeypatch,tokens,audio_end=12,bpm=60)
    chart = ChartSupports(record,support,geo)
    onsets = real_onsets(chart.tokens,chart.timing,sequence_start_tick=0)
    cfg = dict(config,target_beats=2,audio_after_beats=1)
    cached, wanted = [], []
    for start, dropped in ((-96,False),(0,False),(384,True),(576,False),(672,True),(1344,False)):
        row = dict(start_tick=start,end_tick=start+192)
        with patch('maidata.decoder.supports.time_candidates', side_effect=AssertionError('dense candidates')), \
             patch('maidata.time.support.duration_cells', side_effect=AssertionError('dense duration')):
            value = compile_window(chart,row,cfg,12,onsets)
        path = tmp_path/f'{start}.pt'
        torch.save(tensor_pack(value),path)
        restored = tensor_unpack(torch.load(path,weights_only=True))
        cached.append(select_window(restored,dropped))
        w = example(chart,start,start+192,drop=dropped,audio_end=12)
        oracle = AttributionState(PhysicalState(chart.timing,geo,audio_end_seconds=12))
        oracle.restore(chart.before_window(start))
        oracle.duration_end_seconds = min(12,chart.timing.second(w.cap_tick))
        wanted.append(dict(example=w,stages=window_stages(w,oracle),timing=chart.timing,audio_end=12))
    tokens, types, actual = collate_prepared(cached)
    expected_tokens, expected_types, _ = collate_windows([w['example'] for w in wanted],device='cpu')
    for f in fields(tokens):
        torch.testing.assert_close(getattr(tokens,f.name),getattr(expected_tokens,f.name),rtol=0,atol=0)
    assert torch.equal(types,expected_types)
    expected = supervision.prepare(wanted)
    assert expected.counts == actual.counts
    assert expected.groups.keys() == actual.groups.keys()
    for key in actual.groups:
        for field, value in actual.groups[key].items():
            other = expected.groups[key][field]
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,other,rtol=0,atol=0)
            else:
                assert value == other
    assert torch.equal(actual.flags['indices'],expected.flags['indices'])
    assert torch.equal(actual.flags['args'],expected.flags['args'])


def test_window_manifest_supplies_lengths_and_reads_only_tensors(config,tmp_path):
    binding = {k: {} for k in ('plans','physical','geometry','slide_data','pretrain_chart_cache')}
    row = dict(window_id='a',chart_id='1:5',audio_variant='1',start_tick=0,end_tick=288)
    plan = SimpleNamespace(windows=[row])
    config = dict(config,decoder_train_window_cache=str(tmp_path))
    meta = dict(schema=SCHEMA,complete=True,contract=contract(config,binding),
        windows={'a':dict({k:v for k,v in row.items() if k!='window_id'},path='a.pt',lengths=[3,2])})
    (tmp_path/'manifest.json').write_text(json.dumps(meta),encoding='utf-8')
    torch.save(tensor_pack(dict(values=torch.arange(7))),tmp_path/'a.pt')
    cache = WindowCache([plan],config,binding)
    assert plan._lengths == {'a':(3,2)}
    with patch('torch.isfinite',side_effect=AssertionError('runtime validation')):
        assert torch.equal(cache.read('a')['values'],torch.arange(7))


def test_entire_training_batch_avoids_chart_rules_and_tensor_validation(config,monkeypatch):
    item = training_item(0,config,monkeypatch,remaster=True)
    model = DecoderTrainer(config)
    opt = recipe.optimizer(model.parameters(),config)
    sched = recipe.scheduler(opt,[1]*6,config)
    geometry = distances()
    forbidden = ('maidata.chart_cache.cache.load_chart','maidata.chart_cache.training.load_supports',
        'maidata.chart.grammar.Grammar.accept','maidata.physical.attribution.AttributionState.stage_and_accept',
        'maidata.physical.attribution.AttributionState.terminal_stage',
        'maidata.decoder_training.supervision.prepare','maidata.decoder.supports.time_candidates',
        'maidata.decoder.network.validate_memory','maidata.decoder.attention.validate_memory',
        'maidata.decoder.network.require','maidata.decoder.embedding.require','maidata.decoder.attention.require',
        'maidata.decoder.probability.require')
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch(name,side_effect=AssertionError(name)))
        batch = collate_audio([item],config=config,packed=True)
        result, _ = runner.update(model,[batch],opt,sched,config,geometry)
    assert result.counts['main']>0
    assert model.decoder.layers[0].audio_attn.key.weight.grad.abs().sum()>0
