"""Exact-source compilation, local Physical reuse and deferred boundary records."""
from dataclasses import replace
from unittest.mock import patch
import random
import json
from types import SimpleNamespace

import pytest
import torch

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.decoder.windows import real_onsets, build_window
from maidata.decoder.supports import window_stages
from maidata.decoder_training.export_windows import compile_window
from maidata.decoder_training.support_cache import ChartSupports
from maidata.decoder_training.window_cache import (tensor_pack, tensor_unpack, select_window,
    collate_prepared, UnsupportedDurationBoundary)
from maidata.physical import PhysicalState
from maidata.physical.attribution import AttributionState
from maidata.physical.attribution import Evaluation, Evidence
from maidata.physical.conflicts import Conflict
from maidata.time import TimingMap
from test_decoder_supervision import geometry as synthetic_geometry
from maidata.chart import EncodedChart, decode_chart
from maidata.chart_cache import produce
from maidata.chart_cache.training import build_supports
from maidata.chart_cache.cache import save_chart
from maidata.chart_cache.training import save_supports
from test_decoder_training import fields as field_fixture


def config():
    return dict(target_beats=2, audio_after_beats=2, chart_history_beats=4,
                decoder_history_dropout=.15)


def window(chart, start=0):
    return build_window(chart.tokens, chart.timing, sequence_start_tick=chart.start_tick,
        start_tick=start, end_tick=start+192, audio_after_beats=2, audio_end_seconds=12.,
        history_beats=4, history_dropout=0., training=False, rng=random.Random(0))


def test_local_cap_cached_intervals_and_ledger_match_rule_oracle(monkeypatch):
    tokens = [Moment(0), Note.make('hold','A1'), HoldDuration(48),
        Note.make('slide','A2'), Route(0,1), Route(1), SlideDuration(48),
        Moment(500), Note.make('slide','A3'), Route(70,0), SlideDuration(24),
        Moment(500), Note.make('tap','A1')]
    record, supports, geometry = field_fixture(monkeypatch, tokens, audio_end=12., bpm=60)
    chart = ChartSupports(record, supports, geometry)
    example = window(chart)
    cap = chart.timing.second(example.cap_tick)
    got, audit = chart.window(example, duration_cap_seconds=cap)
    oracle = AttributionState(PhysicalState(chart.timing, geometry, audio_end_seconds=12.))
    oracle.restore(chart.before_window(example.start_tick))
    oracle.duration_end_seconds = cap
    want = window_stages(example, oracle)
    assert got == want
    assert audit['reused'] > 0
    assert chart.counterfactuals  # Only absent duration/WiFi candidates were filled.
    assert chart.state.duration_end_seconds is None


def test_compiler_keeps_exact_ticks_query_cost_and_dropout_coordinates(monkeypatch):
    tokens = [Moment(0), Note.make('hold','A1'), HoldDuration(385),
              Moment(5000), Note.make('tap','A2')]
    record, supports, geometry = field_fixture(monkeypatch, tokens, audio_end=12., bpm=60)
    chart = ChartSupports(record, supports, geometry)
    c = dict(config(), audio_after_beats=4)
    row = dict(chart_id='0:5',window_id='0:5:0',start_tick=0,end_tick=192)
    onsets = real_onsets(chart.tokens, chart.timing, sequence_start_tick=0)
    with patch('maidata.time.duration_cells', side_effect=AssertionError('old four-tick labels')):
        value = compile_window(chart, row, c, 12., onsets)
    restored = tensor_unpack(tensor_pack(value))
    exact = restored['metadata']['exact_fields']
    assert next(x for x in exact if x['target'].get('type')=='HOLD_DURATION')['target']['ticks'] == 385
    group = value['labels']['groups']['HOLD_DURATION']
    assert group['target'][0,0,384] and group['target'][0,1,0]
    assert value['metadata']['field_query_cost'] > len(value['next_types'])
    normal = select_window(restored, False)
    dropped = select_window(restored, True)
    _, _, labels = collate_prepared([normal,dropped])
    assert labels.groups['HOLD_DURATION']['allowed'].shape[-1] == 385
    assert labels.counts['main'] == value['labels']['counts']['main']*2
    assert torch.equal(labels.groups['HOLD_DURATION']['query_beats'][0],
                       labels.groups['HOLD_DURATION']['query_beats'][1])


def test_beyond_local_gt_is_preserved_and_explicitly_untrainable(monkeypatch):
    tokens = [Moment(0), Note.make('hold','A1'), HoldDuration(900)]
    record, supports, geometry = field_fixture(monkeypatch, tokens, audio_end=12., bpm=60)
    chart = ChartSupports(record, supports, geometry)
    value = compile_window(chart, dict(chart_id='0:5',window_id='deferred',start_tick=0,end_tick=192),
                           config(),12.,real_onsets(chart.tokens,chart.timing,sequence_start_tick=0))
    assert value['labels'] is None
    assert value['metadata']['unsupported_duration'][0]['exact_ticks'] == 900
    assert value['deferred_supports'][1]['duration']['domain'][1] > 900
    assert value['metadata']['exact_fields'][1]['target']['ticks'] == 900
    with pytest.raises(UnsupportedDurationBoundary, match='deferred'):
        select_window(tensor_unpack(tensor_pack(value)), False)


def test_cap_removes_unique_legal_completion_for_duration_type_and_wifi():
    timing = TimingMap([(0,60)])
    state = AttributionState(PhysicalState(timing, synthetic_geometry(), audio_end_seconds=12.))
    state.inject_boundary(Moment(0), start_tick=0)
    state.physical.accept_source(Note.make('slide','A1'))
    witness = Conflict('slide_too_fast','counterfactual',(('track',0,0,0),))
    bad = Evaluation(Evidence(conflicts=frozenset({witness})),
                     Evidence(conflicts=frozenset({witness}))).record()
    support = dict(domain=[1,96], intervals=[[1,95,bad],[96,96,Evaluation().record()]],
                   outside_basic='duration_outside_audio')
    state.duration_support_provider = lambda _: support
    # The only legal completion is 96; removing it changes WiFi's projected
    # route support. Candidate projection never appends a physical action.
    before = state.snapshot()
    assert not state.query(Route(70,0)).raw.conflicts
    state.duration_end_seconds = timing.second(96)+95*timing.written_tick_seconds(0)
    assert state.query(Route(70,0)).raw.conflicts == frozenset({witness})
    assert state.snapshot() == before
    state.physical.accept_source(Route(0,0))
    assert state.query_types()['SLIDE_DURATION'].penalty.conflicts == frozenset({witness})
    # First responsibility moves to the type; the full time support retains
    # its raw violation while no longer charging that same witness again.
    state.charged |= frozenset({witness})
    values = state._duration_values()['intervals']
    assert all(v.raw.conflicts and not v.penalty.conflicts for _,_,v in values)


def test_note_minimum_hold_proxy_is_reprojected_at_sharp_bpm_boundary(monkeypatch):
    geometry = synthetic_geometry()
    monkeypatch.setattr(produce, 'GEOMETRY', geometry)
    metadata = dict(schema='maimai.chart_ir.v1',tick_rate=180,first_seconds=0., controls=[
        dict(kind='bpm',bpm=60,position_beats=dict(num=0,den=1)),
        dict(kind='bpm',bpm=60000,position_beats=dict(num=2,den=1))])
    timing = TimingMap([(0,60),(2,60000)])
    tokens = (Moment(0),Moment(timing.delta_ms(0,191)),Note.make('tap','A1'))
    row = dict(chart_id='0:5',song_id='0',difficulty='5',audio_variant='0',
        path='charts/0/5.json.gz',timing=dict(bpm=[[0,1,60],[2,1,60000]],first_seconds=0.),
        audio_end_seconds=12.)
    record = produce.build_record(decode_chart(EncodedChart(tokens,metadata),geometry),row)
    supports = build_supports(record,geometry)
    chart = ChartSupports(record,supports,geometry)
    example = window(chart)
    assert not supports['stages'][1]['kind']['hold']['basic']
    got,_ = chart.window(example,duration_cap_seconds=timing.second(example.cap_tick))
    oracle = AttributionState(PhysicalState(chart.timing,geometry,audio_end_seconds=12.))
    oracle.restore(chart.before_window(0))
    oracle.duration_end_seconds = timing.second(example.cap_tick)
    assert got == window_stages(example,oracle)
    note = next(s for s in got if s and 'kind' in s)
    assert note['kind']['hold']['basic'] == ['duration_outside_local_audio']
    assert not note['kind']['tap']['basic']


def test_synthetic_incremental_export_uses_exact_source_without_old_tensor_io(monkeypatch,tmp_path):
    """Synthetic manifests test orchestration, not acceptance of a real old asset."""
    from maidata.decoder_training import export_windows
    from maidata.decoder_training.window_cache import SCHEMA
    record, supports, geometry = field_fixture(monkeypatch,
        [Moment(0),Note.make('hold','A1'),HoldDuration(385)],audio_end=12.,bpm=60)
    supports['source_sha256'] = 'synthetic-source'
    fields = tmp_path/'fields'
    physical = tmp_path/'physical'
    save_chart(fields/'charts/0/5.json.gz',record)
    save_supports(physical/'charts/0/5.json.gz',supports)
    rows = [dict(chart_id='0:5',window_id='0:5:0',audio_variant='0',start_tick=0,end_tick=192),
            dict(chart_id='0:5',window_id='0:5:1',audio_variant='0',start_tick=192,end_tick=384)]
    train = SimpleNamespace(charts={'0:5':dict(chart_cache_path='charts/0/5.json.gz')},windows=rows,
        groups={'0':dict(audio_end_seconds=12.)})
    validation = SimpleNamespace(charts={},windows=[],groups={})
    monkeypatch.setattr('maidata.decoder_training.plan.DecoderPlan',
                        lambda _,c,training: train if training else validation)
    binding = {k:{} for k in ('plans','physical','geometry','slide_data','pretrain_chart_cache')}
    monkeypatch.setattr('maidata.decoder_training.checkpoint.data_binding',
                        lambda plans,c: (binding,{'0:5':'synthetic-source'}))
    monkeypatch.setattr(export_windows.Geometry,'from_resources',lambda *_:geometry)
    class SerialPool:
        def __init__(self,*,max_workers,initializer,initargs):
            initializer(*initargs)
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def map(self,fn,tasks): return map(fn,tasks)
    monkeypatch.setattr(export_windows,'ProcessPoolExecutor',SerialPool)
    old = tmp_path/'old-continuous'
    old.mkdir()
    old_manifest = dict(schema='maidata.decoder-window-tensors.v2',windows={
        row['window_id']:dict(row,path='intentionally-absent-rounded-labels.pt') for row in rows})
    (old/'manifest.json').write_text(json.dumps(old_manifest),'utf-8')
    destination = tmp_path/'new-v3'
    c = dict(config(),audio_after_beats=4,
        decoder_train_train_plan='synthetic-train',decoder_train_validation_plan='synthetic-validation',
        decoder_train_chart_cache=str(fields),decoder_train_physical_cache=str(physical),
        decoder_train_window_cache=str(destination),decoder_train_geometry='synthetic',
        decoder_train_slide_data='synthetic')
    result = export_windows.export(c,workers=12,window_ids=['0:5:0'],source_windows=old)
    assert result['schema'] == SCHEMA and result['complete']
    assert 'target_stride_beats' not in result['contract']['config']
    assert set(result['windows']) == {'0:5:0'}
    value = tensor_unpack(torch.load(destination/result['windows']['0:5:0']['path'],weights_only=True))
    target = value['labels']['groups']['HOLD_DURATION']['target']
    assert target[0,0,384] and target[0,1,0]
    assert json.loads((old/'manifest.json').read_text('utf-8')) == old_manifest
    assert not (old/'intentionally-absent-rounded-labels.pt').exists()
