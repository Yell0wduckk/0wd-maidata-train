"""Early ROUTE gradients and opt-in, non-destructive cache migration."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.chart.routes import route_for
from maidata.chart_cache.training import build_supports, load_supports, save_supports, ROUTE_COVERAGE
from maidata.chart_cache.migrate_route_coverage import migrate_supports
from maidata.decoder.windows import real_onsets
from maidata.decoder_training.export_windows import compile_window
from maidata.decoder_training.support_cache import ChartSupports
from maidata.decoder_training.migrate_physical import replace_physical, equal
from test_decoder_training import fields as field_fixture


def fixture(monkeypatch, *, extension=False):
    tokens = [Moment(0),Note.make('slide','A1'),Route(route_for('1-3'),0),SlideDuration(96),
              Moment(2000),Note.make('slide','A1'),Route(route_for('1-3'),0)]
    if extension:
        tokens.append(Route(route_for('3-5')))
    tokens += [SlideDuration(96),Note.make('hold','A4'),HoldDuration(48),
               Moment(2000),Note.make('tap','A8')]
    with patch('maidata.physical.attribution.first_route_coverage',return_value=frozenset()):
        record, old, geometry = field_fixture(monkeypatch,tokens,audio_end=12.,bpm=60)
    old.pop('route_coverage_attribution')  # The actual pre-fix v2 representation.
    return record,old,geometry


@pytest.mark.parametrize('extension',[False,True])
def test_migration_matches_fresh_oracle_and_deduplicates_witnesses(monkeypatch,extension):
    record,old,geometry = fixture(monkeypatch,extension=extension)
    original = deepcopy(old)
    migrated,audit = migrate_supports(record,old,geometry)
    fresh = build_supports(record,geometry)
    assert migrated == fresh
    assert old == original
    assert 6 in audit['changed_target_indices']
    assert audit['reused_stages'] > 0
    route = migrated['stages'][5]
    chosen = route['route'][route_for('1-3')]
    assert chosen['penalty'] == 4 and not chosen['deferred']
    witnesses = chosen['new_conflicts']
    assert all(w[1]=='path_coverage' and w[3][0]==0 for w in witnesses)
    assert route['attribution']['route'] == witnesses
    following = migrated['stages'][6:8 if extension else 7]
    for stage in following:
        for group in (stage['types'].values(),stage.get('route',[]),
                      [r[2] for r in stage.get('duration',{}).get('intervals',[])]):
            assert all(w not in r['new_conflicts'] for r in group for w in witnesses)
    duration = migrated['stages'][7 if extension else 6]
    assert any(w in r[2]['conflicts'] for r in duration['duration']['intervals'] for w in witnesses)
    again, audit2 = migrate_supports(record,migrated,geometry)
    assert again == migrated and not audit2['changed_target_indices'] and audit2['replayed_stages']==0


def test_route_has_gradient_and_duration_does_not_recharge(monkeypatch):
    from maidata.decoder.supports import categorical_support
    from maidata.decoder.losses import physical_categories
    record,old,geometry = fixture(monkeypatch)
    new,_ = migrate_supports(record,old,geometry)
    records = new['stages'][5]['route']
    allowed,penalty,_ = categorical_support(records,range(71),device='cpu')
    logits = torch.zeros(71,requires_grad=True)
    logp = logits.masked_fill(~allowed,-torch.inf).log_softmax(-1)
    # Real decoder's category penalty primitive supplies route-head pressure.
    loss = physical_categories(logp,penalty).sum()
    loss.backward()
    assert logits.grad[route_for('1-3')] > 0
    assert old['stages'][5]['route'][route_for('1-3')]['penalty']==0


@pytest.mark.parametrize('exempt',[None,'old','new'])
def test_training_boundary_witness_matches_action_without_duration_search(monkeypatch,exempt):
    from maidata.physical import PhysicalState,AttributionState
    from maidata.physical.actions import materialize
    from maidata.physical.conflicts import action_conflicts
    from maidata.chart.types import Completion,Track
    from maidata.time import TimingMap
    _,_,geometry = fixture(monkeypatch)
    timing = TimingMap([(0,60),(Fraction(3,2),120)],first_seconds=.1)
    old = materialize(Completion(8,0,0,Note.make('slide','A1'),
        track=Track((route_for('1-3'),),0,72),track_index=0),timing,geometry)
    # Duration 72 written ticks ends exactly at new onset tick 192 across BPM.
    assert old.motions[0].end == pytest.approx(timing.second(192))
    if exempt=='old':
        old = replace(old,motions=tuple(replace(m,exempt=True) for m in old.motions))
    if exempt=='new':
        geometry.paths['1-3'] = replace(geometry.paths['1-3'],exempt=True)
    physical = PhysicalState(timing,geometry,audio_end_seconds=12.)
    physical.inject_boundary(Moment(0),start_tick=192)
    physical.actions = (old,)
    state = AttributionState(physical)
    state.stage_and_accept(Note.make('slide','A1'))
    saved = state.snapshot()
    token = Route(route_for('1-3'),0)
    with patch('maidata.physical.attribution.duration_support',side_effect=AssertionError('duration search')):
        value = state.query(token)
    assert state.snapshot()==saved
    grammar,_ = physical._trial(token)
    s = grammar.state
    for routes,duration in (((token.route_id,),1),((token.route_id,route_for('3-5')),192)):
        complete = materialize(Completion(s.owner,s.moment,s.cursor,s.note,
            track=Track(routes,0,duration),track_index=s.track_count),timing,geometry)
        expected = {w for w in action_conflicts((old,),complete) if w.rule=='path_coverage' and w.detail[0]==0}
        assert value.raw.conflicts==expected
    assert bool(value.raw.conflicts)==(exempt is None)
    physical.grammar.state = replace(physical.grammar.state,cursor=193)
    assert not state.query(token).raw.conflicts


def test_variable_bpm_branches_wifi_and_unchanged_chart_migration(monkeypatch):
    from maidata.chart import EncodedChart,decode_chart
    from maidata.chart_cache import produce
    from maidata.chart.routes import catalog
    from maidata.physical import Geometry,PathGeometry
    geometry = Geometry({p:PathGeometry(p,1.,((0.,1 << (int(p[0])-1)),))
                         for v in catalog() for p in v['paths_by_start']})
    monkeypatch.setattr(produce,'GEOMETRY',geometry)
    metadata = dict(schema='maimai.chart_ir.v1',tick_rate=180,first_seconds=0,
        controls=[dict(kind='bpm',bpm=bpm,position_beats=dict(num=b,den=1)) for b,bpm in ((0,60),(2,120))])
    tokens = [Moment(0),Note.make('slide','A1'),Route(route_for('1-3'),0),SlideDuration(192),
              Moment(2000),Note.make('slide','A1'),Route(route_for('1-3'),1),Route(route_for('3-5')),
              SlideDuration(96),Route(route_for('1-7'),0),SlideDuration(96),Route(70,0),SlideDuration(96),
              Moment(1000),Note.make('tap','A8')]
    row = dict(chart_id='0:5',audio_end_seconds=12.,timing=dict(bpm=[[0,1,60],[2,1,120]],first_seconds=0))
    record = produce.build_record(decode_chart(EncodedChart(tuple(tokens),metadata),geometry),row)
    with patch('maidata.physical.attribution.first_route_coverage',return_value=frozenset()):
        old = build_supports(record,geometry)
    new,audit = migrate_supports(record,old,geometry)
    assert new == build_supports(record,geometry) and audit['changed_target_indices']
    record,old,geometry = field_fixture(monkeypatch,[Moment(0),Note.make('hold','A1'),HoldDuration(48)],
                                      audio_end=12.,bpm=60)
    old.pop('route_coverage_attribution')
    with patch('maidata.physical.attribution.AttributionState.stage_and_accept',side_effect=AssertionError('replay')):
        new,audit = migrate_supports(record,old,geometry)
    assert not audit['changed_target_indices'] and new['stages']==old['stages']


def test_window_migration_preserves_model_labels_and_old_reader(monkeypatch,tmp_path):
    record,old,geometry = fixture(monkeypatch,extension=True)
    new,_ = migrate_supports(record,old,geometry)
    cfg = dict(target_beats=2,audio_after_beats=2,chart_history_beats=4,decoder_history_dropout=.15)
    charts = [ChartSupports(record,supports,geometry) for supports in (old,new)]
    onsets = real_onsets(charts[0].tokens,charts[0].timing,sequence_start_tick=record['start_tick'])
    for start in (0,192,384,1152):
        row = dict(chart_id='0:5',window_id=f'0:5:{start}',start_tick=start,end_tick=start+192)
        with patch('maidata.physical.attribution.first_route_coverage',return_value=frozenset()):
            before = compile_window(charts[0],row,cfg,12.,onsets)
        fresh = compile_window(charts[1],row,cfg,12.,onsets)
        after,changed = replace_physical(before,fresh)
        assert equal(after['labels'],fresh['labels'])
        assert equal(before['tokens'],after['tokens'])
        assert equal(before['next_types'],after['next_types'])
        if start==192:
            assert changed
            wrong = deepcopy(fresh); wrong['labels']['groups']['route']['target'][0,0] ^= True
            with pytest.raises(ValueError,match='non-Physical'):
                replace_physical(before,wrong)
    # No new minimum-version gate is installed on ordinary v2 readers.
    path = tmp_path/'old.json.gz'
    save_supports(path,old)
    assert load_supports(path)==old


class SerialPool:
    def __init__(self,max_workers,initializer,initargs):
        assert max_workers==12
        initializer(*initargs)
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def map(self,fn,tasks): return map(fn,tasks)


def test_both_migration_entries_resume_and_preserve_source_bytes(monkeypatch,tmp_path):
    from maidata.chart_cache import migrate_route_coverage as pm
    from maidata.decoder_training import migrate_physical as wm
    from maidata.chart_cache.cache import save_chart,write_json
    from maidata.decoder_training.sequence_cache import write_asset,chart_sequence,SCHEMA
    from maidata.decoder_training.window_cache import WindowCache
    record,old,geometry = fixture(monkeypatch)
    field_root,physical,windows = [tmp_path/p for p in ('fields','physical','windows')]
    relative = 'charts/0/5.json.gz'
    field_storage = save_chart(field_root/relative,record)
    old['source_sha256'] = field_storage['sha256']
    physical_storage = save_supports(physical/relative,old)
    row = dict(chart_id='0:5',song_id='0',difficulty='5',path=relative)
    write_json(field_root/'catalog.json',dict(producer_recipe='fields'))
    write_json(field_root/'production-summary.json',dict(results=[dict(row,status='success',**field_storage)]))
    write_json(physical/'catalog.json',dict(schema=pm.SCHEMA,views=['original'],producer_recipe='old',
        source_producer_recipe='fields',charts=[row]))
    write_json(physical/'production-summary.json',dict(recipe='old',results=[
        dict(row,status='success',source_sha256=field_storage['sha256'],**physical_storage)]))
    originals = {p:p.read_bytes() for root in (field_root,physical) for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(pm,'ProcessPoolExecutor',SerialPool)
    monkeypatch.setattr(wm,'ProcessPoolExecutor',SerialPool)
    monkeypatch.setattr(pm.Geometry,'from_resources',lambda *a:geometry)
    new_physical = tmp_path/'new-physical'
    args = (field_root,physical,new_physical,'geometry','slide-data')
    report = pm.migrate(*args)
    assert pm.migrate(*args)==report
    with monkeypatch.context() as rules:
        rules.setattr(pm,'SLIDE_LAUNCH','different-rule')
        with pytest.raises(ValueError,match='different Physical migration'):
            pm.migrate(*args)
    assert report['results'][0]['migration']['changed_target_indices']
    with pytest.raises(ValueError):
        pm.migrate(field_root,physical,physical/'nested','geometry','slide-data')
    cfg = dict(target_beats=2,audio_after_beats=2,chart_history_beats=4,decoder_history_dropout=.15,
        decoder_train_chart_cache=str(field_root),decoder_train_physical_cache=str(new_physical),
        decoder_train_geometry='geometry',decoder_train_slide_data='slide-data')
    chart = ChartSupports(record,old,geometry)
    onsets = real_onsets(chart.tokens,chart.timing,sequence_start_tick=record['start_tick'])
    rows = {}
    for start in (0,192,1152):
        identity = f'0:5:{start}'
        row = dict(chart_id='0:5',audio_variant='0',start_tick=start,end_tick=start+192)
        with patch('maidata.physical.attribution.first_route_coverage',return_value=frozenset()):
            value = compile_window(chart,dict(row,window_id=identity),cfg,12.,onsets)
        storage = write_asset(windows/'windows'/f'{start}.pt',value,windows)
        rows[identity] = dict(row,**storage,lengths=value['metadata']['model_lengths'],
                             logical_lengths=value['metadata']['logical_lengths'])
    seq = write_asset(windows/'sequences'/'0-5.pt',chart_sequence(record),windows)
    write_json(windows/'manifest.json',dict(schema=SCHEMA,complete=True,windows=rows,sequences={'0:5':seq},
        contract=dict(config={k:cfg[k] for k in wm.CONFIG_KEYS},inputs=dict(physical={}))))
    originals.update({p:p.read_bytes() for p in windows.rglob('*') if p.is_file()})
    output = tmp_path/'new-windows'
    result = wm.migrate(cfg,windows,output)
    assert result['complete'] and result['changed_windows']>0
    assert wm.migrate(cfg,windows,output)==result
    assert (output/seq['path']).read_bytes()==(windows/seq['path']).read_bytes()
    for root in (windows,output):
        cache = WindowCache([],dict(cfg,decoder_train_window_cache=str(root)),{})
        assert cache.read('0:5:192')['labels'] is not None
    assert all(p.read_bytes()==v for p,v in originals.items())
    with pytest.raises(ValueError,match='different window migration'):
        wm.migrate(cfg,windows,output,window_ids=['0:5:192'])
