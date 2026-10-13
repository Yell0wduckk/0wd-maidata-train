"""Post-sampling tail repairs, with real CPU Decoder K/V and raw-rule oracles."""
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config, conditions
from test_chart_physical import geometry, track, press
from maidata.chart import Moment, Note
from maidata.chart.types import KINDS
from maidata.decoder.batching import pack_tokens
from maidata.decoder.session import DecodeSession
from maidata.model import build_decoder, decoder_time_distributions
from maidata.physical.inference import InferencePhysicalState
from maidata.inference.generation import decode_window
from maidata.inference.options import GenerationOptions
from maidata.inference.sampling import Sampler
from maidata.inference.tail_collision import GenerationPhysicalState, TailRepair
from maidata.time import TimingMap


def state(geometry, gap=18, *, timing=None, kind=GenerationPhysicalState):
    # BPM 62.5 gives exactly 10 ms/tick. 1-3 enters A3 at tick 144.
    timing = timing or TimingMap([(0, 62.5)])
    s = kind(timing, geometry, audio_end_seconds=10)
    s.inject_boundary(Moment(0), start_tick=144+gap, window_end=384)
    s.actions = (track(timing, geometry, 0, ['1-3'], 96, owner=42),)
    return s


@pytest.mark.parametrize('gap,expected', [(1,'moment'),(14,'moment'),(15,'ex'),(18,'ex'),(20,'ex'),(21,None)])
@pytest.mark.parametrize('kind', ['tap','hold','slide'])
def test_boundaries_and_flags_are_repaired_only_after_note_selection(geometry, gap, expected, kind):
    s = state(geometry, gap)
    note = Note.make(kind, 'A3', break_=1)
    raw = InferencePhysicalState(s.timing, geometry, audio_end_seconds=10)
    raw.restore(s.snapshot())
    assert ('tap_on_slide' in raw.query(note).physical) == (gap <= 20)
    assert 'tap_on_slide' not in s.query(note).physical
    saved = s.snapshot()
    repair = s.tail_repair(note)
    assert s.snapshot() == saved
    if expected == 'ex':
        assert repair.replacement == replace(note, ex=1)
        assert repair.original == note and repair.minimum_tick is None
    elif expected == 'moment':
        assert repair.minimum_tick == 159 and repair.replacement is None
    else:
        assert repair is None


def test_existing_ex_early_collision_touch_and_other_masks(geometry):
    s = state(geometry, 18)
    assert s.tail_repair(Note.make('tap','A3',ex=1)) is None
    assert s.tail_repair(Note.make('touch','A3')) is None
    s = state(geometry, 14)
    assert s.tail_repair(Note.make('tap','A3',ex=1)).minimum_tick == 159
    s.actions += (press(s.timing,geometry,158,'A3',owner=43),)
    assert 'overlap' in s.query(Note.make('tap','A3')).physical
    launch = state(geometry, -35)  # 1.09s is inside A1 launch_guard.
    assert 'tap_on_slide' in launch.query(Note.make('tap','A1')).physical


def test_actual_seconds_through_bpm_change_and_latest_collision_wins(geometry):
    timing = TimingMap([(0,62.5),(Fraction(150,96),125)])
    s = state(geometry, 16, timing=timing)
    assert s.tail_repair(Note.make('tap','A3')).minimum_tick == 168
    s = state(geometry, 18)
    a = s.actions[0]
    recent = replace(a.motions[0], entries=((s.timing.second(152),1<<2),))
    s.actions += (replace(a, owner=43, motions=(recent,)),)
    assert s.tail_repair(Note.make('tap','A3')).minimum_tick == 167


def test_cross_window_active_track_and_snapshot_replay(geometry):
    timing = TimingMap([(0,62.5)])
    s = GenerationPhysicalState(timing, geometry, audio_end_seconds=30)
    s.actions = (track(timing, geometry, 1380, ['1-3'],96,owner=42),)
    s.inject_boundary(Moment(0),start_tick=1542,window_end=3072)
    saved = s.snapshot()
    repair = s.tail_repair(Note.make('tap','A3'))
    assert repair.replacement.ex == 1
    s.accept(repair.replacement)
    assert s.grammar.state.note.ex == 1
    s.restore(saved)
    assert s.tail_repair(Note.make('tap','A3')) == repair


def model_for(c, kind='tap'):
    model = build_decoder(c).eval()
    with torch.no_grad():
        model.type_head.weight.zero_(); model.type_head.bias.fill_(-100); model.type_head.bias[1]=100
        model.heads.kind.weight.zero_(); model.heads.kind.bias.fill_(-100)
        model.heads.kind.bias[KINDS.index(kind)]=100; model.heads.kind.bias[KINDS.index('touch')]=50
        model.heads.button.weight.zero_(); model.heads.button.bias.fill_(-100); model.heads.button.bias[2]=100
        model.heads.button.bias[1]=50
        model.heads.route.weight.zero_(); model.heads.route.bias.fill_(-100); model.heads.route.bias[0]=100
        for head in model.heads.flags:
            head.weight.zero_(); head.bias[0]=10; head.bias[1]=-10
        for head in model.heads.times.values():
            head.weight.zero_(); head.bias.fill_(-100); head.bias[0]=100
        model.heads.times['MOMENT'].bias[0]=-100
        model.heads.times['MOMENT'].bias[19]=100  # 20 ticks
        model.heads.times['MOMENT'].bias[27]=80   # 28 ticks, retry alternative
    return model


def run(c, geometry, *, gap=18, kind='tap', initial_moment=False, sampler=None, plain=False, onset_end_tick=None):
    s = state(geometry, gap, kind=InferencePhysicalState if plain else GenerationPhysicalState)
    if initial_moment:
        s.inject_boundary(Moment(0),start_tick=132,window_end=384)
    model = model_for(c, kind)
    prepared = conditions(c,s.timing,10.)
    with torch.inference_mode():
        memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),prepared.z,
                                      prepared.plans[:1],prepared.plan_valid[:1])
    session = DecodeSession(s)
    sampler = sampler or Sampler(temperature=0,repetition_penalty=0)
    events=[]
    original = sampler.choices
    def choose(*args, **kwargs):
        oracle = args[2]
        if initial_moment and oracle.grammar.state.cursor == 132:
            kwargs['force_moment']=True
        yield from original(*args,**kwargs)
    with torch.inference_mode(), patch.object(sampler,'choices',side_effect=choose), \
         patch.object(model,'begin',wraps=model.begin) as front, patch.object(model,'complete',wraps=model.complete) as back:
        tokens, report = decode_window(model,memory,session,Moment(0),sampler,decoder_time_distributions(c),s.timing,
            start_tick=s.grammar.state.cursor,end_tick=384,cap_tick=768,device='cpu',
            options=GenerationOptions(temperature=0),stream=events.append,
            onset_end_tick=onset_end_tick,
            stop=lambda accepted,rules: rules.grammar.state.phase in ('done','track_done'))
        counts = dict(begin=front.call_count,complete=back.call_count)
    return model,session,memory,tokens,report,events,counts


@pytest.mark.parametrize('kind', ['tap','hold','slide'])
def test_ex_commit_has_no_extra_model_forward_and_finishes_duration(config,geometry,kind):
    _,s,_,tokens,report,events,calls = run(config,geometry,kind=kind,plain=True)
    assert isinstance(tokens[0],Note) and tokens[0].kind==kind and tokens[0].pad=='A3' and tokens[0].ex==1
    assert all(not isinstance(t,Note) or t.kind not in ('touch','touch_hold') for t in tokens)
    assert calls['begin']==calls['complete']==len(tokens)
    assert report['physical_bypass']==[] and report['tail_repairs'][0]['strategy']=='ex'
    assert s.rules.grammar.state.note.ex==1 and len(s.rules.actions)==2


@pytest.mark.parametrize('initial_moment', [False,True])
def test_severe_collision_forces_time_and_kv_matches_accepted_prefix(config,geometry,initial_moment):
    model,s,memory,tokens,report,events,calls = run(config,geometry,gap=8,initial_moment=initial_moment)
    assert len(tokens)==2 and isinstance(tokens[0],Moment) and isinstance(tokens[1],Note)
    assert tokens[1].kind=='tap' and tokens[1].pad=='A3'
    assert report['tail_repairs'][0]['strategy']=='moment'
    assert calls==dict(begin=3 if initial_moment else 2,complete=4 if initial_moment else 3)
    assert any(e['event']=='rollback' for e in events)==initial_moment
    origin = 132 if initial_moment else 152
    end = s.rules.timing.tick_after(origin,tokens[0].delta_ms)
    assert end >= 159
    batch=pack_tokens([[Moment(0),tokens[0]]],[[origin/96,end/96]],device='cpu')
    with torch.inference_mode():
        expected=model(batch,torch.tensor([[0,1]]),memory).cache
    for got,want in zip(s.cache.layers,expected.layers):
        torch.testing.assert_close(got.key,want.key,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(got.value,want.value,atol=2e-5,rtol=2e-4)
    # Replay stream rollback semantics: only the accepted text/IR token prefix remains.
    active=[]
    for event in events:
        if event['event']=='token': active.append(event['token'])
        elif event['event']=='rollback': del active[event['token_count']:]
    assert tuple(active)==tokens


def test_later_backtracking_removes_abandoned_ex_repair(config,geometry):
    class Search:
        physical_hard_mask=True
        def choices(self, model,routed,oracle,*args,**kwargs):
            if oracle.grammar.state.phase!='boundary':
                return
            out=model.complete(routed,torch.ones_like(routed.valid,dtype=torch.long))
            yield oracle.tail_repair(Note.make('hold','A3')),out.cache
            yield Note.make('tap','A2'),out.cache
    _,s,_,tokens,report,events,_ = run(config,geometry,sampler=Search())
    assert tokens==(Note.make('tap','A2'),)
    assert report['tail_repairs']==[] and report['backtracks']==1
    assert len(s.rules.actions)==2 and s.rules.grammar.state.note.pad=='A2'
    assert any(e['event']=='rollback' and e['token_count']==0 for e in events)


def test_retry_crossing_editor_cutoff_leaves_no_empty_moment_or_cache(config,geometry):
    _,s,_,tokens,report,events,calls = run(config,geometry,gap=8,initial_moment=True,onset_end_tick=155)
    assert tokens==() and report['stop_reason']=='onset_limit'
    assert s.rules.grammar.state.phase=='boundary' and s.cache is None
    assert len(s.rules.actions)==1
    assert report['physical_bypass']==[] and report['tail_repairs']==[]


def test_stream_snapshot_text_and_ir_share_repaired_ex_and_rolled_back_time(config,geometry,tmp_path):
    import json
    from test_inference_materialize import write_audio
    from maidata.inference.__main__ import ir_metadata
    from maidata.inference.render import SimaiWriter
    from maidata.inference.streaming import StreamingPreview
    _,s,_,tokens,report,events,_ = run(config,geometry,gap=8,initial_moment=True)
    write_audio(tmp_path/'audio.wav')
    meta=ir_metadata(s.rules.timing,10.)
    end=meta['render_end_beat']
    live=StreamingPreview(SimaiWriter(meta['controls'],Fraction(end['num'],end['den'])),s.rules.timing,
        tmp_path/'audio.wav',tmp_path/'output',ir_geometry=geometry,ir_metadata=meta,difficulty=5,level=13.4)
    for event in events: live(event)
    live.write(live.writer.preview(live.active_groups),force=True)
    snapshot=json.loads(live.snapshot_path.read_text('utf-8'))
    assert '3x' in snapshot['chart']
    notes=[n for m in snapshot['ir']['moments'] for g in m['groups'] for n in g['notes']]
    assert len(notes)==1 and notes[0]['pad']=='A3' and 'ex' in notes[0]['flags']
    position=snapshot['ir']['moments'][0]['position_beats']
    assert Fraction(position['num'],position['den'])==Fraction(160,96)
