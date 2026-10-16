"""Final judge-area Hold repairs, exact probabilities and real Decoder replay."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch
import math

import pytest
import torch

from test_inference import config, producer_config, conditions
from test_chart_physical import geometry, press, track
from test_inference_tail_collision import model_for
from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.chart.routes import route_for
from maidata.chart.types import GENERATED_TYPES
from maidata.decoder.batching import pack_tokens
from maidata.decoder.session import DecodeSession
from maidata.decoder.sequence import expand_inputs, family
from maidata.inference.generation import decode_window, GenerationError
from maidata.inference.hold_tail import (contacts, strict_support, DurationDistribution,
    FieldAnchor, FieldProbabilities, DurationLeaf, repairs)
from maidata.inference.sampling import Sampler
from maidata.inference.options import GenerationOptions
from maidata.inference.tail_collision import GenerationPhysicalState
from maidata.model import decoder_time_distributions
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.support import interval_decision
from maidata.time import TimingMap, advance_ticks


def pending(geometry, *, slide=False, kind='hold', start=32, timing=None):
    timing = timing or TimingMap([(0, 62.5)])
    state = GenerationPhysicalState(timing, geometry, audio_end_seconds=20)
    state.inject_boundary(Moment(0), start_tick=start, window_end=start+768)
    if slide:
        state.actions = (press(timing, geometry, 0, 'A3', kind, duration=200, owner=42),)
        state.accept(Note.make('slide', 'A1'))
        state.accept(Route(route_for('1-3'), 0))
    else:
        state.actions = (track(timing, geometry, 0, ['1-3'], 192, owner=42),)
        state.accept(Note.make(kind, 'A3'))
    return state


@pytest.mark.parametrize('kind', ['hold', 'touch_hold'])
def test_hold_interval_is_real_half_open_and_raw_oracle_unchanged(geometry, kind):
    state = pending(geometry, kind=kind)
    # Entry tick 192; one beat earlier is tick 96. Hold starts at 32.
    assert not contacts(state, HoldDuration(64))
    assert contacts(state, HoldDuration(65))
    assert contacts(state, HoldDuration(96))[0]['checked_seconds'] == pytest.approx(.96)
    raw = InferencePhysicalState(state.timing, geometry, audio_end_seconds=20)
    raw.restore(state.snapshot())
    assert raw.query(HoldDuration(96)).allowed and state.query(HoldDuration(96)).allowed
    # A hold starting after the check instant does not cover it.
    assert not contacts(pending(geometry, kind=kind, start=97), HoldDuration(96))


@pytest.mark.parametrize('kind', ['hold', 'touch_hold'])
def test_slide_interval_boundaries_final_route_only_and_branch(geometry, kind):
    state = pending(geometry, slide=True, kind=kind)
    assert contacts(state, SlideDuration(96))
    assert contacts(state, SlideDuration(335))
    assert not contacts(state, SlideDuration(336))
    # Chaining to a different final area makes the intermediate A3 irrelevant.
    state.accept(Route(route_for('3-5')))
    assert not contacts(state, SlideDuration(96))
    # Both accepted Slide branches are checked independently when a Hold follows.
    hold = pending(geometry, kind=kind)
    hold.actions += (replace(hold.actions[0], track_index=1),)
    assert [w['track_index'] for w in contacts(hold, HoldDuration(96))] == [0, 1]


def test_uses_last_judge_entry_not_endpoint_or_motion_end(geometry):
    geometry.paths['1-3'] = replace(geometry.paths['1-3'], entries=((0., 1), (.25, 1 << 9)))
    timing = TimingMap([(0, 62.5)])
    state = GenerationPhysicalState(timing, geometry, audio_end_seconds=20)
    state.inject_boundary(Moment(0), start_tick=32, window_end=384)
    state.actions = (press(timing, geometry, 0, 'B2', 'touch_hold', duration=100, owner=42),)
    state.accept(Note.make('slide', 'A1')); state.accept(Route(route_for('1-3'), 0))
    witnesses = contacts(state, SlideDuration(96))
    assert len(witnesses) == 1
    assert witnesses[0]['entry_seconds'] == pytest.approx(1.52)
    assert witnesses[0]['checked_seconds'] == pytest.approx(.56)
    assert witnesses[0]['judge_area_mask'] == 1 << 9


@pytest.mark.parametrize('slide', [False, True])
@pytest.mark.parametrize('change', [False, True])
def test_exact_partition_matches_every_tick_through_bpm_change(geometry, slide, change):
    timing = TimingMap([(0, 62.5), (Fraction(150, 96), 125)]) if change else TimingMap([(0,62.5)])
    state = pending(geometry, slide=slide, timing=timing)
    state.set_duration_cap(timing.second(800))
    constructor = SlideDuration if slide else HoldDuration
    support = strict_support(state)
    for ticks in range(*[support['domain'][0], support['domain'][1]+1]):
        expected = state.query(constructor(ticks))
        if contacts(state, constructor(ticks)):
            expected = replace(expected, physical=expected.physical | {'slide_too_fast'})
        assert interval_decision(support, ticks) == expected


def test_cont_probability_is_product_and_queries_reused():
    sampler = Sampler(temperature=1, repetition_penalty=0)
    scores = [torch.full((385,), -torch.inf) for _ in range(3)]
    scores[0][0], scores[0][384] = math.log(.2), math.log(.8)
    scores[1][0], scores[1][384] = math.log(.5), math.log(.5)
    scores[2][0] = 0.
    def output(depth):
        return SimpleNamespace(hidden=torch.full((1,1,1), float(depth)),
                               cache=SimpleNamespace(memory=None, depth=depth))
    decoder = SimpleNamespace(heads=SimpleNamespace(time_logits=lambda h,k: scores[int(h.item())]),
        begin=lambda b,m,cache: cache.depth+1, complete=lambda depth,t: output(depth))
    masks = tuple(tuple(torch.isfinite(s).tolist()) for s in scores)
    tree = SimpleNamespace(kind='HOLD_DURATION', origin_tick=0,
                           queries=((0,0),(384,4),(768,8)), hard=masks, basic=masks)
    distribution = DurationDistribution(sampler, decoder, output(0), tree, TimingMap([(0,120)]))
    # HOLD minimum is two ticks; use class 1 rather than class 0.
    for s in scores:
        s[1], s[0] = s[0].clone(), -torch.inf
    tree.hard = tree.basic = tuple(tuple(torch.isfinite(s).tolist()) for s in scores)
    leaves = list(distribution.leaves())
    assert [leaf.token.ticks for leaf in leaves] == [2,386,770]
    assert [math.exp(leaf.log_probability) for leaf in leaves] == pytest.approx([.2,.4,.4])
    assert distribution.extra_forwards == 2
    list(distribution.leaves())
    assert distribution.extra_forwards == 2


def generate_case(c, geometry, *, slide=False, choose_duration=False, top_k=0,
                  kind='hold', change_duration=False, start=32, stop=None, fail_pad_branch=False,
                  extra_tail=False, fixed_pad=False):
    timing = TimingMap([(0,62.5)])
    state = GenerationPhysicalState(timing, geometry, audio_end_seconds=30)
    state.inject_boundary(Moment(0), start_tick=start, window_end=start+512)
    if slide:
        state.actions = (press(timing, geometry, start-32, 'A3', kind, duration=200, owner=42),)
    else:
        state.actions = (track(timing, geometry, start-32, ['1-3'],192,owner=42),)
    if extra_tail:
        # A prior path entry 18 ticks (180 ms) before this Hold onset, with no
        # simultaneous three-hand demand. The other, future Slide causes the
        # independent final-area conflict after duration is selected.
        state.actions += (track(timing,geometry,start-162,['1-3'],96,owner=43),)
    model = model_for(c, 'slide' if slide else kind)
    old_route, new_route = route_for('1-3'), route_for('1-5')
    with torch.no_grad():
        model.type_head.bias[GENERATED_TYPES.index('ROUTE')] = 0
        model.type_head.bias[GENERATED_TYPES.index('SLIDE_DURATION')] = 10
        model.heads.button.bias.fill_(-1000)
        model.heads.button.bias[0 if slide else 2] = 8
        model.heads.button.bias[1] = -100 if choose_duration else 6
        if fixed_pad:
            model.heads.button.bias.fill_(-torch.inf)
            model.heads.button.bias[0 if slide else 2] = 8
        if kind == 'touch_hold':
            model.heads.touch.weight.zero_(); model.heads.touch.bias.fill_(-1000)
            model.heads.touch.bias[2] = 8
            model.heads.touch.bias[1] = -100 if choose_duration else 6
        model.heads.route.bias.fill_(-1000)
        model.heads.route.bias[old_route] = 8
        model.heads.route.bias[new_route] = -100 if choose_duration else 6
        model.heads.track_break.weight.zero_()
        model.heads.track_break.bias[0], model.heads.track_break.bias[1] = 10, -10
        # A different shape with the SAME endpoint must never be used.
        model.heads.route.bias[route_for('1>3')] = 7
        for head in model.heads.times.values():
            head.bias.fill_(-1000)
        head = model.heads.times['SLIDE_DURATION' if slide else 'HOLD_DURATION']
        head.bias[95] = 8
        head.bias[359 if slide else 31] = 7 if choose_duration else 0
    prepared = conditions(c,timing,30)
    with torch.inference_mode():
        memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),prepared.z,
                                      prepared.plans[:1],prepared.plan_valid[:1])
    session = DecodeSession(state)
    sampler = Sampler(temperature=0, repetition_penalty=0, top_k=top_k)
    original_choices = sampler.choices
    def choices(*args, **kwargs):
        oracle = args[2]
        if fail_pad_branch and oracle.grammar.state.phase=='done' and oracle.grammar.state.note.pad=='A2':
            return
        yield from original_choices(*args, **kwargs)
    sampler.choices = choices
    events = []
    seen_route = [None]
    real_begin, real_time = model.begin, model.heads.time_logits
    def begin(batch,*args,**kwargs):
        if int(batch.route[0,-1]) >= 0:
            seen_route[0] = int(batch.route[0,-1])
        return real_begin(batch,*args,**kwargs)
    def time_logits(h,k):
        result = real_time(h,k)
        if change_duration and seen_route[0] == new_route and k == 'SLIDE_DURATION':
            result = result.clone(); result.fill_(-1000); result[127] = 8
        return result
    with torch.inference_mode(), patch.object(model,'begin',side_effect=begin), \
         patch.object(model.heads,'time_logits',side_effect=time_logits):
        tokens, report = decode_window(model,memory,session,Moment(0),sampler,decoder_time_distributions(c),timing,
            start_tick=start,end_tick=start+512,cap_tick=start+512,device='cpu',options=GenerationOptions(temperature=0),
            stream=events.append,stop=stop or (lambda accepted,rules: rules.grammar.state.phase in ('done','track_done')))
    return model, session, memory, tokens, report, events


def assert_replay(model,session,memory,tokens,start):
    # Last real token has not yet been fed into the causal cache.
    seq = expand_inputs((Moment(0),*tokens), (start/96,)*(len(tokens)+1), session.rules.timing, boundary_index=0)
    inputs = seq['tokens'][:-1]
    batch = pack_tokens([inputs],[seq['beats'][:-1]],device='cpu',slide_positions=[seq['slide_beats'][:-1]])
    types = torch.tensor([[GENERATED_TYPES.index(family(t)) for t in seq['tokens'][1:]]])
    with torch.inference_mode():
        expected = model(batch,types,memory).cache
    for got,want in zip(session.cache.layers,expected.layers):
        torch.testing.assert_close(got.key,want.key,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(got.value,want.value,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('kind', ['hold','touch_hold'])
def test_pad_replacement_keeps_duration_and_kv(config,geometry,kind):
    model,s,memory,tokens,report,events = generate_case(config,geometry,kind=kind)
    assert tokens == (Note.make(kind,'A2'),HoldDuration(96))
    assert report['hold_slide_tail_repairs'][0]['field'] == 'pad'
    assert report['physical_bypass'] == []
    assert_replay(model,s,memory,tokens,32)
    active = []
    for event in events:
        if event['event']=='token': active.append(event['token'])
        elif event['event']=='rollback': del active[event['token_count']:]
    assert tuple(active)==tokens
    assert not any(isinstance(e.get('token'),HoldDuration) and e['token'].ticks==96 and
                   e['complete'] for e in events[:next(i for i,e in enumerate(events) if e['event']=='rollback')])


def test_route_replacement_excludes_same_endpoint_and_reinfers_duration(config,geometry):
    model,s,memory,tokens,report,events = generate_case(config,geometry,slide=True,change_duration=True)
    assert tokens == (Note.make('slide','A1'),Route(route_for('1-5'),0),SlideDuration(128))
    repair = report['hold_slide_tail_repairs'][0]
    assert repair['field']=='route'
    assert repair['original_duration']['ticks']==96 and repair['replacement_duration']['ticks']==128
    assert_replay(model,s,memory,tokens,32)


@pytest.mark.parametrize('slide', [False,True])
def test_duration_only_repair_keeps_position(config,geometry,slide):
    model,s,memory,tokens,report,_ = generate_case(config,geometry,slide=slide,choose_duration=True)
    assert report['hold_slide_tail_repairs'][0]['field']=='duration'
    assert tokens[-1] == (SlideDuration(360) if slide else HoldDuration(32))
    assert_replay(model,s,memory,tokens,32)


def test_cross_window_active_actions_are_checked(config,geometry):
    _,_,_,tokens,report,_ = generate_case(config,geometry,start=1568)
    assert tokens == (Note.make('hold','A2'),HoldDuration(96))
    assert report['hold_slide_tail_repairs'][0]['conflicts'][0]['slide_tick']==1536


def test_original_filtered_mass_is_pooled_not_equal_field_weights(config,geometry):
    state = pending(geometry)
    before = GenerationPhysicalState(state.timing,geometry,audio_end_seconds=20)
    snap = state.snapshot()
    before.restore(replace(snap,grammar=replace(snap.grammar,phase='boundary',note=None,owner=-1)))
    session = DecodeSession(before)
    # Choose duration to avoid needing any speculative model call in this assertion.
    anchor = FieldAnchor(session.snapshot(),0,Note.make('hold','A3'),
        FieldProbabilities((math.log(.9),math.log(.1)),('A3','A2')),None)
    distribution = SimpleNamespace(extra_forwards=0,leaves=lambda: iter([
        DurationLeaf(HoldDuration(96),math.log(.8),None),
        DurationLeaf(HoldDuration(32),math.log(.2),None)]))
    sampler = Sampler(temperature=1,repetition_penalty=0)
    def choose(probabilities, *args, **kwargs):
        assert probabilities.tolist() == pytest.approx([1/3,2/3])
        return torch.tensor([1])
    with patch('torch.multinomial',side_effect=choose):
        result=next(repairs(sampler,None,anchor,state,HoldDuration(96),distribution,
                           contacts(state,HoldDuration(96)),end_tick=384,cap_tick=384))
    assert result.record['field']=='duration'


def test_top_k_cannot_silently_expand_repair_support(config,geometry):
    # The only retained type, pad and duration all collide. Exhaustion of this
    # very small frontier cannot silently enlarge top-k to choose another pad.
    with pytest.raises(GenerationError, match='no legal completion'):
        generate_case(config,geometry,top_k=1)


def test_repair_rollback_drops_abandoned_report_and_stream_content(config,geometry):
    model,s,memory,tokens,report,events = generate_case(config,geometry,fail_pad_branch=True,
        stop=lambda accepted,rules: rules.grammar.state.phase=='done' and rules.grammar.state.note.pad=='A3')
    assert tokens==(Note.make('hold','A3'),HoldDuration(32))
    assert [r['field'] for r in report['hold_slide_tail_repairs']]==['duration']
    active=[]
    for event in events:
        if event['event']=='token': active.append(event['token'])
        elif event['event']=='rollback': del active[event['token_count']:]
    assert tuple(active)==tokens
    assert_replay(model,s,memory,tokens,32)


def test_changed_route_can_require_a_different_legal_duration(config,geometry):
    # Different endpoint, but the same last judge area: a legal longer duration
    # exists. It must be considered without demanding legality at the old value.
    geometry.paths['1-5']=replace(geometry.paths['1-5'],entries=((0.,1),(.5,1<<2)))
    _,_,_,tokens,report,_=generate_case(config,geometry,slide=True)
    assert tokens[1]==Route(route_for('1-5'),0) and tokens[-1]==SlideDuration(360)
    assert report['hold_slide_tail_repairs'][0]['field']=='route'


def test_existing_ex_repair_survives_duration_only_repair(config,geometry):
    _,_,_,tokens,report,_=generate_case(config,geometry,choose_duration=True,extra_tail=True,fixed_pad=True)
    assert tokens==(Note.make('hold','A3',ex=1),HoldDuration(32))
    assert report['hold_slide_tail_repairs'][0]['field']=='duration'
    assert len(report['tail_repairs'])==1 and report['tail_repairs'][0]['strategy']=='ex'


def test_streamed_ir_and_text_contain_the_same_repaired_pair(config,geometry,tmp_path):
    import json
    from test_inference_materialize import write_audio
    from maidata.inference.__main__ import ir_metadata
    from maidata.inference.render import SimaiWriter
    from maidata.inference.streaming import StreamingPreview
    _,session,_,tokens,_,events=generate_case(config,geometry)
    timing=session.rules.timing
    write_audio(tmp_path/'audio.wav')
    metadata=ir_metadata(timing,30.)
    end=metadata['render_end_beat']
    live=StreamingPreview(SimaiWriter(metadata['controls'],Fraction(end['num'],end['den'])),timing,
        tmp_path/'audio.wav',tmp_path/'output',ir_geometry=geometry,ir_metadata=metadata,difficulty=5,level=13.4)
    for event in events: live(event)
    live.write(live.writer.preview(live.active_groups),force=True)
    snapshot=json.loads(live.snapshot_path.read_text('utf-8'))
    notes=[n for m in snapshot['ir']['moments'] for g in m['groups'] for n in g['notes']]
    assert len(notes)==1 and notes[0]['pad']=='A2'
    assert '2h[' in snapshot['chart'] and '3h[' not in snapshot['chart']
    assert tokens[-1]==HoldDuration(96)


@pytest.mark.parametrize('mode', ['continue','range'])
@pytest.mark.parametrize('cancel_first', [False,True])
def test_interactive_rejection_and_prefix_edit_do_not_retain_repair_history(config,geometry,mode,cancel_first):
    import threading
    from test_inference_interactive import stub
    from maidata.decoder.windows import Onset
    c=dict(config,target_beats=16,target_stride_beats=16,chart_history_beats=4,audio_after_beats=4)
    timing=TimingMap([(0,62.5)])
    session=stub(c,geometry,timing,30.)
    session.decoder=model_for(c,'hold')
    with torch.no_grad():
        session.decoder.heads.button.bias.fill_(-1000)
        session.decoder.heads.button.bias[2],session.decoder.heads.button.bias[1]=8,6
        head=session.decoder.heads.times['HOLD_DURATION']
        head.bias.fill_(-1000); head.bias[95],head.bias[31]=8,0
        session.audio_kv=session.decoder.prepare_audio(session.conditions.audio)
    original=(Onset(0,0.,(Note.make('slide','A1'),Route(route_for('1-3'),0),SlideDuration(192))),)
    session.groups=original
    request=dict(id=1,revision=1,mode=mode,start_tick=32,end_tick=64,max_tokens=2,max_beats=0,
        max_four_beat_blocks=0,temperature=0,seed=17,content=True,repetition_penalty=0)
    if cancel_first:
        from maidata.inference.interactive import Cancelled
        calls=[]
        def is_set():
            calls.append(True)
            return len(calls)>1  # Window started, then cancel at the committed pair.
        with pytest.raises(Cancelled):
            session.suggest(request,SimpleNamespace(is_set=is_set))
        assert session.groups==original
    first=session.suggest(request,threading.Event())
    assert first['hold_slide_tail_repairs'][0]['field']=='pad'
    assert session.groups==original  # A suggestion alone never accepts its pair.
    again=session.suggest(dict(request,id=2),threading.Event())
    assert again['text']==first['text'] and again['ir_fragment']==first['ir_fragment']
    # An artificial edit of the real prefix invalidates the old route K/V and
    # conflict witness; the next request must not retain a pad repair.
    session.groups=(replace(original[0],content=(Note.make('slide','A1'),Route(route_for('1-5'),0),SlideDuration(192))),)
    session.revision=2
    changed=session.suggest(dict(request,id=3,revision=2),threading.Event())
    assert changed['hold_slide_tail_repairs']==[]
    assert changed['onsets'][0]['content'][0]['pad']=='A3'
