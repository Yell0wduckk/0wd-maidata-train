from dataclasses import replace
from pathlib import Path
import json
import pytest

from maidata.chart import End, Moment, Note, Route, HoldDuration, SlideDuration
from maidata.chart.routes import path_for, route_for, catalog
from maidata.chart.types import Completion, Track
from maidata.time import TimingMap
from maidata.physical import Geometry, PathGeometry, PhysicalState, PhysicalError
from maidata.physical.actions import materialize
from maidata.physical.rules import reasons, hand_segments, handoff_matches, prune_history


@pytest.fixture
def timing(): return TimingMap([(0,60)])


@pytest.fixture
def geometry():
    return Geometry({p: PathGeometry(p,1.,((0.,1 << (int(p[0])-1)),(.5,1 << (int(p[-1])-1))))
                     for v in catalog() for p in v['paths_by_start']})


def press(timing,geometry,tick,pad,kind='tap',owner=0,duration=None,**flags):
    return materialize(Completion(owner,tick,tick,Note.make(kind,pad,**flags),duration_tick=duration),timing,geometry)


def track(timing,geometry,tick,paths,duration,owner=0,index=0,kind='slide'):
    start='A'+paths[0][0]
    return materialize(Completion(owner,tick,tick,Note.make(kind,start),
                       track=Track(tuple(route_for(p) for p in paths),0,duration),track_index=index),timing,geometry)


def begin(state,tick=0):
    state.accept(End(tick));state.accept(Moment(0))


@pytest.mark.parametrize('kind,pad,flag',[('tap','A1','short_hold'),('touch','C','short_touch_hold')])
def test_flags_do_not_change_physical_actions(timing,geometry,kind,pad,flag):
    base=press(timing,geometry,0,pad,kind)
    assert base == press(timing,geometry,0,pad,kind,**{flag:1})
    others={'ex':1,'break_':1} if kind=='tap' else {'firework':1}
    assert base == press(timing,geometry,0,pad,kind,**others)
    history=[press(timing,geometry,0,'A2')]
    assert reasons(history,base)==reasons(history,press(timing,geometry,0,pad,kind,**{flag:1,**others}))


def test_short_audio_boundary_is_not_physical(timing,geometry):
    s=PhysicalState(timing,geometry,audio_end_seconds=.005)
    begin(s)
    assert s.query(Note.make('tap','A1')).allowed
    d=s.query(Note.make('tap','A1',short_hold=1))
    assert d.basic_errors == ('duration_outside_audio',) and not d.physical


def test_track_modifiers_do_not_enter_physical(timing,geometry):
    c=Completion(0,0,0,Note.make('slide','A1'),track=Track((0,2),0,96),track_index=0)
    modified=replace(c,note=Note.make('slide','A1',ex=1,break_=1),track=replace(c.track,track_break=1))
    assert materialize(c,timing,geometry)==materialize(modified,timing,geometry)


def test_touch_pool_repetition_and_short_equivalence(timing,geometry):
    history=[press(timing,geometry,0,'C','touch_hold',duration=96),press(timing,geometry,0,'A2')]
    assert not reasons(history,press(timing,geometry,1,'B3','touch',owner=2))
    assert 'overlap' in reasons(history,press(timing,geometry,24,'C','touch',owner=2))
    assert 'overlap' not in reasons([press(timing,geometry,0,'C','touch')],press(timing,geometry,25,'C','touch',owner=2))


def test_pending_chain_commit_snapshot_and_branch(timing,geometry):
    s=PhysicalState(timing,geometry,audio_end_seconds=30)
    begin(s)
    s.accept(Note.make('slide','A1',break_=1));s.accept(Route(route_for('1-3'),1))
    pending=s.snapshot()
    s.accept(Route(route_for('3-5')))
    assert s.actions == ()
    s.accept(SlideDuration(192))
    a=s.actions[0]
    assert len(a.motions)==2 and a.launch==1 and a.actual_end==3
    assert a.motions[0].end==a.motions[1].start==2
    assert a.release==pytest.approx(3+22/96)
    assert max(n for _,_,n in hand_segments(s.actions))==1
    complete=s.snapshot()
    s.accept(Route(route_for('1-7'),0));s.accept(SlideDuration(96))
    assert len(s.actions)==2 and not s.actions[1].head and s.actions[1].pad=='A1'
    assert s.query(Route(route_for('1-5'),0)).physical==frozenset({'multitouch'})
    s.restore(complete);assert s.snapshot()==complete
    s.restore(pending);assert s.actions==()
    s.accept(SlideDuration(96));assert len(s.actions[0].motions)==1


def test_nonmonotone_duration_support(timing,geometry):
    geometry.paths['5-3']=replace(geometry.paths['5-3'],length=2.)
    old=track(timing,geometry,0,['5-3','3-5'],288,owner=0)
    assert old.motions[1].start==3 and old.motions[1].end==4
    s=PhysicalState(timing,geometry,audio_end_seconds=20)
    begin(s,96)
    s.actions=(old,)
    s.accept(Note.make('slide','A1'));s.accept(Route(route_for('1-3'),0));s.accept(Route(route_for('3-5')))
    before=s.snapshot()
    support=s.query_duration_ticks(range(1,501))
    assert support.intervals==((1,191),(385,500))
    assert support.decisions[191].physical==frozenset({'slide_too_fast'})
    assert s.snapshot()==before


def test_chain_contacts_and_internal_joint_not_head(timing,geometry):
    a=track(timing,geometry,0,['1-3','3-5'],192)
    assert not reasons([a],press(timing,geometry,192,'A3',owner=1))  # joint is not a new protected head
    assert 'tap_on_slide' in reasons([a],press(timing,geometry,193,'A3',owner=1))
    assert 'tap_on_slide' not in reasons([a],press(timing,geometry,193,'A3','touch',owner=1))


def test_chain_handoff_and_pruning_lock(timing,geometry):
    first=track(timing,geometry,0,['1-3','3-5'],96,owner=0)
    second=track(timing,geometry,96,['5-7'],96,owner=1)
    assert handoff_matches([first,second])==((first,second),)
    kept=prune_history([first,second],2.3)
    assert len(kept)==1 and kept[0].handoff_incoming
    alternate=track(timing,geometry,10,['2-5'],86,owner=2)
    assert not handoff_matches([alternate,*kept])


def test_boundary_queries_and_active_history(timing,geometry):
    s=PhysicalState(timing,geometry,audio_end_seconds=20)
    s.accept(End(0),window_end=96);s.accept(Moment(0));s.accept(Note.make('hold','A1'));s.accept(HoldDuration(480))
    s.accept(Note.make('hold','A2'));s.accept(HoldDuration(480))
    assert s.query(Moment(1000)).allowed  # closes window even if no NOTE can fit
    s.accept(Moment(1000));s.accept(End(96),window_end=192)
    assert len(s.actions)==2
    assert not s.query(Moment(0)).allowed


def test_rejected_commit_is_atomic(timing,geometry):
    s=PhysicalState(timing,geometry,audio_end_seconds=10);begin(s)
    s.accept(Note.make('tap','A1'));before=s.snapshot()
    with pytest.raises(PhysicalError):s.accept(Note.make('tap','A1'))
    assert s.snapshot()==before


def test_kind_and_position_queries_keep_touch_repeat_local(timing,geometry):
    from maidata.chart.types import TOUCH_PADS
    s=PhysicalState(timing,geometry,audio_end_seconds=10)
    begin(s,24)
    s.actions=tuple(press(timing,geometry,0,p,'touch',owner=i) for i,p in enumerate(TOUCH_PADS))
    before=s.snapshot()
    assert all(not d.allowed and d.physical == frozenset({'overlap'}) for d in s.query_note_positions('touch').values())
    assert s.query_note_kinds()['touch'].allowed
    assert s.snapshot()==before


def test_censored_window_keeps_active_actions(timing,geometry):
    from maidata.chart import TimeTarget
    s=PhysicalState(timing,geometry,audio_end_seconds=10)
    s.accept(End(0),window_end=96);s.accept(Moment(0));s.accept(Note.make('hold','A1'));s.accept(HoldDuration(192))
    before=s.snapshot()
    s.close_window(TimeTarget(timing.delta_ms(0,100.5),100.5,True,True))
    assert s.actions==before.actions and s.grammar.state.last_real_tick==0
    s.restore(before);assert s.snapshot()==before


@pytest.fixture(scope='module')
def real_resources():
    outer=Path(__file__).resolve().parents[3]
    root=outer/'repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired'
    samples=outer/'tools/MiaCode/assets/reference/slide_data.json'
    if not (root/'profile/vocabulary.json').is_file() or not samples.is_file():
        pytest.skip('explicit retained geometry resources are unavailable')
    from maimai_baseline.physical.catalog import CompactPathCatalog
    from maimai_dataset.vocabulary import CompactVocabulary
    vocabulary=CompactVocabulary.from_dict(json.loads((root/'profile/vocabulary.json').read_text('utf-8')))
    old=CompactPathCatalog.load(root/'physical/path-catalog',vocabulary)
    new=Geometry.from_resources(root/'physical/path-catalog',samples)
    return old,new


@pytest.mark.integration
def test_retained_single_segment_oracle(real_resources):
    from maimai_baseline.physical.intervals import describe, reasons as old_reasons, hand_segments as old_segments
    old,geometry=real_resources
    timing=TimingMap([(0,120),(2,180)])
    specs=[('slide','A1',0,96,2),('wifi','A3',120,144,70),('hold','A6',24,96,None),
           ('touch_hold','C',12,96,None),('tap','A2',36,None,None),('touch','B3',36,None,None)]
    actions,reference=[],[]
    for owner,(kind,pad,tick,dur,route) in enumerate(specs):
        if route is not None:
            c=Completion(owner,owner,tick,Note.make(kind,pad),track=Track((route,),0,dur),track_index=0)
            row=dict(event='NOTE',kind=kind,start_pad=pad,branch_duration_tick=dur)
        else:
            c=Completion(owner,owner,tick,Note.make(kind,pad),duration_tick=dur)
            row=dict(event='NOTE',kind=kind,pad=pad)
            if dur is not None:row['hold_duration_tick']=dur
        a=materialize(c,timing,geometry)
        b=describe(row,tick=tick,moment=owner,owner=owner,timing=timing,catalog=old,route=route)
        assert (a.press_start,a.press_end,a.launch,a.actual_end,a.release,a.guard_end)==pytest.approx(
               (b.press_start,b.press_end,b.launch,b.actual_end,b.release,b.guard_end))
        assert a.entries==b.entries
        for n in range(len(actions)+1):
            assert reasons(actions[:n],a)==old_reasons(reference[:n],b,old,include_touch_repeat=True)
        actions.append(a);reference.append(b)
    assert hand_segments(actions)==old_segments(reference)


@pytest.mark.integration
def test_retained_handoff_tie_break_order(real_resources):
    from maimai_baseline.physical.intervals import describe, handoff_matches as old_matches
    old,geometry=real_resources
    timing=TimingMap([(0,120)])
    actions,reference=[],[]
    for path,index in [('1-5',0),('1<5',1)]:
        r=route_for(path)
        actions.append(track(timing,geometry,0,[path],96,owner=0,index=index))
        row=dict(event='NOTE' if index==0 else 'SLIDE_BRANCH',start_pad='A1',branch_duration_tick=96)
        if index==0:row['kind']='slide'
        reference.append(describe(row,tick=0,moment=0,owner=0,timing=timing,catalog=old,route=r))
    r=route_for('5-1')
    actions.append(track(timing,geometry,96,['5-1'],96,owner=1))
    reference.append(describe(dict(event='NOTE',kind='slide',start_pad='A5',branch_duration_tick=96),
                             tick=96,moment=1,owner=1,timing=timing,catalog=old,route=r))
    assert [(a.path_key,b.path_key) for a,b in handoff_matches(actions)] == [
        ((old.path_key(a.path),),(old.path_key(b.path),)) for a,b in old_matches(reference)]
