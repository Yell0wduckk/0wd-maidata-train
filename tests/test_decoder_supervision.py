"""Independent main/geometry/Physical accounting and public producer gradients."""
import math
import random
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from maidata.chart.types import (Moment, Note, Route, HoldDuration, SlideDuration, KINDS, BUTTON_PADS)
from maidata.chart.routes import catalog
from maidata.time import TimingMap
from maidata.physical import PhysicalState
from maidata.physical.geometry import Geometry, PathGeometry
from maidata.physical.attribution import AttributionState
from maidata.decoder.supports import window_stages
from maidata.decoder.windows import build_window
from maidata.decoder.batching import collate_windows, pack_tokens
from maidata.decoder.supervision import prediction_loss, decoder_loss
from maidata.decoder.losses import RouteGeometryLoss, physical_mean, physical_categories, Reduction
from maidata.model import build_decoder, decoder_time_distributions
from test_decoder_network import config, inputs, prepare


def record(penalty=0, basic=()):
    return dict(basic=list(basic), penalty=penalty, physical=penalty)


def test_physical_fifty_fifty_global_units_and_flag_gradient_isolation():
    model = build_decoder(config())
    with torch.no_grad():
        for head in (model.heads.kind, model.heads.button):
            head.weight.zero_()
            head.bias.zero_()
    h = torch.randn(24, requires_grad=True)
    type_logits = torch.zeros(5, requires_grad=True)
    stage = dict(cursor_tick=0, types={'NOTE':record(), 'MOMENT':record(1)}, pad_kind='tap',
        kind={k:record(4 if k=='hold' else 0, () if k in ('tap','hold') else ('basic',)) for k in KINDS},
        pad={p:record(2 if p=='A1' else 0, () if p in ('A1','A2') else ('basic',)) for p in BUTTON_PADS})
    kwargs = dict(timing=TimingMap([(0,120)]), window_end=96, cap_tick=192, audio_end_seconds=5,
                  distributions=decoder_time_distributions(config()), geometry=None)
    loss = prediction_loss(model.heads,h,type_logits,Note.make('tap','A1'),stage,**kwargs)
    assert loss.main.count == 6 and loss.physical.count == 2 and loss.judge.count == 0
    torch.testing.assert_close(loss.physical.total, torch.tensor([math.log(2), .5*math.log(2), .5*math.log(2), 0.]))
    torch.testing.assert_close(physical_mean(loss.physical),torch.tensor(math.log(2)*3.5/8))
    other = prediction_loss(model.heads,h,type_logits,Note.make('tap','A1',ex=1,break_=1,short_hold=1),stage,**kwargs)
    torch.testing.assert_close(loss.physical.total,other.physical.total)
    physical_mean(loss.physical).backward()
    assert type_logits.grad.abs().sum() > 0
    assert all(head.weight.grad is None for head in model.heads.flags)
    # Valid zero-loss units stay in the common denominator; chunks add sums/counts.
    no_conflict = deepcopy(stage)
    for group in ('types','kind','pad'):
        for r in no_conflict[group].values():
            r['penalty'] = r['physical'] = 0
    clean = prediction_loss(model.heads,h,type_logits,Note.make('tap','A1'),no_conflict,**kwargs)
    assert clean.physical.count == 2
    combined = loss + clean
    torch.testing.assert_close(physical_mean(combined.physical),physical_mean(loss.physical)/2)
    assert combined.main.count == 12


def geometry():
    return Geometry({p:PathGeometry(p,1.,((0.,1 << (int(p[0])-1)),))
                     for v in catalog() for p in v['paths_by_start']})


def test_actual_grammar_physical_stages_all_fields_and_backward():
    timing = TimingMap([(0,240)])
    tokens = (Moment(0), Note.make('tap','A1',short_hold=1), Note.make('hold','A2'), HoldDuration(2),
        Moment(timing.delta_ms(0,24)), Note.make('touch','C',short_touch_hold=1),
        Note.make('touch_hold','B1'), HoldDuration(4),
        Moment(timing.delta_ms(24,48)), Note.make('slide','A3'), Route(0,0), Route(1), SlideDuration(32),
        Route(70,1), SlideDuration(48))
    w = build_window(tokens,timing,sequence_start_tick=0,start_tick=0,end_tick=96,audio_after_beats=1,
        audio_end_seconds=1.,history_beats=4,history_dropout=0,training=True,rng=random.Random(1))
    attribution = AttributionState(PhysicalState(timing,geometry(),audio_end_seconds=1.))
    before = attribution.snapshot()
    stages = window_stages(w,attribution)
    assert attribution.snapshot() == before
    chain = next(s for s in stages if s and s['phase']=='chain' and 'route' in s)
    assert chain['route'][70]['basic']
    wifi_duration = stages[-2]
    assert wifi_duration['grammar_types'] == ['SLIDE_DURATION']
    model = build_decoder(config())
    batch, types, _ = collate_windows([w],device='cpu')
    _, audio, z, plan = inputs(model)
    z = type(z)(z.values[:1],z.beats[:1],z.valid[:1])
    memory = model.prepare_memory(model.prepare_audio(audio),torch.tensor([0]),z,plan[:1],torch.tensor([True]))
    output = model(batch,types,memory)
    distance = (torch.arange(71)[:,None]-torch.arange(71)[None,:]).abs().float()/70
    fixed = RouteGeometryLoss(distance,distance.square())
    loss = decoder_loss(model.heads,output,[w],[stages],[timing],[1.],
        distributions=decoder_time_distributions(config()),geometry=fixed)
    assert loss.judge.count == 3 and loss.trajectory.count == 3
    assert loss.physical.count > 10 and loss.main.count > loss.physical.count
    objective = loss.objective(physical_weight=.1,judge_weight=.2,trajectory_weight=.3)
    assert torch.isfinite(objective)
    objective.backward()
    for name in ('MOMENT','HOLD_DURATION','SLIDE_DURATION'):
        assert model.heads.times[name].weight.grad.abs().sum() > 0
    assert model.heads.route.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert not fixed.judge.requires_grad


def test_empty_tail_window_and_older_physical_survive_dropout():
    timing = TimingMap([(0,120)])
    tokens = (Moment(0),Note.make('hold','A1'),HoldDuration(960))
    w = build_window(tokens,timing,sequence_start_tick=0,start_tick=5*96,end_tick=6*96,audio_after_beats=1,
        audio_end_seconds=6.,history_beats=4,history_dropout=1,training=True,rng=random.Random(1))
    state = AttributionState(PhysicalState(timing,geometry(),audio_end_seconds=6.))
    state.inject_boundary(tokens[0],start_tick=0)
    for token in tokens[1:]:
        state.stage_and_accept(token)
    saved = state.snapshot()
    stages = window_stages(w,state)
    assert state.snapshot() == saved and state.physical.actions
    assert w.inputs == (Moment(0),) and len(stages) == 1
    assert stages[0]['cursor_tick'] == 480
    assert stages[0]['types']['MOMENT']['penalty'] == 0


def test_extreme_physical_mass_has_finite_gradients_and_fixed_four_reduction():
    logits = torch.tensor([1.,2.],requires_grad=True)
    lp = logits.log_softmax(-1)
    forbidden = torch.tensor([[True,False,True,False],[True,False,False,False]])
    losses = physical_categories(lp,forbidden)
    loss = physical_mean(Reduction(losses,1))
    loss.backward()
    assert torch.isfinite(losses).all() and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(loss,(losses[0]+4*losses[2])/4)


def test_unequal_window_chunks_share_global_denominators_and_gradients():
    torch.manual_seed(82)
    timing = TimingMap([(0,240)])
    tokens = (Moment(0), Note.make('tap','A1'), Note.make('slide','A2'), Route(0,0), SlideDuration(32))
    examples = [build_window(tokens,timing,sequence_start_tick=0,start_tick=start,end_tick=start+96,
        audio_after_beats=1,audio_end_seconds=1.,history_beats=4,history_dropout=drop,
        training=True,rng=random.Random(1)) for start, drop in ((0,0), (96,1))]
    assert len(examples[1].inputs) == 1 < len(examples[0].inputs)
    rules = AttributionState(PhysicalState(timing,geometry(),audio_end_seconds=1.))
    first = window_stages(examples[0],rules)
    rules.inject_boundary(tokens[0],start_tick=0)
    for token in tokens[1:]:
        rules.stage_and_accept(token)
    stages = [first, window_stages(examples[1],rules)]
    model = build_decoder(config())
    length = len(examples[0].inputs)
    output = SimpleNamespace(hidden=torch.randn(2,length,24,requires_grad=True),
                             type_logits=torch.randn(2,length,5,requires_grad=True))
    matrix = (torch.arange(71)[:,None]-torch.arange(71)[None,:]).abs().float()/70
    kwargs = dict(distributions=decoder_time_distributions(config()),geometry=RouteGeometryLoss(matrix,matrix.square()))
    full = decoder_loss(model.heads,output,examples,stages,[timing]*2,[1.]*2,**kwargs)
    chunks = [decoder_loss(model.heads,SimpleNamespace(hidden=output.hidden[i:i+1],
        type_logits=output.type_logits[i:i+1]),[example],[stages[i]],[timing],[1.],**kwargs)
        for i, example in enumerate(examples)]
    assert chunks[0].main.count > chunks[1].main.count == 2  # Router plus terminal time
    assert chunks[0].judge.count == 1 and chunks[1].judge.count == 0
    combined = chunks[0] + chunks[1]
    for name in ('main','physical','judge','trajectory'):
        a,b = getattr(full,name),getattr(combined,name)
        assert a.count == b.count
        torch.testing.assert_close(a.total,b.total)
    weights = dict(physical_weight=.3,judge_weight=.2,trajectory_weight=.1)
    objective = full.objective(**weights)
    # The training loop may backward each chunk separately, using logical-batch counts.
    scaled = [chunk.main.total/full.main.count
        + weights['physical_weight']*physical_mean(Reduction(chunk.physical.total,full.physical.count))
        + weights['judge_weight']*chunk.judge.total/full.judge.count
        + weights['trajectory_weight']*chunk.trajectory.total/full.trajectory.count for chunk in chunks]
    torch.testing.assert_close(objective,sum(scaled))
    assert not torch.isclose(objective,sum(c.objective(**weights) for c in chunks)/2)
    params = (output.hidden,output.type_logits,*tuple(model.heads.parameters()))
    expected = torch.autograd.grad(objective,params,allow_unused=True)
    for part in scaled:
        part.backward()
    for param, grad in zip(params,expected):
        if grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad,grad,atol=1e-6,rtol=1e-5)
    assert output.hidden.grad[1,1:].count_nonzero() == 0
    assert output.type_logits.grad[1,1:].count_nonzero() == 0


@pytest.mark.parametrize('frozen', [False, True])
def test_encoder_draft_planner_to_real_decoder_gradients(frozen):
    from test_draft_integration import small_chain
    from maidata.draft.windows import assemble_full_p0, select_windows
    c,timing,coverage,song,window,e,d,p,r,queries = small_chain()
    c = dict(c, **{k:v for k,v in config().items() if k.startswith('decoder_')})
    model = build_decoder(c)
    if frozen:
        for module in (e,d,p):
            module.requires_grad_(False)
    encoded = e(song,window)
    bank = assemble_full_p0(encoded.P0,window.target_start_beats,window.window_to_song,[coverage],p0_rate=c['fpn_rates'][0])
    z = d(bank,queries)
    local = select_windows(z,torch.tensor([0]),torch.tensor([0.]),torch.tensor([True]),
                           target_beats=3,before_beats=2,after_beats=1)
    predicted = p(local)
    batch = pack_tokens(
        [(Moment(0),Note.make('tap','A1'))],[(0,0)],device='cpu')
    memory = model.prepare_memory(model.prepare_audio(bank),torch.tensor([0]),local.memory,
                                   predicted.plan_values,predicted.valid)
    result = model(batch,torch.tensor([[1,0]]),memory)
    loss = model.heads.kind(result.hidden[0,0]).square().mean()+result.type_logits.square().mean()
    loss.backward()
    for module in (e,d,p):
        grads = [v.grad for v in module.parameters() if v.grad is not None]
        assert not grads if frozen else sum(float(g.abs().sum()) for g in grads)>0
    assert model.layers[0].self_attn.query.weight.grad.abs().sum()>0
    assert all(param.grad is None for param in r.parameters())
