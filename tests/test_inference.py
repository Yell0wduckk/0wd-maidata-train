"""Current fields across windows, real E/D/P/M execution, rollback and sampling."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_pretrain_runtime import config as producer_config, item
from test_decoder_network import config as decoder_config
from test_chart_physical import geometry
from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration, decode_chart, token_to_dict
from maidata.chart.types import GENERATED_TYPES
from maidata.chart.windows import TimeTarget
from maidata.decoder.batching import pack_tokens
from maidata.decoder.session import DecodeSession
from maidata.decoder.windows import Onset, history_prefix, real_onsets
from maidata.draft.types import BeatMemory
from maidata.draft.windows import song_coverage
from maidata.model import build_model, build_draft, build_planner, build_decoder, decoder_time_distributions
from maidata.physical import PhysicalState
from maidata.physical.attribution import AttributionState
from maidata.time import TimingMap
from maidata.inference.checkpoint import load_models
from maidata.inference.conditioning import prepare_conditions, Conditions
from maidata.inference.generation import generate, prefill, decode_window, GenerationOptions, GenerationError
from maidata.inference.sampling import Sampler
from maidata.inference.__main__ import ir_metadata


@pytest.fixture
def config(producer_config):
    c = dict(producer_config)
    c.update({k: v for k, v in decoder_config().items() if k.startswith('decoder_')})
    c['target_stride_beats'] = c['target_beats']
    return c


def conditions(c, timing, duration):
    coverage = song_coverage(timing, duration, first_beat=0, target_beats=c['target_beats'])
    def memory(length, width):
        return BeatMemory(torch.randn(1, length, width), torch.arange(length).float()[None]+.5,
                          torch.ones(1, length, dtype=torch.bool))
    starts = coverage.starts(include_tail=True)
    return Conditions(memory(11, c['d_model']), memory(coverage.draft_beats, c['draft_latent_dim']),
        torch.rand(len(starts), 12), torch.ones(len(starts), dtype=torch.bool), starts, coverage)


def test_checkpoint_and_full_e_d_p_m_no_training_or_hash_dependencies(config, geometry, tmp_path):
    c = config
    modules = [build_model(c), build_draft(c), build_planner(c), build_decoder(c)]
    first = dict(schema='maidata.encoder-draft-training.v1', recipe=c, source={'commit': 'old-code'},
        model_state={name+'.'+key: value for name, m in zip(('encoder','draft','planner'), modules)
                     for key, value in m.state_dict().items()})
    second = dict(schema='maidata.decoder-training.v1', recipe=c, source={'commit': 'different-code'},
                  model_state=modules[3].state_dict())
    torch.save(first, tmp_path/'producer.ckpt')
    torch.save(second, tmp_path/'decoder.ckpt')
    models = load_models(tmp_path/'producer.ckpt', tmp_path/'decoder.ckpt', device='cpu')
    assert all(not m.training for m in (models.encoder,models.draft,models.planner,models.decoder))
    assert not any(p.requires_grad for p in models.encoder.parameters())
    # Force terminal tails without replacing the actual networks or probability heads.
    with torch.no_grad():
        models.decoder.type_head.weight.zero_()
        models.decoder.type_head.bias.fill_(-100)
        models.decoder.type_head.bias[0] = 100
        time_head = models.decoder.heads.times['MOMENT']
        time_head.weight.zero_()
        time_head.bias.reshape(-1, 3)[:, 1] = 100
    timing = TimingMap([(0,120)])
    grid = item(0,c)['grid']
    with patch.object(models.encoder, 'prepare_song', wraps=models.encoder.prepare_song) as e, \
         patch.object(models.draft, 'forward', wraps=models.draft.forward) as d, \
         patch.object(models.planner, 'forward', wraps=models.planner.forward) as p:
        prepared = prepare_conditions(models.encoder, models.draft, models.planner, c, grid,
            timing, 1.2, difficulty=5, level=14.5, device='cpu', encoder_window_batch=1)
        result, windows = generate(models.decoder, prepared, timing, geometry, c, ir_metadata(timing,1.2),
                                    1.2, device='cpu', options=GenerationOptions(temperature=0))
        assert (e.call_count,d.call_count,p.call_count)==(1,1,1)
    assert result.tokens == (Moment(0),)
    assert len(windows)==2 and all(w['censored'] for w in windows)
    assert all(w['boundary_delta_ms']==0 for w in windows)
    assert prepared.z.valid.all() and prepared.plans.shape==(2,12)
    assert not torch.equal(prepared.plans,torch.zeros_like(prepared.plans))
    assert not any('trace' in w for w in windows)


def test_new_fields_history_empty_windows_and_cross_window_actions(config, geometry):
    c = dict(config,target_beats=16,target_stride_beats=16,chart_history_beats=4,audio_after_beats=4)
    timing = TimingMap([(0,120),(24,180)])
    duration = timing.beat_to_second(63)
    prepared = conditions(c,timing,duration)
    decoder = build_decoder(c).eval()
    seen, active = [], []
    # Onsets 2, 15, 16 and 20; then consecutive empty windows. Long hold and
    # complete slide fields survive the boundary; no copied continuation NOTE.
    rows = {
        0: [Moment(timing.delta_ms(0,192)),Note.make('tap','A3'),
            Moment(timing.delta_ms(192,1440)), Note.make('slide','A1'),Route(0,1),SlideDuration(192)],
        16: [Note.make('hold','A7'),HoldDuration(30*96),
             Moment(timing.delta_ms(16*96,20*96)),Note.make('tap','A4')],
    }
    cursors = {start:0 for start in prepared.starts}
    def scripted(self, model, routed, oracle, clock, distributions, *, end_tick, cap_tick,
                 moment_remaining=None):
        start = end_tick//96-16
        if cursors[start]==0:
            active.append((start,tuple(oracle.physical.actions)))
        sequence = rows.get(start,[])
        j = cursors[start]
        cursors[start] += 1
        terminal_tick = end_tick if start==0 else cap_tick
        token = sequence[j] if j < len(sequence) else TimeTarget(
            clock.delta_ms(oracle.physical.grammar.state.cursor,terminal_tick),terminal_tick,start!=0,True)
        kind = 0 if isinstance(token,TimeTarget) else GENERATED_TYPES.index(token.type)
        out = model.complete(routed,torch.full_like(routed.valid,kind,dtype=torch.long))
        yield token,out.cache
    original_prefill = prefill
    def capture(model,memory,groups,clock,start_tick,history_beats,**kwargs):
        seen.append((start_tick,history_prefix(groups,clock,start_tick=start_tick,
                     history_beats=history_beats,dropped=False)))
        return original_prefill(model,memory,groups,clock,start_tick,history_beats,**kwargs)
    # Explicitly disable Physical hard rejection for this synthetic long-action
    # fixture; bypasses remain recorded. Actual rule state is still used.
    with patch.object(Sampler,'choices',scripted), patch('maidata.inference.generation.prefill',capture):
        result, windows = generate(decoder,prepared,timing,geometry,c,ir_metadata(timing,duration),duration,
            device='cpu',options=GenerationOptions(temperature=0,physical_hard_mask=False))
    groups = real_onsets(result.tokens,timing,sequence_start_tick=result.start_tick)
    assert [g.tick for g in groups]==[192,1440,1536,1920]
    assert groups[2].delta_ms==pytest.approx(timing.delta_ms(1440,1536))
    history16, coords16, boundary16, last16 = seen[1][1]
    assert history16==(Moment(timing.delta_ms(192,1440)),Note.make('slide','A1'),Route(0,1),SlideDuration(192))
    assert coords16==(15.,)*4 and last16==1440
    assert boundary16.delta_ms==pytest.approx(timing.delta_ms(1440,1536))
    for index in (2,3,4):
        start, (history,coords,boundary,last) = seen[index]
        assert history==coords==() and last==1920
        assert boundary.delta_ms==pytest.approx(timing.delta_ms(1920,start))
    assert any(a.track for start,actions in active if start==16 for a in actions)
    assert any(a.kind=='hold' for start,actions in active if start==32 for a in actions)
    assert len(windows)==5  # Includes full empty tail, despite earlier empty windows.
    assert windows[0]['terminal_tick']==16*96 and not windows[0]['censored']
    ir = decode_chart(result,geometry)
    hold = ir['moments'][2]['groups'][0]['notes'][0]
    assert hold['duration']['beat_num']==30


def test_encoder_window_chunking_keeps_the_same_full_song_conditions(config):
    c=config
    modules=[build_model(c).eval(),build_draft(c).eval(),build_planner(c).eval()]
    grid=item(0,c)['grid']
    grid={k:(np.concatenate([v,v]) if v.ndim else v) for k,v in grid.items()}
    grid['beat_centers']=np.arange(6,dtype=np.float32)+.5
    grid['grid_k']=np.arange(30).reshape(6,5)
    grid['sample_times_sec']=grid['grid_k']/10
    grid['sample_indices']=np.rint(grid['sample_times_sec']*24000).astype(np.int64)
    grid['sample_valid']=grid['sample_times_sec']<2.7
    timing=TimingMap([(0,120)])
    a=prepare_conditions(*modules,c,grid,timing,2.7,difficulty=5,level=14.5,device='cpu',encoder_window_batch=1)
    b=prepare_conditions(*modules,c,grid,timing,2.7,difficulty=5,level=14.5,device='cpu',encoder_window_batch=8)
    for x,y in ((a.audio.values,b.audio.values),(a.z.values,b.z.values),(a.plans,b.plans)):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
    assert a.starts==(0,3,6) and a.z.valid.all()
    assert a.audio.values.shape[1]==6*c['fpn_rates'][0]


def test_prefill_targets_are_following_new_field_types(config):
    model = build_decoder(config).eval()
    prepared = conditions(config,TimingMap([(0,120)]),3)
    z = prepared.z
    memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),z,
                                   prepared.plans[:1],prepared.plan_valid[:1])
    groups = [Onset(96,500.,(Note.make('hold','A1'),HoldDuration(900)))]
    timing = TimingMap([(0,120)])
    with torch.inference_mode():
        boundary,cache,count = prefill(model,memory,groups,timing,192,4,device='cpu')
        tokens = pack_tokens([[Moment(500),Note.make('hold','A1'),HoldDuration(900),boundary]],[[1,1,1,2]],device='cpu')
        types = torch.tensor([[1,3,0,1]])
        full = model(tokens,types,memory)
        step = model.complete(model.begin(tokens.slice(3,4),memory,cache=cache),types[:,3:])
    torch.testing.assert_close(full.hidden[:,-1:],step.hidden,atol=2e-6,rtol=2e-5)
    assert count==3


def test_rollback_restores_pending_fields_actions_and_model_cache(config,geometry):
    model = build_decoder(config).eval()
    timing = TimingMap([(0,120)])
    prepared = conditions(config,timing,5)
    memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),prepared.z,
                                   prepared.plans[:1],prepared.plan_valid[:1])
    session = DecodeSession(PhysicalState(timing,geometry,audio_end_seconds=5))
    session.rules.inject_boundary(Moment(0),start_tick=0,window_end=288)
    states = []
    class Search:
        physical_hard_mask = True
        def choices(self,decoder,routed,oracle,clock,specs,**kwargs):
            state = oracle.physical.grammar.state
            states.append((state.phase,len(oracle.physical.actions),routed.previous))
            if state.phase=='boundary':
                for kind in ('hold','tap'):
                    out = decoder.complete(routed,torch.ones_like(routed.valid,dtype=torch.long))
                    yield Note.make(kind,'A1'),out.cache
            elif state.phase=='hold':
                return  # Simulated exhausted duration support.
            else:
                out=decoder.complete(routed,torch.zeros_like(routed.valid,dtype=torch.long))
                yield TimeTarget(2000.,384,True,True),out.cache
    with torch.inference_mode():
        tokens,report = decode_window(model,memory,session,Moment(0),Search(),decoder_time_distributions(config),timing,
            start_tick=0,end_tick=288,cap_tick=384,device='cpu',options=GenerationOptions())
    assert tokens==(Note.make('tap','A1'),) and report['backtracks']==1
    assert len(session.rules.actions)==1 and session.rules.grammar.state.phase=='closed'
    assert states[1][0]=='hold' and states[1][1]==0
    assert states[2][0]=='done' and states[2][1]==1


def test_sampler_mask_temperature_and_reproducible_retry():
    logits=torch.tensor([1.,2.,3.,4.])
    assert list(Sampler(temperature=0).order(logits,[True,False,True,False]))==[2,0]
    assert list(Sampler(seed=47).order(logits))==list(Sampler(seed=47).order(logits))
    assert list(Sampler().order(logits,[False]*4))==[]
    assert list(Sampler(temperature=1e-40).order(logits))==[3,2,1,0]
    with pytest.raises(ValueError,match='nonfinite'):
        list(Sampler().order(torch.tensor([float('nan')])))


def test_actual_field_sampling_flags_route_and_time(config,geometry):
    model=build_decoder(config).eval()
    timing=TimingMap([(0,120)])
    physical=PhysicalState(timing,geometry,audio_end_seconds=2.)
    physical.inject_boundary(Moment(0),start_tick=0,window_end=288)
    oracle=AttributionState(physical)
    sampler=Sampler(temperature=0)
    h=torch.zeros(config['decoder_dim'])
    specs=decoder_time_distributions(config)
    with torch.inference_mode():
        note=next(sampler.contents(model,h,'NOTE',oracle,timing,specs,end_tick=288,cap_tick=384))
        assert physical.query(note).allowed
        physical.accept(Note.make('slide','A1'))
        route=next(sampler.contents(model,h,'ROUTE',oracle,timing,specs,end_tick=288,cap_tick=384))
        assert route.track_break is not None
        physical.accept(Route(70,0))
        assert list(oracle.query_types())==['SLIDE_DURATION']
        duration=next(sampler.contents(model,h,'SLIDE_DURATION',oracle,timing,specs,end_tick=288,cap_tick=384))
        assert physical.query(duration).allowed


def test_limit_fails_without_silent_truncation(config,geometry):
    model=build_decoder(config).eval()
    timing=TimingMap([(0,120)])
    prepared=conditions(config,timing,1.)
    def endless(self,model,routed,oracle,clock,specs,**kwargs):
        out=model.complete(routed,torch.ones_like(routed.valid,dtype=torch.long))
        yield Note.make('tap','A1'),out.cache
    with patch.object(Sampler,'choices',endless), pytest.raises(GenerationError,match='token limit'):
        generate(model,prepared,timing,geometry,config,ir_metadata(timing,1),1.,device='cpu',
            options=GenerationOptions(max_tokens_per_window=1,physical_hard_mask=False))


def test_moment_cap_forces_advance_and_appends_required_completion(config, geometry):
    model = build_decoder(config).eval()
    timing = TimingMap([(0,120)])
    prepared = conditions(config,timing,2.)
    memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),
        prepared.z,prepared.plans[:1],prepared.plan_valid[:1])
    physical = PhysicalState(timing,geometry,audio_end_seconds=2.)
    physical.inject_boundary(Moment(0),start_tick=0,window_end=288)
    oracle = AttributionState(physical)
    sampler = Sampler(temperature=0,physical_hard_mask=False)
    specs = decoder_time_distributions(config)
    h = torch.zeros(config['decoder_dim'])
    with torch.inference_mode():
        routed = model.begin(pack_tokens([[Moment(0)]],[[0.]],device='cpu'),memory)
        token,_ = next(sampler.choices(model,routed,oracle,timing,specs,
            end_tick=288,cap_tick=384,moment_remaining=0))
        assert isinstance(token,(Moment,TimeTarget))
        # A chain must finish with duration instead of appending another route.
        physical.accept(Note.make('slide','A1'))
        token,_ = next(sampler.choices(model,routed,oracle,timing,specs,
            end_tick=288,cap_tick=384,moment_remaining=0))
        assert isinstance(token,Route)
        physical.accept(Route(0,0))
        token,_ = next(sampler.choices(model,routed,oracle,timing,specs,
            end_tick=288,cap_tick=384,moment_remaining=0))
        assert isinstance(token,SlideDuration)


def test_generation_caps_repeated_notes_at_one_onset(config, geometry):
    model = build_decoder(config).eval()
    timing = TimingMap([(0,120)])
    prepared = conditions(config,timing,1.)
    with torch.no_grad():
        model.type_head.weight.zero_()
        model.type_head.bias.fill_(-100)
        model.type_head.bias[GENERATED_TYPES.index('NOTE')] = 100
        model.heads.kind.weight.zero_()
        model.heads.kind.bias.fill_(-100)
        model.heads.kind.bias[0] = 100
        head = model.heads.times['MOMENT']
        head.weight.zero_()
        head.bias.reshape(-1,3)[:,1] = 100
    result,windows = generate(model,prepared,timing,geometry,config,ir_metadata(timing,1.),1.,
        device='cpu',options=GenerationOptions(temperature=0,max_tokens_per_moment=3,
                                              physical_hard_mask=False))
    groups = real_onsets(result.tokens,timing,sequence_start_tick=result.start_tick)
    assert groups and all(len(g.content) <= 3 for g in groups)
    assert any(len(g.content) == 3 for g in groups)


@pytest.mark.parametrize('basic_error',[False,True])
def test_exhausted_physical_search_bypasses_only_physical_and_resets_next_window(config,geometry,basic_error):
    from maidata.physical.state import Decision
    from maidata.physical.inference import InferencePhysicalState
    model = build_decoder(config).eval()
    timing = TimingMap([(0,120)])
    prepared = conditions(config,timing,1.)
    starts = []
    query = InferencePhysicalState.query
    def reject(self,token,**kwargs):
        if isinstance(token,Note):
            return Decision(('synthetic_invalid',)) if basic_error else Decision(physical=frozenset({'slide_too_fast'}))
        return query(self,token,**kwargs)
    def choices(self,model,routed,oracle,clock,specs,**kwargs):
        if oracle.grammar.state.phase == 'boundary':
            starts.append(self.physical_hard_mask)
            if self.physical_hard_mask:
                return
            token,index = Note.make('tap','A1'),GENERATED_TYPES.index('NOTE')
        else:
            token = TimeTarget(clock.delta_ms(oracle.grammar.state.cursor,kwargs['end_tick']),
                               kwargs['end_tick'],False,True)
            index = GENERATED_TYPES.index('MOMENT')
        output = model.complete(routed,torch.full_like(routed.valid,index,dtype=torch.long))
        yield token,output.cache
    with patch.object(InferencePhysicalState,'query',reject),patch.object(Sampler,'choices',choices):
        if basic_error:
            with pytest.raises(GenerationError,match='backtracking budget'):
                generate(model,prepared,timing,geometry,config,ir_metadata(timing,1.),1.,device='cpu',
                    options=GenerationOptions(temperature=0,max_backtracks_per_window=0))
        else:
            result,windows = generate(model,prepared,timing,geometry,config,ir_metadata(timing,1.),1.,device='cpu',
                options=GenerationOptions(temperature=0,max_backtracks_per_window=0))
            assert starts == [True,False]*len(windows)
            assert all(w['physical_fallback']['reason']=='backtracking_budget_exhausted' for w in windows)
            assert all(w['physical_bypass'][0]['reasons']==['slide_too_fast'] for w in windows)

