"""Sampling-only chain length bounds, using existing judge-area geometry."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config, conditions
from test_chart_physical import geometry
from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.chart.routes import route_for
from maidata.chart.types import GENERATED_TYPES
from maidata.decoder.batching import pack_tokens
from maidata.model import build_decoder, decoder_time_distributions
from maidata.physical.inference import InferencePhysicalState
from maidata.inference.options import GenerationOptions
from maidata.inference.sampling import Sampler
from maidata.inference.__main__ import parser
from maidata.time import TimingMap


def prefix(geometry):
    clock=TimingMap([(0,120)])
    state=InferencePhysicalState(clock,geometry,audio_end_seconds=12.)
    state.inject_boundary(Moment(0),start_tick=0,window_end=768)
    state.accept_source(Note.make('slide','A1'))
    state.accept_source(Route(route_for('1-3'),0))
    return state,clock


def test_judge_areas_merge_only_shared_chain_junction_and_restore(geometry):
    state,_=prefix(geometry)
    assert Sampler.slide_judge_area_count(state)==2
    saved=state.snapshot()
    state.accept_source(Route(route_for('3-5')))
    assert Sampler.slide_judge_area_count(state)==3
    state.restore(saved)
    assert Sampler.slide_judge_area_count(state)==2
    # One area may list multiple pads; it still counts as one judge area.
    path=geometry.paths['3-5']
    geometry.paths['3-5']=replace(path,entries=((0.,1<<5),(.5,(1<<6)|(1<<7)),(1.,1<<4)))
    state.accept_source(Route(route_for('3-5')))
    assert Sampler.slide_judge_area_count(state)==5  # unshared junction is retained


@pytest.mark.parametrize('hard_mask',[True,False])
def test_crossing_limit_allows_segment_then_requires_duration(config,geometry,hard_mask):
    state,clock=prefix(geometry)
    model=build_decoder(config).eval()
    prepared=conditions(config,clock,12.)
    memory=model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),
                                prepared.z,prepared.plans[:1],prepared.plan_valid[:1])
    with torch.no_grad():
        model.type_head.weight.zero_(); model.type_head.bias.fill_(-100)
        model.type_head.bias[GENERATED_TYPES.index('ROUTE')]=100
        model.type_head.bias[GENERATED_TYPES.index('SLIDE_DURATION')]=90
        for head in model.heads.times.values():
            head.weight.zero_();head.bias.fill_(-100);head.bias[95]=100
    sampler=Sampler(temperature=0,max_slide_judge_areas=2,physical_hard_mask=hard_mask)
    def sample(last):
        routed=model.begin(pack_tokens([[last]],[[0.]],device='cpu'),memory)
        with patch.object(model,'complete',wraps=model.complete) as complete:
            before=state.snapshot()
            result=next(sampler.choices(model,routed,state,clock,decoder_time_distributions(config),
                                       end_tick=768,cap_tick=1152))[0]
            assert state.snapshot()==before
            assert complete.call_args_list[0].args[1].item()==GENERATED_TYPES.index(result.type)
        return result
    with torch.inference_mode():
        assert Sampler.slide_judge_area_count(state)==2
        assert isinstance(sample(Route(route_for('1-3'),0)),Route)  # equality may extend
        extra=Route(route_for('3-5'));state.accept_source(extra)
        assert Sampler.slide_judge_area_count(state)==3
        duration=sample(extra)
        assert isinstance(duration,SlideDuration) and duration.ticks>0
        state.accept_source(duration)
        # The next parallel branch starts a fresh count and can sample a route.
        following=sample(duration)
        assert isinstance(following,Route) and following.track_break is not None
        state.accept_source(following)
        assert Sampler.slide_judge_area_count(state)==2
        assert isinstance(sample(following),Route)


@pytest.mark.parametrize('value',[0,-1,1.5,True])
def test_invalid_slide_limit(value):
    for cls in (GenerationOptions,Sampler):
        with pytest.raises(ValueError,match='max_slide_judge_areas'):
            cls(max_slide_judge_areas=value)


def test_operational_defaults_and_cli():
    options=GenerationOptions()
    assert options.temperature==.4 and Sampler().temperature==.4
    assert options.max_slide_judge_areas==40
    assert options.repetition_settings==dict(pad_repetition_window=32,pad_repetition_penalty=1.,
                                            route_repetition_window=32,route_repetition_penalty=1.)
    required=['--joint-checkpoint','x','--audio','x','--bpm','120','--mel-stats','x',
              '--geometry','x','--slide-data','x','--difficulty','5','--level','13','--output','x']
    assert parser().parse_args(required).max_slide_judge_areas==40
    args=parser().parse_args(required+['--max-slide-judge-areas','17'])
    assert args.max_slide_judge_areas==17 and args.temperature==.4
