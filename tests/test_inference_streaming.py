"""Live playable snapshots follow complete fields and transactional rollback."""
from fractions import Fraction
import os
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config, geometry, conditions
from test_inference_materialize import write_audio, native_miacode, check_roundtrip
from test_inference_render import native_dump
from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration, EncodedChart, decode_chart
from maidata.chart.grammar import Grammar
from maidata.chart.windows import TimeTarget
from maidata.decoder.session import DecodeSession
from maidata.decoder.windows import real_onsets
from maidata.inference.__main__ import ir_metadata
from maidata.inference.generation import generate, decode_window, GenerationOptions
from maidata.inference.materialize import atomic_text
from maidata.inference.render import SimaiWriter
from maidata.inference.sampling import Sampler
from maidata.inference.streaming import StreamingPreview
from maidata.model import build_decoder, decoder_time_distributions
from maidata.physical import PhysicalState
from maidata.time import TimingMap


def preview(tmp_path, timing, duration=20.):
    write_audio(tmp_path/'audio.wav')
    meta = ir_metadata(timing, duration)
    end = meta['render_end_beat']
    writer = SimaiWriter(meta['controls'], Fraction(end['num'], end['den']))
    return StreamingPreview(writer, timing, tmp_path/'audio.wav', tmp_path/'output',
                            difficulty=5, level=14.5, title='测试曲名')


def read_chart(live):
    return (live.last_text or '').split('&inote_5=')[-1].strip()


class Feed:
    """Mirror the public stream contract, with real grammar completion decisions."""
    def __init__(self, live, start_tick=0):
        self.live = live
        self.start_tick = start_tick
        self.grammar = Grammar(live.timing)
        self.grammar.inject_boundary(Moment(0), start_tick=start_tick)
        self.snapshots = [self.grammar.snapshot()]
        self.tokens = []
        live(dict(event='window_start', window=1, start_tick=start_tick))

    def accept(self, token):
        self.grammar.accept(token)
        self.tokens.append(token)
        self.snapshots.append(self.grammar.snapshot())
        self.live(dict(event='token', token=token, token_count=len(self.tokens),
                       complete=self.grammar.state.phase in ('done', 'track_done')))

    def rollback(self, count):
        self.grammar.restore(self.snapshots[count])
        del self.snapshots[count+1:]
        del self.tokens[count:]
        self.live(dict(event='rollback', token_count=count))

    def commit(self):
        groups = real_onsets((Moment(0), *self.tokens), self.live.timing,
                             sequence_start_tick=self.start_tick)
        self.live(dict(event='window_end', groups=groups))


@pytest.mark.parametrize('first', [.02, 0., -1.25])
def test_live_complete_fields_chains_branches_and_rollback(tmp_path, first, geometry, native_miacode):
    timing = TimingMap([(0,120),(Fraction(1,4),180)], first)
    live = preview(tmp_path, timing)
    with patch('maidata.inference.streaming.atomic_text', wraps=atomic_text) as writes:
        feed = Feed(live)
        assert '测试曲名' in live.last_text and '(180)' not in read_chart(live)
        feed.accept(Note.make('tap','A1'))
        tap = live.last_text
        feed.accept(Note.make('hold','A2'))
        assert live.last_text == tap
        feed.accept(HoldDuration(24))
        hold = live.last_text
        assert '2h[' in read_chart(live) and hold != tap
        feed.accept(Moment(timing.delta_ms(0,24)))
        feed.accept(Note.make('slide','A3'))
        feed.accept(Route(0,1))
        feed.accept(Route(2))
        assert live.last_text == hold
        feed.accept(SlideDuration(97))
        first_branch = live.last_text
        assert '(180)' in read_chart(live) and first_branch != hold
        def check_visible_prefix():
            expected = decode_chart(EncodedChart((Moment(0), *feed.tokens), ir_metadata(timing,20.)), geometry)
            last = real_onsets((Moment(0), *feed.tokens), timing, sequence_start_tick=0)[-1].tick
            expected['controls'] = [c for c in expected['controls'] if
                Fraction(c['position_beats']['num'], c['position_beats']['den']) <= Fraction(last,96)]
            check_roundtrip(expected, native_dump(native_miacode, read_chart(live), first))
        check_visible_prefix()
        feed.accept(Route(4,0))
        feed.accept(Route(0))
        assert live.last_text == first_branch
        feed.accept(SlideDuration(121))
        assert '*' in read_chart(live)
        check_visible_prefix()
        feed.rollback(8)
        assert live.last_text == first_branch
        # Drop a completed chain and replace it with a shorter one.
        feed.rollback(6)
        assert live.last_text == hold
        feed.accept(SlideDuration(48))
        assert live.last_text != first_branch
        check_visible_prefix()
        feed.rollback(1)
        assert live.last_text == tap
        feed.accept(Note.make('touch_hold','C'))
        assert live.last_text == tap
        feed.accept(HoldDuration(96))
        assert 'Ch[' in read_chart(live)
        assert writes.call_count == 10  # No writes for incomplete fields.
        check_visible_prefix()
        before = live.last_text
        feed.commit()
        assert live.last_text == before and live.writer.moments == 1
    assert (tmp_path/'output/maidata.txt').read_text(encoding='utf-8') == live.last_text
    assert (tmp_path/'output/track.wav').read_bytes() == (tmp_path/'audio.wav').read_bytes()


def test_committed_empty_windows_spacing_and_final_are_unchanged(tmp_path):
    timing = TimingMap([(0,120),(4,180)])
    live = preview(tmp_path, timing)
    tokens = [Note.make('tap','A1')]
    for tick in (24,48,72,1973):
        previous = 0 if len(tokens) == 1 else (tick-24 if tick != 1973 else 72)
        tokens.extend((Moment(timing.delta_ms(previous,tick)), Note.make('tap','A1')))
    # Commit contiguous windows, including an empty one.
    first = Feed(live)
    for token in tokens[:7]:
        first.accept(token)
    first.commit()
    same = live.last_text
    empty = Feed(live,16*96)
    empty.commit()
    assert live.last_text == same
    last = Feed(live,20*96)
    last.accept(Moment(timing.delta_ms(20*96,1973)))
    last.accept(tokens[-1])
    last.commit()
    assert live.writer.moments == 5
    expected = SimaiWriter(ir_metadata(timing,20.)['controls'], live.writer.end)
    expected.add_groups(real_onsets((Moment(0), *tokens), timing, sequence_start_tick=0))
    assert live.writer.finish() == expected.finish()
    assert '{16}(120)1,1,1,1' in live.last_text
    assert '\n' in read_chart(live) and '(180)' in read_chart(live)


def test_preview_does_not_consume_pending_controls_or_writer_state():
    timing = TimingMap([(0,120),(4,180)])
    writer = SimaiWriter(ir_metadata(timing,20.)['controls'], Fraction(40))
    before = vars(writer).copy()
    before['parts'] = writer.parts.copy()
    before['controls'] = writer.controls.copy()
    assert writer.preview() == '{384}(120),E'
    assert vars(writer) == before
    assert '(180)' in writer.finish()


@pytest.mark.parametrize('failures', [0,2,101])
def test_atomic_replacement_keeps_previous_complete_file(tmp_path, failures):
    path = tmp_path/'maidata.txt'
    path.write_text('old', encoding='utf-8')
    replace = os.replace
    calls = []
    def occupied(source, target):
        calls.append(1)
        assert path.read_text(encoding='utf-8') == 'old'
        assert source.read_text(encoding='utf-8') == 'new'
        if len(calls) <= failures:
            raise PermissionError('reader holds file')
        replace(source,target)
    with patch('maidata.inference.materialize.os.replace', occupied), \
         patch('maidata.inference.materialize.time.sleep') as sleep:
        if failures == 101:
            with pytest.raises(PermissionError):
                atomic_text(path,'new')
        else:
            atomic_text(path,'new')
        assert sleep.call_count == min(failures,100)
    assert path.read_text(encoding='utf-8') == ('old' if failures == 101 else 'new')
    assert not path.with_suffix('.txt.tmp').exists()


@pytest.mark.parametrize('rejected_kind', ['tap','hold','slide'])
def test_decode_stream_observes_window_before_completion_and_actual_rollback(config,geometry,tmp_path,rejected_kind):
    timing = TimingMap([(0,120)])
    model = build_decoder(config).eval()
    prepared = conditions(config,timing,5.)
    memory = model.prepare_memory(model.prepare_audio(prepared.audio),torch.tensor([0]),prepared.z,
                                   prepared.plans[:1],prepared.plan_valid[:1])
    session = DecodeSession(PhysicalState(timing,geometry,audio_end_seconds=5.))
    session.rules.inject_boundary(Moment(0),start_tick=0,window_end=288)
    live = preview(tmp_path,timing)
    events = []
    checking = False
    def observe(event):
        events.append(event)
        live(event)
    class Search:
        physical_hard_mask = True
        def choices(self,decoder,routed,oracle,clock,specs,**kwargs):
            phase = oracle.grammar.state.phase
            if phase == 'boundary':
                for token in (Note.make(rejected_kind,'A1'), Note.make('tap','A2')):
                    out = decoder.complete(routed,torch.ones_like(routed.valid,dtype=torch.long))
                    yield token,out.cache
            elif oracle.grammar.state.note.pad == 'A1':
                if checking:
                    expected = '1' if rejected_kind == 'tap' else ''
                    assert f'&inote_5={{384}}(120){expected},E' in live.last_text
                return  # Fail completed Tap or pending Hold/Slide support.
            else:
                if checking:
                    assert '&inote_5={384}(120)2,E' in live.last_text
                out = decoder.complete(routed,torch.zeros_like(routed.valid,dtype=torch.long))
                yield TimeTarget(2000.,384,True,True),out.cache
    with torch.inference_mode():
        # Same search without publishing; compare actual accepted fields and backtracking.
        reference, reference_report = decode_window(model,memory,session,Moment(0),Search(),
            decoder_time_distributions(config),timing,start_tick=0,end_tick=288,cap_tick=384,
            device='cpu',options=GenerationOptions())
        session = DecodeSession(PhysicalState(timing,geometry,audio_end_seconds=5.))
        session.rules.inject_boundary(Moment(0),start_tick=0,window_end=288)
        checking = True
        tokens, report = decode_window(model,memory,session,Moment(0),Search(),decoder_time_distributions(config),timing,
            start_tick=0,end_tick=288,cap_tick=384,device='cpu',options=GenerationOptions(),stream=observe)
    assert tokens == reference and report == reference_report
    assert tokens == (Note.make('tap','A2'),) and report['backtracks'] == 1
    assert [e['event'] for e in events] == ['window_start','token','rollback','token']
    assert events[2]['token_count'] == 0


@pytest.mark.parametrize('temperature', [0., .7])
def test_streaming_leaves_sampling_reports_rng_and_final_text_unchanged(config,geometry,tmp_path,temperature):
    timing = TimingMap([(0,120)])
    model = build_decoder(config).eval()
    prepared = conditions(config,timing,1.2)
    options = GenerationOptions(temperature=temperature, seed=17, max_tokens_per_moment=2,
                                physical_hard_mask=False)
    # Use real sampling but constrain note kinds to make this CPU test bounded.
    with torch.no_grad():
        model.heads.kind.weight.zero_()
        model.heads.kind.bias.fill_(-100)
        model.heads.kind.bias[0] = 100
        model.type_head.weight.zero_()
        model.type_head.bias.fill_(-100)
        model.type_head.bias[1] = 100
        head = model.heads.times['MOMENT']
        head.weight.zero_()
        head.bias.fill_(-100)
        head.bias[384] = 100
    samplers = []
    def sampler(**kwargs):
        value = Sampler(**kwargs)
        samplers.append(value)
        return value
    with patch('maidata.inference.generation.Sampler',side_effect=sampler):
        reference, reports = generate(model,prepared,timing,geometry,config,{},1.2,device='cpu',options=options)
        state = torch.get_rng_state().clone()
        live = preview(tmp_path,timing,1.2)
        with patch('torch.cuda.synchronize',side_effect=AssertionError('no preview synchronization')):
            observed, streamed_reports = generate(model,prepared,timing,geometry,config,{},1.2,
                                                   device='cpu',options=options,stream=live)
    assert observed == reference and streamed_reports == reports
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(samplers[0].generator.get_state(),samplers[1].generator.get_state())
    writer = SimaiWriter(ir_metadata(timing,1.2)['controls'],live.writer.end)
    writer.add_groups(real_onsets(reference.tokens,timing,sequence_start_tick=0))
    assert live.writer.finish() == writer.finish()
