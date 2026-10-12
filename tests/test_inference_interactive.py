from collections import OrderedDict
from fractions import Fraction
import re
import threading
from unittest.mock import patch

import pytest
import torch

from test_inference import config as config, conditions
from test_pretrain_runtime import config as producer_config
from test_chart_physical import geometry
from maidata.chart import Moment, Note, HoldDuration, token_from_dict
from maidata.chart.types import GENERATED_TYPES
from maidata.decoder.batching import pack_tokens
from maidata.decoder.sequence import family
from maidata.decoder.windows import Onset, real_onsets
from maidata.model import build_decoder, decoder_time_distributions
from maidata.time import TimingMap
from maidata.inference.__main__ import ir_metadata
from maidata.inference.interactive import (Interactive, Server, WindowContext, chart_groups,
    expanded_rows, extend_cache, rebase)
from maidata.inference.render import render_span


@pytest.fixture
def window_config(config):
    return dict(config, target_beats=16, target_stride_beats=16, chart_history_beats=4, audio_after_beats=4)


def stub(c, geometry, timing, duration):
    s = Interactive.__new__(Interactive)
    s.device, s.precision, s.timing, s.config = torch.device('cpu'), 'fp32', timing, c
    s.decoder = build_decoder(c).eval()
    s.duration, s.conditions = duration, conditions(c, timing, duration)
    with torch.inference_mode():
        s.audio_kv = s.decoder.prepare_audio(s.conditions.audio)
    s.geometry, s.distributions = geometry, decoder_time_distributions(c)
    s.window_ticks, s.contexts, s.max_contexts, s.rule_starts = c['target_beats']*96, OrderedDict(), 4, {}
    s.groups, s.skipped, s.revision = (), 0, 1
    return s


def full_cache(decoder, memory, rows):
    batch = pack_tokens([[r[0] for r in rows[:-1]]], [[r[1] for r in rows[:-1]]], device='cpu',
                        slide_positions=[[r[2] for r in rows[:-1]]])
    types = torch.tensor([[GENERATED_TYPES.index(family(r[0])) for r in rows[1:]]])
    return decoder(batch, types, memory).cache


def assert_same_kv(a, b):
    assert len(a.layers) == len(b.layers)
    for x, y in zip(a.layers, b.layers):
        assert x.valid.shape == y.valid.shape
        torch.testing.assert_close(x.key, y.key, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(x.value, y.value, atol=2e-5, rtol=2e-4)


def chart_rows(timing, tail):
    tokens = [Moment(0), Note.make('tap', 'A1'), Moment(timing.delta_ms(0, 96)),
              Note.make('hold', 'A3'), HoldDuration(500), *tail]
    beats, cursor = [], 0
    for token in tokens:
        if isinstance(token, Moment) and beats:
            cursor = timing.tick_after(cursor, token.delta_ms)
        beats.append(cursor / 96)
    return expanded_rows(tuple(tokens), tuple(beats), timing, 0)


def test_lcp_reuses_unchanged_rows_and_recomputes_only_the_suffix(window_config):
    timing = TimingMap([(0, 120)])
    decoder = build_decoder(window_config).eval()
    prepared = conditions(window_config, timing, 20.)
    with torch.inference_mode():
        memory = decoder.prepare_memory(decoder.prepare_audio(prepared.audio), torch.tensor([0]), prepared.z,
                                        prepared.plans[:1], prepared.plan_valid[:1])
        original = chart_rows(timing, [Moment(timing.delta_ms(96, 288)), Note.make('tap', 'A2'),
                                       Moment(timing.delta_ms(288, 384)), Note.make('tap', 'A6')])
        context = WindowContext(memory)
        assert extend_cache(decoder, context, original, device='cpu') == (0, len(original) - 1)
        assert_same_kv(context.cache, full_cache(decoder, memory, original))
        # Unchanged: everything is reused, nothing is computed.
        assert extend_cache(decoder, context, original, device='cpu') == (len(original) - 1, 0)

        # Same family at the first difference: the preceding entry survives.
        same = chart_rows(timing, [Moment(timing.delta_ms(96, 288)), Note.make('tap', 'A4'),
                                   Moment(timing.delta_ms(288, 384)), Note.make('tap', 'A6')])
        first = next(i for i, (a, b) in enumerate(zip(original, same)) if a != b)
        assert extend_cache(decoder, context, same, device='cpu') == (first, len(same) - 1 - first)
        assert_same_kv(context.cache, full_cache(decoder, memory, same))

        # A different following family invalidates the entry before the edit too.
        chord = chart_rows(timing, [Note.make('tap', 'A4'), Moment(timing.delta_ms(96, 384)), Note.make('tap', 'A6')])
        first = next(i for i, (a, b) in enumerate(zip(same, chord)) if a != b)
        assert family(same[first][0]) != family(chord[first][0])
        assert extend_cache(decoder, context, chord, device='cpu') == (first - 1, len(chord) - first)
        assert_same_kv(context.cache, full_cache(decoder, memory, chord))


def user_groups(timing):
    tokens = (Moment(0), Note.make('tap', 'A1'), Moment(timing.delta_ms(0, 96)), Note.make('tap', 'A5'),
              Moment(timing.delta_ms(96, 192)), Note.make('hold', 'A3'), HoldDuration(96),
              Moment(timing.delta_ms(192, 1920)), Note.make('tap', 'A2'))
    return real_onsets(tokens, timing, sequence_start_tick=0)


def test_continuation_respects_floor_budget_and_reuses_accepted_kv(window_config, geometry):
    timing = TimingMap([(0, 120)])
    session = stub(window_config, geometry, timing, timing.beat_to_second(40))
    session.groups = user_groups(timing)
    request = dict(id=1, revision=1, mode='continue', start_tick=384, end_tick=1920, max_tokens=4,
                   temperature=1., seed=7, content=True, boundaries=list(range(408, 1921, 24)))
    result = session.suggest(request, threading.Event())
    assert result['reused_rows'] == 0 and result['computed_rows'] > 0
    assert all(384 <= o['tick'] < 1920 for o in result['onsets'])
    assert result['onsets'], 'fixture seed should generate content'
    assert result['tokens'] >= 4
    assert result['end_tick'] in request['boundaries'] and result['end_tick'] > result['onsets'][-1]['tick']
    assert result['text'].startswith('{')

    # Accept: the chart now holds the suggestion. Its whole prefix is cached.
    generated = session.contexts[0].rows
    session.groups = rebase(sorted(session.groups + tuple(
        Onset(o['tick'], 0., tuple(token_from_dict(t) for t in o['content'])) for o in result['onsets']),
        key=lambda g: g.tick), timing)
    following = dict(request, id=2, start_tick=result['end_tick'])
    again = session.suggest(following, threading.Event())
    assert again['computed_rows'] == 0 and again['reused_rows'] == len(generated) - 1
    assert all(o['tick'] >= result['end_tick'] for o in again['onsets'])


def test_range_replaces_only_the_selected_interval(window_config, geometry):
    timing = TimingMap([(0, 120)])
    session = stub(window_config, geometry, timing, timing.beat_to_second(40))
    session.groups = user_groups(timing)
    result = session.suggest(dict(id=3, revision=1, mode='range', start_tick=96, end_tick=1632,
                                  temperature=1., seed=11), threading.Event())
    assert (result['start_tick'], result['end_tick']) == (96, 1632)
    assert all(96 <= o['tick'] < 1632 for o in result['onsets'])
    assert span_beats(result['text']) == Fraction(1632 - 96, 96)


@pytest.mark.parametrize('mode', ['continue', 'range'])
def test_interactive_random_seed_is_resolved_per_request_and_reported(window_config, geometry, mode):
    timing = TimingMap([(0, 120)])
    session = stub(window_config, geometry, timing, timing.beat_to_second(40))
    request = dict(id=1, revision=1, mode=mode, start_tick=384, end_tick=768,
                   max_tokens=4, temperature=1., seed=-1)
    with patch('maidata.inference.options.secrets.randbelow', side_effect=[7, 11]) as entropy:
        first = session.suggest(request, threading.Event())
        second = session.suggest(dict(request, id=2), threading.Event())
    assert (first['seed'], second['seed']) == (7, 11)
    assert entropy.call_count == 2


def span_beats(text):
    total, subdivision, depth = Fraction(0), 4, 0
    for match in re.finditer(r'\[|\]|\{(\d+)\}|,', text):
        token = match[0]
        if token == '[':
            depth += 1
        elif token == ']':
            depth -= 1
        elif depth == 0 and token.startswith('{'):
            subdivision = int(match[1])
        elif depth == 0 and token == ',':
            total += Fraction(4, subdivision)
    return total


def test_render_span_covers_the_exact_range_with_inner_controls():
    groups = (Onset(384, 0., (Note.make('tap', 'A1'),)), Onset(408, 0., (Note.make('tap', 'A2'),)),
              Onset(600, 0., (Note.make('hold', 'A3'), HoldDuration(48))))
    controls = [dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1)),
                dict(kind='bpm', bpm=150, position_beats=dict(num=6, den=1))]
    text = render_span(groups, 384, 768, controls)
    assert text.startswith('{') and '(150)' in text and '(120)' not in text and 'E' not in text
    assert span_beats(text) == 4
    assert render_span((), 0, 96 * 3) .count(',') == 3


def test_editor_ir_skips_unrepresentable_notes_and_rejects_changed_timing():
    timing = TimingMap([(0, 120)])
    ir = dict(ir_metadata(timing, 20.), moments=[
        dict(position_beats=dict(num=1, den=1), groups=[dict(notes=[dict(kind='tap', pad='A1', flags=[])])]),
        dict(position_beats=dict(num=2, den=1), groups=[dict(notes=[dict(kind='slide', start_pad='A1', flags=[],
            head=dict(kind='star', flags=[]), wait=dict(basis='beat', beat_num=1, beat_den=2), branches=[])])]),
        dict(position_beats=dict(num=3, den=1), groups=[dict(notes=[dict(kind='hold', pad='A3', flags=[],
            duration=dict(basis='beat', beat_num=1, beat_den=1))])]),
    ])
    groups, skipped = chart_groups(ir, timing, 20.)
    assert [g.tick for g in groups] == [96, 288] and skipped == 1
    assert groups[1].delta_ms == pytest.approx(timing.delta_ms(96, 288))
    with pytest.raises(ValueError, match='restart the session'):
        chart_groups(ir, TimingMap([(0, 130)]), 20.)


def test_server_newer_suggest_cancels_running_and_pending():
    events, started = [], threading.Event()

    class Session:
        revision = None
        def suggest(self, request, cancelled):
            if request['id'] == 1:
                started.set()
                assert cancelled.wait(5)
                from maidata.inference.interactive import Cancelled
                raise Cancelled()
            return dict(id=request['id'], text='1,')

    server = Server(Session(), lambda event, **fields: events.append((event, fields.get('id'))))
    worker = threading.Thread(target=server.work)
    worker.start()
    server.submit(dict(op='suggest', id=1))
    assert started.wait(5)
    server.submit(dict(op='suggest', id=2))
    server.submit(dict(op='suggest', id=3))
    server.submit(dict(op='close'))
    worker.join(5)
    assert ('cancelled', 1) in events and ('cancelled', 2) in events and ('result', 3) in events
    assert ('result', 2) not in events
