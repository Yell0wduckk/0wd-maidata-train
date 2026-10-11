"""Independent review probes for prefix isolation and model-only CONT state."""
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch
import random

import pytest
import torch

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.chart.windows import TimeTarget
from maidata.decoder.batching import pack_tokens
from maidata.decoder.types import Memory
from maidata.decoder.windows import build_window, real_onsets
from maidata.decoder_training.export_windows import compile_window
from maidata.decoder_training.support_cache import ChartSupports
from maidata.decoder_training.window_cache import tensor_pack, tensor_unpack
from maidata.draft.chart_data import ChartData, OnsetGroup, from_record
from maidata.draft.targets import reconstruction_target
from maidata.inference.sampling import Sampler
from maidata.model import build_decoder
from maidata.physical import PhysicalState
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.attribution import AttributionState, Evidence, Evaluation
from maidata.physical.conflicts import Conflict
from maidata.time import TimingMap, duration_end
from maidata.time.discrete import CONTINUE, compile_support_tree
from maidata.time.distribution import complete_field_log_probabilities, physical_categories
from test_decoder_network import config as network_config
from test_decoder_supervision import geometry
from test_decoder_training import fields as source_fields
from test_draft_data import record as draft_record


def model_prefix(kind='hold', origin=0):
    torch.manual_seed(944)
    model = build_decoder(network_config()).eval()
    audio = Memory(torch.randn(1, 24, 18), torch.arange(24)[None].float() + .5,
                   torch.ones(1, 24, dtype=torch.bool))
    z = Memory(torch.randn(1, 24, 10), torch.arange(24)[None].float() + .5,
               torch.ones(1, 24, dtype=torch.bool))
    memory = model.prepare_memory(model.prepare_audio(audio), torch.tensor([0]), z,
                                  torch.zeros(1, 12), torch.ones(1, dtype=torch.bool))
    tokens = pack_tokens([[Moment(0), Note.make(kind, 'A1')]],
                          [[origin/96, origin/96]], device='cpu')
    routed = model.begin(tokens, memory)
    types = torch.tensor([[1, 3]])
    return model, routed, types


def test_router_query_metadata_isolation_and_main_kv_invariance():
    model, routed, types = model_prefix()
    original_logits = routed.logits.clone()
    output = model.complete(routed, types)
    alternative = model.complete(routed, (types+2) % 5)
    torch.testing.assert_close(output.type_logits, alternative.type_logits, rtol=0, atol=0)
    keys = tuple(k.key.clone() for k in output.cache.layers)
    # Changing field count, position, role and elapsed cannot reach begin().
    for beats, roles in (([0.], [0]), ([8., 2., 5.], [3, 1, 4])):
        count = len(beats)
        output.field_reader(output, torch.tensor([[0, 1]]*count), torch.tensor(beats),
            torch.tensor(roles), torch.zeros(count, dtype=torch.long),
            torch.arange(count)*384)
    torch.testing.assert_close(routed.logits, original_logits, rtol=0, atol=0)
    for before, after in zip(keys, output.cache.layers):
        torch.testing.assert_close(before, after.key, rtol=0, atol=0)
        assert after.valid.shape[1] == 2


def test_generated_duration_cont_only_moves_temporary_query():
    model, routed, types = model_prefix()
    output = model.complete(routed, types)
    timing = TimingMap([(0, 60), (4, 120)])
    oracle = InferencePhysicalState(timing, geometry(), audio_end_seconds=20.)
    oracle.inject_boundary(Moment(0), start_tick=0)
    oracle.accept_source(Note.make('hold', 'A1'))
    before = oracle.snapshot()
    cache = tuple(k.key.clone() for k in output.cache.layers)
    selected = [CONTINUE, CONTINUE, 0]
    positions = []
    def logits(hidden, kind):
        result = torch.full((385,), -1000.)
        result[selected.pop(0)] = 1000.
        return result
    reader = output.field_reader
    def read(out, indices, beats, roles, origins, elapsed):
        positions.append((float(beats[0]), int(elapsed[0])))
        return reader(out, indices, beats, roles, origins, elapsed)
    output = replace(output, field_reader=read)
    sampler = Sampler(temperature=0)
    with patch.object(model, 'begin', side_effect=AssertionError('CONT wrote main prefix')), \
         patch.object(model, 'complete', side_effect=AssertionError('CONT routed again')), \
         patch.object(model.heads, 'time_logits', side_effect=logits):
        token = next(sampler.contents(model, output.hidden[0,-1], 'HOLD_DURATION', oracle,
            timing, None, end_tick=1536, cap_tick=1920, output=output))
    assert token == HoldDuration(769)
    assert positions == [(0., 0), (4., 384), (12., 768)]
    assert oracle.snapshot() == before
    for old, new in zip(cache, output.cache.layers):
        torch.testing.assert_close(old, new.key, rtol=0, atol=0)
    # Only the assembled complete field can materialize one real action.
    oracle.accept(token)
    assert len(oracle.actions) == 1
    assert oracle.actions[0].press_end > duration_end(timing, 0, 769)


def test_generated_moment_cont_preserves_query_lower_bound_and_real_cursor():
    model, routed, types = model_prefix(origin=123)
    output = model.complete(routed, types)
    timing = TimingMap([(0, 120)])
    oracle = InferencePhysicalState(timing, geometry(), audio_end_seconds=20.)
    oracle.inject_boundary(Moment(0), start_tick=123, window_end=500)
    before = oracle.snapshot()
    scores = torch.full((385,), -1000.)
    scores[CONTINUE] = 1000.
    with patch.object(model.heads, 'time_logits', return_value=scores):
        terminal = next(Sampler(temperature=0).contents(model, output.hidden[0,-1], 'MOMENT',
            oracle, timing, None, end_tick=500, cap_tick=884, output=output))
    assert isinstance(terminal, TimeTarget) and terminal.closed_by_continue
    assert terminal.lower_bound_tick == 508
    assert terminal.query_tick == 507
    assert oracle.snapshot() == before
    oracle.close_window(terminal)
    assert oracle.grammar.state.cursor == 123 and oracle.grammar.state.last_real_tick is None
    assert not oracle.actions


def test_window_close_leaves_zero_penalty_and_full_tail_has_gradient():
    timing = TimingMap([(0, 120)])
    # All real completions violate a cause; close leaves remain legal mass.
    stage = dict(cursor_tick=0, moment=dict(domain=[1, 5000], outside_basic='outside',
        intervals=[[1, 5000, dict(basic=[], physical=1, penalty=1)]]))
    tree = compile_support_tree(stage, timing, kind='MOMENT', window_end=1000,
                               cap_tick=1384, target=Moment(timing.delta_ms(0, 1)))
    allowed = torch.tensor(tree.basic)
    logits = torch.zeros_like(allowed, dtype=torch.float32, requires_grad=True)
    leaves = complete_field_log_probabilities(logits, allowed, torch.tensor(tree.continuation_closes))
    penalties = torch.tensor(tree.penalty)
    assert penalties[-1, :231, 0].all()
    assert not penalties[-1, 231:].any()
    torch.testing.assert_close(leaves.exp().sum(), torch.tensor(1.))
    result = physical_categories(leaves, penalties)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert logits.grad[-1].abs().sum() > 0  # Beyond GT's one depth.
    assert torch.isfinite(logits.grad).all()


def test_first_route_wifi_cap_is_reprojected_from_missing_counterfactual(monkeypatch):
    tokens = [Moment(0), Note.make('slide','A1'), Route(0,0), SlideDuration(24)]
    record, supports, geo = source_fields(monkeypatch, tokens, audio_end=12., bpm=60)
    # Raw accepted old support says the WiFi candidate has a legal long value.
    supports['stages'][1]['route'][70] = Evaluation().record()
    chart = ChartSupports(record, supports, geo)
    witness = Conflict('slide_too_fast', 'review-cap', (('track', 0, 0, 0),))
    bad = Evaluation(Evidence(conflicts=frozenset({witness})),
                     Evidence(conflicts=frozenset({witness}))).record()
    exact = dict(domain=[1, 96], outside_basic='duration_outside_audio',
                 intervals=[[1,95,bad], [96,96,Evaluation().record()]])
    queried = []
    def missing(state):
        queried.append(state.physical.grammar.state.routes)
        return exact
    monkeypatch.setattr(chart, '_duration_source', missing)
    window = build_window(chart.tokens, chart.timing, sequence_start_tick=0,
        start_tick=0, end_tick=96, audio_after_beats=1, audio_end_seconds=12.,
        history_beats=4, history_dropout=0, training=False, rng=random.Random(0))
    stages, _ = chart.window(window, duration_cap_seconds=chart.timing.second(191))
    first_route = stages[1]
    assert first_route['phase'] == 'route'
    assert first_route['route'][70]['physical'] == 4
    assert (70,) in queried
    assert first_route['route'][0] == supports['stages'][1]['route'][0]


def test_window_compiler_idempotence_preserves_exact_source_and_supported_domain(monkeypatch):
    tokens = [Moment(0), Note.make('hold','A1'), HoldDuration(385),
              Moment(5000), Note.make('tap','A2')]
    record, supports, geo = source_fields(monkeypatch, tokens, audio_end=20., bpm=60)
    chart = ChartSupports(record, supports, geo)
    row = dict(chart_id='0:5', window_id='review:0', start_tick=0, end_tick=192)
    cfg = dict(target_beats=2, audio_after_beats=4, chart_history_beats=4, decoder_history_dropout=.15)
    onsets = real_onsets(chart.tokens, chart.timing, sequence_start_tick=0)
    original = repr(record), repr(supports)
    first = compile_window(chart, row, cfg, 20., onsets)
    second = compile_window(chart, row, cfg, 20., onsets)
    # Fresh and migration entry points both import this compiler. Its output
    # depends on exact source/support, never old rounded tensor labels.
    left, right = tensor_pack(first), tensor_pack(second)
    assert left['layout'] == right['layout']
    for dtype in left['buffers']:
        torch.testing.assert_close(left['buffers'][dtype], right['buffers'][dtype], rtol=0, atol=0)
    assert (repr(record), repr(supports)) == original
    restored = tensor_unpack(left)
    source = next(v for v in restored['metadata']['exact_fields'] if v['target'].get('type') == 'HOLD_DURATION')
    assert source['source_token_index'] == 2 and source['target']['ticks'] == 385
    assert restored['metadata']['main_sequence_length'] == len(first['next_types'])
    groups = first['labels']['groups']
    actual_queries = sum(g['indices'].shape[0] for k, g in groups.items() if k in ('kind', 'route'))
    actual_queries += sum(int(g['allowed'].any(-1).sum()) for g in groups.values()
                          if 'target_steps' in g)
    assert restored['metadata']['field_query_cost'] == actual_queries


def test_draft_moment_queries_do_not_see_future_kind_count_or_position():
    timing = TimingMap([(0, 120)])
    def target(groups):
        chart = ChartData(tuple(OnsetGroup(t, tuple(content)) for t, content in groups),
                          (), timing, Fraction(32))
        return reconstruction_target(chart, start_tick=0, target_beats=16)
    a = target([(192, [Note.make('tap','A1')]), (768, [Note.make('touch','C')])])
    b = target([(192, [Note.make('tap','A8')]), (900, [Note.make('slide','A2'),
                Route(0,0), SlideDuration(48), Note.make('hold','A7'), HoldDuration(96)])])
    assert a.rows[0].query == b.rows[0].query
    assert a.rows[0].query.beat == 0
    assert a.rows[0].target_classes == b.rows[0].target_classes
    # Changing pad does not enter an R NOTE query.
    assert a.rows[1].query == b.rows[1].query


def test_draft_real_moment_and_exact_terminal_keep_source_indices():
    chart = from_record(draft_record())
    target = reconstruction_target(chart, start_tick=0, target_beats=3)
    source = [r.source_token_index for r in target.rows if r.query.role == 'MOMENT']
    assert source == [8, 10, None]  # Two real source fields and a censored control.
    one_beat = reconstruction_target(chart, start_tick=0, target_beats=1)
    assert one_beat.rows[0].query.role == 'NOTE'
    assert one_beat.rows[0].source_token_index == 1  # No zero-time MOMENT row.
    assert one_beat.rows[-1].source_token_index == 8  # Exact next-onset terminal.


def test_draft_exact_censored_and_continue_close_metadata_stays_out_of_query():
    timing = TimingMap([(0, 120)])
    empty = ChartData((), (), timing, Fraction(32))
    later = ChartData((OnsetGroup(500, (Note.make('tap','A1'),)),), (), timing, Fraction(32))
    def terminal(chart):
        return reconstruction_target(chart, start_tick=0, target_beats=2,
                                     audio_after_beats=4).rows[-1]
    exact, censor = terminal(later), terminal(empty)
    assert exact.query == censor.query
    assert exact.target_classes == censor.target_classes == (CONTINUE,)
    assert not exact.censored and censor.censored
    assert exact.closes_window and censor.closes_window
    assert exact.closed_by_continue and censor.closed_by_continue
    assert exact.closed_lower_bound_tick == censor.closed_lower_bound_tick == 385
    assert exact.times['moment_position_tick'] == 500
    assert censor.times['moment_position_tick'] == 576
