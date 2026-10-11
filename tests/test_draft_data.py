"""New-field adapters, exact-time simai statistics and empty-window objectives."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import math
from types import SimpleNamespace

import pytest
import torch

from maidata.chart.codec import token_to_dict
from maidata.chart.types import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.time import TimingMap
from maidata.draft.chart_data import from_record, ChartData, OnsetGroup, PlanNote
from maidata.draft.conditions import read_condition, ChartCondition, make_draft_inputs
from maidata.draft.windows import song_coverage
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import (reconstruction_target, collate_reconstruction, collate_planner,
                                  required_query_rows)
from maidata.draft.losses import reconstruction_loss, planner_loss
from maidata.draft.types import ReconstructionOutput, PlannerOutput
from maidata.draft.schema import FIELD_SIZES


def record():
    tokens = (Moment(0), Note.make('slide', 'A1'), Route(0, 0), Route(1), SlideDuration(192),
              Route(70, 1), SlideDuration(96), Note.make('tap', 'A1', short_hold=1),
              Moment(500), Note.make('touch', 'C', short_touch_hold=1),
              Moment(500), Note.make('hold', 'A5'), HoldDuration(200))
    ticks = [0] * 8 + [96] * 2 + [192] * 3
    owners = [None, 0, 0, 0, 0, 0, 0, 1, None, 2, None, 3, 3]
    return dict(schema='maidata.full-chart-original.v2', view='original', start_tick=0,
        metadata=dict(first_seconds=.125, render_end_beat=dict(num=8, den=1),
            controls=[dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1)),
                      dict(kind='bpm', bpm=240, position_beats=dict(num=2, den=1))]),
        tokens=[token_to_dict(t) for t in tokens],
        token_map=[dict(token_index=i, tick=t, owner=o) for i, (t, o) in enumerate(zip(ticks, owners))])


def test_new_field_rows_boundary_and_chain_ownership():
    chart = from_record(record())
    target = reconstruction_target(chart, start_tick=0, target_beats=3)
    assert len(target.rows) == 16
    assert [r.query.role for r in target.rows] == [
        'MOMENT', 'NOTE_KIND', 'NOTE_FLAGS', 'ROUTE_ID', 'TRACK_BREAK', 'ROUTE_ID',
        'ROUTE_ID', 'TRACK_BREAK', 'NOTE_KIND', 'NOTE_FLAGS', 'MOMENT',
        'NOTE_KIND', 'NOTE_FLAGS', 'MOMENT', 'NOTE_KIND', 'NOTE_FLAGS']
    assert target.rows[0].query.beat == .5 and target.rows[0].moment_tick == 0
    assert target.rows[4].fields['track_break'] == 0
    assert 'track_break' not in target.rows[5].fields
    assert target.rows[7].fields['track_break'] == 1
    assert all(row.query.role not in ('HOLD_DURATION', 'SLIDE_DURATION') for row in target.rows)
    assert target.moment_counts == (1, 1, 1)  # Two NOTE owners at beat zero are one MOMENT.
    assert required_query_rows([len(target.rows), 0]) == 16
    batch = collate_reconstruction([target], query_rows=17, target_beats=3, device='cpu')
    assert batch.fields['track_break'][0, 4] == 0
    assert batch.fields['track_break'][0, 5] == -100
    assert batch.queries.valid.shape == (1, 16)
    assert 'type' not in batch.fields and all('pad' not in n for n in batch.fields)
    # Legacy capacity hints neither pad query inputs nor truncate real fields.
    narrow = collate_reconstruction([target], query_rows=1, target_beats=3, device='cpu')
    assert narrow.queries.valid.shape == batch.queries.valid.shape
    assert target.rows[6].query.route_prefix == () and target.rows[6].query.track == 1
    damaged = record()
    damaged['token_map'][3]['owner'] = 1
    with pytest.raises(ValueError, match='owner'):
        from_record(damaged)


def test_empty_window_has_no_moment_queries_and_supervises_empty_beats():
    chart = from_record(record())
    empty = reconstruction_target(chart, start_tick=10 * 96, target_beats=3)
    labels = collate_reconstruction([empty, None], query_rows=7, target_beats=3, device='cpu')
    assert len(empty.rows) == 0 and empty.moment_counts == (0, 0, 0)
    assert labels.valid.tolist() == [True, False]
    assert not labels.queries.valid.any()
    assert labels.moment_tick.eq(-100).all()
    assert all(values.eq(-100).all() for values in labels.fields.values())
    assert not labels.queries.valid[1].any()
    plan = MusicPlanTargets(chart).window(10 * 96, 13 * 96)
    assert plan.continuous == (0.,) * 8 and plan.category == 0
    assert collate_planner([plan, None], device='cpu').valid.tolist() == [True, False]


def test_label_export_uses_new_representation_and_all_empty_export_is_valid(tmp_path, monkeypatch):
    import json
    from maidata.chart_cache import cache
    from maidata.draft.export_targets import main
    monkeypatch.setattr(cache, 'load_chart', lambda _path: record())
    output = tmp_path / 'labels.json'
    assert main(['--chart', str(tmp_path/'selected.json.gz'), '--start-beat', '0',
                 '--start-beat', '10', '--target-beats', '3', '--output', str(output)]) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['schema'] == 'maidata.draft-local-query-targets.v3'
    assert result['representation'] == 'beat-local-tick96'
    assert [row['moment_queries'] for row in result['windows']] == [3, 0]
    assert result['windows'][1]['reconstruction']['rows'] == []
    assert all(row['query']['role'] not in ('HOLD_DURATION', 'SLIDE_DURATION')
               for window in result['windows'] for row in window['reconstruction']['rows'])
    assert main(['--chart', str(tmp_path/'selected.json.gz'), '--start-beat', '10',
                 '--target-beats', '3', '--output', str(output)]) == 0
    assert json.loads(output.read_text(encoding='utf-8'))['query_rows_required'] == 0


def test_simai_statistics_exact_bpm_short_holds_and_category_ties():
    chart = from_record(record())
    p = MusicPlanTargets(chart)
    target = p.window(0, 4 * 96)
    # Window seconds = two beats at 120 + two at 240 = 1.5.
    expected = (math.log1p(4) / math.log1p(64), (4 / 1.5) / 16,
                1 / 3, 1 / 4, .5, 1 / 4, 3 / 4)
    assert target.continuous[:7] == pytest.approx(expected)
    # Slide at 0, Touch at 1, single at 2: one group each, tie chooses Touch.
    assert target.category == 3
    full = p.window(0, 8 * 96)
    assert full.continuous[7] == pytest.approx(.5)
    assert p.window(96, 192).continuous[0] == pytest.approx(math.log1p(1) / math.log1p(64))
    assert p.window(192, 288).category == 0


def test_simai_statistics_match_original_algorithm_on_mixed_sequence():
    # Read-only numerical oracle from the explicitly selected simai source.
    # Execute only its pure _statistics function: no runtime dependency/import.
    import ast
    from collections import Counter
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / 'repos-baiqieji/simai_transformer/src/maichart/training/music_plan_targets.py'
    if not source.exists():
        pytest.skip('read-only simai oracle is not available in this checkout')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_statistics')
    namespace = dict(math=math, Counter=Counter, Fraction=Fraction)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    chart = from_record(record())
    parsed = [SimpleNamespace(beat=Fraction(n.tick, 96), note_type=n.kind,
                               position=n.button if n.button is not None else 'C') for n in chart.notes]
    expected, workload = namespace['_statistics'](parsed, 1.5, 4., Fraction(0))
    from maidata.draft.plan_targets import statistics
    actual, actual_workload = statistics(chart.notes, 1.5, 384, 0)
    assert actual == pytest.approx(expected)
    assert actual_workload == pytest.approx(workload)


def test_metadata_level_and_audio_span_are_separate_from_chart_end(tmp_path):
    path = tmp_path / 'maidata.txt'
    path.write_text('&lv_5=14.600\n&lv_6=14+\n&inote_5=ignored\n', encoding='utf-8')
    source = dict(difficulty='5', chart_id='test:5', source_path=str(path), audio_variant='v')
    c = read_condition(source)
    assert c.level == 14.6 and c.level_text == '14.600' and c.difficulty_id == 5
    with pytest.raises(ValueError, match='exact numeric'):
        read_condition(dict(source, difficulty='6'))
    with pytest.raises(ValueError, match='exact numeric'):
        read_condition(dict(source, difficulty='7'))
    assert read_condition(dict(source, difficulty='6'), numeric_level=14.8).level == 14.8
    bound = dict(source, source_path='unavailable/maidata.txt', level_text='14.600')
    assert read_condition(bound, numeric_level=14.6).level_text == '14.600'
    timing = TimingMap([(0, 120), (2, 240)], first_seconds=.25)
    coverage = song_coverage(timing, 2., first_beat=-1, target_beats=3)
    assert coverage.span_beats == pytest.approx(5.5)
    assert coverage.starts(include_tail=True) == (-1, 2, 5)
    inputs, mapping = make_draft_inputs([c, c, replace(c, level=13.2)], ['v'], [coverage], device='cpu')
    assert mapping.tolist() == [0, 0, 1]
    assert inputs.audio_mapping.tolist() == [0, 0]
    assert inputs.span_beats.tolist() == [5.5, 5.5]
    assert inputs.query_beats.shape == (2, 9)
    assert inputs.query_beats[0, 0] == -.5 and inputs.draft_valid.all()
    missing = record()
    del missing['metadata']['render_end_beat']
    parsed = from_record(missing)
    assert len(reconstruction_target(parsed, start_tick=0, target_beats=3).rows) == 16
    with pytest.raises(ValueError, match='source final'):
        MusicPlanTargets(parsed)
    assert from_record(missing, source_end_beat='17/2').source_end_beat == Fraction(17, 2)


def test_planner_recovers_missing_end_from_selected_source_only(tmp_path):
    from unittest.mock import patch
    from pathlib import Path

    path = tmp_path / 'maidata.txt'
    path.write_text('&bpm=120\n&inote_4={1}1,,,E\n'
                    '&inote_5=(120){4}1,2,,(240){8},\n&lv_5=14.6\n', encoding='utf-8-sig')
    missing = record()
    del missing['metadata']['render_end_beat']
    missing['source'] = dict(source_path=str(path), difficulty='5')
    with patch.object(Path, 'read_text', side_effect=AssertionError('R must not read the source')):
        chart = from_record(missing)
        assert len(reconstruction_target(chart, start_tick=0, target_beats=3).rows) == 16
    # Three full cells and two half cells, including the trailing empty cell.
    planner = MusicPlanTargets(chart)
    assert planner.source_end_beat == 4
    explicit = MusicPlanTargets(from_record(missing, source_end_beat=4))
    assert planner.reference == pytest.approx(explicit.reference)
    assert planner.window(0, 288) == explicit.window(0, 288)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        MusicPlanTargets(from_record(missing))
    # Metadata or an explicit override takes precedence; neither reads the source.
    assert MusicPlanTargets(from_record(dict(missing, metadata=record()['metadata']))).source_end_beat == 8
    assert MusicPlanTargets(from_record(missing, source_end_beat='17/2')).source_end_beat == Fraction(17, 2)


@pytest.mark.parametrize('body,end', [
    ('(120){4}1,2,,,', Fraction(5)),
    ('{4}1,2,,,E,1,1', Fraction(4)),
    ('{3}1,,{7}E1,', Fraction(80, 21)),
    ('||3/4 (120){4}1h[4:1],{8}E1', Fraction(3, 2)),
    ('(120){#0.25}1,,', Fraction(3, 2)),
    ('', Fraction(0)),
])
def test_source_end_uses_timing_cells_not_note_durations(body, end):
    from maidata.draft.source_timing import inote_end_beat
    assert inote_end_beat(body, initial_bpm=120) == end


def test_source_end_matches_read_only_simai_timing_oracle():
    import ast
    import re
    from pathlib import Path
    from maidata.draft.source_timing import inote_end_beat
    root = Path(__file__).resolve().parents[3] / 'repos-baiqieji/simai_transformer/src/maichart'
    if not root.is_dir():
        pytest.skip('read-only simai oracle is not available in this checkout')
    names = {'_split_top_level_commas', '_parse_point_directives', '_skip_whitespace', 'tokenize_event_timing'}
    functions = []
    for filename in ('timing.py', 'event_timing.py'):
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        functions.extend(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    namespace = dict(Fraction=Fraction, re=re, TimingDirective=lambda kind, raw, value:
                     SimpleNamespace(directive_type=kind, raw=raw, value=value),
                     EventTimingPoint=SimpleNamespace, normalize_note_text=lambda x: x,
                     _finite_number=lambda value, *args, **kwargs: float(value),
                     duration_to_ticks_or_none=lambda _: None)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root), 'exec'), namespace)
    for body in ('(120){4}1,2,,,', '{4}1,2,,,E,1', '{3}1,,{7}E1,',
                 '||3/4 (120){4}1h[4:1],{8}E1'):
        last = namespace['tokenize_event_timing'](body)[-1]
        expected = last.beat + (0 if any(d.directive_type == 'end' for d in last.directives)
                                else Fraction(4, last.division))
        assert inote_end_beat(body, initial_bpm=120) == expected


def test_losses_support_na_empty_padding_and_independent_reduction():
    chart = from_record(record())
    real = reconstruction_target(chart, start_tick=0, target_beats=3)
    empty = reconstruction_target(chart, start_tick=9 * 96, target_beats=3)
    target = collate_reconstruction([real, empty, None], query_rows=17, target_beats=3, device='cpu')
    width = target.queries.valid.shape[1]
    logits = {name: torch.zeros(3, width, size, requires_grad=True) for name, size in FIELD_SIZES.items()}
    moments = torch.zeros(3, width, 96, requires_grad=True)
    empty_logits = torch.zeros(3, 3, 2, requires_grad=True)
    positive_logits = torch.zeros(3, 3, 96, requires_grad=True)
    output = ReconstructionOutput(logits, moments, empty_logits, positive_logits,
        target.valid, target.queries.valid, target.valid[:, None].expand(-1, 3))
    terms = reconstruction_loss(output, target, label_smoothing=.05)
    assert sum(term.count for name, term in terms.items() if name.startswith('field/')) == sum((v != -100).sum() for v in target.fields.values())
    assert terms['moment96'].count == 3 and terms['empty'].count == 6
    assert terms['positive_count'].count == 3
    assert terms['empty'].mean.item() == pytest.approx(math.log(2))
    assert terms['positive_count'].mean.item() == pytest.approx(math.log(96))
    sum(term.mean for term in terms.values()).backward()
    assert empty_logits.grad[1].abs().sum() > 0  # Extra empty target is supervised.
    assert empty_logits.grad[2].count_nonzero() == 0
    assert positive_logits.grad[1:].count_nonzero() == 0
    assert all(value.grad[1].count_nonzero() == 0 for value in logits.values())
    assert logits['track_break'].grad[0, 5].count_nonzero() == 0  # Continuation NA.
    assert logits['track_break'].grad[0, 4].abs().sum() > 0  # Applicable zero.
    assert moments.grad[0, 10, 0] < 0
    assert moments.grad[1:].count_nonzero() == 0

    targets = collate_planner([MusicPlanTargets(chart).window(0, 288),
                               MusicPlanTargets(chart).window(864, 1152), None], device='cpu')
    continuous = torch.zeros(3, 8, requires_grad=True)
    category = torch.zeros(3, 4, requires_grad=True)
    plan = torch.cat((continuous.sigmoid(), category.softmax(-1)), -1)
    outputs = PlannerOutput(continuous, category, plan, targets.valid)
    terms = planner_loss(outputs, targets)
    expected = ((.5 - targets.continuous[:2]) ** 2).mean().item()
    assert terms['planner_continuous'].mean.item() == pytest.approx(expected)
    assert terms['planner_category'].mean.item() == pytest.approx(math.log(4))
    sum(term.mean for term in terms.values()).backward()
    assert continuous.grad[1].abs().sum() > 0 and continuous.grad[2].count_nonzero() == 0
    assert category.grad[1].abs().sum() > 0


def test_all_padding_losses_finite_zero_with_gradients():
    r = collate_reconstruction([None], query_rows=2, target_beats=3, device='cpu')
    output = ReconstructionOutput(
        {name: torch.zeros(1, 1, size, requires_grad=True) for name, size in FIELD_SIZES.items()},
        torch.zeros(1, 1, 96, requires_grad=True), torch.zeros(1, 3, 2, requires_grad=True),
        torch.zeros(1, 3, 96, requires_grad=True), r.valid, r.queries.valid,
        r.valid[:, None].expand(-1, 3))
    terms = reconstruction_loss(output, r, label_smoothing=.05)
    p = collate_planner([None], device='cpu')
    prediction = PlannerOutput(torch.zeros(1, 8, requires_grad=True), torch.zeros(1, 4, requires_grad=True),
                                torch.zeros(1, 12), p.valid)
    terms.update(planner_loss(prediction, p))
    loss = sum(term.mean for term in terms.values())
    assert loss.item() == 0 and all(term.count == 0 for term in terms.values())
    loss.backward()
