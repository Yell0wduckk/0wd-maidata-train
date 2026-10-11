"""Local reconstruction conditions, exact time paths and independent Planner."""
from dataclasses import replace
from fractions import Fraction
import math

import pytest
import torch

from maidata.chart.types import Note, Route, SlideDuration, HoldDuration
from maidata.draft.chart_data import ChartData, OnsetGroup
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.draft.types import BeatMemory, WindowLatents
from maidata.draft.losses import reconstruction_loss
from maidata.model import build_reconstruction, build_planner
from maidata.entry import CONFIG
from maidata.time import TimingMap


def chart(groups=()):
    return ChartData(tuple(OnsetGroup(t, tuple(tokens)) for t, tokens in groups), (),
                     TimingMap([(0, 120), (4, 240)]), Fraction(24))


def labels(c, start=0, target=16):
    return collate_reconstruction([reconstruction_target(c, start_tick=start*96,
        target_beats=target)], target_beats=target, device='cpu')


def memory():
    return WindowLatents(BeatMemory(torch.randn(1, 24, 128, requires_grad=True),
        torch.arange(24)[None].float()-3.5, torch.ones(1, 24, dtype=torch.bool)),
        torch.tensor([0.]), 16, torch.tensor([True]))


def model():
    return build_reconstruction(dict(CONFIG, reconstruction_dropout=0.)).eval()


def test_boundary_note_zero_count_and_no_pad_type_task():
    c = chart([(0, [Note.make('tap', 'A1'), Note.make('tap', 'A2')]),
               (385, [Note.make('hold', 'A3'), HoldDuration(385)])])
    t = labels(c)
    assert t.queries.roles[0, :2].tolist() == [1, 1]
    assert t.queries.ordinals[0, :2].tolist() == [0, 1]
    assert 'type' not in t.fields and all('pad' not in key for key in t.fields)
    assert t.moment_counts[0].tolist() == [1, 0, 0, 0, 1]+[0]*11
    assert t.time_valid['hold_duration_tick'].sum() == 1
    assert t.time_classes['hold_duration_tick'].any(-1).sum() == 2
    assert t.time_classes['hold_duration_tick'][0].nonzero()[:, 1].tolist() == [384, 0]


def test_explicit_boundary_and_source_moment_are_not_inferred_from_zero():
    from test_draft_data import record
    from maidata.draft.chart_data import from_record
    c = from_record(record())
    target = reconstruction_target(c, start_tick=0, target_beats=3)
    moments = [r for r in target.rows if r.query.role == 'MOMENT']
    assert [r.source_token_index for r in moments[:2]] == [8, 10]
    assert not moments[0].query.boundary  # A real NOTE at zero already exists.
    empty = labels(chart())
    assert empty.queries.boundary[0, 0] and not empty.queries.last_real_valid[0, 0]
    c = chart([(-96, [Note.make('tap', 'A1')]), (48, [Note.make('tap', 'A2')])])
    t = labels(c)
    assert t.queries.boundary[0, 0] and t.queries.last_real_valid[0, 0]
    assert t.queries.history_gaps[0, 0] == 1


def test_neighborhoods_and_future_query_isolation():
    c = chart([(192, [Note.make('tap', 'A1')]), (768, [Note.make('tap', 'A2')])])
    t, w, r = labels(c), memory(), model()
    out = r(w, t.queries)
    # First MOMENT at known boundary reads beat slots 0..4 only.
    loss = out.time_predictions['moment_position_tick'][0, 0, 7]
    grad = torch.autograd.grad(loss, w.memory.values, retain_graph=True)[0]
    assert grad[0, 4:9].abs().sum() > 0
    assert grad[0, :4].count_nonzero() == grad[0, 9:].count_nonzero() == 0
    # NOTE at beat 2 reads beat slots 1,2,3 only.
    note = (t.queries.roles[0] == 1).nonzero()[0, 0]
    grad = torch.autograd.grad(out.field_logits['kind'][0, note, 0], w.memory.values)[0]
    assert grad[0, 5:8].abs().sum() > 0
    assert grad[0, :5].count_nonzero() == grad[0, 8:].count_nonzero() == 0
    changed = labels(chart([(192, [Note.make('tap', 'A1')]),
                            (800, [Note.make('touch', 'C')]*3)]))
    other = r(w, changed.queries)
    torch.testing.assert_close(out.time_predictions['moment_position_tick'][:, :2],
                               other.time_predictions['moment_position_tick'][:, :2])
    torch.testing.assert_close(out.field_logits['kind'][:, :2], other.field_logits['kind'][:, :2])


def test_pad_answers_absent_and_slide_full_chain_is_known_condition():
    base = [(0, [Note.make('slide', 'A1'), Route(0, 0), Route(1), SlideDuration(769)])]
    a = labels(chart(base))
    b = labels(chart([(0, [Note.make('slide', 'A8'), Route(0, 0), Route(1), SlideDuration(769)])]))
    for name in a.queries.__dataclass_fields__:
        torch.testing.assert_close(getattr(a.queries, name), getattr(b.queries, name))
    slide = (a.queries.roles[0] == 4).nonzero().flatten()
    assert a.queries.route_ids[0, slide[0]].tolist() == [0, 1]
    assert a.queries.beats[0, slide[0]] == 1  # Slide waits one global beat.
    assert a.queries.beats[0, slide[1]] == 6  # Written-BPM 4 beats crosses BPM.
    assert a.queries.route_ids[0, slide].eq(torch.tensor([0, 1])).all()


def test_path_loss_counts_complete_fields_and_empty_tail_is_valid():
    t = labels(chart([(0, [Note.make('hold', 'A1'), HoldDuration(385)])]))
    w, r = memory(), model()
    out = r(w, t.queries)
    terms = reconstruction_loss(out, t, label_smoothing=.05)
    assert terms['time'].count == 2  # One duration and one terminal, not CONT substeps.
    assert torch.isfinite(sum(term.mean for term in terms.values()))
    sum(term.mean for term in terms.values()).backward()
    assert w.memory.values.grad.abs().sum() > 0
    empty = labels(chart())
    assert empty.moment_counts.eq(0).all() and empty.valid.all()
    assert empty.time_valid['moment_position_tick'].sum() == 1
    eo = r(memory(), empty.queries)
    et = reconstruction_loss(eo, empty, label_smoothing=.05)
    assert et['classification'].count == 0 and et['count'].count == 16


def test_planner_reads_all_target_beats_independently():
    w = memory()
    p = build_planner(dict(CONFIG, planner_dropout=0.)).eval()
    grad = torch.autograd.grad((p(w).plan_values*torch.arange(12)).sum(), w.memory.values)[0]
    assert grad[0, 4:20].abs().sum(-1).gt(0).all()
    assert grad[0, :4].count_nonzero() == grad[0, 20:].count_nonzero() == 0


def test_long_duration_retains_gt_and_supervises_crossing_continue():
    target = reconstruction_target(chart([(0, [Note.make('hold', 'A1'), HoldDuration(5000)])]),
                                   start_tick=0, target_beats=16)
    duration = [r for r in target.rows if r.query.role == 'HOLD_DURATION']
    assert all(r.times['hold_duration_tick'] == 5000 for r in duration)
    assert all(r.target_classes == (384,) for r in duration)
    assert duration[-1].closed_by_continue
    assert duration[-1].query.beat < 20
    labels = collate_reconstruction([target], target_beats=16, device='cpu')
    assert labels.time_valid['hold_duration_tick'].sum() == 1
    output = model()(memory(), labels.queries)
    terms = reconstruction_loss(output, labels, label_smoothing=.05)
    sum(term.mean for term in terms.values()).backward()
    assert all(torch.isfinite(term.mean) for term in terms.values())


def test_padding_does_not_condition_other_queries_or_revive_biases():
    c = chart([(0, [Note.make('tap', 'A1')])])
    a = labels(c)
    b = collate_reconstruction([reconstruction_target(c, start_tick=0, target_beats=16), None],
                               target_beats=16, device='cpu')
    w, r = memory(), model()
    two = replace(w, memory=BeatMemory(w.memory.values.expand(2, -1, -1),
        w.memory.beats.expand(2, -1), w.memory.valid.expand(2, -1)),
        target_start=torch.zeros(2), window_valid=torch.tensor([True, False]))
    x, y = r(w, a.queries), r(two, b.queries)
    torch.testing.assert_close(x.field_logits['kind'][0], y.field_logits['kind'][0])
    assert y.count_logits[1].count_nonzero() == 0
    assert all(v[1].count_nonzero() == 0 for v in y.field_logits.values())
