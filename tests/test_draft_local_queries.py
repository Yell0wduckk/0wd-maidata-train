"""Beat-local labels, role isolation, direct gradients and empty supervision."""
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from maidata.chart.types import Note, Route, SlideDuration, HoldDuration
from maidata.draft.chart_data import ChartData, OnsetGroup
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.draft.types import BeatMemory, WindowLatents
from maidata.draft.losses import reconstruction_loss
from maidata.draft.schema import QUERY_ROLES
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
    return WindowLatents(BeatMemory(torch.randn(1, 24, 256, requires_grad=True),
        torch.arange(24)[None].float()-3.5, torch.ones(1, 24, dtype=torch.bool)),
        torch.tensor([0.]), 16, torch.tensor([True]))


def model():
    return build_reconstruction(dict(CONFIG, reconstruction_count_loss='ce', reconstruction_dropout=0.)).eval()


def indices(target, role):
    return (target.queries.roles[0].eq(QUERY_ROLES.index(role)) & target.queries.valid[0]).nonzero().flatten()


def test_tick_zero_95_96_negative_beats_and_same_time_notes():
    c = chart([(-1, [Note.make('tap', 'A8')]),
               (0, [Note.make('tap', 'A1'), Note.make('tap', 'A2')]),
               (95, [Note.make('hold', 'A3'), HoldDuration(5000)]),
               (96, [Note.make('touch', 'C')]), (1536, [Note.make('tap', 'A4')])])
    t = labels(c)
    moments = indices(t, 'MOMENT')
    assert t.moment_tick[0, moments].tolist() == [0, 95, 0]
    assert t.queries.beats[0, moments].tolist() == [.5, .5, 1.5]
    assert t.queries.ordinals[0, moments].tolist() == [0, 1, 0]
    assert t.moment_counts[0].tolist() == [2, 1]+[0]*14
    assert t.queries.note_ordinals[0, indices(t, 'NOTE_KIND')].tolist() == [0, 1, 0, 0]
    assert t.queries.last_real_valid[0, moments].tolist() == [True]*3
    assert t.queries.history_gaps[0, moments].tolist() == pytest.approx([-1/96, 0, -1/96])
    negative = labels(c, start=-1, target=1)
    assert negative.moment_tick[0, indices(negative, 'MOMENT')].tolist() == [95]
    assert not negative.queries.last_real_valid.any()
    assert 'type' not in t.fields and all('pad' not in key for key in t.fields)
    assert not hasattr(t, 'time_classes')


def test_cross_window_history_and_variable_bpm_do_not_change_tick_labels():
    c = chart([(1500, [Note.make('tap', 'A1')]), (1536, [Note.make('tap', 'A2')]),
               (1631, [Note.make('tap', 'A3')]), (1632, [Note.make('tap', 'A4')])])
    t = labels(c, start=16)
    moment = indices(t, 'MOMENT')
    assert t.moment_tick[0, moment].tolist() == [0, 95, 0]
    assert t.queries.history_gaps[0, moment[0]] == pytest.approx(-36/96)
    assert t.queries.last_real_valid[0, moment].all()
    altered = replace(c, timing=TimingMap([(0, 80), (16, 360)]))
    other = labels(altered, start=16)
    torch.testing.assert_close(t.moment_tick, other.moment_tick)


def test_direct_owning_beat_gradients_and_no_attention():
    t, w, r = labels(chart([(192, [Note.make('tap', 'A1')])])), memory(), model()
    out = r(w, t.queries)
    for row, tensor in ((indices(t, 'MOMENT')[0], out.moment_logits),
                        (indices(t, 'NOTE_KIND')[0], out.field_logits['kind'])):
        grad = torch.autograd.grad(tensor[0, row, 0], w.memory.values, retain_graph=True)[0]
        assert grad[0, 6].abs().sum() > 0
        assert grad[0, :6].count_nonzero() == grad[0, 7:].count_nonzero() == 0
    grad = torch.autograd.grad(out.empty_logits[0, 2, 0], w.memory.values)[0]
    assert grad[0, 6].abs().sum() > 0
    assert grad[0, :6].count_nonzero() == grad[0, 7:].count_nonzero() == 0
    assert not any('attention' in name for name, _ in r.named_modules())


def test_current_moment_target_future_count_pad_and_duration_are_isolated():
    a = labels(chart([(1, [Note.make('hold', 'A1'), HoldDuration(2)])]))
    b = labels(chart([(95, [Note.make('hold', 'A8'), HoldDuration(9999)]),
                      (800, [Note.make('touch', 'C')]*3)]))
    w, r = memory(), model()
    x, y = r(w, a.queries), r(w, b.queries)
    moment_a, moment_b = indices(a, 'MOMENT')[0], indices(b, 'MOMENT')[0]
    torch.testing.assert_close(x.moment_logits[0, moment_a], y.moment_logits[0, moment_b])
    torch.testing.assert_close(x.empty_logits, y.empty_logits)
    torch.testing.assert_close(x.positive_count_logits, y.positive_count_logits)
    c = labels(chart([(1, [Note.make('hold', 'A8'), HoldDuration(9999)])]))
    for name in a.queries.__dataclass_fields__:
        torch.testing.assert_close(getattr(a.queries, name), getattr(c.queries, name))
    changed_labels = replace(a, fields={name: torch.zeros_like(value) for name, value in a.fields.items()},
                             moment_tick=torch.zeros_like(a.moment_tick), moment_counts=torch.full_like(a.moment_counts, 96))
    torch.testing.assert_close(x.moment_logits, r(w, changed_labels.queries).moment_logits)


def test_kind_and_current_route_only_condition_their_downstream_roles():
    c = chart([(48, [Note.make('slide', 'A1'), Route(0, 1), Route(1), SlideDuration(192)])])
    a = labels(c)
    changed_kind = labels(chart([(48, [Note.make('touch', 'C')])]))
    w, r = memory(), model()
    x, y = r(w, a.queries), r(w, changed_kind.queries)
    ki, kj = indices(a, 'NOTE_KIND')[0], indices(changed_kind, 'NOTE_KIND')[0]
    torch.testing.assert_close(x.field_logits['kind'][0, ki], y.field_logits['kind'][0, kj])
    # Caller-supplied forbidden conditions cannot leak through the shared network.
    dirty = replace(a.queries, note_kinds=torch.full_like(a.queries.note_kinds, 2),
                    current_route_ids=torch.full_like(a.queries.current_route_ids, 70))
    z = r(w, dirty)
    for role in ('MOMENT', 'NOTE_KIND'):
        selected = indices(a, role)
        torch.testing.assert_close(x.moment_logits[0, selected], z.moment_logits[0, selected])
    assert not torch.allclose(x.field_logits['ex'][0, indices(a, 'NOTE_FLAGS')],
                             z.field_logits['ex'][0, indices(a, 'NOTE_FLAGS')])
    same_kind = replace(a.queries, current_route_ids=torch.full_like(a.queries.current_route_ids, 70))
    z = r(w, same_kind)
    ri, ti = indices(a, 'ROUTE_ID')[0], indices(a, 'TRACK_BREAK')[0]
    torch.testing.assert_close(x.field_logits['route_id'][0, ri], z.field_logits['route_id'][0, ri])
    assert not torch.allclose(x.field_logits['track_break'][0, ti], z.field_logits['track_break'][0, ti])


def test_duration_separator_preserves_parallel_tracks_and_ordered_prefixes():
    c = chart([(0, [Note.make('slide', 'A1'), Route(0, 0), Route(1), Route(3), SlideDuration(192),
                    Route(70, 1), Route(2), SlideDuration(96)])])
    target = reconstruction_target(c, start_tick=0, target_beats=16)
    route = [row for row in target.rows if row.query.role == 'ROUTE_ID']
    assert [row.query.route_prefix for row in route] == [(), (0,), (0, 1), (), (70,)]
    assert [row.query.track for row in route] == [0, 0, 0, 1, 1]
    assert [row.query.segment for row in route] == [0, 1, 2, 0, 1]
    breaks = [row for row in target.rows if row.query.role == 'TRACK_BREAK']
    assert [row.query.current_route_id for row in breaks] == [0, 70]
    assert [row.fields['track_break'] for row in breaks] == [0, 1]
    assert all(row.query.role not in ('HOLD_DURATION', 'SLIDE_DURATION') for row in target.rows)
    t, w, r = labels(c), memory(), model()
    out = r(w, t.queries)
    shuffled = replace(t.queries, route_ids=t.queries.route_ids.flip(-1), route_valid=t.queries.route_valid.flip(-1))
    # Prefix ordering remains an explicit, unbounded positional encoding.
    assert not torch.allclose(out.field_logits['route_id'], r(w, shuffled).field_logits['route_id'])


def test_empty_tail_has_no_field_queries_and_two_count_heads():
    t = labels(chart())
    assert t.moment_counts.eq(0).all() and t.valid.all()
    assert not t.queries.valid.any() and t.moment_tick.eq(-100).all()
    out = model()(memory(), t.queries)
    terms = reconstruction_loss(out, t, label_smoothing=.05)
    assert terms['moment96'].count == terms['positive_count'].count == 0
    assert terms['empty'].count == 16
    assert all(term.count == 0 for name, term in terms.items() if name.startswith('field/'))
    sum(term.mean for term in terms.values()).backward()
    assert out.empty_logits.shape == (1, 16, 2) and out.positive_count_logits.shape == (1, 16, 96)


def test_planner_reads_all_target_beats_independently():
    w = memory()
    p = build_planner(dict(CONFIG, reconstruction_count_loss='ce', planner_dropout=0.)).eval()
    grad = torch.autograd.grad((p(w).plan_values*torch.arange(12)).sum(), w.memory.values)[0]
    assert grad[0, 4:20].abs().sum(-1).gt(0).all()
    assert grad[0, :4].count_nonzero() == grad[0, 20:].count_nonzero() == 0


def test_padding_does_not_condition_queries_or_revive_biases():
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
    assert y.empty_logits[1].count_nonzero() == y.positive_count_logits[1].count_nonzero() == 0
    assert y.moment_logits[1].count_nonzero() == 0
    assert all(v[1].count_nonzero() == 0 for v in y.field_logits.values())
    poison = {}
    for name in b.queries.__dataclass_fields__:
        original = getattr(b.queries, name)
        if original is None:
            poison[name] = None
            continue
        value = original.clone()
        if name != 'valid':
            value[1] = float('nan') if value.is_floating_point() else 999 if value.dtype != torch.bool else True
        poison[name] = value
    dirty = r(two, replace(b.queries, **poison))
    torch.testing.assert_close(dirty.field_logits['kind'][0], y.field_logits['kind'][0])
    assert dirty.moment_logits[1].count_nonzero() == 0
    assert torch.isfinite(dirty.moment_logits).all()
