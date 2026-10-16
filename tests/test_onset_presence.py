"""Exact owner/beat targets, independent empty-beat supervision and logical loss."""
from collections import Counter
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from maidata.draft.losses import (COUNT_TERMS, prepare_supervision, reconstruction_loss,
                                 onset_presence_supervision_enabled)
from maidata.draft.targets import overlap_reconstruction_target, collate_reconstruction, collate_planner
from maidata.draft.types import BeatMemory, OnsetPresenceQueries
from maidata.draft.windows import select_windows
from maidata.training.runner import Diagnostics, supports, query_cost
from maidata.training.transport import pack_batch
from maidata.training.pretrain import weighted_loss
from maidata.joint_training.update import AuxiliaryMetrics, JointMetrics
from test_integer_draft_overlap import chart, head
from maidata.entry import CONFIG


def target(start, *, empty=False, first=0, end=32, stride=2, presence=True):
    source = replace(chart(), groups=(), notes=()) if empty else chart()
    return overlap_reconstruction_target(source, start_tick=start*96, target_beats=16,
        coverage_end_beat=end, first_beat=first, z_stride_beats=stride,
        onset_presence_supervision=presence)


def setup(*, empty=False, stride=2):
    targets = collate_reconstruction([target(0, empty=empty, stride=stride),
        target(16, empty=empty, stride=stride)], target_beats=16, device='cpu')
    planner = collate_planner([None, None], device='cpu')
    prepared = prepare_supervision(targets, planner)
    memory = BeatMemory(torch.randn(1, 32//stride+1, 12, requires_grad=True),
        torch.arange(0, 33, stride).float()[None], torch.ones(1, 32//stride+1, dtype=torch.bool))
    windows = select_windows(memory, torch.tensor([0, 0]), torch.tensor([0., 16.]),
        torch.ones(2, dtype=torch.bool), target_beats=16, before_beats=0, after_beats=0,
        z_stride_beats=stride, anchor_mode='integer')
    return targets, planner, prepared, memory, windows


def terms(model, windows, targets, prepared=None):
    output = model(windows, targets.queries, layout=None if prepared is None else prepared['readout'])
    loss = reconstruction_loss(output, targets, label_smoothing=.05, count_supervision=False,
                               onset_presence_supervision=True, prepared=prepared)
    return output, loss


def test_z2_exact_two_owners_per_real_beat_across_parents_including_empty_tail():
    first, final = target(0), target(16)
    assert first.owner_z_beats == tuple(range(0, 16, 2))
    assert final.owner_z_beats == tuple(range(16, 33, 2))
    rows = [row for t in (first, final) for row in t.onset_presence]
    pairs = [(owner, beat) for owner, beat, label in rows]
    assert len(pairs) == len(set(pairs)) == 64
    assert Counter(beat for owner, beat in pairs) == Counter({beat: 2 for beat in range(32)})
    assert {owner for owner, beat in pairs if beat == 0} == {0, 2}
    assert {owner for owner, beat in pairs if beat == 31} == {30, 32}
    assert {owner for owner, beat in pairs if beat == 15} == {14, 16}
    onset_beats = {group.tick//96 for group in chart().groups}
    assert all(label == (beat in onset_beats) for _, beat, label in rows)
    assert all(not label for owner, beat, label in target(16, empty=True).onset_presence)
    # Distinct real onsets in the same beat contribute only a binary label.
    assert sum(label for owner, beat, label in rows if beat == 0) == 2
    negative_first, negative_last = target(-5, first=-5, end=27), target(11, first=-5, end=27)
    negative = [row for t in (negative_first, negative_last) for row in t.onset_presence]
    assert Counter(beat for _, beat, _ in negative) == Counter({beat: 2 for beat in range(-5, 27)})
    assert all((owner+5) % 2 == 0 for owner, _, _ in negative)
    event_rows = [row for t in (first, final) for row in t.rows if row.query.role == 'MOMENT']
    assert len(event_rows) == len(chart().groups)*2
    assert len({(r.query.owner_z_beat, r.query.origin_tick, r.moment_tick) for r in event_rows}) == len(event_rows)
    # A second onset within beat zero changes no binary label/support count.
    source = chart()
    second = replace(source.groups[0], tick=47)
    denser = replace(source, groups=(source.groups[0], second, *source.groups[1:]))
    dense = overlap_reconstruction_target(denser, start_tick=0, target_beats=16,
        coverage_end_beat=32, z_stride_beats=2, onset_presence_supervision=True)
    assert dense.onset_presence == first.onset_presence


def test_queries_contain_only_geometry_and_transport_preserves_independent_labels():
    targets, planner, prepared, _, _ = setup()
    assert [f.name for f in fields(OnsetPresenceQueries)] == ['owner_z_beats', 'beats', 'valid']
    packet = pack_batch(dict(target=targets, planner=planner, layout=prepared))
    restored = packet.to('cpu')['target']
    torch.testing.assert_close(restored.onset_presence_labels, targets.onset_presence_labels)
    torch.testing.assert_close(restored.queries.onset_presence.owner_z_beats,
                               targets.queries.onset_presence.owner_z_beats)
    assert not hasattr(restored.queries.onset_presence, 'labels')
    batch = SimpleNamespace(reconstruction=targets, planner=planner)
    count = supports(batch)
    assert count['onset_presence'] == 64
    assert count['onset_presence_positive'] + count['onset_presence_negative'] == 64
    assert query_cost(batch) == int(targets.queries.valid.sum()) + 32 + 64


def test_optional_head_keeps_disabled_state_and_rng_exact_and_adds_one_linear():
    torch.manual_seed(801)
    implicit = head(z_stride_beats=2)
    implicit_rng = torch.get_rng_state()
    torch.manual_seed(801)
    explicit = head(z_stride_beats=2, onset_presence_supervision=False)
    assert torch.equal(implicit_rng, torch.get_rng_state())
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
    torch.manual_seed(801)
    enabled = head(z_stride_beats=2, onset_presence_supervision=True)
    assert sum(p.numel() for p in enabled.parameters())-sum(p.numel() for p in implicit.parameters()) == 13
    for name, value in implicit.state_dict().items():
        torch.testing.assert_close(value, enabled.state_dict()[name], atol=0, rtol=0)
    assert not onset_presence_supervision_enabled({})
    assert 'onset_presence' not in COUNT_TERMS


@pytest.mark.parametrize('empty', [False, True])
def test_prepared_loss_fp32_bce_masks_and_presence_only_gradients(empty):
    targets, planner, prepared, memory, windows = setup(empty=empty)
    model = head(z_stride_beats=2, onset_presence_supervision=True)
    output, regular = terms(model, windows, targets)
    fast_output, fast = terms(model, windows, targets, prepared)
    for name in regular:
        assert regular[name].count == fast[name].count
        torch.testing.assert_close(regular[name].total, fast[name].total)
    parameters = (*model.parameters(), memory.values)
    ordinary_gradients = torch.autograd.grad(regular['onset_presence'].mean, parameters,
                                            allow_unused=True, retain_graph=True)
    prepared_gradients = torch.autograd.grad(fast['onset_presence'].mean, parameters,
                                            allow_unused=True, retain_graph=True)
    for ordinary_gradient, prepared_gradient in zip(ordinary_gradients, prepared_gradients):
        if ordinary_gradient is None:
            assert prepared_gradient is None
        else:
            torch.testing.assert_close(ordinary_gradient, prepared_gradient)
    presence = fast['onset_presence']
    assert presence.total.dtype == torch.float32 and presence.count == 64
    valid = output.onset_presence_valid
    expected = F.binary_cross_entropy_with_logits(output.onset_presence_logits[valid].float(),
        targets.onset_presence_labels[valid].float(), reduction='sum')
    torch.testing.assert_close(presence.total, expected)
    assert not set(regular) & COUNT_TERMS
    presence.mean.backward()
    assert memory.values.grad.abs().sum() > 0
    assert model.onset_presence_head.weight.grad.abs().sum() > 0
    assert model.role.weight.grad.abs().sum() > 0
    assert model.ffn[0].weight.grad.abs().sum() > 0
    assert model.count_head.weight.grad is None and model.count_head.bias.grad is None
    assert all(p.grad is None for p in model.field_heads.parameters())
    if empty:
        assert not output.query_valid.any()
        assert presence.positive_count == 0 and presence.negative_count == 64
    # Labels do not affect any forward prediction, and padding never gets loss.
    flipped = replace(targets, onset_presence_labels=1-targets.onset_presence_labels)
    same, changed = terms(model, windows, flipped, prepared)
    torch.testing.assert_close(same.onset_presence_logits, fast_output.onset_presence_logits, atol=0, rtol=0)
    missing = replace(output, onset_presence_valid=output.onset_presence_valid & (targets.queries.onset_presence.owner_z_beats != 0))
    masked = reconstruction_loss(missing, targets, label_smoothing=.05, count_supervision=False,
        onset_presence_supervision=True, prepared=prepared)['onset_presence']
    assert masked.count == 62


def test_bf16_readout_uses_fp32_loss_and_finite_gradient():
    targets, _, prepared, _, windows = setup(empty=True)
    model = head(z_stride_beats=2, onset_presence_supervision=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output, loss = terms(model, windows, targets, prepared)
    assert output.onset_presence_logits.dtype == torch.bfloat16
    assert loss['onset_presence'].total.dtype == torch.float32
    loss['onset_presence'].mean.backward()
    assert torch.isfinite(model.onset_presence_head.weight.grad).all()


def test_presence_common_denominator_reporting_and_final_confusion():
    targets, planner, prepared, _, windows = setup()
    model = head(z_stride_beats=2, onset_presence_supervision=True)
    output, loss = terms(model, windows, targets, prepared)
    config = dict(CONFIG, reconstruction_count_supervision=False,
        reconstruction_onset_presence_supervision=True, reconstruction_onset_presence_weight=1.,
        decoder_train_physical_weight=1., decoder_train_judge_weight=1.,
        decoder_train_trajectory_weight=1.)
    count = supports(SimpleNamespace(reconstruction=targets, planner=planner))
    term = loss['onset_presence']
    result = weighted_loss({'onset_presence': term}, config, denominators=count)
    torch.testing.assert_close(result, term.total/64)
    gradients = torch.autograd.grad(result, tuple(model.parameters()), allow_unused=True, retain_graph=True)
    half = weighted_loss({'onset_presence': term}, config, denominators=dict(count, onset_presence=128))
    gradients_half = torch.autograd.grad(half, tuple(model.parameters()), allow_unused=True)
    for full, part in zip(gradients, gradients_half):
        if full is not None:
            torch.testing.assert_close(full/2, part)
    # Zero planner terms follow the same production aggregation shape.
    from maidata.draft.losses import LossTerm
    loss.update({name: LossTerm(term.total*0, term.count*0) for name in ('planner_continuous','planner_category')})
    metric = AuxiliaryMetrics()
    metric.add(loss, count)
    report = metric.report(config)
    assert report['terms']['onset_presence']['positive_count'] == int(term.positive_count)
    assert report['terms']['onset_presence']['negative_count'] == int(term.negative_count)
    merged = AuxiliaryMetrics(); merged.merge(metric); merged.merge(metric)
    assert merged.report(config)['terms']['onset_presence']['count'] == 128
    joint = JointMetrics(); joint.auxiliary = metric
    assert joint.report(config)['losses']['drafter_onset_presence'] == pytest.approx(float(term.mean.detach()))
    diagnostic = Diagnostics()
    logits = torch.where(targets.onset_presence_labels.bool(), 2., -2.)
    diagnostic.add(replace(output, onset_presence_logits=logits), targets)
    final = diagnostic.report()['onset_presence']
    assert final['count'] == 64
    assert final['precision'] == final['recall'] == final['f1'] == 1.
    assert final['unit'] == 'owner_z_beat'
