"""Integer centers, exact gathers and once-owned overlapping R supervision."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import patch

from maidata.chart import Note, Route, SlideDuration
from maidata.draft.chart_data import ChartData, OnsetGroup, PlanNote
from maidata.draft.conditions import ChartCondition, make_draft_inputs
from maidata.draft.layout import prepare_centers, prepare_selection
from maidata.draft.losses import prepare_supervision, reconstruction_loss
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.reconstruction import ReconstructionHead
from maidata.draft.targets import (reconstruction_target, overlap_reconstruction_target,
                                  collate_reconstruction, collate_planner, planner_target)
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage, draft_queries, gather_centers, select_windows
from maidata.time import TimingMap
from maidata.training.data import AudioDataset
from maidata.training.runner import supports
from test_pretrain_runtime import config as producer_config


def chart():
    ticks = [23, 96, 14*96+11, 15*96+95, 16*96, 17*96+32, 31*96+47]
    groups, notes = [], []
    for index, tick in enumerate(ticks):
        tokens = (Note.make('slide', 'A2', ex=1), Route(0, 1), SlideDuration(24))
        groups.append(OnsetGroup(tick, tokens, tuple(index*10+k for k in range(3)), index*10+9))
        notes.append(PlanNote(tick, 'slide', 2))
    return ChartData(tuple(groups), tuple(notes), TimingMap([(0, 120), (8, 180)]), Fraction(32))


def target(start):
    return overlap_reconstruction_target(chart(), start_tick=start*96, target_beats=16, coverage_end_beat=32)


def head(**overrides):
    options = dict(latent_dim=12, ffn_dim=19, dropout=0., norm_eps=1e-5,
        target_beats=16, position_base=10000., position_scale=1., z_stride_beats=1,
        count_loss='mse', anchor_mode='integer', span_beats=4)
    return ReconstructionHead(**dict(options, **overrides))


def pair_key(row):
    q = row.query
    return (q.owner_z_beat, row.source_token_index, q.role, q.origin_tick,
            q.note_ordinal, q.track, q.segment, q.current_route_id)


def test_integer_queries_include_coverage_endpoint_and_keep_legacy_grid():
    coverages = [SongCoverage(0, 2, 16, 30.), SongCoverage(5, 1, 16, 20.)]
    mapping = torch.tensor([0, 1])
    centers, valid = draft_queries(coverages, mapping, z_stride_beats=1, anchor_mode='integer')
    assert centers[0].tolist() == list(range(49))
    assert centers[1, :33].tolist() == list(range(5, 38))
    assert valid.sum(-1).tolist() == [49, 33] and not valid[1, 33:].any()
    old, old_valid = draft_queries(coverages, mapping, z_stride_beats=4)
    assert old[0].tolist() == list(range(2, 48, 4))
    assert old[1, :8].tolist() == list(range(7, 37, 4))
    assert old_valid.sum(-1).tolist() == [12, 8]
    condition = ChartCondition('audio', 5, 14.5, '14.5')
    inputs, groups = make_draft_inputs([condition], ['audio'], coverages[:1], device='cpu',
        z_stride_beats=1, anchor_mode='integer')
    torch.testing.assert_close(inputs.query_beats, centers[:1])
    assert groups.tolist() == [0]


def test_exact_center_gather_masks_edges_keeps_duplicates_and_gradients():
    values = torch.randn(2, 33, 5, requires_grad=True)
    beats = torch.arange(33).float()[None].expand(2, -1)
    valid = torch.ones(2, 33, dtype=torch.bool)
    valid[1] = False
    z = BeatMemory(values, beats, valid)
    desired = torch.tensor([[-3., 0., 4., 4., 32., 33.], [-3., 0., 4., 4., 32., 33.]])
    layout = prepare_centers(beats, valid, torch.tensor([0, 1]), desired, torch.ones(2, dtype=torch.bool))
    out = gather_centers(z, torch.tensor([0, 1]), desired, torch.ones(2, dtype=torch.bool), layout=layout)
    torch.testing.assert_close(out.beats, desired)
    assert out.valid.tolist() == [[False, True, True, True, True, False], [False]*6]
    assert out.values[~out.valid].count_nonzero() == 0
    out.values.sum().backward()
    assert values.grad[0, 4].eq(2).all() and values.grad[0, 0].eq(1).all()
    assert values.grad[0, 32].eq(1).all() and values.grad[1].count_nonzero() == 0
    selected = prepare_selection(beats, valid, torch.tensor([0]), torch.tensor([16.]),
        torch.tensor([True]), target_beats=16, before_beats=0, after_beats=0,
        z_stride_beats=1, anchor_mode='integer')
    assert selected['beats'][0].tolist() == list(range(16, 33))
    assert selected['valid'].all()


def test_global_overlap_keeps_all_distinct_owners_and_no_parent_duplicates():
    source = reconstruction_target(chart(), start_tick=0, target_beats=32)
    first, last = target(0), target(16)
    assert first.owner_z_beats == tuple(float(t) for t in range(16))
    assert last.owner_z_beats == tuple(float(t) for t in range(16, 33))
    actual = [row for item in (first, last) for row in item.rows]
    expected = []
    for row in source.rows:
        beat = int(row.query.beat)
        for owner in range(max(0, beat-1), min(33, beat+3)):
            expected.append(replace(row, query=replace(row.query, owner_z_beat=float(owner))))
    assert Counter(map(pair_key, actual)) == Counter(map(pair_key, expected))
    assert len(set(map(pair_key, actual))) == len(actual)
    original = {(r.source_token_index, r.query.role): r for r in source.rows}
    for row in actual:
        reference = original[row.source_token_index, row.query.role]
        assert replace(row.query, owner_z_beat=None) == reference.query
        assert row.fields == reference.fields and row.moment_tick == reference.moment_tick
        offset = int(row.query.beat)-(row.query.owner_z_beat-2)
        assert offset in (0, 1, 2, 3)
    before = {row.query.owner_z_beat for row in first.rows if int(row.query.beat) == 15}
    after = {row.query.owner_z_beat for row in last.rows if int(row.query.beat) == 15}
    assert before == {14., 15.} and after == {16., 17.}
    zero = [row for row in actual if row.query.role == 'MOMENT' and row.query.beat == 16.5]
    assert len(zero) == 4 and all(row.moment_tick == 0 for row in zero)


def test_overlap_r_exact_owner_gradient_offsets_and_prepared_loss_denominators():
    torch.manual_seed(71)
    targets = collate_reconstruction([target(0), target(16)], target_beats=16, device='cpu')
    assert targets.queries.owner_z_beats is not None
    plans = MusicPlanTargets(chart())
    p_target = collate_planner([planner_target(plans, 0, 16*96, layout='centered-overlap'),
        planner_target(plans, 16*96, 32*96, layout='centered-overlap', last_parent=True)], device='cpu')
    prepared = prepare_supervision(targets, p_target)
    counts = supports(SimpleNamespace(reconstruction=targets, planner=p_target))
    assert counts['planner_category'] == 9
    assert counts['moment96'] == sum(row.query.role == 'MOMENT' for t in (target(0), target(16)) for row in t.rows)
    assert prepared['counts']['moment96'] == counts['moment96']
    values = torch.randn(1, 33, 12, requires_grad=True)
    z = BeatMemory(values, torch.arange(33).float()[None], torch.ones(1, 33, dtype=torch.bool))
    windows = select_windows(z, torch.tensor([0, 0]), torch.tensor([0., 16.]), torch.ones(2, dtype=torch.bool),
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=1, anchor_mode='integer')
    model = head()
    output = model(windows, targets.queries)
    fast_output = model(windows, targets.queries, layout=prepared['readout'])
    torch.testing.assert_close(output.moment_logits, fast_output.moment_logits)
    regular = reconstruction_loss(output, targets, label_smoothing=.05, count_supervision=False)
    fast = reconstruction_loss(fast_output, targets, label_smoothing=.05, count_supervision=False, prepared=prepared)
    for key in regular:
        assert regular[key].count == fast[key].count
        torch.testing.assert_close(regular[key].total, fast[key].total)
    row = targets.queries.valid[0].nonzero().flatten()[0]
    owner = int(targets.queries.owner_z_beats[0, row])
    gradient, = torch.autograd.grad((output.moment_logits[0, row]*torch.arange(96)).sum(), values, retain_graph=True)
    assert gradient[0, owner].abs().sum() > 0
    mask = torch.arange(33) != owner
    assert gradient[0, mask].count_nonzero() == 0
    altered = replace(z, values=z.values.clone())
    altered.values[:, 16] = altered.values[:, 16]+torch.arange(12)
    changed_windows = select_windows(altered, torch.tensor([0, 0]), torch.tensor([0., 16.]), torch.ones(2, dtype=torch.bool),
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=1, anchor_mode='integer')
    changed = model(changed_windows, targets.queries)
    unrelated = targets.queries.owner_z_beats != 16
    torch.testing.assert_close(changed.moment_logits[unrelated], output.moment_logits[unrelated])
    related = (targets.queries.owner_z_beats == 16) & targets.queries.valid
    assert not torch.allclose(changed.moment_logits[related], output.moment_logits[related])


def test_empty_overlap_tail_keeps_owner_metadata_and_no_fake_queries():
    empty = replace(chart(), groups=(), notes=())
    target_empty = overlap_reconstruction_target(empty, start_tick=16*96, target_beats=16, coverage_end_beat=32)
    collated = collate_reconstruction([target_empty], target_beats=16, device='cpu')
    assert target_empty.owner_z_beats[-1] == 32 and not target_empty.rows
    assert collated.queries.owner_z_beats is not None and not collated.queries.valid.any()
    memory = BeatMemory(torch.randn(1, 17, 12), torch.arange(16, 33).float()[None], torch.ones(1, 17, dtype=torch.bool))
    from maidata.draft.types import WindowLatents
    output = head()(WindowLatents(memory, torch.tensor([16.]), 16, torch.tensor([True])), collated.queries)
    assert not output.query_valid.any() and output.moment_logits.count_nonzero() == 0


def test_legacy_r_constructor_and_owner_path_are_unchanged():
    torch.manual_seed(72)
    implicit = head(anchor_mode='block-center', span_beats=None, z_stride_beats=4)
    explicit = head(anchor_mode='block-center', span_beats=4, z_stride_beats=4)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    targets = collate_reconstruction([reconstruction_target(chart(), start_tick=0, target_beats=16)],
                                    target_beats=16, device='cpu')
    assert targets.queries.owner_z_beats is None
    from maidata.draft.types import WindowLatents
    memory = BeatMemory(torch.randn(1, 4, 12), torch.tensor([[2.,6.,10.,14.]]), torch.ones(1,4,dtype=torch.bool))
    windows = WindowLatents(memory, torch.tensor([0.]), 16, torch.tensor([True]))
    a, b = implicit(windows, targets.queries), explicit(windows, targets.queries)
    torch.testing.assert_close(a.moment_logits, b.moment_logits, atol=0., rtol=0.)
    assert implicit.state_dict().keys() == head().state_dict().keys()


def test_dataset_labels_global_overlap_and_last_plan_endpoint_without_extra_reads(monkeypatch):
    import maidata.training.data as data
    rows = [dict(window_id=f'chart:{s}', chart_id='chart', start_beat=s, start_tick=s*96, end_tick=(s+16)*96)
            for s in (16, 0)]
    plan = SimpleNamespace(groups={'audio':dict(chart_ids=['chart'], audio_end_seconds=8.,
        timing=dict(first_seconds=0., bpm=[[0,1,120.]]))}, charts={'chart':dict(chart_cache_path='unused')},
        rows=lambda variant, epoch:rows)
    config = dict(pretrain_audio_mode='joint', pretrain_audio_cache='unused', pretrain_mel_root='unused',
        pretrain_chart_cache='unused', pretrain_source_root='unused', target_beats=16, audio_after_beats=4,
        draft_z_anchor_mode='integer', draft_z_stride_beats=1, reconstruction_span_beats=4,
        planner_layout='centered-overlap', planner_span_beats=16, planner_stride_beats=4)
    loaded = []
    monkeypatch.setattr(data, 'load_grid', lambda *a, **k:dict(first_beat=np.array(0)))
    monkeypatch.setattr(data, 'load_chart', lambda path, **k:loaded.append(path) or dict(source=dict(song_id='1',difficulty='5')))
    monkeypatch.setattr(data, 'from_record', lambda *a, **k:chart())
    monkeypatch.setattr(data, 'read_condition', lambda *a, **k:ChartCondition('audio', 5, 14.5, '14.5'))
    item = AudioDataset(plan, config)[('audio', 1)]
    assert len(loaded) == 1
    assert item['windows'][0][3].owner_z_beats[-1] == 32
    assert item['windows'][1][3].owner_z_beats[-1] == 15
    assert item['windows'][0][4][-1] is not None and item['windows'][1][4][-1] is None


def test_real_e_d_r_p_packed_forward_backward_owns_terminal_center_once(producer_config):
    from maidata.training.data import collate_audio
    from maidata.training.network import Pretrainer
    from maidata.training.pretrain import weighted_loss
    c = dict(producer_config, target_beats=16, target_stride_beats=16, audio_before_beats=4, audio_after_beats=4,
        decoder_z_before_beats=4, decoder_z_after_beats=4, draft_z_stride_beats=1,
        draft_z_anchor_mode='integer', reconstruction_span_beats=4, reconstruction_count_supervision=False,
        planner_span_beats=16, planner_layout='centered-overlap', planner_stride_beats=4)
    rng = np.random.default_rng(73)
    coordinates = np.arange(32*5).reshape(32, 5)/5
    sample_valid = coordinates < 31.8
    mel = rng.normal(size=(32, 5, 7)).astype(np.float32)
    aux = rng.normal(size=(32, 5, 32)).astype(np.float32)
    mel[~sample_valid], aux[~sample_valid] = 0, 0
    grid = dict(mel_patches=mel, sample_valid=sample_valid,
        beat_centers=np.arange(32, dtype=np.float32)+.5, aux32_grid=aux, aux32_centers=aux.mean(1),
        sample_times_sec=coordinates*.5, sample_indices=np.rint(coordinates*.5*24000).astype(np.int64),
        grid_k=np.arange(32*5).reshape(32, 5), first_beat=np.array(0))
    coverage = SongCoverage(0, 2, 16, 31.8)
    plans = MusicPlanTargets(chart())
    windows = [(str(start), 'chart', start, overlap_reconstruction_target(chart(), start_tick=start*96,
        target_beats=16, coverage_end_beat=48), planner_target(plans, start*96, (start+16)*96,
        layout='centered-overlap', last_parent=start == 32)) for start in (0, 16, 32)]
    item = dict(variant='audio', grid=grid, coverage=coverage,
        conditions={'chart':ChartCondition('audio', 5, 14.5, '14.5')}, windows=windows)
    packet = collate_audio([item], config=c, packed=True)
    batch = packet.to('cpu')
    assert batch.draft_inputs.query_beats.tolist() == [list(range(49))]
    assert batch.reconstruction.queries.owner_z_beats is not None
    assert batch.planner.valid.tolist() == [[True]*4+[False], [True]*4+[False], [True]*5]
    assert supports(batch)['planner_category'] == 13
    assert batch.layouts['reconstruction']['beats'].shape == (3, 17)
    assert batch.layouts['planner']['beats'].shape == (3, 33)
    model = Pretrainer(c)
    with patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as encode, \
         patch.object(model.draft, 'forward', wraps=model.draft.forward) as draft:
        terms, (_, plan, z) = model(batch)
    assert encode.call_count == draft.call_count == 1
    assert z.values.shape == (1, 49, 24)
    assert plan.plan_values.shape == (3, 5, 12)
    assert plan.plan_beats.tolist() == [[0.,4.,8.,12.,16.], [16.,20.,24.,28.,32.], [32.,36.,40.,44.,48.]]
    assert 'count' not in terms and int(terms['planner_category'].count) == 13
    loss = weighted_loss(terms, c, denominators=supports(batch))
    loss.backward()
    for name in ('encoder', 'draft', 'reconstruction', 'planner'):
        assert sum(float(p.grad.abs().sum()) for p in getattr(model, name).parameters() if p.grad is not None) > 0
