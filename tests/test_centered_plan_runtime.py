"""Centered Plan16 pools 17 integer D nodes and keeps five independent keys."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.draft.planner import PlannerHead, select_planner_windows
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage
from maidata.decoder_training import gt_plan_cache, dp_cache
from maidata.decoder_training.frozen_cache import FrozenCache
from maidata.decoder_training.gt_plan_cache import select_plan_conditions
from maidata.inference.conditioning import (Conditions, decode_joint_draft, decoder_z_window,
    empty_plan_geometry, plan_centers_for_starts, with_plan_source)


def config(**updates):
    return dict(dict(planner_layout='centered-overlap', planner_span_beats=16, planner_stride_beats=4,
        draft_z_anchor_mode='integer', draft_z_stride_beats=1, draft_latent_dim=4,
        reconstruction_span_beats=4, target_beats=16, decoder_z_cross_attention=True,
        decoder_z_before_beats=4, decoder_z_after_beats=4, decoder_train_plan_source='predicted'), **updates)


def head():
    return PlannerHead(latent_dim=4, dim=8, heads=2, mlp_dim=12, dropout=0., norm_eps=1e-5,
        rope_base=10000., rope_position_scale=1., span_beats=16,
        layout='centered-overlap', stride_beats=4).eval()


def memory(first=0, last=32):
    beats = torch.arange(first, last+1).float()[None]
    return BeatMemory(torch.randn(1, len(beats[0]), 4), beats, torch.ones_like(beats, dtype=torch.bool))


def test_pool_reads_both_endpoints_and_keeps_each_independent_query_differentiable():
    torch.manual_seed(88)
    z, model = memory(0, 64), head()
    windows = select_planner_windows(z, torch.tensor([0]), torch.tensor([16.]), torch.tensor([True]), config())
    with patch('maidata.draft.planner.F.scaled_dot_product_attention',
               wraps=torch.nn.functional.scaled_dot_product_attention) as attention:
        output = model(windows)
    mask = attention.call_args.kwargs['attn_mask']
    assert mask.sum(-1).flatten().tolist() == [17]*5
    assert output.plan_beats.tolist() == [[16, 20, 24, 28, 32]]
    assert windows.memory.beats.tolist() == [list(range(8, 41))]
    # At center16, nodes8 and24 are both included and node25 is excluded.
    assert mask[0, 0, 0, 0] and mask[0, 0, 0, 16] and not mask[0, 0, 0, 17]
    changed = BeatMemory(windows.memory.values.clone(), windows.memory.beats, windows.memory.valid)
    changed.values[:, 17] += 100
    altered = model(type(windows)(changed, windows.target_start, windows.target_beats, windows.window_valid))
    torch.testing.assert_close(output.plan_values[:, 0], altered.plan_values[:, 0], rtol=0, atol=0)
    output.continuous_logits.square().sum().backward()
    assert model.pool.query.weight.grad is not None
    # The learned constant query is zero initialized, so its input-weight gradient
    # may be zero; the query bias/key projection still learn multi-key selection.
    assert model.pool.query.bias.grad.abs().sum() > 0 and model.pool.key.weight.grad.abs().sum() > 0


def test_full_curve_endpoint_is_valid_and_missing_context_never_copies_an_edge():
    z, model = memory(), head()
    windows = select_planner_windows(z, torch.tensor([0, 0, 0]), torch.tensor([0., 16., 32.]),
                                     torch.ones(3, dtype=torch.bool), config())
    out = model(windows)
    assert out.valid.tolist() == [[True]*5, [True]*5, [True, False, False, False, False]]
    assert out.plan_values[2, 1:].count_nonzero() == 0
    torch.testing.assert_close(out.plan_values[0, 4], out.plan_values[1, 0])
    assert windows.memory.valid[0].sum() == 25 and windows.memory.valid[1].sum() == 25


def test_gt_producer_emits_each_chart_center_once_and_preserves_exact_statistics(tmp_path):
    from test_decoder_gt_plan_cache import record, write_chart
    from maidata.chart_cache.cache import load_chart
    from maidata.draft.chart_data import from_record
    from maidata.draft.plan_targets import MusicPlanTargets
    chart = record()
    charts, output = tmp_path/'charts', tmp_path/'out'
    relative = write_chart(charts, chart)
    rows = [dict(chart_id='42:5', window_id=f'42:5:{start}', start_tick=start*96,
                 end_tick=(start+16)*96) for start in (0, 16)]
    c = config(decoder_train_chart_cache=str(charts), decoder_train_source_root=str(tmp_path/'source'))
    gt_plan_cache.initialize(c, output)
    with patch('maidata.chart_cache.cache.load_chart', wraps=load_chart) as load:
        _, meta = gt_plan_cache.export_variant(('song', rows, {'42:5': relative}))
    assert load.call_count == 1 and meta['anchors'] == 9 and meta['consumer_read_verified']
    bank = torch.load(output/meta['path'], weights_only=True)
    assert bank['center_beats'].tolist() == list(range(0, 33, 4))
    assert len(set(zip(bank['chart_ids'], bank['center_beats'].tolist()))) == 9
    targets = MusicPlanTargets(from_record(chart))
    expected = [targets.window((center-8)*96, (center+8)*96) for center in range(0, 33, 4)]
    torch.testing.assert_close(bank['continuous'], torch.tensor([target.continuous for target in expected]))
    assert bank['category'].tolist() == [target.category for target in expected]
    (output/'manifest.json').write_text(json.dumps(dict(schema=gt_plan_cache.CENTERED_SCHEMA,
        planner_layout='centered-overlap', plan_span_beats=16, plan_stride_beats=4, tick_rate=96,
        groups={'song': meta})))
    loaded = gt_plan_cache.GTPlanCache(output, c).read('song')
    plan, valid, centers = select_plan_conditions(loaded, 'ignored', '42:5', 16, c)
    assert centers.tolist() == [16, 20, 24, 28, 32] and valid.all()
    torch.testing.assert_close(plan, bank['plan_values'][4:])
    with pytest.raises(ValueError, match='planner_layout'):
        gt_plan_cache.GTPlanCache(output, dict(c, planner_layout='legacy'))


def test_centered_dp_export_deduplicates_shared_boundary_and_frozen_consumer_reads_it(tmp_path):
    c, z, model = config(), memory(), head()
    starts = torch.tensor([0., 16.])
    windows = select_planner_windows(z, torch.tensor([0, 0]), starts, torch.ones(2, dtype=torch.bool), c)
    plans = model(windows)
    batch = dict(variant='song', chart_to_group={'A': 0}, conditions=[], groups=torch.tensor([0, 0]),
                 starts=starts, window_ids=['A:0', 'A:16'], chart_ids=['A', 'A'])
    path = tmp_path/'banks/song.pt'
    meta = dp_cache.save_outputs(path, batch, z, plans, 'producer', z_stride_beats=1,
        plan_span_beats=16, planner_layout='centered-overlap', planner_stride_beats=4,
        z_anchor_mode='integer', reconstruction_span_beats=4)
    from maidata.decoder_training.cache_schema import DP_SCHEMA, Z_REPRESENTATION
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION,
        z_width=4, z_stride_beats=1, z_anchor_mode='integer', planner_layout='centered-overlap',
        plan_span_beats=16, plan_stride_beats=4,
        producer={'recipe': {'draft_z_stride_beats': 1, 'draft_z_anchor_mode': 'integer'}},
        groups={'song': meta})))
    frozen = FrozenCache([], dict(c, decoder_train_dp_cache=str(tmp_path)), {})
    bank = frozen.read('song')['planner']
    assert bank['plan_values'].shape == (9, 12)
    chosen, valid, centers = select_plan_conditions(bank, 'A:16', 'A', 16, c)
    assert valid.all() and centers.tolist() == [16, 20, 24, 28, 32]
    torch.testing.assert_close(chosen, plans.plan_values[1])


def test_inference_keeps_z23_plan5_independent_positions_and_gt_none_shapes():
    c, coverage = config(), SongCoverage(0, 1, 16, 8.)
    starts = coverage.starts(include_tail=True)
    z_empty, plans, valid = empty_plan_geometry(starts, coverage, c, 'cpu')
    audio = memory(0, 1)
    local = Conditions(audio, z_empty, plans, valid, starts, coverage, 'window',
                       plan_centers_for_starts(starts, c, 'cpu'))
    z = memory()
    result = decode_joint_draft(lambda audio, inputs: z, head(), c, local, audio,
                                difficulty=5, level=14., device='cpu')
    assert result.plans.shape == (2, 5, 12) and result.plan_valid.all()
    assert result.plan_beats.tolist() == [[0, 4, 8, 12, 16], [16, 20, 24, 28, 32]]
    selected = decoder_z_window(result, 0, c, 'cpu')
    assert selected.beats.tolist() == [list(range(-3, 20))]
    assert selected.valid[0, :3].eq(False).all() and selected.valid[0, 3:].all()
    selected = decoder_z_window(result, 1, c, 'cpu')
    assert selected.valid[0, :-3].all() and selected.valid[0, -3:].eq(False).all()
    no_z = decoder_z_window(result, 0, dict(c, decoder_z_cross_attention=False), 'cpu')
    assert no_z.values.shape == (1, 23, 4) and not no_z.valid.any()
    doc = dict(planner_layout='centered-overlap', plan_span_beats=16, plan_stride_beats=4,
        windows=[dict(start_beat=start, center_beats=result.plan_beats[i].tolist(),
                      valid=[True]*5, plan12=result.plans[i].tolist()) for i, start in enumerate(starts)])
    gt = with_plan_source(local, 'gt', doc)
    assert gt.plan_valid.all() and gt.plan_beats.equal(result.plan_beats)
    absent = with_plan_source(gt, 'none')
    assert absent.plans.shape == (2, 5, 12) and not absent.plan_valid.any()
    assert absent.plan_beats.equal(gt.plan_beats)


def test_centered_export_rejects_existing_output_without_touching_it(tmp_path):
    marker = tmp_path/'keep.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError, match='new or empty'):
        gt_plan_cache.export(config(), tmp_path)
    assert marker.read_text() == 'keep'


def test_centered_export_completes_with_an_empty_existing_validation_split(tmp_path):
    from test_decoder_gt_plan_cache import record, write_chart
    chart_root = tmp_path/'charts'
    relative = write_chart(chart_root, record())
    c = config(decoder_train_chart_cache=str(chart_root), decoder_train_source_root=str(tmp_path/'source'))
    for side in ('train', 'validation'):
        root = tmp_path/side
        root.mkdir()
        populated = side == 'train'
        (root/'plan.json').write_text(json.dumps(dict(files={'windows': 'windows.jsonl'},
            charts=[dict(chart_id='42:5', chart_cache_path=relative)] if populated else [],
            audio_groups=[dict(audio_variant='song')] if populated else [])))
        rows = [dict(chart_id='42:5', audio_variant='song', window_id=f'42:5:{start}',
                     start_tick=start*96, end_tick=(start+16)*96) for start in (0, 16)] if populated else []
        (root/'windows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        c['decoder_train_'+side+'_plan'] = str(root)
    output = tmp_path/'export'
    manifest = gt_plan_cache.export(c, output, workers=1)  # Independent tiny CPU fixture.
    assert manifest['schema'] == gt_plan_cache.CENTERED_SCHEMA and manifest['complete']
    assert manifest['splits'] == dict(train=['song'], validation=[])
    assert manifest['groups']['song']['anchors'] == 9
    reader = gt_plan_cache.GTPlanCache(output, c)
    assert reader.read('song')['valid'].all()
