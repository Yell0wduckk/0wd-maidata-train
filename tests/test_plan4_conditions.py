"""Four-beat condition slots use neighboring chart segments and exact geometry."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.decoder.types import Memory
from maidata.decoder_training import conditioning, preparation
from maidata.decoder_training import gt_plan_cache
from maidata.decoder_training.gt_plan_cache import (GTPlanCache, PLAN4_SCHEMA, SCHEMA,
                                                   select_plan_window)
from maidata.draft.planner import PlannerHead
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage
from maidata.inference.conditioning import (Conditions, decode_joint_draft,
    decoder_z_window, empty_plan_geometry, with_plan_source)


def config4(**overrides):
    return dict(dict(planner_span_beats=4, target_beats=16, draft_z_stride_beats=4,
        draft_latent_dim=4, decoder_z_before_beats=4, decoder_z_after_beats=4,
        decoder_z_cross_attention=True, decoder_train_plan_source='gt'), **overrides)


def bank4():
    # The second difficulty shares beats, but must never supply chart A's slots.
    charts = ['A', 'A', 'A', 'B', 'B', 'B']
    starts = torch.tensor([0, 16, 32, 0, 16, 32])[:, None]+torch.arange(4)*4
    centers = starts.double()+2
    values = torch.zeros(6, 4, 12)
    values[..., 0] = centers.float()
    values[..., 8] = 1
    values[3:, :, 0] += 100
    values[2, :, :8] = 0  # Complete chart-A extra tail is valid and empty.
    return dict(schema=PLAN4_SCHEMA, plan_span_beats=4, chart_ids=charts,
        window_ids=[f'{chart}:{start}' for chart, start in zip(charts, starts[:, 0].tolist())],
        plan_values=values, valid=torch.ones(6, 4, dtype=torch.bool),
        center_beats=centers, start_ticks=starts*96, end_ticks=(starts+4)*96)


def write_bank(root, bank=None, schema=PLAN4_SCHEMA):
    torch.save(bank4() if bank is None else bank, root/'bank.pt')
    manifest = dict(schema=schema, groups={'song': {'path': 'bank.pt'}})
    if schema == PLAN4_SCHEMA:
        manifest.update(plan_span_beats=4, parent_window_beats=16, plans_per_window=4, tick_rate=96)
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_plan4_reader_gathers_actual_neighbors_without_replaying_or_rescanning(tmp_path):
    write_bank(tmp_path)
    with patch('torch.load', side_effect=AssertionError('construction reads only manifest')):
        reader = GTPlanCache(tmp_path, config4())
    with patch('torch.isfinite', side_effect=AssertionError('accepted banks are not rescanned')):
        bank = reader.read('song')
    first, first_valid = select_plan_window(bank, 'A:0', 'A', 0, config4())
    assert first.shape == (6, 12)
    assert first_valid.tolist() == [False, True, True, True, True, True]
    assert first[:, 0].tolist() == [0, 2, 6, 10, 14, 18]
    middle, valid = select_plan_window(bank, 'A:16', 'A', 16, config4())
    assert valid.all() and middle[0, 0] == 14 and middle[4, 0] == 30
    # The next slot is an actual empty tail plan, not a copy of the edge at 30.
    assert middle[5, :8].count_nonzero() == 0 and middle[5, 8] == 1
    tail, tail_valid = select_plan_window(bank, 'A:32', 'A', 32, config4())
    assert tail_valid.tolist() == [True, True, True, True, True, False]
    assert tail[0, 0] == 30 and tail[1:5, :8].count_nonzero() == 0
    assert tail[1:5, 8].eq(1).all() and tail[-1].count_nonzero() == 0
    other, _ = select_plan_window(bank, 'B:16', 'B', 16, config4())
    assert other[0, 0] == 114  # Absolute center alone is not the ownership key.


def test_plan4_cache_and_configuration_never_silently_fall_back(tmp_path):
    write_bank(tmp_path)
    with pytest.raises(ValueError, match='schema differs'):
        GTPlanCache(tmp_path, dict(planner_span_beats=16))
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    manifest['schema'] = SCHEMA
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema differs'):
        GTPlanCache(tmp_path, config4())
    bank = bank4()
    del bank['plan_span_beats']
    with pytest.raises(ValueError, match='cached plan_span_beats'):
        select_plan_window(bank, 'A:0', 'A', 0, config4())


def test_explicit_plan4_export_produces_four_real_intervals_per_base_window(tmp_path):
    from test_decoder_gt_plan_cache import record, write_chart
    from maidata.chart_cache.cache import load_chart
    from maidata.draft.chart_data import from_record
    from maidata.draft.plan_targets import MusicPlanTargets
    chart = record()
    chart_root, output = tmp_path/'charts', tmp_path/'output'
    relative = write_chart(chart_root, chart)
    rows = [dict(window_id=f'42:5:{start}', chart_id='42:5', start_tick=start*96,
                 end_tick=(start+16)*96) for start in (0, 16)]
    gt_plan_cache.initialize(config4(decoder_train_chart_cache=str(chart_root),
        decoder_train_source_root=str(tmp_path/'source')), output)
    with patch('maidata.chart_cache.cache.load_chart', wraps=load_chart) as load:
        _, metadata = gt_plan_cache.export_variant(('42', rows, {'42:5': relative}))
    assert load.call_count == 1 and metadata['shape'] == [2, 4, 12]
    saved = torch.load(output/metadata['path'], weights_only=True)
    assert saved['schema'] == PLAN4_SCHEMA and saved['chart_ids'] == ['42:5', '42:5']
    assert saved['valid'].shape == (2, 4) and saved['valid'].all()
    assert saved['center_beats'].tolist() == [[2, 6, 10, 14], [18, 22, 26, 30]]
    targets = MusicPlanTargets(from_record(chart))
    expected = [targets.window(start, start+384) for start in range(0, 32*96, 384)]
    torch.testing.assert_close(saved['continuous'],
        torch.tensor([target.continuous for target in expected]).reshape(2, 4, 8))
    assert saved['category'].flatten().tolist() == [target.category for target in expected]


def frozen_items(bank):
    beats = torch.arange(2, 48, 4).float()[None]
    dp = dict(chart_to_group={'A': 0}, window_index={'A:0': 0, 'A:16': 1, 'A:32': 2},
        z=dict(values=torch.ones(1, 12, 4), beats=beats, valid=torch.ones_like(beats, dtype=torch.bool)))
    windows = [dict(identity=(f'A:{start}', occurrence), chart_id='A', start=start)
               for start, occurrence in ((0, 0), (16, 0), (16, 1), (32, 0))]
    return [dict(variant='song', dp=dp, gt_plan=bank, windows=windows)], windows


@pytest.mark.parametrize('use_z', [True, False])
def test_frozen_condition_collation_deduplicates_occurrences_and_aligns_six_slots(use_z):
    bank = bank4()
    bank['window_index'] = {key: i for i, key in enumerate(bank['window_ids'])}
    items, windows = frozen_items(bank)
    audio = Memory(torch.empty(1, 2, 0), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    with patch.object(conditioning, 'select_plan_window', wraps=select_plan_window) as gather:
        conditions = conditioning.collate_conditions(items, windows,
            config4(decoder_z_cross_attention=use_z), audio=audio)
    assert gather.call_count == 3
    assert conditions.occurrence_mapping.tolist() == [0, 1, 1, 2]
    assert conditions.plan.shape == (3, 6, 12) and conditions.plan_valid.shape == (3, 6)
    assert conditions.decoder_z.beats.tolist() == [
        [-2, 2, 6, 10, 14, 18], [14, 18, 22, 26, 30, 34], [30, 34, 38, 42, 46, 50]]
    if use_z:
        assert torch.equal(conditions.plan_valid, conditions.decoder_z.valid)
    else:
        assert not conditions.decoder_z.valid.any() and conditions.plan_valid.any()


def test_predicted_plan4_bank_uses_the_same_chart_center_lookup():
    bank = bank4()
    items, windows = frozen_items(bank)
    items[0]['dp']['planner'] = bank
    del items[0]['gt_plan']
    audio = Memory(torch.empty(1, 2, 0), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    conditions = conditioning.collate_conditions(items, windows,
        config4(decoder_train_plan_source='predicted'), audio=audio)
    assert conditions.plan[1, 0, 0] == 14 and conditions.plan[1, 5, 8] == 1
    legacy = dict(bank, plan_span_beats=16)
    items[0]['dp']['planner'] = legacy
    with pytest.raises(ValueError, match='cached plan_span_beats'):
        conditioning.collate_conditions(items, windows,
            config4(decoder_train_plan_source='predicted'), audio=audio)


def test_prepared_attention_layout_keeps_each_plan_validity_slot(monkeypatch):
    bank = bank4()
    bank['window_index'] = {key: i for i, key in enumerate(bank['window_ids'])}
    items, windows = frozen_items(bank)
    audio = Memory(torch.empty(1, 2, 0), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    c = config4(decoder_z_cross_attention=False, decoder_train_row_budget=8192,
                decoder_train_chunk_token_budget=None, decoder_train_plan_input=True)
    conditions = conditioning.collate_conditions(items, windows, c, audio=audio)
    batch = SimpleNamespace(conditions=conditions, audio_lengths=[2],
        windows=[dict(window, length=2, history_dropped=False) for window in windows])
    monkeypatch.setattr(preparation, 'collate_prepared', lambda rows:
        (SimpleNamespace(valid=torch.ones(len(rows), 2, dtype=torch.bool)), None, None))
    preparation.prepare_batch(batch, c)
    assert len(batch.chunks) == 1
    expected = torch.cat((torch.zeros(4, 6, dtype=torch.bool),
        conditions.plan_valid[conditions.occurrence_mapping]), -1)
    assert torch.equal(batch.chunks[0].audio_layout['condition_rows']['key_indices'],
                       expected.flatten().nonzero().flatten())


def inference_case():
    coverage = SongCoverage(0, 2, 16, 32.)
    starts = coverage.starts(include_tail=True)
    z, plans, valid = empty_plan_geometry(starts, coverage, config4(), 'cpu')
    audio = BeatMemory(torch.ones(1, 2, 4), torch.tensor([[0., 1.]]), torch.ones(1, 2, dtype=torch.bool))
    return Conditions(audio, z, plans, valid, starts, coverage, 'window')


def test_predicted_inference_pools_all_six_independent_four_beat_positions():
    local = inference_case()
    beats = torch.arange(2, 48, 4).float()[None]
    z = BeatMemory(torch.randn(1, 12, 4), beats, torch.ones(1, 12, dtype=torch.bool))
    planner = PlannerHead(latent_dim=4, dim=8, heads=2, mlp_dim=12, dropout=0., norm_eps=1e-5,
                          rope_base=10000., rope_position_scale=1., span_beats=4)
    with patch.object(planner, 'forward', wraps=planner.forward) as forward:
        conditions = decode_joint_draft(lambda audio, inputs: z, planner, config4(), local, local.audio,
                                       difficulty=5, level=14., device='cpu')
    window = forward.call_args.args[0]
    assert window.target_start.tolist() == [-4, 12, 28] and window.target_beats == 24
    assert conditions.plans.shape == (3, 6, 12)
    assert torch.equal(conditions.plan_valid, window.memory.valid)
    assert conditions.plan_valid.tolist() == [[False, True, True, True, True, True],
        [True]*6, [True, True, True, True, True, False]]
    for use_z in (True, False):
        chosen = decoder_z_window(conditions, 1, config4(decoder_z_cross_attention=use_z), 'cpu')
        assert chosen.beats.tolist() == [[14, 18, 22, 26, 30, 34]]
        assert chosen.valid.all() if use_z else not chosen.valid.any()
    # Z-off GT/none uses the same six coordinates without executing D/P.
    chosen = decoder_z_window(local, 2, config4(decoder_z_cross_attention=False), 'cpu')
    assert chosen.values.shape == (1, 6, 4) and chosen.beats.tolist() == [[30, 34, 38, 42, 46, 50]]


def gt_document(conditions):
    rows = []
    for start in conditions.starts:
        centers = [start+offset for offset in (-2, 2, 6, 10, 14, 18)]
        rows.append(dict(start_beat=start, center_beats=centers, plan12=[[.5]*12 for _ in range(6)],
            valid=[conditions.coverage.first_beat <= center < conditions.coverage.draft_beats for center in centers]))
    return dict(plan_span_beats=4, windows=rows)


def test_gt_and_none_inference_keep_six_slot_shapes_and_valid_empty_tail():
    original = inference_case()
    conditions = with_plan_source(original, 'gt', gt_document(original))
    assert conditions.plans.shape == (3, 6, 12) and conditions.plan_valid.shape == (3, 6)
    assert conditions.plan_valid[2, 1:5].all() and not conditions.plan_valid[2, -1]
    assert conditions.plans[0, 0].count_nonzero() == 0
    absent = with_plan_source(conditions, 'none')
    assert absent.plans.shape == conditions.plans.shape
    assert absent.plans.count_nonzero() == absent.plan_valid.count_nonzero() == 0


@pytest.mark.parametrize('error', ['span', 'centers', 'mask', 'mask_type', 'shape'])
def test_gt_plan4_json_rejects_ambiguous_positions_or_masks(error):
    conditions = inference_case()
    document = gt_document(conditions)
    if error == 'span':
        document.pop('plan_span_beats')
    elif error == 'centers':
        document['windows'][0]['center_beats'][0] = 0
    elif error == 'mask':
        document['windows'][0]['valid'][0] = True
    elif error == 'mask_type':
        document['windows'][0]['valid'][1] = 1
    else:
        document['windows'][0]['plan12'].pop()
    with pytest.raises(ValueError, match='GT'):
        with_plan_source(conditions, 'gt', document)
