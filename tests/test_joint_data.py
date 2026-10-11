"""Joint complete-group batches preserve supervision and local audio ownership."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.audio.types import ScaleMemory
from maidata.draft.windows import assemble_full_p0
from maidata.joint_training import data
from maidata.joint_training.data import JointDataset, JointPlan, collate_audio
from maidata.training.data import AudioBatchSampler, AudioPlan
from maidata.training.runner import supports
from test_decoder_training import config, encoder_config, training_item
from test_pretrain_runtime import item as auxiliary_item, make_plan


def joint_item(index, config, monkeypatch, *, remaster=False, training=True):
    base = auxiliary_item(index, config, remaster)
    decoder = training_item(index, config, monkeypatch, remaster=remaster, training=training)
    windows = [dict(window, sequence_index=i)
               for i, window in enumerate(decoder['windows'])]
    unique = list(dict.fromkeys(window['identity'][0] for window in windows))
    # GT plan values are deliberately independent of the predicted-plan fixture.
    plan = torch.arange(len(unique)*12, dtype=torch.float32).reshape(len(unique), 12)/100
    return dict(base, m_windows=windows, gt_plan=dict(window_ids=unique,
        window_index={key: i for i, key in enumerate(unique)},
        plan_values=plan, valid=torch.ones(len(unique), dtype=torch.bool)))


def test_joint_collation_keeps_base_auxiliary_and_all_decoder_occurrences(config, monkeypatch):
    items = [joint_item(0, config, monkeypatch, remaster=True), joint_item(1, config, monkeypatch)]
    for i, item in enumerate(items):
        item['m_windows'] = [dict(window, sequence_index=2*j+i)
                             for j, window in enumerate(item['m_windows'])]
    expected = sorted([window for item in items for window in item['m_windows']],
                      key=lambda window: window['sequence_index'])
    with patch('maidata.audio.layout.prepare_encoder_layout',
               wraps=__import__('maidata.audio.layout', fromlist=['prepare_encoder_layout']).prepare_encoder_layout) as layout:
        batch = collate_audio(items, config=config)
    assert layout.call_count == 1
    assert len(batch.window_ids) == len(set(batch.window_ids)) == 6
    assert len(batch.windows) == len(expected) == 9
    assert [window['identity'] for window in batch.windows] == [window['identity'] for window in expected]
    assert supports(batch)['planner_category'] == 6
    assert supports(batch)['empty'] == 18
    assert sum(chunk.labels.counts['main'] for chunk in batch.chunks) == sum(
        window['labels']['counts']['main'] for window in expected)
    assert batch.audio_windows.target_start_beats.tolist() == [0., 3., 0., 3.]
    assert batch.audio_windows.window_to_song.tolist() == [0, 0, 1, 1]
    assert batch.coverage_audio_indices.tolist() == [0, 2]
    assert batch.conditions.audio.values.shape == (4, 18, 0)
    assert batch.draft_inputs.audio_mapping.tolist() == [0, 0, 1]
    assert len(batch.decoder_groups) == 6
    assert batch.conditions.occurrence_mapping.unique().numel() == 6
    for occurrence, window in enumerate(expected):
        unique = batch.conditions.occurrence_mapping[occurrence]
        variant = window['chart_id'].split(':')[0]
        item = items[int(variant)]
        gt = item['gt_plan']
        torch.testing.assert_close(batch.conditions.plan[unique],
            gt['plan_values'][gt['window_index'][window['identity'][0]]])
        audio_index = batch.conditions.audio_mapping[unique]
        assert batch.audio_windows.target_start_beats[audio_index] == window['start']
        assert batch.audio_windows.window_to_song[audio_index] == int(variant)
        assert batch.decoder_groups[unique] == batch.groups[batch.window_ids.index(window['identity'][0])]
    assert all(set(window) == {'identity', 'history_dropped'} for window in batch.windows)


def test_tail_local_audio_is_kept_for_m_but_never_assembled_into_d_bank(config, monkeypatch):
    item = joint_item(0, config, monkeypatch)
    batch = collate_audio([item], config=config)
    geometry = batch.conditions.audio
    values = torch.arange(geometry.valid.numel(), dtype=torch.float32).reshape(*geometry.valid.shape, 1)
    local = ScaleMemory(values, geometry.beats, geometry.valid)
    coverage = batch.coverage_audio_indices
    selected = ScaleMemory(local.values[coverage], local.beats[coverage], local.valid[coverage])
    bank = assemble_full_p0(selected, batch.audio_windows.target_start_beats[coverage],
        batch.audio_windows.window_to_song[coverage], batch.coverages,
        p0_rate=config['fpn_rates'][0], layout=batch.layouts['p0'])
    assert bank.values.shape == (1, config['target_beats']*config['fpn_rates'][0], 1)
    assert bank.beats.max() < 3
    tail_unique = (batch.decoder_starts == 3).nonzero().flatten().item()
    tail_local = batch.conditions.audio_mapping[tail_unique]
    assert tail_local not in coverage.tolist()
    # This exact local tail includes real preceding context. It is not a slice
    # of the concatenated center bank and is not replaced by an all-empty bank.
    assert local.valid[tail_local].any()
    assert local.beats[tail_local].max() > bank.beats.max()


def test_chunk_budget_only_changes_m_computation_and_packed_context_is_shared(config, monkeypatch):
    items = [joint_item(0, config, monkeypatch, remaster=True)]
    together = collate_audio(items, config=dict(config, decoder_train_chunk_token_budget=10000))
    packet = collate_audio(items, config=dict(config, decoder_train_chunk_token_budget=1), packed=True)
    batch = packet.to('cpu')
    assert len(together.chunks) == 1 and len(batch.chunks) == len(batch.windows)
    assert together.window_ids == batch.window_ids
    assert supports(together) == supports(batch)
    assert {key: sum(chunk.labels.counts[key] for chunk in together.chunks)
            for key in together.chunks[0].labels.counts} == {
        key: sum(chunk.labels.counts[key] for chunk in batch.chunks)
        for key in together.chunks[0].labels.counts}
    assert len(packet.context.buffers) <= 5
    assert batch.songs.mel_patches.data_ptr() == packet.songs.mel_patches.data_ptr()
    assert all(hasattr(chunk, 'buffers') and len(chunk.buffers) <= 5 for chunk in batch.chunks)
    assert batch.conditions.decoder_z.valid.equal(batch.layouts['decoder']['valid'])
    assert batch.conditions.decoder_z.beats.equal(batch.layouts['decoder']['beats'])
    assert not batch.conditions.plan.requires_grad


def test_joint_dataset_reads_v5_window_once_for_repeated_occurrence(config, monkeypatch):
    item = joint_item(0, config, monkeypatch)
    rows, cached = [], {}
    for index, window in enumerate(item['m_windows']):
        identity = window['identity'][0]
        rows.append(dict(window_id=identity, chart_id=window['chart_id'],
            start_beat=window['start'], occurrence=window['identity'][1], sequence_index=index))
        cached[identity] = dict(history=window['offset'], tokens=window['tokens'],
            next_types=window['next_types'], labels=window['labels'],
            metadata=dict(logical_lengths=[window['length']-window['offset'], window['offset']]))
    reads, gt_reads = [], []
    windows = SimpleNamespace(read=lambda identity: reads.append(identity) or cached[identity])
    gt = SimpleNamespace(read=lambda variant: gt_reads.append(variant) or item['gt_plan'])
    plan = SimpleNamespace(groups={'0': {}}, training=True,
        decoder_plan=SimpleNamespace(rows=lambda variant, epoch: rows))
    with patch.object(data, 'AudioDataset', return_value={('0', 2): item}):
        dataset = JointDataset(plan, config, windows, gt)
        result = dataset[('0', 2)]
    assert len(reads) == len(set(reads)) == 2
    assert gt_reads == ['0']
    assert len(result['windows']) == 2 and len(result['m_windows']) == 3
    assert [window['identity'] for window in result['m_windows']] == [
        (row['window_id'], row['occurrence']) for row in rows]
    assert all(window['history_dropped'] == data.history_dropped(config, True, 2, row)
               for window, row in zip(result['m_windows'], rows))


def test_joint_plan_preserves_stage_one_groups_and_boundaries(config, tmp_path):
    make_plan(tmp_path, config)
    stage_one = AudioPlan(tmp_path, config)
    decoder = SimpleNamespace(rows=lambda variant, epoch: [])
    c = dict(config, decoder_train_train_plan='accepted-decoder-plan')
    with patch.object(data, 'DecoderPlan', return_value=decoder) as constructor:
        joint = JointPlan(tmp_path, c)
    constructor.assert_called_once_with('accepted-decoder-plan', c, training=True)
    for epoch in (1, 2, 7):
        assert joint.order(epoch) == stage_one.order(epoch)
        assert joint.batches(epoch, c) == stage_one.batches(epoch, c)
        assert joint.rows('0', epoch) == stage_one.rows('0', epoch)
    sampler = AudioBatchSampler(joint, c)
    sampler.epoch = 2
    assert [variant for batch in sampler for variant, epoch in batch] == joint.order(2)


def test_joint_loader_keeps_formal_workers_and_group_sampler(config, monkeypatch):
    plan = SimpleNamespace(manifest={'epoch': 1}, groups={})
    with patch.object(data, 'JointDataset', return_value=object()), \
         patch.object(data, 'DataLoader', side_effect=lambda *args, **kwargs: kwargs):
        loader = data.make_loader(plan, dict(config, pretrain_workers=12))
    assert loader['num_workers'] == 12 and loader['persistent_workers']
    assert isinstance(loader['batch_sampler'], AudioBatchSampler)
    assert loader['collate_fn'].keywords['packed']
    assert loader['worker_init_fn'] is data.worker_init
