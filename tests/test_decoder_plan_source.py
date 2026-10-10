"""Plan-source switching changes only the selected plan12, never Z or sampling."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from maidata.decoder_training import checkpoint, recipe, runner
from maidata.decoder_training.__main__ import main
from maidata.decoder_training.data import collate_audio, DecoderDataset
from maidata.decoder_training.network import DecoderTrainer
from test_decoder_training import config, encoder_config, training_item, distances


def gt_values(item):
    identities = list(dict.fromkeys(w['identity'][0] for w in item['windows']))[::-1]
    values = torch.zeros(len(identities), 12)
    for i, identity in enumerate(identities):
        if identity.endswith(':3'):
            values[i, 8] = 1  # Valid empty tail: single, with eight zero statistics.
        else:
            values[i, :8] = (i+1)/10
            values[i, 8+i%4] = 1
    return dict(plan_values=values, valid=torch.ones(len(values), dtype=torch.bool),
                window_index={key: i for i, key in enumerate(identities)})


def comparison_items(config, monkeypatch):
    items = [training_item(0, config, monkeypatch, remaster=True), training_item(1, config, monkeypatch)]
    for i, item in enumerate(items):
        item['gt_plan'] = gt_values(item)
        item['windows'] = [dict(w, sequence_index=2*j+i) for j, w in enumerate(item['windows'])]
    return items


def test_gt_and_predicted_select_same_window_identity_and_leave_z_audio_labels_unchanged(config, monkeypatch):
    items = comparison_items(config, monkeypatch)
    predicted = collate_audio(items, config=config, packed=True).to('cpu')
    gt = collate_audio(items, config=dict(config, decoder_train_plan_source='gt'), packed=True).to('cpu')
    expected = sorted([w for item in items for w in item['windows']], key=lambda w: w['sequence_index'])
    for batch in (predicted, gt):
        assert [w['identity'] for w in batch.windows] == [w['identity'] for w in expected]
    for a, b in ((predicted.conditions.decoder_z, gt.conditions.decoder_z),
                 (predicted.conditions.audio, gt.conditions.audio)):
        for name in ('values', 'beats', 'valid'):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    assert predicted.budget_cost == gt.budget_cost
    assert torch.equal(predicted.conditions.occurrence_mapping, gt.conditions.occurrence_mapping)
    assert torch.equal(predicted.conditions.audio_mapping, gt.conditions.audio_mapping)
    for a, b in zip(predicted.chunks, gt.chunks):
        assert a.labels.counts == b.labels.counts and a.cost == b.cost
        assert torch.equal(a.tokens.types, b.tokens.types)
    for index, window in enumerate(expected):
        unique = gt.conditions.occurrence_mapping[index]
        item = items[gt.online_audio.windows.window_to_song[gt.conditions.audio_mapping[unique]]]
        cached = item['gt_plan']
        torch.testing.assert_close(gt.conditions.plan[unique],
            cached['plan_values'][cached['window_index'][window['identity'][0]]], rtol=0, atol=0)
        dp = item['dp']
        torch.testing.assert_close(predicted.conditions.plan[unique],
            dp['planner']['plan_values'][dp['window_index'][window['identity'][0]]], rtol=0, atol=0)
        assert gt.conditions.plan_valid[unique]
    assert not gt.conditions.plan.requires_grad
    assert not torch.equal(predicted.conditions.plan, gt.conditions.plan)


def test_gt_update_trains_encoder_and_decoder_and_records_source(config, monkeypatch):
    c = dict(config, decoder_train_plan_source='gt')
    batch = collate_audio(comparison_items(c, monkeypatch), config=c, packed=True)
    model = DecoderTrainer(c)
    opt = recipe.optimizer(model.parameters(), c)
    sched = recipe.scheduler(opt, [1]*6, c)
    metrics, audit = runner.update(model, [batch], opt, sched, c, distances(), audit=True)
    assert metrics.counts['main'] > 0
    for name in ('encoder', 'decoder'):
        assert audit['gradients'][name] > 0 and audit['changed_parameter_tensors'][name] > 0
    payload = checkpoint.capture(model, opt, sched, epoch=1, step=1, config=c, binding={},
        source={}, producer={}, steps=[1]*6, loaders=[], best=None, metrics={})
    assert payload['recipe']['decoder_train_plan_source'] == 'gt'


def test_predicted_never_reads_gt_and_gt_never_falls_back(config, monkeypatch):
    items = comparison_items(config, monkeypatch)
    for item in items:
        del item['gt_plan']
    collate_audio(items, config=config)
    with pytest.raises(KeyError, match='gt_plan'):
        collate_audio(items, config=dict(config, decoder_train_plan_source='gt'))


def test_dataset_reads_gt_once_per_variant_without_replaying_charts(config, tmp_path, monkeypatch):
    item = training_item(0, config, monkeypatch, remaster=True, training=False)
    gt = gt_values(item)
    torch.save(dict(gt, window_ids=list(gt['window_index'])), tmp_path/'0.pt')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(groups={'0': {'path': '0.pt'}})))
    windows = {w['identity'][0]: dict(history=w['offset'], tokens=w['tokens'],
        next_types=w['next_types'], labels=w['labels']) for w in item['windows']}
    rows = [dict(window_id=w['identity'][0], occurrence=w['identity'][1], chart_id=w['chart_id'],
                 start_beat=w['start'], sequence_index=i) for i, w in enumerate(item['windows'])]
    selected = (0, 1, 5)
    plan = SimpleNamespace(training=True, groups={'0': dict(timing=dict(bpm=[[0,1,120]], first_seconds=0.),
        audio_end_seconds=1.2)}, rows=lambda variant, epoch, selection: [rows[i] for i in selection])
    c = dict(config, decoder_train_plan_source='gt', decoder_train_gt_plan_cache=str(tmp_path))
    cache = SimpleNamespace(read=lambda variant: item['dp'])
    window_cache = SimpleNamespace(read=lambda identity: windows[identity])
    dataset = DecoderDataset(plan, c, {}, cache, window_cache)
    with patch.object(dataset.gt_plan, 'read', wraps=dataset.gt_plan.read) as read, \
         patch('maidata.decoder_training.data.load_grid', return_value=item['grid']), \
         patch('maidata.chart_cache.cache.load_chart', side_effect=AssertionError('no online chart replay')), \
         patch('maidata.draft.plan_targets.MusicPlanTargets', side_effect=AssertionError('no online GT statistics')):
        value = dataset[('0', 2, selected)]
        batch = collate_audio([value], config=c)
    assert read.call_count == 1 and value['gt_plan']['window_index'] == gt['window_index']
    assert [w['identity'] for w in batch.windows] == [(rows[i]['window_id'], rows[i]['occurrence']) for i in selected]
    for i, row_index in enumerate(selected):
        unique = batch.conditions.occurrence_mapping[i]
        torch.testing.assert_close(batch.conditions.plan[unique],
            gt['plan_values'][gt['window_index'][rows[row_index]['window_id']]], rtol=0, atol=0)


def test_recipe_requires_gt_cache_only_when_selected(config, tmp_path):
    c = dict(config, **{key: str(tmp_path/key) for key in recipe.PATH_KEYS})
    c['decoder_train_gt_plan_cache'] = None
    recipe.validate(c)
    with pytest.raises(ValueError, match='decoder_train_gt_plan_cache'):
        recipe.validate(dict(c, decoder_train_plan_source='gt'))
    recipe.validate(dict(c, decoder_train_plan_source='gt', decoder_train_gt_plan_cache='labels'))
    with pytest.raises(ValueError, match='predicted or gt'):
        recipe.validate(dict(c, decoder_train_plan_source='other'))


def test_cli_can_override_both_sources_outputs_and_gt_cache(tmp_path, capsys):
    path = tmp_path/'base.json'
    path.write_text(json.dumps(dict(decoder_train_plan_source='gt', decoder_train_output='base')))
    assert main(['--config', str(path), '--plan-source', 'predicted', '--output', 'prediction', '--show-config']) == 0
    assert json.loads(capsys.readouterr().out)['decoder_train_plan_source'] == 'predicted'
    assert main(['--config', str(path), '--plan-source', 'gt', '--gt-plan-cache', 'labels',
                 '--output', 'truth', '--show-config']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['decoder_train_plan_source'] == 'gt'
    assert config['decoder_train_gt_plan_cache'] == 'labels' and config['decoder_train_output'] == 'truth'
