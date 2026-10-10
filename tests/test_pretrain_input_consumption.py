"""Accepted inputs reach E/D/R/P without repeated input validation."""
from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_draft_data import record
from test_pretrain_runtime import config, item
from test_pretrain_layout import tensors
from maidata.chart.codec import token_from_dict
from maidata.chart.types import Moment, Note, Route, HoldDuration, SlideDuration
from maidata.draft.chart_data import from_record
from maidata.draft.conditions import ChartCondition
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.preparation.cache import load_grid
from maidata.training.data import AudioDataset, collate_audio
from maidata.training.network import Pretrainer
from maidata.training.pretrain import weighted_loss, make_optimizer, make_scheduler
from maidata.training.runner import update
from maidata.training.checkpoint import data_provenance


@pytest.mark.parametrize('reference_mel', [False, True])
def test_dataset_to_forward_has_no_input_scans(config, tmp_path, reference_mel):
    c = dict(config, reconstruction_query_rows=16, pretrain_audio_cache=str(tmp_path),
             pretrain_chart_cache=str(tmp_path), pretrain_source_root=str(tmp_path), pretrain_mel_root=str(tmp_path))
    reference = item(0, c)
    (tmp_path/'grids').mkdir()
    grid = dict(reference['grid'])
    if reference_mel:
        mel = grid.pop('mel_patches').reshape(-1,c['mel_bins']).T
        np.save(tmp_path/'mel.npy',mel)
        grid.update(mel_source_indices=np.arange(15).reshape(3,5),
                    mel_new_flat_indices=np.empty(0,dtype=np.int64),
                    mel_new_values=np.empty((0,c['mel_bins']),dtype=np.float32))
        (tmp_path/'grids/0.json').write_text(json.dumps(dict(
            mel_reference=dict(feature_path='mel.npy',shape=list(mel.shape)))),encoding='utf-8')
    np.savez(tmp_path/'grids/0.npz', **grid)
    source = tmp_path/'song'
    source.mkdir()
    (source/'maidata.txt').write_text('&lv_5=14.5\n&inote_5={1}1,,E\n', encoding='utf-8')
    raw = record()
    raw['source'] = dict(chart_id='chart', audio_variant='0', song_id='song', difficulty='5',
                         source_path=str(source/'maidata.txt'))
    if reference_mel:
        del raw['metadata']['render_end_beat']
    with gzip.open(tmp_path/'chart.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(raw, f)
    rows = [dict(window_id=f'chart:{b}',chart_id='chart',start_beat=b,start_tick=b*96,end_tick=(b+3)*96)
            for b in (0,3)]
    plan = SimpleNamespace(groups={'0':dict(timing=dict(bpm=[[0,1,120]],first_seconds=0.),
        audio_end_seconds=1.2,chart_ids=['chart'])}, charts={'chart':dict(chart_cache_path='chart.json.gz')},
        rows=lambda variant,epoch: rows)
    chart = from_record(raw)
    planner = MusicPlanTargets(chart)
    expected = dict(reference, conditions={'chart':ChartCondition('0',5,14.5,'14.5')},
        windows=[(w['window_id'],'chart',w['start_beat'],
            reconstruction_target(chart,start_tick=w['start_tick'],target_beats=3),
            planner.window(w['start_tick'],w['end_tick'])) for w in rows])
    expected_batch = collate_audio([expected], config=c)
    model = Pretrainer(c).train()
    checked = deepcopy(model)
    scans = ('maidata.preparation.cache.validate_grid', 'maidata.chart_cache.cache.validate',
             'maidata.chart.grammar.Grammar.accept', 'maidata.draft.types.validate_memory',
             'numpy.isfinite', 'numpy.array_equal', 'torch.isfinite', 'torch.equal', 'torch.allclose')
    with ExitStack() as stack:
        for name in scans:
            stack.enter_context(patch(name, side_effect=AssertionError(f'input scan: {name}')))
        for cls in (Moment,Note,Route,HoldDuration,SlideDuration):
            stack.enter_context(patch.object(cls,'__post_init__',side_effect=AssertionError('token validation')))
        batch = collate_audio([AudioDataset(plan,c)[('0',1)]], config=c)
        terms, output = model(batch)
    expected_terms, expected_output = checked(expected_batch)
    for a,b in zip(tensors(batch),tensors(expected_batch),strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(tensors(output),tensors(expected_output),strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    weighted_loss(terms,c).backward()
    weighted_loss(expected_terms,c).backward()
    for (name,a),(other,b) in zip(model.named_parameters(),checked.named_parameters(),strict=True):
        assert name == other
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_offline_checks_remain_separate_from_training(tmp_path, config):
    grid = item(0,config)['grid']
    grid['mel_patches'][0,0,0] = np.nan
    np.savez(tmp_path/'grid.npz',**grid)
    with pytest.raises(ValueError, match='finite'):
        load_grid(tmp_path/'grid.npz')
    assert np.isnan(load_grid(tmp_path/'grid.npz',validate_input=False)['mel_patches'][0,0,0])
    for raw in record()['tokens']:
        assert token_from_dict(raw,validate_input=False) == token_from_dict(raw)
    bad = dict(type='NOTE',kind='tap',pad='A1',ex=2,break_=0,short_hold=0)
    with pytest.raises(ValueError,match='binary'):
        token_from_dict(bad)
    assert token_from_dict(bad,validate_input=False).ex == 2
    # Fixed query capacity still cannot silently truncate accepted rows.
    target = reconstruction_target(from_record(record()),start_tick=0,target_beats=3)
    with pytest.raises(IndexError):
        collate_reconstruction([target],query_rows=1,target_beats=3,device='cpu',validate_input=False)


def test_input_provenance_does_not_read_or_hash_inputs(config):
    with patch.object(Path,'read_bytes',side_effect=AssertionError('input hash')), \
         patch.object(Path,'read_text',side_effect=AssertionError('input scan')):
        result = data_provenance(dict(config,pretrain_chart_cache='missing-cache'))
    assert result['paths']['pretrain_chart_cache'].endswith('missing-cache')


def test_nonfinite_loss_still_stops_before_optimizer_update(config):
    model = Pretrainer(config)
    batch = collate_audio([item(0,config)],config=config)
    with torch.no_grad():
        model.draft.output.weight.fill_(float('nan'))
    optimizer = make_optimizer(model.parameters(),config)
    scheduler = make_scheduler(optimizer,[1]*8,config)
    with patch.object(optimizer,'step',side_effect=AssertionError('nonfinite update')):
        with pytest.raises(ValueError,match='nonfinite pretraining loss'):
            update(model,[batch],optimizer,scheduler,config)
