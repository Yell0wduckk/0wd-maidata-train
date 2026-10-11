"""Stage-II Encoder -> public P0 -> frozen-E Drafter, and joint-mode parity."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_draft_data import record
from test_pretrain_runtime import config
from test_decoder_network import config as decoder_config
from maidata.model import build_model, build_decoder
from maidata.draft.windows import song_coverage, assemble_full_p0
from maidata.time import TimingMap
from maidata.training import checkpoint, runner
from maidata.training.data import AudioPlan, AudioDataset, collate_audio
from maidata.training.frozen_p0 import FrozenP0, resolve_audio_config
from maidata.training.network import Pretrainer
from maidata.training.p0_cache import export
from maidata.training.pretrain import make_optimizer, make_scheduler, autocast, weighted_loss


def inputs(tmp_path, config, precision='fp32'):
    c = dict(config, pretrain_precision=precision, pretrain_workers=0,
        pretrain_audio_cache=str(tmp_path), pretrain_chart_cache=str(tmp_path),
        pretrain_source_root=str(tmp_path), pretrain_mel_root=None,
        pretrain_train_plan=str(tmp_path/'train'), pretrain_validation_plan=str(tmp_path/'validation'))
    (tmp_path/'grids').mkdir()
    groups, charts, rows = [], [], []
    for index in range(3):
        variant, chart_id = str(index), f'chart{index}'
        first, duration = (-1, 2.1) if index == 1 else (0, 1.2)
        count, rate = (6 if index == 1 else 3), c['samples_per_beat']
        k = (np.arange(count*rate) + first*rate).reshape(count, rate)
        seconds = k/rate*.5
        valid = (seconds >= 0) & (seconds < duration)
        rng = np.random.default_rng(index)
        mel = rng.normal(size=(count, rate, c['mel_bins'])).astype(np.float32)
        aux = rng.normal(size=(count, rate, 32)).astype(np.float32)
        mel[~valid], aux[~valid] = 0, 0
        np.savez(tmp_path/f'grids/{variant}.npz', mel_patches=mel, sample_valid=valid,
            beat_centers=np.arange(first,first+count,dtype=np.float32)+.5,
            aux32_grid=aux, aux32_centers=aux.mean(1), sample_times_sec=seconds,
            sample_indices=np.rint(seconds*24000).astype(np.int64), grid_k=k, first_beat=np.array(first))
        timing = dict(first_seconds=0., bpm=[[0,1,120]])
        coverage = song_coverage(TimingMap.from_audio_timing(timing), duration, first_beat=first, target_beats=3)
        starts = coverage.starts(include_tail=True)
        groups.append(dict(audio_variant=variant, audio_asset='asset'+variant, timing=timing,
            audio_end_seconds=duration, first_beat=first, actual_audio_end_beat=duration*2,
            encoder_starts_beats=list(coverage.starts(include_tail=False)),
            draft_end_beat=first+coverage.draft_beats, occurrences=len(starts), chart_ids=[chart_id]))
        chart = record()
        chart['source'] = dict(chart_id=chart_id, audio_variant=variant, song_id='song'+variant,
                              difficulty='5', source_path='unused')
        (tmp_path/('song'+variant)).mkdir()
        (tmp_path/('song'+variant)/'maidata.txt').write_text('&lv_5=14.5\n&inote_5={1}1,,E\n')
        with gzip.open(tmp_path/f'{chart_id}.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(chart, f)
        charts.append(dict(chart_id=chart_id, chart_cache_path=f'{chart_id}.json.gz'))
        rows.extend(dict(audio_variant=variant, chart_id=chart_id, window_id=f'{chart_id}:{start}',
                         start_beat=start, start_tick=start*96, end_tick=(start+3)*96) for start in starts)
    for side, chosen in (('train', groups[:2]), ('validation', groups[2:])):
        root = tmp_path/side
        root.mkdir()
        (root/'plan.json').write_text(json.dumps(dict(seed=c['pretrain_seed'], epoch=1,
            config=dict(target_beats=3), audio_groups=chosen, charts=charts, files=dict(windows='windows.jsonl'))))
        (root/'windows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows
            if r['audio_variant'] in {g['audio_variant'] for g in chosen}))
    # E_M's width differs from the default pretrainer's width. Only Encoder keys
    # should be adopted from this checkpoint; D/R/P are freshly initialized.
    source = dict(c, d_model=24, decoder_z_cross_attention=False)
    source.update({k:v for k,v in decoder_config().items() if k.startswith('decoder_')})
    source['decoder_z_cross_attention'] = False
    model = torch.nn.ModuleDict(dict(encoder=build_model(source), decoder=build_decoder(source)))
    path = tmp_path/'epoch-045.ckpt'
    torch.save(dict(schema='maidata.decoder-training.v4', architecture='maidata.decoder-causal-cont-tick384.v2',
        completed_epochs=45, step=100, source={'commit':'stage-ii'}, recipe=source, model_state=model.state_dict()), path)
    return c, path


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_export_import_matches_live_frozen_encoder_and_updates_only_drp(config, tmp_path, precision):
    c, source = inputs(tmp_path, config, precision)
    root = tmp_path/'p0'
    with patch('maidata.preparation.cache.validate_grid', side_effect=AssertionError('old input scan')), \
         patch('maidata.model.build_decoder', side_effect=AssertionError('must not construct M')):
        report = export(c, source, root)
    assert report['complete'] and report['producer']['completed_epochs'] == 45
    assert report['decoder_forward_calls'] == report['draft_forward_calls'] == 0
    assert report['first_bank_roundtrip_exact'] and len(report['groups']) == 3
    cached = resolve_audio_config(dict(c, pretrain_audio_mode='cached', pretrain_p0_cache=str(root)))
    assert cached['d_model'] == 24 and cached['draft_dim'] == c['draft_dim']
    joint = dict(cached, pretrain_audio_mode='joint', pretrain_p0_cache=None)
    plan = AudioPlan(cached['pretrain_train_plan'], cached)
    for epoch in (1,2,50):
        assert plan.batches(epoch, cached) == plan.batches(epoch, joint)
    with patch('maidata.training.data.load_grid', side_effect=AssertionError('cache training read Mel')):
        data = AudioDataset(plan, cached)
        rows = [data[(v,1)] for v in ('0','1')]
        batch = collate_audio(rows, config=cached, packed=True).to('cpu')
    assert batch.songs is batch.audio_windows is None
    assert batch.p0.values.shape == (2,18,24)
    assert not batch.p0.valid[0,9:].any() and batch.p0.values[0,9:].count_nonzero() == 0
    direct_rows = [AudioDataset(plan,joint)[(v,1)] for v in ('0','1')]
    live = collate_audio(direct_rows, config=joint)
    model = Pretrainer(cached)
    model.train()
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    # Compare one song at a time, exactly matching the offline export batch shape.
    for i, row in enumerate(direct_rows):
        b = collate_audio([row],config=joint)
        with torch.no_grad(), autocast('cpu',joint):
            encoded = model.encoder(b.songs,b.audio_windows)
            bank = assemble_full_p0(encoded.P0,b.audio_windows.target_start_beats,b.audio_windows.window_to_song,
                b.coverages,p0_rate=joint['fpn_rates'][0],layout=b.layouts['p0'])
        n = bank.values.shape[1]
        for key in ('values','beats','valid'):
            torch.testing.assert_close(getattr(batch.p0,key)[i,:n],getattr(bank,key)[0],rtol=0,atol=0)
    # An online frozen-E path with this identical bank gives the same D/R/P
    # outputs and gradients; cached training has no E invocation at all.
    online = Pretrainer(joint).eval()
    online.load_state_dict(model.state_dict(),strict=True)
    model.eval()
    with patch.object(online.encoder,'forward') as encode, autocast('cpu',cached):
        # Use the already assembled bank to avoid batch-size numerical effects.
        from types import SimpleNamespace
        encode.return_value = SimpleNamespace(P0=None)
        with patch('maidata.training.network.assemble_full_p0',return_value=batch.p0):
            expected, _ = online(live)
        actual, _ = model(batch)
    assert weighted_loss(actual,cached).item() == weighted_loss(expected,joint).item()
    weighted_loss(actual,cached).backward()
    weighted_loss(expected,joint).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,dict(online.named_parameters())[name].grad,rtol=0,atol=0)
    optimizer = make_optimizer(model.parameters(),cached)
    scheduler = make_scheduler(optimizer,[1]*8,cached)
    model.train()
    with patch.object(model.encoder,'forward',side_effect=AssertionError('cached training ran E')):
        _, audit = runner.update(model,[batch],optimizer,scheduler,cached,audit=True)
    assert audit['gradients']['encoder'] == audit['changed_parameter_tensors']['encoder'] == 0
    assert all(audit['gradients'][k] > 0 and audit['changed_parameter_tensors'][k] > 0
               for k in ('draft','reconstruction','planner'))
    frozen_ids = {id(p) for p in model.encoder.parameters()}
    assert all(id(p) not in frozen_ids for g in optimizer.param_groups for p in g['params'])


def test_cached_runner_resume_and_portable_teacher_checkpoint(config,tmp_path,monkeypatch):
    c, source = inputs(tmp_path, config)
    export(c,source,tmp_path/'p0')
    c = dict(c,pretrain_audio_mode='cached',pretrain_p0_cache=str(tmp_path/'p0'),
             pretrain_stop_after_epochs=3,pretrain_scheduler_epochs=4)
    full = dict(c,pretrain_output=str(tmp_path/'full'))
    interrupted = dict(c,pretrain_output=str(tmp_path/'resumed'))
    runner.run(full)
    real_update, calls = runner.update, 0
    def fail(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('interrupted')
        return real_update(*args,**kwargs)
    monkeypatch.setattr(runner,'update',fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        runner.run(interrupted)
    path = tmp_path/'resumed/checkpoints/last.ckpt'
    assert torch.load(path,weights_only=False)['completed_epochs'] == 1
    monkeypatch.setattr(runner,'update',real_update)
    with patch('maidata.training.data.load_grid',side_effect=AssertionError('resume read Mel')):
        runner.run(interrupted,resume=path)
    a = torch.load(tmp_path/'full/checkpoints/last.ckpt',weights_only=False)
    b = torch.load(path,weights_only=False)
    assert a['step'] == b['step'] == 3 and a['scheduler_state'] == b['scheduler_state']
    assert a['audio_producer'] == b['audio_producer'] and a['recipe']['pretrain_audio_mode'] == 'cached'
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)
    for name in ('optimizer_state','rng','loader_rng'):
        from test_pretrain_layout import tensors
        for x,y in zip(tensors(a[name]),tensors(b[name]),strict=True):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(ValueError,match='producer changed'):
        bad = dict(b,audio_producer={'different':'encoder'})
        model = Pretrainer(resolve_audio_config(c))
        checkpoint.restore(bad,model,None,None,config=resolve_audio_config(c),binding={},source={},steps=[1]*4,loaders=[])
    # The finished checkpoint embeds E, so inference/export does not depend on
    # the training P0 cache, original M checkpoint, or their original locations.
    from maidata.inference.checkpoint import load_models
    with patch('maidata.training.frozen_p0.FrozenP0',side_effect=AssertionError('teacher reread P0')):
        teacher = load_models(path,source,device='cpu',z_cross_attention=False,plan_source='predicted')
    assert teacher.teacher_config['pretrain_audio_mode'] == 'cached'
    for key,value in teacher.encoder.state_dict().items():
        torch.testing.assert_close(value,b['model_state']['encoder.'+key],rtol=0,atol=0)


def test_audio_mode_is_explicit_and_cannot_silently_ignore_cache(config,tmp_path):
    with pytest.raises(ValueError,match='requires pretrain_p0_cache'):
        resolve_audio_config(dict(config,pretrain_audio_mode='cached'))
    with pytest.raises(ValueError,match='cannot consume'):
        resolve_audio_config(dict(config,pretrain_p0_cache='unused'))
    with pytest.raises(ValueError,match='must be joint or cached'):
        resolve_audio_config(dict(config,pretrain_audio_mode='frozen-online'))
    root = tmp_path/'incomplete'
    root.mkdir()
    (root/'manifest.json').write_text('{"schema":"maidata.frozen-p0-bank.v1","complete":false}')
    with pytest.raises(ValueError,match='complete P0'):
        FrozenP0(root)


def test_cli_exposes_cached_and_joint_modes_without_reading_inputs(tmp_path,capsys):
    from maidata.training.__main__ import main
    assert main(['--audio-mode','cached','--p0-cache',str(tmp_path/'future-cache'),'--show-config']) == 0
    cached = json.loads(capsys.readouterr().out)
    assert cached['pretrain_audio_mode'] == 'cached'
    assert cached['pretrain_p0_cache'] == str(tmp_path/'future-cache')
    assert cached['pretrain_workers'] == 12
    assert main(['--audio-mode','joint','--show-config']) == 0
    joint = json.loads(capsys.readouterr().out)
    assert joint['pretrain_audio_mode'] == 'joint' and joint['pretrain_p0_cache'] is None
