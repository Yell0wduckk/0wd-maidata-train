"""Joint inference cache acceptance with real small E/D/P/M and atomic files."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_inference import config, producer_config
from test_inference_joint import joint_config, joint_checkpoint, arguments
from test_chart_physical import geometry
from test_pretrain_runtime import item
from test_inference_materialize import write_audio
from maidata.inference.__main__ import main
from maidata.inference.preencode import main as preencode
from maidata.inference.predraft import main as predraft
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.cache import (JointCache, audio_identity, atomic_save,
    draft_identity, file_identity, identity_key, load_audio)
from maidata.inference.conditioning import prepare_joint_conditions
from maidata.inference.sampling import Sampler
from maidata.time import TimingMap
from maidata.model import build_model, build_draft, build_planner, build_decoder


def inputs(c, tmp_path, *, notes=False):
    checkpoint, _, payload = joint_checkpoint(c, tmp_path)
    if notes:
        state = payload['model_state']
        state['decoder.type_head.bias'][0] = 80
        state['decoder.type_head.bias'][1] = 100
        state['decoder.heads.kind.weight'].zero_()
        state['decoder.heads.kind.bias'].fill_(-100)
        state['decoder.heads.kind.bias'][0] = 100
        torch.save(payload, checkpoint)
    write_audio(tmp_path/'audio.wav')
    grid = item(0, c)['grid']
    np.savez(tmp_path/'grid.npz', **grid)
    return checkpoint, grid


def audio_args(checkpoint, tmp_path, precision='fp32', timing=None):
    return ['--joint-checkpoint', str(checkpoint), '--audio', str(tmp_path/'audio.wav'),
        '--grid', str(tmp_path/'grid.npz'), '--device', 'cpu', '--precision', precision,
        '--cache-dir', str(tmp_path/'cache'), '--no-progress',
        *(['--timing', str(timing)] if timing else ['--bpm', '120'])]


def spies(stack, *, forbid=()):
    result = {}
    factories = dict(encoder=('build_model', build_model), draft=('build_draft', build_draft),
                     planner=('build_planner', build_planner), decoder=('build_decoder', build_decoder))
    for name, (function, factory) in factories.items():
        result[name] = stack.enter_context(patch('maidata.inference.checkpoint.'+function,
            **(dict(side_effect=AssertionError(name+' must not be constructed')) if name in forbid
               else dict(wraps=factory))))
    return result


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
@pytest.mark.parametrize('temperature', [0, .7])
def test_online_cold_hot_exact_and_only_required_modules(joint_config, tmp_path, geometry, precision, temperature):
    checkpoint, _ = inputs(joint_config, tmp_path, notes=True)
    args = [*arguments(checkpoint, tmp_path, precision), '--grid', str(tmp_path/'grid.npz'),
            '--temperature', str(temperature), '--max-tokens-per-moment', '4', '--trace']
    reports, charts, samplers = [], [], []
    def sampler(**kwargs):
        value = Sampler(**kwargs)
        samplers.append(value)
        return value
    with patch('maidata.physical.Geometry.from_resources', return_value=geometry), \
         patch('maidata.inference.generation.Sampler', side_effect=sampler):
        for name in ('online', 'cold', 'hot', 'level', 'seed'):
            extra = [] if name == 'online' else ['--cache-dir', str(tmp_path/'cache')]
            if name == 'level':
                extra += ['--level', '13.5']
            if name == 'seed':
                extra += ['--seed', '42', '--temperature', '.9']
            with ExitStack() as stack:
                forbid = ('encoder', 'draft', 'planner') if name in ('hot', 'seed') else ()
                calls = spies(stack, forbid=forbid)
                if name in ('hot', 'level', 'seed'):
                    stack.enter_context(patch('maidata.preparation.cache.load_grid', side_effect=AssertionError('grid read')))
                    stack.enter_context(patch('soundfile.info', side_effect=AssertionError('audio read')))
                main([*args, *extra, '--output', str(tmp_path/name)])
            if name in ('online', 'cold'):
                assert all(call.call_count == 1 for call in calls.values())
            if name == 'level':
                assert calls['encoder'].call_count == 0
                assert calls['draft'].call_count == calls['planner'].call_count == 1
            reports.append(json.loads((tmp_path/name/'generation.json').read_text('utf-8')))
            charts.append((tmp_path/name/'maidata.txt').read_bytes())
    assert charts[0] == charts[1] == charts[2]
    assert reports[0]['moments'] > 0
    assert reports[0]['windows'] == reports[1]['windows'] == reports[2]['windows']
    assert torch.equal(samplers[0].generator.get_state(), samplers[1].generator.get_state())
    assert torch.equal(samplers[0].generator.get_state(), samplers[2].generator.get_state())
    assert reports[1]['cache']['audio']['hit'] is False and reports[1]['cache']['draft']['hit'] is False
    assert reports[2]['cache']['audio']['hit'] and reports[2]['cache']['draft']['hit']
    assert reports[3]['cache']['audio']['hit'] and not reports[3]['cache']['draft']['hit']
    assert reports[4]['cache']['audio']['hit'] and reports[4]['cache']['draft']['hit']
    # --force only removes generation products, retaining reusable caches.
    identities = {p: file_identity(p) for p in (tmp_path/'cache').rglob('*.pt')}
    with patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        main([*args, '--cache-dir', str(tmp_path/'cache'), '--output', str(tmp_path/'hot'), '--force'])
    assert identities == {p: file_identity(p) for p in identities}


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
@pytest.mark.parametrize('first', [.02, -1.25])
def test_independent_stages_and_cached_tensor_equality(joint_config, tmp_path, capsys, precision, first):
    checkpoint_path, grid = inputs(joint_config, tmp_path)
    timing_data = dict(first_seconds=first, bpm=[[0, 1, 120], [1, 1, 180]])
    timing_path = tmp_path/'timing.json'
    timing_path.write_text(json.dumps(timing_data), 'utf-8')
    with ExitStack() as stack:
        calls = spies(stack, forbid=('draft', 'planner', 'decoder'))
        preencode(audio_args(checkpoint_path, tmp_path, precision, timing_path))
        assert calls['encoder'].call_count == 1
    result = json.loads(capsys.readouterr().out)
    audio_path = Path(result['audio_cache']['path'])
    assert result['windows'] == 2 and not result['audio_cache']['hit']
    with ExitStack() as stack:
        spies(stack, forbid=('encoder', 'draft', 'planner', 'decoder'))
        stack.enter_context(patch('maidata.preparation.cache.load_grid', side_effect=AssertionError('grid read')))
        preencode(audio_args(checkpoint_path, tmp_path, precision, timing_path))
    assert json.loads(capsys.readouterr().out)['audio_cache']['hit']
    saved_audio = (tmp_path/'audio.wav').read_bytes()
    saved_grid = (tmp_path/'grid.npz').read_bytes()
    (tmp_path/'audio.wav').unlink()
    (tmp_path/'grid.npz').unlink()
    args = ['--joint-checkpoint', str(checkpoint_path), '--audio-cache', str(audio_path),
        '--difficulty', '5', '--level', '14.5', '--device', 'cpu', '--precision', precision,
        '--cache-dir', str(tmp_path/'cache'), '--no-progress']
    with ExitStack() as stack:
        calls = spies(stack, forbid=('encoder', 'decoder'))
        predraft(args)
        assert calls['draft'].call_count == calls['planner'].call_count == 1
    draft_path = Path(json.loads(capsys.readouterr().out)['draft_cache']['path'])
    with ExitStack() as stack:
        spies(stack, forbid=('encoder', 'draft', 'planner', 'decoder'))
        predraft(args)
    assert json.loads(capsys.readouterr().out)['draft_cache']['hit']
    (tmp_path/'audio.wav').write_bytes(saved_audio)
    (tmp_path/'grid.npz').write_bytes(saved_grid)
    checkpoint = JointCheckpoint(checkpoint_path)
    online = checkpoint.models(device='cpu')
    encoded = load_audio(audio_path, online.config, device='cpu')
    cache = JointCache(tmp_path/'cache', checkpoint, device='cpu', precision=precision)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        expected = prepare_joint_conditions(online.encoder, online.draft, online.planner, online.config,
            grid, TimingMap.from_audio_timing(timing_data), encoded.duration,
            difficulty=5, level=14.5, device='cpu')
        actual = cache.draft(encoded, difficulty=5, level=14.5)
    for kind in ('audio', 'z'):
        for field in ('values', 'beats', 'valid'):
            left, right = getattr(getattr(expected, kind), field), getattr(getattr(actual, kind), field)
            assert left.dtype == right.dtype and torch.equal(left, right)
    assert torch.equal(expected.plans, actual.plans) and torch.equal(expected.plan_valid, actual.plan_valid)
    assert encoded.local.audio.valid[0, :4*joint_config['fpn_rates'][0]].sum() == 0
    assert draft_path.exists() and not (tmp_path/'cache/maidata.txt').exists()


def test_cache_identity_invalidation_and_independent_condition_keys(joint_config, tmp_path):
    checkpoint_path, _ = inputs(joint_config, tmp_path)
    checkpoint = JointCheckpoint(checkpoint_path)
    kwargs = dict(grid=tmp_path/'grid.npz', device='cpu', precision='fp32', encoder_window_batch=8)
    base = audio_identity(checkpoint, tmp_path/'audio.wav', TimingMap([(0, 120)]), **kwargs)
    for name, value in [('precision', 'bf16'), ('encoder_window_batch', 1), ('device', 'meta')]:
        assert identity_key(audio_identity(checkpoint, tmp_path/'audio.wav', TimingMap([(0, 120)]),
            **dict(kwargs, **{name: value}))) != identity_key(base)
    for timing in (TimingMap([(0, 120)], first_seconds=.02), TimingMap([(0, 120), (1, 180)])):
        assert identity_key(audio_identity(checkpoint, tmp_path/'audio.wav', timing, **kwargs)) != identity_key(base)
    for name in ('audio.wav', 'grid.npz', 'joint.ckpt'):
        file = tmp_path/name
        stat = file.stat()
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
        changed_checkpoint = JointCheckpoint(checkpoint_path)
        assert identity_key(audio_identity(changed_checkpoint, tmp_path/'audio.wav', TimingMap([(0, 120)]),
                                          **kwargs)) != identity_key(base)
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    checkpoint.config['fpn_p0_normalization'] = not checkpoint.config['fpn_p0_normalization']
    assert identity_key(audio_identity(checkpoint, tmp_path/'audio.wav', TimingMap([(0, 120)]), **kwargs)) != identity_key(base)
    standard = draft_identity(checkpoint, identity_key(base), difficulty=5, level=14.5, device='cpu', precision='fp32')
    for difficulty, level in ((4, 14.5), (5, 14.)):
        assert identity_key(draft_identity(checkpoint, identity_key(base), difficulty=difficulty, level=level,
                                         device='cpu', precision='fp32')) != identity_key(standard)


@pytest.mark.parametrize('source', ['gt', 'none'])
def test_z_off_gt_none_skip_dp_on_cold_and_hot(joint_config, tmp_path, geometry, source):
    checkpoint, _ = inputs(joint_config, tmp_path)
    plan = tmp_path/'gt.json'
    plan.write_text(json.dumps(dict(windows=[dict(start_beat=s, plan12=[1.]*12) for s in (0, 16)])), 'utf-8')
    for name in ('online', 'cold', 'hot'):
        with ExitStack() as stack:
            calls = spies(stack, forbid=('draft', 'planner', *(('encoder',) if name == 'hot' else ())))
            stack.enter_context(patch('maidata.physical.Geometry.from_resources', return_value=geometry))
            main([*arguments(checkpoint, tmp_path), '--grid', str(tmp_path/'grid.npz'),
                '--no-z-cross-attention', '--plan-source', source, '--gt-plan', str(plan),
                '--output', str(tmp_path/name), *([] if name == 'online' else ['--cache-dir', str(tmp_path/'cache')])])
        assert calls['decoder'].call_count == 1
        report = json.loads((tmp_path/name/'generation.json').read_text('utf-8'))
        assert all(row['plan_valid'] == (source == 'gt') for row in report['windows'])
        if name != 'online':
            assert report['cache']['draft'] is None
    assert (tmp_path/'online/maidata.txt').read_bytes() == (tmp_path/'hot/maidata.txt').read_bytes()


@pytest.mark.parametrize('mode', ['identity', 'shape', 'truncated', 'incomplete'])
def test_corrupt_cache_is_reported_and_not_overwritten(joint_config, tmp_path, capsys, mode):
    checkpoint, _ = inputs(joint_config, tmp_path)
    args = audio_args(checkpoint, tmp_path)
    preencode(args)
    path = Path(json.loads(capsys.readouterr().out)['audio_cache']['path'])
    payload = torch.load(path, weights_only=True)
    if mode == 'identity':
        payload['identity']['encoder']['d_model'] += 1
    elif mode == 'shape':
        payload['local']['values'] = payload['local']['values'][:, :1]
    elif mode == 'incomplete':
        payload['complete'] = False
    if mode == 'truncated':
        path.write_bytes(b'partial')
    else:
        torch.save(payload, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='invalid inference cache'):
        preencode(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize('failures', [0, 2, 101])
def test_atomic_save_retry_and_failure_leave_complete_old_file(tmp_path, failures):
    path = tmp_path/'cache.pt'
    torch.save(dict(old=True), path)
    original_replace = os.replace
    attempts = []
    def replace(source, destination):
        assert torch.load(path, weights_only=True) == dict(old=True)
        assert torch.load(source, weights_only=True) == dict(new=True)
        attempts.append(source)
        if len(attempts) <= failures:
            raise PermissionError('occupied')
        return original_replace(source, destination)
    with patch('maidata.inference.cache.os.replace', side_effect=replace), patch('maidata.inference.cache.time.sleep'):
        if failures == 101:
            with pytest.raises(PermissionError):
                atomic_save(path, dict(new=True))
        else:
            atomic_save(path, dict(new=True))
    assert len(attempts) == min(failures+1, 101)
    assert torch.load(path, weights_only=True) == (dict(old=True) if failures == 101 else dict(new=True))
    assert list(tmp_path.glob('*.tmp')) == []


def test_interrupted_save_does_not_publish_or_leave_temporary(tmp_path):
    with patch('maidata.inference.cache.torch.save', side_effect=RuntimeError('interrupted')):
        with pytest.raises(RuntimeError, match='interrupted'):
            atomic_save(tmp_path/'cache.pt', {})
    assert list(tmp_path.iterdir()) == []


def test_cli_legacy_cache_rejected_before_runtime():
    with pytest.raises(SystemExit):
        main(['--decoder-checkpoint', 'legacy.ckpt', '--cache-dir', 'cache', '--audio', 'audio.wav',
              '--bpm', '120', '--grid', 'grid.npz', '--geometry', 'geometry', '--slide-data', 'slides',
              '--difficulty', '5', '--level', '14.5', '--output', 'output'])


def test_failed_draft_keeps_audio_cache_and_can_resume(joint_config, tmp_path, geometry):
    checkpoint, _ = inputs(joint_config, tmp_path)
    args = [*arguments(checkpoint, tmp_path), '--grid', str(tmp_path/'grid.npz'),
            '--cache-dir', str(tmp_path/'cache')]
    with patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        with patch('maidata.inference.cache.decode_joint_draft', side_effect=RuntimeError('draft interrupted')):
            with pytest.raises(RuntimeError, match='draft interrupted'):
                main(args)
        report = json.loads((tmp_path/'result/generation-error.json').read_text('utf-8'))
        assert report['complete'] is False and Path(report['cache']['audio']['path']).is_file()
        assert not (tmp_path/'result/generation.json').exists()
        assert list((tmp_path/'cache').rglob('*.tmp')) == []
        assert list((tmp_path/'cache').rglob('draft/*.pt')) == []
        with patch('maidata.inference.checkpoint.build_model', side_effect=AssertionError('E must be reused')):
            main([*args, '--force'])
    assert json.loads((tmp_path/'result/generation.json').read_text('utf-8'))['cache']['audio']['hit']


def test_draft_cache_corruption_and_checkpoint_binding(joint_config, tmp_path, capsys):
    checkpoint, _ = inputs(joint_config, tmp_path)
    preencode(audio_args(checkpoint, tmp_path))
    audio_path = json.loads(capsys.readouterr().out)['audio_cache']['path']
    args = ['--joint-checkpoint', str(checkpoint), '--audio-cache', audio_path, '--difficulty', '5',
        '--level', '14.5', '--device', 'cpu', '--precision', 'fp32', '--cache-dir', str(tmp_path/'cache'), '--no-progress']
    predraft(args)
    draft_path = Path(json.loads(capsys.readouterr().out)['draft_cache']['path'])
    payload = torch.load(draft_path, weights_only=True)
    payload['plans'] = torch.zeros((1, 12))
    torch.save(payload, draft_path)
    original = draft_path.read_bytes()
    with pytest.raises(ValueError, match='invalid inference cache'):
        predraft(args)
    assert draft_path.read_bytes() == original
    with pytest.raises(ValueError, match='precision differs'):
        predraft([*args, '--precision', 'bf16'])
    checkpoint_stat = checkpoint.stat()
    os.utime(checkpoint, ns=(checkpoint_stat.st_atime_ns, checkpoint_stat.st_mtime_ns+1000000000))
    with pytest.raises(ValueError, match='checkpoint/device/precision differs'):
        predraft(args)


def test_referenced_mel_and_stats_metadata_invalidate_keys(joint_config, tmp_path):
    checkpoint_path, _ = inputs(joint_config, tmp_path)
    checkpoint = JointCheckpoint(checkpoint_path)
    timing = TimingMap([(0, 120)])
    referenced = tmp_path/'mel.npy'
    np.save(referenced, np.zeros((128, 24), dtype=np.float32))
    metadata = tmp_path/'grid.json'
    metadata.write_text(json.dumps(dict(mel_reference=dict(feature_path=str(referenced)))), 'utf-8')
    kwargs = dict(grid=tmp_path/'grid.npz', device='cpu', precision='fp32', encoder_window_batch=8)
    original = audio_identity(checkpoint, tmp_path/'audio.wav', timing, **kwargs)
    stat = referenced.stat()
    os.utime(referenced, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
    changed = audio_identity(checkpoint, tmp_path/'audio.wav', timing, **kwargs)
    assert identity_key(original) != identity_key(changed)
    stats = tmp_path/'stats.json'
    stats.write_text('{}', 'utf-8')
    kwargs = dict(mel_stats=stats, device='cpu', precision='fp32', encoder_window_batch=8)
    original = audio_identity(checkpoint, tmp_path/'audio.wav', timing, **kwargs)
    stats.write_text('{"changed":true}', 'utf-8')
    assert identity_key(original) != identity_key(audio_identity(checkpoint, tmp_path/'audio.wav', timing, **kwargs))


@pytest.mark.parametrize('first', [.02, -1.25])
def test_real_audio_extraction_is_cached_with_first_and_tempo(joint_config, tmp_path, capsys, first):
    from maidata.preparation.features import MEL_RECIPE
    c = dict(joint_config, mel_bins=128)
    checkpoint, _ = inputs(c, tmp_path)
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps(dict(MEL_RECIPE, mean=[0.]*128, std=[1.]*128)), 'utf-8')
    timing = tmp_path/'timing.json'
    timing.write_text(json.dumps(dict(first_seconds=first, bpm=[[0, 1, 120], [1, 1, 180]])), 'utf-8')
    args = audio_args(checkpoint, tmp_path, timing=timing)
    index = args.index('--grid')
    args[index:index+2] = ['--mel-stats', str(stats)]
    preencode(args)
    result = json.loads(capsys.readouterr().out)
    encoded = load_audio(result['audio_cache']['path'], JointCheckpoint(checkpoint).config, device='cpu')
    assert encoded.local.starts == (0, 16) and encoded.identity['timing']['first_seconds'] == first
    assert not encoded.local.audio.valid[0, :4*c['fpn_rates'][0]].any()
    if first < 0:
        assert not encoded.local.audio.valid[0, 4*c['fpn_rates'][0]]
    with patch('maidata.inference.conditioning.audio_grid', side_effect=AssertionError('must not extract again')):
        preencode(args)
    assert json.loads(capsys.readouterr().out)['audio_cache']['hit']


def test_old_joint_recipe_does_not_inherit_new_p0_normalization_default(joint_config, tmp_path):
    checkpoint, _, payload = joint_checkpoint(dict(joint_config, fpn_p0_normalization=False), tmp_path)
    payload['recipe'].pop('fpn_p0_normalization')
    torch.save(payload, checkpoint)
    assert JointCheckpoint(checkpoint).config['fpn_p0_normalization'] is False
