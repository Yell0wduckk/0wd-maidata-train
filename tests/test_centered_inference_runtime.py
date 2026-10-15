"""Real new-checkpoint inference and independent E/D/P caches use integer grids."""
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from maidata.inference.__main__ import main as infer
from maidata.inference.preencode import main as preencode
from maidata.inference.predraft import main as predraft
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.cache import draft_identity, identity_key
from test_chart_physical import geometry
from test_inference import config, producer_config
from test_inference_cache import audio_args, spies
from test_inference_joint import arguments, joint_checkpoint
from test_inference_materialize import write_audio
from test_integer_joint_runtime import integer_config
from test_plan4_runtime import plan4_config
from test_pretrain_runtime import item as auxiliary_item


def inputs(c, root):
    path, _, _ = joint_checkpoint(c, root)
    write_audio(root/'audio.wav')
    np.savez(root/'grid.npz', **auxiliary_item(0, c)['grid'])
    return path


@pytest.mark.parametrize('z_enabled,source', [(True, 'predicted'), (False, 'predicted'),
                                            (True, 'none'), (False, 'none'), (False, 'gt')])
def test_real_integer_generation_online_cold_hot_and_independent_plan_keys(
        integer_config, tmp_path, geometry, z_enabled, source):
    path = inputs(integer_config, tmp_path)
    checkpoint = JointCheckpoint(path, z_cross_attention=z_enabled, plan_source=source)
    strict = checkpoint.models(device='cpu')
    assert strict.planner.layout == 'centered-overlap'
    assert strict.config['draft_z_anchor_mode'] == 'integer'
    assert all(getattr(strict, name) is not None for name in ('encoder', 'draft', 'planner', 'decoder'))
    args = [*arguments(path, tmp_path), '--grid', str(tmp_path/'grid.npz'), '--plan-source', source]
    if not z_enabled:
        args += ['--no-z-cross-attention']
    if source == 'gt':
        document = dict(planner_layout='centered-overlap', plan_span_beats=16, plan_stride_beats=4,
            windows=[dict(start_beat=start, center_beats=list(range(start, start+17, 4)),
                valid=[True]*5, plan12=[[0.]*8+[1., 0., 0., 0.] for _ in range(5)]) for start in (0, 16)])
        (tmp_path/'gt.json').write_text(json.dumps(document))
        args += ['--gt-plan', str(tmp_path/'gt.json')]
    reports, charts = [], []
    for mode in ('online', 'cold', 'hot'):
        forbid = ('encoder', 'draft', 'planner') if mode == 'hot' else (
            ('draft', 'planner') if not z_enabled and source != 'predicted' else ())
        with ExitStack() as stack:
            calls = spies(stack, forbid=forbid)
            stack.enter_context(patch('maidata.physical.Geometry.from_resources', return_value=geometry))
            if mode == 'hot':
                stack.enter_context(patch('maidata.preparation.cache.load_grid', side_effect=AssertionError('hot grid read')))
                stack.enter_context(patch('soundfile.info', side_effect=AssertionError('hot audio read')))
            infer([*args, '--output', str(tmp_path/mode),
                *([] if mode == 'online' else ['--cache-dir', str(tmp_path/'cache')])])
            assert calls['decoder'].call_count == 1
        reports.append(json.loads((tmp_path/mode/'generation.json').read_text('utf-8')))
        charts.append((tmp_path/mode/'maidata.txt').read_bytes())
    assert charts[0] == charts[1] == charts[2]
    assert reports[0]['windows'] == reports[1]['windows'] == reports[2]['windows']
    assert all(report['complete'] for report in reports)
    assert [row['plan_beats'] for row in reports[0]['windows']] == [[0,4,8,12,16], [16,20,24,28,32]]
    assert all(np.asarray(row['plan12']).shape == (5, 12) for row in reports[0]['windows'])
    assert all(row['plan_valid'] == [source != 'none']*5 for row in reports[0]['windows'])
    assert reports[1]['cache']['audio']['hit'] is False and reports[2]['cache']['audio']['hit'] is True


def test_integer_preencode_predraft_strict_load_and_semantic_cache_rejection(integer_config, tmp_path, capsys):
    path = inputs(integer_config, tmp_path)
    with ExitStack() as stack:
        calls = spies(stack, forbid=('draft', 'planner', 'decoder'))
        preencode(audio_args(path, tmp_path))
        assert calls['encoder'].call_count == 1
    audio_path = Path(json.loads(capsys.readouterr().out)['audio_cache']['path'])
    args = ['--joint-checkpoint', str(path), '--audio-cache', str(audio_path),
        '--difficulty', '5', '--level', '14.5', '--device', 'cpu', '--precision', 'fp32',
        '--cache-dir', str(tmp_path/'cache'), '--no-progress']
    with ExitStack() as stack:
        calls = spies(stack, forbid=('encoder', 'decoder'))
        predraft(args)
        assert calls['draft'].call_count == calls['planner'].call_count == 1
    draft_path = Path(json.loads(capsys.readouterr().out)['draft_cache']['path'])
    saved = torch.load(draft_path, weights_only=True)
    assert saved['z']['beats'].tolist() == [list(range(33))]
    assert saved['plans'].shape == (2, 5, 12) and saved['plan_valid'].all()
    assert saved['plan_beats'].tolist() == [[0,4,8,12,16], [16,20,24,28,32]]
    with ExitStack() as stack:
        spies(stack, forbid=('encoder', 'draft', 'planner', 'decoder'))
        predraft(args)
    assert json.loads(capsys.readouterr().out)['draft_cache']['hit']
    checkpoint = JointCheckpoint(path)
    first = draft_identity(checkpoint, 'audio', difficulty=5, level=14.5, device='cpu', precision='fp32')
    checkpoint.config['planner_layout'] = 'legacy'
    second = draft_identity(checkpoint, 'audio', difficulty=5, level=14.5, device='cpu', precision='fp32')
    assert identity_key(first) != identity_key(second)
    # Same tensor shapes with a shifted phase must still fail and preserve cache.
    saved['plan_beats'][0, 0] = 2
    torch.save(saved, draft_path)
    original = draft_path.read_bytes()
    with pytest.raises(ValueError, match='cached centered Plan coordinates'):
        predraft(args)
    assert draft_path.read_bytes() == original
