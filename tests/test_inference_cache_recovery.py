"""Invalid files recover once; operational failures preserve valid cache files."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config
from test_inference_joint import joint_config
from test_inference_cache import inputs, audio_args
from maidata.inference.cache import JointCache, load_audio
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.preencode import main as preencode
from maidata.inference.predraft import main as predraft


@pytest.fixture
def valid_cache(joint_config, tmp_path, capsys):
    checkpoint_path, _ = inputs(joint_config, tmp_path)
    preencode(audio_args(checkpoint_path, tmp_path))
    audio_path = Path(json.loads(capsys.readouterr().out)['audio_cache']['path'])
    predraft(['--joint-checkpoint', str(checkpoint_path), '--audio-cache', str(audio_path),
        '--difficulty', '5', '--level', '14.5', '--device', 'cpu', '--precision', 'fp32',
        '--cache-dir', str(tmp_path/'cache'), '--no-progress'])
    draft_path = Path(json.loads(capsys.readouterr().out)['draft_cache']['path'])
    checkpoint = JointCheckpoint(checkpoint_path)
    encoded = load_audio(audio_path, checkpoint.config, device='cpu')
    cache = JointCache(tmp_path/'cache', checkpoint, device='cpu', precision='fp32')
    return cache, encoded, audio_path, draft_path


def read(cache, encoded, kind):
    return (cache.find_audio(encoded.identity) if kind == 'audio'
            else cache.draft(encoded, difficulty=5, level=14.5))


@pytest.mark.parametrize('kind', ['audio', 'draft'])
@pytest.mark.parametrize('error', [PermissionError('cache read denied'),
                                  torch.OutOfMemoryError('cache read out of memory')])
def test_operational_read_failure_does_not_invalidate_valid_cache(valid_cache, kind, error):
    cache, encoded, audio_path, draft_path = valid_cache
    before = {path: path.read_bytes() for path in (audio_path, draft_path)}
    with patch('maidata.inference.cache.torch.load', side_effect=error), \
         patch.object(cache.checkpoint, 'models', side_effect=AssertionError('must not rebuild')):
        with pytest.raises(type(error), match=str(error)):
            read(cache, encoded, kind)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not cache.report[kind].get('invalidated', False)


@pytest.mark.parametrize('kind', ['audio', 'draft'])
def test_device_transfer_oom_does_not_invalidate_valid_cache(valid_cache, kind):
    cache, encoded, audio_path, draft_path = valid_cache
    before = {path: path.read_bytes() for path in (audio_path, draft_path)}
    with patch('torch.Tensor.to', side_effect=torch.OutOfMemoryError('device transfer out of memory')), \
         patch.object(cache.checkpoint, 'models', side_effect=AssertionError('must not rebuild')):
        with pytest.raises(torch.OutOfMemoryError, match='device transfer out of memory'):
            read(cache, encoded, kind)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not cache.report[kind].get('invalidated', False)


@pytest.mark.parametrize('error', [RuntimeError('model calculation failed'),
                                  torch.OutOfMemoryError('model calculation out of memory')])
def test_draft_rebuild_failure_is_not_retried_and_keeps_valid_audio(valid_cache, error):
    cache, encoded, audio_path, draft_path = valid_cache
    original_audio = audio_path.read_bytes()
    draft_path.write_bytes(b'partial')
    with patch('maidata.inference.cache.decode_joint_draft', side_effect=error) as decode:
        with pytest.raises(type(error), match=str(error)):
            cache.draft(encoded, difficulty=5, level=14.5)
    assert decode.call_count == 1
    assert audio_path.read_bytes() == original_audio and not draft_path.exists()
    assert cache.report['draft']['hit'] is False
    assert cache.report['draft']['invalidated'] is True
    assert 'invalid inference cache' in cache.report['draft']['error']
