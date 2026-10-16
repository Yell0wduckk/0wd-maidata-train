"""Cache recovery also reaches resident sessions and keeps operational failures."""
from contextlib import ExitStack
from io import StringIO
from unittest.mock import patch

import pytest
import torch

from test_chart_physical import geometry
from test_inference import config, producer_config
from test_inference_joint import joint_config
from test_inference_cache import audio_args, spies
from test_inference_cache_recovery import valid_cache, read
from maidata.inference.cache import InvalidCacheError, load_audio
from maidata.inference.interactive import Interactive, parser
from maidata.inference.predraft import main as predraft
from maidata.inference.progress import StderrProgress, emit


def test_interactive_repairs_invalid_audio_once_and_reuses_valid_draft(valid_cache, tmp_path, geometry):
    cache, encoded, audio_path, draft_path = valid_cache
    draft_bytes = draft_path.read_bytes()
    audio_path.write_bytes(b'partial')
    args = parser().parse_args([*audio_args(cache.checkpoint.path, tmp_path),
        '--difficulty', '5', '--level', '14.5', '--geometry', 'geometry', '--slide-data', 'slides'])
    for cold in (True, False):
        with ExitStack() as stack:
            calls = spies(stack, forbid=('draft', 'planner') if cold else ('encoder', 'draft', 'planner'))
            stack.enter_context(patch('maidata.physical.Geometry.from_resources', return_value=geometry))
            session = Interactive(args, emit=lambda event: None)
            assert calls['encoder'].call_count == int(cold)
            assert calls['decoder'].call_count == 1
        assert session.duration == encoded.duration
        assert draft_path.read_bytes() == draft_bytes
        repaired = load_audio(audio_path, cache.checkpoint.config, device='cpu')
        torch.testing.assert_close(repaired.local.audio.values, encoded.local.audio.values)


@pytest.mark.parametrize('kind', ['audio', 'draft'])
def test_cpu_allocator_error_keeps_valid_file(valid_cache, kind):
    cache, encoded, audio_path, draft_path = valid_cache
    before = {path: path.read_bytes() for path in (audio_path, draft_path)}
    with patch('maidata.inference.cache.torch.load', side_effect=RuntimeError('DefaultCPUAllocator: not enough memory')):
        with pytest.raises(RuntimeError, match='DefaultCPUAllocator'):
            read(cache, encoded, kind)
    assert all(path.read_bytes() == original for path, original in before.items())
    assert not cache.report[kind].get('invalidated', False)


def test_predraft_never_deletes_an_explicit_invalid_audio_path(valid_cache, tmp_path):
    cache, _, audio_path, draft_path = valid_cache
    outside = tmp_path/'explicit-input.pt'
    outside.write_bytes(b'not an audio cache')
    before = {path: path.read_bytes() for path in (outside, audio_path, draft_path)}
    with pytest.raises(InvalidCacheError, match='invalid inference cache'):
        predraft(['--joint-checkpoint', str(cache.checkpoint.path), '--audio-cache', str(outside),
            '--difficulty', '5', '--level', '14.5', '--device', 'cpu', '--precision', 'fp32',
            '--cache-dir', str(tmp_path/'cache'), '--no-progress'])
    assert all(path.read_bytes() == original for path, original in before.items())


def test_cache_recovery_progress_uses_stderr_message():
    stream = StringIO()
    reporter = StderrProgress(stream=stream, clock=lambda: 0.)
    emit(reporter, 'cache_invalid', cache='draft', path='draft/KEY.pt', error='invalid cached memory shape')
    assert 'Draft/Plan cache removed; rebuilding: draft/KEY.pt' in stream.getvalue()
    assert 'invalid cached memory shape' in stream.getvalue()
