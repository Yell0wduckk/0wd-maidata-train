"""Observable progress stays separate from sampling, tensors and stdout products."""
from io import StringIO
from unittest.mock import patch

import pytest
import torch

from test_inference import config, producer_config
from test_pretrain_runtime import item
from maidata.inference.__main__ import parser
from maidata.inference.conditioning import prepare_conditions
from maidata.inference.progress import StderrProgress, emit, stage
from maidata.model import build_model, build_draft, build_planner
from maidata.time import TimingMap


def test_optional_callback_and_failed_stage_preserve_exception():
    events = []
    emit(None, 'window_progress', tokens=1)
    with pytest.raises(RuntimeError, match='original failure'):
        with stage(events.append, 'generation'):
            raise RuntimeError('original failure')
    assert events == [dict(event='stage_start', stage='generation'),
                      dict(event='stage_error', stage='generation', error='original failure')]


def test_stderr_progress_throttles_tokens_and_reports_window_boundaries(capsys):
    now = [10.]
    reporter = StderrProgress(interval=2, clock=lambda: now[0])
    emit(reporter, 'stage_start', stage='generation')
    emit(reporter, 'window_start', window=1, windows=2, start_beat=-1, end_beat=15)
    for token in range(1, 101):
        now[0] = 10 + token / 100
        emit(reporter, 'window_progress', window=1, tokens=token, backtracks=0, cursor_beat=2.)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert len(captured.err.splitlines()) == 2
    assert 'Window 1/2 beats [-1, 15): started' in captured.err
    now[0] = 12.
    emit(reporter, 'window_progress', window=1, tokens=101, backtracks=1, cursor_beat=3.)
    emit(reporter, 'window_done', window=1, tokens=102, backtracks=1)
    emit(reporter, 'stage_done', stage='generation')
    captured = capsys.readouterr()
    assert captured.out == '' and len(captured.err.splitlines()) == 3
    assert '101 tokens, 1 backtracks; beat 3' in captured.err
    assert 'Window 1/2: done; 102 tokens' in captured.err
    assert 'Autoregressive generation: done (2.0s)' in captured.err


def test_encoder_final_progress_is_visible_without_waiting_for_interval():
    stream = StringIO()
    reporter = StderrProgress(stream=stream, clock=lambda: 0.)
    emit(reporter, 'encoder_progress', completed=8, total=9)
    assert stream.getvalue() == ''
    emit(reporter, 'encoder_progress', completed=9, total=9)
    assert 'Encoder audio windows: 9/9' in stream.getvalue()


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_invalid_progress_interval(interval):
    with pytest.raises(ValueError, match='progress-interval'):
        StderrProgress(interval=interval)


def test_cli_progress_defaults_and_opt_out():
    common = ['--encoder-draft-checkpoint', 'ed.ckpt', '--decoder-checkpoint', 'm.ckpt',
              '--audio', 'audio.wav', '--grid', 'grid.npz', '--bpm', '120',
              '--geometry', 'geometry', '--slide-data', 'slides.json', '--difficulty', '5',
              '--level', '14.5', '--output', 'result']
    args = parser().parse_args(common)
    assert args.progress and args.progress_interval == 2 and not args.trace and args.stream_output
    args = parser().parse_args([*common, '--no-progress', '--progress-interval', '5', '--no-stream-output'])
    assert not args.progress and args.progress_interval == 5 and not args.trace and not args.stream_output


def test_conditioning_events_leave_model_outputs_unchanged(config, capsys):
    encoder, draft, planner = (build_model(config).eval(), build_draft(config).eval(), build_planner(config).eval())
    timing = TimingMap([(0, 120)])
    grid = item(0, config)['grid']
    arguments = dict(difficulty=5, level=14.5, device='cpu', encoder_window_batch=1)
    events = []
    with patch('torch.cuda.synchronize', side_effect=AssertionError('progress must not synchronize CUDA')):
        silent = prepare_conditions(encoder, draft, planner, config, grid, timing, 1.2, **arguments)
        observed = prepare_conditions(encoder, draft, planner, config, grid, timing, 1.2,
                                      progress=events.append, **arguments)
    for name in ('audio', 'z'):
        for field in ('values', 'beats', 'valid'):
            assert torch.equal(getattr(getattr(silent, name), field), getattr(getattr(observed, name), field))
    assert torch.equal(silent.plans, observed.plans) and silent.starts == observed.starts
    assert [(e['event'], e['stage']) for e in events if 'stage' in e] == [
        (event, name) for name in ('encoder_global', 'encoder_windows', 'draft', 'planner')
        for event in ('stage_start', 'stage_done')]
    updates = [e for e in events if e['event'] == 'encoder_progress']
    assert updates and updates[-1]['completed'] == updates[-1]['total']
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''
