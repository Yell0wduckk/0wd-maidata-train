"""Optional host-side progress events; no tensors, synchronization or trace data."""
from contextlib import contextmanager
import math
import sys
import time


DEFAULT_INTERVAL = 2.0


def emit(progress, event, **fields):
    """Python APIs stay silent unless the caller supplies a callback."""
    if progress is not None:
        progress(dict(event=event, **fields))


@contextmanager
def stage(progress, name):
    emit(progress, 'stage_start', stage=name)
    try:
        yield
    except Exception as exc:
        emit(progress, 'stage_error', stage=name, error=str(exc))
        raise
    else:
        emit(progress, 'stage_done', stage=name)


class StderrProgress:
    """Line-oriented CLI progress, with throttled encoder/token updates."""
    LABELS = {
        'runtime': 'Runtime imports', 'models': 'Model loading', 'audio': 'Audio input',
        'geometry': 'Physical geometry', 'encoder_global': 'Draft encoder global audio',
        'encoder_windows': 'Draft encoder audio windows', 'draft': 'Full-song Draft',
        'decoder_encoder_global': 'Decoder encoder global audio',
        'decoder_encoder_windows': 'Decoder encoder audio windows',
        'planner': 'Window Planner', 'generation': 'Autoregressive generation',
        'decoder_audio': 'Decoder audio projection',
        'chart_ir': 'Chart IR conversion', 'write_outputs': 'Writing outputs',
        'audio_cache_read': 'Audio cache loading', 'audio_cache_write': 'Audio cache saving',
        'draft_cache_read': 'Draft/Plan cache loading', 'draft_cache_write': 'Draft/Plan cache saving',
    }

    def __init__(self, *, interval=DEFAULT_INTERVAL, stream=None, clock=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('progress-interval must be finite and positive')
        self.interval = interval
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.started = self.clock()
        self.stages = {}
        self.last_update = {}
        self.window = {}

    def __call__(self, event):
        now = self.clock()
        kind = event['event']
        message = None
        if kind.startswith('stage_'):
            name = event['stage']
            label = self.LABELS.get(name, name)
            if kind == 'stage_start':
                self.stages[name] = now
                message = f'{label}: started'
            else:
                elapsed = now - self.stages.pop(name, now)
                status = 'done' if kind == 'stage_done' else 'failed'
                message = f'{label}: {status} ({elapsed:.1f}s)'
                if kind == 'stage_error':
                    message += f"; {event['error']}"
        elif kind == 'cache_invalid':
            label = 'Audio' if event['cache'] == 'audio' else 'Draft/Plan'
            message = f"Invalid {label} cache removed; rebuilding: {event['path']}; {event['error']}"
        elif kind == 'encoder_progress':
            key = ('encoder', event.get('encoder'))
            complete = event['completed'] == event['total']
            if complete or now - self.last_update.get(key, self.started) >= self.interval:
                label = {'draft': 'Draft encoder', 'decoder': 'Decoder encoder'}.get(event.get('encoder'), 'Encoder')
                message = f"{label} audio windows: {event['completed']}/{event['total']}"
                self.last_update[key] = now
        elif kind == 'window_start':
            self.window = dict(event)
            self.last_update[('window', event['window'])] = now
            message = (f"Window {event['window']}/{event['windows']} "
                       f"beats [{event['start_beat']:g}, {event['end_beat']:g}): started")
        elif kind in ('window_progress', 'window_done'):
            key = ('window', event['window'])
            if kind == 'window_done' or now - self.last_update.get(key, self.started) >= self.interval:
                values = dict(self.window, **event)
                message = f"Window {values['window']}/{values['windows']}: "
                if kind == 'window_done':
                    message += 'done; '
                message += f"{values['tokens']} tokens, {values['backtracks']} backtracks"
                if 'cursor_beat' in values:
                    message += f"; beat {values['cursor_beat']:g}"
                self.last_update[key] = now
        if message is not None:
            print(f'[inference {now-self.started:.1f}s] {message}', file=self.stream, flush=True)
