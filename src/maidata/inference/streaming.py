"""Live Python simai snapshots from accepted fields and exact prefix rollbacks."""
from maidata.chart import Moment
from maidata.decoder.windows import real_onsets
from .materialize import atomic_text, install_audio, maidata_text
from fractions import Fraction
from copy import deepcopy
import json
import time


class StreamingPreview:
    """Own the writer's commits; only the active window's complete prefix is tentative."""
    def __init__(self, writer, timing, audio_path, output, *, ir_geometry=None, ir_metadata=None, publish_live=True, **metadata):
        self.writer, self.timing = writer, timing
        self.paths = install_audio(audio_path, output)
        self.metadata = metadata
        self.tokens, self.complete = [], []
        self.start_tick = 0
        self.last_text = None
        self.geometry, self.ir_metadata = ir_geometry, ir_metadata
        self.publish_live = publish_live
        self.committed_ir, self.active_groups = [], ()
        self.last_publish, self.revision = float('-inf'), 0
        from pathlib import Path
        self.snapshot_path = Path(output)/'chart.snapshot.json'

    def decode(self, groups):
        from maidata.chart import decode_chart
        from .generation import chart_from_groups
        return decode_chart(chart_from_groups(groups, self.timing, self.ir_metadata), self.geometry)['moments']

    def write(self, chart, *, complete=False, force=False):
        if self.geometry is not None and not force and time.monotonic()-self.last_publish < .25:
            return
        text = maidata_text(chart, self.timing.first_seconds, self.timing.points[0][1], **self.metadata)
        if text != self.last_text:
            atomic_text(self.paths['maidata'], text)
            self.last_text = text
        if self.geometry is not None:
            ir = deepcopy(self.ir_metadata)
            ir['moments'] = deepcopy(self.committed_ir) + self.decode(self.active_groups)
            for index, moment in enumerate(ir['moments']):
                moment['id'] = f'moment-{index}'
                for group_index, group in enumerate(moment['groups']):
                    for note_index, note in enumerate(group['notes']):
                        note['id'] = f'note-{index}-{group_index}-{note_index}'
            if not complete:
                end = Fraction(self.active_groups[-1].tick, 96) if self.active_groups else self.writer.pending[0]
                end += Fraction(1, 96)
                ir['render_end_beat'] = dict(num=end.numerator, den=end.denominator)
                ir['controls'] = [c for c in ir['controls']
                    if Fraction(c['position_beats']['num'], c['position_beats']['den']) < end]
            self.revision += 1
            atomic_text(self.snapshot_path, json.dumps(dict(revision=self.revision, chart=chart, ir=ir, complete=complete), ensure_ascii=False, allow_nan=False))
            if complete:
                atomic_text(self.snapshot_path.parent/'model.ir.json', json.dumps(ir, ensure_ascii=False, allow_nan=False))
            self.last_publish = time.monotonic()

    def refresh(self):
        count = next((i+1 for i in range(len(self.complete)-1, -1, -1) if self.complete[i]), 0)
        groups = real_onsets((Moment(0), *self.tokens[:count]), self.timing,
                             sequence_start_tick=self.start_tick) if count else ()
        self.active_groups = groups
        if self.publish_live:
            self.write(self.writer.preview(groups))

    def __call__(self, event):
        kind = event['event']
        if kind == 'window_start':
            self.start_tick = event['start_tick']
            self.tokens.clear()
            self.complete.clear()
            self.refresh()
        elif kind == 'token':
            self.tokens.append(event['token'])
            self.complete.append(event['complete'])
            if event['complete']:
                self.refresh()
        elif kind == 'rollback':
            del self.tokens[event['token_count']:]
            del self.complete[event['token_count']:]
            self.refresh()
        elif kind == 'window_end':
            if self.geometry is not None:
                self.committed_ir.extend(self.decode(event['groups']))
            self.writer.add_groups(event['groups'])
            self.tokens.clear()
            self.complete.clear()
            self.refresh()
