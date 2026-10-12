"""Live Python simai snapshots from accepted fields and exact prefix rollbacks."""
from maidata.chart import Moment
from maidata.decoder.windows import real_onsets
from .materialize import atomic_text, install_audio, maidata_text


class StreamingPreview:
    """Own the writer's commits; only the active window's complete prefix is tentative."""
    def __init__(self, writer, timing, audio_path, output, **metadata):
        self.writer, self.timing = writer, timing
        self.paths = install_audio(audio_path, output)
        self.metadata = metadata
        self.tokens, self.complete = [], []
        self.start_tick = 0
        self.last_text = None

    def write(self, chart):
        text = maidata_text(chart, self.timing.first_seconds, self.timing.points[0][1], **self.metadata)
        if text != self.last_text:
            atomic_text(self.paths['maidata'], text)
            self.last_text = text

    def refresh(self):
        count = next((i+1 for i in range(len(self.complete)-1, -1, -1) if self.complete[i]), 0)
        groups = real_onsets((Moment(0), *self.tokens[:count]), self.timing,
                             sequence_start_tick=self.start_tick) if count else ()
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
            self.writer.add_groups(event['groups'])
            self.tokens.clear()
            self.complete.clear()
            self.refresh()
