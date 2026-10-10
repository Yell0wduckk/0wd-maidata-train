"""Ordered incremental rows and persistent preview materialization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from fractions import Fraction

from maimai_baseline.runtime_versions import policy, require_compatible

PREFIX = 'MAIMAI_STREAM '


class StreamEmitter:
    def __init__(self):
        self.run_id, self.sequence = uuid.uuid4().hex, 0
        self.version = policy()['components']['stream_protocol']

    def __call__(self, kind, **fields):
        self.sequence += 1
        print(PREFIX + json.dumps(dict(type=kind, run_id=self.run_id, sequence=self.sequence,
            protocol_version=self.version, **fields), ensure_ascii=False), flush=True)


def atomic_text(path, text):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf8', newline='\n')
    # Windows readers may briefly hold a handle without FILE_SHARE_DELETE.
    # Keep the previous complete snapshot until that read finishes.
    for attempt in range(101):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 100: raise
            time.sleep(.01)


class PersistentRenderer:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable), 'render-stream'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def render(self, document):
        self.process.stdin.write(json.dumps(document, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line: raise RuntimeError('Persistent chart-IR renderer exited')
        reply=json.loads(line)
        if not reply.get('ok'): raise ValueError(reply.get('error', 'Rendering failed'))
        return reply['chart']

    def close(self):
        self.process.stdin.close()
        try: self.process.wait(timeout=5)
        except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.process.stdout.close()


def materialize_stream(args):
    from maimai_baseline.inference.generation import _append_window_chunks, _global_rows
    from maimai_baseline.inference.materialize import resolve_chart_ir_executable, maidata_text, _install_audio
    from maimai_baseline.inference.simai import bounded_render_document, compact_spacing, four_beat_lines
    from maimai_dataset.codec import restore_v1_rows
    from maimai_dataset.modifiers import materialize_row
    from maimai_dataset.materialize import materialize_chart_ir, normalize_each_semantics
    from maimai_dataset.timing import TimingMap
    preview=Path(args.preview_dir)
    preview.mkdir(parents=True, exist_ok=False)
    timing=TimingMap.from_dict(json.loads(Path(args.timing).read_text('utf-8-sig')))
    _install_audio(Path(args.audio), preview/('track'+Path(args.audio).suffix.lower()), force=False)
    renderer=PersistentRenderer(resolve_chart_ir_executable(Path(args.miacode_chart_ir_exe)))
    active, init, sequence, run_id = [], None, 0, None
    protocol_version = policy()['components']['stream_protocol']
    closed_moments, closed_rows, tail_document, last_tick = [], 0, None, 0
    windows=0
    try:
        for line in sys.stdin:
            e=json.loads(line)
            if e.get('protocol_version') != protocol_version:
                raise ValueError('Unsupported streaming protocol')
            if e['sequence'] != sequence+1 or (run_id is not None and run_id != e['run_id']):
                raise ValueError('Stream sequence/run mismatch')
            sequence,run_id=e['sequence'],e['run_id']
            kind=e['type']
            if kind=='init':
                require_compatible(e,'materialize'); init=e
            elif kind=='window_start':
                if active: raise ValueError('Unclosed streaming window')
                windows=e['index']
            elif kind=='rollback': active=active[:e['row_count']]
            elif kind=='row': active.append(e['row'])
            elif kind=='window_end':
                if tail_document is not None: closed_moments.extend(tail_document['moments'])
                closed_rows += len(_global_rows(_append_window_chunks(active,target_start_tick=windows*1536)))
                active=[]; tail_document=None
            elif kind=='inference_complete':
                return {'stream_complete':True, 'preview':str(preview/'maidata.txt')}
            else: continue
            if init is None: raise ValueError('Rows received before stream initialization')
            # A bare MOMENT is tentative until its first NOTE succeeds.
            renderable=list(active)
            if renderable and renderable[-1]['event']=='MOMENT':renderable.pop()
            chunks=_append_window_chunks(renderable,target_start_tick=windows*1536) if renderable else []
            rows=[materialize_row(row) for row in _global_rows(chunks)]
            last_tick=max(last_tick,max([tick for tick,_ in chunks],default=0))
            end=Fraction(last_tick+1,96)
            tail_document=normalize_each_semantics(materialize_chart_ir(restore_v1_rows(rows),timing=timing,
                difficulty='MAS' if init['difficulty']==5 else 'REM'))
            document=dict(tail_document, moments=closed_moments+tail_document['moments'],
                render_end_beat={'num':end.numerator,'den':end.denominator})
            document['controls']=[c for c in document['controls']
                if Fraction(c['position_beats']['num'],c['position_beats']['den'])<end]
            chart=four_beat_lines(compact_spacing(renderer.render(bounded_render_document(document))))
            text=maidata_text(chart,timing=timing,difficulty_id=init['difficulty'],level=init['level'],
                cabinet=init['cabinet'],title=args.title or init['song_id'],artist=args.artist,designer=args.designer)
            atomic_text(preview/'maidata.txt',text)
            print(PREFIX+json.dumps(dict(type='preview',run_id=run_id,sequence=sequence,
                path=str(preview/'maidata.txt'),difficulty=init['difficulty'],rows=closed_rows+len(rows))),flush=True)
        raise ValueError('Stream ended before inference completion')
    finally:
        renderer.close()
