"""Standalone Chart IR -> playable maidata.txt using the Python simai renderer.

No model, Torch, training loader or old repository Python imports are required.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import time

from maidata.time import TimingMap
from .render import render_document


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def render_origin(document):
    """simai begins at beat zero. Shift negative beat coordinates, not audio time."""
    result = deepcopy(document)
    positions = [Fraction(m['position_beats']['num'], m['position_beats']['den']) for m in result['moments']]
    shift = min(0, math.floor(min(positions, default=0)))
    if shift:
        timing = TimingMap.from_ir(document)
        result['first_seconds'] = timing.beat_to_second(shift)
        for item in [*result['moments'], *result['controls']]:
            p = item['position_beats']
            beat = Fraction(p['num'], p['den'])-shift
            item['position_beats'] = dict(num=beat.numerator, den=beat.denominator)
        result['controls'].insert(0, dict(kind='bpm', bpm=timing.bpm_at_beat(shift),
            position_beats=dict(num=0, den=1), tick=math.floor(result['first_seconds']*180+.5)))
        if 'render_end_beat' in result:
            p = result['render_end_beat']
            beat = Fraction(p['num'], p['den'])-shift
            result['render_end_beat'] = dict(num=beat.numerator, den=beat.denominator)
    return result


def maidata_text(chart, first, bpm, *, difficulty, level,
                 title='', artist='', designer='maidata', cabinet='DX'):
    clean = lambda s: str(s).replace('\r', ' ').replace('\n', ' ').replace('&', ' ').strip()
    return '\n'.join([f'&title={clean(title)}', f'&artist={clean(artist)}', f'&first={first:.17g}',
        f'&wholebpm={bpm:.17g}', f'&cabinet={clean(cabinet)}', '&shortid=generated',
        f'&lv_{difficulty}={level:.4g}', f'&des_{difficulty}={clean(designer)}',
        f'&inote_{difficulty}={chart.strip()}', ''])


def atomic_text(path, text):
    """Readers keep a complete snapshot, including briefly occupied Windows files."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        temporary.write_text(text, encoding='utf-8', newline='\n')
        for attempt in range(101):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 100:
                    raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def install_audio(audio_path, output):
    output, audio_path = Path(output), Path(audio_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_audio = output/('track'+audio_path.suffix.lower())
    if target_audio.resolve() != audio_path:
        shutil.copy2(audio_path, target_audio)
    return dict(maidata=str((output/'maidata.txt').resolve()), track=str(target_audio.resolve()))


def write_playable(chart, first, bpm, audio_path, output, *, difficulty, level,
                   title='', artist='', designer='maidata', cabinet='DX'):
    paths = install_audio(audio_path, output)
    atomic_text(paths['maidata'], maidata_text(chart, first, bpm, difficulty=difficulty, level=level,
        title=title, artist=artist, designer=designer, cabinet=cabinet))
    return paths


def materialize(ir_path, audio_path, output, *, difficulty, level, title='', artist='',
                designer='maidata', cabinet='DX', force=False):
    if difficulty not in range(1, 8) or not math.isfinite(level):
        raise ValueError('difficulty must be 1..7 and level must be a finite numeric value')
    output, audio_path = Path(output), Path(audio_path).resolve()
    target_audio = output/('track'+audio_path.suffix.lower())
    destinations = [output/'maidata.txt', target_audio, output/'render.ir.json',
                    output/'reparsed.ir.json', output/'materialization.json']
    if not force and any(p.exists() for p in destinations):
        raise FileExistsError(f'output already exists: {output}; use a new directory or --force')
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    document = render_origin(read_json(ir_path))
    chart = render_document(document)
    timing = TimingMap.from_ir(document)
    paths = write_playable(chart, document['first_seconds'], timing.points[0][1], audio_path, output,
        difficulty=difficulty, level=level, title=title, artist=artist, designer=designer, cabinet=cabinet)
    if force and (output/'reparsed.ir.json').is_file():
        (output/'reparsed.ir.json').unlink()
    write_json(output/'render.ir.json', document)
    report = dict(schema='maidata.materialization.v2', ir=str(Path(ir_path).resolve()),
        audio=str(audio_path), renderer='python-simai', moments=len(document['moments']), **paths)
    write_json(output/'materialization.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Materialize current maidata Chart IR as playable maidata.txt + track audio.')
    parser.add_argument('--ir', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--difficulty', type=int, required=True)
    parser.add_argument('--level', type=float, required=True)
    parser.add_argument('--title', default='')
    parser.add_argument('--artist', default='')
    parser.add_argument('--designer', default='maidata')
    parser.add_argument('--cabinet', default='DX')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    report = materialize(args.ir, args.audio, args.output, difficulty=args.difficulty,
        level=args.level, title=args.title, artist=args.artist, designer=args.designer,
        cabinet=args.cabinet, force=args.force)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
