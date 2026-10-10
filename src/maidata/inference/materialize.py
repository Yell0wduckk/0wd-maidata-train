"""Standalone Chart IR -> playable maidata.txt, through the native MiaCode CLI.

No model, Torch, training loader or old repository Python imports are required.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

from maidata.chart import encode_chart, token_to_dict
from maidata.chart.types import Note
from maidata.decoder.windows import real_onsets
from maidata.time import TimingMap
from .simai import bounded_render_document, compact_spacing, four_beat_lines


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def resolve_executable(requested):
    p = Path(requested).expanduser().resolve()
    if p.is_dir():
        choices = [p/'miacode_chart_ir.exe', p/'miacode_chart_ir']
    elif p.name.lower() in ('miacode.exe', 'miacode'):
        choices = [p.with_name('miacode_chart_ir.exe'), p.with_name('miacode_chart_ir')]
    else:
        choices = [p]
    for candidate in choices:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'MiaCode chart IR executable not found: {p}')


def native(executable, arguments, payload, *, timeout):
    result = subprocess.run([str(executable), *arguments], input=payload.encode('utf-8'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'MiaCode {arguments[0]} failed ({result.returncode}): '
                           +result.stderr.decode('utf-8', errors='replace'))
    return result.stdout.decode('utf-8-sig')


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


def semantic_events(document):
    """Compare current fields, allowing native ordering within an each group.

    Derived IDs, 180 Hz rounding and per-segment length projections are not
    generated fields. Onsets, flags, route order/branches and duration ticks are.
    """
    encoded = encode_chart(document)
    groups = real_onsets(encoded.tokens, encoded.timing, sequence_start_tick=encoded.start_tick)
    events = []
    for group in groups:
        notes, current = [], None
        for token in group.content:
            if isinstance(token, Note):
                current = []
                notes.append(current)
            current.append(token_to_dict(token))
        events.append((group.tick, sorted(json.dumps(n, sort_keys=True) for n in notes)))
    return encoded.timing, events


def check_roundtrip(expected, actual):
    first, events = semantic_events(expected)
    second, recovered = semantic_events(actual)
    if events != recovered:
        raise ValueError('MiaCode round-trip changed generated onset/NOTE/ROUTE/duration fields')
    if abs(first.first_seconds-second.first_seconds) > 1e-8:
        raise ValueError('MiaCode round-trip changed first_seconds')
    # Native rendering may add redundant tempo controls. Compare both timelines
    # at every original/recovered change point rather than their spelling.
    beats = {b for b, _ in first.points} | {b for b, _ in second.points}
    for beat in beats:
        if (abs(first.bpm_at_beat(beat)-second.bpm_at_beat(beat)) > 1e-8 or
                abs(first.beat_to_second(beat)-second.beat_to_second(beat)) > 1e-8):
            raise ValueError('MiaCode round-trip changed the BPM timeline')


def materialize(ir_path, audio_path, output, miacode, *, difficulty, level, title='', artist='',
                designer='maidata', cabinet='DX', timeout=30., force=False):
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
    executable = resolve_executable(miacode)
    chart = four_beat_lines(compact_spacing(native(executable, ['render', '--input', '-', '--output', '-'],
        json.dumps(bounded_render_document(document), ensure_ascii=False), timeout=timeout)))
    first = document['first_seconds']
    recovered = json.loads(native(executable,
        ['dump', '--file', '-', '--first', format(first, '.17g'), '--output', '-'], chart, timeout=timeout))
    check_roundtrip(document, recovered)
    timing = TimingMap.from_ir(document)
    clean = lambda s: str(s).replace('\r', ' ').replace('\n', ' ').replace('&', ' ').strip()
    text = '\n'.join([f'&title={clean(title)}', f'&artist={clean(artist)}', f'&first={first:.17g}',
        f'&wholebpm={timing.points[0][1]:.17g}', f'&cabinet={clean(cabinet)}', '&shortid=generated',
        f'&lv_{difficulty}={level:.4g}', f'&des_{difficulty}={clean(designer)}',
        f'&inote_{difficulty}={chart.strip()}', ''])
    output.mkdir(parents=True, exist_ok=True)
    if target_audio.resolve() != audio_path:
        shutil.copy2(audio_path, target_audio)
    (output/'maidata.txt').write_text(text, encoding='utf-8', newline='\n')
    write_json(output/'render.ir.json', document)
    write_json(output/'reparsed.ir.json', recovered)
    report = dict(schema='maidata.materialization.v1', ir=str(Path(ir_path).resolve()),
        audio=str(audio_path), miacode=str(executable), roundtrip=True, moments=len(document['moments']),
        maidata=str((output/'maidata.txt').resolve()), track=str(target_audio.resolve()))
    write_json(output/'materialization.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Materialize current maidata Chart IR as playable maidata.txt + track audio.')
    parser.add_argument('--ir', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--miacode', required=True, help='miacode_chart_ir executable or its directory (Windows/Linux)')
    parser.add_argument('--difficulty', type=int, required=True)
    parser.add_argument('--level', type=float, required=True)
    parser.add_argument('--title', default='')
    parser.add_argument('--artist', default='')
    parser.add_argument('--designer', default='maidata')
    parser.add_argument('--cabinet', default='DX')
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    report = materialize(args.ir, args.audio, args.output, args.miacode, difficulty=args.difficulty,
        level=args.level, title=args.title, artist=args.artist, designer=args.designer,
        cabinet=args.cabinet, timeout=args.timeout, force=args.force)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
