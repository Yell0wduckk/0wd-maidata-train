"""Original-only official full-chart production CLI, safely resumable."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from maidata.chart.codec import encode_chart, token_to_dict
from maidata.chart.types import ChartError, Note, Route, Moment
from maidata.physical import Geometry, PhysicalState
from maidata.time import TimingMap
from .cache import SCHEMA, read_json, write_json, digest, save_chart

GEOMETRY = None


def fields(text):
    matches = list(re.finditer(r'^&([^=\r\n]+)=', text, re.M))
    return {m[1]: text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def catalog(source, audio_root, parser, geometry_root, slide_data):
    audio = read_json(audio_root/'catalog.json')
    songs = {s['song_id']: s for s in audio['songs']}
    rows, audio_only = [], []
    versions = Counter()
    for path in sorted(source.glob('*/maidata.txt')):
        text = path.read_text(encoding='utf-8-sig')
        f, song = fields(text), songs[path.parent.name]
        versions[f.get('version', '')] += 1
        charts = {k.split('_', 1)[1]: v for k, v in f.items() if k.startswith('inote_') and v.strip()}
        if set(charts) != set(song['charts']):
            raise ValueError(f'source/audio difficulty coverage mismatch: {path}')
        if not charts:
            audio_only.append(dict(song_id=song['song_id'], variant=song['audio_only_variant']))
        for difficulty, body in charts.items():
            variant = song['charts'][difficulty]
            rows.append(dict(chart_id=f"{song['song_id']}:{difficulty}", song_id=song['song_id'],
                difficulty=difficulty, version=f.get('version', ''), source_path=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                source_header_line=next(i+1 for i, line in enumerate(text.splitlines()) if line.startswith(f'&inote_{difficulty}=')),
                audio_variant=variant, audio_asset=song['asset'],
                timing=audio['variants'][variant]['timing'],
                audio_end_seconds=audio['assets'][song['asset']]['duration_seconds'],
                path=f"charts/{song['song_id']}/{difficulty}.json.gz"))
    if set(songs) != {p.parent.name for p in source.glob('*/maidata.txt')}:
        raise ValueError('source/audio song coverage mismatch')
    return dict(schema=SCHEMA, view='original', views=['original'], source_root=str(source),
                audio_catalog=str(audio_root/'catalog.json'), parser=str(parser),
                geometry_root=str(geometry_root), slide_data=str(slide_data),
                song_count=len(songs), chart_count=len(rows), versions=dict(versions),
                audio_only=audio_only, charts=rows)


def initialize(root, slide):
    global GEOMETRY
    GEOMETRY = Geometry.from_resources(root, slide)


def source_failure_locations(ir):
    """On failure only, locate every offending note without dropping any GT."""
    failures = []
    for mi, moment in enumerate(ir['moments']):
        for gi, group in enumerate(moment['groups']):
            for ni, note in enumerate(group['notes']):
                one = dict(ir, moments=[dict(moment, groups=[dict(group, notes=[note])])])
                try:
                    encode_chart(one)
                except (ChartError, ValueError) as exc:
                    failures.append(dict(moment_index=mi, group_index=gi, note_index=ni,
                                         note_id=note['id'], source=note.get('source'), reason=str(exc)))
    return failures


def build_record(ir, row):
    encoded = encode_chart(ir)
    expected = TimingMap.from_audio_timing(row['timing'])
    actual = encoded.timing
    compact = []
    for point in actual.points:
        if not compact or compact[-1][1] != point[1]:
            compact.append(point)
    if tuple(compact) != expected.points or abs(actual.first_seconds-expected.first_seconds) > 1e-9:
        raise ValueError('chart/audio timing mismatch')
    state = PhysicalState(actual, GEOMETRY, audio_end_seconds=row['audio_end_seconds'])
    source_notes = [(mi, gi, ni, n) for mi, m in enumerate(ir['moments'])
                    for gi, g in enumerate(m['groups']) for ni, n in enumerate(g['notes'])]
    token_map, actions, decisions = [], [], []
    source_note, segment, track = None, -1, 0
    for i, token in enumerate(encoded.tokens):
        c, decision = state.accept_source(token)
        s = state.grammar.state
        if isinstance(token, Note):
            source_note = source_notes[s.owner]
            segment, track = -1, 0
        elif isinstance(token, Route):
            if token.track_break is not None:
                segment, track = 0, s.track_count
            else:
                segment += 1
        location = None if isinstance(token, Moment) or source_note is None else dict(
            moment_index=source_note[0], group_index=source_note[1], note_index=source_note[2],
            note_id=source_note[3]['id'], source=source_note[3].get('source'))
        token_map.append(dict(token_index=i, tick=s.cursor, second=actual.second(s.cursor),
                              moment=s.moment, owner=None if location is None else s.owner,
                              track_index=track if isinstance(token, Route) or token.type == 'SLIDE_DURATION' else None,
                              segment_index=segment if isinstance(token, Route) else None, source=location))
        decisions.append(dict(basic_errors=list(decision.basic_errors), physical=sorted(decision.physical), deferred=decision.deferred))
        if c is not None:
            actions.append(dict(owner=c.owner, track_index=c.track_index, token_index=i,
                                duration_tick=c.track.duration_tick if c.track else c.duration_tick,
                                action=asdict(state.actions[-1])))
    state.grammar.finish()
    if state.grammar.state.owner+1 != len(source_notes):
        raise ValueError('source note count changed')
    return dict(schema=SCHEMA, view='original', source=row, metadata=encoded.metadata,
                tokens=[token_to_dict(t) for t in encoded.tokens], token_map=token_map,
                actions=actions, decisions=decisions, note_count=len(source_notes),
                physical_categories=dict(Counter(p for d in decisions for p in d['physical'])),
                basic_errors=dict(Counter(p for d in decisions for p in d['basic_errors'])))


def produce(row, output, parser, recipe):
    output = Path(output)
    status_path = output/'status'/f"{row['song_id']}-{row['difficulty']}.json"
    identity = digest(dict(row=row, recipe=recipe))
    if status_path.exists():
        old = read_json(status_path)
        if old['input_key'] != identity:
            raise ValueError(f"resume identity mismatch: {row['chart_id']}")
        if old['status'] != 'failed':
            return dict(old, resumed=True)
    result = dict(chart_id=row['chart_id'], input_key=identity, path=row['path'], view='original')
    ir = None
    stage = 'parse'
    try:
        process = subprocess.run([parser, 'dump', '--maidata', row['source_path'],
            '--difficulty', row['difficulty'], '--output', '-'], capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if process.returncode:
            result.update(status='unrepresentable', reason=process.stderr.decode('utf-8', errors='replace'),
                          parser_exit=process.returncode, source_path=row['source_path'],
                          source_header_line=row['source_header_line'])
        else:
            ir = json.loads(process.stdout)
            stage = 'encode'
            # MiaCode's maidata CLI uses global first; the accepted audio catalog
            # selects first_<difficulty> when present. Preserve that source anchor.
            ir['first_seconds'] = row['timing']['first_seconds']
            try:
                record = build_record(ir, row)
            except ChartError as exc:
                locations = source_failure_locations(ir)
                result.update(status='unrepresentable', reason=str(exc), locations=locations,
                              offending_note_count=len(locations))
            else:
                stage = 'save'
                storage = save_chart(output/row['path'], record)
                result.update(status='success', **storage, tokens=len(record['tokens']),
                              notes=record['note_count'], actions=len(record['actions']),
                              physical_categories=record['physical_categories'], basic_errors=record['basic_errors'])
    except Exception as exc:
        result.update(status='failed', stage=stage, reason=f'{type(exc).__name__}: {exc}')
    if result['status'] != 'success' and ir is not None:
        failure_path = output/'failures'/f"{row['song_id']}-{row['difficulty']}.ir.json.gz"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(failure_path, 'wt', encoding='utf-8') as stream:
            json.dump(ir, stream, ensure_ascii=False, allow_nan=False)
        result['failure_ir'] = str(failure_path)
    write_json(status_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audio-cache', type=Path, required=True)
    p.add_argument('--parser', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--slide-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--song-id', action='append')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    for name in ('source', 'audio_cache', 'parser', 'geometry', 'slide_data', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    cat = catalog(args.source, args.audio_cache, args.parser, args.geometry, args.slide_data)
    catalog_path = args.output/'catalog.json'
    if catalog_path.exists() and read_json(catalog_path) != cat:
        raise ValueError('existing catalog differs; refusing overwrite')
    write_json(catalog_path, cat)
    # Bind only producer code, not concurrently edited draft/planner documents.
    code_paths = [*Path(__file__).parent.glob('*.py')]
    for package in ('chart', 'time', 'physical'):
        code_paths.extend((Path(__file__).parents[1]/package).rglob('*.py'))
    recipe = digest({str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(code_paths)})
    rows = [r for r in cat['charts'] if not args.song_id or r['song_id'] in args.song_id]
    if args.limit is not None:
        rows = rows[:args.limit]
    started, results = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.geometry), str(args.slide_data))) as pool:
        futures = [pool.submit(produce, r, str(args.output), str(args.parser), recipe) for r in rows]
        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            if len(results) % 100 == 0 or result['status'] == 'failed':
                print(json.dumps(dict(completed=len(results), expected=len(rows), latest=result['chart_id'],
                                      counts=dict(Counter(r['status'] for r in results))), ensure_ascii=False), flush=True)
    summary = dict(expected=len(rows), source_chart_count=cat['chart_count'], song_count=cat['song_count'],
                   views=['original'], counts=dict(Counter(r['status'] for r in results)),
                   seconds=time.monotonic()-started, recipe=recipe,
                   resumed=sum(bool(r.get('resumed')) for r in results),
                   physical_categories=dict(sum((Counter(r.get('physical_categories', {})) for r in results), Counter())),
                   basic_errors=dict(sum((Counter(r.get('basic_errors', {})) for r in results), Counter())),
                   bytes=sum(r.get('bytes', 0) for r in results),
                   results=sorted(results, key=lambda r:r['chart_id']))
    write_json(args.output/'production-summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}, ensure_ascii=False), flush=True)
    return 1 if summary['counts'].get('failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
