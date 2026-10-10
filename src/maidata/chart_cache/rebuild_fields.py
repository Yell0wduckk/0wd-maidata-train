"""Rebuild only original v2 fields/actions from accepted v1 cache and failure IR."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

from maidata.chart.types import ChartError
from . import produce
from .cache import SCHEMA, digest, read_json, save_chart, write_json
from .migrate import convert, select_master_remaster


def one(row, old_status, source, output, recipe):
    source, output = Path(source), Path(output)
    status_path = output/'status'/f"{row['song_id']}-{row['difficulty']}.json"
    identity = digest(dict(row=row, source_key=old_status['input_key'], recipe=recipe))
    if status_path.exists():
        old = read_json(status_path)
        if old['input_key'] != identity:
            raise ValueError('field cache resume identity mismatch')
        if old['status'] != 'failed':
            return dict(old, resumed=True)
    result = dict(chart_id=row['chart_id'], input_key=identity,
                  path=row['path'], view='original')
    ir = None
    try:
        if old_status['status'] == 'success':
            with gzip.open(source/row['path'], 'rt', encoding='utf-8') as stream:
                ir = convert(json.load(stream), produce.GEOMETRY)
        else:
            failure = source/'failures'/f"{row['song_id']}-{row['difficulty']}.ir.json.gz"
            if failure.exists():
                with gzip.open(failure, 'rt', encoding='utf-8') as stream:
                    ir = json.load(stream)
            else:
                result.update(status='unrepresentable', reason=old_status['reason'])
        if ir is not None:
            try:
                record = produce.build_record(ir, row)
            except ChartError as exc:
                locations = produce.source_failure_locations(ir)
                result.update(status='unrepresentable', reason=str(exc),
                              locations=locations, offending_note_count=len(locations))
            else:
                storage = save_chart(output/row['path'], record)
                result.update(status='success', **storage, tokens=len(record['tokens']),
                              notes=record['note_count'], actions=len(record['actions']),
                              physical_categories=record['physical_categories'],
                              basic_errors=record['basic_errors'])
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    if result['status'] == 'unrepresentable' and ir is not None:
        relative = f"failures/{row['song_id']}-{row['difficulty']}.ir.json.gz"
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(ir, ensure_ascii=False,
                                                  allow_nan=False).encode(), mtime=0))
        result['failure_ir'] = relative
    write_json(status_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'geometry', 'slide-data'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    cat = read_json(args.source/'catalog.json')
    if cat['schema'] != 'maidata.full-chart-original.v1':
        raise ValueError('expected accepted v1 source cache')
    statuses = {r['chart_id']: r for r in read_json(args.source/'production-summary.json')['results']}
    rows, selection = select_master_remaster(cat)
    package = Path(__file__).parents[1]
    code = [Path(__file__), Path(__file__).with_name('migrate.py'),
            Path(__file__).with_name('produce.py'), Path(__file__).with_name('cache.py')]
    code += [p for name in ('chart', 'time', 'physical') for p in (package/name).rglob('*.py')]
    recipe = digest({str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(code)})
    value = dict(cat, schema=SCHEMA, charts=rows, chart_count=len(rows),
                 song_count=selection['selected_songs'], audio_only=[],
                 versions=dict(Counter(r['version'] for r in rows)), selection=selection,
                 producer_recipe=recipe, source_cache=str(args.source.resolve()),
                 geometry_root=str(args.geometry.resolve()), slide_data=str(args.slide_data.resolve()))
    dest = args.output/'catalog.json'
    if dest.exists() and read_json(dest) != value:
        raise ValueError('existing output catalog identity mismatch')
    write_json(dest, value)
    write_json(args.output/'selection.json', selection)
    started, results = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=produce.initialize,
                             initargs=(str(args.geometry), str(args.slide_data))) as pool:
        jobs = [pool.submit(one, r, statuses[r['chart_id']], str(args.source),
                            str(args.output), recipe) for r in rows]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            if len(results) % 100 == 0 or result['status'] == 'failed':
                print(json.dumps(dict(completed=len(results), expected=len(rows),
                      counts=dict(Counter(r['status'] for r in results)), latest=result['chart_id'])), flush=True)
    results.sort(key=lambda r: r['chart_id'])
    counts = dict(Counter(r['status'] for r in results))
    summary = dict(schema=SCHEMA, expected=len(rows), source_chart_count=cat['chart_count'],
                   song_count=selection['selected_songs'], views=['original'], recipe=recipe,
                   counts=counts, seconds=time.monotonic()-started,
                   training_supports_computed=False, results=results,
                   totals={k: sum(r.get(k, 0) for r in results) for k in ('tokens', 'notes', 'actions', 'bytes')})
    write_json(args.output/'production-summary.json', summary)
    quarantined = [r for r in results if r['status'] == 'unrepresentable']
    write_json(args.output/'quarantine.json', dict(count=len(quarantined),
               reasons=dict(Counter(r['reason'] for r in quarantined)), charts=quarantined))
    write_json(args.output/'training-index.json', dict(schema=SCHEMA, views=['original'],
               charts=[dict(chart_id=r['chart_id'], path=r['path']) for r in results if r['status'] == 'success']))
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}), flush=True)
    return int(bool(counts.get('failed')))


if __name__ == '__main__':
    raise SystemExit(main())
