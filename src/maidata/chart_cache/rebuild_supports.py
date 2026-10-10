"""Rebuild only training Physical supports from accepted original v2 fields."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path

from .cache import SCHEMA as FIELD_SCHEMA, digest, load_chart, read_json, write_json
from .produce import initialize
from . import produce
from .training import SCHEMA, build_supports, save_supports


def rebuild_one(row, source_status, source, output, recipe):
    """Keep fields read-only; the new status identifies their actual producer."""
    source, output = Path(source), Path(output)
    storage = source_status.get('original_storage', source_status)
    source_sha = storage['sha256']
    identity = digest(dict(chart_id=row['chart_id'], source_sha256=source_sha, recipe=recipe))
    status_path = output/'status'/f"{row['song_id']}-{row['difficulty']}.json"
    if status_path.exists():
        previous = read_json(status_path)
        if previous['input_key'] != identity:
            raise ValueError('support resume identity mismatch')
        if previous['status'] == 'success':
            return dict(previous, resumed=True)
    result = dict(chart_id=row['chart_id'], path=row['path'], view='original',
                  input_key=identity, source_sha256=source_sha)
    try:
        record = load_chart(source/row['path'])
        supports = build_supports(record, produce.GEOMETRY)
        supports['source_sha256'] = source_sha
        result.update(status='success', **save_supports(output/row['path'], supports),
                      tokens=supports['token_count'], stages=len(supports['stages']))
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    write_json(status_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'geometry', 'slide-data'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--chart-id', action='append', help='Rebuild only these chart IDs; repeatable.')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Physical output must be separate from accepted fields')
    cat = read_json(args.source/'catalog.json')
    if cat['schema'] != FIELD_SCHEMA or cat['views'] != ['original']:
        raise ValueError('expected accepted original v2 field catalog')
    summary = read_json(args.source/'production-summary.json')
    statuses = {r['chart_id']:r for r in summary['results'] if r['status'] == 'success'}
    selected = set(args.chart_id) if args.chart_id is not None else set(statuses)
    if selected-set(statuses):
        raise ValueError('requested chart lacks successful source fields')
    rows = [r for r in cat['charts'] if r['chart_id'] in selected]
    if {r['chart_id'] for r in rows} != selected:
        raise ValueError('source summary/catalog chart mismatch')
    package = Path(__file__).parents[1]
    paths = sorted(p for name in ('chart', 'time', 'physical') for p in (package/name).rglob('*.py'))
    paths += [Path(__file__), Path(__file__).with_name('training.py'), Path(__file__).with_name('cache.py')]
    recipe = digest({str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output_catalog = dict(cat, schema=SCHEMA, charts=rows, chart_count=len(rows),
                          song_count=len({r['song_id'] for r in rows}),
                          versions=dict(Counter(r['version'] for r in rows)),
                          selected_chart_ids=sorted(selected),
                          source_cache=str(args.source.resolve()), producer_recipe=recipe,
                          source_producer_recipe=cat.get('producer_recipe'),
                          geometry_root=str(args.geometry.resolve()), slide_data=str(args.slide_data.resolve()))
    destination = args.output/'catalog.json'
    if destination.exists() and read_json(destination) != output_catalog:
        raise ValueError('existing support catalog identity mismatch')
    write_json(destination, output_catalog)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.geometry), str(args.slide_data))) as pool:
        jobs = [pool.submit(rebuild_one, row, statuses[row['chart_id']], args.source, args.output, recipe)
                for row in rows]
        for job in as_completed(jobs):
            results.append(job.result())
    results.sort(key=lambda r:r['chart_id'])
    failed = sum(r['status'] == 'failed' for r in results)
    write_json(args.output/'production-summary.json', dict(schema=SCHEMA, views=['original'],
               recipe=recipe, expected=len(rows), counts=dict(success=len(rows)-failed, failed=failed), results=results))
    write_json(args.output/'training-index.json', dict(views=['original'], charts=[
               dict(chart_id=r['chart_id'], path=r['path']) for r in results if r['status'] == 'success']))
    print(f'Physical supports: {len(rows)-failed} success, {failed} failed')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
