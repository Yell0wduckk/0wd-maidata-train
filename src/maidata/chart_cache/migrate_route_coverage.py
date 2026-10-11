"""Opt-in, resumable v2 Physical attribution migration; source assets stay read-only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from maidata.chart.types import Moment, Route
from maidata.physical import Geometry
from maidata.physical.conflicts import first_route_coverage
from .cache import digest, load_chart, read_json, write_json
from .training import SCHEMA, ROUTE_COVERAGE, load_supports, save_supports


def migrate_supports(record, supports, geometry):
    """Reuse unchanged stages/actions and exact raw duration partitions.

    After a changed ROUTE stage, replay through the next MOMENT (which resets
    the ledger), including all intervening type and time projections. Merely
    OR-ing a route mask would leave later fields charging the same witness.
    """
    from maidata.decoder_training.support_cache import ChartSupports
    chart = ChartSupports(record, supports, geometry)
    for i, stage in enumerate(chart.stages, 1):
        if 'duration' in stage:
            prefix = chart.prefixes[i-1].physical
            key = (replace(prefix.grammar, token_count=0, window_end=None), prefix.actions)
            chart.counterfactuals[key] = stage['duration']
    state = chart.state
    state.restore(chart.prefixes[0])
    state.duration_support_provider = chart._duration_source
    stages, changed, replayed, dirty = [], [], 0, False
    for i, token in enumerate(chart.tokens[1:], 1):
        old = chart.stages[i-1]
        if isinstance(token, Route) and not dirty:
            p, s = state.physical, state.physical.grammar.state
            for route in range(71):
                candidate = Route(route, 0 if s.phase in ('route', 'track_done') else None)
                try:
                    grammar, _ = p._trial(candidate)
                except ValueError:
                    continue
                # No duration search is needed to discover affected prefixes.
                if (first_route_coverage(p.actions, grammar.state, p.timing, geometry)
                        and state.query(candidate).record() != old['route'][route]):
                    dirty = True
                    break
        if dirty:
            stage = dict(state.stage_and_accept(token), target_token_index=i)
            replayed += 1
        else:
            stage = old
            chart._accept_cached(i)
        if stage != old:
            changed.append(i)
        stages.append(stage)
        if isinstance(token, Moment):
            dirty = False
    terminal = (dict(state.terminal_stage(), target_token_index=len(chart.tokens))
                if dirty else chart.stages[-1])
    if terminal != chart.stages[-1]:
        changed.append(len(chart.tokens))
    stages.append(terminal)
    result = dict(supports, stages=stages, route_coverage_attribution=ROUTE_COVERAGE)
    return result, dict(changed_target_indices=changed, replayed_stages=replayed,
                        reused_stages=len(stages)-replayed-int(dirty))


def initialize(geometry, slide_data):
    global GEOMETRY
    GEOMETRY = Geometry.from_resources(geometry, slide_data)


def migrate_one(task):
    row, status, fields, source, output, recipe = task
    fields, source, output = map(Path, (fields, source, output))
    identity = digest(dict(chart_id=row['chart_id'], source_sha256=status['sha256'], recipe=recipe))
    marker = output/'status'/f"{row['song_id']}-{row['difficulty']}.json"
    if marker.exists():
        prior = read_json(marker)
        if prior['input_key'] != identity:
            raise ValueError('Physical migration resume identity differs')
        return prior
    record = load_chart(fields/row['path'])
    old = load_supports(source/row['path'])
    if old['source_sha256'] != status['source_sha256']:
        raise ValueError('Physical/source ownership differs')
    new, audit = migrate_supports(record, old, GEOMETRY)
    result = dict(status, input_key=identity, status='success', migration=audit,
                  source_physical_sha256=status['sha256'],
                  **save_supports(output/row['path'], new))
    write_json(marker, result)
    return result


def migrate(fields, source, output, geometry, slide_data, *, workers=12, chart_ids=None):
    fields, source, output = (Path(p).resolve() for p in (fields, source, output))
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (fields, source)):
        raise ValueError('migration output must be separate from source trees')
    cat, summary = read_json(source/'catalog.json'), read_json(source/'production-summary.json')
    field_cat = read_json(fields/'catalog.json')
    if cat['schema'] != SCHEMA or cat['views'] != ['original']:
        raise ValueError('requires original v2 Physical supports')
    if cat['producer_recipe'] != summary['recipe']:
        raise ValueError('Physical catalog/summary producer differs')
    if cat.get('source_producer_recipe',field_cat['producer_recipe']) != field_cat['producer_recipe']:
        raise ValueError('Physical/fields producer differs')
    field_status = {r['chart_id']:r for r in read_json(fields/'production-summary.json')['results']
                    if r['status']=='success'}
    statuses = {r['chart_id']:r for r in summary['results'] if r['status']=='success'}
    selected = set(statuses) if chart_ids is None else set(chart_ids)
    rows = [r for r in cat['charts'] if r['chart_id'] in selected]
    if selected-set(statuses) or {r['chart_id'] for r in rows} != selected:
        raise ValueError('selected chart lacks successful Physical source')
    for row in rows:
        key = row['chart_id']
        status = field_status.get(key,{})
        if (status.get('path') != row['path']
                or status.get('original_storage',status).get('sha256') != statuses[key]['source_sha256']):
            raise ValueError(f'{key}: Physical/fields source identity differs')
    # Only metadata identities, never a pre-scan of accepted payloads.
    recipe = digest(dict(migration=ROUTE_COVERAGE, source_recipe=cat['producer_recipe'],
                         source_summary=digest(summary)))
    intent = dict(fields=str(fields), source=str(source), selected=sorted(selected), recipe=recipe,
                  geometry=str(Path(geometry).resolve()), slide_data=str(Path(slide_data).resolve()))
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'migration.json'
    if marker.exists():
        if read_json(marker) != intent:
            raise ValueError('output belongs to a different Physical migration')
    elif any(output.iterdir()):
        raise FileExistsError('nonempty migration output')
    else:
        write_json(marker, intent)
    output_catalog = dict(cat, charts=rows, chart_count=len(rows), selected_chart_ids=sorted(selected),
        song_count=len({r['song_id'] for r in rows}), producer_recipe=recipe,
        source_producer_recipe=field_cat['producer_recipe'], source_cache=str(fields),
        source_physical_cache=str(source), route_coverage_attribution=ROUTE_COVERAGE,
        geometry_root=str(Path(geometry).resolve()), slide_data=str(Path(slide_data).resolve()))
    if 'versions' in cat:
        output_catalog['versions'] = dict(Counter(r['version'] for r in rows))
    write_json(output/'catalog.json', output_catalog)
    tasks = [(r, statuses[r['chart_id']], str(fields), str(source), str(output), recipe) for r in rows]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(str(geometry), str(slide_data))) as pool:
        for result in pool.map(migrate_one, tasks):
            results.append(result)
            print(f"Physical migration {len(results)}/{len(tasks)}: {result['chart_id']}", flush=True)
    results.sort(key=lambda r:r['chart_id'])
    result = dict(schema=SCHEMA, views=['original'], recipe=recipe, expected=len(rows),
                  counts=dict(success=len(results), failed=0), results=results,
                  route_coverage_attribution=ROUTE_COVERAGE)
    write_json(output/'production-summary.json', result)
    write_json(output/'training-index.json', dict(views=['original'], charts=[
        dict(chart_id=r['chart_id'], path=r['path']) for r in results]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'source-physical', 'output', 'geometry', 'slide-data'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--chart-id', action='append')
    args = parser.parse_args()
    migrate(args.fields, args.source_physical, args.output, args.geometry, args.slide_data,
            workers=args.workers, chart_ids=args.chart_id)


if __name__ == '__main__':
    main()
