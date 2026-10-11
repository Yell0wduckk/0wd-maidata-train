"""Measure independent R query/substep cost on selected base windows."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from maidata.chart_cache.cache import load_chart
from .chart_data import from_record
from .schema import ROW_TYPES
from .targets import reconstruction_target, required_query_rows


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _summary(rows):
    lengths = sorted(row['query_rows'] for row in rows)
    return dict(charts=len({row['chart_id'] for row in rows}), windows=len(rows),
                empty_windows=sum(not any(row['moment_counts']) for row in rows),
                max=required_query_rows(lengths),
                p50=lengths[math.ceil(.5 * len(lengths)) - 1],
                p95=lengths[math.ceil(.95 * len(lengths)) - 1],
                p99=lengths[math.ceil(.99 * len(lengths)) - 1])


def measure_capacity(plan_path, split_path, *, chart_cache=None):
    """Reuse the base-window plan; repeated occurrences do not change capacity."""
    plan_path, split_path = Path(plan_path), Path(split_path)
    plan, split = _read(plan_path), _read(split_path)
    selection = _read(plan['sources']['selection'])
    charts = {row['chart_id']: row for row in selection['charts']}
    owners = {}
    for side in ('train', 'validation'):
        for chart_id in split[side]['chart_ids']:
            if chart_id in owners:
                raise ValueError('a chart cannot belong to both data splits')
            owners[chart_id] = side
    if set(owners) != set(charts):
        raise ValueError('split and selected charts must match')
    windows = defaultdict(list)
    with plan_path.with_name('windows.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['chart_id'] in owners:
                windows[row['chart_id']].append(row)
    if set(windows) != set(owners):
        raise ValueError('every selected chart needs its base windows')
    cache = Path(chart_cache or plan['sources']['chart_cache'])
    target_beats = plan['config']['target_beats']
    rows = []
    for chart_id, selected in charts.items():
        record = load_chart(cache / selected['chart_cache_path'])
        if record['source']['chart_id'] != chart_id:
            raise ValueError('chart cache identity mismatch')
        chart = from_record(record)
        for window in windows[chart_id]:
            if window['end_tick'] - window['start_tick'] != target_beats * 96:
                raise ValueError('window length differs from the sampling plan')
            target = reconstruction_target(chart, start_tick=window['start_tick'], target_beats=target_beats)
            counts = Counter(row.query.role for row in target.rows)
            rows.append(dict(window_id=window['window_id'], chart_id=chart_id, split=owners[chart_id],
                             start_tick=window['start_tick'], end_tick=window['end_tick'],
                             query_rows=len(target.rows), query_cost=target.query_cost,
                             original_time_fields=sum(r.first_time_step for r in target.rows),
                             moment_counts=target.moment_counts, query_roles=dict(counts),
                             truncated_time_fields=sum(r.first_time_step and r.closed_by_continue
                                 and r.query.role != 'MOMENT' for r in target.rows)))
    summary = _summary(rows)
    report = dict(schema='maidata.draft-query-cost.v2', plan=str(plan_path.resolve()),
                  split=str(split_path.resolve()), chart_cache=str(cache.resolve()),
                  target_beats=target_beats, target_stride_beats=plan['config']['target_stride_beats'],
                  representation='relative-tick384-continue', max_query_substeps=summary['max'], all=summary,
                  splits={side: _summary([r for r in rows if r['split'] == side])
                          for side in ('train', 'validation')},
                  max_windows=[r for r in rows if r['query_rows'] == summary['max']])
    return report, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='existing plan.json with sibling windows.jsonl')
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--chart-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report, rows = measure_capacity(args.plan, args.split, chart_cache=args.chart_cache)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'capacity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'window-lengths.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
