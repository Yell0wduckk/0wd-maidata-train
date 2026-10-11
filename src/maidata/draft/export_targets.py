"""Export inspectable R and simai plan labels for explicitly selected cached windows."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', type=Path, required=True)
    parser.add_argument('--start-beat', type=int, action='append', required=True)
    parser.add_argument('--target-beats', type=int)
    parser.add_argument('--source-end-beat', help='explicit end-beat override; otherwise use metadata or source timing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from maidata.entry import CONFIG
    from maidata.chart_cache.cache import load_chart
    from .chart_data import from_record
    from .targets import reconstruction_target, required_query_rows
    from .plan_targets import MusicPlanTargets
    target_beats = CONFIG['target_beats'] if args.target_beats is None else args.target_beats
    chart = from_record(load_chart(args.chart), source_end_beat=args.source_end_beat)
    planner = MusicPlanTargets(chart)
    rows = []
    for start in args.start_beat:
        r = reconstruction_target(chart, start_tick=start * 96, target_beats=target_beats)
        p = planner.window(start * 96, (start + target_beats) * 96)
        rows.append(dict(start_beat=start, target_beats=target_beats,
                         reconstruction=asdict(r), planner=asdict(p), query_rows_required=len(r.rows),
                         query_cost=r.query_cost, original_time_fields=sum(t.first_time_step for t in r.rows),
                         trainable=not r.deferred_durations))
    result = dict(schema='maidata.draft-local-query-targets.v2',
                  representation='relative-tick384-continue', chart=str(args.chart), source_end_beat=str(planner.source_end_beat), windows=rows,
                  query_rows_required=required_query_rows([r['query_rows_required'] for r in rows]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
