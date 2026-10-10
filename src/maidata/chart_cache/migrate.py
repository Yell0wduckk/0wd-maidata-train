"""Migrate accepted original v1 records, then compute new training supports.

This consumes the already transferred original cache, including preserved IR
for unrepresentable charts. It never reparses/recomputes audio or mirrors GT.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

from maidata.chart import EncodedChart, decode_chart, token_from_dict
from maidata.chart.types import ChartError
from . import produce
from .cache import SCHEMA, digest, read_json, write_json, save_chart
from .training import build_supports, save_supports, SCHEMA as TRAINING_SCHEMA


def convert(old, geometry):
    """Explicit legacy adapter: END anchor is metadata, WiFi moves to ROUTE."""
    if old['schema'] != 'maidata.full-chart-original.v1' or old['view'] != 'original':
        raise ValueError('expected accepted original v1 input')
    raw = old['tokens']
    if raw[0]['type'] != 'END':
        raise ValueError('legacy anchor missing')
    anchor = raw[0]['start_tick']
    tokens = [token_from_dict(dict(type='MOMENT',delta_ms=0))]
    for i,t in enumerate(raw[1:],1):
        t = dict(t)
        if t['type'] == 'MOMENT' and t['delta_ms'] == 0 and i == 1:
            continue
        if t['type'] == 'NOTE' and t['kind'] == 'wifi':
            t['kind'] = 'slide'
        tokens.append(token_from_dict(t))
    ir = decode_chart(EncodedChart(tuple(tokens),old['metadata'],anchor),geometry)
    locations = [m['source'] for t,m in zip(old['tokens'],old['token_map']) if t['type'] == 'NOTE']
    notes = [n for m in ir['moments'] for g in m['groups'] for n in g['notes']]
    if len(notes) != len(locations):
        raise ValueError('migration changed note count')
    for n,loc in zip(notes,locations):
        n['id'], n['source'] = loc['note_id'], loc['source']
        n['original_location'] = loc
    return ir


def one(row, old_status, source, output, physical, recipe):
    source,output,physical = Path(source),Path(output),Path(physical)
    status_path = physical/'status'/f"{row['song_id']}-{row['difficulty']}.json"
    identity = digest(dict(row=row,source_key=old_status['input_key'],recipe=recipe))
    if status_path.exists():
        old_result = read_json(status_path)
        if old_result['input_key'] != identity:
            raise ValueError('resume identity mismatch')
        if old_result['status'] != 'failed':
            return dict(old_result,resumed=True)
    result = dict(chart_id=row['chart_id'],view='original',input_key=identity,path=row['path'])
    started = time.monotonic()
    ir = None
    try:
        if old_status['status'] == 'success':
            with gzip.open(source/row['path'],'rt',encoding='utf-8') as stream:
                old = json.load(stream)
            ir = convert(old,produce.GEOMETRY)
        else:
            failure = source/'failures'/f"{row['song_id']}-{row['difficulty']}.ir.json.gz"
            if not failure.exists():
                result.update(status='unrepresentable',reason=old_status['reason'])
            else:
                with gzip.open(failure,'rt',encoding='utf-8') as stream:
                    ir = json.load(stream)
        if ir is not None:
            try:
                record = produce.build_record(ir,row)
            except ChartError as exc:
                result.update(status='unrepresentable',reason=str(exc),locations=produce.source_failure_locations(ir))
            else:
                original_storage = save_chart(output/row['path'],record)
                supports = build_supports(record,produce.GEOMETRY)
                supports['source_sha256'] = original_storage['sha256']
                storage = save_supports(physical/row['path'],supports)
                result.update(status='success',**storage,original_storage=original_storage,
                              tokens=len(record['tokens']),notes=record['note_count'],actions=len(record['actions']),
                              stages=len(supports['stages']),
                              time_intervals=sum(len(s.get(k,{}).get('intervals',[])) for s in supports['stages'] for k in ('moment','duration')),
                              physical_categories=record['physical_categories'],basic_errors=record['basic_errors'])
    except Exception as exc:
        result.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    if result['status'] == 'unrepresentable' and ir is not None:
        path = output/'failures'/f"{row['song_id']}-{row['difficulty']}.ir.json.gz"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(ir,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    result['seconds'] = time.monotonic()-started
    write_json(status_path,result)
    write_json(output/'status'/status_path.name,result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','physical-output','geometry','slide-data'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--limit',type=int)
    args = p.parse_args()
    cat = read_json(args.source/'catalog.json')
    summary = read_json(args.source/'production-summary.json')
    statuses = {r['chart_id']:r for r in summary['results']}
    paths = [p for package in ('chart_cache','chart','time','physical')
             for p in (Path(__file__).parents[1]/package).rglob('*.py')]
    recipe = digest({str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    for root,schema in ((args.output,SCHEMA),(args.physical_output,TRAINING_SCHEMA)):
        value = dict(cat,schema=schema,producer_recipe=recipe,source_cache=str(args.source.resolve()),
                     geometry_root=str(args.geometry.resolve()),slide_data=str(args.slide_data.resolve()))
        dest = root/'catalog.json'
        if dest.exists() and read_json(dest) != value:
            raise ValueError('existing output catalog identity mismatch')
        write_json(dest,value)
    rows = cat['charts'][:args.limit] if args.limit else cat['charts']
    started,results = time.monotonic(),[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=produce.initialize,
                             initargs=(str(args.geometry),str(args.slide_data))) as pool:
        jobs = [pool.submit(one,r,statuses[r['chart_id']],str(args.source),str(args.output),str(args.physical_output),recipe) for r in rows]
        for job in as_completed(jobs):
            r = job.result()
            results.append(r)
            if len(results)%25 == 0 or r['status'] == 'failed':
                print(json.dumps(dict(completed=len(results),expected=len(rows),counts=dict(Counter(r['status'] for r in results)),latest=r['chart_id'])),flush=True)
    final = dict(expected=len(rows),source_chart_count=cat['chart_count'],views=['original'],recipe=recipe,
                 seconds=time.monotonic()-started,counts=dict(Counter(r['status'] for r in results)),
                 totals={k:sum(r.get(k,0) for r in results) for k in ('tokens','notes','actions','stages','time_intervals','bytes')},
                 results=sorted(results,key=lambda r:r['chart_id']))
    write_json(args.output/'production-summary.json',dict(final,schema=SCHEMA))
    write_json(args.physical_output/'production-summary.json',dict(final,schema=TRAINING_SCHEMA))
    print(json.dumps({k:v for k,v in final.items() if k != 'results'}),flush=True)
    return int(bool(final['counts'].get('failed')))


if __name__ == '__main__':
    raise SystemExit(main())
