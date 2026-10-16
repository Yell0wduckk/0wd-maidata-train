"""Opt-in v5 window migration: replace Physical penalties, preserve every model/GT tensor."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import torch

from maidata.chart_cache.cache import digest, load_chart, read_json, write_json
from maidata.chart_cache.training import ROUTE_COVERAGE, SLIDE_LAUNCH, load_supports
from maidata.decoder.windows import real_onsets
from maidata.physical import Geometry
from .export_windows import compile_window
from .sequence_cache import SCHEMA, write_asset
from .support_cache import ChartSupports
from .window_cache import tensor_unpack, CONFIG_KEYS


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def replace_physical(old, compiled):
    """A migration must not silently adopt a changed sampler or label compiler."""
    for key in ('tokens', 'next_types', 'history'):
        if not equal(old[key], compiled[key]):
            raise ValueError(f'Physical migration changes {key}')
    before, after = old['labels'], compiled['labels']
    if (before is None or not equal(before['counts'],after['counts'])
            or not equal(before['flags'],after['flags']) or before['groups'].keys()!=after['groups'].keys()):
        raise ValueError('Physical migration changes label inventory')
    groups, changed = {}, False
    for key, group in before['groups'].items():
        new = after['groups'][key]
        if not equal({k:v for k,v in group.items() if k!='penalty'},
                     {k:v for k,v in new.items() if k!='penalty'}):
            raise ValueError(f'Physical migration changes non-Physical labels: {key}')
        changed |= not equal(group['penalty'],new['penalty'])
        groups[key] = dict(group, penalty=new['penalty'])
    ignored = {'support_reuse', 'route_coverage_attribution', 'slide_launch_attribution'}
    if not equal({k:v for k,v in old['metadata'].items() if k not in ignored},
                 {k:v for k,v in compiled['metadata'].items() if k not in ignored}):
        raise ValueError('Physical migration changes window metadata')
    return dict(old, labels=dict(before, groups=groups), metadata=dict(old['metadata'],
        route_coverage_attribution=ROUTE_COVERAGE, slide_launch_attribution=SLIDE_LAUNCH)), changed


def initialize(config, source, output):
    global CONFIG, SOURCE, OUTPUT, GEOMETRY
    torch.set_num_threads(1)
    CONFIG, SOURCE, OUTPUT = config, Path(source), Path(output)
    GEOMETRY = Geometry.from_resources(config['decoder_train_geometry'],config['decoder_train_slide_data'])


def migrate_chart(task):
    key, relative, rows, source_sha = task
    marker = OUTPUT/'completed-charts'/f'{key.replace(":","-")}.json'
    if marker.exists():
        return read_json(marker)
    record = load_chart(Path(CONFIG['decoder_train_chart_cache'])/relative)
    supports = load_supports(Path(CONFIG['decoder_train_physical_cache'])/relative)
    if supports['source_sha256'] != source_sha:
        raise ValueError('window migration Physical/source identity differs')
    # This explicit migration needs the new attribution. Ordinary old-cache
    # readers deliberately have no such check and continue to work unchanged.
    if (supports.get('route_coverage_attribution') != ROUTE_COVERAGE
            or supports.get('slide_launch_attribution') != SLIDE_LAUNCH):
        raise ValueError('migrate Physical supports before migrating windows')
    chart = ChartSupports(record, supports, GEOMETRY)
    onsets = real_onsets(chart.tokens,chart.timing,sequence_start_tick=chart.start_tick)
    result = dict(chart_id=key, windows={}, changed_windows=0)
    for identity, row in rows:
        old = tensor_unpack(torch.load(SOURCE/row['path'],map_location='cpu',weights_only=True))
        compiled = compile_window(chart, dict(row,window_id=identity), CONFIG,
                                  record['source']['audio_end_seconds'],onsets)
        value, changed = replace_physical(old,compiled)
        result['windows'][identity] = dict(row, **write_asset(OUTPUT/row['path'],value,OUTPUT))
        result['changed_windows'] += int(changed)
    write_json(marker,result)
    return result


def migrate(config, source, output, *, workers=12, window_ids=None):
    source, output = Path(source).resolve(),Path(output).resolve()
    inputs = [source,Path(config['decoder_train_chart_cache']).resolve(),
              Path(config['decoder_train_physical_cache']).resolve()]
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError('window output must be separate from source trees')
    prior = read_json(source/'manifest.json')
    if prior['schema']!=SCHEMA or not prior.get('complete'):
        raise ValueError('requires completed v5 windows; migrate earlier sequence formats separately')
    if any(prior['contract']['config'][k]!=config[k] for k in CONFIG_KEYS):
        raise ValueError('window geometry/config must remain unchanged')
    selected = set(prior['windows']) if window_ids is None else set(window_ids)
    if not selected <= prior['windows'].keys():
        raise ValueError('unknown source window')
    physical = Path(config['decoder_train_physical_cache'])
    catalog, summary = read_json(physical/'catalog.json'),read_json(physical/'production-summary.json')
    fields = Path(config['decoder_train_chart_cache'])
    field_catalog = read_json(fields/'catalog.json')
    field_statuses = {r['chart_id']:r for r in read_json(fields/'production-summary.json')['results']
                      if r['status']=='success'}
    if (catalog['producer_recipe']!=summary['recipe']
            or catalog['source_producer_recipe']!=field_catalog['producer_recipe']):
        raise ValueError('window migration Physical/fields producer differs')
    rows = {r['chart_id']:r for r in catalog['charts']}
    successful = {r['chart_id']:r for r in summary['results'] if r['status']=='success'}
    grouped = defaultdict(list)
    for identity in sorted(selected):
        row = prior['windows'][identity]
        if row['chart_id'] not in successful or row['chart_id'] not in rows:
            raise ValueError('selected window lacks migrated Physical support')
        key = row['chart_id']
        status = field_statuses.get(key,{})
        if (status.get('path')!=rows[key]['path']
                or status.get('original_storage',status).get('sha256')!=successful[key]['source_sha256']):
            raise ValueError(f'{key}: window migration Physical/fields identity differs')
        grouped[row['chart_id']].append((identity,row))
    intent = dict(source=str(source), source_manifest=digest(prior), config=config,
                  physical_catalog=digest(catalog), physical_summary=digest(summary), windows=sorted(selected))
    output.mkdir(parents=True,exist_ok=True)
    marker = output/'migration.json'
    if marker.exists():
        if read_json(marker)!=intent:
            raise ValueError('output belongs to a different window migration')
    elif any(output.iterdir()):
        raise FileExistsError('nonempty migration output')
    else:
        write_json(marker,intent)
    from maidata.training.checkpoint import digest_file
    inputs_binding = dict(prior['contract']['inputs'],physical={
        name:digest_file(physical/name) for name in ('catalog.json','production-summary.json')})
    manifest = dict(prior, complete=False, windows={}, sequences={},
        source_windows=str(source), route_coverage_attribution=ROUTE_COVERAGE, slide_launch_attribution=SLIDE_LAUNCH,
        contract=dict(prior['contract'],inputs=inputs_binding), changed_windows=0)
    write_json(output/'manifest.json',manifest)
    tasks = [(k,rows[k]['path'],v,successful[k]['source_sha256']) for k,v in grouped.items()]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,
                             initargs=(config,str(source),str(output))) as pool:
        for n,result in enumerate(pool.map(migrate_chart,tasks),1):
            manifest['windows'].update(result['windows'])
            manifest['changed_windows'] += result['changed_windows']
            print(f'Physical windows {n}/{len(tasks)}',flush=True)
    # Accepted sequence assets do not contain Physical. Copy bytes without
    # reparsing/rebuilding or rehashing them; retain their original identities.
    for key in grouped:
        storage = prior['sequences'][key]
        path = output/storage['path']
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/storage['path'],path)
        manifest['sequences'][key] = storage
    manifest.update(complete=True,selected_window_ids=sorted(selected))
    for name,key in (('logical_rows','logical_lengths'),('model_rows','lengths')):
        if name in manifest:
            manifest[name] = sum(sum(r[key]) for r in manifest['windows'].values())
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    from maidata.entry import CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','source-windows','physical-cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--window-id',action='append')
    args = parser.parse_args()
    config = dict(CONFIG,**read_json(args.config))
    config['decoder_train_physical_cache'] = str(args.physical_cache.resolve())
    migrate(config,args.source_windows,args.output,workers=args.workers,window_ids=args.window_id)


if __name__=='__main__':
    main()
