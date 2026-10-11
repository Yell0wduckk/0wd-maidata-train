"""Low-cost, resumable v4 -> v5 migration, grouped once per source chart."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
import argparse
import json
import random
import time
import torch

from maidata.chart.codec import token_from_dict
from maidata.chart_cache.cache import load_chart, write_json
from maidata.time import TimingMap
from maidata.decoder.sequence import expand_inputs
from maidata.decoder.batching import pack_tokens
from maidata.decoder.windows import real_onsets, window_from_config
from .sequence_cache import expand_window, SCHEMA, SEQUENCE_SCHEMA
from .window_cache import tensor_pack, tensor_unpack
from .plan import DecoderPlan
from maidata.training.checkpoint import digest_file


def initialize(config, source, output):
    global CONFIG, SOURCE, OUTPUT
    torch.set_num_threads(1)
    CONFIG, SOURCE, OUTPUT = config, Path(source), Path(output)


def save_value(path,value):
    from .sequence_cache import write_asset
    return write_asset(path,value,OUTPUT)


def migrate_chart(task):
    key, chart, rows, previous, audio_end = task
    marker = OUTPUT/'completed-charts'/f'{key.replace(":","-")}.json'
    if marker.exists():
        return json.loads(marker.read_text('utf-8'))
    record = load_chart(Path(CONFIG['decoder_train_chart_cache'])/chart['chart_cache_path'])
    tokens = tuple(token_from_dict(t) for t in record['tokens'])
    timing = TimingMap.from_ir(record['metadata'])
    onsets = real_onsets(tokens,timing,sequence_start_tick=record['start_tick'])
    from .sequence_cache import chart_sequence
    sequence=dict(chart_sequence(record),source_chart=chart['chart_cache_path'])
    seq = save_value(OUTPUT/'sequences'/f'{key.replace(":","-")}.pt',sequence)
    result = dict(chart_id=key,sequence=seq,windows={},physical_recomputed=0)
    for row in rows:
        identity = row['window_id']; old = previous[identity]
        value = tensor_unpack(torch.load(SOURCE/old['path'],map_location='cpu',weights_only=True))
        example = window_from_config(tokens,timing,CONFIG,sequence_start_tick=record['start_tick'],
            start_tick=row['start_tick'],audio_end_seconds=audio_end,training=False,rng=random.Random(0),onsets=onsets)
        new = expand_window(value,example,timing)
        name = sha256(identity.encode()).hexdigest()+'.pt'
        storage = save_value(OUTPUT/'windows'/name,new)
        result['windows'][identity] = dict(**{k:old[k] for k in ('chart_id','audio_variant','start_tick','end_tick')},
            **storage,lengths=new['metadata']['model_lengths'],logical_lengths=old['lengths'],
            duration_window_crossings=old.get('duration_window_crossings',[]))
    write_json(marker,result)
    return result


def migrate(config, source, output, *, workers=16, window_ids=None):
    source, output = Path(source).resolve(),Path(output).resolve()
    if source == output:
        raise ValueError('migration output must be separate')
    prior = json.loads((source/'manifest.json').read_text('utf-8-sig'))
    if prior['schema'] != 'maidata.decoder-window-tensors.v4' or not prior.get('complete'):
        raise ValueError('requires completed exact v4 source')
    selected = set(prior['windows']) if window_ids is None else set(window_ids)
    if not selected <= prior['windows'].keys():
        raise ValueError('unknown source windows')
    output.mkdir(parents=True,exist_ok=True)
    intent = dict(schema=SCHEMA,source=str(source),config=config,windows=sorted(selected))
    intent_path = output/'migration.json'
    if intent_path.exists():
        if json.loads(intent_path.read_text('utf-8')) != intent:
            raise ValueError('output belongs to a different migration')
    elif any(output.iterdir()):
        raise FileExistsError('nonempty destination without migration identity')
    else:
        write_json(intent_path,intent)
    tasks = []
    for side in ('train','validation'):
        plan = DecoderPlan(config[f'decoder_train_{side}_plan'],config,training=side=='train')
        for key,chart in plan.charts.items():
            rows = [r for r in plan.windows if r['chart_id']==key and r['window_id'] in selected]
            if rows:
                for row in rows:
                    if any(row[k]!=prior['windows'][row['window_id']][k] for k in ('chart_id','audio_variant','start_tick','end_tick')):
                        raise ValueError('window membership or geometry differs')
                tasks.append((key,chart,rows,{r['window_id']:prior['windows'][r['window_id']] for r in rows},
                    plan.groups[rows[0]['audio_variant']]['audio_end_seconds']))
    if {r['window_id'] for t in tasks for r in t[2]} != selected:
        raise ValueError('selected source not covered by supplied plans')
    (output/'completed-charts').mkdir(exist_ok=True)
    start = time.perf_counter()
    manifest = dict(schema=SCHEMA,complete=False,contract=dict(prior['contract'],
        time_representation='causal-token-tick384-continue',physical_objective='gt-prefix-step-mean'),
        windows={},sequences={},source_windows=str(source),physical_recomputed=0)
    write_json(output/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(config,str(source),str(output))) as pool:
        pending = [pool.submit(migrate_chart,task) for task in tasks]
        for n,future in enumerate(as_completed(pending),1):
            result = future.result()
            manifest['windows'].update(result['windows'])
            manifest['sequences'][result['chart_id']] = result['sequence']
            print(json.dumps(dict(charts=n,total_charts=len(tasks),windows=len(manifest['windows']))),flush=True)
    manifest.update(complete=True,seconds=time.perf_counter()-start,workers=workers)
    manifest['logical_rows'] = sum(sum(r['logical_lengths']) for r in manifest['windows'].values())
    manifest['model_rows'] = sum(sum(r['lengths']) for r in manifest['windows'].values())
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    from maidata.entry import CONFIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-windows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--window-id',action='append')
    args=parser.parse_args()
    config=dict(CONFIG,**json.loads(args.config.read_text('utf-8-sig')))
    result=migrate(config,args.source_windows,args.output,workers=args.workers,window_ids=args.window_id)
    print(json.dumps({k:result[k] for k in ('complete','seconds','workers','logical_rows','model_rows','physical_recomputed')}))


if __name__=='__main__':
    main()
