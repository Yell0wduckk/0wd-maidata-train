"""Export compact window tensors directly from accepted field/Physical intervals."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import torch

from maidata.chart_cache.cache import load_chart, write_json
from maidata.chart_cache.training import load_supports
from maidata.decoder.batching import collate_windows
from maidata.decoder.windows import real_onsets, window_from_config
from maidata.physical import Geometry
from maidata.training.checkpoint import digest_file
from .support_cache import ChartSupports
from .supervision import prepare
from .window_cache import SCHEMA, contract, tensor_pack


def compile_window(chart, row, config, audio_end, onsets):
    example = window_from_config(chart.tokens, chart.timing, config, sequence_start_tick=chart.start_tick,
        start_tick=row['start_tick'], audio_end_seconds=audio_end, training=False, rng=random.Random(0), onsets=onsets)
    if example.end_tick != row['end_tick']:
        raise ValueError('window geometry differs')
    stages, _ = chart.window(example)
    tokens, types, _ = collate_windows([example], device='cpu')
    labels = prepare([dict(example=example, stages=stages, timing=chart.timing, audio_end=audio_end)])
    # Store the shared supervision once; history dropout changes only input and
    # row offsets, never real Physical state or the target decision supports.
    for group in labels.groups.values():
        group['indices'][:, 1] -= example.boundary_index
    labels.flags['indices'][:, 1] -= example.boundary_index
    return dict(tokens={f.name: getattr(tokens, f.name)[0] for f in fields(tokens)}, next_types=types[0],
        history=example.boundary_index, labels=dict(groups=labels.groups, counts=labels.counts, flags=labels.flags))


def initialize(config, hashes):
    global CONFIG, HASHES, GEOMETRY
    torch.set_num_threads(1)
    CONFIG, HASHES = config, hashes
    GEOMETRY = Geometry.from_resources(config['decoder_train_geometry'], config['decoder_train_slide_data'])


def export_chart(task):
    key, relative, rows, audio_end = task
    c = CONFIG
    record = load_chart(Path(c['decoder_train_chart_cache']) / relative)
    support = load_supports(Path(c['decoder_train_physical_cache']) / relative)
    if (record['source']['chart_id'] != key or support['source_sha256'] != HASHES[key]
            or any(r['audio_variant'] != record['source']['audio_variant'] for r in rows)):
        raise ValueError('field/Physical ownership differs')
    chart = ChartSupports(record, support, GEOMETRY)
    onsets = real_onsets(chart.tokens, chart.timing, sequence_start_tick=chart.start_tick)
    result = {}
    for row in rows:
        value = compile_window(chart, row, c, audio_end, onsets)
        payload = tensor_pack(value)
        name = sha256(row['window_id'].encode()).hexdigest()+'.pt'
        path = Path(c['decoder_train_window_cache']) / 'windows' / name
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        # Acceptance of this newly generated asset happens here, once.
        saved = torch.load(temporary, map_location='cpu', weights_only=True)
        if (saved['layout'] != payload['layout'] or saved['buffers'].keys() != payload['buffers'].keys()
                or any(not torch.equal(v, saved['buffers'][k]) for k, v in payload['buffers'].items())):
            raise ValueError('window tensor storage roundtrip differs')
        temporary.replace(path)
        result[row['window_id']] = dict(
            **{k: row[k] for k in ('chart_id','audio_variant','start_tick','end_tick')},
            lengths=[len(value['next_types'])-value['history'], value['history']],
            path='windows/'+name, bytes=path.stat().st_size, sha256=digest_file(path))
    return result


def export(config, *, workers=12):
    from .plan import DecoderPlan
    from .checkpoint import data_binding
    root = Path(config['decoder_train_window_cache'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('window output must be new or empty')
    plans = [DecoderPlan(config['decoder_train_'+name], config, training=i==0)
             for i, name in enumerate(('train_plan','validation_plan'))]
    binding, hashes = data_binding(plans, config)
    tasks = []
    for plan in plans:
        for key, chart in plan.charts.items():
            rows = [w for w in plan.windows if w['chart_id'] == key]
            tasks.append((key, chart['chart_cache_path'], rows,
                          plan.groups[rows[0]['audio_variant']]['audio_end_seconds']))
    (root/'windows').mkdir(parents=True)
    meta = dict(schema=SCHEMA, complete=False, contract=contract(config, binding), windows={},
                time_support='merged-equal-target-and-physical-intervals')
    write_json(root/'manifest.json', meta)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(config, hashes)) as pool:
        for n, result in enumerate(pool.map(export_chart, tasks), 1):
            meta['windows'].update(result)
            print(json.dumps(dict(charts=n,total_charts=len(tasks),windows=len(meta['windows']))), flush=True)
    meta['complete'] = True
    write_json(root/'manifest.json', meta)
    return meta


def main():
    from maidata.entry import CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    config = dict(CONFIG, **json.loads(args.config.read_text(encoding='utf-8-sig')))
    config['decoder_train_window_cache'] = str(args.output)
    export(config, workers=args.workers)


if __name__ == '__main__':
    main()
