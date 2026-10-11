"""Compile exact field/query supervision from accepted original-v2 assets.

Fresh production and incremental migration share compile_window. Old rounded
window tensors are never a source of tick labels.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, asdict
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
from maidata.chart.codec import token_to_dict
from maidata.chart.windows import TimeTarget
from maidata.chart.types import HoldDuration, SlideDuration, Moment
from maidata.training.checkpoint import digest_file
from .support_cache import ChartSupports
from .source_labels import prepare
from .window_cache import SCHEMA, contract, tensor_pack


def source_window(chart, row, config, audio_end, onsets):
    example = window_from_config(chart.tokens, chart.timing, config, sequence_start_tick=chart.start_tick,
        start_tick=row['start_tick'], audio_end_seconds=audio_end, training=False, rng=random.Random(0), onsets=onsets)
    if example.end_tick != row['end_tick']:
        raise ValueError('window geometry differs')
    from maidata.time.discrete import duration_cap
    crossings = []
    cursor = example.start_tick
    for offset, target in enumerate(example.targets[example.boundary_index:]):
        if isinstance(target, Moment):
            cursor = chart.timing.tick_after(cursor, target.delta_ms)
        elif isinstance(target, (HoldDuration, SlideDuration)):
            cap = duration_cap(target.type, timing=chart.timing, onset_tick=cursor,
                               local_end_tick=example.cap_tick, audio_end_seconds=audio_end)
            if target.ticks > cap:
                crossings.append(dict(target_offset=offset, kind=target.type, onset_tick=cursor,
                                      exact_ticks=target.ticks, cap_ticks=cap))
    # The GT action and candidate ledger retain the complete source duration.
    # Only query supervision stops at the local observation boundary.
    stages, audit = chart.window(example)
    tokens, types, _ = collate_windows([example], device='cpu', expand=False)
    labels = prepare([dict(example=example, stages=stages,
                           timing=chart.timing, audio_end=audio_end)])
    # Store the shared supervision once; history dropout changes only input and
    # row offsets, never real Physical state or the target decision supports.
    for group in labels.groups.values():
        group['indices'][:, 1] -= example.boundary_index
    labels.flags['indices'][:, 1] -= example.boundary_index
    exact = []
    for offset, (index, target) in enumerate(zip(chart.source_indices(example),
                                                example.targets[example.boundary_index:])):
        exact.append(dict(target_offset=offset, source_token_index=index,
                          target=asdict(target) if isinstance(target, TimeTarget) else token_to_dict(target),
                          origin_tick=stages[example.boundary_index+offset]['cursor_tick']))
    query_cost = sum(int(g['allowed'].any(-1).sum()) if g['allowed'].ndim == 3 else g['allowed'].shape[0]
                     for key, g in labels.groups.items()
                     if key in ('kind','route') or g['allowed'].ndim == 3)
    return dict(tokens={f.name: getattr(tokens, f.name)[0] for f in fields(tokens)}, next_types=types[0],
        history=example.boundary_index,
        labels=dict(groups=labels.groups, counts=labels.counts, flags=labels.flags),
        metadata=dict(chart_id=row.get('chart_id'), window_id=row.get('window_id'),
            source_schema=chart.record['schema'], time_representation='relative-tick384-continue',
            start_tick=example.start_tick, end_tick=example.end_tick, local_end_tick=example.cap_tick,
            history_beats=example.beats[:example.boundary_index], boundary_index=example.boundary_index,
            last_real_tick=example.last_real_tick, exact_fields=exact,
            main_sequence_length=len(types[0]), field_query_cost=query_cost,
            duration_window_crossings=crossings, support_reuse=audit))


def compile_window(chart, row, config, audio_end, onsets):
    from .sequence_cache import expand_window
    example = window_from_config(chart.tokens, chart.timing, config, sequence_start_tick=chart.start_tick,
        start_tick=row["start_tick"], audio_end_seconds=audio_end, training=False, rng=random.Random(0), onsets=onsets)
    return expand_window(source_window(chart,row,config,audio_end,onsets),example,chart.timing)


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
    from .sequence_cache import chart_sequence,write_asset
    root=Path(c['decoder_train_window_cache'])
    sequence=dict(chart_sequence(record),source_chart=relative)
    seq=write_asset(root/'sequences'/f'{key.replace(chr(58),chr(45))}.pt',sequence,root)
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
            logical_lengths=value['metadata']['logical_lengths'],
            duration_window_crossings=value['metadata']['duration_window_crossings'],
            path='windows/'+name, bytes=path.stat().st_size, sha256=digest_file(path))
    return key,result,seq


def export(config, *, workers=12, window_ids=None, source_windows=None):
    from .plan import DecoderPlan
    from .checkpoint import data_binding
    root = Path(config['decoder_train_window_cache'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('window output must be new or empty')
    plans = [DecoderPlan(config['decoder_train_'+name], config, training=i==0)
             for i, name in enumerate(('train_plan','validation_plan'))]
    binding, hashes = data_binding(plans, config)
    previous = None
    if source_windows is not None:
        previous = json.loads((Path(source_windows)/'manifest.json').read_text(encoding='utf-8-sig'))
        if previous['schema'] not in ('maidata.decoder-window-tensors.v1', 'maidata.decoder-window-tensors.v2',
                                      'maidata.decoder-window-tensors.v3', 'maidata.decoder-window-tensors.v4'):
            raise ValueError('migration source must be a previous v1/v2/v3 window cache')
        if window_ids is None:
            window_ids = set(previous['windows'])
    selected = None if window_ids is None else set(window_ids)
    known = {w['window_id'] for p in plans for w in p.windows}
    if selected is not None and not selected <= known:
        raise ValueError('selected window_id absent from unchanged plans')
    tasks = []
    for plan in plans:
        for key, chart in plan.charts.items():
            rows = [w for w in plan.windows if w['chart_id'] == key
                    and (selected is None or w['window_id'] in selected)]
            if not rows:
                continue
            if previous is not None:
                for row in rows:
                    old = previous['windows'][row['window_id']]
                    if any(old[k] != row[k] for k in ('chart_id','audio_variant','start_tick','end_tick')):
                        raise ValueError('migration changes source window membership or geometry')
            tasks.append((key, chart['chart_cache_path'], rows,
                          plan.groups[rows[0]['audio_variant']]['audio_end_seconds']))
    (root/'windows').mkdir(parents=True)
    meta = dict(schema=SCHEMA, complete=False, contract=contract(config, binding), windows={}, sequences={},
                time_support='exact-integer-intervals-gt-prefix-step-mean',
                source_windows=None if source_windows is None else str(Path(source_windows).resolve()),
                selected_window_ids=None if selected is None else sorted(selected))
    write_json(root/'manifest.json', meta)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(config, hashes)) as pool:
        for n, (key,result,seq) in enumerate(pool.map(export_chart, tasks), 1):
            meta['windows'].update(result)
            meta['sequences'][key]=seq
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
    parser.add_argument('--window-id', action='append', help='Compile only explicitly selected existing windows')
    args = parser.parse_args()
    config = dict(CONFIG, **json.loads(args.config.read_text(encoding='utf-8-sig')))
    config['decoder_train_window_cache'] = str(args.output)
    export(config, workers=args.workers, window_ids=args.window_id)


if __name__ == '__main__':
    main()
