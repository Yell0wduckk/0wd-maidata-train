"""Export GT plan12 from selected original charts; training reads only small banks."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path


SCHEMA = 'maidata.gt-plan12.v1'
CONTINUOUS = ('note_count', 'note_density', 'chord_ratio', 'slide_ratio',
              'last_position', 'touch_ratio', 'hold_ratio', 'relative_intensity')
CATEGORIES = ('single', 'chord', 'slide', 'touch')


class GTPlanCache:
    def __init__(self, root, config=None):
        self.root = Path(root)
        path = self.root/'manifest.json'
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') not in (None, SCHEMA):
            raise ValueError('unsupported GT Plan16 cache schema')
        if config is not None and config.get('planner_span_beats', 16) != 16:
            raise ValueError('GT plan cache requires the original single Plan16')
        self.groups = self.manifest['groups']
        self.binding = str(path)

    def read(self, variant):
        import torch
        value = torch.load(self.root/self.groups[variant]['path'], map_location='cpu', weights_only=True)
        return dict(value, window_index={key: i for i, key in enumerate(value['window_ids'])})


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def initialize(config, output):
    import torch
    global CONFIG, OUTPUT
    torch.set_num_threads(1)
    CONFIG, OUTPUT = config, Path(output)


def export_variant(task):
    """Read each selected chart once; no audio, geometry, supports or networks."""
    import torch
    from maidata.chart_cache.cache import load_chart
    from maidata.draft.chart_data import from_record
    from maidata.draft.plan_targets import MusicPlanTargets
    variant, rows, charts = task
    by_chart = defaultdict(list)
    for row in rows:
        by_chart[row['chart_id']].append(row)
    targets = {}
    for key, windows in by_chart.items():
        record = load_chart(Path(CONFIG['decoder_train_chart_cache'])/charts[key], validate_input=False)
        source = dict(record['source'])
        source['source_path'] = str(Path(CONFIG['decoder_train_source_root'])/source['song_id']/'maidata.txt')
        chart = from_record(dict(record, source=source), validate_input=False)
        planner = MusicPlanTargets(chart, validate_input=False)
        for row in windows:
            targets[row['window_id']] = planner.window(row['start_tick'], row['end_tick'])
    ordered = [targets[row['window_id']] for row in rows]
    continuous = torch.tensor([target.continuous for target in ordered], dtype=torch.float32)
    category = torch.tensor([target.category for target in ordered], dtype=torch.long)
    values = torch.cat((continuous, torch.nn.functional.one_hot(category, 4).float()), -1)
    payload = dict(plan_values=values, valid=torch.ones(len(rows), dtype=torch.bool),
                   window_ids=[row['window_id'] for row in rows])
    path = OUTPUT/'banks'/f'{variant}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    # Accept only this newly generated asset, once. Training never repeats this.
    saved = torch.load(temporary, map_location='cpu', weights_only=True)
    if (saved['window_ids'] != payload['window_ids'] or
            saved['plan_values'].dtype != torch.float32 or saved['plan_values'].shape != (len(rows), 12) or
            saved['valid'].dtype != torch.bool or saved['valid'].shape != (len(rows),) or
            not torch.isfinite(saved['plan_values']).all() or not saved['valid'].all() or
            not torch.equal(saved['plan_values'], values) or not torch.equal(saved['valid'], payload['valid'])):
        raise ValueError('GT plan12 storage roundtrip differs')
    temporary.replace(path)
    return variant, dict(path='banks/'+path.name, windows=len(rows), shape=list(values.shape),
                         dtype=str(values.dtype), bytes=path.stat().st_size)


def export(config, output, workers=12):
    if config.get('planner_span_beats', 16) != 16:
        raise ValueError('GT plan export requires the original single Plan16')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('GT plan output must be new or empty')
    grouped, charts, splits = defaultdict(list), {}, {}
    for side in ('train', 'validation'):
        root = Path(config['decoder_train_'+side+'_plan'])
        plan = json.loads((root/'plan.json').read_text(encoding='utf-8-sig'))
        charts.update({row['chart_id']: row['chart_cache_path'] for row in plan['charts']})
        splits[side] = [row['audio_variant'] for row in plan['audio_groups']]
        for line in (root/plan['files']['windows']).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            grouped[row['audio_variant']].append(row)
    tasks = [(variant, rows, {key: charts[key] for key in dict.fromkeys(row['chart_id'] for row in rows)})
             for variant, rows in grouped.items()]
    meta = dict(schema=SCHEMA, complete=False, continuous=list(CONTINUOUS), categories=list(CATEGORIES),
                encoding='continuous8 + category one_hot4', splits=splits, groups={}, workers=workers,
                target_beats=config['target_beats'])
    _write_json(output/'manifest.json', meta)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(config, str(output))) as pool:
        for variant, row in pool.map(export_variant, tasks):
            meta['groups'][variant] = row
            print(json.dumps(dict(completed=len(meta['groups']), total=len(tasks), variant=variant)), flush=True)
    meta['complete'] = True
    _write_json(output/'manifest.json', meta)
    return meta


def main(argv=None):
    from maidata.entry import CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    config = dict(CONFIG, **json.loads(args.config.read_text(encoding='utf-8-sig')))
    export(config, args.output, workers=args.workers)


if __name__ == '__main__':
    main()
