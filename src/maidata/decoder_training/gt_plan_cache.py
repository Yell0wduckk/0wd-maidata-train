"""Export GT plan12 from selected original charts; training reads only small banks."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path


SCHEMA = 'maidata.gt-plan12.v1'
PLAN4_SCHEMA = 'maidata.gt-plan4x12.v1'
CENTERED_SCHEMA = 'maidata.gt-plan16-centered-stride4.v1'
CONTINUOUS = ('note_count', 'note_density', 'chord_ratio', 'slide_ratio',
              'last_position', 'touch_ratio', 'hold_ratio', 'relative_intensity')
CATEGORIES = ('single', 'chord', 'slide', 'touch')


class GTPlanCache:
    def __init__(self, root, config=None):
        self.root = Path(root)
        path = self.root/'manifest.json'
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        schema = self.manifest.get('schema')
        if schema not in (None, SCHEMA, CENTERED_SCHEMA):
            raise ValueError('unsupported GT plan cache schema')
        # Older unschematized Plan16 metadata remains readable.
        self.span_beats = 16
        self.layout = 'centered-overlap' if schema == CENTERED_SCHEMA else 'legacy'
        consumer_layout = config.get('planner_layout', 'legacy') if config is not None else self.layout
        compatible_window_statistics = (consumer_layout == 'window-single'
                                        and self.layout == 'legacy' and self.span_beats == 16)
        if consumer_layout != self.layout and not compatible_window_statistics:
            raise ValueError('GT plan cache schema differs from planner_layout')
        if config is not None and config.get('planner_span_beats', 16) != self.span_beats:
            raise ValueError('GT plan cache requires the original single Plan16')
        if self.span_beats == 4 and any(self.manifest.get(key) != value for key, value in (
                ('plan_span_beats', 4), ('parent_window_beats', 16),
                ('plans_per_window', 4), ('tick_rate', 96))):
            raise ValueError('GT Plan4 manifest interval semantics differ')
        if self.layout == 'centered-overlap' and any(self.manifest.get(key) != value for key, value in (
                ('plan_span_beats', 16), ('plan_stride_beats', 4), ('tick_rate', 96),
                ('planner_layout', 'centered-overlap'))):
            raise ValueError('centered Plan manifest interval semantics differ')
        self.groups = self.manifest['groups']
        self.binding = str(path)

    def read(self, variant):
        import torch
        value = torch.load(self.root/self.groups[variant]['path'], map_location='cpu', weights_only=True)
        value = dict(value, plan_span_beats=self.span_beats, planner_layout=self.layout,
                     window_index={key: i for i, key in enumerate(value['window_ids'])})
        if self.layout == 'centered-overlap':
            value['chart_center_index'] = centered_index(value)
        elif self.span_beats == 4:
            value['chart_center_index'] = plan4_index(value)
        return value


def centered_index(bank):
    count = len(bank['chart_ids'])
    if (bank['plan_values'].shape != (count, 12) or bank['valid'].shape != (count,)
            or bank['center_beats'].shape != (count,)):
        raise ValueError('centered Plan bank requires unique [N,12] values and [N] coordinates/masks')
    lookup = {}
    for index, (chart, center) in enumerate(zip(bank['chart_ids'], bank['center_beats'].tolist(), strict=True)):
        key = chart, float(center)
        if key in lookup:
            raise ValueError('duplicate centered Plan chart/anchor')
        lookup[key] = index
    return lookup


def select_plan_conditions(bank, identity, chart_id, start, config, *, window_index=None):
    """Positioned Plan conditions keep their own independent coordinate grid."""
    import torch
    if config.get('planner_layout', 'legacy') != 'centered-overlap':
        values, valid = select_plan_window(bank, identity, chart_id, start, config, window_index=window_index)
        return values, valid, None
    if (bank.get('planner_layout', 'legacy') != 'centered-overlap'
            or bank.get('plan_span_beats') != 16 or bank.get('plan_stride_beats') != 4):
        raise ValueError('cached centered Plan layout/span/stride differs')
    if 'chart_center_index' not in bank:
        bank['chart_center_index'] = centered_index(bank)
    centers = torch.arange(5, dtype=torch.float32)*4+float(start)
    values, valid = bank['plan_values'].new_zeros((5, 12)), torch.zeros(5, dtype=torch.bool)
    for slot, center in enumerate(centers.tolist()):
        index = bank['chart_center_index'].get((chart_id, center))
        if index is not None:
            values[slot], valid[slot] = bank['plan_values'][index], bank['valid'][index]
    return values.masked_fill(~valid[:, None], 0), valid, centers


def plan4_index(bank):
    """Index a loaded small bank once by its chart and absolute segment center."""
    values, valid, beats = bank['plan_values'], bank['valid'], bank['center_beats']
    count = len(bank['chart_ids'])
    if values.shape != (count, 4, 12) or valid.shape != (count, 4) or beats.shape != (count, 4):
        raise ValueError('Plan4 bank requires aligned [N,4,12] values and [N,4] coordinates/masks')
    lookup = {}
    for row, (chart, centers) in enumerate(zip(bank['chart_ids'], beats.tolist(), strict=True)):
        for segment, center in enumerate(centers):
            key = (chart, float(center))
            if key in lookup:
                raise ValueError('duplicate Plan4 chart/center in bank')
            lookup[key] = (row, segment)
    return lookup


def select_plan_window(bank, identity, chart_id, start, config, *, window_index=None):
    """Gather actual neighboring four-beat plans; absent context stays invalid."""
    import torch
    if config.get('planner_layout', 'legacy') == 'centered-overlap':
        return select_plan_conditions(bank, identity, chart_id, start, config, window_index=window_index)[:2]
    consumer_layout = config.get('planner_layout', 'legacy')
    allowed = ('legacy', 'window-single') if consumer_layout == 'window-single' else ('legacy',)
    if bank.get('planner_layout', 'legacy') not in allowed:
        raise ValueError('cached planner_layout differs from window-statistic recipe')
    span = config.get('planner_span_beats', 16)
    if span != 16:
        raise ValueError('GT plan cache requires the original single Plan16')
    if bank.get('plan_span_beats', 16) != span:
        raise ValueError('cached plan_span_beats differs from planner_span_beats')
    if span == 16:
        if bank['plan_values'].ndim != 2 or bank['plan_values'].shape[-1] != 12:
            raise ValueError('Plan16 bank requires [N,12] values')
        lookup = bank['window_index'] if window_index is None else window_index
        row = lookup[identity]
        return bank['plan_values'][row], bank['valid'][row]
    if (config['target_beats'] != 16 or config['draft_z_stride_beats'] != 4
            or config['decoder_z_before_beats'] != 4 or config['decoder_z_after_beats'] != 4):
        raise ValueError('Plan4 conditions require 16 target beats and four-beat Z/context spacing')
    if 'chart_center_index' not in bank:
        bank['chart_center_index'] = plan4_index(bank)
    values = bank['plan_values'].new_zeros((6, 12))
    valid = torch.zeros(6, dtype=torch.bool)
    for slot, offset in enumerate((-2, 2, 6, 10, 14, 18)):
        index = bank['chart_center_index'].get((chart_id, float(start)+offset))
        if index is not None:
            values[slot] = bank['plan_values'][index]
            valid[slot] = bank['valid'][index]
    return values.masked_fill(~valid[:, None], 0), valid


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
    if CONFIG.get('planner_layout', 'legacy') == 'centered-overlap':
        return export_centered_variant(task)
    span = CONFIG.get('planner_span_beats', 16)
    if span != 16:
        raise ValueError('GT plan cache requires the original single Plan16')
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
            if span == 4:
                if row['end_tick']-row['start_tick'] != 16*96:
                    raise ValueError('GT Plan4 export requires complete sixteen-beat base windows')
                targets[row['window_id']] = [planner.window(row['start_tick']+k*384,
                    row['start_tick']+(k+1)*384) for k in range(4)]
            else:
                targets[row['window_id']] = planner.window(row['start_tick'], row['end_tick'])
    ordered = [targets[row['window_id']] for row in rows]
    if span == 4:
        continuous = torch.tensor([[target.continuous for target in group] for group in ordered], dtype=torch.float32)
        category = torch.tensor([[target.category for target in group] for group in ordered], dtype=torch.long)
    else:
        continuous = torch.tensor([target.continuous for target in ordered], dtype=torch.float32)
        category = torch.tensor([target.category for target in ordered], dtype=torch.long)
    values = torch.cat((continuous, torch.nn.functional.one_hot(category, 4).float()), -1)
    shape = (len(rows), 4) if span == 4 else (len(rows),)
    payload = dict(plan_values=values, valid=torch.ones(shape, dtype=torch.bool),
                   window_ids=[row['window_id'] for row in rows])
    if span == 4:
        starts = torch.tensor([row['start_tick'] for row in rows], dtype=torch.long)[:, None]+torch.arange(4)*384
        payload.update(schema=PLAN4_SCHEMA, plan_span_beats=4, chart_ids=[row['chart_id'] for row in rows],
            continuous=continuous, category=category, start_ticks=starts, end_ticks=starts+384,
            center_beats=starts.to(torch.float64)/96+2)
    path = OUTPUT/'banks'/f'{variant}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    # Accept only this newly generated asset, once. Training never repeats this.
    saved = torch.load(temporary, map_location='cpu', weights_only=True)
    if (saved['window_ids'] != payload['window_ids'] or
            saved['plan_values'].dtype != torch.float32 or saved['plan_values'].shape != (*shape, 12) or
            saved['valid'].dtype != torch.bool or saved['valid'].shape != shape or
            not torch.isfinite(saved['plan_values']).all() or not saved['valid'].all() or
            not torch.equal(saved['plan_values'], values) or not torch.equal(saved['valid'], payload['valid'])):
        raise ValueError('GT plan12 storage roundtrip differs')
    if span == 4:
        for key in ('start_ticks', 'end_ticks', 'center_beats', 'continuous', 'category'):
            if not torch.equal(saved[key], payload[key]):
                raise ValueError('GT Plan4 storage roundtrip differs: '+key)
        if saved['chart_ids'] != payload['chart_ids']:
            raise ValueError('GT Plan4 storage roundtrip differs: chart_ids')
    temporary.replace(path)
    return variant, dict(path='banks/'+path.name, windows=len(rows), shape=list(values.shape),
                         dtype=str(values.dtype), bytes=path.stat().st_size)


def export_centered_variant(task):
    """Each original chart is read once; overlapping anchors are emitted once."""
    import hashlib
    import torch
    from maidata.chart_cache.cache import load_chart
    from maidata.draft.chart_data import from_record
    from maidata.draft.plan_targets import MusicPlanTargets
    variant, rows, charts = task
    by_chart = defaultdict(list)
    for row in rows:
        if row['end_tick']-row['start_tick'] != 1536:
            raise ValueError('centered Plan export requires complete sixteen-beat base windows')
        by_chart[row['chart_id']].append(row)
    keys, centers, targets = [], [], []
    for chart_id, windows in by_chart.items():
        record = load_chart(Path(CONFIG['decoder_train_chart_cache'])/charts[chart_id], validate_input=False)
        source = dict(record['source'])
        source['source_path'] = str(Path(CONFIG['decoder_train_source_root'])/source['song_id']/'maidata.txt')
        planner = MusicPlanTargets(from_record(dict(record, source=source), validate_input=False), validate_input=False)
        anchors = {tick for row in windows for tick in range(row['start_tick'], row['end_tick'], 384)}
        anchors.add(max(row['end_tick'] for row in windows))
        for tick in sorted(anchors):
            keys.append(chart_id); centers.append(tick/96)
            targets.append(planner.window(tick-768, tick+768))
    continuous = torch.tensor([target.continuous for target in targets], dtype=torch.float32)
    category = torch.tensor([target.category for target in targets], dtype=torch.long)
    values = torch.cat((continuous, torch.nn.functional.one_hot(category, 4).float()), -1)
    center = torch.tensor(centers, dtype=torch.float64)
    payload = dict(schema=CENTERED_SCHEMA, planner_layout='centered-overlap', plan_span_beats=16,
        plan_stride_beats=4, chart_ids=keys, center_beats=center,
        start_ticks=((center-8)*96).long(), end_ticks=((center+8)*96).long(),
        window_ids=[f'{key}:center:{beat:g}' for key, beat in zip(keys, centers)],
        plan_values=values, valid=torch.ones(len(keys), dtype=torch.bool),
        continuous=continuous, category=category)
    path = OUTPUT/'banks'/f'{variant}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    saved = torch.load(temporary, map_location='cpu', weights_only=True)
    if saved.keys() != payload.keys() or not torch.isfinite(saved['plan_values']).all():
        raise ValueError('centered Plan storage roundtrip differs')
    for key, value in payload.items():
        if not (torch.equal(saved[key], value) if isinstance(value, torch.Tensor) else saved[key] == value):
            raise ValueError('centered Plan storage roundtrip differs: '+key)
    centered_index(saved)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return variant, dict(path='banks/'+path.name, anchors=len(keys), charts=len(by_chart),
        shape=list(values.shape), dtype=str(values.dtype), bytes=path.stat().st_size,
        sha256=digest.hexdigest(), consumer_read_verified=True)


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
    span = config.get('planner_span_beats', 16)
    if span != 16:
        raise ValueError('GT plan cache requires the original single Plan16')
    meta = dict(schema=PLAN4_SCHEMA if span == 4 else SCHEMA, complete=False,
                continuous=list(CONTINUOUS), categories=list(CATEGORIES),
                encoding='continuous8 + category one_hot4', splits=splits, groups={}, workers=workers,
                target_beats=config['target_beats'])
    if span == 4:
        if config['target_beats'] != 16:
            raise ValueError('GT Plan4 export requires target_beats=16')
        meta.update(parent_window_beats=16, plan_span_beats=4, plans_per_window=4, tick_rate=96)
    if config.get('planner_layout', 'legacy') == 'centered-overlap':
        if span != 16 or config.get('planner_stride_beats') != 4 or config['target_beats'] != 16:
            raise ValueError('centered Plan export requires span16/stride4/parent16')
        meta.update(schema=CENTERED_SCHEMA, planner_layout='centered-overlap', plan_span_beats=16,
            plan_stride_beats=4, parent_window_beats=16, tick_rate=96,
            anchor_mode='integer', read_interval='[center-8,center+8]',
            statistics_interval='[center-8,center+8)', unique_key='chart_id,center_beats')
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
