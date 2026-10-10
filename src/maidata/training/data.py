"""Whole-audio Dataset/Collator and deterministic, base-window epoch ordering.

PyTorch loader lifecycle follows the original pretrainer. No historical model,
Decoder sampler, Physical supports or frozen audio representations are read.
"""
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from maidata.chart_cache.cache import load_chart
from maidata.draft.chart_data import from_record
from maidata.draft.conditions import read_condition, make_draft_inputs
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target, collate_reconstruction, collate_planner
from maidata.draft.windows import song_coverage, audio_windows
from maidata.preparation.cache import load_grid, song_inputs, window_inputs
from maidata.time import TimingMap
from .batching import optimizer_batches, group_token_cost


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class AudioPlan:
    def __init__(self, root, config):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'plan.json')
        p = self.manifest
        if (p['schema'] != 'maidata.window-sampling-plan.v1'
                or p['policy']['views'] != ['original']
                or p['policy'].get('consumer') != 'encoder-draft-pretraining'
                or p['config']['complex_rhythm_multiplier'] != 1
                or p['seed'] != config['pretrain_seed']):
            raise ValueError('stage one requires the bound original, base-only plan and seed')
        for key in ('target_beats', 'audio_before_beats', 'audio_after_beats'):
            if p['config'][key] != config[key]:
                raise ValueError(f'plan/config geometry differs: {key}')
        if p['config']['target_stride_beats'] != config['target_beats']:
            raise ValueError('stage one requires disjoint base target windows')
        self.windows = [json.loads(line) for line in
                        (self.root / p['files']['windows']).read_text(encoding='utf-8').splitlines()]
        self.by_audio = defaultdict(list)
        ids = set()
        for w in self.windows:
            if w['window_id'] in ids:
                raise ValueError('duplicate base supervision window')
            ids.add(w['window_id'])
            self.by_audio[w['audio_variant']].append(w)
        self.groups = {g['audio_variant']: g for g in p['audio_groups']}
        self.charts = {c['chart_id']: c for c in p['charts']}
        if set(self.by_audio) != set(self.groups) or len(ids) != p['totals']['base_windows']:
            raise ValueError('plan audio/window inventory mismatch')
        # Check the existing first-epoch ordering, without reopening any payload.
        saved = [json.loads(line) for line in
                 (self.root / p['files']['occurrences']).read_text(encoding='utf-8').splitlines()]
        actual = [(g, w['window_id']) for g in self.order(p['epoch']) for w in self.rows(g, p['epoch'])]
        if actual != [(o['audio_variant'], o['window_id']) for o in saved] or any(o['occurrence'] != 0 for o in saved):
            raise ValueError('plan ordering/occurrences differ from base-only epoch ordering')
        if self.order(p['epoch']) != list(self.groups):
            raise ValueError('plan audio group boundaries differ')
        for variant, group in self.groups.items():
            if group['occurrences'] != len(self.by_audio[variant]):
                raise ValueError('incomplete audio group in plan')

    def order(self, epoch):
        result = list(self.by_audio)
        random.Random(f'{self.manifest["seed"]}:{epoch}:groups').shuffle(result)
        return result

    def rows(self, variant, epoch):
        result = list(self.by_audio[variant])
        random.Random(f'{self.manifest["seed"]}:{epoch}:within:{variant}').shuffle(result)
        return result

    def batches(self, epoch, config):
        order = self.order(epoch)
        return optimizer_batches(order, config, token_costs=[group_token_cost(self.groups[k], config) for k in order])


class AudioDataset(Dataset):
    def __init__(self, plan, config):
        self.plan, self.config = plan, config

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch = index
        c, plan = self.config, self.plan
        meta = plan.groups[variant]
        grid = load_grid(Path(c['pretrain_audio_cache']) / 'grids' / f'{variant}.npz',
                         mel_root=c['pretrain_mel_root'])
        if grid['mel_patches'].shape[1:] != (c['samples_per_beat'], c['mel_bins']):
            raise ValueError('audio grid/config representation mismatch')
        coverage = song_coverage(TimingMap.from_audio_timing(meta['timing']), meta['audio_end_seconds'],
                                first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        if (list(coverage.starts(include_tail=False)) != meta['encoder_starts_beats']
                or coverage.first_beat + coverage.draft_beats != meta['draft_end_beat']):
            raise ValueError('audio-derived coverage differs from bound plan')
        charts, conditions, plans = {}, {}, {}
        for chart_id in meta['chart_ids']:
            item = plan.charts[chart_id]
            record = load_chart(Path(c['pretrain_chart_cache']) / item['chart_cache_path'])
            source = dict(record['source'])
            if source['chart_id'] != chart_id or source['audio_variant'] != variant:
                raise ValueError('chart/audio ownership differs from plan')
            source['source_path'] = str(Path(c['pretrain_source_root']) / source['song_id'] / 'maidata.txt')
            conditions[chart_id] = read_condition(source)
            chart = from_record(dict(record, source=source))
            charts[chart_id], plans[chart_id] = chart, MusicPlanTargets(chart)
        windows = []
        expected = {(key, beat) for key in charts for beat in coverage.starts(include_tail=True)}
        rows = plan.rows(variant, epoch)
        if {(w['chart_id'], w['start_beat']) for w in rows} != expected or len(rows) != len(expected):
            raise ValueError('R/P supervision must cover every base and tail window exactly once')
        for w in rows:
            if w['start_tick'] != w['start_beat'] * 96 or w['end_tick'] != w['start_tick'] + 96*c['target_beats']:
                raise ValueError('window tick/beat geometry differs')
            key = w['chart_id']
            windows.append((w['window_id'], key, w['start_beat'],
                reconstruction_target(charts[key], start_tick=w['start_tick'], target_beats=c['target_beats']),
                plans[key].window(w['start_tick'], w['end_tick'])))
        return dict(variant=variant, grid=grid, coverage=coverage, conditions=conditions, windows=windows)


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if is_dataclass(value):
        return type(value)(**{f.name: map_tensors(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, fn) for v in value)
    return value


@dataclass
class Batch:
    songs: object
    audio_windows: object
    coverages: list
    draft_inputs: object
    groups: torch.Tensor
    starts: torch.Tensor
    valid: torch.Tensor
    reconstruction: object
    planner: object
    window_ids: list
    audio_variants: list
    layouts: dict | None = None

    def pin_memory(self):
        return map_tensors(self, lambda t: t.pin_memory())

    def to(self, device):
        return map_tensors(self, lambda t: t.to(device, non_blocking=True))


def collate_audio(items, *, config, packed=False):
    grids, coverages = [i['grid'] for i in items], [i['coverage'] for i in items]
    variants = [i['variant'] for i in items]
    starts, mapping = audio_windows(coverages, device='cpu')
    conditions = {k: v for i in items for k, v in i['conditions'].items()}
    draft, chart_groups = make_draft_inputs(list(conditions.values()), variants, coverages, device='cpu')
    ownership = dict(zip(conditions, chart_groups.tolist()))
    windows = [w for i in items for w in i['windows']]
    batch = Batch(song_inputs(grids), window_inputs(grids, starts.numpy(), mapping.numpy(),
        target_beats=config['target_beats'], before_beats=config['audio_before_beats'],
        after_beats=config['audio_after_beats']), coverages, draft,
        torch.tensor([ownership[w[1]] for w in windows], dtype=torch.long),
        torch.tensor([w[2] for w in windows], dtype=torch.float32),
        torch.ones(len(windows), dtype=torch.bool),
        collate_reconstruction([w[3] for w in windows], query_rows=config['reconstruction_query_rows'],
                               target_beats=config['target_beats'], device='cpu'),
        collate_planner([w[4] for w in windows], device='cpu'), [w[0] for w in windows], variants)
    from maidata.audio.layout import prepare_encoder_layout
    from maidata.draft.layout import prepare_p0, prepare_selection
    layout = prepare_encoder_layout(batch.songs.beat_centers, batch.songs.sample_valid.any(-1),
        batch.audio_windows, radii=config['stage_attention_radii'], rates=config['fpn_rates'],
        before=config['audio_before_beats'], context=config['audio_before_beats']+config['target_beats']+config['audio_after_beats'])
    batch.audio_windows.layout = layout
    batch.draft_inputs.valid_indices = draft.draft_valid.flatten().nonzero().flatten()
    batch.layouts = dict(p0=prepare_p0(layout['grids'][0], layout['resample'][0]['valid'],
        starts, mapping, coverages, p0_rate=config['fpn_rates'][0]))
    for name, before, after in (('reconstruction', config['reconstruction_before_beats'], config['reconstruction_after_beats']),
                                ('planner', 0, 0)):
        batch.layouts[name] = prepare_selection(draft.query_beats, draft.draft_valid, batch.groups,
            batch.starts, batch.valid, target_beats=config['target_beats'], before_beats=before, after_beats=after)
    if packed:
        from .transport import pack_batch
        return pack_batch(batch)
    return batch



class AudioBatchSampler(Sampler):
    def __init__(self, plan, config):
        self.plan, self.config, self.epoch = plan, config, plan.manifest['epoch']
        self.selected_step = None

    def steps(self):
        steps = self.plan.batches(self.epoch, self.config)
        return steps if self.selected_step is None else (steps[self.selected_step],)

    def __iter__(self):
        for step in self.steps():
            for batch in step:
                yield [(variant, self.epoch) for variant in batch]

    def __len__(self):
        return sum(len(s) for s in self.steps())


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(plan, config):
    workers = config['pretrain_workers']
    if type(workers) is not int or workers < 0:
        raise ValueError('workers must be a nonnegative integer')
    generator = torch.Generator().manual_seed(config['pretrain_seed'])
    sampler = AudioBatchSampler(plan, config)
    return DataLoader(AudioDataset(plan, config), batch_sampler=sampler,
        collate_fn=partial(collate_audio, config=config, packed=True), num_workers=workers,
        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
        pin_memory=torch.device(config['pretrain_device']).type == 'cuda')


def epoch_iterator(loader, restored_rng=None):
    iterator = iter(loader)
    # Creating replacement persistent workers after resume consumes one extra
    # base_seed. Dataset has no random transforms: restore the saved generator.
    if restored_rng is not None and loader.persistent_workers:
        loader.generator.set_state(restored_rng.cpu())
    return iterator
