"""Whole-audio Dataset/Collator and deterministic, base-window epoch ordering.

PyTorch loader lifecycle follows the original pretrainer. No historical model,
Decoder sampler or Physical supports are read. Cached mode reads public P0 only.
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
from maidata.draft.targets import reconstruction_target, overlap_reconstruction_target, planner_target
from maidata.draft.supervision import prepare_auxiliary
from maidata.draft.windows import song_coverage, audio_windows
from maidata.preparation.cache import load_grid, song_inputs, window_inputs
from maidata.time import TimingMap
from .batching import optimizer_batches, group_token_cost
from .frozen_p0 import FrozenP0, collate_banks


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class AudioPlan:
    def __init__(self, root, config):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'plan.json')
        p = self.manifest
        self.windows = [json.loads(line) for line in
                        (self.root / p['files']['windows']).read_text(encoding='utf-8').splitlines()]
        self.by_audio = defaultdict(list)
        for w in self.windows:
            self.by_audio[w['audio_variant']].append(w)
        self.groups = {g['audio_variant']: g for g in p['audio_groups']}
        self.charts = {c['chart_id']: c for c in p['charts']}

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
        self.p0 = FrozenP0(config['pretrain_p0_cache']) if config['pretrain_audio_mode'] == 'cached' else None
        if self.p0 is not None:
            self.p0.check_plan(plan)

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch = index
        c, plan = self.config, self.plan
        meta = plan.groups[variant]
        grid, p0 = None, None
        if self.p0 is not None:
            p0, coverage = self.p0.read(variant)
        else:
            grid = load_grid(Path(c['pretrain_audio_cache']) / 'grids' / f'{variant}.npz',
                             mel_root=c['pretrain_mel_root'], validate_input=False)
            coverage = song_coverage(TimingMap.from_audio_timing(meta['timing'], validate_input=False), meta['audio_end_seconds'],
                                    first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        charts, conditions, plans = {}, {}, {}
        for chart_id in meta['chart_ids']:
            item = plan.charts[chart_id]
            record = load_chart(Path(c['pretrain_chart_cache']) / item['chart_cache_path'], validate_input=False)
            source = dict(record['source'])
            source['source_path'] = str(Path(c['pretrain_source_root']) / source['song_id'] / 'maidata.txt')
            conditions[chart_id] = read_condition(source, validate_input=False)
            chart = from_record(dict(record, source=source), validate_input=False)
            charts[chart_id], plans[chart_id] = chart, MusicPlanTargets(chart, validate_input=False)
        windows = []
        rows = plan.rows(variant, epoch)
        for w in rows:
            key = w['chart_id']
            end = coverage.first_beat+coverage.draft_beats
            if c.get('draft_z_anchor_mode', 'block-center') == 'integer':
                target = overlap_reconstruction_target(charts[key], start_tick=w['start_tick'],
                    target_beats=c['target_beats'], coverage_end_beat=end, first_beat=coverage.first_beat,
                    span_beats=c.get('reconstruction_span_beats') or c['draft_z_stride_beats'],
                    validate_input=False)
            else:
                target = reconstruction_target(charts[key], start_tick=w['start_tick'], target_beats=c['target_beats'],
                    audio_after_beats=c['audio_after_beats'], audio_end_seconds=meta['audio_end_seconds'], validate_input=False)
            windows.append((w['window_id'], key, w['start_beat'],
                target,
                planner_target(plans[key], w['start_tick'], w['end_tick'],
                    span_beats=c.get('planner_span_beats', 16), layout=c.get('planner_layout', 'legacy'),
                    stride_beats=c.get('planner_stride_beats', 4),
                    last_parent=w['start_beat']+c['target_beats'] == end)))
        return dict(variant=variant, grid=grid, p0=p0, coverage=coverage, conditions=conditions, windows=windows)


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
    p0: object = None

    def pin_memory(self):
        return map_tensors(self, lambda t: t.pin_memory())

    def to(self, device):
        return map_tensors(self, lambda t: t.to(device, non_blocking=True))


def collate_audio(items, *, config, packed=False):
    cached = config['pretrain_audio_mode'] == 'cached'
    coverages = [i['coverage'] for i in items]
    grids = None if cached else [i['grid'] for i in items]
    variants = [i['variant'] for i in items]
    starts, mapping = audio_windows(coverages, device='cpu')
    conditions = {k: v for i in items for k, v in i['conditions'].items()}
    draft, chart_groups = make_draft_inputs(list(conditions.values()), variants, coverages, device='cpu', validate_input=False,
        z_stride_beats=config['draft_z_stride_beats'], anchor_mode=config.get('draft_z_anchor_mode', 'block-center'))
    ownership = dict(zip(conditions, chart_groups.tolist()))
    auxiliary = prepare_auxiliary(items, config, ownership, draft)
    batch = Batch(None if cached else song_inputs(grids, validate_input=False), None if cached else window_inputs(grids, starts.numpy(), mapping.numpy(),
        target_beats=config['target_beats'], before_beats=config['audio_before_beats'],
        after_beats=config['audio_after_beats'], validate_input=False), coverages, draft,
        auxiliary['groups'], auxiliary['starts'], auxiliary['valid'], auxiliary['reconstruction'],
        auxiliary['planner'], auxiliary['window_ids'], variants)
    from maidata.audio.layout import prepare_encoder_layout
    from maidata.draft.layout import prepare_p0, prepare_selection
    batch.layouts = auxiliary['layouts']
    if cached:
        batch.p0 = collate_banks(items)
    else:
        layout = prepare_encoder_layout(batch.songs.beat_centers, batch.songs.sample_valid.any(-1),
            batch.audio_windows, radii=config['stage_attention_radii'], rates=config['fpn_rates'],
            before=config['audio_before_beats'], context=config['audio_before_beats']+config['target_beats']+config['audio_after_beats'])
        batch.audio_windows.layout = layout
        batch.layouts['p0'] = prepare_p0(layout['grids'][0], layout['resample'][0]['valid'],
            starts, mapping, coverages, p0_rate=config['fpn_rates'][0])
    batch.draft_inputs.valid_indices = draft.draft_valid.flatten().nonzero().flatten()
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
