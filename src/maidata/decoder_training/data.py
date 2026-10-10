"""Random window IO with per-batch audio deduplication; no model ownership."""
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from maidata.chart_cache.cache import load_chart
from maidata.chart_cache.training import load_supports
from maidata.decoder.windows import window_from_config, real_onsets
from maidata.draft.conditions import read_condition
from maidata.physical import Geometry
from maidata.training.data import map_tensors, worker_init
from .support_cache import ChartSupports
from .ordering import history_rng
from .conditioning import collate_conditions


class WindowBatchSampler(torch.utils.data.Sampler):
    def __init__(self, plan, config):
        self.plan, self.config, self.epoch = plan, config, plan.manifest['epoch']
        self.selected_step = None

    def steps(self):
        steps = self.plan.batches(self.epoch, self.config)
        return steps if self.selected_step is None else (steps[self.selected_step],)

    def __iter__(self):
        for step in self.steps():
            for batch in step:
                grouped = defaultdict(list)
                for row in batch:
                    grouped[row['audio_variant']].append(row['sequence_index'])
                yield [(variant, self.epoch, tuple(indices)) for variant, indices in grouped.items()]

    def __len__(self):
        return sum(len(step) for step in self.steps())


class DecoderDataset(Dataset):
    def __init__(self, plan, config, field_hashes, cache):
        self.plan, self.config, self.field_hashes = plan, config, field_hashes
        self.cache = cache
        self.geometry = None

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch, *selection = index
        c, plan = self.config, self.plan
        if self.geometry is None:
            self.geometry = Geometry.from_resources(c['decoder_train_geometry'], c['decoder_train_slide_data'])
        meta = plan.groups[variant]
        rows = plan.rows(variant, epoch, selection[0] if selection else None)
        p0, dp = self.cache.read(variant)
        charts, onsets = {}, {}
        for key in dict.fromkeys(r['chart_id'] for r in rows):
            path = plan.charts[key]['chart_cache_path']
            record = load_chart(Path(c['decoder_train_chart_cache']) / path)
            support = load_supports(Path(c['decoder_train_physical_cache']) / path)
            source = dict(record['source'])
            if source['chart_id'] != key or source['audio_variant'] != variant:
                raise ValueError('chart/audio ownership differs')
            if support['source_sha256'] != self.field_hashes[key]:
                raise ValueError('Physical supports reference different source fields')
            source['source_path'] = str(Path(c['decoder_train_source_root']) / source['song_id'] / 'maidata.txt')
            saved = dp['chart_conditions'][key]
            if read_condition(source).key != (saved['audio_variant'], saved['level'], saved['difficulty_id']):
                raise ValueError('cached Draft difficulty/level condition differs from fields')
            charts[key] = ChartSupports(record, support, self.geometry)
            chart = charts[key]
            onsets[key] = real_onsets(chart.tokens, chart.timing, sequence_start_tick=chart.start_tick)
        windows, supports = [], {}
        for row in rows:
            key, identity = row['chart_id'], row['window_id']
            chart = charts[key]
            example = window_from_config(chart.tokens, chart.timing, c, sequence_start_tick=chart.start_tick,
                start_tick=row['start_tick'], audio_end_seconds=meta['audio_end_seconds'],
                training=plan.training, rng=history_rng(c['decoder_train_seed'], epoch, identity, row['occurrence']),
                onsets=onsets[key])
            if row['end_tick'] != example.end_tick or row['start_tick'] != 96*row['start_beat']:
                raise ValueError('window time geometry differs')
            # Only history/boundary input changes under dropout, not real Physical.
            if identity not in supports:
                stages, audit = chart.window(example)
                supports[identity] = (stages[example.boundary_index:], audit)
            stages, audit = supports[identity]
            windows.append(dict(identity=(identity, row['occurrence']), chart_id=key, start=row['start_beat'],
                sequence_index=row['sequence_index'],
                example=example, stages=(None,)*example.boundary_index+stages, timing=chart.timing,
                audio_end=meta['audio_end_seconds'], support_audit=audit))
        return dict(variant=variant, p0=p0, dp=dp, windows=windows)


@dataclass
class Batch:
    conditions: object
    windows: list
    audio_variants: list
    audio_lengths: list
    chunks: list | None = None
    token_budget: int | None = None
    budget_cost: dict | None = None

    def pin_memory(self):
        return map_tensors(self, lambda t: t.pin_memory())

    def to(self, device):
        return self._tensors(lambda t: t.to(device, non_blocking=True))

    def _tensors(self, fn):
        # Only the active Decoder chunk moves to GPU, not a full song's labels.
        from dataclasses import replace
        return replace(map_tensors(replace(self, chunks=None), fn), chunks=self.chunks)


def collate_audio(items, *, config, packed=False):
    variants = [i['variant'] for i in items]
    if len(set(variants)) != len(variants):
        raise ValueError('batch audio variants must be deduplicated')
    windows = [w for i in items for w in i['windows']]
    if any('sequence_index' in w for w in windows):
        windows.sort(key=lambda w: w['sequence_index'])
        if len({w['sequence_index'] for w in windows}) != len(windows):
            raise ValueError('duplicate occurrence sequence index')
    batch = Batch(collate_conditions(items, windows, config), windows, variants, [len(i['p0']['values']) for i in items])
    from .preparation import prepare_batch
    prepare_batch(batch, config)
    if packed:
        from .transport import pack
        return pack(batch)
    return batch


def make_loader(plan, config, field_hashes, cache):
    workers = config['decoder_train_workers']
    generator = torch.Generator().manual_seed(config['decoder_train_seed'])
    return DataLoader(DecoderDataset(plan, config, field_hashes, cache),
        batch_sampler=WindowBatchSampler(plan, config), collate_fn=partial(collate_audio, config=config, packed=True),
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=worker_init,
        generator=generator, pin_memory=torch.device(config['decoder_train_device']).type == 'cuda')
