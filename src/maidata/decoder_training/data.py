"""Whole-song IO and deterministic GT windows. No learned model ownership."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader

from maidata.chart_cache.cache import load_chart
from maidata.chart_cache.training import load_supports
from maidata.decoder.windows import window_from_config
from maidata.draft.conditions import read_condition, make_draft_inputs
from maidata.draft.windows import song_coverage, audio_windows
from maidata.physical import Geometry
from maidata.preparation.cache import load_grid, song_inputs, window_inputs
from maidata.time import TimingMap
from maidata.training.data import map_tensors, AudioBatchSampler, worker_init, epoch_iterator
from .support_cache import ChartSupports


def history_rng(seed, epoch, window_id, occurrence):
    return random.Random(f'decoder-history:{seed}:{epoch}:{window_id}:{occurrence}')


class DecoderDataset(Dataset):
    def __init__(self, plan, config, field_hashes):
        self.plan, self.config, self.field_hashes = plan, config, field_hashes
        self.geometry = None

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch = index
        c, plan = self.config, self.plan
        if self.geometry is None:
            self.geometry = Geometry.from_resources(c['decoder_train_geometry'], c['decoder_train_slide_data'])
        meta = plan.groups[variant]
        grid = load_grid(Path(c['decoder_train_audio_cache']) / 'grids' / f'{variant}.npz',
                         mel_root=c['decoder_train_mel_root'])
        if grid['mel_patches'].shape[1:] != (c['samples_per_beat'], c['mel_bins']):
            raise ValueError('audio grid/config representation differs')
        coverage = song_coverage(TimingMap.from_audio_timing(meta['timing']), meta['audio_end_seconds'],
                                first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        if (list(coverage.starts(include_tail=False)) != meta['encoder_starts_beats']
                or coverage.first_beat+coverage.draft_beats != meta['draft_end_beat']):
            raise ValueError('incomplete audio or Draft coverage')
        charts, conditions = {}, {}
        for key in meta['chart_ids']:
            path = plan.charts[key]['chart_cache_path']
            record = load_chart(Path(c['decoder_train_chart_cache']) / path)
            support = load_supports(Path(c['decoder_train_physical_cache']) / path)
            source = dict(record['source'])
            if source['chart_id'] != key or source['audio_variant'] != variant:
                raise ValueError('chart/audio ownership differs')
            if support['source_sha256'] != self.field_hashes[key]:
                raise ValueError('Physical supports reference different source fields')
            source['source_path'] = str(Path(c['decoder_train_source_root']) / source['song_id'] / 'maidata.txt')
            conditions[key] = read_condition(source)
            charts[key] = ChartSupports(record, support, self.geometry)
        rows = plan.rows(variant, epoch)
        expected = {(k, b) for k in charts for b in coverage.starts(include_tail=True)}
        if {(r['chart_id'], r['start_beat']) for r in rows if r['occurrence'] == 0} != expected:
            raise ValueError('Decoder base/tail coverage differs')
        windows, supports = [], {}
        for row in rows:
            key, identity = row['chart_id'], row['window_id']
            chart = charts[key]
            example = window_from_config(chart.tokens, chart.timing, c, sequence_start_tick=chart.start_tick,
                start_tick=row['start_tick'], audio_end_seconds=meta['audio_end_seconds'],
                training=plan.training, rng=history_rng(c['decoder_train_seed'], epoch, identity, row['occurrence']))
            if row['end_tick'] != example.end_tick or row['start_tick'] != 96*row['start_beat']:
                raise ValueError('window time geometry differs')
            # Only history/boundary input changes under dropout, not real Physical.
            if identity not in supports:
                stages, audit = chart.window(example)
                supports[identity] = (stages[example.boundary_index:], audit)
            stages, audit = supports[identity]
            windows.append(dict(identity=(identity, row['occurrence']), chart_id=key, start=row['start_beat'],
                example=example, stages=(None,)*example.boundary_index+stages, timing=chart.timing,
                audio_end=meta['audio_end_seconds'], support_audit=audit))
        return dict(variant=variant, grid=grid, coverage=coverage, conditions=conditions, windows=windows)


@dataclass
class Batch:
    songs: object
    audio_windows: object
    coverages: list
    draft_inputs: object
    groups: torch.Tensor
    starts: torch.Tensor
    valid: torch.Tensor
    windows: list
    audio_variants: list

    def pin_memory(self):
        return self._tensors(lambda t: t.pin_memory())

    def to(self, device):
        return self._tensors(lambda t: t.to(device, non_blocking=True))

    def _tensors(self, fn):
        # Keep structured supports, timing and GT on CPU (no GPU scalar collation).
        return Batch(*(map_tensors(getattr(self, k), fn) for k in
            ('songs','audio_windows','coverages','draft_inputs','groups','starts','valid')),
            self.windows, self.audio_variants)


def collate_audio(items, *, config):
    grids, coverages = [i['grid'] for i in items], [i['coverage'] for i in items]
    variants = [i['variant'] for i in items]
    starts, mapping = audio_windows(coverages, device='cpu')
    conditions = {k:v for i in items for k,v in i['conditions'].items()}
    draft, groups = make_draft_inputs(list(conditions.values()), variants, coverages, device='cpu')
    ownership = dict(zip(conditions, groups.tolist()))
    windows = [w for i in items for w in i['windows']]
    return Batch(song_inputs(grids), window_inputs(grids, starts.numpy(), mapping.numpy(),
        target_beats=config['target_beats'], before_beats=config['audio_before_beats'],
        after_beats=config['audio_after_beats']), coverages, draft,
        torch.tensor([ownership[w['chart_id']] for w in windows], dtype=torch.long),
        torch.tensor([w['start'] for w in windows], dtype=torch.float32),
        torch.ones(len(windows), dtype=torch.bool), windows, variants)


def make_loader(plan, config, field_hashes):
    workers = config['decoder_train_workers']
    generator = torch.Generator().manual_seed(config['decoder_train_seed'])
    return DataLoader(DecoderDataset(plan, config, field_hashes),
        batch_sampler=AudioBatchSampler(plan, config), collate_fn=partial(collate_audio, config=config),
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=worker_init,
        generator=generator, pin_memory=torch.device(config['decoder_train_device']).type == 'cuda')
