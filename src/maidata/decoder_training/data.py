"""Random window IO with per-batch audio deduplication; no model ownership."""
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from maidata.training.data import map_tensors, worker_init
from maidata.preparation.cache import load_grid, song_inputs, window_inputs
from maidata.draft.windows import song_coverage, audio_windows
from maidata.time import TimingMap
from .ordering import history_dropped
from .conditioning import collate_conditions
from .window_cache import select_window


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
    def __init__(self, plan, config, field_hashes, cache, windows):
        self.plan, self.config = plan, config
        self.cache, self.windows = cache, windows
        self.gt_plan = None
        if config['decoder_train_plan_source'] == 'gt':
            from .gt_plan_cache import GTPlanCache
            self.gt_plan = GTPlanCache(config['decoder_train_gt_plan_cache'])

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch, *selection = index
        c, plan = self.config, self.plan
        rows = plan.rows(variant, epoch, selection[0] if selection else None)
        dp = self.cache.read(variant)
        grid = load_grid(Path(c['decoder_train_audio_cache'])/'grids'/f'{variant}.npz',
                         mel_root=c['decoder_train_mel_root'], validate_input=False)
        meta = plan.groups[variant]
        coverage = song_coverage(TimingMap.from_audio_timing(meta['timing'], validate_input=False),
            meta['audio_end_seconds'], first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        windows, loaded = [], {}
        for row in rows:
            key, identity = row['chart_id'], row['window_id']
            if identity not in loaded:
                loaded[identity] = self.windows.read(identity)
            tensors = select_window(loaded[identity], history_dropped(c, plan.training, epoch, row))
            windows.append(dict(identity=(identity, row['occurrence']), chart_id=key, start=row['start_beat'],
                sequence_index=row['sequence_index'], **tensors))
        result = dict(variant=variant, grid=grid, coverage=coverage, dp=dp, windows=windows)
        if self.gt_plan is not None:
            result['gt_plan'] = self.gt_plan.read(variant)
        return result


@dataclass
class OnlineAudio:
    """One full coverage per unique batch variant; learned values are never cached."""
    songs: object
    windows: object
    coverages: list
    p0_layout: dict


@dataclass
class Batch:
    conditions: object
    windows: list
    audio_variants: list
    audio_lengths: list
    chunks: list | None = None
    token_budget: int | None = None
    budget_cost: dict | None = None
    online_audio: OnlineAudio | None = None

    def pin_memory(self):
        return map_tensors(self, lambda t: t.pin_memory())

    def to(self, device):
        return self._tensors(lambda t: t.to(device, non_blocking=True))

    def _tensors(self, fn):
        # Only the active Decoder chunk moves to GPU, not a full song's labels.
        from dataclasses import replace
        return replace(map_tensors(replace(self, chunks=None), fn), chunks=self.chunks)


def collate_online_audio(items, config):
    """Reuse stage-I worker layouts and the exact public center-P0 geometry."""
    from maidata.audio.layout import prepare_encoder_layout
    from maidata.draft.layout import prepare_p0
    from maidata.decoder.types import Memory
    grids, coverages = [item['grid'] for item in items], [item['coverage'] for item in items]
    starts, mapping = audio_windows(coverages, device='cpu')
    songs = song_inputs(grids, validate_input=False)
    windows = window_inputs(grids, starts.numpy(), mapping.numpy(), target_beats=config['target_beats'],
        before_beats=config['audio_before_beats'], after_beats=config['audio_after_beats'], validate_input=False)
    layout = prepare_encoder_layout(songs.beat_centers, songs.sample_valid.any(-1), windows,
        radii=config['stage_attention_radii'], rates=config['fpn_rates'], before=config['audio_before_beats'],
        context=config['audio_before_beats']+config['target_beats']+config['audio_after_beats'])
    windows.layout = layout
    p0 = prepare_p0(layout['grids'][0], layout['resample'][0]['valid'], starts, mapping, coverages,
                    p0_rate=config['fpn_rates'][0])
    indices, present = p0['indices'], p0['present']
    beats = layout['grids'][0].flatten()[indices].masked_fill(~present, 0)
    valid = layout['resample'][0]['valid'].flatten()[indices] & present
    # No dummy d_model-wide bank crosses IPC/H2D. The trainable Encoder replaces
    # these zero-width values before M prepares its audio K/V.
    placeholder = Memory(torch.empty((*beats.shape, 0), dtype=torch.float32), beats, valid)
    return OnlineAudio(songs, windows, coverages, p0), placeholder


def collate_audio(items, *, config, packed=False):
    variants = [i['variant'] for i in items]
    windows = [w for i in items for w in i['windows']]
    if any('sequence_index' in w for w in windows):
        windows.sort(key=lambda w: w['sequence_index'])
    online, audio = collate_online_audio(items, config)
    lengths = [coverage.audio_windows*coverage.target_beats*config['fpn_rates'][0] for coverage in online.coverages]
    batch = Batch(collate_conditions(items, windows, config, audio=audio), windows, variants, lengths,
                  online_audio=online)
    from .preparation import prepare_batch
    prepare_batch(batch, config)
    if packed:
        from .transport import pack
        return pack(batch)
    return batch


def make_loader(plan, config, field_hashes, cache, windows):
    workers = config['decoder_train_workers']
    generator = torch.Generator().manual_seed(config['decoder_train_seed'])
    return DataLoader(DecoderDataset(plan, config, field_hashes, cache, windows),
        batch_sampler=WindowBatchSampler(plan, config), collate_fn=partial(collate_audio, config=config, packed=True),
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=worker_init,
        generator=generator, pin_memory=torch.device(config['decoder_train_device']).type == 'cuda')
