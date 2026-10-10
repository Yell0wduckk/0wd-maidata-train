"""Random window IO with per-batch audio deduplication; no model ownership."""
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from maidata.training.data import map_tensors, worker_init
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

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch, *selection = index
        c, plan = self.config, self.plan
        rows = plan.rows(variant, epoch, selection[0] if selection else None)
        p0, dp = self.cache.read(variant)
        windows, loaded = [], {}
        for row in rows:
            key, identity = row['chart_id'], row['window_id']
            if identity not in loaded:
                loaded[identity] = self.windows.read(identity)
            tensors = select_window(loaded[identity], history_dropped(c, plan.training, epoch, row))
            windows.append(dict(identity=(identity, row['occurrence']), chart_id=key, start=row['start_beat'],
                sequence_index=row['sequence_index'], **tensors))
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
    windows = [w for i in items for w in i['windows']]
    if any('sequence_index' in w for w in windows):
        windows.sort(key=lambda w: w['sequence_index'])
    batch = Batch(collate_conditions(items, windows, config), windows, variants, [len(i['p0']['values']) for i in items])
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
