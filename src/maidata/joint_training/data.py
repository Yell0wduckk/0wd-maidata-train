"""Complete audio-group joint batches; all geometry is prepared by workers.

The accepted stage-I plan owns group order and logical update boundaries. The
Decoder plan owns its occurrence inventory, while auxiliary labels are consumed
once per base window. No frozen learned P0/Z or online Physical reconstruction is
used by this path.
"""
from dataclasses import dataclass, replace
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

from maidata.decoder.types import Memory
from maidata.decoder_training.conditioning import Conditions
from maidata.decoder_training.gt_plan_cache import GTPlanCache, select_plan_conditions
from maidata.decoder_training.ordering import history_dropped
from maidata.decoder_training.plan import DecoderPlan
from maidata.decoder_training.preparation import prepare_batch
from maidata.decoder_training.transport import pack
from maidata.decoder_training.window_cache import WindowCache, select_window
from maidata.draft.conditions import make_draft_inputs
from maidata.draft.layout import prepare_p0, prepare_selection, prepare_centers
from maidata.draft.supervision import prepare_auxiliary
from maidata.preparation.cache import song_inputs, window_inputs
from maidata.training.data import (AudioBatchSampler, AudioDataset, AudioPlan,
                                   epoch_iterator, map_tensors, worker_init)


class JointPlan(AudioPlan):
    """Reuse stage-I grouping and the existing M occurrence sampler verbatim."""
    def __init__(self, root, config, *, training=True):
        super().__init__(root, config)
        self.training = training
        side = 'train' if training else 'validation'
        self.decoder_plan = DecoderPlan(config[f'decoder_train_{side}_plan'],
                                        config, training=training)


class JointDataset(Dataset):
    def __init__(self, plan, config, windows=None, gt_plan=None):
        self.plan, self.config = plan, config
        self.base = AudioDataset(plan, config)
        self.windows = windows if windows is not None else WindowCache(
            [plan.decoder_plan], config, {})
        self.gt_plan = gt_plan if gt_plan is not None else GTPlanCache(
            config['decoder_train_gt_plan_cache'], config)

    def __len__(self):
        return len(self.plan.groups)

    def __getitem__(self, index):
        variant, epoch = index
        item = self.base[index]
        selected, loaded = [], {}
        for row in self.plan.decoder_plan.rows(variant, epoch):
            identity = row['window_id']
            if identity not in loaded:
                loaded[identity] = self.windows.read(identity)
            tensors = select_window(loaded[identity], history_dropped(
                self.config, self.plan.training, epoch, row))
            selected.append(dict(identity=(identity, row['occurrence']),
                chart_id=row['chart_id'], start=row['start_beat'],
                sequence_index=row['sequence_index'], **tensors))
        return dict(item, m_windows=selected, gt_plan=self.gt_plan.read(variant))


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
    coverage_audio_indices: torch.Tensor
    decoder_groups: torch.Tensor
    decoder_starts: torch.Tensor
    decoder_valid: torch.Tensor
    conditions: Conditions
    layouts: dict
    windows: list
    audio_lengths: list
    chunks: list | None = None
    token_budget: int | None = None
    budget_cost: dict | None = None

    def pin_memory(self):
        return map_tensors(self, lambda tensor: tensor.pin_memory())

    def to(self, device):
        # M support/inputs move only for the active computation chunk. The live
        # audio/Draft context is transferred once for the whole logical group.
        context = map_tensors(replace(self, chunks=None),
            lambda tensor: tensor.to(device, non_blocking=True))
        return replace(context, chunks=self.chunks)


def _audio_union(items, config):
    """Encode coverage plus selected M local windows exactly once per start."""
    from maidata.audio.layout import prepare_encoder_layout
    keys, owners, coverage_indices = [], [], []
    for song, item in enumerate(items):
        seen = {}
        for start in item['coverage'].starts(include_tail=False):
            seen[start] = len(keys)
            coverage_indices.append(len(keys))
            keys.append((item['variant'], start))
            owners.append(song)
        for window in item['m_windows']:
            start = window['start']
            if start not in seen:
                seen[start] = len(keys)
                keys.append((item['variant'], start))
                owners.append(song)
    starts = torch.tensor([key[1] for key in keys], dtype=torch.float32)
    mapping = torch.tensor(owners, dtype=torch.long)
    grids = [item['grid'] for item in items]
    songs = song_inputs(grids, validate_input=False)
    windows = window_inputs(grids, starts.numpy(), mapping.numpy(),
        target_beats=config['target_beats'], before_beats=config['audio_before_beats'],
        after_beats=config['audio_after_beats'], validate_input=False)
    layout = prepare_encoder_layout(songs.beat_centers, songs.sample_valid.any(-1),
        windows, radii=config['stage_attention_radii'], rates=config['fpn_rates'],
        before=config['audio_before_beats'], context=config['audio_before_beats']+
        config['target_beats']+config['audio_after_beats'])
    windows.layout = layout
    beats, valid = layout['grids'][0], layout['resample'][0]['valid']
    coverage_indices = torch.tensor(coverage_indices, dtype=torch.long)
    p0 = prepare_p0(beats[coverage_indices], valid[coverage_indices],
        starts[coverage_indices], mapping[coverage_indices],
        [item['coverage'] for item in items], p0_rate=config['fpn_rates'][0])
    audio = Memory(torch.empty((*beats.shape, 0), dtype=torch.float32), beats, valid)
    return songs, windows, keys, coverage_indices, p0, audio


def collate_audio(items, *, config, packed=False):
    variants = [item['variant'] for item in items]
    coverages = [item['coverage'] for item in items]
    conditions = {key: value for item in items for key, value in item['conditions'].items()}
    draft, chart_groups = make_draft_inputs(list(conditions.values()), variants,
        coverages, device='cpu', validate_input=False,
        z_stride_beats=config['draft_z_stride_beats'],
        anchor_mode=config.get('draft_z_anchor_mode', 'block-center'))
    ownership = dict(zip(conditions, chart_groups.tolist()))
    draft.valid_indices = draft.draft_valid.flatten().nonzero().flatten()

    auxiliary = prepare_auxiliary(items, config, ownership, draft)
    groups, starts, valid = (auxiliary[key] for key in ('groups', 'starts', 'valid'))
    layouts = auxiliary['layouts']
    centered = config.get('planner_layout', 'legacy') == 'centered-overlap'

    songs, audio_windows, audio_keys, coverage_indices, layouts['p0'], audio = _audio_union(items, config)
    audio_lookup = {key: index for index, key in enumerate(audio_keys)}
    m_windows = sorted([window for item in items for window in item['m_windows']],
                       key=lambda window: window['sequence_index'])
    item_lookup = {item['variant']: item for item in items}
    chart_variant = {key: item['variant'] for item in items for key in item['conditions']}
    unique, occurrence, decoder_groups, decoder_starts = {}, [], [], []
    plans, plan_valid, plan_beats, audio_mapping = [], [], [], []
    for window in m_windows:
        identity = window['identity'][0]
        if identity not in unique:
            unique[identity] = len(unique)
            variant = chart_variant[window['chart_id']]
            gt = item_lookup[variant]['gt_plan']
            plan, mask, centers = select_plan_conditions(gt, identity, window['chart_id'], window['start'], config)
            plans.append(plan)
            plan_valid.append(mask)
            plan_beats.append(centers)
            decoder_groups.append(ownership[window['chart_id']])
            decoder_starts.append(window['start'])
            audio_mapping.append(audio_lookup[(variant, window['start'])])
        occurrence.append(unique[identity])
    decoder_groups = torch.tensor(decoder_groups, dtype=torch.long)
    decoder_starts = torch.tensor(decoder_starts, dtype=torch.float32)
    decoder_valid = torch.ones(len(unique), dtype=torch.bool)
    layouts['decoder'] = prepare_selection(draft.query_beats, draft.draft_valid,
        decoder_groups, decoder_starts, decoder_valid, target_beats=config['target_beats'],
        before_beats=config['decoder_z_before_beats'], after_beats=config['decoder_z_after_beats'],
        z_stride_beats=config['draft_z_stride_beats'],
        anchor_mode=config.get('draft_z_anchor_mode', 'block-center'))
    if centered:
        layouts['decoder'] = prepare_centers(draft.query_beats, draft.draft_valid,
            decoder_groups, decoder_starts[:, None]+torch.arange(-3, 20).float(), decoder_valid)
    z_layout = layouts['decoder']
    z = Memory(torch.zeros((*z_layout['beats'].shape, config['draft_latent_dim'])),
               z_layout['beats'], z_layout['valid'])
    decoder_conditions = Conditions(audio, z, torch.stack(plans), torch.stack(plan_valid),
        torch.tensor(occurrence, dtype=torch.long), torch.tensor(audio_mapping, dtype=torch.long),
        audio.valid.flatten().nonzero().flatten(),
        plan_beats=torch.stack(plan_beats) if centered else None)
    batch = Batch(songs, audio_windows, coverages, draft, groups, starts, valid,
        auxiliary['reconstruction'], auxiliary['planner'], auxiliary['window_ids'], variants, coverage_indices,
        decoder_groups, decoder_starts, decoder_valid, decoder_conditions, layouts, m_windows,
        [sum(key[0] == variant for key in audio_keys)*audio.valid.shape[1] for variant in variants])
    from maidata.draft.losses import prepare_supervision
    batch.layouts['auxiliary'] = prepare_supervision(batch.reconstruction, batch.planner)
    prepare_batch(batch, config)
    return pack(batch) if packed else batch


def make_loader(plan, config):
    workers = config['pretrain_workers']
    generator = torch.Generator().manual_seed(config['pretrain_seed'])
    return DataLoader(JointDataset(plan, config), batch_sampler=AudioBatchSampler(plan, config),
        collate_fn=partial(collate_audio, config=config, packed=True), num_workers=workers,
        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
        pin_memory=torch.device(config['pretrain_device']).type == 'cuda')
