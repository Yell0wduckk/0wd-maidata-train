"""CPU selection of fixed Draft/Plan and online-audio geometry, without networks."""
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.draft.types import BeatMemory
from maidata.draft.spacing import integer_offsets
from maidata.decoder.types import Memory
from .gt_plan_cache import select_plan_window


@dataclass
class Conditions:
    audio: Memory
    decoder_z: Memory
    plan: torch.Tensor
    plan_valid: torch.Tensor
    occurrence_mapping: torch.Tensor
    audio_mapping: torch.Tensor
    audio_indices: torch.Tensor | None = None
    plan_beats: torch.Tensor | None = None


def collate_conditions(items, windows, config, *, audio, audio_keys=None):
    # Full audio geometry is shared by songs. Only selected frozen Z context and
    # plan tensors cross IPC; current Encoder values are supplied by the model.
    songs = {item['variant']: i for i, item in enumerate(items)}
    local_audio = None if audio_keys is None else {key: i for i, key in enumerate(audio_keys)}
    use_z = config['decoder_z_cross_attention']
    chart_sources = ({chart: (item, group) for item in items for chart, group in item['dp']['chart_to_group'].items()}
                     if use_z else {w['chart_id']: (item, 0) for item in items for w in item['windows']})
    groups, unique, indices = {}, {}, []
    z_rows, window_groups, starts, plans, plan_valid, plan_beats, audio_mapping = [], [], [], [], [], [], []
    for window in windows:
        identity = window['identity'][0]
        if identity not in unique:
            unique[identity] = len(unique)
            item, group = chart_sources[window['chart_id']]
            dp = item['dp']
            key = (item['variant'], group)
            if use_z and key not in groups:
                groups[key] = len(z_rows)
                if dp['z']['values'].shape[-1] != config['draft_latent_dim']:
                    raise ValueError('M Z projection width differs from the frozen teacher Z')
                z_rows.append({k: dp['z'][k][group] for k in ('values', 'beats', 'valid')})
            planner = item['gt_plan'] if config['decoder_train_plan_source'] == 'gt' else dp['planner']
            lookup = None if config['decoder_train_plan_source'] == 'gt' else dp['window_index']
            if config.get('planner_layout', 'legacy') == 'centered-overlap':
                from .gt_plan_cache import select_plan_conditions
                plan, mask, centers = select_plan_conditions(planner, identity, window['chart_id'], window['start'],
                                                            config, window_index=lookup)
            else:
                plan, mask = select_plan_window(planner, identity, window['chart_id'], window['start'],
                                                config, window_index=lookup)
                centers = None
            if use_z:
                window_groups.append(groups[key])
            starts.append(window['start'])
            plans.append(plan)
            plan_valid.append(mask)
            plan_beats.append(centers)
            audio_mapping.append(songs[item['variant']] if local_audio is None else
                                 local_audio[(item['variant'], window['start'])])
        indices.append(unique[identity])
    if not use_z:
        integer_context = config.get('planner_layout', 'legacy') in ('centered-overlap', 'window-single')
        offsets = (torch.tensor(integer_offsets(-3, config['target_beats']+3, config['draft_z_stride_beats']))
                   if integer_context else torch.tensor((-2, 2, 6, 10, 14, 18)))
        slots = len(offsets) if integer_context else (6 if config.get('planner_span_beats', 16) == 4 else 0)
        beats = torch.tensor(starts, dtype=torch.float32)[:, None] + offsets
        z = Memory(torch.zeros((len(starts), slots, config['draft_latent_dim'])),
                   beats[:, :slots], torch.zeros((len(starts), slots), dtype=torch.bool))
        return Conditions(audio, z, torch.stack(plans), torch.stack(plan_valid),
            torch.tensor(indices, dtype=torch.long), torch.tensor(audio_mapping, dtype=torch.long),
            audio.valid.flatten().nonzero().flatten(),
            None if plan_beats[0] is None else torch.stack(plan_beats))
    z = BeatMemory(*(pad_sequence([r[k] for r in z_rows], batch_first=True) for k in ('values','beats','valid')))
    mapping = torch.tensor(window_groups, dtype=torch.long)
    starts = torch.tensor(starts, dtype=torch.float32)
    valid = torch.ones(len(starts), dtype=torch.bool)
    from maidata.decoder.batching import select_condition_windows
    selected = select_condition_windows(z, mapping, starts, valid, config).memory
    return Conditions(audio, Memory(selected.values, selected.beats, selected.valid), torch.stack(plans),
        torch.stack(plan_valid), torch.tensor(indices, dtype=torch.long), torch.tensor(audio_mapping, dtype=torch.long),
        audio.valid.flatten().nonzero().flatten(),
        None if plan_beats[0] is None else torch.stack(plan_beats))
