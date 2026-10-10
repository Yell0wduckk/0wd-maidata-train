"""CPU selection of fixed Draft/Plan and online-audio geometry, without networks."""
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.draft.types import BeatMemory
from maidata.decoder.types import Memory


@dataclass
class Conditions:
    audio: Memory
    decoder_z: Memory
    plan: torch.Tensor
    plan_valid: torch.Tensor
    occurrence_mapping: torch.Tensor
    audio_mapping: torch.Tensor
    audio_indices: torch.Tensor | None = None


def collate_conditions(items, windows, config, *, audio):
    # Full audio geometry is shared by songs. Only selected frozen Z context and
    # plan tensors cross IPC; current Encoder values are supplied by the model.
    songs = {item['variant']: i for i, item in enumerate(items)}
    chart_sources = {chart: (item, group) for item in items for chart, group in item['dp']['chart_to_group'].items()}
    groups, unique, indices = {}, {}, []
    z_rows, window_groups, starts, plans, plan_valid, audio_mapping = [], [], [], [], [], []
    for window in windows:
        identity = window['identity'][0]
        if identity not in unique:
            unique[identity] = len(unique)
            item, group = chart_sources[window['chart_id']]
            dp = item['dp']
            key = (item['variant'], group)
            if key not in groups:
                groups[key] = len(z_rows)
                z_rows.append({k: dp['z'][k][group] for k in ('values', 'beats', 'valid')})
            row = dp['window_index'][identity]
            window_groups.append(groups[key])
            starts.append(window['start'])
            plans.append(dp['planner']['plan_values'][row])
            plan_valid.append(dp['planner']['valid'][row])
            audio_mapping.append(songs[item['variant']])
        indices.append(unique[identity])
    z = BeatMemory(*(pad_sequence([r[k] for r in z_rows], batch_first=True) for k in ('values','beats','valid')))
    mapping = torch.tensor(window_groups, dtype=torch.long)
    starts = torch.tensor(starts, dtype=torch.float32)
    valid = torch.ones(len(starts), dtype=torch.bool)
    count = config['decoder_z_before_beats'] + config['target_beats'] + config['decoder_z_after_beats']
    desired = starts[:, None] - config['decoder_z_before_beats'] + torch.arange(count).float() + .5
    position = (desired-z.beats[mapping, :1]).long()
    safe = position.clamp(0, z.values.shape[1]-1)
    selected_valid = (position >= 0) & (position < z.values.shape[1]) & z.valid[mapping[:, None], safe]
    selected = Memory(z.values[mapping[:, None], safe].masked_fill(~selected_valid[..., None], 0),
                      desired, selected_valid)
    return Conditions(audio, Memory(selected.values, selected.beats, selected.valid), torch.stack(plans),
        torch.stack(plan_valid), torch.tensor(indices, dtype=torch.long), torch.tensor(audio_mapping, dtype=torch.long),
        audio.valid.flatten().nonzero().flatten())
