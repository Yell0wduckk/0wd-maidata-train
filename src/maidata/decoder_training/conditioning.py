"""CPU selection of public cached conditions, independent of labels and networks."""
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.draft.types import BeatMemory
from maidata.draft.layout import prepare_selection
from maidata.draft.windows import select_windows
from maidata.decoder.types import Memory


@dataclass
class Conditions:
    audio: Memory
    decoder_z: Memory
    plan: torch.Tensor
    plan_valid: torch.Tensor
    occurrence_mapping: torch.Tensor
    audio_mapping: torch.Tensor


def collate_conditions(items, windows, config):
    # Full P0 is shared by songs. Only selected Z context/plan tensors cross IPC.
    audio = Memory(*(pad_sequence([item['p0'][k] for item in items], batch_first=True)
                     for k in ('values', 'beats', 'valid')))
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
    layout = prepare_selection(z.beats, z.valid, mapping, starts, valid, target_beats=config['target_beats'],
        before_beats=config['decoder_z_before_beats'], after_beats=config['decoder_z_after_beats'])
    selected = select_windows(z, mapping, starts, valid, target_beats=config['target_beats'],
        before_beats=config['decoder_z_before_beats'], after_beats=config['decoder_z_after_beats'], layout=layout).memory
    return Conditions(audio, Memory(selected.values, selected.beats, selected.valid), torch.stack(plans),
        torch.stack(plan_valid), torch.tensor(indices, dtype=torch.long), torch.tensor(audio_mapping, dtype=torch.long))
