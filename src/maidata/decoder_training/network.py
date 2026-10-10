"""Stage II owns frozen E/D/P and trainable M; each network remains independent."""
from dataclasses import dataclass
import torch
from torch import nn

from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.draft.windows import assemble_full_p0, select_windows
from maidata.decoder.types import Memory


@dataclass
class Conditions:
    audio: object
    z: object
    groups: torch.Tensor
    starts: torch.Tensor
    valid: torch.Tensor
    plan: torch.Tensor
    plan_valid: torch.Tensor
    occurrence_mapping: torch.Tensor
    audio_mapping: torch.Tensor


class DecoderTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.draft, self.planner = build_model(config), build_draft(config), build_planner(config)
        self.decoder = build_decoder(config)
        for module in (self.encoder, self.draft, self.planner):
            module.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        for module in (self.encoder, self.draft, self.planner):
            module.eval()
        return self

    @torch.no_grad()
    def produce(self, batch):
        local = self.encoder(batch.songs, batch.audio_windows)
        bank = assemble_full_p0(local.P0, batch.audio_windows.target_start_beats,
            batch.audio_windows.window_to_song, batch.coverages, p0_rate=self.config['fpn_rates'][0])
        z = self.draft(bank, batch.draft_inputs)
        # Duplicate occurrences share target-only Planner predictions.
        keys, mapping, unique = {}, [], []
        for i, key in enumerate(zip(batch.groups.tolist(), batch.starts.tolist())):
            if key not in keys:
                keys[key] = len(unique)
                unique.append(i)
            mapping.append(keys[key])
        index = torch.tensor(unique, device=batch.groups.device)
        groups, starts, valid = batch.groups[index], batch.starts[index], batch.valid[index]
        target = select_windows(z, groups, starts, valid, target_beats=self.config['target_beats'],
                                before_beats=0, after_beats=0)
        plan = self.planner(target)
        return Conditions(bank, z, groups, starts, valid, plan.plan_values, plan.valid,
            torch.tensor(mapping, device=groups.device), batch.draft_inputs.audio_mapping)

    def memory(self, conditions, audio_kv, indices):
        unique = conditions.occurrence_mapping[indices]
        groups = conditions.groups[unique]
        z = select_windows(conditions.z, groups, conditions.starts[unique], conditions.valid[unique],
            target_beats=self.config['target_beats'], before_beats=self.config['decoder_z_before_beats'],
            after_beats=self.config['decoder_z_after_beats']).memory
        return self.decoder.prepare_memory(audio_kv, conditions.audio_mapping[groups],
            Memory(z.values, z.beats, z.valid), conditions.plan[unique], conditions.plan_valid[unique])
