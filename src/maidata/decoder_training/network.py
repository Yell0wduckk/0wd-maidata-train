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
    decoder_z: object
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
            batch.audio_windows.window_to_song, batch.coverages, p0_rate=self.config['fpn_rates'][0], layout=batch.layouts['p0'])
        z = self.draft(bank, batch.draft_inputs)
        # Duplicate occurrences share target-only Planner predictions.
        index = batch.layouts['unique']
        groups, starts, valid = batch.groups[index], batch.starts[index], batch.valid[index]
        target = select_windows(z, groups, starts, valid, target_beats=self.config['target_beats'],
                                before_beats=0, after_beats=0, layout=batch.layouts['planner'])
        plan = self.planner(target)
        decoder_z = select_windows(z, groups, starts, valid, target_beats=self.config['target_beats'],
            before_beats=self.config['decoder_z_before_beats'], after_beats=self.config['decoder_z_after_beats'],
            layout=batch.layouts['decoder']).memory
        return Conditions(bank, z, decoder_z, groups, starts, valid, plan.plan_values, plan.valid,
            batch.layouts['occurrences'], batch.draft_inputs.audio_mapping)

    def memory(self, conditions, audio_kv, indices, audio_layout=None):
        unique = conditions.occurrence_mapping[indices]
        groups = conditions.groups[unique]
        z = conditions.decoder_z
        return self.decoder.prepare_memory(audio_kv, conditions.audio_mapping[groups],
            Memory(z.values[unique], z.beats[unique], z.valid[unique]), conditions.plan[unique], conditions.plan_valid[unique],
            audio_layout=audio_layout)
