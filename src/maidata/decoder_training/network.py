"""Stage II owns only M; frozen public conditions arrive from the data layer."""
from dataclasses import replace
import torch
from torch import nn

from maidata.model import build_decoder
from maidata.decoder.types import Memory


class DecoderTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = build_decoder(config)

    def conditions(self, batch):
        # Cached BF16 outputs can also feed FP32 training. Autocast training
        # keeps the stored dtype, without doubling bank transport/storage.
        conditions = batch.conditions
        device = conditions.audio.values.device.type
        if torch.is_autocast_enabled(device):
            return conditions
        dtype = next(self.decoder.parameters()).dtype
        return replace(conditions, audio=replace(conditions.audio, values=conditions.audio.values.to(dtype)),
            decoder_z=replace(conditions.decoder_z, values=conditions.decoder_z.values.to(dtype)),
            plan=conditions.plan.to(dtype))

    def memory(self, conditions, audio_kv, indices, audio_layout=None):
        unique = conditions.occurrence_mapping[indices]
        z = conditions.decoder_z
        return self.decoder.prepare_memory(audio_kv, conditions.audio_mapping[unique],
            Memory(z.values[unique], z.beats[unique], z.valid[unique]), conditions.plan[unique],
            conditions.plan_valid[unique], audio_layout=audio_layout)
