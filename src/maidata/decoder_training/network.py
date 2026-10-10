"""Stage II trains a fresh audio Encoder and M with fixed Draft/Plan conditions."""
from dataclasses import replace
import torch
from torch import nn

from maidata.model import build_model, build_decoder, decoder_time_distributions
from maidata.decoder.types import Memory


class DecoderTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_model(config)
        self.decoder = build_decoder(config)
        self.distributions = decoder_time_distributions(config)
        for module in self.modules():
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs = False

    def conditions(self, batch):
        inputs = batch.online_audio
        local = self.encoder(inputs.songs, inputs.windows,
                             window_chunks=inputs.windows.layout['training_chunks'])
        audio = local.P0
        conditions = replace(batch.conditions, audio=Memory(audio.values, audio.beats, audio.valid))
        # Fixed D/P caches may have BF16 values while this run uses FP32.
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
