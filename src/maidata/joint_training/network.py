"""One trainable E/D/R/P/M composition with shared, exact local audio."""
from dataclasses import replace
from contextlib import nullcontext

import torch
from torch import nn

from maidata.model import (build_model, build_draft, build_reconstruction,
                           build_planner, build_decoder, decoder_time_distributions)
from maidata.decoder.types import Memory
from maidata.draft.windows import assemble_full_p0, select_windows
from maidata.draft.supervision import predict_auxiliary, auxiliary_losses


class JointTrainer(nn.Module):
    """Own each component once; no frozen teacher or learned feature cache."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = build_model(config)
        self.draft = build_draft(config)
        self.reconstruction = build_reconstruction(config)
        self.planner = build_planner(config)
        self.decoder = build_decoder(config)
        self.distributions = decoder_time_distributions(config)
        for module in self.modules():
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs = False

    def forward_shared(self, batch, *, auxiliary=True, draft_grad=True, recorder=None, fixed_z=None):
        """E and D execute once; both consumers retain their real graphs.

        Every M audio bank is the public output of its original local Encoder
        window. D alone receives the centers of the coverage windows, excluding
        the extra complete supervised tail.
        """
        config = self.config
        section = lambda name: nullcontext() if recorder is None else recorder.section(name)
        with section('encoder_forward'):
            local = self.encoder(batch.songs, batch.audio_windows).P0
        if fixed_z is not None:
            conditions = replace(batch.conditions,
                audio=Memory(local.values, local.beats, local.valid), decoder_z=fixed_z)
            return {}, conditions, (None, None, None, local)
        rows = batch.coverage_audio_indices
        coverage_local = replace(local, **{name: getattr(local, name).index_select(0, rows)
            for name in ('values', 'beats', 'valid')})
        bank = assemble_full_p0(coverage_local,
            batch.audio_windows.target_start_beats.index_select(0, rows),
            batch.audio_windows.window_to_song.index_select(0, rows), batch.coverages,
            p0_rate=config['fpn_rates'][0], layout=batch.layouts['p0'])
        with section('draft_forward'), torch.set_grad_enabled(torch.is_grad_enabled() and draft_grad):
            z = self.draft(bank, batch.draft_inputs)
        reconstruction = planner = None
        terms = {}
        if auxiliary:
            with section('auxiliary_forward'):
                reconstruction, planner = predict_auxiliary(z, batch.groups, batch.starts, batch.valid,
                    self.reconstruction, self.planner, config, layouts=batch.layouts,
                    queries=batch.reconstruction.queries)
                terms = auxiliary_losses(reconstruction, planner, batch.reconstruction, batch.planner, config,
                                         prepared=batch.layouts.get('auxiliary'))
        selected = select_windows(z, batch.decoder_groups, batch.decoder_starts, batch.decoder_valid,
            target_beats=config['target_beats'], before_beats=config['decoder_z_before_beats'],
            after_beats=config['decoder_z_after_beats'], z_stride_beats=config['draft_z_stride_beats'],
            layout=batch.layouts['decoder']).memory
        conditions = replace(batch.conditions,
            audio=Memory(local.values, local.beats, local.valid),
            decoder_z=Memory(selected.values, selected.beats, selected.valid))
        return terms, conditions, (reconstruction, planner, z, local)

    def forward(self, batch):
        return self.forward_shared(batch)

    def memory(self, conditions, audio_kv, indices, audio_layout=None):
        unique = conditions.occurrence_mapping[indices]
        z = conditions.decoder_z
        return self.decoder.prepare_memory(audio_kv, conditions.audio_mapping[unique],
            Memory(z.values[unique], z.beats[unique], z.valid[unique]), conditions.plan[unique],
            conditions.plan_valid[unique], audio_layout=audio_layout)
