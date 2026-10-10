"""Training composition owns independent E/D/R/P; networks own no peers."""
from torch import nn

from maidata.model import build_model, build_draft, build_reconstruction, build_planner
from maidata.draft.windows import assemble_full_p0
from maidata.draft.supervision import predict_auxiliary, auxiliary_losses


class Pretrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = build_model(config)
        self.draft = build_draft(config)
        self.reconstruction = build_reconstruction(config)
        self.planner = build_planner(config)

    def forward(self, batch):
        local = self.encoder(batch.songs, batch.audio_windows)
        bank = assemble_full_p0(local.P0, batch.audio_windows.target_start_beats,
            batch.audio_windows.window_to_song, batch.coverages, p0_rate=self.config['fpn_rates'][0])
        z = self.draft(bank, batch.draft_inputs)
        r, p = predict_auxiliary(z, batch.groups, batch.starts, batch.valid,
                                self.reconstruction, self.planner, self.config)
        return auxiliary_losses(r, p, batch.reconstruction, batch.planner, self.config), (r, p, z)
