"""Small CPU-only encoder inputs; no dataset or training configuration."""

import pytest
import torch

from maidata.audio.types import SongInputs, WindowInputs
from maidata.entry import CONFIG


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def encoder_config():
    return dict(CONFIG, mel_bins=7, samples_per_beat=5, d_model=24, heads=3,
                ffn_dim=40, stage_depths=(2, 1, 1), temporal_kernel_size=3,
                stem_channels=4, frequency_dim=6, frequency_queries=2,
                fpn_rates=(3, 2, 1), target_beats=3, audio_before_beats=2,
                audio_after_beats=1, dropout=0.0)


@pytest.fixture
def audio_inputs(encoder_config):
    torch.manual_seed(14)
    config = encoder_config
    r, m = config["samples_per_beat"], config["mel_bins"]
    centers = (torch.arange(10).float() - 2.5).expand(2, -1).clone()
    grid = centers[..., None] - 0.5 + (torch.arange(r).float() + 0.5) / r
    valid = (grid >= torch.tensor([-1.6, -2.7])[:, None, None]) & (
        grid < torch.tensor([4.9, 6.3])[:, None, None]
    )
    mel = torch.randn(2, 10, r, m)
    center_valid = (centers >= torch.tensor([-1.6, -2.7])[:, None]) & (
        centers < torch.tensor([4.9, 6.3])[:, None]
    )
    aux_centers = torch.randn(2, 10, 32).masked_fill(~center_valid[..., None], 0)
    aux_grid = torch.randn(2, 10, r, 32).masked_fill(~valid[..., None], 0)
    song = SongInputs(mel, valid, centers, aux_centers)
    starts, mapping = torch.tensor([-1, 1, 0]), torch.tensor([0, 0, 1])
    length = r * (config["target_beats"] + config["audio_before_beats"] + config["audio_after_beats"])
    slices = [(int(s) - config["audio_before_beats"] + 3) * r for s in starts]
    def windows(value):
        flattened = value.flatten(1, 2)
        return torch.stack([flattened[b, i:i + length] for b, i in zip(mapping, slices)])
    window = WindowInputs(windows(mel), windows(aux_grid), windows(grid), windows(valid), starts, mapping)
    return song, window
