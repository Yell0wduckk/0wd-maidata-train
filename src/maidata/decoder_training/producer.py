"""Offline exporters own E/D/P; Decoder training never constructs this module."""
from torch import nn
from maidata.model import build_model, build_draft, build_planner


class FrozenProducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_model(config)
        self.draft = build_draft(config)
        self.planner = build_planner(config)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)
