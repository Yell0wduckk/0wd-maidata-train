"""Offline exporters own E/D/P; Decoder training never constructs this module."""
from torch import nn
import torch
from maidata.entry import CONFIG
from maidata.model import build_model, build_draft, build_planner
from maidata.training.checkpoint import SCHEMA as TEACHER_SCHEMA


def teacher_configuration(payload):
    """The producer recipe owns E_D/D/P, independently of the M recipe."""
    if payload.get('schema') != TEACHER_SCHEMA:
        raise ValueError('teacher export/load requires a split Drafter stage-I v3 checkpoint')
    config = dict(CONFIG, **payload['recipe'])
    if config['draft_latent_dim'] != config['draft_dim']:
        raise ValueError('split Drafter Z must equal the final normalized draft width')
    return config


def load_export_producer(config, binding):
    """Read teacher architecture before construction; keep export runtime separate."""
    from .checkpoint import load_producer
    payload = torch.load(config['decoder_train_checkpoint'], map_location='cpu', weights_only=False)
    teacher = teacher_configuration(payload)
    # Paths, workers and export precision belong to this operation, not E_D.
    teacher.update({k: v for k, v in config.items() if k.startswith('decoder_train_') or k.startswith('_')})
    model = FrozenProducer(teacher)
    identity = load_producer(model, teacher, binding, payload=payload)
    return model, identity, teacher


class FrozenProducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_model(config)
        self.draft = build_draft(config)
        self.planner = build_planner(config)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)
