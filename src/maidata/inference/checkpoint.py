"""Load inference weights and architecture, without training plans or source guards."""
from dataclasses import dataclass
import json

import torch

from maidata.entry import CONFIG
from maidata.model import ENCODER_KEYS, build_model, build_draft, build_planner, build_decoder


@dataclass
class Models:
    encoder: object
    draft: object
    planner: object
    decoder: object
    config: dict
    provenance: dict


def load_models(encoder_draft_checkpoint, decoder_checkpoint, *, device):
    producer = torch.load(encoder_draft_checkpoint, map_location='cpu', weights_only=False)
    consumer = torch.load(decoder_checkpoint, map_location='cpu', weights_only=False)
    if producer.get('schema') != 'maidata.encoder-draft-training.v1':
        raise ValueError('expected an Encoder/Draft stage-I checkpoint')
    if consumer.get('schema') != 'maidata.decoder-training.v1':
        raise ValueError('expected a Decoder stage-II checkpoint')
    # These are architecture/representation settings, not source or asset identities.
    keys = [k for k in producer['recipe'] if k in ENCODER_KEYS or k.startswith(('draft_', 'planner_'))]
    for key in keys:
        if key in consumer['recipe'] and json.dumps(producer['recipe'][key]) != json.dumps(consumer['recipe'][key]):
            raise ValueError(f'Encoder/Draft and Decoder representation differs: {key}')
    config = dict(CONFIG, **producer['recipe'])
    config.update(consumer['recipe'])
    modules = [build_model(config), build_draft(config), build_planner(config), build_decoder(config)]
    for name, module in zip(('encoder', 'draft', 'planner'), modules):
        prefix = name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in producer['model_state'].items()
                                if k.startswith(prefix)}, strict=True)
    modules[-1].load_state_dict(consumer['model_state'], strict=True)
    for model in modules:
        model.to(device).eval().requires_grad_(False)
        for module in model.modules():
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs = False
    def describe(path, payload):
        return dict(path=str(path), epoch=payload.get('completed_epochs'), step=payload.get('step'),
                    source=payload.get('source'))
    return Models(*modules, config, dict(encoder_draft=describe(encoder_draft_checkpoint, producer),
                                        decoder=describe(decoder_checkpoint, consumer)))
