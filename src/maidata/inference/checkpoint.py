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
    decoder_encoder: object = None


def load_models(encoder_draft_checkpoint, decoder_checkpoint, *, device):
    producer = torch.load(encoder_draft_checkpoint, map_location='cpu', weights_only=False)
    consumer = torch.load(decoder_checkpoint, map_location='cpu', weights_only=False)
    if producer.get('schema') not in ('maidata.encoder-draft-training.v1','maidata.encoder-draft-training.v2'):
        raise ValueError('expected an Encoder/Draft stage-I checkpoint')
    if (consumer.get('schema') != 'maidata.decoder-training.v4'
            or consumer.get('architecture') != 'maidata.decoder-causal-cont-tick384.v2'):
        raise ValueError('expected a v4 causal CONT Decoder checkpoint with normalized-plan architecture v2; older architectures are not resume-compatible')
    # These are architecture/representation settings, not source or asset identities.
    keys = [k for k in producer['recipe'] if k in ENCODER_KEYS or k.startswith(('draft_', 'planner_'))]
    for key in keys:
        if key in consumer['recipe'] and json.dumps(producer['recipe'][key]) != json.dumps(consumer['recipe'][key]):
            raise ValueError(f'Encoder/Draft and Decoder representation differs: {key}')
    config = dict(CONFIG, **producer['recipe'])
    config.update(consumer['recipe'])
    config['decoder_train_audio_scope'] = consumer['recipe'].get('decoder_train_audio_scope', 'full_song')
    modules = [build_model(config), build_draft(config), build_planner(config), build_decoder(config)]
    for name, module in zip(('encoder', 'draft', 'planner'), modules):
        prefix = name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in producer['model_state'].items()
                                if k.startswith(prefix)}, strict=True)
    decoder_encoder = None
    if consumer['schema'] == 'maidata.decoder-training.v4':
        decoder_encoder = build_model(config)
        # The stage-II encoder is independently trained with M. It must never
        # replace the stage-I encoder used by the fixed Draft/Planner.
        torch.nn.ModuleDict(dict(encoder=decoder_encoder, decoder=modules[-1])).load_state_dict(
            consumer['model_state'], strict=True)
    for model in modules + ([decoder_encoder] if decoder_encoder is not None else []):
        model.to(device).eval().requires_grad_(False)
        for module in model.modules():
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs = False
    def describe(path, payload):
        return dict(path=str(path), schema=payload['schema'], epoch=payload.get('completed_epochs'), step=payload.get('step'),
                    source=payload.get('source'))
    return Models(*modules, config, dict(encoder_draft=describe(encoder_draft_checkpoint, producer),
                                        decoder=describe(decoder_checkpoint, consumer),
                                        training_plan_source=consumer['recipe'].get('decoder_train_plan_source', 'gt'),
                                        inference_plan_source='predicted',
                                        decoder_audio_scope=config['decoder_train_audio_scope'],
                                        decoder_audio_encoder='stage_ii' if decoder_encoder is not None else 'stage_i'),
                  decoder_encoder=decoder_encoder)
