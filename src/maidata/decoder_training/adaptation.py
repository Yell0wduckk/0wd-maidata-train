"""Explicit weight-only initialization and frozen public-P0 adaptation contract."""
import json
import torch

from maidata.model import ENCODER_KEYS, encoder_config, with_legacy_defaults
from maidata.training.frozen_p0 import FrozenP0
from .checkpoint import SCHEMA, ARCHITECTURE


def prepare(config, resume=None):
    path = resume or config['decoder_train_init_checkpoint']
    if not path:
        raise ValueError('cached adaptation requires --init-checkpoint (or its own --resume checkpoint)')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('Decoder adaptation requires a v4 normalized-plan CONT v2 checkpoint')
    if resume and payload['recipe'].get('decoder_train_audio_mode', 'joint') != 'cached':
        raise ValueError('use --init-checkpoint for joint M weights; --resume requires an adaptation checkpoint')
    if not config['decoder_train_p0_cache']:
        raise ValueError('cached adaptation requires --p0-cache including its frozen Encoder snapshot')
    p0 = FrozenP0(config['decoder_train_p0_cache'])
    saved = with_legacy_defaults(payload['recipe'])
    if saved.get('decoder_train_audio_scope') != 'window':
        raise ValueError('adaptation must preserve the source Decoder local-window audio scope')
    encoder = encoder_config(saved)
    if json.loads(json.dumps(encoder)) != p0.encoder_config:
        raise ValueError('initial Decoder Encoder configuration differs from frozen P0')
    # M architecture comes from the selected checkpoint; the new optimization
    # recipe, cache paths, Z width and Z switch belong to this adaptation run.
    architecture = {k: v for k, v in saved.items() if k in ENCODER_KEYS or
                    k.startswith('decoder_') and not k.startswith('decoder_train_')}
    architecture.pop('decoder_z_cross_attention', None)
    architecture['planner_span_beats'] = saved.get('planner_span_beats', 16)
    architecture['planner_layout'] = saved.get('planner_layout', 'legacy')
    architecture['planner_stride_beats'] = saved.get('planner_stride_beats')
    if architecture['planner_layout'] in ('centered-overlap', 'window-single'):
        # These coordinates determine M's 23-Z layout even though the
        # adaptation does not execute R. Preserve the source representation
        # before FrozenCache checks it; its Z width remains consumer-selected.
        architecture.update(draft_z_anchor_mode=saved.get('draft_z_anchor_mode', 'block-center'),
            draft_z_stride_beats=saved.get('draft_z_stride_beats', 1),
            reconstruction_span_beats=saved.get('reconstruction_span_beats'))
    architecture.update(encoder)
    return dict(config, **architecture), payload


def initialize(model, payload, path):
    state = payload['model_state']
    # Cache E is kept in checkpoints for standalone inference. Never substitute
    # an unrelated M Encoder while retaining P0 produced by a different one.
    encoder = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
    actual = model.encoder.state_dict()
    if encoder.keys() != actual.keys() or any(not torch.equal(actual[k], v) for k, v in encoder.items()):
        raise ValueError('initial Decoder Encoder weights differ from frozen P0 snapshot')
    decoder = {k.removeprefix('decoder.'): v for k, v in state.items() if k.startswith('decoder.')}
    current = model.decoder.state_dict()
    if decoder.keys() != current.keys():
        raise ValueError('initial Decoder parameter inventory differs')
    reset = []
    # No-Z M still owns the unused Z Linear. Reset exactly that projection if
    # it was untrained or its input width changed; every other weight is strict.
    if (not payload['recipe']['decoder_z_cross_attention'] or
            decoder['condition.z.weight'].shape != current['condition.z.weight'].shape):
        reset = ['condition.z.weight', 'condition.z.bias']
        for key in reset:
            decoder[key] = current[key]
    model.decoder.load_state_dict(decoder, strict=True)
    model.initialization = dict(path=str(path), schema=payload['schema'], architecture=payload['architecture'],
        completed_epochs=payload.get('completed_epochs'), step=payload.get('step'), source=payload.get('source'),
        reset_decoder_parameters=reset, optimizer='fresh', schedule='fresh')


def bind_conditions(cache, p0):
    if cache.producer.get('audio_producer') != p0.identity:
        raise ValueError('fixed Z must come from the Drafter trained on this frozen P0 producer')
