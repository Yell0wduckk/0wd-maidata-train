"""Load inference weights and architecture, without training plans or source guards."""
from dataclasses import dataclass

import torch

from maidata.entry import CONFIG
from maidata.model import build_model, build_draft, build_planner, build_decoder
from .input_contract import check_shared_grid_config


@dataclass
class Models:
    encoder: object
    draft: object
    planner: object
    decoder: object
    config: dict
    provenance: dict
    decoder_encoder: object = None
    teacher_config: dict | None = None


class JointCheckpoint:
    """Load required components strictly; training version tags are provenance."""

    def __init__(self, checkpoint, *, z_cross_attention=None, plan_source='predicted'):
        from pathlib import Path
        self.path = Path(checkpoint).resolve()
        stat = self.path.stat()
        self.identity = dict(path=str(self.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        self.payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        payload = self.payload
        # Training schema/architecture labels and auxiliary heads do not decide
        # inference compatibility. Each consumed component is constructed from
        # its saved recipe and load_state_dict(strict=True) checks real weights.
        if plan_source not in ('predicted', 'gt', 'none'):
            raise ValueError('plan_source must be predicted, gt or none')
        self.config = dict(CONFIG, **payload['recipe'])
        # Historical recipes must not inherit the new P0 normalization default.
        self.config['fpn_p0_normalization'] = payload['recipe'].get('fpn_p0_normalization', False)
        saved_z = self.config['decoder_z_cross_attention']
        self.actual_z = saved_z if z_cross_attention is None else z_cross_attention
        if type(self.actual_z) is not bool or self.actual_z and not saved_z:
            raise ValueError('cannot enable Z on a checkpoint trained without its projection')
        self.config['decoder_train_audio_scope'] = 'window'
        description = dict(path=str(checkpoint), schema=payload.get('schema'), architecture=payload.get('architecture'),
            epoch=payload.get('completed_epochs'), step=payload.get('step'), source=payload.get('source'))
        self.provenance = dict(joint=description, encoder_draft=description, decoder=description,
            training_plan_source=self.config.get('joint_train_plan_source', 'gt'),
            inference_plan_source=plan_source, z_cross_attention=self.actual_z,
            decoder_audio_scope='window', decoder_audio_encoder='shared_joint')

    def models(self, *, device, components=('encoder', 'draft', 'planner', 'decoder')):
        factories = dict(encoder=build_model, draft=build_draft, planner=build_planner, decoder=build_decoder)
        if not set(components) <= factories.keys():
            raise ValueError('unknown joint checkpoint component')
        modules = {name: None for name in factories}
        for name in components:
            model = factories[name](self.config)
            prefix = name + '.'
            model.load_state_dict({k[len(prefix):]: v for k, v in self.payload['model_state'].items()
                                   if k.startswith(prefix)}, strict=True)
            model.to(device).eval().requires_grad_(False)
            for module in model.modules():
                if hasattr(module, 'validate_inputs'):
                    module.validate_inputs = False
            modules[name] = model
        config = dict(self.config, decoder_z_cross_attention=self.actual_z)
        if modules['decoder'] is not None:
            modules['decoder'].condition.z_cross_attention = self.actual_z
        return Models(**modules, config=config, provenance=self.provenance,
                      decoder_encoder=modules['encoder'], teacher_config=config)


def load_joint_models(checkpoint, *, device, z_cross_attention=None, plan_source='predicted', components=None):
    """Default online loader; cached inference requests components separately."""
    checkpoint = JointCheckpoint(checkpoint, z_cross_attention=z_cross_attention, plan_source=plan_source)
    return checkpoint.models(device=device, **(dict(components=components) if components is not None else {}))


def load_models(encoder_draft_checkpoint, decoder_checkpoint, *, device, z_cross_attention=None, plan_source='predicted'):
    consumer = torch.load(decoder_checkpoint, map_location='cpu', weights_only=False)
    if plan_source not in ('predicted', 'gt', 'none'):
        raise ValueError('plan_source must be predicted, gt or none')
    saved_z = consumer['recipe']['decoder_z_cross_attention']
    actual_z = saved_z if z_cross_attention is None else z_cross_attention
    if type(actual_z) is not bool or actual_z and not saved_z:
        raise ValueError('cannot enable Z on a checkpoint trained without its projection')
    producer = None
    if actual_z or plan_source == 'predicted':
        if not encoder_draft_checkpoint:
            raise ValueError('Z or predicted plan requires an Encoder/Draft checkpoint')
        producer = torch.load(encoder_draft_checkpoint, map_location='cpu', weights_only=False)
    teacher = None
    if producer is not None:
        # This inference path must not call the training export loader's schema
        # gate. D/P architecture and cross-model dimensions are still checked.
        teacher = dict(CONFIG, **producer['recipe'])
        teacher['fpn_p0_normalization'] = producer['recipe'].get('fpn_p0_normalization', False)
        if teacher['draft_latent_dim'] != teacher['draft_dim']:
            raise ValueError('split Drafter Z must equal the final normalized draft width')
    config = dict(CONFIG, **consumer['recipe'])
    config['fpn_p0_normalization'] = consumer['recipe'].get('fpn_p0_normalization', False)
    config['decoder_train_audio_scope'] = consumer['recipe'].get('decoder_train_audio_scope', 'full_song')
    if teacher is not None:
        check_shared_grid_config(teacher, config)
        if teacher['target_beats'] != config['target_beats']:
            raise ValueError('teacher Planner and M target-window spans differ')
        if actual_z and teacher['draft_latent_dim'] != config['draft_latent_dim']:
            raise ValueError('teacher Z width differs from the trained M input projection; old Z cannot be expanded')
        if actual_z and teacher['draft_z_stride_beats'] != config['draft_z_stride_beats']:
            raise ValueError('teacher Z spacing differs from the trained M conditions')
    modules = ([build_model(teacher), build_draft(teacher), build_planner(teacher)]
               if producer is not None else [None, None, None]) + [build_decoder(config)]
    for name, module in zip(('encoder', 'draft', 'planner'), modules):
        if module is None:
            continue
        prefix = name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in producer['model_state'].items()
                                if k.startswith(prefix)}, strict=True)
    decoder_encoder = build_model(config)
    # E/M can be jointly trained or adapted with a frozen E; in both cases load
    # the saved Encoder, never use a version label to skip loading actual weights.
    # Other training-only components are irrelevant to these two consumers.
    for name, module in (('encoder', decoder_encoder), ('decoder', modules[-1])):
        prefix = name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in consumer['model_state'].items()
                                if k.startswith(prefix)}, strict=True)
    for model in modules + ([decoder_encoder] if decoder_encoder is not None else []):
        if model is None:
            continue
        model.to(device).eval().requires_grad_(False)
        for module in model.modules():
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs = False
    # Load the original structure strictly before applying the inference ablation.
    config['decoder_z_cross_attention'] = actual_z
    modules[-1].condition.z_cross_attention = actual_z
    def describe(path, payload):
        return dict(path=str(path), schema=payload.get('schema'), architecture=payload.get('architecture'),
                    epoch=payload.get('completed_epochs'), step=payload.get('step'),
                    source=payload.get('source'))
    return Models(*modules, config, dict(encoder_draft=describe(encoder_draft_checkpoint, producer) if producer is not None else None,
                                        decoder=describe(decoder_checkpoint, consumer),
                                        training_plan_source=consumer['recipe'].get('decoder_train_plan_source', 'gt'),
                                        inference_plan_source=plan_source, z_cross_attention=actual_z,
                                        decoder_audio_scope=config['decoder_train_audio_scope'],
                                        decoder_audio_encoder='stage_ii' if decoder_encoder is not None else 'stage_i'),
                  decoder_encoder=decoder_encoder, teacher_config=teacher)
