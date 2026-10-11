"""Load inference weights and architecture, without training plans or source guards."""
from dataclasses import dataclass

import torch

from maidata.entry import CONFIG
from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.decoder_training.producer import teacher_configuration


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


def load_models(encoder_draft_checkpoint, decoder_checkpoint, *, device, z_cross_attention=None, plan_source='predicted'):
    consumer = torch.load(decoder_checkpoint, map_location='cpu', weights_only=False)
    if (consumer.get('schema') != 'maidata.decoder-training.v4'
            or consumer.get('architecture') != 'maidata.decoder-causal-cont-tick384.v2'):
        raise ValueError('expected a v4 causal CONT Decoder checkpoint with normalized-plan architecture v2; older architectures are not resume-compatible')
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
    teacher = teacher_configuration(producer) if producer is not None else None
    config = dict(CONFIG, **consumer['recipe'])
    config['decoder_train_audio_scope'] = consumer['recipe'].get('decoder_train_audio_scope', 'full_song')
    if teacher is not None:
        if teacher['target_beats'] != config['target_beats']:
            raise ValueError('teacher Planner and M target-window spans differ')
        if actual_z and teacher['draft_latent_dim'] != config['draft_latent_dim']:
            raise ValueError('teacher Z width differs from the trained M input projection; old Z cannot be expanded')
    modules = ([build_model(teacher), build_draft(teacher), build_planner(teacher)]
               if producer is not None else [None, None, None]) + [build_decoder(config)]
    for name, module in zip(('encoder', 'draft', 'planner'), modules):
        if module is None:
            continue
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
        return dict(path=str(path), schema=payload['schema'], epoch=payload.get('completed_epochs'), step=payload.get('step'),
                    source=payload.get('source'))
    return Models(*modules, config, dict(encoder_draft=describe(encoder_draft_checkpoint, producer) if producer is not None else None,
                                        decoder=describe(decoder_checkpoint, consumer),
                                        training_plan_source=consumer['recipe'].get('decoder_train_plan_source', 'gt'),
                                        inference_plan_source=plan_source, z_cross_attention=actual_z,
                                        decoder_audio_scope=config['decoder_train_audio_scope'],
                                        decoder_audio_encoder='stage_ii' if decoder_encoder is not None else 'stage_i'),
                  decoder_encoder=decoder_encoder, teacher_config=teacher)
