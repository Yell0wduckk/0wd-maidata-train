"""Joint recipe and explicit adapters to the two established data/loss APIs."""
import math
from pathlib import Path


TRAIN_KEYS = ('initialization', 'optimizer', 'learning_rate', 'weight_decay',
              'gradient_clip', 'warmup_steps', 'scheduler_epochs', 'stop_after_epochs',
              'precision', 'workers', 'seed', 'device')
PATH_KEYS = ('pretrain_train_plan', 'pretrain_validation_plan', 'pretrain_chart_cache',
             'pretrain_audio_cache', 'pretrain_mel_root', 'pretrain_source_root',
             'decoder_train_train_plan', 'decoder_train_validation_plan',
             'decoder_train_chart_cache', 'decoder_train_audio_cache',
             'decoder_train_mel_root', 'decoder_train_source_root',
             'decoder_train_physical_cache', 'decoder_train_window_cache',
             'decoder_train_gt_plan_cache', 'decoder_train_geometry',
             'decoder_train_slide_data', 'decoder_train_route_distances',
             'joint_train_output')


def resolve(config):
    """No fallback to an old training recipe; aliases only serve existing helpers."""
    result = dict(config)
    for name in TRAIN_KEYS:
        result['pretrain_' + name] = config['joint_train_' + name]
        if name != 'initialization':
            result['decoder_train_' + name] = config['joint_train_' + name]
    result.update(draft_z_stride_beats=config['joint_train_z_stride_beats'],
                  decoder_z_cross_attention=True, pretrain_audio_mode='joint',
                  pretrain_p0_cache=None, decoder_train_audio_mode='joint',
                  decoder_train_audio_scope='window', decoder_train_plan_source='gt',
                  decoder_train_fused_optimizer=config['joint_train_fused_optimizer'],
                  decoder_train_checkpoint=None, decoder_train_init_checkpoint=None,
                  decoder_train_p0_cache=None, decoder_train_local_p0_cache=None,
                  decoder_train_dp_cache=None, pretrain_output=config['joint_train_output'],
                  decoder_train_output=config['joint_train_output'])
    for name in ('physical', 'judge', 'trajectory'):
        result['decoder_train_' + name + '_weight'] = config['joint_train_' + name + '_weight']
    return result


def validate(config):
    if config['joint_train_initialization'] != 'scratch':
        raise ValueError('joint training initializes every component from scratch')
    if config['joint_train_plan_source'] != 'gt':
        raise ValueError('this joint recipe uses GT plan in training and validation')
    if config['pretrain_gradient_accumulation_steps'] != config['decoder_train_gradient_accumulation_steps']:
        raise ValueError('joint training has one shared gradient accumulation count')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip', 'decoder_weight',
                'reconstruction_weight', 'planner_weight'):
        value = config['joint_train_' + key]
        if (isinstance(value, bool) or not math.isfinite(value) or value < 0
                or key in ('learning_rate', 'gradient_clip') and value == 0):
            raise ValueError('invalid joint training ' + key)
    for key in ('scheduler_epochs', 'stop_after_epochs'):
        if type(config['joint_train_' + key]) is not int or config['joint_train_' + key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for key in ('workers', 'warmup_steps'):
        if type(config['joint_train_' + key]) is not int or config['joint_train_' + key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    if config['joint_train_stop_after_epochs'] > config['joint_train_scheduler_epochs']:
        raise ValueError('absolute stop exceeds scheduler horizon')
    if config['joint_train_optimizer'] != 'adamw' or config['joint_train_precision'] not in ('bf16', 'fp32'):
        raise ValueError('joint training supports AdamW and bf16/fp32')
    if type(config['joint_train_seed']) is not int:
        raise ValueError('seed must be an integer')
    required = [k for k in PATH_KEYS if config.get(k) is None]
    if required:
        raise ValueError('set joint input/output paths: ' + ', '.join(required))
    for name in ('physical', 'judge', 'trajectory'):
        value = config['decoder_train_' + name + '_weight']
        if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('set explicit Decoder ' + name + ' weight')


def provenance(config):
    return dict(paths={key: str(Path(config[key]).resolve()) for key in PATH_KEYS
                       if key != 'joint_train_output' and config.get(key) is not None})


def optimizer(parameters, config):
    from maidata.decoder_training.recipe import optimizer as create
    return create(parameters, config)


def scheduler(opt, steps, config):
    from maidata.training.pretrain import make_scheduler
    return make_scheduler(opt, steps, config)


def seed(config):
    from maidata.training.pretrain import seed_scratch
    seed_scratch(config)


def autocast(device, config):
    from maidata.training.pretrain import autocast as context
    return context(device, config)
