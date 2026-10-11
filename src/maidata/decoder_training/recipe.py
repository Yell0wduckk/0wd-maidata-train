"""Explicit stage-II recipe consumers. No fallback to the stage-I recipe."""
from contextlib import nullcontext
import math
import random
import numpy as np
import torch


PATH_KEYS = tuple('decoder_train_'+k for k in ('checkpoint','init_checkpoint','p0_cache','dp_cache','gt_plan_cache','window_cache','train_plan','validation_plan','chart_cache',
    'physical_cache','audio_cache','mel_root','source_root','geometry','slide_data','route_distances','output'))


def validate(config):
    cached = config['decoder_train_audio_mode'] == 'cached'
    if config['decoder_train_audio_mode'] not in ('joint', 'cached'):
        raise ValueError('decoder_train_audio_mode must be joint or cached')
    if config['decoder_train_audio_scope'] != ('full_song' if cached else 'window'):
        raise ValueError('joint stage II requires window audio; cached adaptation requires full_song P0')
    if cached and not config['decoder_z_cross_attention']:
        raise ValueError('cached Decoder adaptation requires Z cross-attention')
    if not cached and config['decoder_train_init_checkpoint'] is not None:
        raise ValueError('Decoder weight initialization belongs to cached adaptation')
    keys = [k for k in config if k.startswith('decoder_train_')]
    if config['decoder_train_plan_source'] not in ('predicted','gt'):
        raise ValueError('decoder_train_plan_source must be predicted or gt')
    export_only = {'decoder_train_checkpoint', 'decoder_train_init_checkpoint'}
    if not cached:
        export_only.add('decoder_train_p0_cache')
    else:
        export_only.update(('decoder_train_audio_cache', 'decoder_train_mel_root'))
    if not config['decoder_z_cross_attention'] and config['decoder_train_plan_source'] == 'gt':
        export_only.add('decoder_train_dp_cache')
    if config['decoder_train_plan_source'] == 'predicted':
        export_only.add('decoder_train_gt_plan_cache')
    missing = [k for k in keys if k not in export_only and config[k] is None]
    if missing:
        raise ValueError('set explicit Decoder training recipe: '+', '.join(missing))
    for key in ('gradient_accumulation_steps','row_budget','chunk_token_budget','scheduler_epochs','stop_after_epochs'):
        if type(config['decoder_train_'+key]) is not int or config['decoder_train_'+key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('warmup_steps','workers'):
        if type(config['decoder_train_'+key]) is not int or config['decoder_train_'+key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if type(config['decoder_train_seed']) is not int:
        raise ValueError('seed must be an integer')
    for key in ('learning_rate','weight_decay','gradient_clip','physical_weight','judge_weight','trajectory_weight'):
        value = config['decoder_train_'+key]
        if isinstance(value,bool) or not math.isfinite(value) or value < 0 or (key in ('learning_rate','gradient_clip') and value == 0):
            raise ValueError(f'invalid {key}')
    if config['decoder_train_stop_after_epochs'] > config['decoder_train_scheduler_epochs']:
        raise ValueError('absolute stop exceeds scheduler horizon')
    if config['decoder_train_precision'] not in ('fp32','bf16') or config['decoder_train_optimizer'] != 'adamw':
        raise ValueError('supported recipe: AdamW with fp32 or CUDA bf16')


def seed(config):
    value = config['decoder_train_seed']
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def optimizer(parameters, config):
    parameters = [p for p in parameters if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=config['decoder_train_learning_rate'],
                             weight_decay=config['decoder_train_weight_decay'],
                             fused=config['decoder_train_fused_optimizer'] and parameters[0].device.type == 'cuda')


def restore_optimizer(opt, state):
    # The current implementation selects the kernel, while the checkpoint owns
    # learning rates, moments and steps. Set flags BEFORE load_state_dict so
    # PyTorch places fused step tensors on the parameter device correctly.
    groups = [dict(saved) for saved in state['param_groups']]
    for group, current in zip(groups, opt.param_groups):
        group['fused'] = current['fused']
        if current['fused']:
            group['foreach'] = False
    opt.load_state_dict(dict(state, param_groups=groups))


def scheduler(opt, steps, config):
    warmup, total = config['decoder_train_warmup_steps'], sum(steps)
    if not steps or any(n < 1 for n in steps):
        raise ValueError('empty optimizer schedule')
    def factor(step):
        if warmup and step < warmup:
            return min(1.,max(1e-6,(step+1)/warmup))
        progress = min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
        return .5*(1.+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(opt,factor)


def autocast(device, config):
    return (torch.autocast('cuda',dtype=torch.bfloat16)
            if torch.device(device).type == 'cuda' and config['decoder_train_precision'] == 'bf16' else nullcontext())
