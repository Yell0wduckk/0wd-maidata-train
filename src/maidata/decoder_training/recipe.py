"""Explicit stage-II recipe consumers. No fallback to the stage-I recipe."""
from contextlib import nullcontext
import math
import random
import numpy as np
import torch


PATH_KEYS = tuple('decoder_train_'+k for k in ('checkpoint','p0_cache','dp_cache','window_cache','train_plan','validation_plan','chart_cache',
    'physical_cache','audio_cache','mel_root','source_root','geometry','slide_data','route_distances','output'))


def validate(config):
    keys = [k for k in config if k.startswith('decoder_train_')]
    missing = [k for k in keys if config[k] is None]
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
    return torch.optim.AdamW(parameters, lr=config['decoder_train_learning_rate'],
                             weight_decay=config['decoder_train_weight_decay'])


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
