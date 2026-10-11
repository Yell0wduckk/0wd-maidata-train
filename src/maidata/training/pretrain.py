"""Explicit E/D/R/P recipe consumers; no dataset, model ownership or training loop."""
from contextlib import nullcontext
import math
import random

import numpy as np
import torch

from maidata.draft.losses import FIELD_PREFIX


LOSS_WEIGHT_KEYS = {
    'classification': 'reconstruction_classification_weight',
    'moment96': 'reconstruction_moment_weight',
    'empty': 'reconstruction_empty_weight',
    'positive_count': 'reconstruction_positive_count_weight',
    'planner_continuous': 'planner_continuous_weight',
    'planner_category': 'planner_category_weight',
}


def weighted_loss(terms, config, *, denominators=None):
    """Use common logical supports and equal weighting of active fields.

    With denominators, terms may describe one microbatch. A field absent there
    still participates in the equal-field average if supervised anywhere else
    in the same logical update.
    """
    counts = ({name: int(term.count) for name, term in terms.items()}
              if denominators is None else denominators)
    active_fields = sum(name.startswith(FIELD_PREFIX) and count > 0
                        for name, count in counts.items())
    result = None
    for key in LOSS_WEIGHT_KEYS.values():
        weight = config[key]
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('loss weights must be finite and nonnegative')
    for name, term in terms.items():
        field = name.startswith(FIELD_PREFIX)
        key = LOSS_WEIGHT_KEYS['classification' if field else name]
        divisor = max(1, counts[name]) * (max(1, active_fields) if field else 1)
        value = term.total / divisor * config[key]
        result = value if result is None else result + value
    return result


def seed_scratch(config):
    """Call once before fresh model construction; resume must restore RNG instead."""
    if config['pretrain_initialization'] != 'scratch':
        raise ValueError('fresh E/D/R/P construction requires scratch initialization')
    seed = config['pretrain_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(parameters, config):
    if config['pretrain_optimizer'] != 'adamw':
        raise ValueError('unsupported pretraining optimizer')
    # Keep original AdamW semantics: one trainable group, standard betas/eps.
    return torch.optim.AdamW((p for p in parameters if p.requires_grad),
                             lr=config['pretrain_learning_rate'],
                             weight_decay=config['pretrain_weight_decay'])


def clip_gradients(parameters, config):
    return torch.nn.utils.clip_grad_norm_(parameters, config['pretrain_gradient_clip'],
                                         error_if_nonfinite=True)


def epoch_range(completed_epochs, config):
    stop, horizon = config['pretrain_stop_after_epochs'], config['pretrain_scheduler_epochs']
    if (any(type(n) is not int for n in (completed_epochs, stop, horizon))
            or not 0 <= completed_epochs <= stop <= horizon or stop < 1):
        raise ValueError('require 0 <= completed epochs <= absolute stop <= scheduler horizon')
    return range(completed_epochs + 1, stop + 1)


def make_scheduler(optimizer, steps_by_epoch, config):
    """Original linear warmup + cosine, using the full horizon's optimizer steps."""
    epoch_range(0, config)
    horizon, warmup = config['pretrain_scheduler_epochs'], config['pretrain_warmup_steps']
    if (len(steps_by_epoch) != horizon
            or any(type(n) is not int or n < 1 for n in steps_by_epoch)
            or type(warmup) is not int or warmup < 0):
        raise ValueError('scheduler needs positive update counts for every horizon epoch and nonnegative warmup')
    total = sum(steps_by_epoch)

    def factor(step):
        if warmup and step < warmup:
            return min(1., max(1e-6, (step + 1) / warmup))
        progress = (step - warmup) / max(1, total - warmup)
        return .5 * (1. + math.cos(math.pi * min(1., max(0., progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device, config):
    precision = config['pretrain_precision']
    if precision not in ('bf16', 'fp32'):
        raise ValueError('pretraining precision must be bf16 or fp32')
    return (torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
            if precision == 'bf16' else nullcontext())
