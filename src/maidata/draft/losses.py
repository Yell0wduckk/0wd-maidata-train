"""Independent FP32 numerators/supports; the caller owns logical reduction."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


FIELD_PREFIX = 'field/'


@dataclass
class LossTerm:
    total: Tensor
    count: Tensor

    @property
    def mean(self):
        return self.total / self.count.clamp_min(1)


def prepare_supervision(target, planner_target):
    """Selectors for complete D coverage, including the supervised empty tail.

    Every valid target must have an owning Z, as supplied by the joint collator.
    Public callers with partial Z coverage retain the unprepared mask-aware path.
    """
    active = target.valid[:, None] & target.queries.valid
    beats = target.valid[:, None].expand_as(target.moment_counts)
    masks = {FIELD_PREFIX + name: active & (labels != -100)
             for name, labels in target.fields.items()}
    masks.update(moment96=active & (target.moment_tick != -100), empty=beats,
                 positive_count=beats & (target.moment_counts > 0))
    indices = {name: mask.flatten().nonzero().flatten() for name, mask in masks.items()}
    return dict(readout=dict(query_indices=active.flatten().nonzero().flatten(),
                             count_indices=beats.flatten().nonzero().flatten()),
                indices=indices, counts={name: len(rows) for name, rows in indices.items()},
                planner_indices=planner_target.valid.nonzero().flatten(),
                planner_count=int(planner_target.valid.sum()))


def _categorical(logits, labels, active, smoothing, indices=None):
    """An empty field stays differentiably zero without inventing a label."""
    if indices is not None:
        selected = logits.float().flatten(0, 1).index_select(0, indices)
        truth = labels.flatten().index_select(0, indices)
        total = (F.cross_entropy(selected, truth, label_smoothing=smoothing, reduction='sum')
                 if len(indices) else logits.float().sum() * 0)
        return LossTerm(total, labels.new_tensor(len(indices)))
    total = (F.cross_entropy(logits.float()[active], labels[active],
                             label_smoothing=smoothing, reduction='sum')
             if active.any() else logits.float().sum() * 0)
    return LossTerm(total, active.sum())


def reconstruction_loss(output, target, *, label_smoothing, prepared=None):
    if not 0 <= label_smoothing <= 1:
        raise ValueError('invalid reconstruction label smoothing')
    device = output.empty_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        query_valid = target.valid[:, None] & target.queries.valid & output.query_valid
        terms = {}
        selected = lambda name: None if prepared is None else prepared['indices'][name]
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            terms[FIELD_PREFIX + name] = _categorical(
                logits, labels, query_valid & (labels != -100), label_smoothing,
                selected(FIELD_PREFIX + name))
        terms['moment96'] = _categorical(output.moment_logits, target.moment_tick,
                                        query_valid & (target.moment_tick != -100), 0., selected('moment96'))
        beat_valid = target.valid[:, None] & output.count_valid
        terms['empty'] = _categorical(output.empty_logits, (target.moment_counts > 0).long(),
                                      beat_valid, label_smoothing, selected('empty'))
        terms['positive_count'] = _categorical(output.positive_count_logits,
            target.moment_counts - 1, beat_valid & (target.moment_counts > 0), label_smoothing,
            selected('positive_count'))
    return terms


def planner_loss(output, target, *, prepared=None):
    valid = target.valid
    with torch.autocast(device_type=output.plan_values.device.type, enabled=False):
        if prepared is None:
            predicted = output.continuous_logits.float()[valid].sigmoid()
            truth = target.continuous.float()[valid]
            category_logits, category = output.config_logits.float()[valid], target.category[valid]
            count = valid.sum()
        else:
            index = prepared['planner_indices']
            predicted = output.continuous_logits.float().index_select(0, index).sigmoid()
            truth = target.continuous.float().index_select(0, index)
            category_logits = output.config_logits.float().index_select(0, index)
            category = target.category.index_select(0, index)
            count = valid.new_tensor(len(index), dtype=torch.long)
        mse = (predicted - truth).square().mean(-1).sum()
        ce = (F.cross_entropy(category_logits, category, reduction='sum') if len(category_logits)
              else output.config_logits.float().sum() * 0)
    return {'planner_continuous': LossTerm(mse, count),
            'planner_category': LossTerm(ce, count)}
