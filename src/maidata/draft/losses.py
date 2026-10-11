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


def _categorical(logits, labels, active, smoothing):
    """An empty field stays differentiably zero without inventing a label."""
    total = (F.cross_entropy(logits.float()[active], labels[active],
                             label_smoothing=smoothing, reduction='sum')
             if active.any() else logits.float().sum() * 0)
    return LossTerm(total, active.sum())


def reconstruction_loss(output, target, *, label_smoothing):
    if not 0 <= label_smoothing <= 1:
        raise ValueError('invalid reconstruction label smoothing')
    device = output.empty_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        query_valid = target.valid[:, None] & target.queries.valid & output.query_valid
        terms = {}
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            terms[FIELD_PREFIX + name] = _categorical(
                logits, labels, query_valid & (labels != -100), label_smoothing)
        terms['moment96'] = _categorical(output.moment_logits, target.moment_tick,
                                        query_valid & (target.moment_tick != -100), 0.)
        beat_valid = target.valid[:, None] & output.count_valid
        terms['empty'] = _categorical(output.empty_logits, (target.moment_counts > 0).long(),
                                      beat_valid, label_smoothing)
        terms['positive_count'] = _categorical(output.positive_count_logits,
            target.moment_counts - 1, beat_valid & (target.moment_counts > 0), label_smoothing)
    return terms


def planner_loss(output, target):
    valid = target.valid
    with torch.autocast(device_type=output.plan_values.device.type, enabled=False):
        predicted = output.continuous_logits.float()[valid].sigmoid()
        mse = (predicted - target.continuous.float()[valid]).square().mean(-1).sum()
        ce = F.cross_entropy(output.config_logits.float()[valid], target.category[valid], reduction='sum')
    return {'planner_continuous': LossTerm(mse, valid.sum()),
            'planner_category': LossTerm(ce, valid.sum())}
