"""FP32 sums and supports; training chooses weights, never hidden defaults."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class LossTerm:
    total: Tensor
    count: Tensor

    @property
    def mean(self):
        return self.total / self.count.clamp_min(1)


def reconstruction_loss(output, target, *, label_smoothing, smooth_l1_beta):
    if not 0 <= label_smoothing <= 1 or smooth_l1_beta <= 0:
        raise ValueError('invalid reconstruction smoothing/beta')
    if output.valid.shape != target.valid.shape or not torch.equal(output.valid, target.valid):
        raise ValueError('R prediction and target window validity differ')
    device = output.count_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        ce = output.count_logits.float().sum() * 0
        ce_count = torch.zeros((), device=device, dtype=torch.long)
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            if labels.shape != logits.shape[:2]:
                raise ValueError(f'R {name} row shape mismatch')
            active = (labels != -100) & target.valid[:, None]
            labels = labels.masked_fill(~active, -100)
            ce = ce + F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100,
                                     label_smoothing=label_smoothing, reduction='sum')
            ce_count = ce_count + active.sum()
        time = output.count_logits.float().sum() * 0
        time_count = torch.zeros((), device=device, dtype=torch.long)
        for name, prediction in output.time_predictions.items():
            values, active = target.times[name].float(), target.time_valid[name] & target.valid[:, None]
            if values.shape != prediction.shape:
                raise ValueError(f'R {name} time shape mismatch')
            values = values[active] / 96
            if name != 'moment_position_tick':
                values = values.log1p()
            time = time + F.smooth_l1_loss(prediction.float()[active], values,
                                         beta=smooth_l1_beta, reduction='sum')
            time_count = time_count + active.sum()
        logits = output.count_logits.float()
        if target.moment_counts.shape != logits.shape[:2]:
            raise ValueError('R count shape mismatch')
        active = target.valid[:, None].expand(logits.shape[:2])
        labels = target.moment_counts.masked_fill(~active, -100)
        count_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100,
                                    label_smoothing=label_smoothing, reduction='sum')
    return {'classification': LossTerm(ce, ce_count), 'time': LossTerm(time, time_count),
            'count': LossTerm(count_loss, active.sum())}


def planner_loss(output, target):
    if (output.valid.shape != target.valid.shape or not torch.equal(output.valid, target.valid)
            or target.continuous.shape != output.continuous_logits.shape
            or target.category.shape != output.valid.shape):
        raise ValueError('Planner prediction/target shapes or validity differ')
    valid = target.valid
    with torch.autocast(device_type=output.plan_values.device.type, enabled=False):
        predicted = output.continuous_logits.float()[valid].sigmoid()
        mse = (predicted - target.continuous.float()[valid]).square().mean(-1).sum()
        ce = F.cross_entropy(output.config_logits.float()[valid], target.category[valid], reduction='sum')
    return {'planner_continuous': LossTerm(mse, valid.sum()),
            'planner_category': LossTerm(ce, valid.sum())}
