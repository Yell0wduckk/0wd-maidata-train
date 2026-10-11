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


def reconstruction_loss(output, target, *, label_smoothing):
    if not 0 <= label_smoothing <= 1:
        raise ValueError('invalid reconstruction label smoothing')
    device = output.count_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        ce = output.count_logits.float().sum() * 0
        ce_count = torch.zeros((), device=device, dtype=torch.long)
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            active = (labels != -100) & target.valid[:, None]
            labels = labels.masked_fill(~active, -100)
            ce = ce + F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100,
                                     label_smoothing=label_smoothing, reduction='sum')
            ce_count = ce_count + active.sum()
        time = output.count_logits.float().sum() * 0
        time_count = torch.zeros((), device=device, dtype=torch.long)
        for name, prediction in output.time_predictions.items():
            selected = target.time_classes[name] & target.valid[:, None, None]
            active = selected.any(-1)
            logits = prediction.float()[active]
            basic = target.time_basic[name][active]
            logits = logits.masked_fill(~basic, -torch.inf)
            numerator = torch.logsumexp(logits.masked_fill(~selected[active], -torch.inf), -1)
            time = time + (torch.logsumexp(logits, -1)-numerator).sum()
            time_count = time_count + (target.time_valid[name] & target.valid[:, None]).sum()
        logits = output.count_logits.float()
        active = target.valid[:, None].expand(logits.shape[:2])
        labels = target.moment_counts.masked_fill(~active, -100)
        count_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100,
                                    label_smoothing=label_smoothing, reduction='sum')
    return {'classification': LossTerm(ce, ce_count), 'time': LossTerm(time, time_count),
            'count': LossTerm(count_loss, active.sum())}


def planner_loss(output, target):
    valid = target.valid
    with torch.autocast(device_type=output.plan_values.device.type, enabled=False):
        predicted = output.continuous_logits.float()[valid].sigmoid()
        mse = (predicted - target.continuous.float()[valid]).square().mean(-1).sum()
        ce = F.cross_entropy(output.config_logits.float()[valid], target.category[valid], reduction='sum')
    return {'planner_continuous': LossTerm(mse, valid.sum()),
            'planner_category': LossTerm(ce, valid.sum())}
