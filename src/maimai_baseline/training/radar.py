"""MajRadar-only prior objective over the original batch's selected windows."""
import math
import torch
from maimai_baseline.radar_contract import LOSS_COEFFICIENT


def prior_loss(prediction, *, collect_metrics=False):
    selected = prediction['target_windows']
    groups = prediction['window_group'].index_select(0, selected)
    windows = prediction['window_index'].index_select(0, selected)
    valid = prediction['valid'][groups, windows]
    with torch.autocast(device_type=prediction['mean'].device.type, enabled=False):
        mu = prediction['mean'][groups, windows].float().masked_fill(~valid[:, None, None], 0.)
        lv = prediction['logvar'][groups, windows].float().masked_fill(~valid[:, None, None], 0.)
        target = prediction['target'][groups, windows].float().masked_fill(~valid[:, None], 0.)
        log_pi = prediction['logits'][groups, windows].float().masked_fill(~valid[:, None], 0.).log_softmax(-1)
        component = -.5 * (math.log(2 * math.pi) + lv +
            (target[:, None, :] - mu).square() * (-lv).exp()).sum(-1)
        nll = -(log_pi + component).logsumexp(-1).masked_fill(~valid, 0.).sum()
        coordinates = valid.sum() * target.shape[-1]
        loss = nll / coordinates.clamp_min(1)
    metrics = {}
    if collect_metrics:
        metrics['radar:nll'] = dict(loss=float(loss.detach()), count=int(coordinates),
            windows=int(valid.sum()), coefficient=LOSS_COEFFICIENT)
    return loss * LOSS_COEFFICIENT, metrics
