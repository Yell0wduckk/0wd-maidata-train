"""Separate per-row router CE from the decoder's conditional event CE."""
import torch
from maimai_baseline.dual_contract import event_family
from maimai_baseline.loss.common import masked_mean


def router_loss(outputs, batch, vocabulary, conditional_loss, *, collect_metrics):
    target = batch['target_ids']['event']
    active = target.ne(-100)
    family = event_family(target, vocabulary)
    if bool((active & outputs['decoder_family'].ne(family)).any()):
        raise ValueError('expert routing differs from the supervised next event')
    support = outputs['router_support']
    if bool((active & ~support.gather(-1, family[..., None]).squeeze(-1)).any()):
        raise ValueError('router target outside legal family support')
    logits = outputs['router_logits'].float().masked_fill(~support, -10_000.)
    logp = logits.log_softmax(-1)
    nll = -logp.gather(-1, family[..., None]).squeeze(-1)
    smooth = -logp.masked_fill(~support, 0.).sum(-1) / support.sum(-1).clamp_min(1)
    router = (.95 * nll + .05 * smooth).masked_fill(~active, 0.)
    total = conditional_loss + router
    if not collect_metrics:
        return router, {}
    correct = logits.argmax(-1).eq(family)
    conditional_correct = outputs['field_logits']['event'].argmax(-1).eq(target)
    count = active.sum()
    metrics = {'router': dict(loss=masked_mean(router.detach(), active),
        accuracy=masked_mean(correct.float(), active), count=count),
        'conditional_event': dict(loss=masked_mean(conditional_loss.detach(), active),
        accuracy=masked_mean(conditional_correct.float(), active), count=count),
        'event': dict(loss=masked_mean(total.detach(), active),
        accuracy=masked_mean((correct & conditional_correct).float(), active), count=count)}
    for family_id, name in ((0, 'time'), (1, 'note')):
        selected = active & family.eq(family_id)
        metrics['router_' + name] = dict(loss=masked_mean(router.detach(), selected),
            accuracy=masked_mean(correct.float(), selected), count=selected.sum())
    return router, metrics
