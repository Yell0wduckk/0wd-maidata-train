"""Conditional-stage illegal mass, evaluated before hard Physical masking."""
from __future__ import annotations
import torch
from .contract import PHYSICAL_CATEGORIES, PHYSICAL_CATEGORY_WEIGHTS, PHYSICAL_COEFFICIENT
from maimai_baseline.loss.common import L3LossResult
from maimai_baseline.time.duration import interval_log_probability, distribution_name


def compute(chart,outputs,batch,vocabulary,*,runtime_cache,warmup_coefficient,collect_metrics):
    if 'physical_stages' not in batch:
        raise ValueError('row-field Physical requires the new shared interval cache')
    from maimai_baseline.loss.difficulty import weights as difficulty_weights
    dw,_=difficulty_weights(batch,batch['target_ids']['event'].float())
    from maimai_baseline.loss.common import apply_field_grammar_mask, mask_audio_support
    fields=mask_audio_support(apply_field_grammar_mask(outputs['field_logits'],batch['target_ids'],vocabulary,
        sd_forbidden_kind=batch['supervision']['sd_forbidden_kind']),batch,vocabulary)
    heads={c:[] for c in PHYSICAL_CATEGORIES}
    def record(name,q,weight,rows,categories=(0,1,2,3)):
        if dw is not None and batch.get('difficulty_loss_preset','uniform')!='uniform':
            weight=weight*dw[rows[:,0]]
        for i,c in enumerate(PHYSICAL_CATEGORIES):
            if i not in categories:continue
            # A stage with no applicable illegal values has a zero contribution,
            # but is still an applicable stage in the fixed category reduction.
            qi=q[:,i].clamp(0,1-1e-6)
            denom=weight.sum().clamp_min(1e-12)
            loss=(-torch.log1p(-qi)*weight).sum()/denom
            mass=(qi*weight).sum()/denom
            heads[c].append((name,loss,mass,weight.sum()))
    for name,plan in batch['physical_stages'].items():
        b,r=plan['rows'].unbind(1)
        if name=='router':raw=outputs['router_logits'][b,r]
        elif name=='route':raw=outputs['route_logits'][b,r]
        elif '_short_' in name:
            field,flag=name.split('_short_',1)
            position=fields[field][b,r].float()
            bit=fields['modifier_short_'+flag][b,r].float()
            # Parallel logits form a small product support, before Physical.
            position=position.masked_fill(~plan['allowed'].any(-1),-torch.inf).softmax(-1)
            # A ban shared by both short states is solely a position event.
            # Marginalize it analytically, without sending cancellation noise
            # (or the new Touch rule's responsibility) into the flag head.
            common=plan['bad'].all(-1)
            partial=plan['bad'] & ~common[...,None]
            q=(position[:,None,:]*common).sum(-1)
            q=q+(position[:,None,:,None]*bit.softmax(-1)[:,None,None,:]*partial).sum((-1,-2))
            record(name,q,plan['weight'],plan['rows'])
            continue
        else:raw=fields[name][b,r]
        probability=raw.float().flatten(1).masked_fill(~plan['allowed'].flatten(1),-torch.inf).softmax(-1)
        bad=plan['bad'].flatten(2)
        q=(probability[:,None,:]*bad).sum(-1)
        record(name,q,plan['weight'],plan['rows'])
    for field,plan in batch['physical_duration_tails'].items():
        b,r=plan['rows'].unbind(1)
        p=interval_log_probability(outputs['time_outputs'][field][b,r],plan['bounds'],field=field,
                                    distribution=distribution_name(outputs['duration_distribution'])).exp()*plan['active']
        q=torch.stack((p,torch.zeros_like(p),torch.zeros_like(p),torch.zeros_like(p)),-1)
        record(field,q,plan['weight'],plan['rows'],categories=(0,))
    distribution=runtime_cache['delta_log_probability']
    if distribution.numel():
        q=(distribution.exp()*batch['physical_delta_bad']).sum(-1)
        rows=batch['time_targets']['delta_ms'].ne(-100).nonzero()
        record('delta_ms',torch.stack((q,torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(q)),-1),torch.ones_like(q),rows,categories=(0,))
    zero=chart.loss.new_zeros(())
    category=torch.stack([torch.stack([x[1] for x in heads[c]]).mean() if heads[c] else zero for c in PHYSICAL_CATEGORIES])
    physical=(category*category.new_tensor(PHYSICAL_CATEGORY_WEIGHTS)).sum()/4
    coefficient=PHYSICAL_COEFFICIENT*float(warmup_coefficient)
    total=chart.loss+coefficient*physical
    metrics=dict(chart.metrics)
    if collect_metrics:
        for i,c in enumerate(PHYSICAL_CATEGORIES):
            mass = torch.stack([x[2] for x in heads[c]]).mean() if heads[c] else zero
            metrics['physical:'+c]=dict(loss=float(category[i].detach()),bad_mass=float(mass.detach()),
                count=sum(float(x[3]) for x in heads[c]),category_weight=PHYSICAL_CATEGORY_WEIGHTS[i],
                exact_weighted_mean=True)
            for name,value,mass,count in heads[c]:
                metrics['physical:'+c+':'+name]=dict(loss=float(value.detach()),bad_mass=float(mass.detach()),count=float(count),
                    weight_sum=float(count),exact_weighted_mean=True)
        metrics['physical:total']=dict(loss=float(physical.detach()),coefficient=coefficient,
            category_reduction='weighted-sum-divided-by-fixed-four-v1',probability_scope='conditional-stage-before-hard-mask',
            backend='shared-interval-stages',**batch.get('physical_interval_cost',{}))
        metrics['component:physical']=dict(loss=float(physical.detach()),count=1.)
        metrics['total']={**metrics.get('total',{}),'loss':float(total.detach()),'chart_loss':float(chart.loss.detach()),'physical_loss':float(physical.detach()),'physical_coefficient':coefficient}
    return L3LossResult(loss=total,metrics=metrics if collect_metrics else {},components={**chart.components,'physical':physical})
