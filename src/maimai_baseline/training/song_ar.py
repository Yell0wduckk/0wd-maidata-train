"""Deterministic frozen mu supervision; joint NLL over valid coordinates."""
import math

import torch

def posterior(batch, generator=None):
    """Read raw cached mu without sampling or an encoder call."""
    with torch.no_grad():
        clean=[]
        for row in batch['groups']:
            if 'mu' not in row:
                raise ValueError('frozen AR B requires cached deterministic mu')
            clean.append(row['mu'].detach().masked_fill(~row['valid'][...,None],0.))
    return clean


def mixture_terms(logits, mean, logvar, target, valid, *, collect_metrics=False):
    """Sum dimensions BEFORE marginalizing components; diagnostics never add loss."""
    mask = valid[..., None, None]
    mu = mean.float().masked_fill(~mask, 0.)
    lv = logvar.float().masked_fill(~mask, 0.)
    z = target.float().masked_fill(~valid[..., None], 0.)
    log_pi = logits.float().masked_fill(~valid[..., None], 0.).log_softmax(-1)
    component_logp = -.5*(math.log(2*math.pi) + lv + (z[..., None, :]-mu).square()*(-lv).exp()).sum(-1)
    joint = log_pi + component_logp
    nll = -joint.logsumexp(-1).masked_fill(~valid, 0.).sum()
    coordinates = valid.sum()*target.shape[-1]
    with torch.no_grad():
        pi = log_pi.exp()
        center = (pi[..., None]*mu).sum(-2)
        within = (pi[..., None]*lv.exp()).sum(-2)
        between = (pi[..., None]*(mu-center[..., None, :]).square()).sum(-2)
        stats = dict(nll_sum=float(nll.detach()), coordinates=int(coordinates), blocks=int(valid.sum()),
            mixture_squared_error=float((center-z).square().masked_fill(~valid[..., None], 0.).sum()),
            within_variance_sum=float(within.masked_fill(~valid[..., None], 0.).sum()),
            between_variance_sum=float(between.masked_fill(~valid[..., None], 0.).sum()))
        responsibility = joint.softmax(-1)
        for k in range(logits.shape[-1]):
            stats[f'component_{k}_probability_sum'] = float(pi[..., k].masked_fill(~valid, 0.).sum())
            stats[f'component_{k}_responsibility_sum'] = float(responsibility[..., k].masked_fill(~valid, 0.).sum())
        if collect_metrics:
            count = logits.shape[-1]
            pairs = torch.triu_indices(count, count, offset=1, device=mu.device)
            distances = (mu[..., pairs[0], :]-mu[..., pairs[1], :]).square().mean(-1).sqrt()
            stats.update(pair_distance_sum=float(distances.masked_fill(~valid[..., None], 0.).sum()),
                pair_support=int(valid.sum())*pairs.shape[1],
                logvar_lower_hits=int(((lv <= -6.) & mask).sum()),
                logvar_upper_hits=int(((lv >= 2.) & mask).sum()),
                component_coordinates=int(coordinates)*count)
    return nll, coordinates, stats


def _loss(model, batch, *, generator=None, return_diagnostics=False, collect_metrics=False):
    if 'prior_inputs' in batch:
        row = batch['prior_inputs']
        if not torch.equal(row['weights'], row['valid'].float()):
            raise ValueError('AR requires unit weight on every valid full-song latent')
        prior = model.style_prior
        z = ((row['mu'].detach()-prior.latent_mean)/prior.latent_std).masked_fill(~row['valid'][..., None], 0.)
        conditions = {k: row[k] for k in ('audio', 'audio_valid', 'audio_beats', 'audio_mapping',
                     'difficulty_id', 'span_beats', 'valid', 'level')}
        projected = prior.project_audio(row['audio'], row['audio_valid'], row['audio_beats'], z.shape[1])
        logits, mean, logvar = prior(prior.shifted(z, row['valid']), **conditions, projected=projected)
        numerator, support, stats = mixture_terms(logits, mean, logvar, z, row['valid'], collect_metrics=collect_metrics)
        batch.update(prior_audio_kv_calls=len(prior.blocks), prior_audio_kv_batch_variants=len(row['audio']),
                     prior_control_kv_calls=0, prior_local_audio_calls=0)
        value = numerator/support.clamp_min(1)
        return (value, stats) if return_diagnostics else value
    clean = posterior(batch, generator)
    prior = model.style_prior
    numerators, supports, stats = [], [], {}
    for row, z in zip(batch['groups'], clean, strict=True):
        if not torch.equal(row['weights'], row['valid'].float()):
            raise ValueError("AR requires unit weight on every valid full-song latent")
        if 'audio_conditioned' in row and not bool(row['audio_conditioned'].all()):
            raise ValueError("AR does not support audio dropout/CFG")
        z = (z-prior.latent_mean)/prior.latent_std
        conditions = {k: row[k] for k in ('audio', 'audio_valid', 'audio_beats', 'difficulty_id', 'span_beats', 'valid', 'level')}
        projected = prior.project_audio(row['audio'], row['audio_valid'], row['audio_beats'], z.shape[1])
        logits, mean, logvar = prior(prior.shifted(z, row['valid']), **conditions, projected=projected)
        numerator, support, current = mixture_terms(logits, mean, logvar, z, row['valid'], collect_metrics=collect_metrics)
        numerators.append(numerator)
        supports.append(support)
        merge_stats(stats, current)
    if not numerators:
        raise ValueError("AR batch has no audio groups")
    value = sum(numerators)/sum(supports).clamp_min(1)
    batch['prior_audio_kv_calls'] = len(numerators)*8
    batch['prior_control_kv_calls'] = 0
    batch['prior_local_audio_calls'] = 0
    return (value, stats) if return_diagnostics else value


def loss(model, batch, *, generator=None, return_diagnostics=False, collect_metrics=False):
    from maimai_baseline.model.token_moe import capture_moe_metrics
    with capture_moe_metrics(model.style_prior, collect_metrics) as metrics:
        value, stats = _loss(model, batch, generator=generator,
                             return_diagnostics=True, collect_metrics=collect_metrics)
    for name, values in metrics.items():
        for key, number in values.items():
            stats[f'moe:{name}:{key}'] = number if key == 'count' else number * values['count']
    return (value, stats) if return_diagnostics else value


def merge_stats(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0)+value


def summarize(stats):
    coordinates, blocks = max(1, stats.get('coordinates', 0)), max(1, stats.get('blocks', 0))
    within = stats.get('within_variance_sum', 0.)/coordinates
    between = stats.get('between_variance_sum', 0.)/coordinates
    result = dict(stats, nll=stats.get('nll_sum', 0.)/coordinates,
        mixture_mean_mse=stats.get('mixture_squared_error', 0.)/coordinates,
        mixture_within_variance=within, mixture_between_variance=between,
        mixture_total_variance=within+between)
    moe = {}
    for key, value in stats.items():
        if key.startswith('moe:'):
            name, field = key[4:].rsplit(':', 1)
            count = stats[f'moe:{name}:count']
            moe.setdefault(name, {})[field] = value if field == 'count' else value / max(1, count)
    if moe:
        result['moe'] = moe
    for key in stats:
        if key.startswith('component_') and key.endswith(('_probability_sum', '_responsibility_sum')):
            result[key.removesuffix('_sum')] = stats[key]/blocks
    if 'pair_support' in stats:
        result.update(component_mean_rms_distance=stats['pair_distance_sum']/max(1, stats['pair_support']),
            logvar_lower_rate=stats['logvar_lower_hits']/max(1, stats['component_coordinates']),
            logvar_upper_rate=stats['logvar_upper_hits']/max(1, stats['component_coordinates']))
    return result


def train_step(model, optimizer, batch, *, generator=None, collect_metrics=False, gradient_clip=1., precision="fp32"):
    optimizer.zero_grad(set_to_none=True)
    device_type = next(model.style_prior.parameters()).device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        value, stats = loss(model, batch, generator=generator, return_diagnostics=True, collect_metrics=collect_metrics)
    if not torch.isfinite(value):
        raise RuntimeError("non-finite AR mixture NLL")
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.style_prior.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return stats, float(norm)
