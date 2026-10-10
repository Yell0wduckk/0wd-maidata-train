"""Clean-prefix GMM NLL plus frozen-decoder L3 through sampled latent inputs."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from maimai_baseline.loss.common import GROUP_BUDGET
from maimai_baseline.loss.l3 import l3_loss, logical_support
from maimai_baseline.loss.supervision import SpecializedSupervision
from maimai_baseline.model.baseline import baseline_forward
from maimai_baseline.model.song_ar import sample_mixture
from maimai_baseline.model.style import sampled_context
from maimai_baseline.training.song_ar import mixture_terms, merge_stats


def _sample_prior(model, row, *, generator=None, collect_metrics=False):
    """One TF prefix and real AR outside its GT support; both temperatures are 1."""
    prior = model.style_prior
    target_valid, condition_valid = row['target_valid'], row['condition_valid']
    mu = row['mu'].detach()
    if (mu.shape != (*target_valid.shape, 128) or target_valid.dtype != torch.bool
            or condition_valid.dtype != torch.bool or condition_valid.ndim != 2
            or len(condition_valid) != len(mu) or condition_valid.shape[1] < mu.shape[1]):
        raise ValueError('prior decoder requires aligned GT-prefix and condition masks')
    target_lengths = target_valid.sum(-1)
    condition_lengths = condition_valid.sum(-1)
    if 'weights' in row and not torch.equal(row['weights'], target_valid.float()):
        raise ValueError('prior decoder requires unit GT-prefix NLL weights')
    if (not torch.equal(target_valid, torch.arange(mu.shape[1], device=mu.device)[None] < target_lengths[:, None])
            or not torch.equal(condition_valid, torch.arange(condition_valid.shape[1], device=mu.device)[None] < condition_lengths[:, None])
            or bool((condition_lengths < target_lengths).any())):
        raise ValueError('prior decoder masks must be contiguous prefixes and cover GT')
    z = ((mu-prior.latent_mean)/prior.latent_std).masked_fill(~target_valid[..., None], 0.)
    conditions = {key: row[key] for key in ('audio', 'audio_valid', 'audio_beats',
                                           'audio_mapping', 'difficulty_id', 'span_beats', 'level')}
    projected = prior.project_audio(row['audio'], row['audio_valid'], row['audio_beats'], condition_valid.shape[1])
    if z.shape[1]:
        logits, mean, logvar, cache = prior(prior.shifted(z, target_valid),
            **conditions, valid=target_valid, projected=projected, return_cache=True)
        numerator, support, stats = mixture_terms(logits, mean, logvar, z, target_valid,
                                                   collect_metrics=collect_metrics)
        prefix_samples = sample_mixture(logits, mean, logvar, generator=generator,
            temperature=1., mixture_temperature=1.).masked_fill(~target_valid[..., None], 0.)
        nll = numerator/support.clamp_min(1)
    else:
        cache = None
        prefix_samples = z
        nll = prior.bos.sum()*0.
        stats = dict(nll_sum=0., coordinates=0, blocks=0)
    output_rows = [None]*len(mu)
    tail_steps = 0
    for length in torch.unique(target_lengths).tolist():
        indices = target_lengths.eq(length).nonzero().flatten()
        values = [prefix_samples.index_select(0, indices)[:, :length]]
        state = None if cache is None else cache.select_prefix(indices, length)
        inputs = (values[0][:, -1:] if length else prior.bos[None, None].expand(len(indices), 1, -1))
        subset = {key: value if key in ('audio', 'audio_valid', 'audio_beats')
                  else value.index_select(0, indices) for key, value in conditions.items()}
        stop = int(condition_lengths.index_select(0, indices).max())
        for index in range(length, stop):
            valid = condition_valid.index_select(0, indices)[:, index:index+1]
            logits, mean, logvar, state = prior(inputs, **subset, valid=valid,
                projected=projected, cache=state, return_cache=True)
            inputs = sample_mixture(logits, mean, logvar, generator=generator,
                temperature=1., mixture_temperature=1.).masked_fill(~valid[..., None], 0.)
            values.append(inputs)
            tail_steps += 1
        samples = torch.cat(values, 1)
        samples = F.pad(samples, (0, 0, 0, condition_valid.shape[1]-samples.shape[1]))
        for local, group in enumerate(indices.tolist()):
            output_rows[group] = samples[local]
    normalized = torch.stack(output_rows)
    raw = (normalized*prior.latent_std+prior.latent_mean).masked_fill(~condition_valid[..., None], 0.)
    stats.update(prior_audio_kv_calls=len(prior.blocks), prior_audio_kv_batch_variants=len(row['audio']),
                 prior_tail_steps=tail_steps, condition_blocks=int(condition_valid.sum()))
    return nll, raw, stats


def sample_prior(model, row, *, generator=None, collect_metrics=False):
    from maimai_baseline.model.token_moe import capture_moe_metrics
    with capture_moe_metrics(model.style_prior, collect_metrics) as metrics:
        nll, raw, stats = _sample_prior(model, row, generator=generator, collect_metrics=collect_metrics)
    for name, values in metrics.items():
        for key, number in values.items():
            stats[f'moe:{name}:{key}'] = number if key == 'count' else number*values['count']
    return nll, raw, stats


def _decoder_keys(model, batch):
    """Frozen audio projections are constant, shared across target microbatches."""
    if 'decoder_audio_context' not in batch:
        return None
    from maimai_baseline.model.common import AudioContext
    context = batch['decoder_audio_context']
    context = AudioContext(**context) if isinstance(context, dict) else context
    with torch.no_grad():
        return tuple(layer.audio_kv(context.memories[-1], context.beats[-1]) for layer in model.decoder_layers)


def _microbatches(model, batch):
    return [dict(micro, difficulty_loss_preset=getattr(model, 'difficulty_loss_preset',
                                                    micro.get('difficulty_loss_preset', 'uniform')))
            for micro in batch['decoder_batches']]


def _condition(batch, tokens, valid, logical_batch=None, decoder_keys=None):
    execution = dict(batch['song_dense_inputs'])
    groups, blocks = execution['latent_group'], execution['latent_block']
    index = blocks[:, None]+torch.arange(4, device=blocks.device)[None]
    if bool(((index < 0) | (index >= tokens.shape[1])).any()):
        raise ValueError('decoder condition window escaped prior coverage')
    execution['sampled_style_context'] = sampled_context(tokens[groups[:, None], index], valid[groups[:, None], index])
    if 'audio_context' not in execution and logical_batch is not None:
        from maimai_baseline.model.common import AudioContext
        continuous = logical_batch['decoder_audio_context']
        local = logical_batch['decoder_local_audio_context']
        continuous = AudioContext(**continuous) if isinstance(continuous, dict) else continuous
        local = AudioContext(**local) if isinstance(local, dict) else local
        execution['audio_context'] = continuous.map_audio_windows(execution['audio_indices'])
        execution['local_audio_context'] = local.map_audio_windows(execution['audio_window_index'])
    if decoder_keys is not None:
        execution['decoder_audio_keys'] = tuple(tuple(value.index_select(0, execution['audio_indices']) for value in kv)
                                               for kv in decoder_keys)
    return dict(batch, song_dense_inputs=execution)


def decoder_loss(model, batch, tokens, *, collect_metrics=False):
    """Differentiable full objective, useful for validation and small references."""
    micros = _microbatches(model, batch)
    supports = [logical_support(micro, model.vocabulary) for micro in micros]
    global_support = {name: sum(row[name] for row in supports) for name in GROUP_BUDGET}
    total = tokens.sum()*0.
    components = {name: 0. for name in GROUP_BUDGET}
    audio_keys = _decoder_keys(model, batch)
    if collect_metrics:
        batch['decoder_diagnostics'] = []
    for micro in micros:
        conditioned = _condition(micro, tokens, batch['prior_inputs']['condition_valid'], batch, audio_keys)
        outputs = baseline_forward(model, conditioned)
        result = l3_loss(outputs, conditioned, model.vocabulary, collect_metrics=collect_metrics,
            reduction_support=global_support, supervision=getattr(model, 'specialized_supervision', SpecializedSupervision()))
        total = total+result.loss
        for name, value in result.components.items():
            components[name] += float(value.detach())
        if collect_metrics:
            batch['decoder_diagnostics'].append(dict(metrics=result.metrics,
                reduction_support={name:float(value) for name,value in global_support.items()},
                components={name:float(value.detach()) for name,value in result.components.items()}))
    return total, components, global_support


def loss(model, batch, *, generator=None, return_diagnostics=False, collect_metrics=False):
    nll, tokens, stats = sample_prior(model, batch['prior_inputs'], generator=generator,
                                     collect_metrics=collect_metrics)
    l3, components, supports = decoder_loss(model, batch, tokens, collect_metrics=collect_metrics)
    stats.update(l3_sum=float(l3.detach()), nll_objective_sum=float(nll.detach()),
                 total_sum=float((nll+l3).detach()), logical_batches=1,
                 decoder_microbatches=len(batch['decoder_batches']), decoder_audio_encoder_calls=0,
                 **{'l3_'+name+'_sum':value for name,value in components.items()},
                 **{'l3_'+name+'_support':float(value) for name,value in supports.items()})
    batch.update(prior_audio_kv_calls=stats['prior_audio_kv_calls'],
                 prior_audio_kv_batch_variants=stats['prior_audio_kv_batch_variants'],
                 prior_control_kv_calls=0, prior_local_audio_calls=0)
    return (nll+l3, stats) if return_diagnostics else nll+l3


def train_step(model, optimizer, batch, *, generator=None, collect_metrics=False, gradient_clip=1., precision='fp32'):
    """Free each decoder graph, then apply its input gradient to the prior once."""
    if precision != 'fp32':
        raise ValueError('prior decoder training fixes FP32')
    if any(parameter.requires_grad for name,parameter in model.named_parameters() if not name.startswith('style_prior.')):
        raise ValueError('prior decoder training requires every A parameter frozen')
    model.eval()
    model.style_prior.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.style_prior.parameters()).device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        nll, samples, stats = sample_prior(model, batch['prior_inputs'], generator=generator,
                                          collect_metrics=collect_metrics)
    proxy = samples.detach().requires_grad_(True)
    micros = _microbatches(model, batch)
    supports = [logical_support(micro, model.vocabulary) for micro in micros]
    global_support = {name: sum(row[name] for row in supports) for name in GROUP_BUDGET}
    l3_value = 0.
    components = {name: 0. for name in GROUP_BUDGET}
    audio_keys = _decoder_keys(model, batch)
    if collect_metrics:
        batch['decoder_diagnostics'] = []
    for micro in micros:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
            conditioned = _condition(micro, proxy, batch['prior_inputs']['condition_valid'], batch, audio_keys)
            outputs = baseline_forward(model, conditioned)
            result = l3_loss(outputs, conditioned, model.vocabulary, collect_metrics=collect_metrics,
                reduction_support=global_support, supervision=getattr(model, 'specialized_supervision', SpecializedSupervision()))
        if not torch.isfinite(result.loss):
            raise RuntimeError('non-finite frozen decoder L3')
        result.loss.backward()
        l3_value += float(result.loss.detach())
        for name, value in result.components.items():
            components[name] += float(value.detach())
        if collect_metrics:
            batch['decoder_diagnostics'].append(dict(metrics=result.metrics,
                reduction_support={name:float(value) for name,value in global_support.items()},
                components={name:float(value.detach()) for name,value in result.components.items()}))
        del outputs, result, conditioned
    if not torch.isfinite(nll) or proxy.grad is None or not bool(torch.isfinite(proxy.grad).all()):
        raise RuntimeError('non-finite prior NLL or missing decoder input gradient')
    torch.autograd.backward((nll, samples), (None, proxy.grad))
    norm = torch.nn.utils.clip_grad_norm_(model.style_prior.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    stats.update(l3_sum=l3_value, nll_objective_sum=float(nll.detach()),
        total_sum=float(nll.detach())+l3_value, logical_batches=1,
        decoder_microbatches=len(micros), decoder_audio_encoder_calls=0,
        decoder_audio_kv_calls=len(model.decoder_layers) if audio_keys is not None else len(model.decoder_layers)*len(micros),
        **{'l3_'+name+'_sum':value for name,value in components.items()},
        **{'l3_'+name+'_support':float(value) for name,value in global_support.items()})
    batch.update(prior_audio_kv_calls=stats['prior_audio_kv_calls'],
                 prior_audio_kv_batch_variants=stats['prior_audio_kv_batch_variants'],
                 prior_control_kv_calls=0, prior_local_audio_calls=0)
    return stats, float(norm)


def summarize(stats):
    from maimai_baseline.training.song_ar import summarize as summarize_nll
    result = summarize_nll(stats)
    count = max(1, stats.get('logical_batches', 0))
    result.update(l3=stats.get('l3_sum', 0.)/count, total=stats.get('total_sum', 0.)/count)
    result['mean_batch_nll'] = stats.get('nll_objective_sum', 0.)/count
    result['l3_components'] = {name:stats.get('l3_'+name+'_sum', 0.)/count for name in GROUP_BUDGET}
    return result
