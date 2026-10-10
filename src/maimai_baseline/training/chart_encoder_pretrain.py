"""Independent reconstruction, frozen-audio predictability and mu dispersion."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from maimai_baseline.training import chart_encoder_checkpoint as checkpoint

DEFAULT_RECIPE = dict(epochs=60, scheduler_epochs=155, offset=0, seed=20260811, workers=12,
    token_budget=4096, precision='bf16', learning_rate=.000565685424949238,
    weight_decay=.01, gradient_clip=1., warmup_steps=79, validation='disabled')
ARGUMENTS = dict(stop_after_epochs='epochs', scheduler_epochs='scheduler_epochs', plan_offset='offset',
    seed='seed', num_workers='workers', token_budget='token_budget', precision='precision',
    learning_rate='learning_rate', weight_decay='weight_decay', gradient_clip='gradient_clip',
    warmup_steps='warmup_steps')


def resolve_recipe(args):
    """Consume only numerical recipe fields, with flat or stages documents."""
    config = getattr(args, 'experiment_config', None)
    document = {} if config is None else (dict(config) if isinstance(config, Mapping)
                else json.loads(Path(config).read_text('utf-8-sig')))
    if not isinstance(document, dict):
        raise ValueError('chart encoder recipe must be a JSON object')
    document = document.get('recipe', document)
    result = dict(DEFAULT_RECIPE)
    result.update({key: value for key, value in document.items() if key in result})
    stages = document.get('stages', {})
    local = stages.get('encoder_pretrain', stages.get('A', {}))
    result.update({key: value for key, value in local.items() if key in result})
    for name, key in ARGUMENTS.items():
        value = getattr(args, name, None)
        if value is not None:
            result[key] = value
    for key in ('epochs', 'scheduler_epochs', 'workers', 'token_budget'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('offset', 'warmup_steps'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if type(result['seed']) is not int:
        raise ValueError('seed must be an integer')
    if result['epochs'] > result['scheduler_epochs']:
        raise ValueError('scheduler horizon must cover the absolute stop epoch')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError(f'invalid {key}')
    if result['workers'] != 12 or result['precision'] not in ('fp32', 'bf16') or result['validation'] != 'disabled':
        raise ValueError('chart encoder pretraining requires workers12 and disabled validation')
    args.resolved_recipe = result
    for name, key in ARGUMENTS.items():
        setattr(args, name, result[key])
    return result


def plan_step_counts(plan, recipe):
    end = recipe['offset'] + recipe['scheduler_epochs']
    if end > plan.available_epochs:
        raise ValueError('fixed epoch plan does not cover the scheduler horizon')
    counts = [int(plan.batch_count(epoch)) for epoch in range(recipe['offset'], end)]
    if any(count < 1 for count in counts):
        raise ValueError('fixed epoch plan contains an empty epoch')
    return counts


def make_optimizer(model, recipe):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])


def make_scheduler(optimizer, recipe, steps_by_epoch):
    total, warmup = sum(steps_by_epoch), recipe['warmup_steps']

    def factor(step):
        if warmup and step < warmup:
            return min(1., max(1e-6, (step + 1) / warmup))
        progress = (step - warmup) / max(1, total - warmup)
        return .5 * (1. + math.cos(math.pi * min(1., max(0., progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _device(args):
    requested = getattr(args, 'device', 'cuda')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('chart encoder pretraining requested CUDA, but CUDA is unavailable')
    return torch.device(requested)


def _move(value, device):
    if hasattr(value, 'to_device'):
        return value.to_device(device)
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_move(child, device) for child in value)
    return value


def _autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else nullcontext()


def epoch_iterator(loader, restored_loader_rng=None):
    """A restart creates workers; persistent uninterrupted workers already exist."""
    iterator = iter(loader)
    if restored_loader_rng is not None and loader.persistent_workers:
        loader.generator.set_state(restored_loader_rng)
    return iterator


def _accumulate(table, metrics):
    for name, row in metrics.items():
        total = table.setdefault(name, {'count': 0})
        count = int(row['count'])
        total['count'] += count
        for key, value in row.items():
            if key != 'count':
                total[key] = total.get(key, 0.) + float(value) * count


def _means(table):
    return {name: {key: value if key == 'count' else value / max(1, row['count'])
                  for key, value in row.items()} for name, row in table.items()}


def update(model, batch, optimizer, scheduler, recipe, *, collect_metrics=False):
    from maimai_baseline.model.chart_encoder_pretrain import pretraining_loss

    device = next(model.parameters()).device
    batch = _move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, recipe['precision']):
        outputs = model(batch)
        loss, metrics = pretraining_loss(outputs, batch['targets'], collect_metrics=collect_metrics)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('non-finite chart encoder reconstruction loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                         recipe['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), metrics, float(norm), outputs


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


@torch.no_grad()
def final_diagnostics(model, batch):
    """Training-plan samples, not held-out evaluation or an acceptance threshold."""
    from maimai_baseline.model.chart_encoder_pretrain import reconstruction_loss

    model.eval()
    batch = _move(batch, next(model.parameters()).device)
    actual = model(batch)
    values, valid = actual['latents'], actual['valid']
    mismatch = values.roll(1, 0) if len(values) > 1 else values.roll(1, 1)
    variants = {'real': actual, 'zero': model.decode(torch.zeros_like(values), valid),
                'mismatched': model.decode(mismatch, valid)}
    # Fixed Gaussian intervention describes sensitivity without altering RNG
    # state used by training/checkpoint continuation.
    noise_generator = torch.Generator(device=values.device).manual_seed(20260811)
    noise = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=noise_generator)*.05
    variants['perturbed'] = model.decode((values+noise).masked_fill(~valid[...,None],0.), valid)
    metrics, examples = {}, []
    for name, outputs in variants.items():
        loss, fields = reconstruction_loss(outputs, batch['targets'], collect_metrics=True)
        metrics[name] = dict(loss=float(loss), fields=fields)
    end_id = model.vocabulary.event_id('SEGMENT_END')
    for b in range(len(values)):
        target = batch['targets']['ids']['event'][b]
        endings = target.eq(end_id).nonzero().flatten()
        truth_end = int(endings[0]) if len(endings) else None
        sample = dict(reference=_jsonable(batch.get('references', list(range(len(values))))[b]),
            segment_id=batch.get('segment_id', [None] * len(values))[b],
            spatial_view_id=batch.get('spatial_view_id', [None] * len(values))[b],
            target_end_row=truth_end, target_ids={k: v[b].cpu().tolist() for k, v in batch['targets']['ids'].items()},
            target_times_ticks={k: v[b].cpu().tolist() for k, v in batch['targets']['times'].items()}, variants={})
        for name, outputs in variants.items():
            fields = {k: v[b].argmax(-1).cpu().tolist() for k, v in outputs['field_logits'].items()}
            endings = [i for i, event in enumerate(fields['event']) if event == end_id]
            end = endings[0] if endings else None
            length = model.query_rows if end is None else end + 1
            sample['variants'][name] = dict(end_row=end,
                ids={k: v[:length] for k, v in fields.items()},
                times_beats={k: (v[b].float() if k == 'moment_position_tick' else v[b].float().expm1().clamp_min(0.))[:length].cpu().tolist()
                             for k, v in outputs['time_predictions'].items()})
        examples.append(sample)
    return dict(scope='fixed-training-plan-sample', held_out=False, metrics=metrics, examples=examples,
                mismatch='previous-window-in-sample' if len(values) > 1 else 'previous-four-beat-slot')


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare(args):
    from maimai_baseline.data.chart_encoder_pretrain import prepare_inputs
    inputs = prepare_inputs(args)
    if inputs.collator.audio_teacher is None:
        raise ValueError('native encoder pretraining requires --teacher-audio-cache')
    return inputs


def train(args):
    if getattr(args, 'cuda_check', False):
        return cuda_check(args)
    from maimai_baseline.model.chart_encoder_pretrain import ChartEncoderPretrainer
    from maimai_baseline.data.chart_encoder_pretrain import fixed_diagnostic_batch

    recipe = resolve_recipe(args)
    device = _device(args)
    _seed(recipe['seed'])
    inputs = _prepare(args)
    counts = plan_step_counts(inputs.plan_store, recipe)
    output = Path(args.output).resolve()
    resume = getattr(args, 'resume', None)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite an existing chart encoder run')
    if resume:
        model, payload = checkpoint.load_checkpoint(resume, device=device)
        start_epoch, step = checkpoint.validate_resume(payload, recipe=recipe,
            data_binding=inputs.binding, steps_by_epoch=counts)
        if model.query_rows != inputs.query_rows:
            raise ValueError('chart encoder resume fixed query capacity changed')
    else:
        model = ChartEncoderPretrainer(inputs.vocabulary, inputs.prototypes, inputs.query_rows,
                                       seed=recipe['seed']).to(device)
        start_epoch = step = 0
    optimizer = make_optimizer(model, recipe)
    scheduler = make_scheduler(optimizer, recipe, counts)
    restored_loader_rng = None
    if resume:
        checkpoint.restore_training_state(payload, optimizer=optimizer, scheduler=scheduler,
                                           loader_generator=inputs.loader_generator)
        restored_loader_rng = inputs.loader_generator.get_state().clone()
    output.mkdir(parents=True, exist_ok=True)
    run = dict(recipe=recipe, data_binding=inputs.binding, steps_by_epoch=counts,
               total_scheduler_steps=sum(counts), query_rows=inputs.query_rows,
               resume=None if not resume else dict(checkpoint=str(Path(resume).resolve()),
                                                    completed_epochs=start_epoch, step=step))
    if resume and (output / 'run.json').is_file():
        previous = json.loads((output / 'run.json').read_text('utf8'))
        with (output / 'resume-history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(previous=previous, current=run), ensure_ascii=False, allow_nan=False) + '\n')
    _write_json(output / 'run.json', run)
    for epoch in range(start_epoch, recipe['epochs']):
        begun = time.perf_counter()
        inputs.sampler.set_epoch(recipe['offset'] + epoch)
        model.train()
        diagnostics, losses = {}, []
        last = epoch + 1 == recipe['epochs']
        loader_rng = restored_loader_rng if epoch == start_epoch else None
        for raw in epoch_iterator(inputs.loader, loader_rng):
            loss, metrics, _, _ = update(model, raw, optimizer, scheduler, recipe, collect_metrics=last)
            losses.append(loss)
            _accumulate(diagnostics, metrics)
            step += 1
        if len(losses) != counts[epoch]:
            raise ValueError('chart encoder loader changed the fixed plan batch boundary')
        row = dict(epoch=epoch + 1, step=step, loss=sum(losses)/len(losses),
            seconds=time.perf_counter()-begun, diagnostics=_means(diagnostics) if last else {})
        if last:
            report = final_diagnostics(model, fixed_diagnostic_batch(inputs, args, count=4))
            _write_json(output / 'reconstruction.json', report)
            row['sample_diagnostics'] = report['metrics']
        checkpoint.save_checkpoint(output / 'checkpoints/last.ckpt', model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, step=step, recipe=recipe, data_binding=inputs.binding,
            steps_by_epoch=counts, loader_generator=inputs.loader_generator, metrics=row,
            source_provenance=getattr(args, 'source_identity', None))
        if (epoch + 1) % 5 == 0 or last:
            # Serialize the same complete-epoch state; no best or within-epoch saves.
            last_payload = torch.load(output / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
            checkpoint._atomic_save(output / f'checkpoints/epoch-{epoch + 1:03d}.ckpt', last_payload)
        with (output / 'metrics.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    checkpoint.export_encoder(output / 'encoder.pt', model=model, recipe=recipe,
        data_binding=inputs.binding, source_provenance=getattr(args, 'source_identity', None),
        completed_epochs=recipe['epochs'])
    return output / 'encoder.pt'


def cuda_check(args):
    """One original plan batch update and cost report; never launches a full run."""
    from maimai_baseline.model.chart_encoder_pretrain import ChartEncoderPretrainer

    recipe = resolve_recipe(args)
    if not torch.cuda.is_available():
        raise RuntimeError('original-batch chart encoder CUDA check requires CUDA')
    device = torch.device('cuda')
    _seed(recipe['seed'])
    inputs = _prepare(args)
    inputs.sampler.set_epoch(recipe['offset'])
    raw = next(iter(inputs.loader))
    model = ChartEncoderPretrainer(inputs.vocabulary, inputs.prototypes, inputs.query_rows,
                                   seed=recipe['seed']).to(device).train()
    optimizer = make_optimizer(model, recipe)
    scheduler = make_scheduler(optimizer, recipe, plan_step_counts(inputs.plan_store, recipe))
    frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    loss, metrics, norm, outputs = update(model, raw, optimizer, scheduler, recipe, collect_metrics=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begun
    parameters = dict(model.named_parameters())
    scope = {prefix: dict(trainable=sum(p.numel() for n, p in parameters.items() if n.startswith(prefix) and p.requires_grad),
        gradients=sum(p.numel() for n, p in parameters.items() if n.startswith(prefix) and p.grad is not None),
        updated=sum(p.numel() for n, p in parameters.items() if n.startswith(prefix) and n in before and not torch.equal(p.detach().cpu(), before[n])))
        for prefix in ('encoder.', 'audio_predictor.', 'layers.', 'field_heads.', 'time_heads.')}
    if any(not torch.equal(parameters[name].detach().cpu(), value) for name, value in frozen.items()):
        raise RuntimeError('chart encoder CUDA check changed frozen parameters')
    if any(scope[prefix]['updated'] == 0 for prefix in ('encoder.', 'layers.', 'audio_predictor.')):
        raise RuntimeError('chart encoder CUDA check has no encoder/reconstruction update')
    cpu = raw.unpack() if hasattr(raw, 'unpack') else raw
    report = dict(operation='original-plan-batch-chart-encoder-update', recipe=recipe,
        step=1, loss=loss, metrics=metrics, gradient_norm=norm, scope=scope,
        device=torch.cuda.get_device_name(), seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        encoding_stats=outputs['encoding_stats'], references=_jsonable(cpu.get('references', [])),
        windows=len(cpu['window_groups']), query_rows=model.query_rows, fixed_plan_batch=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'cuda-check.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return output / 'cuda-check.json'
