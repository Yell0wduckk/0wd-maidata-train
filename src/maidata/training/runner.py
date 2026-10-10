"""Whole-song E/D/R/P updates, validation and complete-epoch continuation."""
from itertools import islice
import json
import math
from pathlib import Path
import time

import torch

from . import checkpoint
from .batching import group_token_cost
from .data import AudioPlan, make_loader, epoch_iterator
from .network import Pretrainer
from .pretrain import (LOSS_WEIGHT_KEYS, seed_scratch, make_optimizer, make_scheduler,
                       clip_gradients, autocast, epoch_range)


def supports(batch):
    r, p = batch.reconstruction, batch.planner
    return dict(classification=sum(int(((v != -100) & r.valid[:, None]).sum()) for v in r.fields.values()),
        time=sum(int((v & r.valid[:, None]).sum()) for v in r.time_valid.values()),
        count=int(r.valid.sum()) * r.moment_counts.shape[1],
        planner_continuous=int(p.valid.sum()), planner_category=int(p.valid.sum()))


class Metrics:
    def __init__(self):
        self.sums = {k: [0., 0] for k in LOSS_WEIGHT_KEYS}

    def add(self, terms):
        for name, term in terms.items():
            self.sums[name][0] += float(term.total.detach())
            self.sums[name][1] += int(term.count)

    def report(self, config):
        terms = {k: dict(total=v[0], count=v[1], mean=v[0]/max(1, v[1])) for k,v in self.sums.items()}
        return dict(loss=sum(terms[k]['mean'] * config[w] for k,w in LOSS_WEIGHT_KEYS.items()), terms=terms)


def update(model, batches, optimizer, scheduler, config, *, audit=False):
    device = next(model.parameters()).device
    if any(not math.isfinite(config[k]) or config[k] < 0 for k in LOSS_WEIGHT_KEYS.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    total_counts = {k: 0 for k in LOSS_WEIGHT_KEYS}
    for batch in batches:
        for k, n in supports(batch).items():
            total_counts[k] += n
    optimizer.zero_grad(set_to_none=True)
    metrics = Metrics()
    before = {k: [p.detach().clone() for p in m.parameters()] for k,m in model.named_children()} if audit else {}
    for raw in batches:
        batch = raw.to(device)
        with autocast(device, config):
            terms, _ = model(batch)
            loss = sum(terms[k].total / max(1, total_counts[k]) * config[w] for k,w in LOSS_WEIGHT_KEYS.items())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite pretraining loss')
        loss.backward()
        metrics.add(terms)
    norm = clip_gradients(model.parameters(), config)
    gradients = {k: sum(float(p.grad.float().square().sum()) for p in m.parameters() if p.grad is not None)**.5
                 for k,m in model.named_children()} if audit else {}
    optimizer.step()
    scheduler.step()
    changed = {k: sum(not torch.equal(a, b) for a,b in zip(before[k], m.parameters()))
               for k,m in model.named_children()} if audit else {}
    if audit and (any(v <= 0 for v in gradients.values()) or any(v == 0 for v in changed.values())):
        raise ValueError('E/D/R/P must all receive gradients and update in the CUDA batch check')
    return metrics, dict(gradient_norm=float(norm), gradients=gradients, changed_parameter_tensors=changed)


@torch.no_grad()
def validate(model, loader, config, *, final, restored_rng=None):
    mode, rng = model.training, checkpoint.rng_state()
    model.eval()
    metrics, accuracy, examples = Metrics(), {}, []
    try:
        for raw in epoch_iterator(loader, restored_rng):
            batch = raw.to(next(model.parameters()).device)
            with autocast(next(model.parameters()).device, config):
                terms, (r, p, z) = model(batch)
            metrics.add(terms)
            if final:
                for key, logits in r.field_logits.items():
                    labels = batch.reconstruction.fields[key]
                    mask = (labels != -100) & batch.valid[:, None]
                    correct, count = accuracy.setdefault(key, [0, 0])
                    accuracy[key] = [correct + int(((logits.argmax(-1) == labels) & mask).sum()), count + int(mask.sum())]
                if not examples:
                    for i in range(min(4, len(raw.window_ids))):
                        examples.append(dict(window_id=raw.window_ids[i],
                            predicted_fields={k: v[i].argmax(-1).cpu().tolist() for k,v in r.field_logits.items()},
                            target_fields={k: v[i].cpu().tolist() for k,v in batch.reconstruction.fields.items()},
                            plan12=p.plan_values[i].float().cpu().tolist()))
        result = metrics.report(config)
        if not torch.isfinite(torch.tensor(result['loss'])):
            raise ValueError('nonfinite validation loss')
        if final:
            result['diagnostics'] = dict(field_accuracy={k: dict(correct=v[0], count=v[1]) for k,v in accuracy.items()},
                                         examples=examples)
        return result
    finally:
        model.train(mode)
        checkpoint.restore_rng(rng)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def run(config, *, resume=None, cuda_check=False):
    epoch_range(0, config)
    device = torch.device(config['pretrain_device'])
    if device.type == 'cuda' and (not torch.cuda.is_available()
            or (config['pretrain_precision'] == 'bf16' and not torch.cuda.is_bf16_supported())):
        raise ValueError('requested CUDA/bf16 runtime is unavailable')
    if cuda_check and (device.type != 'cuda' or resume is not None):
        raise ValueError('CUDA check uses a separate scratch CUDA model')
    output = Path(config['pretrain_output'])
    if not cuda_check and not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('fresh training output must be empty')
    plans = [AudioPlan(config[k], config) for k in ('pretrain_train_plan', 'pretrain_validation_plan')]
    binding, source = checkpoint.data_provenance(config), checkpoint.source_identity()
    loaders = [make_loader(p, config) for p in plans]
    steps = [len(plans[0].batches(e, config)) for e in range(1, config['pretrain_scheduler_epochs'] + 1)]
    if not resume:
        seed_scratch(config)
    model = Pretrainer(config).to(device)
    optimizer = make_optimizer(model.parameters(), config)
    scheduler = make_scheduler(optimizer, steps, config)
    completed, step, best = 0, 0, None
    restored = [None, None]
    if resume:
        payload = torch.load(resume, map_location='cpu', weights_only=False)
        completed, step, best = checkpoint.restore(payload, model, optimizer, scheduler, config=config,
            binding=binding, source=source, steps=steps, loaders=loaders)
        restored = [rng if started else None for rng,started in zip(payload['loader_rng'], payload['loader_started'])]
        del payload
    if cuda_check:
        # Select an actual, unchanged epoch-one batch near the size ceiling;
        # do not load preceding payloads or synthesize a different grouping.
        planned = plans[0].batches(1, config)
        costs = [max(sum(group_token_cost(plans[0].groups[k], config) for k in b) for b in s) for s in planned]
        selected = max(range(len(planned)), key=costs.__getitem__)
        loaders[0].batch_sampler.epoch = 1
        loaders[0].batch_sampler.selected_step = selected
        batches = list(islice(epoch_iterator(loaders[0]), config['pretrain_gradient_accumulation_steps']))
        calls = {k: 0 for k in ('encoder', 'draft', 'reconstruction', 'planner')}
        handles = []
        for key in calls:
            def count_call(module, args, name=key):
                calls[name] += 1
            handles.append(getattr(model, key).register_forward_pre_hook(count_call))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics, audit = update(model, batches, optimizer, scheduler, config, audit=True)
        torch.cuda.synchronize(device)
        report = dict(metrics=metrics.report(config), **audit, component_forward_calls=calls,
            epoch=1, optimizer_step_in_epoch=selected+1, weighted_token_cost=costs[selected],
            seconds=time.perf_counter()-started, peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device), audio_groups=[b.audio_variants for b in batches],
            windows=sum(len(b.window_ids) for b in batches), config=config, binding=binding, source=source)
        for handle in handles:
            handle.remove()
        write_json(output.with_name(output.name + '-cuda-check.json'), report)
        print(json.dumps({k:report[k] for k in ('seconds','peak_allocated','peak_reserved','windows','changed_parameter_tensors')}, ensure_ascii=False), flush=True)
        return report
    remaining = epoch_range(completed, config)
    if not remaining:
        return dict(completed_epochs=completed, step=step)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ('resume.json' if resume else 'run.json'), dict(config=config, binding=binding,
        source=source, steps_by_epoch=steps, resume=str(resume) if resume else None,
        starting_completed_epochs=completed, parameters=sum(p.numel() for p in model.parameters())))
    for epoch in remaining:
        started = time.perf_counter()
        model.train()
        loaders[0].batch_sampler.epoch = epoch
        iterator = epoch_iterator(loaders[0], restored[0])
        restored[0] = None
        epoch_metrics, updates, windows, groups = Metrics(), 0, 0, 0
        while batches := list(islice(iterator, config['pretrain_gradient_accumulation_steps'])):
            metrics, audit = update(model, batches, optimizer, scheduler, config)
            for k, value in metrics.sums.items():
                epoch_metrics.sums[k][0] += value[0]
                epoch_metrics.sums[k][1] += value[1]
            updates += 1
            step += 1
            windows += sum(len(b.window_ids) for b in batches)
            groups += sum(len(b.audio_variants) for b in batches)
            print(json.dumps(dict(epoch=epoch, update=updates, updates=steps[epoch-1], step=step,
                                  loss=metrics.report(config)['loss'], lr=optimizer.param_groups[0]['lr'])), flush=True)
        if updates != steps[epoch-1] or groups != len(plans[0].groups) or windows != len(plans[0].windows):
            raise ValueError('incomplete training epoch; checkpoint not saved')
        final = epoch == config['pretrain_stop_after_epochs']
        result = dict(epoch=epoch, step=step, train=epoch_metrics.report(config),
                      seconds=time.perf_counter()-started, audio_groups=groups, windows=windows)
        improved = False
        if epoch % 5 == 0 or final:
            result['validation'] = validate(model, loaders[1], config, final=final, restored_rng=restored[1])
            restored[1] = None
            value = result['validation']['loss']
            improved = best is None or value < best
            best = value if improved else best
        payload = checkpoint.capture(model, optimizer, scheduler, completed_epochs=epoch, step=step,
            config=config, binding=binding, source=source, steps=steps, loaders=loaders, best=best, metrics=result)
        checkpoint.atomic_save(output / 'checkpoints/last.ckpt', payload)
        if improved:
            checkpoint.atomic_save(output / 'checkpoints/best.ckpt', payload)
        if epoch % 5 == 0 or final:
            checkpoint.atomic_save(output / f'checkpoints/epoch-{epoch:03d}.ckpt', payload)
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(dict(completed_epoch=epoch, step=step, validation_loss=result.get('validation', {}).get('loss'))), flush=True)
    return dict(completed_epochs=epoch, step=step)
