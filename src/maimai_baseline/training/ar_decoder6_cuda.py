"""Authorized original-batch CUDA diagnostics, isolated from formal training."""
import json
from copy import deepcopy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from maimai_baseline.runtime_versions import stamp
from maimai_baseline.training.steps import _move, _optimizer, _learning_rate, _autocast, _loss_step


def select_a_batches(batches, segments):
    """Target-row padding cost from accepted metadata; never open payloads."""
    costs = [len(batch)*max(int(segments[r.canonical_index]['target_rows']) for r in batch)
             for batch in batches]
    if not costs:
        raise ValueError('A diagnostic requires a nonempty first epoch')
    largest = max(range(len(costs)), key=costs.__getitem__)
    return list(dict.fromkeys((0, largest))), costs


def unchanged(model, before, *, prior):
    state = model.state_dict()
    names = [n for n in before if n.startswith('style_prior.') == prior]
    if any(not torch.equal(state[n].detach().cpu(), before[n]) for n in names):
        raise ValueError('diagnostic changed frozen tensors')


def snapshot(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def changed_parameters(model, before):
    return [n for n,p in model.named_parameters() if not torch.equal(p.detach().cpu(), before[n])]


def measure(device, action):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = action()
    torch.cuda.synchronize(device)
    return result, dict(seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def shutdown(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def check_a(args, device):
    from maimai_baseline.training.inputs import _prepare_inputs, _loaders
    from maimai_baseline.training.runner import _model_and_state
    from maimai_baseline.experiment import horizon, stage
    inputs = _prepare_inputs(args)
    loader, sampler, validation = _loaders(inputs, args)
    if validation is not None or args.resume:
        raise ValueError('A diagnostic requires fresh train-only inputs')
    batches = sampler.store.load_epoch(0)
    selected, costs = select_a_batches(batches, loader.dataset.segments)
    # Restrict only this independent diagnostic iterator. Keep complete refs,
    # views, occurrence, order and boundaries of each selected original batch.
    sampler._batches = tuple(batches[i] for i in selected)
    model, _, _ = _model_and_state(inputs, args, device)
    model.train()
    optimizer = _optimizer(model, learning_rate=_learning_rate(args)/79)
    prior_ids = {id(p) for p in model.style_prior.parameters()}
    if any(id(p) in prior_ids for g in optimizer.param_groups for p in g['params']):
        raise ValueError('A optimizer contains AR')
    def forbidden(*_):
        raise ValueError('A executed AR prior')
    hook = model.style_prior.register_forward_pre_hook(forbidden)
    rows = []
    try:
        for index, raw in zip(selected, loader, strict=True):
            batch = _move(raw, device)
            before = snapshot(model)
            optimizer.zero_grad(set_to_none=True)
            def update():
                with _autocast(device):
                    outputs, result = _loss_step(model, batch, inputs.vocabulary,
                        warmup_coefficient=0., style_progress=0., collect_metrics=True)
                if not torch.isfinite(result.loss):
                    raise ValueError('nonfinite A diagnostic loss')
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
                optimizer.step()
                return outputs, result
            (outputs, result), cost = measure(device, update)
            unchanged(model, before, prior=True)
            updated = changed_parameters(model, before)
            if any(p.grad is not None for p in model.style_prior.parameters()):
                raise ValueError('A produced AR gradients')
            for layer in range(len(model.decoder_layers)):
                if not any(n.startswith(f'decoder_layers.{layer}.') for n in updated):
                    raise ValueError('decoder layer did not update')
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in model.decoder_layers[layer].parameters()):
                    raise ValueError('decoder layer produced no gradient')
            # Existing forward counters describe sharing, without a second run.
            reuse = dict(model.a_encoding_stats, **model.audio_kv_stats)
            rows.append(dict(batch_index=index, target_padded_tokens=costs[index],
                decoder_padded_tokens=int(batch['row_padding_mask'].numel()),
                loss=float(result.loss.detach()), updated_parameters=updated, reuse=reuse, **cost))
        return dict(stage='A', selected_batches=selected, batch_count=len(batches),
            largest_target_padded_tokens=max(costs), workers=args.num_workers,
            scheduler_steps=horizon(inputs.manifest['experiment'], 'A'),
            physical_warmup=stage('A',inputs.manifest['experiment'])['physical_warmup_steps'], batches=rows)
    finally:
        hook.remove()
        shutdown(loader)


def check_b(args, device):
    from maimai_baseline.training.ar_runner import inputs, RECIPE
    from maimai_baseline.ar_checkpoint import attach_ar, require_frozen, capture_rng, restore_training_state
    from maimai_baseline.training.steps import _scheduler, set_scheduler_progress
    from maimai_baseline.data.song_context import SongBatches
    from maimai_baseline.data.style_audio_cache import BatchRequests, worker_init
    from maimai_baseline.training.song_ar import train_step, summarize
    model, parent, cache, plans, identity, _ = inputs(args)
    if args.resume:
        raise ValueError('B diagnostic requires initial A90 state')
    attach_ar(model, seed=RECIPE['seed'])
    norm = identity['normalization']['local']
    model.style_prior.set_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    model.to(device)
    optimizer = torch.optim.AdamW(model.style_prior.parameters(), lr=RECIPE['learning_rate'],
                                 weight_decay=RECIPE['weight_decay'])
    scheduler = _scheduler(optimizer, warmup_steps=50, total_steps=identity['total_steps'])
    requests = BatchRequests()
    requests.set_batches([cache.plan_indices(plans.load_epoch(90)[0])], epoch=90)
    dataset = SongBatches(args.cache, args.audio_condition_cache, args.song_context_cache,
                         RECIPE['seed'], packed=True, whole_song=True)
    loader = DataLoader(dataset, batch_size=None, sampler=requests, num_workers=12,
        worker_init_fn=worker_init, persistent_workers=True, pin_memory=True,
        generator=torch.Generator().manual_seed(RECIPE['seed']+90))
    try:
        batch = _move(next(iter(loader)), device)
        before = snapshot(model)
        rng = torch.Generator(device=device).manual_seed(RECIPE['seed'] ^ 0x4152504F)
        (stats, grad), cost = measure(device, lambda: train_step(
            model, optimizer, batch, generator=rng, collect_metrics=True))
        set_scheduler_progress(scheduler, 1)
        saved = dict(optimizer_state=deepcopy(optimizer.state_dict()),
                     scheduler_state=deepcopy(scheduler.state_dict()), rng_state=capture_rng(rng))
        # Exercise the same CUDA optimizer/scheduler/RNG restore mechanics.
        # This is a diagnostic state, never a partial-epoch checkpoint.
        restored_opt = torch.optim.AdamW(model.style_prior.parameters(),
            lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
        restored_scheduler = _scheduler(restored_opt,warmup_steps=50,total_steps=identity['total_steps'])
        restored_rng = torch.Generator(device=device)
        restore_training_state(saved,restored_opt,restored_scheduler,restored_rng)
        if (restored_scheduler.state_dict()!=scheduler.state_dict() or
                not torch.equal(rng.get_state(),restored_rng.get_state())):
            raise ValueError('B restore mechanics mismatch')
        require_frozen(model, parent)
        unchanged(model, before, prior=False)
        updated = changed_parameters(model, before)
        if not updated or any(not n.startswith('style_prior.') for n in updated):
            raise ValueError('B update scope mismatch')
        return dict(stage='B', selected_batches=[0], plan_epoch=91, workers=args.num_workers,
            scheduler_steps=identity['total_steps'], restored_training_state=True,
            complete_epoch_checkpoint=False, updated_parameters=updated, gradient_norm=grad,
            reuse={k:batch[k] for k in ('prior_audio_kv_calls','prior_control_kv_calls','prior_local_audio_calls')},
            metrics=summarize(stats), **cost)
    finally:
        shutdown(loader)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ('A', 'B'):
        raise ValueError('choose A or B original-batch diagnostic')
    which = argv.pop(0)
    if which == 'A':
        from maimai_baseline.cli import parser
        args = parser().parse_args(['fit', *argv])
    else:
        from maimai_baseline.training.ar_runner import parser
        args = parser().parse_args(argv)
    args.output = Path(args.output)
    if args.num_workers != 12 or args.device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('diagnostic requires CUDA and 12 workers')
    seed = 20260811
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    report = (check_a if which == 'A' else check_b)(args, torch.device('cuda'))
    args.output.mkdir(parents=True, exist_ok=True)
    report.update(**stamp(), diagnostic_only=True, formal_weights_reused=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', 'utf8')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
