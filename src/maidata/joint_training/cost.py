"""Explicit original-batch CUDA diagnostics, never part of normal startup."""
from contextlib import contextmanager
from dataclasses import replace
import statistics
import time

import torch

from maidata.training.checkpoint import rng_state, restore_rng
from maidata.decoder_training.recipe import restore_optimizer
from maidata.training.runner import query_cost
from .recipe import autocast
from .update import update


class Events:
    def __init__(self):
        self.pairs = {}
        self.cpu_seconds = {}

    @contextmanager
    def section(self, name):
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            last.record()
            self.pairs.setdefault(name, []).append((first, last))
            self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.) + time.perf_counter() - start

    def report(self):
        return dict(cuda_seconds={name: sum(a.elapsed_time(b) for a, b in pairs)/1000
                                  for name, pairs in self.pairs.items()},
                    cpu_wall_seconds=self.cpu_seconds)


def workload(batches):
    return dict(audio_groups=[batch.audio_variants for batch in batches],
        auxiliary_base_windows=sum(len(batch.window_ids) for batch in batches),
        decoder_occurrences=sum(len(batch.windows) for batch in batches),
        unique_decoder_windows=sum(len(batch.decoder_starts) for batch in batches),
        local_encoder_windows=sum(len(batch.audio_windows.target_start_beats) for batch in batches),
        coverage_encoder_windows=sum(len(batch.coverage_audio_indices) for batch in batches),
        valid_audio_input_samples=sum(int(batch.audio_windows.valid.sum()) for batch in batches),
        actual_reconstruction_queries=sum(query_cost(batch) for batch in batches),
        valid_m_rows=sum(int(chunk.tokens.valid.sum()) for batch in batches for chunk in batch.chunks),
        padded_m_rows=sum(chunk.tokens.valid.numel() for batch in batches for chunk in batch.chunks),
        m_chunks=sum(len(batch.chunks) for batch in batches),
        m_batch_costs=[batch.budget_cost for batch in batches])


class Calls:
    def __init__(self, model):
        self.forward = dict.fromkeys(dict(model.named_children()), 0)
        self.backward = dict.fromkeys(self.forward, 0)
        self.handles = []
        for name, module in model.named_children():
            def forward(_module, _args, key=name):
                self.forward[key] += 1
            self.handles.append(module.register_forward_pre_hook(forward))
            parameter = next(module.parameters())
            def backward(gradient, key=name):
                self.backward[key] += 1
            self.handles.append(parameter.register_hook(backward))

    def close(self):
        for handle in self.handles:
            handle.remove()


def audit_update(model, batches, opt, sched, config, geometry):
    device = next(model.parameters()).device
    calls = Calls(model)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        metrics, audit = update(model, batches, opt, sched, config, geometry, audit=True)
        torch.cuda.synchronize(device)
    finally:
        calls.close()
    for name in calls.forward:
        if audit['gradients'][name] <= 0 or audit['changed_parameter_tensors'][name] <= 0:
            raise ValueError(name + ': expected trainable joint gradient and update')
    if calls.forward['encoder'] != len(batches) or calls.forward['draft'] != len(batches):
        raise ValueError('joint E/D forward reuse was lost')
    if calls.backward['encoder'] != len(batches) or calls.backward['draft'] != len(batches):
        raise ValueError('joint E/D shared backward reuse was lost')
    return dict(kind='correctness_audit_not_throughput', metrics=metrics.report(config), **audit,
        forward_calls=calls.forward, backward_parameter_calls=calls.backward,
        audit_seconds=time.perf_counter()-start, workload=workload(batches),
        peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def measure(model, batches, opt, sched, config, geometry):
    """Same weights/RNG/workload before every timed step; reset costs excluded."""
    device = next(model.parameters()).device
    # Populate Adam moments and warm up the complete path before snapshots.
    for _ in range(2):
        update(model, batches, opt, sched, config, geometry)
    torch.cuda.synchronize(device)
    snapshot = dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(opt.state_dict()),
                    scheduler=cpu_copy(sched.state_dict()), rng=rng_state())

    def reset():
        model.load_state_dict(snapshot['model'], strict=True)
        restore_optimizer(opt, cpu_copy(snapshot['optimizer']))
        sched.load_state_dict(snapshot['scheduler'])
        restore_rng(snapshot['rng'])
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

    reset()
    # Fixed-condition M baseline excludes D/R/P execution, using the identical
    # Z values produced from the common initial state outside every timer.
    fixed_z = []
    with torch.no_grad(), autocast(device, config):
        for raw in batches:
            terms, conditions, outputs = model.forward_shared(raw.to(device), auxiliary=False)
            fixed_z.append(replace(conditions.decoder_z, values=conditions.decoder_z.values.detach().clone()))
            del terms, conditions, outputs
    reset()
    reports = {}
    for mode in ('joint', 'detach_z', 'auxiliary', 'decoder'):
        extra = {'fixed_z': fixed_z} if mode == 'decoder' else {}
        for _ in range(2):
            reset()
            update(model, batches, opt, sched, config, geometry, mode=mode, **extra)
        samples, last_metrics, last_audit = [], None, None
        for _ in range(7):
            reset()
            recorder, calls = Events(), Calls(model)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                last_metrics, last_audit = update(model, batches, opt, sched, config, geometry,
                    mode=mode, recorder=recorder, **extra)
                torch.cuda.synchronize(device)
            finally:
                calls.close()
            samples.append(dict(seconds=time.perf_counter()-start, **recorder.report(),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device),
                forward_calls=dict(calls.forward), backward_parameter_calls=dict(calls.backward)))
        values = [row['seconds'] for row in samples]
        reports[mode] = dict(samples=samples, median_seconds=statistics.median(values),
            min_seconds=min(values), max_seconds=max(values), metrics=last_metrics.report(config),
            audit=last_audit)
    reset()
    # One extra profiler step explains dispatch/call counts; excluded from timing.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
        update(model, batches, opt, sched, config, geometry)
    operators = []
    for event in profiler.key_averages(group_by_input_shape=True):
        operators.append(dict(name=event.key, count=event.count,
            cpu_time_us=event.cpu_time_total, device_time_us=event.device_time_total,
            shapes=event.input_shapes))
    operators.sort(key=lambda event: event['device_time_us'], reverse=True)
    return dict(kind='original_batch_cost_comparison', warmup=2, repetitions=7,
        reset_outside_timing=True, diagnostics=False, parameter_audit=False,
        workload=workload(batches), modes=reports,
        sdpa_operators=[event for event in operators if 'attention' in event['name'].lower()],
        synchronization_operators=[event for event in operators if any(name in event['name']
            for name in ('nonzero', '_local_scalar_dense', 'item'))], top_device_operators=operators[:40],
        notes=['GT plan is fixed in all modes.',
               'detach_z retains auxiliary supervision and the M-to-audio path.',
               'decoder uses the same fixed Z and skips D/R/P; E still encodes the same local windows.',
               'Shared E makes sums of standalone wall times an approximation, not an epoch prediction.'])
