"""Independent CUDA updates and profiler evidence; never a training launcher."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def trace_summary(path):
    """Stream exported Kineto events; avoid building another multi-GB event tree."""
    from collections import defaultdict
    from bisect import bisect_right
    operators = defaultdict(lambda: dict(calls=0, cpu_us=0., cuda_us=0.))
    ranges, launches, kernels = defaultdict(list), {}, []
    event_lines = None
    with path.open(encoding='utf8') as source:
        for line in source:
            if line.rstrip() == '  {':
                event_lines = [line]
            elif event_lines is not None:
                event_lines.append(line)
                if line.rstrip() not in ('  },', '  }'):
                    continue
                event = json.loads(''.join(event_lines).rstrip().removesuffix(','))
                event_lines = None
                if event.get('ph') != 'X':
                    continue
                name, category = event.get('name', ''), event.get('cat', '')
                duration = event.get('dur', 0.)
                if category == 'cpu_op':
                    operators[name]['calls'] += 1
                    operators[name]['cpu_us'] += duration
                    external = event.get('args', {}).get('External id')
                    if external is not None:
                        launches[external] = event['ts']
                elif category == 'kernel':
                    operators[name]['calls'] += 1
                    operators[name]['cuda_us'] += duration
                    kernels.append((event.get('args', {}).get('External id'), duration))
                elif category == 'user_annotation' and name.startswith(('phase.', 'encoding.', 'loss.', 'dense.', 'attention.')):
                    ranges[name].append((event['ts'], event['ts']+duration))
    phases = {}
    for name, intervals in ranges.items():
        intervals.sort()
        starts = [a for a, _ in intervals]
        device = 0.
        for external, duration in kernels:
            timestamp = launches.get(external)
            if timestamp is None:
                continue
            i = bisect_right(starts, timestamp)-1
            if i >= 0 and timestamp <= intervals[i][1]:
                device += duration
        phases[name] = dict(calls=len(intervals), cpu_us=sum(b-a for a,b in intervals), cuda_kernel_us=device)
    return dict(operators=[dict(name=name, **cost) for name,cost in operators.items()],
        sdpa_operators=[name for name in operators if 'attention' in name.lower() or 'fmha' in name.lower()],
        phases=phases, profiler_note='CPU inclusive ranges and summed kernel durations; nested ranges overlap, not GPU busy time')


def prepare(args):
    import torch
    from maimai_baseline.training import queue
    from maimai_baseline.cli import parser
    from maimai_baseline.query6_parent import A_CONFIG
    from maimai_baseline.training.inputs import _prepare_inputs, _loaders
    from maimai_baseline.training.ar_decoder6_cuda import select_a_batches, shutdown
    from maimai_baseline.data.curriculum import EpochPlanStore
    q = queue.parser().parse_args(['--dataset-root', '/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911',
        '--epoch-plan-cache', str(args.plan), '--physical-interval-cache',
        '/root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache',
        '--song-chart-cache', '/root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart',
        '--song-audio-cache', '/root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio',
        '--output', str(args.output)])
    q.experiment_config = A_CONFIG
    config = parser().parse_args(['fit', *queue.a_flags(q, args.output/'unused')])
    inputs = _prepare_inputs(config)
    loader, sampler, validation = _loaders(inputs, config)
    assert validation is None and config.num_workers == 12
    original = EpochPlanStore(args.original_plan, verify_payloads=False).load_epoch(0)
    selected, costs = select_a_batches(original, loader.dataset.segments)
    jobs = [(f'original-{i}', original[i], dict(kind='original', batch_index=i,
             target_padded_tokens=costs[i])) for i in selected]
    sampler._batches = tuple(refs for _, refs, _ in jobs)
    records = []
    try:
        for (name, refs, info), raw in zip(jobs, loader, strict=True):
            target = args.output/(name+'.pt')
            torch.save(raw, target)
            batch = raw.unpack()
            execution = batch.get('song_dense_inputs', {}).get('batch', batch)
            info.update(name=name, file=target.name, references=[asdict(r) for r in refs],
                target_rows=int(batch['target_ids']['event'].ne(-100).sum()),
                original_shape=list(batch['row_padding_mask'].shape),
                execution_shape=list(execution['row_padding_mask'].shape),
                unique_audio_windows=len(execution['audio']))
            records.append(info)
            print(json.dumps(info), flush=True)
        from maimai_baseline.model.common import BaselineModelConfig
        save(args.output/'batches.json', dict(jobs=records, vocabulary=inputs.vocabulary.to_dict(),
            route_artifact=str(config.route_artifact), model_config=BaselineModelConfig().to_dict(),
            workers=12, preparation_only=True))
    finally:
        shutdown(loader)


def run(args):
    import torch
    torch.set_num_threads(1)
    from maimai_baseline.model.common import BaselineModelConfig
    from maimai_baseline.model.route import load_route_artifacts
    from maimai_baseline.training.initialization import _initialize_model
    from maimai_baseline.training.steps import _move, _optimizer, _loss_step, _autocast
    from maimai_baseline.loss.supervision import SpecializedSupervision
    from maimai_dataset.vocabulary import CompactVocabulary
    from torch.profiler import record_function
    meta = json.loads((args.output/'batches.json').read_text())
    vocab = CompactVocabulary.from_dict(meta['vocabulary'])
    report = dict(source=str(args.source.resolve()), batch=args.batch, diagnostic_only=True,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        workers=12, precision='bf16-autocast', status='running')
    destination = args.output/f'{args.label}-{args.batch}.json'
    save(destination, report)
    model, initialization = _initialize_model(vocab, BaselineModelConfig(**meta['model_config']), seed=20260811,
        artifacts=load_route_artifacts(Path(meta['route_artifact']), verify_payloads=False))
    report['initialization'] = initialization['audio_cross_initialization']
    initial_layers = [layer.experts[0].linear1.weight.detach().clone() for layer in model.decoder_layers]
    model = model.cuda().train()
    model.difficulty_loss_preset = 'uniform'
    model.specialized_supervision = SpecializedSupervision(enabled=False)
    model.style_prior.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('A executed prior')))
    started = time.perf_counter()
    packed = torch.load(args.output/(args.batch+'.pt'), map_location='cpu', weights_only=False)
    raw = packed.unpack()
    report['input_load_unpack_seconds'] = time.perf_counter()-started
    report['input_cpu_prepare_seconds'] = packed.cpu_prepare_seconds
    if not getattr(model, 'song_dense', False):
        raw.pop('song_dense_inputs', None)
    started = time.perf_counter()
    batch = _move(raw, torch.device('cuda'))
    torch.cuda.synchronize()
    report['h2d_seconds'] = time.perf_counter()-started
    report['model_parameters'] = model.parameter_count()
    report['phase'] = 'ready'
    save(destination, report)
    optimizer = _optimizer(model, learning_rate=0.000565685424949238/79)
    prior_ids = {id(p) for p in model.style_prior.parameters()}
    assert not any(id(p) in prior_ids for g in optimizer.param_groups for p in g['params'])
    def update(diagnostics=False):
        report['phase'] = 'forward_loss'
        with record_function('phase.zero_grad'):
            optimizer.zero_grad(set_to_none=True)
        with record_function('phase.forward_loss'), _autocast(torch.device('cuda')):
            out, result = _loss_step(model, batch, vocab, warmup_coefficient=1., style_progress=.1,
                                     collect_metrics=diagnostics)
        report['phase'] = 'backward'
        with record_function('phase.backward'):
            result.loss.backward()
        with record_function('phase.clip'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True, error_if_nonfinite=True)
        report['phase'] = 'optimizer'
        with record_function('phase.optimizer'):
            optimizer.step()
        return float(result.loss.detach()), float(norm), out
    def measured(diagnostics=False):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, norm, out = update(diagnostics)
        torch.cuda.synchronize()
        cost = dict(seconds=time.perf_counter()-started, loss=loss, gradient_norm=norm,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if diagnostics:
            d = out['audio_diagnostics']
            cost['whole_batch_diagnostics'] = {k:(v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v)
                for k,v in d.items() if k.startswith(('decoder_', 'sparse_', 'supervision_', 'diagnostic_scope', 'song_decoder_rows'))}
            cost['mmoe_counts'] = [{task:values['count'] for task,values in layer.metrics.items()}
                                   for layer in model.decoder_layers]
        del out
        return cost
    profile = None
    if args.profile:
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    try:
        if args.oom_profile:
            with profile:
                report['first_update'] = measured(False)
        else:
            report['first_update'] = measured(False)
        save(destination, report)
        print(json.dumps(dict(first_update=report['first_update'])), flush=True)
        report['steady_updates'] = []
        for _ in range(args.repeats):
            report['steady_updates'].append(measured())
            save(destination, report)
            print(json.dumps(dict(update=report['steady_updates'][-1])), flush=True)
        report['diagnostic_update'] = measured(True)
        report['layers_with_gradient'] = [i for i, layer in enumerate(model.decoder_layers)
            if any(p.grad is not None and bool(p.grad.abs().sum()) for p in layer.parameters())]
        report['prior_frozen'] = all(p.grad is None and not p.requires_grad for p in model.style_prior.parameters())
        report['layers_updated'] = [i for i, (layer, before) in enumerate(zip(model.decoder_layers, initial_layers, strict=True))
                                   if not torch.equal(layer.experts[0].linear1.weight.detach().cpu(), before)]
        report['optimizer_state_parameters'] = len(optimizer.state)
        report['reuse'] = dict(model.a_encoding_stats)
        if args.profile:
            from functools import wraps
            import maimai_baseline.physical.loss as objective
            def scope(function, label):
                @wraps(function)
                def wrapped(*values, **keywords):
                    with record_function(label):
                        return function(*values, **keywords)
                return wrapped
            for name in ('chart_style_encoder', 'song_audio_encoder', 'audio_encoder'):
                module = getattr(model, name)
                module.forward = scope(module.forward, 'encoding.'+name)
            objective.l3_loss = scope(objective.l3_loss, 'loss.L3')
            objective.physical_bad_mass_loss = scope(objective.physical_bad_mass_loss, 'loss.Physical')
            import maimai_baseline.model.dual as dual
            original_attention = dual.attention
            def attention_scope(*values, **keywords):
                with record_function('attention.self' if keywords.get('causal') else 'attention.cross'):
                    return original_attention(*values, **keywords)
            dual.attention = attention_scope
            import maimai_baseline.model.song_chart as chart
            chart.window_posteriors = scope(chart.window_posteriors, 'encoding.chart_posterior')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False) as prof:
                measured(True)
            trace = args.output/f'{args.label}-{args.batch}-trace.json'
            prof.export_chrome_trace(str(trace))
            report.update(trace_summary(trace))
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='oom', error=str(error), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save(destination, report)
        raise
    finally:
        if args.oom_profile and profile is not None:
            trace = args.output/f'{args.label}-{args.batch}-trace.json'
            profile.export_chrome_trace(str(trace))
            report.update(trace_summary(trace))
        save(destination, report)
    print(json.dumps({k:v for k,v in report.items() if k!='operators'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--original-plan', type=Path)
    p.add_argument('--batch')
    p.add_argument('--label', default='dense')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--oom-profile', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()/'src'))
    args.output.mkdir(parents=True, exist_ok=True)
    (prepare if args.mode == 'prepare' else run)(args)
