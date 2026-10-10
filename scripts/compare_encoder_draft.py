"""Replay one existing full-song batch against two source checkouts.

Run prepare with the optimized checkout, then run/eval and run/train separately
with each checkout's PYTHONPATH. No training assets or checkpoints are modified.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import statistics
import time

import torch


def flatten(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value.detach().cpu()
    elif is_dataclass(value):
        for f in fields(value):
            yield from flatten(getattr(value, f.name), prefix+'.'+f.name)
    elif isinstance(value, dict):
        for k,v in value.items():
            yield from flatten(v, prefix+'.'+str(k))
    elif isinstance(value, (tuple, list)):
        for i,v in enumerate(value):
            yield from flatten(v, prefix+'.'+str(i))


def prepare(args):
    from maidata.training.__main__ import load_config
    from maidata.training.data import AudioPlan, make_loader, epoch_iterator
    c = load_config(args.config)
    loader = make_loader(AudioPlan(c['pretrain_train_plan'], c), c)
    loader.batch_sampler.selected_step = args.batch_index
    started = time.perf_counter()
    batch = next(epoch_iterator(loader))
    if hasattr(batch, 'unpack'):
        batch = batch.unpack()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torch.save(dict(batch=batch, config=c, checkpoint=payload,
                    batch_index=args.batch_index, prepare_seconds=time.perf_counter()-started), args.bundle)
    print(json.dumps(dict(batch_index=args.batch_index, groups=batch.audio_variants,
        audio_windows=len(batch.audio_windows.mel), windows=len(batch.window_ids),
        checkpoint_epoch=payload['completed_epochs'], checkpoint_step=payload['step']), ensure_ascii=False), flush=True)


def run(args):
    from maidata.training.network import Pretrainer
    from maidata.training.pretrain import autocast, weighted_loss, make_optimizer, make_scheduler
    from maidata.training.runner import update
    from maidata.training.checkpoint import restore_rng
    torch.set_num_threads(1)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=False)
    c, ckpt, raw = bundle['config'], bundle['checkpoint'], bundle['batch']
    if args.precision:
        c = dict(c, pretrain_precision=args.precision)
    model = Pretrainer(c).to('cuda')
    model.load_state_dict(ckpt['model_state'], strict=True)
    optimizer = make_optimizer(model.parameters(), c)
    scheduler = make_scheduler(optimizer, ckpt['steps_by_epoch'], c)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    restore_rng(ckpt['rng'])
    report = dict(mode=args.mode, checkpoint_epoch=ckpt['completed_epochs'], checkpoint_step=ckpt['step'],
        batch_index=bundle['batch_index'], audio_groups=raw.audio_variants,
        window_ids=raw.window_ids, precision=c['pretrain_precision'], workers=c['pretrain_workers'])
    if args.mode == 'eval':
        model.eval()  # Autograd stays on; only stochastic dropout is disabled.
        batch = raw.to('cuda')
        captured = {}
        def capture(name):
            return lambda module, inputs, outputs: captured.update(dict(flatten(outputs, name)))
        hooks = [getattr(model, name).register_forward_hook(capture(name))
                 for name in ('encoder', 'draft', 'reconstruction', 'planner')]
        torch.cuda.reset_peak_memory_stats()
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload_saved_tensors else nullcontext()
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend = sdpa_kernel(SDPBackend.MATH) if args.math_attention else nullcontext()
        report.update(math_attention=args.math_attention, offload_saved_tensors=args.offload_saved_tensors)
        with backend, offload, autocast('cuda', c):
            terms, _ = model(batch)
            loss = weighted_loss(terms, c)
        loss.backward()
        gradients = {k: p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(v).all() for v in captured.values())
        assert all(torch.isfinite(v).all() for v in gradients.values())
        for h in hooks:
            h.remove()
        report.update(loss=float(loss.detach()), terms={k: dict(total=float(v.total.detach()), count=int(v.count)) for k,v in terms.items()},
                      peak_allocated=torch.cuda.max_memory_allocated())
        torch.save(dict(outputs=captured, gradients=gradients, report=report), args.output.with_suffix('.pt'))
    else:
        model.train()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.repeats + 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics, audit = update(model, [raw], optimizer, scheduler, c)
            torch.cuda.synchronize()
            if i >= 2:
                timings.append(time.perf_counter()-start)
        report.update(seconds=timings, median_seconds=statistics.median(timings),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            last_metrics=metrics.report(c), gradient_norm=audit['gradient_norm'],
            module_gradient_norms={name: sum(float(p.grad.float().square().sum()) for p in module.parameters()
                if p.grad is not None)**.5 for name,module in model.named_children()})
        assert all(v > 0 for v in report['module_gradient_norms'].values())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('window_ids','audio_groups','last_metrics')},
                     ensure_ascii=False), flush=True)


def compare(args):
    torch.set_num_threads(1)
    old = torch.load(args.before, map_location='cpu', weights_only=False)
    new = torch.load(args.after, map_location='cpu', weights_only=False)
    report = {}
    for group in ('outputs', 'gradients'):
        a, b = old[group], new[group]
        assert a.keys() == b.keys(), group
        rows = {}
        for k in a:
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
            if not a[k].is_floating_point():
                assert torch.equal(a[k], b[k]), k
                continue
            x, y = a[k].double().flatten(), b[k].double().flatten()
            diff = (x-y).abs()
            norm = x.norm().clamp_min(1e-30)
            rows[k] = dict(max_abs=float(diff.max()) if len(diff) else 0.,
                relative_l2=float(diff.norm()/norm), reference_norm=float(norm),
                cosine=float(torch.nn.functional.cosine_similarity(x, y, dim=0, eps=1e-30)))
        report[group] = rows
    report['loss'] = dict(before=old['report']['loss'], after=new['report']['loss'])
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(loss=report['loss'], worst_outputs=sorted(report['outputs'].items(),
        key=lambda kv:kv[1]['relative_l2'], reverse=True)[:5], worst_gradients=sorted(report['gradients'].items(),
        key=lambda kv:kv[1]['relative_l2'], reverse=True)[:8])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-index', type=int, default=0); p.add_argument('--bundle', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('eval','train'), required=True); p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--precision', choices=('fp32','bf16'))
    p.add_argument('--offload-saved-tensors', action='store_true', help='CPU activation storage for numerical replay only')
    p.add_argument('--math-attention', action='store_true', help='FP32 math reference for numerical replay only')
    p = sub.add_parser('compare')
    p.add_argument('--before', type=Path, required=True); p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'compare':compare}[args.command](args)


if __name__ == '__main__':
    main()
