"""One affected frozen-latent A update on the original plan batch."""
import json
from pathlib import Path
import time

import torch

from maimai_baseline.style_contract import is_frozen, is_draft
from maimai_baseline.training.chart_encoder_pretrain import _seed
from maimai_baseline.training.recipe import bind_stage_recipe, plan_steps
from maimai_baseline.training.runner import _loaders, _model_and_state, _prepare_inputs
from maimai_baseline.training.steps import _autocast, _learning_rate, _loss_step, _move, _optimizer, _scheduler


def cuda_check(args):
    if not torch.cuda.is_available():
        raise RuntimeError('frozen latent A original-batch CUDA check requires CUDA')
    if not is_frozen(args.chart_style) or getattr(args, 'resume', None):
        raise ValueError('this A CUDA check requires a fresh frozen encoder60 model')
    bind_stage_recipe(args, 'A')
    args.command = 'fit'
    _seed(args.seed)
    device = torch.device('cuda')
    inputs = _prepare_inputs(args)
    loader, sampler, _ = _loaders(inputs, args)
    sampler.set_epoch(args.plan_offset)
    raw = next(iter(loader))
    model, _, _ = _model_and_state(inputs, args, device)
    model.training_precision = args.precision
    model.train()
    optimizer = _optimizer(model, learning_rate=_learning_rate(args), weight_decay=args.weight_decay)
    scheduler = _scheduler(optimizer, warmup_steps=args.warmup_steps,
        total_steps=plan_steps(inputs.plan_store, args.scheduler_epochs, offset=args.plan_offset))
    frozen = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    batch = _move(raw, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, args.precision):
        # Exercise the Physical gradient path in qualification; formal A still
        # starts at its unmodified 988-step Physical warmup.
        outputs, result = _loss_step(model, batch, inputs.vocabulary,
            warmup_coefficient=1., collect_metrics=True)
    result.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip,
                                         foreach=True, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begun
    parameters = dict(model.named_parameters())
    modules = ([model.draft_generator, model.audio_encoder, model.song_audio_encoder] if is_draft(model.config)
               else [model.chart_style_encoder, model.style_prior])
    if any(p.grad is not None for module in modules for p in module.parameters()):
        raise RuntimeError('frozen encoder/prior received A CUDA gradients')
    if any(not torch.equal(parameters[n].detach().cpu(), value) for n, value in frozen.items()):
        raise RuntimeError('frozen encoder/prior changed in A CUDA update')
    if 'style_posterior' in outputs or 'radar_prior' in outputs:
        raise RuntimeError('deterministic latent A unexpectedly used a posterior or radar objective')
    scope = {}
    for prefix in ('audio_encoder.', 'song_audio_encoder.', 'router.', 'style_conditioner.',
                   'decoder_layers.', 'time_heads.', 'chart_style_encoder.', 'style_prior.',
                   'draft_generator.', 'condition_view.'):
        members = [(n, p) for n, p in parameters.items() if n.startswith(prefix)]
        scope[prefix] = dict(trainable=sum(p.numel() for _, p in members if p.requires_grad),
            gradients=sum(p.numel() for _, p in members if p.grad is not None),
            updated=sum(p.numel() for n, p in members if n in before
                        and not torch.equal(p.detach().cpu(), before[n])))
    if any(scope[p]['updated'] == 0 for p in ('style_conditioner.', 'decoder_layers.')):
        raise RuntimeError('latent conditioner or decoder has no A CUDA update')
    if is_draft(model.config) and scope['condition_view.']['updated'] == 0:
        raise RuntimeError('draft condition view embedding has no A CUDA update')
    cpu = raw.unpack() if hasattr(raw, 'unpack') else raw
    report = dict(operation='original-plan-batch-audio-draft-A' if is_draft(model.config) else 'original-plan-batch-frozen-latent-A', recipe=args.resolved_recipe,
        step=1, loss=float(result.loss.detach()), metrics=result.metrics, gradient_norm=float(norm),
        physical_check_coefficient=1., formal_physical_warmup=args.physical_warmup_steps,
        device=torch.cuda.get_device_name(), seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope=scope, encoding_stats=model.a_encoding_stats,
        plan_epoch=args.plan_offset, references=[list(map(int, (r.canonical_index, r.view_id, r.occurrence)))
                                               for r in sampler._batches[0]],
        target_windows=len(cpu['start_tick']), fixed_plan_batch=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cuda-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + '\n', encoding='utf8')
    return report
