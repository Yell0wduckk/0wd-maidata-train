"""Complete-epoch orchestration for the one shared E/D/R/P/M model."""
from itertools import islice
import json
from pathlib import Path
import time

import torch

from maidata.decoder.resources import load_route_distances
from maidata.decoder_training.runner import chunk_loss
from maidata.training.runner import Diagnostics, supports
from maidata.training.data import epoch_iterator
from . import checkpoint, recipe
from .data import JointPlan, make_loader
from .network import JointTrainer
from .update import JointMetrics, update


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@torch.no_grad()
def validate(model, loader, config, geometry, *, final, restored_rng=None):
    mode, rng = model.training, checkpoint.rng_state()
    model.eval()
    device = next(model.parameters()).device
    metrics = JointMetrics()
    auxiliary_diagnostics = Diagnostics() if final else None
    decoder_diagnostics = dict(field_accuracy={}, time_target_probability={}, windows=0)
    try:
        for raw in epoch_iterator(loader, restored_rng):
            batch = raw.to(device)
            with recipe.autocast(device, config):
                terms, conditions, outputs = model.forward_shared(batch)
                metrics.add_auxiliary(terms, supports(raw))
                if final:
                    auxiliary_diagnostics.add(outputs[0], batch.reconstruction)
                audio_kv = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
                for cpu_chunk in batch.chunks:
                    chunk = cpu_chunk.to(device)
                    decoder_terms, output = chunk_loss(model, conditions, audio_kv, chunk, config, geometry,
                        diagnostics=decoder_diagnostics['field_accuracy'] if final else None,
                        time_diagnostics=decoder_diagnostics['time_target_probability'] if final else None)
                    metrics.add_decoder(decoder_terms)
                    del decoder_terms, output, chunk
            decoder_diagnostics['windows'] += len(raw.windows)
            del terms, outputs, batch, conditions, audio_kv
        result = metrics.report(config)
        if final:
            result['diagnostics'] = dict(auxiliary=auxiliary_diagnostics.report(), decoder=decoder_diagnostics)
        if not torch.isfinite(torch.tensor(result['loss'])):
            raise ValueError('nonfinite joint validation loss')
        return result
    finally:
        model.train(mode)
        checkpoint.restore_rng(rng)


def select_check_step(plan, config, steps, requested):
    if requested is not None:
        if not 0 <= requested < len(steps):
            raise ValueError('check step outside first epoch')
        return requested
    from maidata.training.batching import group_token_cost
    costs = [max(sum(group_token_cost(plan.groups[key], config) for key in batch)
                 for batch in step) for step in steps]
    return max(range(len(costs)), key=costs.__getitem__)


def run(config, *, resume=None, cuda_check=False, cost_check=False, check_step=None,
        upgrade_z_normalization=False, upgrade_single_plan16=False):
    config = recipe.resolve(config)
    if upgrade_z_normalization and not resume:
        raise ValueError('Z normalization upgrade requires a complete joint resume checkpoint')
    if upgrade_single_plan16 and (not resume or upgrade_z_normalization):
        raise ValueError('single Plan16 upgrade requires its own complete joint resume without another upgrade')
    recipe.validate(config)
    device = torch.device(config['joint_train_device'])
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            config['joint_train_precision'] == 'bf16' and not torch.cuda.is_bf16_supported()):
        raise ValueError('requested CUDA/bf16 runtime is unavailable')
    if (cuda_check or cost_check) and (device.type != 'cuda' or resume):
        raise ValueError('joint CUDA diagnostics require an independent scratch CUDA model')
    output = Path(config['joint_train_output'])
    if not resume and not (cuda_check or cost_check) and output.exists() and any(output.iterdir()):
        raise FileExistsError('fresh joint output must be empty')
    plans = [JointPlan(config['pretrain_' + name], config, training=index == 0)
             for index, name in enumerate(('train_plan', 'validation_plan'))]
    loaders = [make_loader(plan, config) for plan in plans]
    steps = [len(plans[0].batches(epoch, config))
             for epoch in range(1, config['joint_train_scheduler_epochs'] + 1)]
    binding, source = recipe.provenance(config), checkpoint.source_identity()
    if not resume:
        recipe.seed(config)
    model = JointTrainer(config).to(device)
    geometry = load_route_distances(config['decoder_train_route_distances']).to(device)
    opt = recipe.optimizer(model.parameters(), config)
    sched = recipe.scheduler(opt, steps, config)
    completed, step, best, restored = 0, 0, None, [None, None]
    plan_upgrade = None
    if resume:
        payload = torch.load(resume, map_location='cpu', weights_only=False)
        completed, step, best = checkpoint.restore(payload, model, opt, sched,
            config=config, loaders=loaders, upgrade_z_normalization=upgrade_z_normalization,
            upgrade_single_plan16=upgrade_single_plan16, binding=binding, steps=steps)
        if upgrade_single_plan16:
            plan_upgrade = dict(from_layout=payload['recipe']['planner_layout'],
                to_layout=config['planner_layout'],
                from_gt_plan_cache=payload['binding']['paths']['decoder_train_gt_plan_cache'],
                to_gt_plan_cache=binding['paths']['decoder_train_gt_plan_cache'],
                source_checkpoint=payload.get('source'))
        restored = [state if started else None for state, started in
                    zip(payload['loader_rng'], payload['loader_started'])]
        del payload
    if cuda_check or cost_check:
        selected = select_check_step(plans[0], config, plans[0].batches(1, config), check_step)
        loaders[0].batch_sampler.epoch = 1
        loaders[0].batch_sampler.selected_step = selected
        start = time.perf_counter()
        batches = list(islice(epoch_iterator(loaders[0]), config['pretrain_gradient_accumulation_steps']))
        data_seconds = time.perf_counter() - start
        from .cost import audit_update, measure
        report = (measure if cost_check else audit_update)(model, batches, opt, sched, config, geometry)
        report.update(source=source, binding=binding, config=config, optimizer_step_in_epoch=selected + 1,
                      data_wait_seconds=data_seconds, steps_by_epoch=steps)
        from maidata.training.batching import group_token_cost
        report['group_proxy_costs'] = [sum(group_token_cost(plans[0].groups[key], config)
                                         for key in batch.audio_variants) for batch in batches]
        write_json(output.parent / 'diagnostics' / ('cuda-cost.json' if cost_check else 'cuda-audit.json'), report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return report
    if completed >= config['joint_train_stop_after_epochs']:
        return dict(completed_epochs=completed, step=step)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ('resume.json' if resume else 'run.json'),
        dict(config=config, source=source, binding=binding, steps_by_epoch=steps,
             resume=str(resume) if resume else None, starting_completed_epochs=completed))
    if upgrade_z_normalization:
        write_json(output / 'z-normalization-upgrade.json', dict(
            resume=str(resume), completed_epochs=completed, step=step,
            change='parameter-free LayerNorm after M Z projection plus source embedding',
            inherited='all E/D/R/P/M parameters, optimizer, scheduler, complete-epoch cursor and all RNG',
            source=source))
    if plan_upgrade is not None:
        write_json(output / 'single-plan16-upgrade.json', dict(plan_upgrade,
            resume=str(resume), completed_epochs=completed, step=step,
            inherited='all E/D/R/P/M parameters, optimizer, scheduler, complete-epoch cursor and all RNG',
            source=source))
    for epoch in range(completed + 1, config['joint_train_stop_after_epochs'] + 1):
        epoch_start = time.perf_counter()
        final = epoch == config['joint_train_stop_after_epochs']
        training_diagnostics = (Diagnostics() if final and
            config.get('reconstruction_onset_presence_supervision', False) else None)
        loaders[0].batch_sampler.epoch = epoch
        iterator = epoch_iterator(loaders[0], restored[0])
        restored[0] = None
        metrics, updates, windows, auxiliary_windows, audio_groups = JointMetrics(), 0, 0, 0, set()
        while True:
            data_start = time.perf_counter()
            batches = list(islice(iterator, config['pretrain_gradient_accumulation_steps']))
            data_seconds = time.perf_counter() - data_start
            if not batches:
                break
            update_start = time.perf_counter()
            current, audit = update(model, batches, opt, sched, config, geometry,
                **({'diagnostics': training_diagnostics} if training_diagnostics is not None else {}))
            update_seconds = time.perf_counter() - update_start
            metrics.merge(current)
            updates += 1
            step += 1
            windows += sum(len(batch.windows) for batch in batches)
            auxiliary_windows += sum(len(batch.window_ids) for batch in batches)
            audio_groups.update(key for batch in batches for key in batch.audio_variants)
            report = current.report(config)
            print(json.dumps(dict(epoch=epoch, update=updates, updates=steps[epoch - 1], step=step,
                loss=report['loss'], losses=report['losses'], lr=opt.param_groups[0]['lr'],
                data_wait_seconds=data_seconds, update_seconds=update_seconds,
                chunks=audit['chunks'], batch_costs=audit['batch_costs']), allow_nan=False), flush=True)
        result = dict(epoch=epoch, step=step, train=metrics.report(config),
                      seconds=time.perf_counter() - epoch_start, audio_groups=len(audio_groups),
                      decoder_occurrences=windows, auxiliary_base_windows=auxiliary_windows)
        if training_diagnostics is not None:
            result['train']['diagnostics'] = dict(auxiliary=training_diagnostics.report(),
                population='final training epoch, teacher-conditioned reconstruction',
                numeric_count_supervision=config.get('reconstruction_count_supervision', True))
        improved = False
        if plans[1].groups and (epoch % 5 == 0 or final):
            loaders[1].batch_sampler.epoch = epoch
            result['validation'] = validate(model, loaders[1], config, geometry,
                                             final=final, restored_rng=restored[1])
            restored[1] = None
            value = result['validation']['loss']
            improved = best is None or value < best
            best = value if improved else best
        payload = checkpoint.capture(model, opt, sched, epoch=epoch, step=step, config=config,
            binding=binding, source=source, steps=steps, loaders=loaders, best=best, metrics=result)
        checkpoint.atomic_save(output / 'checkpoints/last.ckpt', payload)
        if improved:
            checkpoint.atomic_save(output / 'checkpoints/best.ckpt', payload)
        if epoch % 5 == 0 or final:
            checkpoint.atomic_save(output / f'checkpoints/epoch-{epoch:03d}.ckpt', payload)
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(completed_epoch=epoch, step=step,
            validation_loss=result.get('validation', {}).get('loss'))), flush=True)
    return dict(completed_epochs=epoch, step=step)
