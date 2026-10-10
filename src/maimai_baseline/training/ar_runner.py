"""B-only full-song AR training; no A training, CFG, validation or structure stages."""
import argparse
from maimai_baseline.runtime_versions import stamp
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from maimai_baseline.ar_contract import (CONFIG, RECIPE, ARCHITECTURE, IDENTITY_SCHEMA,
    load_ar_config, validate_identity, stage_recipe, plan_counts, DECODER_OBJECTIVE, TAIL_CONTRACT)
from maimai_baseline.ar_checkpoint import (attach_ar, make_payload,
    save_checkpoint, load_checkpoint, restore_training, serialize_batches)
from maimai_baseline.checkpoint import load_frozen_stage_a
from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.frozen_latent import FrozenLatentCache
from maimai_baseline.data.style_audio_cache import BatchRequests, worker_init
from maimai_baseline.data.frozen_context import FrozenSongContext, SongBatches, SongLoader, plan_requests
from maimai_baseline.model.song_ar import CONTRACT
from maimai_baseline.runtime_versions import operational
from maimai_baseline.training.song_ar import train_step, merge_stats, summarize
from maimai_baseline.training.steps import _move, _scheduler, set_scheduler_progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ar-config', '--recipe', dest='ar_config', type=Path, default=CONFIG)
    for name in ('cache', 'song-context-cache', 'route-artifact', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('stage-a-checkpoint', 'epoch-plan-cache', 'whole-song-plan',
                 'audio-condition-cache', 'dataset-manifest', 'decoder-context-cache', 'teacher-audio-cache'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--num-workers', type=int, default=12, choices=[12])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--execute', action='store_true', default=True, help=argparse.SUPPRESS)
    return p


def inputs(args):
    """Resolve final inputs once; workers receive the resulting read-only index."""
    plans = EpochPlanStore(getattr(args,'whole_song_plan',None) or args.epoch_plan_cache, verify_payloads=False)
    cache = FrozenLatentCache(getattr(args,'latent_cache',None) or args.cache)
    song = FrozenSongContext(args.song_context_cache, cache,
                             teacher_audio_cache=getattr(args, 'teacher_audio_cache', None))
    cache.song_descriptor = song.descriptor()
    if args.resume:
        model, resumed = load_checkpoint(args.resume, route_artifact=args.route_artifact)
        parent = resumed['parent_metadata']
        recipe = getattr(args,'resolved_recipe',None) or load_ar_config(getattr(args,'ar_config',CONFIG))['recipe']
        if stage_recipe(resumed['identity']).get('objective') != recipe.get('objective', 'nll'):
            raise ValueError('B resume objective differs; old NLL-only B cannot resume Decoder-supervised B')
    else:
        resumed = None
        recipe = getattr(args, 'resolved_recipe', None) or load_ar_config(getattr(args,'ar_config',CONFIG))['recipe']
        if not args.stage_a_checkpoint:
            raise ValueError('new B training requires --stage-a-checkpoint')
        model, parent = load_frozen_stage_a(args.stage_a_checkpoint, route_artifact=args.route_artifact)
        from maimai_baseline.style_contract import is_frozen
        if not is_frozen(model.config) or bool(model.style_prior.ready):
            raise ValueError('new B requires an untrained prior in a frozen-mu A checkpoint')
    counts = [plans.batch_count(e) for e in range(plans.available_epochs)]
    from maimai_baseline.style_contract import NATIVE_STYLE_ID
    native = model.config.chart_style == NATIVE_STYLE_ID
    if native != getattr(song, 'native', False):
        raise ValueError('B native-audio model and song context representation differ')
    if native and not getattr(args, 'teacher_audio_cache', None):
        raise ValueError('native-audio B requires --teacher-audio-cache')
    if native and operational(parent.get('baseline_manifest', {}).get('teacher_audio')) != operational(song.manifest.get('teacher_audio')):
        raise ValueError('B teacher audio differs from the native A binding')
    decoder_supervised = recipe.get('objective') == DECODER_OBJECTIVE
    parent_binding = parent.get('baseline_manifest',{}).get('frozen_latent')
    if decoder_supervised:
        from maimai_baseline.data.latent_rebuild import validate_decoder_latent
        from maimai_baseline.training.decoder_context import DecoderContextCache
        if native:
            if operational(parent_binding) != operational(cache.identity):
                raise ValueError('native B latent data differs from the native A binding')
        else:
            validate_decoder_latent(cache, parent)
        if resumed is None and parent.get('epoch') != 89:
            raise ValueError('Decoder-supervised B requires the completed A90 checkpoint (epoch=89)')
        if not getattr(args, 'decoder_context_cache', None):
            raise ValueError('Decoder-supervised B requires --decoder-context-cache')
        candidate = DecoderContextCache(args.decoder_context_cache, song)
        cache.decoder_descriptor = candidate.descriptor()
        model.prior_condition_contract = TAIL_CONTRACT
        model.difficulty_loss_preset = parent.get('difficulty_loss_weighting', {}).get('preset', 'uniform')
    elif operational(parent_binding) != operational(cache.identity):
        raise ValueError('B latent data differs from the frozen A binding')
    plan = dict(optimizer_steps_by_epoch=counts)
    offset = recipe['offset']
    identity = dict(**stamp(), schema=IDENTITY_SCHEMA, architecture=ARCHITECTURE, model=CONTRACT,
        recipe=recipe, plan=plan, total_steps=sum(counts[offset:offset+recipe['scheduler_epochs']]),
        normalization=cache.manifest['normalization'],
        source_identity=parent.get('source_identity'),
        data=dict(latent=operational(cache.identity), context=dict(charts=song.charts,
            record_to_chart=song.mapping,audio=song.manifest['audio'],
            **(dict(schema=song.manifest['schema'], teacher_audio=song.manifest['teacher_audio']) if native else {})),
            plan=dict(segment_order_sha256=plans.manifest.get('segment_order_sha256'),
                      modes=plans.manifest['modes'])))
    if decoder_supervised:
        identity['condition_contract'] = TAIL_CONTRACT
        identity['data']['decoder_context'] = operational(candidate.identity)
    validate_identity(identity)
    if resumed is not None:
        old = resumed['identity']
        saved_recipe = stage_recipe(old)
        if ({k:v for k,v in saved_recipe.items() if k!='epochs'} != {k:v for k,v in recipe.items() if k!='epochs'}
                or recipe['epochs'] < saved_recipe['epochs']):
            raise ValueError('B resume recipe/horizon changed or stop shortened')
        if operational(old.get('data')) != operational(identity['data']):
            raise ValueError('B resume latent/P0 data or fixed plan differs')
        if old['normalization']['local'] != identity['normalization']['local']:
            raise ValueError('AR resume normalization differs from cached targets')
        saved_batches = resumed.get('training', {}).get('plan_batches')
        if saved_batches is not None:
            actual = serialize_batches(plans.load_epoch(recipe['offset'] + resumed['epoch']))
            if saved_batches != actual:
                raise ValueError('AR resume plan references or boundaries differ')
        if plan_counts(old) != counts:
            raise ValueError('AR resume update schedule differs from plan')
    args._resumed_payload = resumed
    levels = {str(c['key'][1]): float(c['level']) for c in song.charts}
    return model, parent, cache, plans, identity, levels


def step_api(identity):
    if stage_recipe(identity).get('objective') == DECODER_OBJECTIVE:
        from maimai_baseline.training import prior_decoder
        return prior_decoder.train_step, prior_decoder.merge_stats, prior_decoder.summarize
    return train_step, merge_stats, summarize


def batch_dataset(args, cache, identity):
    if stage_recipe(identity).get('objective') == DECODER_OBJECTIVE:
        from maimai_baseline.data.prior_decoder_batches import PriorDecoderBatches
        descriptor = dict(target=cache.descriptor(), context=cache.song_descriptor['manifest'],
                          decoder=cache.decoder_descriptor,
                          teacher_audio_cache=cache.song_descriptor.get('teacher_audio_cache'))
        return PriorDecoderBatches(cache.root, args.song_context_cache, args.decoder_context_cache,
            descriptor=descriptor)
    return SongBatches(cache.root, args.song_context_cache, descriptor=getattr(cache,'song_descriptor',None))


def train(args):
    if getattr(args,'cuda_check',False):
        return cuda_check(args)
    if args.num_workers != 12:
        raise ValueError('AR production DataLoader workers must be 12')
    model, parent, cache, plans, identity, levels = inputs(args)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise ValueError('refusing to overwrite AR output')
    if not getattr(args,'execute',True):
        return dict(operation='AR-B', identity=identity, output=str(output), written=False)
    from maimai_baseline.training.steps import _device
    device = _device(args.device)
    recipe = stage_recipe(identity)
    update, merge, summarize_step = step_api(identity)
    seed = recipe['seed']
    seed_rng(seed)
    attach_ar(model, seed=seed)
    norm = identity['normalization']['local']
    model.style_prior.set_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    resumed = getattr(args, '_resumed_payload', None)
    model.to(device)
    model.eval()
    model.style_prior.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(model.style_prior.parameters(), lr=recipe['learning_rate'],
                                  weight_decay=recipe['weight_decay'])
    scheduler = _scheduler(optimizer, warmup_steps=recipe['warmup_steps'], total_steps=identity['total_steps'])
    loader_rng = torch.Generator().manual_seed(seed)
    start_epoch = step = 0
    if resumed is not None:
        start_epoch, step = restore_training(resumed, optimizer, scheduler, loader_rng)
        if start_epoch >= recipe['epochs']:
            raise ValueError('AR stage is already complete')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'run.json', dict(**stamp(), resumed_from_version=None if resumed is None else resumed.get('runtime_version'), identity=identity, device=str(device),
        trainable_prior_parameters=sum(p.numel() for p in model.style_prior.parameters()),
        frozen='all-non-prior', stages=['B'], validation=False))
    write_json(output/'level-conditions.json', dict(levels=levels))
    dataset = batch_dataset(args, cache, identity)
    requests = BatchRequests()
    loader = SongLoader(dataset, batch_size=None, sampler=requests, num_workers=12,
        worker_init_fn=worker_init, persistent_workers=True, pin_memory=device.type == 'cuda',
        generator=loader_rng)
    restored_loader_rng = loader_rng.get_state().clone() if resumed is not None else None
    for epoch in range(start_epoch, recipe['epochs']):
        started = time.perf_counter()
        pe = recipe['offset']+epoch
        epoch_batches = plans.load_epoch(pe)
        requests.set_batches((plan_requests(cache,refs) for refs in epoch_batches), epoch=pe)
        totals, gradients, batches = {}, [], 0
        peak_allocated = peak_reserved = 0
        final = epoch + 1 == recipe['epochs']
        iterator = epoch_iterator(loader,restored_loader_rng)
        restored_loader_rng = None
        for raw in iterator:
            batch = _move(raw, device)
            set_scheduler_progress(scheduler, step)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            stats, grad = update(model, optimizer, batch, collect_metrics=final,
                                     gradient_clip=recipe['gradient_clip'], precision=recipe['precision'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            batches += 1
            set_scheduler_progress(scheduler, step)
            merge(totals, stats)
            gradients.append(grad)
            cost = dict(epoch=epoch+1, step=step, seconds=time.perf_counter()-tick,
                audio_groups=batch['audio_count'], charts=batch['chart_count'],
                coordinates=stats['coordinates'],
                memory_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                memory_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
            for key in ('decoder_microbatches', 'decoder_audio_encoder_calls', 'condition_blocks', 'prior_tail_steps'):
                if key in stats:
                    cost[key] = stats[key]
            if device.type == 'cuda':
                peak_allocated = max(peak_allocated, cost['memory_allocated'])
                peak_reserved = max(peak_reserved, cost['memory_reserved'])
            if final:
                cost.update(summarize_step(stats))
                cost.update(prior_audio_kv_calls=batch['prior_audio_kv_calls'],
                    prior_control_kv_calls=batch['prior_control_kv_calls'],
                    prior_local_audio_calls=batch['prior_local_audio_calls'])
                if 'decoder_diagnostics' in batch:
                    cost['decoder_diagnostics'] = batch['decoder_diagnostics']
            with (output/'batch-costs.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(cost, allow_nan=False)+'\n')
            if batches == 1 or batches % 50 == 0:
                print(json.dumps(dict(stage='AR-B', epoch=epoch+1, step=step, **summarize_step(stats)), allow_nan=False), flush=True)
        if batches != plans.batch_count(pe) or not batches:
            raise ValueError('AR incomplete epoch batch count')
        model.style_prior.ready.fill_(True)
        metrics = dict(stage='AR-B', epoch=epoch+1, plan_epoch=pe+1, step=step,
            prediction='joint-diagonal-gaussian-mixture', components=4, validation_performed=False, collect_metrics=final,
            gradient_norm=sum(gradients)/len(gradients), learning_rate=optimizer.param_groups[0]['lr'],
            seconds=time.perf_counter()-started, optimizer_updates=batches,
            peak_memory_allocated=peak_allocated if device.type == 'cuda' else None,
            peak_memory_reserved=peak_reserved if device.type == 'cuda' else None, **summarize_step(totals))
        save_started = time.perf_counter()
        payload = make_payload(model, parent, identity, epoch=epoch, step=step, metrics=metrics,
            optimizer=optimizer, scheduler=scheduler, loader_generator=loader_rng,
            plan_batches=serialize_batches(epoch_batches))
        save_checkpoint(output/'last.ckpt', payload)
        if (epoch+1) % 5 == 0 or final:
            save_checkpoint(output/'checkpoints'/f'epoch-{epoch+1:03d}.ckpt', payload)
        if final:
            save_checkpoint(output/'ar.ckpt', payload)
        metrics.update(save_seconds=time.perf_counter()-save_started,
                       epoch_wall_seconds=time.perf_counter()-started)
        if final:
            write_json(output/'final-metrics.json', metrics)
        with (output/'metrics.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
        print(json.dumps(metrics, allow_nan=False), flush=True)
    return dict(operation='AR-B', checkpoint=str(output/'ar.ckpt'), epochs=recipe['epochs'], step=step)


def epoch_iterator(loader,restored_loader_rng=None):
    iterator = iter(loader)  # SongLoader eagerly creates workers/base_seed.
    if restored_loader_rng is not None and loader.persistent_workers:
        loader.generator.set_state(restored_loader_rng)
    return iterator


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_check(args):
    """One actual fixed B batch/update; no epoch training or qualification marker."""
    if not torch.cuda.is_available():
        raise RuntimeError('B original-batch CUDA check requires CUDA')
    output=Path(args.output).resolve()
    if output.exists():
        raise ValueError('refusing to overwrite B CUDA check output')
    model,parent,cache,plans,identity,_ = inputs(args)
    recipe = stage_recipe(identity)
    update, _, summarize_step = step_api(identity)
    seed_rng(recipe['seed'])
    attach_ar(model,seed=recipe['seed'])
    norm = identity['normalization']['local']
    model.style_prior.set_normalization(torch.tensor(norm['mean']),torch.tensor(norm['std']))
    device = torch.device('cuda')
    model.to(device)
    dataset = batch_dataset(args,cache,identity)
    refs=plans.load_epoch(recipe['offset'])[0]
    request = (plan_requests(cache,refs),recipe['offset'],False)
    batch = _move(dataset[request],device)
    optimizer = torch.optim.AdamW(model.style_prior.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    from maimai_baseline.ar_checkpoint import require_frozen
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    stats,grad = update(model,optimizer,batch,collect_metrics=True,
        gradient_clip=recipe['gradient_clip'],precision=recipe['precision'])
    torch.cuda.synchronize(device)
    require_frozen(model,parent)
    report=dict(operation='B-original-batch-cuda',seconds=time.perf_counter()-started,
        memory_allocated=torch.cuda.max_memory_allocated(device),memory_reserved=torch.cuda.max_memory_reserved(device),
        gradient_norm=grad,charts=batch['chart_count'],unique_audio=batch['audio_count'],
        precision=recipe['precision'],budget=recipe['token_budget'])
    report.update(summarize_step(stats))
    report['prior_audio_kv_calls'] = batch['prior_audio_kv_calls']
    for key in ('decoder_microbatches', 'decoder_audio_encoder_calls', 'condition_blocks', 'prior_tail_steps'):
        if key in stats:
            report[key] = stats[key]
    if 'decoder_diagnostics' in batch:
        report['decoder_diagnostics'] = batch['decoder_diagnostics']
    write_cuda_report(output,report,refs,recipe)
    return report


def write_cuda_report(output,report,refs,recipe):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'cuda-check.json',dict(**stamp(),**report,resolved_recipe=recipe,
        plan_epoch=recipe['offset']+1,references=serialize_batches([refs])[0]))


def main():
    print(json.dumps(train(parser().parse_args()), allow_nan=False))
