"""Independent audio-draft reconstruction; complete-epoch training and CUDA cost check."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import time

import torch

from maimai_baseline.model.audio_draft import AudioDraftPretrainer, pretraining_loss
from maimai_baseline.training import audio_draft_checkpoint as checkpoint
from maimai_baseline.training.chart_encoder_pretrain import (
    DEFAULT_RECIPE as ENCODER_RECIPE, _accumulate, _autocast, _device, _jsonable, _means, _move, _seed,
    _write_json, epoch_iterator, make_optimizer, make_scheduler, plan_step_counts,
    resolve_recipe as _resolve_recipe)

DEFAULT_RECIPE = dict(ENCODER_RECIPE, validation='required')


def resolve_recipe(args):
    # Share numerical optimizer/scheduler validation, not an encoder model or route.
    original = getattr(args, 'experiment_config', None)
    document = {} if original is None else (dict(original) if isinstance(original, Mapping)
                else json.loads(Path(original).read_text('utf-8-sig')))
    if not isinstance(document, dict):
        raise ValueError('audio draft recipe must be a JSON object')
    document = dict(document.get('recipe', document))
    stages = document.get('stages', {})
    if 'audio_draft' in stages:
        document.update({key: value for key, value in stages['audio_draft'].items() if key in DEFAULT_RECIPE})
    document.pop('stages', None)
    validation = document.get('validation', DEFAULT_RECIPE['validation'])
    if validation not in ('disabled', 'required'):
        raise ValueError('audio draft validation must be required or explicitly disabled for CPU fixtures')
    # Keep the independent encoder resolver's disabled-only policy unchanged.
    document['validation'] = 'disabled'
    args.experiment_config = document
    try:
        result = _resolve_recipe(args)
        result['validation'] = validation
        return result
    finally:
        args.experiment_config = original


@torch.no_grad()
def evaluate(model, validation, recipe, *, restored_loader_rng=None):
    """Aggregate coordinate sums globally, preserving the next training update."""
    mode, rng = model.training, checkpoint.rng_state()
    sums = {name: dict(sum=0., count=0) for name in ('ce', 'time', 'count')}
    diagnostics = {}
    endings = dict(windows=0, coordinate_hits=0, first_hits=0, row_error=0, missing=0)
    try:
        model.eval()
        device = next(model.parameters()).device
        for raw in epoch_iterator(validation.loader, restored_loader_rng):
            batch = _move(raw, device)
            with _autocast(device, recipe['precision']):
                outputs = model(batch)
                loss, metrics = pretraining_loss(outputs, batch['targets'], collect_metrics=True)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('non-finite audio draft validation loss')
            for name, term in outputs['loss_terms'].items():
                sums[name]['sum'] += float(term['sum'].detach())
                sums[name]['count'] += int(term['count'])
            _accumulate(diagnostics, metrics)
            event_id = model.vocabulary.event_id('SEGMENT_END')
            predicted = outputs['field_logits']['event'].argmax(-1)
            target = batch['targets']['ids']['event']
            for actual, expected in zip(predicted, target):
                expected_rows = expected.eq(event_id).nonzero().flatten()
                if not len(expected_rows):
                    continue  # Batch padding never contributes EOS statistics.
                if len(expected_rows) != 1:
                    raise ValueError('canonical draft validation window must have exactly one EOS')
                truth = int(expected_rows[0])
                candidates = actual.eq(event_id).nonzero().flatten()
                found = int(candidates[0]) if len(candidates) else len(actual)
                endings['windows'] += 1
                endings['coordinate_hits'] += int(actual[truth]) == event_id
                endings['first_hits'] += found == truth
                endings['row_error'] += abs(found-truth)
                endings['missing'] += not len(candidates)
    finally:
        model.train(mode)
        checkpoint.restore_rng(rng)
    components = {name: term['sum'] / max(1, term['count']) for name, term in sums.items()}
    denominator = max(1, endings['windows'])
    eos = dict(windows=endings['windows'], target_coordinate_accuracy=endings['coordinate_hits']/denominator,
        first_end_accuracy=endings['first_hits']/denominator, first_end_row_mae=endings['row_error']/denominator,
        missing_rate=endings['missing']/denominator, missing_position='fixed-query-capacity')
    return dict(loss=sum(components.values()), components=components, loss_terms=sums,
                diagnostics=_means(diagnostics), eos=eos, windows=len(validation.dataset))


def update(model, batch, optimizer, scheduler, recipe, *, collect_metrics=False):
    device = next(model.parameters()).device
    batch = _move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, recipe['precision']):
        outputs = model(batch)
        loss, metrics = pretraining_loss(outputs, batch['targets'], collect_metrics=collect_metrics)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('non-finite audio draft reconstruction loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), metrics, float(norm), outputs


@torch.no_grad()
def final_diagnostics(model, batch, *, held_out=False):
    model.eval()
    batch = _move(batch, next(model.parameters()).device)
    actual = model(batch)
    values, valid = actual['latents'], actual['valid']
    mismatch = values.roll(1, 0) if len(values) > 1 else values.roll(1, 1)
    variants = dict(real=actual, zero=model.decode(torch.zeros_like(values), valid),
                    mismatched=model.decode(mismatch, valid))
    metrics, examples = {}, []
    for name, outputs in variants.items():
        loss, fields = pretraining_loss(outputs, batch['targets'], collect_metrics=True)
        metrics[name] = dict(loss=float(loss), fields=fields)
    end_id = model.vocabulary.event_id('SEGMENT_END')
    for index in range(len(values)):
        target = batch['targets']['ids']['event'][index]
        endings = target.eq(end_id).nonzero().flatten()
        sample = dict(reference=_jsonable(batch.get('references', list(range(len(values))))[index]),
            segment_id=batch.get('segment_id', [None]*len(values))[index],
            target_end_row=int(endings[0]) if len(endings) else None,
            target_ids={key: value[index].cpu().tolist() for key, value in batch['targets']['ids'].items()},
            target_times_ticks={key: value[index].cpu().tolist() for key, value in batch['targets']['times'].items()},
            variants={})
        if 'moment_counts' in batch['targets']:
            sample['target_moment_counts'] = batch['targets']['moment_counts'][index].cpu().tolist()
        for name, outputs in variants.items():
            fields = {key: value[index].argmax(-1).cpu().tolist() for key, value in outputs['field_logits'].items()}
            endings = [row for row, event in enumerate(fields['event']) if event == end_id]
            end = endings[0] if endings else None
            length = model.query_rows if end is None else end+1
            sample['variants'][name] = dict(end_row=end, ids={key: value[:length] for key, value in fields.items()},
                times_beats={key: (value[index].float() if key == 'moment_position_tick'
                    else value[index].float().expm1().clamp_min(0.))[:length].cpu().tolist()
                    for key, value in outputs['time_predictions'].items()})
            if 'count_logits' in outputs:
                sample['variants'][name]['moment_counts'] = outputs['count_logits'][index].argmax(-1).cpu().tolist()
        examples.append(sample)
    return dict(scope='fixed-validation-sample' if held_out else 'fixed-training-plan-sample',
        held_out=held_out, metrics=metrics, examples=examples,
        mismatch='previous-window-in-sample' if len(values) > 1 else 'previous-beat-slot')


def _prepare(args):
    from maimai_baseline.data.audio_draft import prepare_inputs
    return prepare_inputs(args)


def train(args):
    if getattr(args, 'cuda_check', False):
        return cuda_check(args)
    from maimai_baseline.data.chart_encoder_pretrain import fixed_diagnostic_batch
    recipe = resolve_recipe(args)
    device = _device(args)
    _seed(recipe['seed'])
    inputs = _prepare(args)
    validation = getattr(inputs, 'validation', None)
    if recipe['validation'] != 'disabled' and validation is None:
        raise ValueError('draft training requires the declared held-out validation dataset')
    counts = plan_step_counts(inputs.plan_store, recipe)
    output = Path(args.output).resolve()
    resume = getattr(args, 'resume', None)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite an existing audio draft run')
    if resume:
        model, payload = checkpoint.load_checkpoint(resume, device=device)
        start_epoch, step = checkpoint.validate_resume(payload, recipe=recipe,
            data_binding=inputs.binding, steps_by_epoch=counts)
        if model.query_rows != inputs.query_rows:
            raise ValueError('audio draft resume fixed row query capacity changed')
    else:
        model = AudioDraftPretrainer(inputs.vocabulary, inputs.query_rows, seed=recipe['seed']).to(device)
        start_epoch = step = 0
    optimizer = make_optimizer(model, recipe)
    scheduler = make_scheduler(optimizer, recipe, counts)
    restored_loader_rng = restored_validation_rng = None
    best_validation_loss = None
    if resume:
        checkpoint.restore_training_state(payload, optimizer=optimizer, scheduler=scheduler,
            loader_generator=inputs.loader_generator,
            validation_generator=None if validation is None else validation.loader_generator)
        restored_loader_rng = inputs.loader_generator.get_state().clone()
        if validation is not None and payload.get('best_validation_loss') is not None:
            restored_validation_rng = validation.loader_generator.get_state().clone()
        best_validation_loss = payload.get('best_validation_loss')
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = dict(recipe=recipe, data_binding=inputs.binding, steps_by_epoch=counts,
        total_scheduler_steps=sum(counts), query_rows=inputs.query_rows,
        resume=None if not resume else dict(checkpoint=str(Path(resume).resolve()),
                                            completed_epochs=start_epoch, step=step))
    if resume and (output/'run.json').is_file():
        previous = json.loads((output/'run.json').read_text('utf8'))
        with (output/'resume-history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(previous=previous, current=run_metadata), ensure_ascii=False, allow_nan=False)+'\n')
    _write_json(output/'run.json', run_metadata)
    for epoch in range(start_epoch, recipe['epochs']):
        begun = time.perf_counter()
        inputs.sampler.set_epoch(recipe['offset']+epoch)
        model.train()
        diagnostics, losses = {}, []
        last = epoch+1 == recipe['epochs']
        for raw in epoch_iterator(inputs.loader, restored_loader_rng if epoch == start_epoch else None):
            loss, metrics, _, _ = update(model, raw, optimizer, scheduler, recipe, collect_metrics=last)
            losses.append(loss)
            _accumulate(diagnostics, metrics)
            step += 1
        if len(losses) != counts[epoch]:
            raise ValueError('audio draft loader changed the canonical fixed plan batch boundary')
        row = dict(epoch=epoch+1, step=step, loss=sum(losses)/len(losses),
            seconds=time.perf_counter()-begun, diagnostics=_means(diagnostics) if last else {})
        improved = False
        if validation is not None and ((epoch+1) % 5 == 0 or last):
            row['validation'] = evaluate(model, validation, recipe, restored_loader_rng=restored_validation_rng)
            restored_validation_rng = None
            improved = best_validation_loss is None or row['validation']['loss'] < best_validation_loss
            if improved:
                best_validation_loss = row['validation']['loss']
        if last:
            report = final_diagnostics(model, fixed_diagnostic_batch(inputs, args, count=4))
            _write_json(output/'reconstruction.json', report)
            row['sample_diagnostics'] = report['metrics']
            if validation is not None:
                references = [ref for batch in validation.sampler.batches for ref in batch][:4]
                validation_batch = validation.collator([validation.dataset[ref] for ref in references])
                heldout = final_diagnostics(model, validation_batch, held_out=True)
                _write_json(output/'validation-reconstruction.json', heldout)
                row['validation_sample_diagnostics'] = heldout['metrics']
        checkpoint.save_checkpoint(output/'checkpoints/last.ckpt', model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, step=step, recipe=recipe, data_binding=inputs.binding,
            steps_by_epoch=counts, loader_generator=inputs.loader_generator, metrics=row,
            source_provenance=getattr(args, 'source_identity', None),
            validation_generator=None if validation is None else validation.loader_generator,
            best_validation_loss=best_validation_loss)
        if improved:
            saved = torch.load(output/'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
            checkpoint._atomic_save(output/'checkpoints/best.ckpt', saved)
        if (epoch+1) % 5 == 0 or last:
            saved = torch.load(output/'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
            checkpoint._atomic_save(output/f'checkpoints/epoch-{epoch+1:03d}.ckpt', saved)
        with (output/'metrics.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    checkpoint.export_draft(output/'draft.pt', model=model, recipe=recipe, data_binding=inputs.binding,
        source_provenance=getattr(args, 'source_identity', None), completed_epochs=recipe['epochs'])
    return output/'draft.pt'


def cuda_check(args):
    """One unchanged logical batch update; cost evidence, not formal training."""
    recipe = resolve_recipe(args)
    if not torch.cuda.is_available():
        raise RuntimeError('original-batch audio draft CUDA check requires CUDA')
    device = torch.device('cuda')
    _seed(recipe['seed'])
    inputs = _prepare(args)
    inputs.sampler.set_epoch(recipe['offset'])
    raw = next(iter(inputs.loader))
    model = AudioDraftPretrainer(inputs.vocabulary, inputs.query_rows, seed=recipe['seed']).to(device).train()
    optimizer = make_optimizer(model, recipe)
    scheduler = make_scheduler(optimizer, recipe, plan_step_counts(inputs.plan_store, recipe))
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    loss, metrics, norm, outputs = update(model, raw, optimizer, scheduler, recipe, collect_metrics=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-begun
    scope = {prefix: dict(trainable=sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix)),
        gradients=sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None),
        updated=sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix)
                    and not torch.equal(p.detach().cpu(), before[n])))
        for prefix in ('generator.', 'reconstruction.')}
    if any(row['updated'] == 0 for row in scope.values()):
        raise RuntimeError('audio draft CUDA check has no generator/reconstruction update')
    cpu = raw.unpack() if hasattr(raw, 'unpack') else raw
    report = dict(operation='original-plan-batch-audio-draft-update', recipe=recipe,
        step=1, loss=loss, metrics=metrics, gradient_norm=norm, scope=scope,
        device=torch.cuda.get_device_name(), seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        encoding_stats=outputs['encoding_stats'], references=_jsonable(cpu.get('references', [])),
        windows=len(cpu['window_groups']), query_rows=model.query_rows, fixed_plan_batch=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'cuda-check.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return output/'cuda-check.json'


def run(args, command_args=None):
    return train(args)
