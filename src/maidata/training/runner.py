"""Whole-song E/D/R/P updates, validation and complete-epoch continuation."""
from itertools import islice
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from maidata.draft.losses import FIELD_PREFIX
from maidata.draft.schema import FIELD_SIZES
from maidata.draft.route_vocabulary import route_class_count, route_groups, route_targets

from . import checkpoint
from .batching import group_token_cost
from .data import AudioPlan, make_loader, epoch_iterator
from .network import Pretrainer
from .frozen_p0 import resolve_audio_config
from .pretrain import (LOSS_WEIGHT_KEYS, seed_scratch, make_optimizer, make_scheduler,
                       clip_gradients, autocast, epoch_range, weighted_loss, loss_weight)


def supports(batch):
    r, p = batch.reconstruction, batch.planner
    active = r.valid[:, None] & r.queries.valid
    result = dict({FIELD_PREFIX + name: int(((labels != -100) & active).sum())
                 for name, labels in r.fields.items()},
        moment96=int(((r.moment_tick != -100) & active).sum()),
        count=int(r.valid.sum()) * r.moment_counts.shape[1],
        empty=int(r.valid.sum()) * r.moment_counts.shape[1],
        positive_count=int(((r.moment_counts > 0) & r.valid[:, None]).sum()),
        planner_continuous=int(p.valid.sum()), planner_category=int(p.valid.sum()))
    presence = getattr(r.queries, 'onset_presence', None)
    if presence is not None:
        mask = r.valid[:, None] & presence.valid
        positive = int(((r.onset_presence_labels > 0) & mask).sum())
        count = int(mask.sum())
        result.update(onset_presence=count, onset_presence_positive=positive,
                      onset_presence_negative=count-positive)
    return result


def query_cost(batch):
    """Report actual local queries separately from the preserved batch proxy."""
    r = batch.reconstruction
    count = int(r.queries.valid.sum()) + int(r.valid.sum()) * r.moment_counts.shape[1]
    if getattr(r.queries, 'onset_presence', None) is not None:
        count += int((r.queries.onset_presence.valid & r.valid[:, None]).sum())
    return count


class Metrics:
    def __init__(self):
        self.sums = {k: [0., 0] for k in
                     (*[FIELD_PREFIX + name for name in FIELD_SIZES],
                      *[name for name in LOSS_WEIGHT_KEYS if name != 'classification'])}
        self.presence_positive = self.presence_negative = 0

    def add(self, terms):
        for name, term in terms.items():
            self.sums[name][0] += float(term.total.detach())
            self.sums[name][1] += int(term.count)
            if name == 'onset_presence':
                self.presence_positive += int(term.positive_count)
                self.presence_negative += int(term.negative_count)

    def report(self, config):
        terms = {k: dict(total=v[0], count=v[1], mean=v[0]/max(1, v[1])) for k,v in self.sums.items()}
        if terms['onset_presence']['count']:
            terms['onset_presence'].update(positive_count=self.presence_positive,
                                           negative_count=self.presence_negative,
                                           unit='owner_z_beat')
        fields = [v['mean'] for k,v in terms.items() if k.startswith(FIELD_PREFIX) and v['count']]
        classification = sum(fields) / max(1, len(fields))
        loss = classification * config[LOSS_WEIGHT_KEYS['classification']]
        loss += sum(terms[k]['mean'] * loss_weight(config, w) for k,w in LOSS_WEIGHT_KEYS.items() if k != 'classification')
        return dict(loss=loss, classification=dict(mean=classification, active_fields=len(fields)), terms=terms)


def update(model, batches, optimizer, scheduler, config, *, audit=False):
    device = next(model.parameters()).device
    if any(not math.isfinite(loss_weight(config, k)) or loss_weight(config, k) < 0 for k in LOSS_WEIGHT_KEYS.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    total_counts = {k: 0 for k in Metrics().sums}
    for batch in batches:
        for k, n in supports(batch).items():
            if k in total_counts:
                total_counts[k] += n
    optimizer.zero_grad(set_to_none=True)
    metrics = Metrics()
    before = {k: [p.detach().clone() for p in m.parameters()] for k,m in model.named_children()} if audit else {}
    for raw in batches:
        batch = raw.to(device)
        with autocast(device, config):
            terms, _ = model(batch)
            loss = weighted_loss(terms, config, denominators=total_counts)
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
    if audit:
        for name, module in model.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            if trainable and (gradients[name] <= 0 or changed[name] == 0):
                raise ValueError(f'{name} must receive gradients and update in the CUDA batch check')
            if not trainable and (gradients[name] != 0 or changed[name] != 0):
                raise ValueError(f'frozen {name} changed in the CUDA batch check')
    return metrics, dict(gradient_norm=float(norm), gradients=gradients, changed_parameter_tensors=changed)


class Diagnostics:
    """Exact final-only confusion counts, with no quality acceptance threshold."""
    def __init__(self):
        self.route_vocabulary = None
        self.confusion = {name: torch.zeros(size, size, dtype=torch.long)
                          for name, size in FIELD_SIZES.items()}
        self.moment = torch.zeros(96, 96, dtype=torch.long)
        self.empty = torch.zeros(2, 2, dtype=torch.long)
        self.positive = torch.zeros(96, 96, dtype=torch.long)
        self.composed = [0, 0]
        self.regression = dict(squared_error=0., absolute_error=0., correct=0, count=0)
        self.count_truth, self.count_prediction = Counter(), Counter()
        self.onset_presence = torch.zeros(2, 2, dtype=torch.long)
        self.onset_presence_seen = False

    @staticmethod
    def add_confusion(table, logits, labels, active):
        truth = labels[active].detach().cpu()
        prediction = logits.argmax(-1)[active].detach().cpu()
        size = table.shape[0]
        table += torch.bincount(truth * size + prediction, minlength=size * size).reshape(size, size)

    def add(self, output, target):
        vocabulary = getattr(output, 'route_vocabulary', 'route71')
        if self.route_vocabulary is None:
            self.route_vocabulary = vocabulary
            size = route_class_count(self.route_vocabulary)
            self.confusion['route_id'] = torch.zeros(size, size, dtype=torch.long)
        elif self.route_vocabulary != vocabulary:
            raise ValueError('cannot mix Draft route vocabularies in one diagnostic')
        active = target.valid[:, None] & target.queries.valid & output.query_valid
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            if name == 'route_id':
                labels = route_targets(labels, self.route_vocabulary)
            self.add_confusion(self.confusion[name], logits, labels, active & (labels != -100))
        self.add_confusion(self.moment, output.moment_logits, target.moment_tick,
                           active & (target.moment_tick != -100))
        if getattr(output, 'onset_presence_logits', None) is not None:
            presence = target.queries.onset_presence
            valid = target.valid[:, None] & presence.valid & output.onset_presence_valid
            truth = target.onset_presence_labels[valid].long().detach().cpu()
            prediction = (output.onset_presence_logits[valid] >= 0).long().detach().cpu()
            self.onset_presence += torch.bincount(truth*2+prediction, minlength=4).reshape(2, 2)
            self.onset_presence_seen = True
        beats = target.valid[:, None] & output.count_valid
        if getattr(output, 'count_values', None) is not None:
            prediction = output.count_values.float()[beats].detach().cpu()
            truth = target.moment_counts[beats].detach().cpu()
            rounded = prediction.round().clamp_min(0).long()
            error = prediction-truth
            self.regression['squared_error'] += float(error.square().sum())
            self.regression['absolute_error'] += float(error.abs().sum())
            self.regression['correct'] += int((rounded == truth).sum())
            self.regression['count'] += len(truth)
            self.count_truth.update(truth.tolist())
            self.count_prediction.update(rounded.tolist())
            return
        self.add_confusion(self.empty, output.empty_logits, (target.moment_counts > 0).long(), beats)
        self.add_confusion(self.positive, output.positive_count_logits, target.moment_counts - 1,
                           beats & (target.moment_counts > 0))
        composed = torch.where(output.empty_logits.argmax(-1).bool(),
                               output.positive_count_logits.argmax(-1) + 1, 0)
        self.composed[0] += int(((composed == target.moment_counts) & beats).sum())
        self.composed[1] += int(beats.sum())

    @staticmethod
    def accuracy(table):
        correct, count = int(table.diag().sum()), int(table.sum())
        return dict(correct=correct, count=count, accuracy=correct/max(1, count))

    def report(self):
        fields = {}
        for name, table in self.confusion.items():
            support = table.sum(1)
            recall = [None if not int(n) else int(table[i, i])/int(n) for i,n in enumerate(support)]
            least = int(support[support > 0].min()) if (support > 0).any() else 0
            rare = {str(i): recall[i] for i,n in enumerate(support) if least and int(n) == least}
            fields[name] = dict(**self.accuracy(table), confusion=table.tolist(),
                class_support=support.tolist(), class_recall=recall,
                rare_class_rule='least-supported observed classes', rare_class_recall=rare)
        vocabulary = self.route_vocabulary or 'route71'
        fields['route_id']['route71_groups'] = [list(group) for group in route_groups(vocabulary)]
        result = dict(fields=fields, route_vocabulary=vocabulary,
            moment96_conditional=dict(**self.accuracy(self.moment),
                true_distribution=self.moment.sum(1).tolist(), predicted_distribution=self.moment.sum(0).tolist()),
            empty=self.accuracy(self.empty),
            positive_count_gt_nonempty=self.accuracy(self.positive),
            composed_count=dict(correct=self.composed[0], count=self.composed[1],
                                accuracy=self.composed[0]/max(1, self.composed[1])))
        if self.regression['count']:
            row, n = self.regression, self.regression['count']
            result = {key: value for key, value in result.items()
                      if key not in ('empty', 'positive_count_gt_nonempty', 'composed_count')}
            result['count_regression'] = dict(count=n, mse=row['squared_error']/n,
                mae=row['absolute_error']/n, rounded_correct=row['correct'],
                rounded_accuracy=row['correct']/n,
                rounding='nearest integer, clamped only below zero; raw values used for MSE/MAE',
                true_distribution=dict(sorted(self.count_truth.items())),
                predicted_distribution=dict(sorted(self.count_prediction.items())))
        if self.onset_presence_seen:
            tn, fp, fn, tp = self.onset_presence.flatten().tolist()
            result['onset_presence'] = dict(confusion=self.onset_presence.tolist(),
                labels=['empty', 'onset'], count=tn+fp+fn+tp,
                positive_count=fn+tp, negative_count=tn+fp,
                precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
                f1=2*tp/max(1, 2*tp+fp+fn), unit='owner_z_beat', threshold_logit=0.)
        return result


def count_example(output, index):
    if getattr(output, 'count_values', None) is not None:
        raw = output.count_values[index].float()
        return dict(predicted_count_raw=raw.cpu().tolist(),
                    predicted_count=raw.round().clamp_min(0).long().cpu().tolist())
    return dict(predicted_empty=output.empty_logits[index].argmax(-1).cpu().tolist(),
                predicted_positive_count=(output.positive_count_logits[index].argmax(-1)+1).cpu().tolist())


@torch.no_grad()
def validate(model, loader, config, *, final, restored_rng=None):
    mode, rng = model.training, checkpoint.rng_state()
    model.eval()
    metrics, diagnostics, examples = Metrics(), Diagnostics() if final else None, []
    try:
        for raw in epoch_iterator(loader, restored_rng):
            batch = raw.to(next(model.parameters()).device)
            with autocast(next(model.parameters()).device, config):
                terms, (r, p, z) = model(batch)
            metrics.add(terms)
            if final:
                diagnostics.add(r, batch.reconstruction)
                if not examples:
                    for i in range(min(4, len(raw.window_ids))):
                        examples.append(dict(window_id=raw.window_ids[i],
                            predicted_fields={k: v[i].argmax(-1).cpu().tolist() for k,v in r.field_logits.items()},
                            target_fields={k: v[i].cpu().tolist() for k,v in batch.reconstruction.fields.items()},
                            predicted_moment96=r.moment_logits[i].argmax(-1).cpu().tolist(),
                            target_moment96=batch.reconstruction.moment_tick[i].cpu().tolist(),
                            **count_example(r, i),
                            target_count=batch.reconstruction.moment_counts[i].cpu().tolist(),
                            plan12=p.plan_values[i].float().cpu().tolist()))
        result = metrics.report(config)
        if not torch.isfinite(torch.tensor(result['loss'])):
            raise ValueError('nonfinite validation loss')
        if final:
            result['diagnostics'] = dict(**diagnostics.report(), examples=examples)
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
    config = resolve_audio_config(config)
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
        report['reconstruction_query_substeps'] = sum(query_cost(b) for b in batches)
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
        starting_completed_epochs=completed, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        audio_producer=model.audio_producer))
    for epoch in remaining:
        started = time.perf_counter()
        model.train()
        loaders[0].batch_sampler.epoch = epoch
        iterator = epoch_iterator(loaders[0], restored[0])
        restored[0] = None
        epoch_metrics, updates, windows, groups, queries = Metrics(), 0, 0, 0, 0
        while batches := list(islice(iterator, config['pretrain_gradient_accumulation_steps'])):
            metrics, audit = update(model, batches, optimizer, scheduler, config)
            for k, value in metrics.sums.items():
                epoch_metrics.sums[k][0] += value[0]
                epoch_metrics.sums[k][1] += value[1]
            epoch_metrics.presence_positive += metrics.presence_positive
            epoch_metrics.presence_negative += metrics.presence_negative
            updates += 1
            step += 1
            windows += sum(len(b.window_ids) for b in batches)
            groups += sum(len(b.audio_variants) for b in batches)
            queries += sum(query_cost(b) for b in batches)
            print(json.dumps(dict(epoch=epoch, update=updates, updates=steps[epoch-1], step=step,
                                  loss=metrics.report(config)['loss'], lr=optimizer.param_groups[0]['lr'])), flush=True)
        if updates != steps[epoch-1] or groups != len(plans[0].groups) or windows != len(plans[0].windows):
            raise ValueError('incomplete training epoch; checkpoint not saved')
        final = epoch == config['pretrain_stop_after_epochs']
        result = dict(epoch=epoch, step=step, train=epoch_metrics.report(config),
                      seconds=time.perf_counter()-started, audio_groups=groups, windows=windows,
                      reconstruction_query_substeps=queries)
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
