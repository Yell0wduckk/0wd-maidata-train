"""Joint logical-update reductions and one optimizer step for E/D/R/P/M."""
from dataclasses import replace
from contextlib import nullcontext

import torch

from maidata.draft.losses import FIELD_PREFIX, COUNT_TERMS, count_supervision_enabled
from maidata.training.pretrain import weighted_loss, autocast
from maidata.training.runner import supports
from maidata.decoder_training.runner import Metrics as DecoderMetrics, chunk_loss
from maidata.decoder_training.supervision import KEYS, objective
from .shared_gradients import SharedJointGradients


def auxiliary_objective(terms, counts, config):
    reconstruction = {k: v for k, v in terms.items() if not k.startswith('planner_')}
    planner = {k: v for k, v in terms.items() if k.startswith('planner_')}
    return (weighted_loss(reconstruction, config, denominators=counts) * config['joint_train_reconstruction_weight']
            + weighted_loss(planner, config, denominators=counts) * config['joint_train_planner_weight'])


class AuxiliaryMetrics:
    """Keep detached totals on their device until the reporting boundary."""
    def __init__(self):
        self.names = None
        self.totals = None
        self.counts = {}

    def add(self, terms, counts=None):
        names = tuple(terms)
        if self.names is None:
            self.names = names
            self.counts = dict.fromkeys(names, 0)
        values = torch.stack([terms[k].total.detach().double() for k in self.names])
        self.totals = values if self.totals is None else self.totals + values
        for key in self.names:
            self.counts[key] += int(terms[key].count) if counts is None else counts[key]

    def merge(self, other):
        if other.names is None:
            return
        if self.names is None:
            self.names, self.counts = other.names, dict.fromkeys(other.names, 0)
        self.totals = other.totals if self.totals is None else self.totals + other.totals
        for key in self.names:
            self.counts[key] += other.counts[key]

    def report(self, config):
        if self.names is None:
            return dict(loss=0., reconstruction=0., planner=0., terms={}, classification=dict(mean=0., active_fields=0))
        terms = {key: dict(total=total, count=self.counts[key], mean=total/max(1, self.counts[key]))
                 for key, total in zip(self.names, self.totals.cpu().tolist())}
        fields = [value['mean'] for key, value in terms.items()
                  if key.startswith(FIELD_PREFIX) and value['count']]
        classification = sum(fields)/max(1, len(fields))
        reconstruction = classification * config['reconstruction_classification_weight']
        reconstruction += sum(terms[key]['mean'] * config[weight] for key, weight in (
            ('moment96', 'reconstruction_moment_weight'), ('count', 'reconstruction_count_weight'),
            ('empty', 'reconstruction_empty_weight'), ('positive_count', 'reconstruction_positive_count_weight'))
            if key in terms and (key not in COUNT_TERMS or count_supervision_enabled(config)))
        planner = sum(terms[key]['mean'] * config[key+'_weight']
                      for key in ('planner_continuous', 'planner_category'))
        return dict(loss=reconstruction+planner, reconstruction=reconstruction, planner=planner,
                    classification=dict(mean=classification, active_fields=len(fields)), terms=terms)


class JointMetrics:
    def __init__(self):
        self.auxiliary = AuxiliaryMetrics()
        self.decoder = DecoderMetrics()

    def add_auxiliary(self, terms, counts=None):
        self.auxiliary.add(terms, counts)

    add_aux = add_auxiliary

    def add_decoder(self, terms):
        self.decoder.add(terms)

    def merge(self, other):
        self.auxiliary.merge(other.auxiliary)
        self.decoder.merge(other.decoder)

    def report(self, config):
        auxiliary, decoder = self.auxiliary.report(config), self.decoder.report(config)
        weighted = dict(decoder=decoder['loss'] * config['joint_train_decoder_weight'],
            reconstruction=auxiliary['reconstruction'] * config['joint_train_reconstruction_weight'],
            planner=auxiliary['planner'] * config['joint_train_planner_weight'])
        terms = auxiliary['terms']
        mean = lambda key: terms.get(key, {}).get('mean', 0.)
        r_weight = config['joint_train_reconstruction_weight']
        parts = dict(fields=auxiliary['classification']['mean'] * config['reconstruction_classification_weight'] * r_weight,
            moment=mean('moment96') * config['reconstruction_moment_weight'] * r_weight,
            count=(mean('count') * config['reconstruction_count_weight']
                   + mean('empty') * config['reconstruction_empty_weight']
                   + mean('positive_count') * config['reconstruction_positive_count_weight']) * r_weight
                   if count_supervision_enabled(config) else 0.)
        losses = dict(main_decoder=weighted['decoder'],
            drafter=weighted['reconstruction'] + weighted['planner'],
            drafter_fields=parts['fields'], drafter_moment=parts['moment'], drafter_count=parts['count'],
            planner=weighted['planner'])
        return dict(loss=sum(weighted.values()), losses=losses,
                    weighted=weighted, auxiliary=auxiliary, decoder=decoder)


def update(model, batches, opt, sched, config, geometry, *, audit=False, mode='joint', recorder=None, fixed_z=None):
    """Common supports across all physical batches and all M chunks."""
    if mode not in ('joint', 'detach_z', 'auxiliary', 'decoder'):
        raise ValueError('unsupported joint update comparison mode')
    model.train()
    aux_supports = [supports(batch) for batch in batches]
    counts = {key: sum(row[key] for row in aux_supports) for key in aux_supports[0]}
    decoder_counts = {key: sum(chunk.labels.counts[key] for batch in batches for chunk in batch.chunks)
                      for key in KEYS}
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in model.named_children()} if audit else {}
    device = next(model.parameters()).device
    section = lambda name: nullcontext() if recorder is None else recorder.section(name)
    opt.zero_grad(set_to_none=True)
    metrics, costs, finite = JointMetrics(), [], None
    for batch_index, (raw, batch_supports) in enumerate(zip(batches, aux_supports)):
        with section('H2D'):
            batch = raw.to(device)
        with section('shared_forward'), autocast(device, config):
            terms, conditions, outputs = model.forward_shared(batch,
                auxiliary=mode != 'decoder', draft_grad=mode != 'decoder', recorder=recorder,
                fixed_z=None if fixed_z is None else fixed_z[batch_index])
            auxiliary = (conditions.audio.values.sum()*0 if mode == 'decoder'
                         else auxiliary_objective(terms, counts, config))
        finite_auxiliary = torch.isfinite(auxiliary)
        finite = finite_auxiliary if finite is None else finite & finite_auxiliary
        if mode != 'decoder':
            metrics.add_auxiliary(terms, batch_supports)
        del terms, outputs
        if mode == 'auxiliary':
            with section('shared_backward'):
                auxiliary.backward()
            del batch, conditions, auxiliary
            continue
        if mode in ('detach_z', 'decoder'):
            conditions = replace(conditions, decoder_z=replace(conditions.decoder_z,
                values=conditions.decoder_z.values.detach()))
        with section('audio_projection'), autocast(device, config):
            bank = conditions.audio
            audio_kv = model.decoder.prepare_audio(bank, indices=conditions.audio_indices)
        bridge = SharedJointGradients(audio_kv, conditions.decoder_z) if len(batch.chunks) > 1 else None
        chunk_conditions = conditions if bridge is None else replace(conditions, decoder_z=bridge.z)
        chunk_audio = audio_kv if bridge is None else bridge.memory
        for cpu_chunk in batch.chunks:
            with section('H2D'):
                chunk = cpu_chunk.to(device)
            with section('M_forward_loss'), autocast(device, config):
                decoder_terms, output = chunk_loss(model, chunk_conditions, chunk_audio, chunk, config, geometry)
                value = objective(decoder_terms, decoder_counts, config) * config['joint_train_decoder_weight']
                if bridge is None:
                    value = value + auxiliary
            finite = finite & torch.isfinite(value)
            with section('direct_backward' if bridge is None else 'M_backward'):
                value.backward()
                if bridge is not None:
                    bridge.collect()
            metrics.add_decoder(decoder_terms)
            costs.append(chunk.cost)
            del decoder_terms, output, value, chunk
        if bridge is not None:
            with section('shared_backward'):
                bridge.backward(auxiliary)
        elif not batch.chunks:
            with section('shared_backward'):
                auxiliary.backward()
        del batch, conditions, bank, audio_kv, chunk_conditions, chunk_audio, bridge, auxiliary
    if finite is not None and not bool(finite):
        raise ValueError('nonfinite joint training loss')
    with section('clip'):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['pretrain_gradient_clip'], error_if_nonfinite=True)
    gradients = {name: sum(float(p.grad.float().square().sum()) for p in module.parameters()
                          if p.grad is not None)**.5 for name, module in model.named_children()} if audit else {}
    with section('optimizer'):
        opt.step()
        sched.step()
    changed = {name: sum(not torch.equal(a, b) for a, b in zip(before[name], module.parameters()))
               for name, module in model.named_children()} if audit else {}
    return metrics, dict(gradient_norm=float(norm), gradients=gradients, changed_parameter_tensors=changed,
        chunks=len(costs), padded_token_costs=costs, batch_costs=[batch.budget_cost for batch in batches])
