"""Independent FP32 numerators/supports; the caller owns logical reduction."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from .route_vocabulary import route_targets
from .route_acceptance import acceptance_nll, validate_route_loss


FIELD_PREFIX = 'field/'
COUNT_TERMS = frozenset(('count', 'empty', 'positive_count'))


def count_supervision_enabled(config):
    # Recipes saved before the switch always supervised COUNT.
    enabled = config.get('reconstruction_count_supervision', True)
    if type(enabled) is not bool:
        raise ValueError('reconstruction_count_supervision must be boolean')
    return enabled


def onset_presence_supervision_enabled(config):
    enabled = config.get('reconstruction_onset_presence_supervision', False)
    if type(enabled) is not bool:
        raise ValueError('reconstruction_onset_presence_supervision must be boolean')
    return enabled


@dataclass
class LossTerm:
    total: Tensor
    count: Tensor
    positive_count: Tensor | None = None
    negative_count: Tensor | None = None

    @property
    def mean(self):
        return self.total / self.count.clamp_min(1)


def prepare_supervision(target, planner_target):
    """Selectors for complete D coverage, including the supervised empty tail.

    Every valid target must have an owning Z, as supplied by the joint collator.
    Public callers with partial Z coverage retain the unprepared mask-aware path.
    """
    active = target.valid[:, None] & target.queries.valid
    beats = target.valid[:, None].expand_as(target.moment_counts)
    masks = {FIELD_PREFIX + name: active & (labels != -100)
             for name, labels in target.fields.items()}
    masks.update(moment96=active & (target.moment_tick != -100), count=beats, empty=beats,
                 positive_count=beats & (target.moment_counts > 0))
    indices = {name: mask.flatten().nonzero().flatten() for name, mask in masks.items()}
    readout = dict(query_indices=active.flatten().nonzero().flatten(),
                   count_indices=beats.flatten().nonzero().flatten())
    if getattr(target.queries, 'onset_presence', None) is not None:
        presence = target.valid[:, None] & target.queries.onset_presence.valid
        indices['onset_presence'] = presence.flatten().nonzero().flatten()
        readout['onset_presence_indices'] = indices['onset_presence']
    return dict(readout=readout,
                indices=indices, counts={name: len(rows) for name, rows in indices.items()},
                planner_indices=planner_target.valid.flatten().nonzero().flatten(),
                planner_count=int(planner_target.valid.sum()))


def _categorical(logits, labels, active, smoothing, indices=None, *, acceptance=False):
    """An empty field stays differentiably zero without inventing a label."""
    if indices is not None:
        selected = logits.float().flatten(0, 1).index_select(0, indices)
        truth = labels.flatten().index_select(0, indices)
        count = labels.new_tensor(len(indices))
    else:
        selected, truth = logits.float()[active], labels[active]
        count = active.sum()
    if not len(truth):
        total = logits.float().sum() * 0
    elif acceptance:
        total = acceptance_nll(selected, truth).sum()
    else:
        total = F.cross_entropy(selected, truth, label_smoothing=smoothing, reduction='sum')
    return LossTerm(total, count)


def reconstruction_loss(output, target, *, label_smoothing, prepared=None, count_supervision=True,
                        route_loss='ce', onset_presence_supervision=False):
    if type(count_supervision) is not bool:
        raise ValueError('count_supervision must be boolean')
    if not 0 <= label_smoothing <= 1:
        raise ValueError('invalid reconstruction label smoothing')
    vocabulary = getattr(output, 'route_vocabulary', 'route71')
    validate_route_loss(route_loss, vocabulary)
    device = output.moment_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        query_valid = target.valid[:, None] & target.queries.valid & output.query_valid
        terms = {}
        selected = lambda name: None if prepared is None else prepared['indices'][name]
        for name, logits in output.field_logits.items():
            labels = target.fields[name]
            if name == 'route_id':
                labels = route_targets(labels, vocabulary)
            terms[FIELD_PREFIX + name] = _categorical(
                logits, labels, query_valid & (labels != -100), label_smoothing,
                selected(FIELD_PREFIX + name), acceptance=name == 'route_id' and route_loss == 'acceptance37')
        terms['moment96'] = _categorical(output.moment_logits, target.moment_tick,
                                        query_valid & (target.moment_tick != -100), 0., selected('moment96'))
        if onset_presence_supervision:
            if (output.onset_presence_logits is None or output.onset_presence_valid is None
                    or target.onset_presence_labels is None or target.queries.onset_presence is None):
                raise ValueError('onset presence loss requires independent logits, labels and geometry')
            presence_valid = (target.valid[:, None] & target.queries.onset_presence.valid
                              & output.onset_presence_valid)
            # Prepared selectors do not override a genuinely missing owner/mask.
            index = selected('onset_presence')
            if index is None:
                prediction = output.onset_presence_logits.float()[presence_valid]
                truth = target.onset_presence_labels.float()[presence_valid]
            else:
                enabled = presence_valid.flatten().index_select(0, index)
                prediction = output.onset_presence_logits.float().flatten().index_select(0, index)
                truth = target.onset_presence_labels.float().flatten().index_select(0, index)
            weights = torch.ones_like(truth, dtype=torch.bool) if index is None else enabled
            total = (F.binary_cross_entropy_with_logits(prediction, truth, reduction='none')*weights).sum()
            positive = ((truth > 0) & weights).sum()
            count = weights.sum()
            terms['onset_presence'] = LossTerm(total, count, positive, count-positive)
        if not count_supervision:
            return terms
        beat_valid = target.valid[:, None] & output.count_valid
        if getattr(output, 'count_values', None) is not None:
            # Linear raw count, no rounding/clipping/sigmoid in the training graph.
            # Empty beats and the complete empty tail share the same denominator.
            index = selected('count')
            if index is None:
                prediction = output.count_values.float()[beat_valid]
                truth = target.moment_counts.float()[beat_valid]
                count = beat_valid.sum()
            else:
                prediction = output.count_values.float().flatten().index_select(0, index)
                truth = target.moment_counts.float().flatten().index_select(0, index)
                count = target.moment_counts.new_tensor(len(index))
            terms['count'] = LossTerm((prediction-truth).square().sum(), count)
            return terms
        terms['empty'] = _categorical(output.empty_logits, (target.moment_counts > 0).long(),
                                      beat_valid, label_smoothing, selected('empty'))
        terms['positive_count'] = _categorical(output.positive_count_logits,
            target.moment_counts - 1, beat_valid & (target.moment_counts > 0), label_smoothing,
            selected('positive_count'))
    return terms


def planner_loss(output, target, *, prepared=None):
    valid = target.valid & output.valid
    with torch.autocast(device_type=output.plan_values.device.type, enabled=False):
        if prepared is None:
            predicted = output.continuous_logits.float()[valid].sigmoid()
            truth = target.continuous.float()[valid]
            category_logits, category = output.config_logits.float()[valid], target.category[valid]
            count = valid.sum()
        else:
            index = prepared['planner_indices']
            predicted = output.continuous_logits.float().reshape(-1, 8).index_select(0, index).sigmoid()
            truth = target.continuous.float().reshape(-1, 8).index_select(0, index)
            category_logits = output.config_logits.float().reshape(-1, 4).index_select(0, index)
            category = target.category.flatten().index_select(0, index)
            selected_valid = valid.flatten().index_select(0, index)
            count = selected_valid.sum()
        mse_rows = (predicted - truth).square().mean(-1)
        ce_rows = (F.cross_entropy(category_logits, category, reduction='none') if len(category_logits)
                   else category_logits.sum(-1) * 0)
        if prepared is not None:
            mse_rows = mse_rows * selected_valid
            ce_rows = ce_rows * selected_valid
        mse, ce = mse_rows.sum(), ce_rows.sum()
    return {'planner_continuous': LossTerm(mse, count),
            'planner_category': LossTerm(ce, count)}
