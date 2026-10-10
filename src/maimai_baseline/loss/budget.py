"""Fixed per-kind field budgets and support-aware diagnostic statistics."""
import torch
from .common import GROUP_NOTE_BUDGET, NOTE_FIELD_BUDGET, BRANCH_FIELD_BUDGET


def note_semantics(losses, active, targets, vocabulary, *, raw_losses=None, collect_metrics=True):
    """Return NOTE/branch row objectives, without redistributing missing fields.

    ``position`` and ``route`` are prepared once by L3. Each diagnostic term is
    (raw values, weighted values, applicable mask, coefficient within N).
    """
    event, kind = targets['event'], targets['kind']
    note = event.eq(vocabulary.event_id('NOTE'))
    branch = event.eq(vocabulary.event_id('SLIDE_BRANCH'))
    zero = torch.zeros_like(event, dtype=torch.float32)
    total, raw_total, terms = zero, zero, {}
    raw_losses = losses if raw_losses is None else raw_losses

    def field(name, selector):
        mask = selector & active.get(name, torch.zeros_like(selector))
        return (raw_losses.get(name, zero).detach().masked_fill(~mask, 0) if collect_metrics else zero,
                losses.get(name, zero).masked_fill(~mask, 0), mask)

    def add(name, term, coefficient):
        nonlocal total, raw_total
        raw, weighted, mask = term
        total = total + coefficient * weighted
        if collect_metrics:
            raw_total = raw_total + coefficient * raw
            terms[name] = (raw, weighted, mask, coefficient)

    for name, budget in NOTE_FIELD_BUDGET.items():
        select = note & kind.eq(vocabulary.fields['kind'].encode(name))
        add(name + ':kind', field('kind', select), GROUP_NOTE_BUDGET['kind'])
        for group, weight in budget.items():
            key = ('branch_duration_tick' if name in ('slide', 'wifi') else 'hold_duration_tick') if group == 'duration' else group
            term = field(key, select)
            add(name + ':' + group, term, GROUP_NOTE_BUDGET['fields'] * weight)
    for group, weight in BRANCH_FIELD_BUDGET.items():
        key = {'flags': 'modifier_track_break', 'duration': 'branch_duration_tick'}.get(group, group)
        add('branch:' + ('modifier_track_break' if group == 'flags' else group), field(key, branch), weight)
    return total, raw_total, note | branch, terms


def budget_statistics(raw, values, mask, row_weights, denominator_mask, coefficient):
    """Detached additive statistics; contribution uses the parent denominator."""
    weight = row_weights * mask
    denominator = (row_weights * denominator_mask).sum()
    weighted_sum = (values.detach().masked_fill(~mask, 0) * weight).sum()
    return dict(raw_loss_sum=raw.detach().masked_fill(~mask, 0).sum(), count=mask.sum(),
                loss_sum=weighted_sum, weight_sum=weight.sum(),
                contribution_sum=coefficient * weighted_sum, denominator_weight_sum=denominator)


def reject_legacy_sft():
    from maimai_baseline.contract import L3_LOSS_ID
    raise ValueError(f'independent SFT is not adapted to {L3_LOSS_ID}')
