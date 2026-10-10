"""Conditional readouts over one immutable decoder hidden row.

Targets are supplied here, after all causal attention, never to the decoder.
Sampling calls one stage at a time with discrete decisions already made.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from maimai_dataset.modifiers import HEADS, LEGAL

from maimai_baseline.row_field_contract import IDENTITY, parameter_count


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.mlp = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))

    def forward(self, h, condition):
        x = h + condition
        return x + self.mlp(self.norm(x))


class RowFields(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.embeddings = nn.ModuleDict({n: nn.Embedding(len(vocabulary.fields[n].values), 256)
                                        for n in ('event', 'kind', 'pad', 'start_pad')})
        self.flags = nn.Linear(len(HEADS), 256, bias=False)
        self.route = nn.Linear(70, 256, bias=False)
        self.adapters = nn.ModuleDict({s: Adapter() for s in IDENTITY['stages']})

    def condition(self, values, stage, prototypes):
        event = values['event']
        out = self.embeddings['event'](event.clamp_min(0))
        if stage != 'kind':
            kind = values['kind']
            out = out + self.embeddings['kind'](kind.clamp_min(0)) * kind.ge(0)[..., None]
        if stage in ('route', 'duration'):
            for name in ('pad', 'start_pad'):
                ids = values[name]
                out = out + self.embeddings[name](ids.clamp_min(0)) * ids.ge(0)[..., None]
            flags = torch.stack([values['modifier_' + n].clamp_min(0) for n in HEADS], -1)
            out = out + self.flags(flags.to(out.dtype))
        if stage == 'duration':
            ids = values['route']
            proto = prototypes[(ids + 1).clamp_min(0)]
            out = out + self.route(proto) * ids.ge(0)[..., None]
        return out


def install(model, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) ^ 0x524F5746)
        model.row_fields = RowFields(model.vocabulary)


def owner_starts(batch, vocabulary):
    events = batch['input_ids']['event']
    positions = torch.arange(events.shape[1], device=events.device)[None].expand_as(events)
    owner = torch.where(events.eq(vocabulary.event_id('NOTE')), positions, -1).cummax(1).values
    reset = torch.where(events.eq(vocabulary.event_id('MOMENT')) | events.eq(vocabulary.event_id('TARGET_START')),
                        positions, -1).cummax(1).values
    start = batch['input_ids']['start_pad'].gather(1, owner.clamp_min(0))
    return start.masked_fill(owner.lt(0) | owner.lt(reset), -100)


def teacher_values(batch, vocabulary):
    values = dict(batch['target_ids'])
    branch = values['event'].eq(vocabulary.event_id('SLIDE_BRANCH'))
    values['start_pad'] = torch.where(branch, owner_starts(batch, vocabulary), values['start_pad'])
    return values


def read_stage(model, decoded, values, stage, *, row_valid=None):
    vocabulary = model.vocabulary
    event = values['event']
    note = event.eq(vocabulary.event_id('NOTE'))
    branch = event.eq(vocabulary.event_id('SLIDE_BRANCH'))
    valid = torch.ones_like(note) if row_valid is None else row_valid
    note, branch = note & valid, branch & valid
    kind = values.get('kind', torch.full_like(event, -100))
    kinds = {k: note & kind.eq(vocabulary.fields['kind'].encode(k)) for k in LEGAL}
    masks = {}
    if stage == 'kind':
        masks['kind'] = note
    elif stage == 'position_flags':
        masks['pad'] = kinds['tap'] | kinds['touch'] | kinds['hold'] | kinds['touch_hold']
        masks['start_pad'] = kinds['slide'] | kinds['wifi']
        for n in HEADS:
            mask = branch.clone() if n == 'track_break' else torch.zeros_like(note)
            for k, legal in LEGAL.items():
                if n in legal:
                    mask |= kinds[k]
            masks['modifier_' + n] = mask
    elif stage == 'route':
        masks['route'] = kinds['slide'] | kinds['wifi'] | branch
    elif stage == 'duration':
        masks['hold_duration_tick'] = kinds['hold'] | kinds['touch_hold']
        masks['branch_duration_tick'] = kinds['slide'] | kinds['wifi'] | branch
    else:
        raise ValueError('unknown field stage: ' + stage)
    active = torch.stack(list(masks.values())).any(0)
    index = active.flatten().nonzero().flatten()
    h = decoded.flatten(0, 1)
    outputs = {}
    # No adapter/head call on inapplicable rows: no gradients or weight decay.
    if index.numel():
        selected = {k: v.flatten().index_select(0, index) for k,v in values.items()}
        condition = model.row_fields.condition(selected, stage, model.route_embedding.prototypes)
        hidden = model.row_fields.adapters[stage](h.index_select(0, index), condition)
    for field, mask in masks.items():
        size = 71 if field == 'route' else model.config.time_components * 3 if 'duration' in field else model.field_heads[field].out_features
        result = h.new_zeros((len(h), size))
        local = mask.flatten().index_select(0, index).nonzero().flatten()
        if local.numel():
            x = hidden.index_select(0, local)
            if field == 'route':
                value = model.route_head(x, model.route_embedding.prototypes[1:])
            elif 'duration' in field:
                head = model.time_heads[field]
                with torch.autocast(device_type=x.device.type, enabled=False):
                    value = F.linear(x.float(), head.weight.float(), head.bias.float())
            else:
                value = model.field_heads[field](x)
            result = result.to(value.dtype).index_copy(0, index.index_select(0, local), value)
        outputs[field] = result.reshape(*event.shape, *(((model.config.time_components, 3)) if 'duration' in field else (size,)))
    return outputs, masks


def supervised_readout(model, outputs, batch):
    result = dict(outputs, field_logits=dict(outputs['field_logits']), time_outputs=dict(outputs['time_outputs']),
                  field_valid=dict(outputs['field_valid']), time_valid=dict(outputs['time_valid']))
    values = teacher_values(batch, model.vocabulary)
    for stage in IDENTITY['stages']:
        logits, masks = read_stage(model, outputs['decoded'], values, stage, row_valid=~batch['row_padding_mask'])
        for field, value in logits.items():
            if field == 'route':
                result['route_logits'], result['route_valid'] = value, masks[field]
            elif 'duration' in field:
                result['time_outputs'][field], result['time_valid'][field] = value, masks[field]
            else:
                result['field_logits'][field], result['field_valid'][field] = value, masks[field]
    result['row_field_identity'] = dict(IDENTITY)
    return result


def sample_readout(model, outputs, row, stage, *, owner_start=None, route=None):
    h = outputs['decoded'][:, -1:]
    values = {name: torch.full(h.shape[:2], -100, device=h.device, dtype=torch.long)
              for name in ('event','kind','pad','start_pad','route', *('modifier_' + n for n in HEADS))}
    for name in ('event', 'kind', 'pad', 'start_pad'):
        if name in row:
            values[name].fill_(model.vocabulary.fields[name].encode(row[name]))
    if row['event'] == 'SLIDE_BRANCH' and owner_start is not None:
        values['start_pad'].fill_(model.vocabulary.fields['start_pad'].encode(owner_start))
    from maimai_dataset.modifiers import row_bits
    # A generated kind is already normalized. Duration is a later field and
    # must neither be required here nor change the preceding flag conditions.
    bits = row_bits(row, normalize=False)
    for n in HEADS:
        role = 'head_' + n if row.get('kind') in ('slide', 'wifi') and n in ('ex','break') else n
        values['modifier_' + n].fill_(bits.get(role, 0))
    if route is not None:
        values['route'].fill_(route)
    logits, _ = read_stage(model, h, values, stage)
    return logits
