"""Read-only, tensorized supervision windows. No rules or chart parsing at runtime."""
from dataclasses import fields
import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from maidata.decoder.types import TokenBatch
from .supervision import Prepared, KEYS

SCHEMA = 'maidata.decoder-window-tensors.v3'
CONFIG_KEYS = ('target_beats', 'audio_after_beats', 'chart_history_beats')


def contract(config, binding):
    # Source assets retain their producer identities; path relocation is harmless.
    return dict(config={k: config[k] for k in CONFIG_KEYS},
                time_representation='relative-tick384-continue',
                duration_domain='local-audio-window-intersect-song-end',
                inputs={k: binding[k] for k in ('plans', 'physical', 'geometry', 'slide_data',
                                               'pretrain_chart_cache')})


def tensor_pack(value):
    """Plain container metadata plus one contiguous tensor per dtype; weights_only safe."""
    tensors, sizes = {}, {}
    def describe(v):
        if isinstance(v, torch.Tensor):
            offset = sizes.get(v.dtype, 0)
            sizes[v.dtype] = offset + v.numel()
            tensors.setdefault(v.dtype, []).append(v.contiguous().flatten())
            return ('tensor', v.dtype, offset, v.numel(), tuple(v.shape))
        if isinstance(v, dict):
            return ('dict', {k: describe(x) for k, x in v.items()})
        if isinstance(v, (tuple, list)):
            return ('tuple' if isinstance(v, tuple) else 'list', [describe(x) for x in v])
        return ('value', v)
    layout = describe(value)
    return dict(layout=layout, buffers={k: torch.cat(v) for k, v in tensors.items()})


def tensor_unpack(payload):
    def restore(spec):
        kind = spec[0]
        if kind == 'tensor':
            _, dtype, offset, count, shape = spec
            return payload['buffers'][dtype].narrow(0, offset, count).view(shape)
        if kind == 'dict':
            return {k: restore(v) for k, v in spec[1].items()}
        if kind == 'tuple':
            return tuple(restore(v) for v in spec[1])
        if kind == 'list':
            return [restore(v) for v in spec[1]]
        return spec[1]
    return restore(payload['layout'])


class WindowCache:
    def __init__(self, plans, config, binding):
        self.root = Path(config['decoder_train_window_cache'])
        path = self.root / 'manifest.json'
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('schema') != SCHEMA:
            raise ValueError('Decoder requires exact tick384 query windows (v3); recompile from original-v2 fields/supports')
        self.windows = meta['windows']
        for plan in plans:
            plan._lengths = {w['window_id']: tuple(self.windows[w['window_id']]['lengths']) for w in plan.windows}
            plan._query_costs = {w['window_id']: self.windows[w['window_id']].get('field_query_cost')
                                 for w in plan.windows}
        self.binding = str(path)

    def read(self, identity):
        # New tensors were accepted once by the exporter. No payload validation,
        # decompression, grammar replay or Physical query on the training path.
        return tensor_unpack(torch.load(self.root / self.windows[identity]['path'],
                                        map_location='cpu', weights_only=True))


class UnsupportedDurationBoundary(ValueError):
    """An exact GT duration needs the explicitly deferred beyond-window mechanism."""


def select_window(value, dropped):
    unsupported = value.get('metadata', {}).get('unsupported_duration', ())
    if unsupported:
        meta = value['metadata']
        raise UnsupportedDurationBoundary(
            f"{meta.get('window_id')}: GT duration exceeds local audio window; "
            f"exact source/support retained, training labels deferred: {unsupported}")
    history = value['history']
    start = history if dropped else 0
    tokens = {k: v[start:] for k, v in value['tokens'].items()}
    if dropped:
        tokens['time'] = tokens['time'].clone()
        tokens['time'][0] = 0
    return dict(tokens=tokens, next_types=value['next_types'][start:], labels=value['labels'],
                offset=0 if dropped else history, length=len(tokens['types']), history_dropped=dropped,
                field_query_cost=value.get('metadata', {}).get('field_query_cost'))


def collate_prepared(rows):
    """Pad tensors and shift row indices only; all decisions were prepared offline."""
    tokens = TokenBatch(**{f.name: pad_sequence([r['tokens'][f.name] for r in rows], batch_first=True,
        padding_value=-1 if f.name in ('types','kind','pad','flags','route','track_break') else 0)
        for f in fields(TokenBatch)})
    types = pad_sequence([r['next_types'] for r in rows], batch_first=True)
    groups, flags, args, counts, flag_offset = {}, [], [], dict.fromkeys(KEYS, 0), 0
    for b, row in enumerate(rows):
        labels, offset = row['labels'], row['offset']
        def indices(values):
            result = values.clone()
            result[:, 0] = b
            result[:, 1] += offset
            return result
        flags.append(indices(labels['flags']['indices']))
        args.append(labels['flags']['args'])
        for key, value in labels['counts'].items():
            counts[key] += value
        for key, group in labels['groups'].items():
            g = dict(group, indices=indices(group['indices']))
            if 'flag_rows' in g:
                g['flag_rows'] = g['flag_rows'] + flag_offset
            groups.setdefault(key, []).append(g)
        flag_offset += len(labels['flags']['indices'])
    packed = {}
    for key, parts in groups.items():
        width = max(p['allowed'].shape[1] for p in parts)
        g = {}
        for name in parts[0]:
            if name in ('owner_spans','query_rows'):
                continue  # These CPU layouts are rebuilt after row concatenation.
            if name == 'has_physical':
                g[name] = any(p[name] for p in parts)
                continue
            values = [p[name] for p in parts]
            if name in ('allowed','target','lower','upper','penalty','query_beats','elapsed','closes','target_steps'):
                # Time groups carry [field, depth, 385, (category)]. Content
                # groups retain [field, class]. Only the second axis is padded.
                values = [pad(v, (0, 0)*(v.ndim-2)+(0, width-v.shape[1]),
                              value=1 if name == 'upper' else 0) if v.ndim >= 2 else v for v in values]
            g[name] = torch.cat(values)
        owners = g['indices'][:,0].tolist()
        spans = []
        for i, owner in enumerate(owners):
            if spans and spans[-1][0] == owner:
                spans[-1] = [owner, spans[-1][1], i+1]
            else:
                spans.append([owner, i, i+1])
        g['owner_spans'] = spans
        packed[key] = g
    lookup = {}
    for key in ('kind','route'):
        if key in packed:
            for index in packed[key]['indices'].tolist():
                lookup[tuple(index)] = len(lookup)
    for key, group in packed.items():
        if key != 'type' and 'target_steps' not in group:
            group['query_rows'] = torch.tensor([lookup[tuple(index)] for index in group['indices'].tolist()],
                                               dtype=torch.long)
    flag_indices = torch.cat(flags)
    flag_query_rows = torch.tensor([lookup[tuple(index)] for index in flag_indices.tolist()],dtype=torch.long)
    return tokens, types, Prepared(packed, counts, dict(indices=flag_indices, args=torch.cat(args),
                                                       query_rows=flag_query_rows))
