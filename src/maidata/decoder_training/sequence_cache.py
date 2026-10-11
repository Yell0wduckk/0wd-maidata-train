"""Shared v5 compiler: exact model sequence and GT-prefix step supervision.

The v4 representation is an offline source interchange, never a runtime model.
"""
from dataclasses import fields
import torch

from maidata.decoder.sequence import expand_inputs, family, continue_token
from maidata.decoder.batching import pack_tokens
from maidata.chart.types import GENERATED_TYPES
from maidata.time.discrete import TimeQuery, CONTINUE

SCHEMA = 'maidata.decoder-window-tensors.v5'
SEQUENCE_SCHEMA = 'maidata.model-chart-sequence.v1'


def chart_sequence(record):
    """The same full-chart model representation for fresh export and migration."""
    from maidata.chart.codec import token_from_dict
    from maidata.time import TimingMap
    tokens=tuple(token_from_dict(t) for t in record['tokens'])
    timing=TimingMap.from_ir(record['metadata'])
    seq=expand_inputs(tokens,[r['tick']/96 for r in record['token_map']],timing,boundary_index=0)
    batch=pack_tokens([seq['tokens']],[seq['beats']],device='cpu',slide_positions=[seq['slide_beats']])
    return dict(schema=SEQUENCE_SCHEMA,chart_id=record['source']['chart_id'],
        tokens={f.name:getattr(batch,f.name)[0] for f in fields(batch)},
        source_rows=seq['source_rows'],elapsed_ticks=seq['elapsed'])


def write_asset(path,value,root):
    """Accept each new asset once; both producers publish the same format."""
    from .window_cache import tensor_pack
    from maidata.training.checkpoint import digest_file
    payload=tensor_pack(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    saved=torch.load(temporary,map_location='cpu',weights_only=True)
    if saved['layout']!=payload['layout'] or any(not torch.equal(v,saved['buffers'][k]) for k,v in payload['buffers'].items()):
        raise ValueError('new asset storage roundtrip differs')
    temporary.replace(path)
    return dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=digest_file(path))


def step_penalty(group, field, depth):
    """Project complete leaves onto CONT; keep first-attribution category bits.

    A closing CONT is already a zero-penalty leaf. Before it, a category is
    forbidden only when every basic continuation completion carries its bit.
    """
    penalty = group['penalty'][field,depth].clone()
    if bool(group['closes'][field,depth]):
        penalty[CONTINUE] = False
    else:
        allowed = group['allowed'][field,depth+1:].clone()
        allowed[:,CONTINUE] &= group['closes'][field,depth+1:]
        tail = group['penalty'][field,depth+1:]
        penalty[CONTINUE] = ((~allowed[...,None]) | tail).flatten(0,1).all(0)
    return penalty


def expand_window(value, example, timing):
    expanded = expand_inputs(example.inputs,example.beats,timing,boundary_index=example.boundary_index)
    rows, positions, slides = map(list,(expanded['tokens'],expanded['beats'],expanded['slide_beats']))
    row_map = expanded['last_rows']
    history = row_map[example.boundary_index]
    # Terminal is an output only. Earlier CONT choices become real model inputs.
    terminal = value['labels']['groups']['MOMENT']
    terminal_field = next(i for i,p in enumerate(terminal['indices'][:,1].tolist())
                          if p+value['history'] == len(example.inputs)-1)
    steps = int(terminal['target_steps'][terminal_field].sum())
    origin = int(terminal['origins'][terminal_field])
    for depth in range(1,steps):
        elapsed = depth*384
        rows.append(continue_token('MOMENT',timing,origin,elapsed))
        beat = TimeQuery('MOMENT',origin,elapsed).query_beat(timing)
        positions.append(beat); slides.append(beat)
    tokens = pack_tokens([rows],[positions],device='cpu',slide_positions=[slides])
    types = [GENERATED_TYPES.index(family(t)) for t in rows[1:]]+[0]
    groups = {}
    counts = dict(value['labels']['counts'])
    obsolete = {'query_rows','owner_spans','query_beats','origins','elapsed','roles','closes','target_steps'}
    def mapped(indices):
        result = indices.clone()
        result[:,1] = torch.tensor([row_map[int(i)+value['history']]-history for i in indices[:,1]])
        return result
    for key,g in value['labels']['groups'].items():
        if 'target_steps' not in g:
            groups[key] = {k:(mapped(v) if k == 'indices' else v) for k,v in g.items() if k not in obsolete}
            continue
        entries = []
        physical_fields = 0
        for i in range(len(g['indices'])):
            active = g['target_steps'][i].nonzero().flatten().tolist()
            choices = [d for d in active if int(g['allowed'][i,d].sum()) > 1]
            physical_fields += bool(choices)
            base = int(mapped(g['indices'][i:i+1])[0,1])
            for depth in active:
                entries.append(dict(indices=torch.tensor([0,base+depth]), allowed=g['allowed'][i,depth],
                    target=g['target'][i,depth],penalty=step_penalty(g,i,depth),
                    scale=torch.tensor(1/len(choices) if depth in choices else 0.),
                    main=g['main'][i],args=g['args'][i],field_id=torch.tensor(i)))
        counts['physical'] += physical_fields-int((g['scale']>0).sum())
        groups[key] = {k:torch.stack([r[k] for r in entries]) for k in entries[0]}
        groups[key]['has_physical'] = bool(physical_fields)
    flags = {k:(mapped(v) if k == 'indices' else v) for k,v in value['labels']['flags'].items()
             if k != 'query_rows'}
    metadata = dict(value['metadata'],time_representation='causal-token-tick384-continue',
        physical_objective='gt-prefix-step-mean', main_sequence_length=len(rows),
        logical_lengths=[len(example.inputs)-example.boundary_index,example.boundary_index],
        model_lengths=[len(rows)-history,history],boundary_index=history,
        source_rows=list(expanded['source_rows'])+[-1]*(len(rows)-len(expanded['tokens'])),
        elapsed_ticks=list(expanded['elapsed'])+[384*d for d in range(1,steps)])
    metadata.pop('field_query_cost',None)
    return dict(tokens={f.name:getattr(tokens,f.name)[0] for f in fields(tokens)},
        next_types=torch.tensor(types,dtype=torch.long),history=history,
        labels=dict(groups=groups,counts=counts,flags=flags),metadata=metadata)
