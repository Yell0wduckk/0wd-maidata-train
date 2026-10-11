"""Valid-row projections and bucketed SDPA; no state or learned parameters."""
import math
import torch
from torch.nn import functional as F
from maidata.nn.position import apply_rope
from .types import KV
from .condition_attention import attend as attend_condition


def restore_rows(value, valid, indices):
    return value.new_zeros((valid.numel(), *value.shape[1:])).index_copy(0, indices, value).reshape(
        *valid.shape, *value.shape[1:])


def rotate(module, value, beats):
    # Compact [N,H,D] rows share one positional calculation, retaining absolute beats.
    return apply_rope(value.transpose(0, 1)[None], beats[None], module.base, module.scale)[0].transpose(0, 1)


def self_attention(module, x, beats, valid, layout):
    layers = (module.query, module.key, module.value)
    parts = F.linear(x, torch.cat([m.weight for m in layers]), torch.cat([m.bias for m in layers]))
    raw, key, value = parts.reshape(len(x), 3, module.heads, module.dim//module.heads).unbind(1)
    query, key = rotate(module, raw, beats), rotate(module, key, beats)
    sizes = [count*width for count, width, _ in layout['groups']]
    queries, keys, values = [t.index_select(0, layout['gather']).split(sizes) for t in (query, key, value)]
    pieces = []
    for q, k, v, (count, width, _) in zip(queries, keys, values, layout['groups']):
        q, k, v = [t.reshape(count, width, module.heads, module.dim//module.heads).transpose(1, 2)
                   for t in (q, k, v)]
        # Only end padding exists here. Real queries cannot see padded future keys.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=module.dropout if module.training else 0.)
        pieces.append(y.transpose(1, 2).flatten(0, 1))
    selected = torch.cat(pieces).index_select(0, layout['restore']) if pieces else query
    result = module.output(selected.reshape(len(x), module.dim))
    cache = KV(*(restore_rows(t, valid, layout['indices']).transpose(1, 2) for t in (key, value)), valid)
    return result, cache


def condition_multiples(valid, layout, masks):
    """Bind the fast path to the actual projected mask, once per memory.

    A public layout can outlive a Z switch or a changed condition mask. Its
    bucket width alone therefore cannot certify two valid keys. All layers
    share one validity tensor, so cache this small decision by identity and
    mutation version rather than synchronizing for every layer/field query.
    Inference tensors lack a version counter; keep their exact fallback.
    """
    if 'key_gather' not in layout or not masks:
        return (False,)*len(masks)
    try:
        version = valid._version
    except RuntimeError:
        return (False,)*len(masks)
    cached = layout.get('_condition_multiples')
    if cached is not None and cached[0] is valid and cached[1] == version:
        return cached[2]
    counts = torch.cat([mask.reshape(count, -1).sum(-1)
                        for mask, (count, _, _) in zip(masks, layout['groups'])]).cpu().tolist()
    flags, offset = [], 0
    for count, _, _ in layout['groups']:
        flags.append(all(value > 1 for value in counts[offset:offset+count]))
        offset += count
    flags = tuple(flags)
    layout['_condition_multiples'] = (valid, version, flags)
    return flags


def condition_attention(module, x, beats, valid, kv, plan_index, layout):
    raw = module.query(x).reshape(len(x), module.heads, module.dim//module.heads)
    query = torch.cat((rotate(module, raw, beats), raw), -1)
    z, plan = kv.key[:, :, :plan_index], kv.key[:, :, plan_index:]
    key = torch.cat((torch.cat((z, torch.zeros_like(z)), -1),
                     torch.cat((torch.zeros_like(plan), plan), -1)), 2)
    value = F.pad(kv.value, (0, module.dim//module.heads))
    groups = layout['groups']
    queries = query.index_select(0, layout['gather']).split([c*w for c, w, _ in groups])
    if 'key_gather' in layout:
        sizes = [c*k for c, _, k in groups]
        keys, values = [t.transpose(1, 2).flatten(0, 1).index_select(0, layout['key_gather']).split(sizes)
                        for t in (key, value)]
        masks = (layout['key_valid'] & kv.valid.flatten().index_select(0, layout['key_gather'])).split(sizes)
    else:
        # Public callers may provide only a query layout. Preserve arbitrary key masks.
        sizes = [c for c, _, _ in groups]
        keys, values = [t.transpose(1, 2).index_select(0, layout['windows']).split(sizes) for t in (key, value)]
        masks = kv.valid.index_select(0, layout['windows']).split(sizes)
    multiples = condition_multiples(kv.valid, layout, masks)
    pieces = []
    for q, k, v, mask, (count, width, key_width), all_multiple in zip(queries, keys, values, masks, groups, multiples):
        key_width = kv.valid.shape[1] if key_width is None else key_width
        q = q.reshape(count, width, module.heads, 2*module.dim//module.heads).transpose(1, 2)
        k, v = [t.reshape(count, key_width, module.heads, 2*module.dim//module.heads).transpose(1, 2) for t in (k, v)]
        y = attend_condition(q, k, v, mask.reshape(count, key_width),
            scale=1/math.sqrt(module.dim//module.heads), dropout_p=module.dropout if module.training else 0.,
            all_multiple=all_multiple)
        pieces.append(y[..., :module.dim//module.heads].transpose(1, 2).flatten(0, 1))
    selected = torch.cat(pieces).index_select(0, layout['restore']) if pieces else raw
    present = kv.valid.any(-1)[:, None].expand_as(valid).flatten().index_select(0, layout['indices'])
    return module.output(selected.reshape(len(x), module.dim)).masked_fill(~present[:, None], 0)
