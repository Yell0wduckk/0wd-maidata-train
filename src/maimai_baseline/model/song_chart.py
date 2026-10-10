"""Whole-song MOMENT attention, with four-beat pooling and shared posterior draws."""
import torch
from torch.nn import functional as F
from maimai_baseline.model.style import StyleContext, StylePosterior


def transformer_layer(layer, x, chunk=128, valid=None):
    """Exact all-key attention; query chunking does not restrict receptive field."""
    y = layer.norm1(x)
    attn = layer.self_attn
    q, k, v = F.linear(y, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
    def heads(t):
        return t.reshape(t.shape[0], t.shape[1], attn.num_heads, -1).transpose(1, 2)
    q, k, v = map(heads, (q, k, v))
    pieces = [F.scaled_dot_product_attention(q[:, :, i:i+chunk], k, v,
        attn_mask=None if valid is None else valid[:, None, None, :],
        dropout_p=attn.dropout if layer.training else 0.) for i in range(0, x.shape[1], chunk)]
    y = torch.cat(pieces, 2).transpose(1, 2).reshape_as(x)
    x = x + layer.dropout1(attn.out_proj(y))
    x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
    return x if valid is None else x.masked_fill(~valid[..., None], 0.)


def encode_songs(encoder, songs):
    """One packed embedding/aggregation and padded Transformer batch per step.

    Only query positions are chunked; every valid query sees its whole song.
    Indices use disjoint row/note/MOMENT/block offsets, including empty charts.
    """
    if not songs:
        raise ValueError('empty chart batch')
    device = encoder.empty.device
    offsets = dict(rows=0, notes=0, moments=0, blocks=0)
    indices = {k: [] for k in ('note_rows', 'branch_rows', 'moment_rows',
                              'branch_note', 'note_moment', 'moment_slot')}
    block_lengths = []
    for p in songs:
        count = p['valid'].numel()
        if p['valid'].shape != (1, count) or count < 1:
            raise ValueError('song validity must be [1,K], K >= 1')
        block_lengths.append(count)
        for key, offset in (('note_rows','rows'), ('branch_rows','rows'), ('moment_rows','rows'),
                            ('branch_note','notes'), ('note_moment','moments'), ('moment_slot','blocks')):
            indices[key].append(p[key] + offsets[offset])
        offsets['rows'] += p['ids']['event'].numel()
        offsets['notes'] += p['note_rows'].numel()
        offsets['moments'] += p['moment_rows'].numel()
        offsets['blocks'] += count
    indices = {k: torch.cat(v) for k, v in indices.items()}
    ids = {k: torch.cat([p['ids'][k] for p in songs]) for k in songs[0]['ids']}
    value = encoder.modifier_projection(ids)
    mass = (ids['event'].eq(encoder.note_event_id) | ids['event'].eq(encoder.branch_event_id))[:, None].to(value.dtype)
    for name, embedding in encoder.embeddings.items():
        active = ids[name].ne(0)[:, None]
        value = value + embedding(ids[name]) * active
        mass = mass + active
    for name, embedding in encoder.time_embeddings.items():
        active = torch.cat([p['time_active'][name] for p in songs])[:, None]
        value = value + embedding(torch.cat([p['times'][name] for p in songs]).float()/96.) * active
        mass = mass + active
    value = value / mass.clamp_min(1).sqrt()
    notes = value.index_select(0, indices['note_rows'])
    branches = value.new_zeros(notes.shape).index_add(0, indices['branch_note'], value.index_select(0, indices['branch_rows']))
    notes = encoder.note_combine(torch.cat((notes, branches), -1))
    moments = value.index_select(0, indices['moment_rows'])
    members = value.new_zeros(moments.shape).index_add(0, indices['note_moment'], notes.to(value.dtype))
    moments = encoder.moment_combine(torch.cat((moments, members), -1))
    moments = moments + encoder.position(torch.cat([p['moment_absolute'] for p in songs]))
    moments = moments + encoder.position(torch.cat([p['moment_delta'] for p in songs]))
    # Assemble sequence metadata only. No per-song encoder or per-block pool calls.
    sequences, sequence_blocks = [], []
    moment_offset = block_offset = 0
    for p, count in zip(songs, block_lengths, strict=True):
        blocks = p['moment_slot']
        occupancy = torch.zeros(count, device=device, dtype=torch.long).scatter_add_(0, blocks, torch.ones_like(blocks))
        empty_blocks = (occupancy == 0).nonzero().flatten()
        empty = encoder.empty[None] + encoder.position(empty_blocks.float()*4+2) + encoder.position(torch.zeros_like(empty_blocks).float())
        length = p['moment_rows'].numel()
        sequences.append(torch.cat((moments[moment_offset:moment_offset+length], empty.to(moments.dtype))))
        sequence_blocks.append(torch.cat((blocks, empty_blocks)) + block_offset)
        moment_offset += length
        block_offset += count
    lengths = [len(x) for x in sequences]
    x = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    valid = torch.arange(x.shape[1], device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    for layer in encoder.transformer.layers:
        x = transformer_layer(layer, x, valid=valid)
    x = encoder.transformer.norm(x).float()[valid]
    blocks = torch.cat(sequence_blocks)
    scores = encoder.pool(x).squeeze(-1).float()
    # Stable segment softmax, followed by one differentiable segment sum.
    maxima = scores.new_full((block_offset,), -torch.inf).scatter_reduce(0, blocks, scores.detach(), reduce='amax')
    weights = (scores - maxima[blocks]).exp()
    sums = weights.new_zeros(block_offset).index_add(0, blocks, weights)
    pooled = x.new_zeros((block_offset, 128)).index_add(0, blocks, x * (weights/sums[blocks])[:, None])
    features = [h[None].masked_fill(~p['valid'][..., None], 0.)
                for h, p in zip(pooled.split(block_lengths), songs, strict=True)]
    stats = dict(unique_charts=len(songs), chart_encoder_calls=1, chart_encoder_batch_size=len(songs),
                 chart_tokens=sum(lengths), padded_chart_tokens=len(songs)*max(lengths),
                 chart_padding_tokens=len(songs)*max(lengths)-sum(lengths), activation_checkpoint_calls=0)
    return features, stats


def encode_song(encoder, packed):
    """Single-song cache/inference adapter using the same batched implementation."""
    return encode_songs(encoder, [packed])[0][0]


def window_posteriors(model, inputs, *, sample):
    """Encode and sample once per unique chart/view; replicate only small windows."""
    features, stats = encode_songs(model.chart_style_encoder, inputs['songs'])
    lengths = [h.shape[1] for h in features]
    h = torch.nn.utils.rnn.pad_sequence([h[0] for h in features], batch_first=True)
    valid = torch.nn.utils.rnn.pad_sequence([p['valid'][0] for p in inputs['songs']], batch_first=True)
    all_posteriors = model.chart_style_encoder.posterior(h, valid)
    posteriors = [StylePosterior(all_posteriors.mu[i:i+1,:n], all_posteriors.logvar[i:i+1,:n],
                                all_posteriors.beats[i:i+1,:n], all_posteriors.valid[i:i+1,:n])
                  for i, n in enumerate(lengths)]
    model.a_encoding_stats = dict(stats, posterior_calls=1, shared_window_references=len(inputs['window_groups'])-len(features))
    tokens = [p.mu + torch.exp(.5*p.logvar)*torch.randn_like(p.mu) if sample else p.mu for p in posteriors]
    mus, logvars, zs, masks = [], [], [], []
    for group, start in zip(inputs['window_groups'].tolist(), inputs['window_blocks'].tolist(), strict=True):
        n = min(4, posteriors[group].mu.shape[1]-start)
        if n < 1:
            raise ValueError('window outside full chart context')
        def take(value):
            return F.pad(value[:, start:start+n], (0, 0, 0, 4-n))
        mus.append(take(posteriors[group].mu)); logvars.append(take(posteriors[group].logvar)); zs.append(take(tokens[group]))
        masks.append(F.pad(posteriors[group].valid[:, start:start+n], (0, 4-n)))
    valid = torch.cat(masks) & inputs['window_valid']
    beats = (torch.arange(4, device=features[0].device).float()*4+2)[None].expand_as(valid)
    posterior = StylePosterior(torch.cat(mus), torch.cat(logvars), beats, valid)
    context = StyleContext(torch.cat(zs).masked_fill(~valid[..., None], 0.), beats, valid)
    return context, posterior
