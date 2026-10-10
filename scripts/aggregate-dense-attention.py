"""Time-bin recorded attention for plotting, retaining all heads separately.

Keys are summed into time bins; queries within each bin are averaged. The raw
archive remains unaggregated. No local-only crop or renormalization is applied.
"""
import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def seconds(beats, timing):
    points = timing['bpm_points']
    b = np.array([x['position_beats']['num']/x['position_beats']['den'] for x in points])
    rates = np.array([x['bpm'] for x in points])
    offsets = np.zeros(len(b)) + timing['first_seconds']
    for i in range(1, len(b)):
        offsets[i] = offsets[i-1] + (b[i]-b[i-1])*60/rates[i-1]
    indices = np.maximum(np.searchsorted(b, beats, side='right')-1, 0)
    return offsets[indices] + (beats-b[indices])*60/rates[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--timing', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--bin-seconds', type=float, default=.5)
    args = parser.parse_args()
    root = args.root
    timing = json.loads(args.timing.read_text(encoding='utf-8-sig'))
    index = json.loads((root/'index.json').read_text())
    journal = json.loads((root/'journal.json').read_text())
    generation = json.loads((root/'inference/generation.json').read_text())
    duration = generation['audio_preparation']['audio_end_seconds']
    axis_end = duration
    # A partially filled last beat can have its nominal center after the last
    # audio sample. Keep that real valid key rather than silently losing mass.
    with zipfile.ZipFile(root/'raw-attention.zip') as archive:
        first_global = next(x for x in index if x['kind'] == 'audio_global')
        with archive.open(first_global['path']) as stream, np.load(stream, allow_pickle=False) as a:
            axis_end = max(axis_end, float(seconds(a['key_beats'][a['key_valid']], timing).max())+1e-6)
    n = math.ceil(axis_end/args.bin_seconds)
    final = set(journal['final_history_node_ids'])
    accepted = {row['query_node_id'] for row in journal['accepted_predictions']}
    assert accepted <= final, 'accepted output points to a rolled-back query'
    streams = {}
    stats = dict(raw_records=len(index), accepted_queries=len(accepted), final_history_nodes=len(final),
                 total_observed_nodes=len(journal['nodes']), masked_probability_max=0.,
                 stored_row_sum_max_error=0., future_causal_probability_max=0.)

    def accumulate(label, p, qb, kb, valid):
        # Stored tensors: [batch, heads, query, key], including all real keys.
        p = p.astype(np.float32)
        heads = p.shape[1]
        item = streams.setdefault(label, dict(values=np.zeros((heads, n, n), np.float64),
            counts=np.zeros(n, np.int64), query_count=0, outside_key_mass=0., rowsum_min=math.inf, rowsum_max=0.))
        qt, kt = seconds(qb, timing), seconds(kb, timing)
        qbin = np.floor(qt/args.bin_seconds).astype(int)
        kbin = np.floor(kt/args.bin_seconds).astype(int)
        valid = valid & (qbin >= 0) & (qbin < n)
        for batch in range(len(p)):
            key_good = (kbin[batch] >= 0) & (kbin[batch] < n)
            for row in np.flatnonzero(valid[batch]):
                weights = p[batch, :, row]
                sums = weights.sum(-1)
                stats['stored_row_sum_max_error'] = max(stats['stored_row_sum_max_error'], float(np.abs(sums-1).max()))
                item['rowsum_min'] = min(item['rowsum_min'], float(sums.min()))
                item['rowsum_max'] = max(item['rowsum_max'], float(sums.max()))
                item['outside_key_mass'] += float(weights[:, ~key_good].sum()/heads)
                destination = qbin[batch, row]
                for head in range(heads):
                    item['values'][head, destination] += np.bincount(kbin[batch, key_good],
                        weights=weights[head, key_good], minlength=n)
                item['counts'][destination] += 1
                item['query_count'] += 1

    with zipfile.ZipFile(root/'raw-attention.zip') as archive:
        for i, record in enumerate(index):
            with archive.open(record['path']) as stream, np.load(stream, allow_pickle=False) as arrays:
                if record['kind'] == 'decoder':
                    ids = arrays['query_node_ids']
                    valid = arrays['query_valid'] & np.isin(ids, list(accepted))[None]
                    for layer in range(8):
                        for kind in ['self', 'cross']:
                            p = arrays[f'{kind}_{layer:02d}']
                            allowed = arrays[f'{kind}_key_valid'][:, None, None, :]
                            if kind == 'self':
                                qindex = len(arrays['key_node_ids'])-len(ids)+np.arange(len(ids))
                                causal = np.arange(p.shape[-1])[None, :] <= qindex[:, None]
                                forbidden = p * (~causal)[None, None]
                                stats['future_causal_probability_max'] = max(stats['future_causal_probability_max'], float(forbidden.max()))
                                allowed = allowed & causal[None, None]
                            stats['masked_probability_max'] = max(stats['masked_probability_max'], float((p*~allowed).max()))
                            accumulate(f'decoder_{kind}_L{layer+1}', p, arrays['query_beats'],
                                       arrays[f'{kind}_key_beats'], valid)
                else:
                    p = arrays['probability']
                    stats['masked_probability_max'] = max(stats['masked_probability_max'], float((p*~arrays['allowed']).max()))
                    valid = arrays['query_valid']
                    if record['kind'] == 'audio_local':
                        valid = valid & arrays['canonical_query']
                    accumulate(record['layer'], p, arrays['query_beats'], arrays['key_beats'], valid)
            if i % 250 == 0:
                print('aggregated', i, '/', len(index), flush=True)
    output = {}
    metadata = []
    for label, item in streams.items():
        value = (item['values']/np.maximum(1, item['counts'])[None, :, None]).astype(np.float32)
        output[label] = value
        output[label+'__counts'] = item['counts']
        metadata.append(dict(label=label, heads=value.shape[0], shape=list(value.shape),
            query_count=item['query_count'], outside_key_mass_mean=item['outside_key_mass']/max(1,item['query_count']),
            rowsum_range=[item['rowsum_min'], item['rowsum_max']]))
    np.savez_compressed(root/'time-binned-attention.npz', **output)
    import soundfile as sf
    audio, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    mono = audio.mean(-1)
    window = max(1, sr//100)
    padded = np.pad(mono, (0, (-len(mono)) % window)).reshape(-1, window)
    waveform = dict(sample_rate=sr, envelope_step_seconds=window/sr,
        minimum=padded.min(-1).round(5).tolist(), maximum=padded.max(-1).round(5).tolist())
    report = dict(duration_seconds=duration, axis_end_seconds=n*args.bin_seconds, bin_seconds=args.bin_seconds, bins=n, streams=metadata,
        statistics=stats, timing=timing, waveform=waveform, generation=dict(
            rows=generation['generated_rows'], moments=generation['generated_moments'],
            truncated=generation['truncated'], checkpoint_epoch=generation['checkpoint_epoch'],
            checkpoint_step=generation['checkpoint_step'], checkpoint_sha256=generation['checkpoint_sha256'],
            sampling=generation['sampling'], physical=generation['physical_hard_mask']),
        note='Mean over queries, sum over keys in each time bin; no row renormalization; final accepted prediction queries only. Local audio uses each target window central 16-beat queries, with all original allowed keys.')
    (root/'aggregation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
