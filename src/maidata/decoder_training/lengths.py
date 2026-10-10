"""Small exact input-length index from selected field records; no Physical/audio IO."""
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


def chart_lengths(record, windows, history_beats):
    # Every real onset contributes one MOMENT and its complete content. A
    # target onset exactly at the boundary shares the injected MOMENT instead.
    tokens, mapping = record['tokens'], record['token_map']
    if len(tokens) != len(mapping):
        raise ValueError('unaligned field token map')
    counts = Counter()
    for i, (token, row) in enumerate(zip(tokens, mapping)):
        if row['token_index'] != i or type(row['tick']) is not int:
            raise ValueError('invalid field token index/time')
        if token['type'] != 'MOMENT':
            counts[row['tick']] += 1
    ticks = sorted(counts)
    prefix = [0]
    for tick in ticks:
        prefix.append(prefix[-1] + counts[tick] + 1)
    result = {}
    for row in windows:
        start, end = row['start_tick'], row['end_tick']
        a, b = bisect_left(ticks, start), bisect_left(ticks, end)
        h = bisect_left(ticks, start - 96 * history_beats)
        at_start = a < len(ticks) and ticks[a] == start
        target = 1 + prefix[b] - prefix[a] - int(at_start)
        result[row['window_id']] = (target, prefix[a] - prefix[h])
    return result


def load_lengths(plan, config):
    """Read only this split's fields once to size its batches and schedule.

    No rehash, grammar/Physical replay or persistent cache/acceptance marker.
    Keep only two integers per window; workers read normal payloads as usual.
    """
    windows = defaultdict(list)
    for row in plan.windows:
        windows[row['chart_id']].append(row)
    result = {}
    for key, rows in windows.items():
        path = Path(config['decoder_train_chart_cache']) / plan.charts[key]['chart_cache_path']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            record = json.load(stream)
        if (record['schema'] != 'maidata.full-chart-original.v2' or record['view'] != 'original'
                or record['source']['chart_id'] != key
                or record['source']['audio_variant'] != plan.charts[key]['audio_variant']):
            raise ValueError('length index field identity differs from plan')
        result.update(chart_lengths(record, rows, config['chart_history_beats']))
    return result
