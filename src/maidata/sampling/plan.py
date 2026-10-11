"""Audio-derived base windows and deterministic grouped supervision occurrences."""

from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import gzip
import json
import math
import os
from pathlib import Path
import random

from maidata.time import TimingMap, TICKS_PER_BEAT


SCHEMA = 'maidata.window-sampling-plan.v1'
CHART_SCHEMA = 'maidata.full-chart-original.v2'
CONFIG_KEYS = ('target_beats', 'target_stride_beats', 'audio_before_beats',
               'audio_after_beats', 'chart_history_beats',
               'complex_rhythm_min_kinds', 'complex_rhythm_multiplier')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _config(config):
    result = {key: config[key] for key in CONFIG_KEYS}
    for key in CONFIG_KEYS[:-1]:
        minimum = 1 if key in ('target_beats', 'target_stride_beats',
                              'complex_rhythm_min_kinds') else 0
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if result['target_stride_beats'] > result['target_beats']:
        raise ValueError('target_stride_beats must be <= target_beats; gaps are not supported')
    _multiplier(result['complex_rhythm_multiplier'])
    result['complex_rhythm_multiplier'] = float(result['complex_rhythm_multiplier'])
    return result


def _multiplier(value):
    if isinstance(value, bool):
        raise ValueError('complex rhythm multiplier must be finite and >= 1')
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError('complex rhythm multiplier must be finite and >= 1') from exc
    if not result.is_finite() or result < 1:
        raise ValueError('complex rhythm multiplier must be finite and >= 1')
    return result


def _seed_epoch(seed, epoch):
    if type(seed) is not int or type(epoch) is not int or epoch < 1:
        raise ValueError('seed must be an integer and epoch must be an integer >= 1')


def _coverage(source, config):
    timing = TimingMap.from_audio_timing(source['timing'])
    duration = source['audio_end_seconds']
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('audio_end_seconds must be finite and positive')
    start, end = timing.second_to_beat(0), timing.second_to_beat(duration)
    first = math.floor(start)
    target, stride = config['target_beats'], config['target_stride_beats']
    # One shortest prefix of the stride grid whose last full target reaches audio end.
    count = max(1, math.ceil((end - first - target) / stride) + 1)
    starts = [first + i * stride for i in range(count)]
    # The extra full empty target begins after the final audio-coverage target.
    starts.append(starts[-1] + target)
    encoder_windows = math.ceil((end - first) / target)
    return dict(first_beat=first, actual_audio_start_beat=start, actual_audio_end_beat=end,
                span_beats=end - start, target_starts_beats=starts,
                encoder_starts_beats=[first + i * target for i in range(encoder_windows)],
                draft_end_beat=first + (encoder_windows + 1) * target,
                supervision_end_beat=starts[-1] + target)


def _note_onsets(record):
    if record.get('schema') != CHART_SCHEMA or record.get('view') != 'original':
        raise ValueError('expected original v2 chart cache')
    tokens, mapping = record.get('tokens'), record.get('token_map')
    if not isinstance(tokens, list) or not isinstance(mapping, list) or len(tokens) != len(mapping):
        raise ValueError('tokens and token_map must be aligned lists')
    onsets, slides = set(), set()
    for i, (token, row) in enumerate(zip(tokens, mapping)):
        if (not isinstance(token, dict) or not isinstance(row, dict)
                or row.get('token_index') != i or type(row.get('tick')) is not int):
            raise ValueError(f'invalid token time/index mapping at {i}')
        tick = row['tick']
        # Boundary MOMENT, ROUTE, durations and their endpoints are not real onsets.
        if token.get('type') == 'NOTE':
            onsets.add(tick)
            if token.get('kind') == 'slide':
                slides.add(tick)
    return sorted(onsets), sorted(slides)


def chart_windows(record, *, config):
    """Return unique base windows in chart order, from NOTE onsets and real audio.

    Target ranges and history ranges are left-closed/right-open. Only adjacent
    distinct NOTE onsets inside the same target contribute rhythm deltas.
    This consumes ordinary cache shapes; it does not revalidate chart grammar.
    """
    config = _config(config)
    source = record['source']
    onsets, slides = _note_onsets(record)
    coverage = _coverage(source, config)
    first, starts = coverage['first_beat'], coverage['target_starts_beats']
    tail_tick = starts[-1] * TICKS_PER_BEAT
    if onsets and (onsets[0] < first * TICKS_PER_BEAT or onsets[-1] >= tail_tick):
        raise ValueError(f"{source['chart_id']}: NOTE outside audio-covering base targets or inside extra tail")
    windows = []
    for index, start in enumerate(starts):
        end = start + config['target_beats']
        lo, hi = start * TICKS_PER_BEAT, end * TICKS_PER_BEAT
        a, b = bisect_left(onsets, lo), bisect_left(onsets, hi)
        local = onsets[a:b]
        distinct = sorted({right - left for left, right in zip(local, local[1:])
                           if 0 < right - left < TICKS_PER_BEAT})
        upper = end + config['audio_after_beats']
        windows.append(dict(window_id=f"{source['chart_id']}:{index}",
            chart_id=source['chart_id'], song_id=source['song_id'],
            difficulty=str(source['difficulty']), audio_variant=source['audio_variant'],
            window_index=index, start_tick=lo, end_tick=hi,
            extra_tail=index == len(starts) - 1, first_beat=first,
            start_beat=start, end_beat=end,
            audio_start_beat=start - config['audio_before_beats'], audio_end_beat=upper,
            U=upper, history_start_tick=lo - config['chart_history_beats'] * TICKS_PER_BEAT,
            history_end_tick=lo, unique_note_onsets=len(local),
            slide=bisect_left(slides, lo) < bisect_left(slides, hi),
            subquarter_delta_ticks=distinct, subquarter_delta_kinds=len(distinct),
            complex=len(distinct) >= config['complex_rhythm_min_kinds']))
    return windows


def make_occurrences(windows, *, multiplier, seed, epoch):
    """Keep all bases, then add complex copies; shuffle complete audio groups.

    For a noninteger multiplier m, every complex window gets floor(m) total
    copies and round_half_up(frac(m) * complex_count) windows get one more.
    The fractional subset is sampled globally without replacement. Occurrence
    zero always exists; positive occurrence numbers identify added copies.
    """
    _seed_epoch(seed, epoch)
    multiplier = _multiplier(multiplier)
    windows = list(windows)
    identifiers = [w['window_id'] for w in windows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('window_id must be unique')
    complex_ids = [w['window_id'] for w in windows if w['complex']]
    whole = int(multiplier)
    fractional_count = int(((multiplier - whole) * len(complex_ids)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    fractional = set(random.Random(f'{seed}:{epoch}:fractional').sample(complex_ids, fractional_count))
    grouped = defaultdict(list)
    for window in windows:
        copies = whole + int(window['window_id'] in fractional) if window['complex'] else 1
        for occurrence in range(copies):
            grouped[window['audio_variant']].append(dict(window_id=window['window_id'],
                chart_id=window['chart_id'], audio_variant=window['audio_variant'],
                occurrence=occurrence))
    order = list(grouped)
    random.Random(f'{seed}:{epoch}:groups').shuffle(order)
    result = []
    for variant in order:
        group = grouped[variant]
        random.Random(f'{seed}:{epoch}:within:{variant}').shuffle(group)
        result.extend(group)
    for sequence, row in enumerate(result):
        row['sequence_index'] = sequence
    return result


def _summary(windows, occurrences):
    total, complex_count = len(windows), sum(w['complex'] for w in windows)
    return dict(base_windows=total, complex_windows=complex_count,
                slide_windows=sum(w['slide'] for w in windows),
                empty_windows=sum(w['unique_note_onsets'] == 0 for w in windows),
                extra_tail_windows=sum(w['extra_tail'] for w in windows),
                occurrences=len(occurrences), added_occurrences=len(occurrences) - total,
                complex_fraction=complex_count / total if total else 0,
                actual_total_multiplier=len(occurrences) / total if total else 0)


def _path(value, base):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _binding(selected, catalog, source):
    """Check only explicit identity/timing fields, without hashing old payloads."""
    for key in ('chart_id', 'song_id', 'audio_variant', 'audio_asset', 'audio_end_seconds'):
        if source.get(key) != catalog.get(key) or selected.get(key) != catalog.get(key):
            raise ValueError(f"{selected['chart_id']}: {key} binding mismatch")
    if str(source.get('difficulty')) != str(catalog.get('difficulty')) or str(selected.get('difficulty')) != str(catalog.get('difficulty')):
        raise ValueError(f"{selected['chart_id']}: difficulty binding mismatch")
    if source.get('timing') != catalog.get('timing'):
        raise ValueError(f"{selected['chart_id']}: timing binding mismatch")
    if selected.get('chart_cache_path') != catalog.get('path'):
        raise ValueError(f"{selected['chart_id']}: chart path binding mismatch")


def _payload_timing(record):
    if 'metadata' not in record:
        raise ValueError('chart metadata missing')
    expected = TimingMap.from_audio_timing(record['source']['timing'])
    actual = TimingMap.from_ir(record['metadata'])
    compact = []
    for point in actual.points:
        if not compact or compact[-1][1] != point[1]:
            compact.append(point)
    if tuple(compact) != expected.points or abs(actual.first_seconds - expected.first_seconds) > 1e-9:
        raise ValueError(f"{record['source']['chart_id']}: payload/audio timing mismatch")


def _split_metadata(split_path, selected, *, selection_path):
    """Validate an existing partition using only selection and split metadata."""
    path = Path(split_path).resolve()
    split = _read_json(path)
    if split.get('schema') != 'maidata.encoder-draft-split.v1':
        raise ValueError('expected maidata.encoder-draft-split.v1')
    if _path(split['source_selection'], path.parent).resolve() != Path(selection_path).resolve():
        raise ValueError('split source_selection does not reference this selection')
    rows = {row['chart_id']: row for row in selected}
    subsets = {}
    for partition in ('train', 'validation'):
        declared = split[partition]
        ids = declared['chart_ids']
        if len(ids) != len(set(ids)) or not set(ids) <= set(rows):
            raise ValueError(f'{partition}: duplicate or unknown split chart IDs')
        subsets[partition] = set(ids)
        for listed, field in (('song_ids', 'song_id'), ('audio_assets', 'audio_asset'),
                              ('audio_variants', 'audio_variant')):
            expected = {rows[chart_id][field] for chart_id in ids}
            actual = declared[listed]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError(f'{partition}: split {listed} disagree with selected chart bindings')
    if subsets['train'] & subsets['validation'] or subsets['train'] | subsets['validation'] != set(rows):
        raise ValueError('split chart IDs must be disjoint and cover the full selection')
    for key in ('song_ids', 'audio_assets', 'audio_variants'):
        if set(split['train'][key]) & set(split['validation'][key]):
            raise ValueError(f'train/validation {key} overlap')
    return path, subsets


def _partition(partition, split):
    if partition not in ('all', 'train', 'validation'):
        raise ValueError('partition must be all, train or validation')
    if partition != 'all' and split is None:
        raise ValueError('train/validation partition requires an explicit split file')


def _assemble(plan, windows, chart_meta, groups):
    """Shared occurrence/statistics assembly for fresh and metadata-only plans."""
    occurrences = make_occurrences(windows, multiplier=plan['config']['complex_rhythm_multiplier'],
                                   seed=plan['seed'], epoch=plan['epoch'])
    per_chart = Counter(row['chart_id'] for row in occurrences)
    chart_meta = [dict(chart, occurrences=per_chart[chart['chart_id']],
                       added_occurrences=per_chart[chart['chart_id']] - chart['base_windows'])
                  for chart in chart_meta]
    audio_groups = []
    offset = 0
    while offset < len(occurrences):
        variant = occurrences[offset]['audio_variant']
        stop = offset + 1
        while stop < len(occurrences) and occurrences[stop]['audio_variant'] == variant:
            stop += 1
        audio_groups.append(dict(groups[variant], group_index=len(audio_groups),
            occurrence_start=offset, occurrence_stop=stop, occurrences=stop - offset))
        offset = stop
    difficulty_by_chart = {r['chart_id']: r['difficulty'] for r in chart_meta}
    by_difficulty = {difficulty: _summary(
        [w for w in windows if w['difficulty'] == difficulty],
        [r for r in occurrences if difficulty_by_chart[r['chart_id']] == difficulty])
        for difficulty in dict.fromkeys(r['difficulty'] for r in chart_meta)}
    totals = dict(_summary(windows, occurrences), charts=len(chart_meta),
                  songs=len({r['song_id'] for r in chart_meta}), audio_groups=len(audio_groups))
    return dict(plan, totals=totals, by_difficulty=by_difficulty, charts=chart_meta,
                audio_groups=audio_groups, windows=windows, occurrences=occurrences)


def partition_plan(plan, split_path, *, partition):
    """Derive a partition from a complete all-plan without rereading source caches.

    The split JSON is the only file read. Input dictionaries are not modified.
    Validation always keeps base occurrences only; all retains the full selection.
    """
    _partition(partition, split_path)
    if plan.get('policy', {}).get('split') not in ('all', 'unassigned'):
        raise ValueError('partition_plan requires a complete all-selection plan')
    group_lookup = {g['audio_variant']: g for g in plan['audio_groups']}
    selected = [dict(chart, audio_asset=group_lookup[chart['audio_variant']]['audio_asset'])
                for chart in plan['charts']]
    path, subsets = _split_metadata(split_path, selected, selection_path=plan['sources']['selection'])
    wanted = {row['chart_id'] for row in selected} if partition == 'all' else subsets[partition]
    windows = [w for w in plan['windows'] if w['chart_id'] in wanted]
    charts = [c for c in plan['charts'] if c['chart_id'] in wanted]
    groups = {}
    for variant, group in group_lookup.items():
        owned = [c for c in charts if c['audio_variant'] == variant]
        if owned:
            groups[variant] = dict(group, chart_ids=[c['chart_id'] for c in owned],
                                  song_ids=list(dict.fromkeys(c['song_id'] for c in owned)))
    config = dict(plan['config'])
    if partition == 'validation':
        config['complex_rhythm_multiplier'] = 1.0
    header = {key: value for key, value in plan.items()
              if key not in ('totals', 'by_difficulty', 'charts', 'audio_groups', 'windows', 'occurrences', 'files')}
    header.update(config=config, sources=dict(plan['sources'], split=str(path)),
                  policy=dict(plan['policy'], split=partition))
    return _assemble(header, windows, charts, groups)


def build_plan(selection_path, *, config, seed=20261010, epoch=1, chart_cache=None,
               split=None, partition='all'):
    """Materialize references for the fixed selection; only selected payloads are read."""
    _seed_epoch(seed, epoch)
    _partition(partition, split)
    config = _config(config)
    selection_path = Path(selection_path).resolve()
    selection = _read_json(selection_path)
    cache = (Path(chart_cache).resolve() if chart_cache is not None else
             _path(selection['sources']['chart_catalog'], selection_path.parent).resolve().parent)
    catalog_path, summary_path = cache / 'catalog.json', cache / 'production-summary.json'
    catalog, production = _read_json(catalog_path), _read_json(summary_path)
    if catalog.get('schema') != CHART_SCHEMA or catalog.get('views') != ['original']:
        raise ValueError('expected original-only v2 chart catalog')
    if selection['sources'].get('chart_schema') != CHART_SCHEMA:
        raise ValueError('selection chart schema mismatch')
    selected = selection['charts']
    selected_ids = selection['selected_chart_ids']
    if not selected or [r['chart_id'] for r in selected] != selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError('selection charts must match its ordered unique selected_chart_ids')
    split_path = None
    if split is not None:
        split_path, subsets = _split_metadata(split, selected, selection_path=selection_path)
        if partition != 'all':
            selected = [row for row in selected if row['chart_id'] in subsets[partition]]
    if partition == 'validation':
        config['complex_rhythm_multiplier'] = 1.0
    wanted = {row['chart_id'] for row in selected}
    rows = {r['chart_id']: r for r in catalog['charts'] if r['chart_id'] in wanted}
    statuses = {r['chart_id']: r for r in production['results'] if r['chart_id'] in wanted}
    windows, chart_meta, groups = [], [], {}
    for index, chosen in enumerate(selected):
        chart_id = chosen['chart_id']
        if chart_id not in rows or statuses.get(chart_id, {}).get('status') != 'success' or chosen.get('cache_status') != 'success':
            raise ValueError(f'{chart_id}: selected chart must have success status')
        row = rows[chart_id]
        if statuses[chart_id].get('view', 'original') != 'original':
            raise ValueError(f'{chart_id}: selected chart is not original')
        relative = Path(row['path'])
        payload_path = (cache / relative).resolve()
        if relative.is_absolute() or not payload_path.is_relative_to(cache):
            raise ValueError(f'{chart_id}: cache path must stay inside chart cache')
        with gzip.open(payload_path, 'rt', encoding='utf-8-sig') as stream:
            record = json.load(stream)
        _binding(chosen, row, record['source'])
        _payload_timing(record)
        local = chart_windows(record, config=config)
        coverage = _coverage(row, config)
        variant = row['audio_variant']
        if variant not in groups:
            groups[variant] = dict(audio_variant=variant, audio_asset=row['audio_asset'],
                audio_end_seconds=row['audio_end_seconds'], timing=row['timing'],
                **{k: v for k, v in coverage.items() if k != 'target_starts_beats'},
                chart_ids=[], song_ids=[])
        group = groups[variant]
        if any(group[key] != row[key] for key in ('audio_asset', 'audio_end_seconds', 'timing')):
            raise ValueError(f'{chart_id}: audio variant has inconsistent shared metadata')
        group['chart_ids'].append(chart_id)
        if row['song_id'] not in group['song_ids']:
            group['song_ids'].append(row['song_id'])
        chart_meta.append(dict(chart_id=chart_id, song_id=row['song_id'],
            difficulty=str(row['difficulty']), audio_variant=variant,
            chart_cache_path=row['path'], selection_index=chosen.get('selection_index'),
            chart_index=chosen.get('chart_index', index + 1),
            source_group=chosen.get('source_group'), version=chosen.get('version'),
            first_beat=coverage['first_beat'], base_windows=len(local),
            complex_windows=sum(w['complex'] for w in local)))
        windows.extend(local)
    header = dict(schema=SCHEMA, name=selection.get('name'),
        sources=dict(selection=str(selection_path), selection_schema=selection.get('schema'), chart_cache=str(cache),
                     chart_catalog=str(catalog_path), chart_production_summary=str(summary_path),
                     chart_schema=CHART_SCHEMA, chart_recipe=catalog.get('producer_recipe'),
                     audio_catalog=selection['sources'].get('audio_catalog')),
        config=config, seed=seed, epoch=epoch,
        policy=dict(views=['original'], physical_violations_filtered=False,
                    base_windows='all retained once; include one complete audio-external tail',
                    rhythm='distinct adjacent NOTE-onset tick deltas within the target; 0 < delta < 96',
                    complex_comparison='>=', slide_multiplier=1.0,
                    fractional_selection='global complex subset without replacement; count rounded half up',
                    ordering='shuffle audio_variant groups and occurrences within each group',
                    offsets='zero-based [occurrence_start, occurrence_stop)', split=partition))
    if split_path is not None:
        header['sources']['split'] = str(split_path)
    return _assemble(header, windows, chart_meta, groups)


def write_plan(plan, output):
    """Write a compact manifest and reference-only JSONL files, without copying tokens."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output / filename for key, filename in (
        ('plan', 'plan.json'), ('windows', 'windows.jsonl'), ('occurrences', 'occurrences.jsonl'))}
    for key in ('windows', 'occurrences'):
        temporary = paths[key].with_suffix(paths[key].suffix + '.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for row in plan[key]:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        os.replace(temporary, paths[key])
    manifest = {key: value for key, value in plan.items() if key not in ('windows', 'occurrences')}
    manifest['files'] = dict(windows=paths['windows'].name, occurrences=paths['occurrences'].name)
    temporary = paths['plan'].with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, paths['plan'])
    return paths
