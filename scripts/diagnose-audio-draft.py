"""Terminal clean/zero/mismatched draft generations; no acceptance threshold."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from maimai_baseline.checkpoint import load_baseline_checkpoint
from maimai_baseline.data.audio_draft import resolve_paths
from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.draft_cache import AudioDraftCache
from maimai_baseline.data.frozen_latent import selected_segments
from maimai_baseline.data.prior_decoder import _read
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.teacher_audio import TeacherAudioCache, draft_producer_identity
from maimai_baseline.inference.generation import generate_baseline_rows
from maimai_baseline.inference.style import DraftSongStyle
from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_baseline.physical.miacode_inference import MiaCodeInferenceState
from maimai_baseline.style_contract import is_draft
from maimai_baseline.time.audio_end import audio_end
from maimai_dataset.timing import TimingMap


def fixed_samples(cache, count=3):
    chosen, seen = [], set()
    for record in cache.records:
        if record['view_id'] == 0 and record['condition'][0] not in seen:
            chosen.append(record); seen.add(record['condition'][0])
            if len(chosen) == count:
                break
    if not chosen:
        raise ValueError('no canonical draft samples in selected training data')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-manifest', 'epoch-plan-cache', 'checkpoint', 'draft-cache',
                 'teacher-audio-cache', 'route-artifact', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--physical-path-catalog', type=Path)
    args = resolve_paths(parser.parse_args())
    if args.physical_path_catalog is None:
        mappings = {key.replace('-', '_'): value for key, value in _read(args.dataset_manifest)['inputs'].items()}
        path = Path(mappings['physical_path_catalog'])
        args.physical_path_catalog = path if path.is_absolute() else args.dataset_manifest.parent/path
    output = args.output.resolve()
    if output.exists():
        raise ValueError('refusing to overwrite final draft samples')
    model, payload = load_baseline_checkpoint(args.checkpoint, route_artifact=args.route_artifact,
        device=args.device, strict_source=False, verify_artifact_payloads=False)
    if not is_draft(model.config):
        raise ValueError('draft samples require an audio draft A checkpoint')
    model.eval()
    teacher, cache = TeacherAudioCache(args.teacher_audio_cache), AudioDraftCache(args.draft_cache)
    if draft_producer_identity(payload['baseline_manifest']['teacher_audio']) != draft_producer_identity(teacher.identity):
        raise ValueError('diagnostic teacher producer differs from frozen A')
    plans = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    segments = selected_segments(args.dataset, plans)
    by_record = {}
    for segment in segments:
        by_record.setdefault(str(segment['record_id']), []).append(segment)
    import json
    records = [json.loads(line) for line in (Path(args.dataset)/'manifest.jsonl').read_text('utf8').splitlines() if line.strip()]
    cabinets = {str(record['record_id']): record['cabinet'] for record in records}
    catalog = CompactPathCatalog.load(args.physical_path_catalog, model.vocabulary)
    samples = fixed_samples(cache)
    output.mkdir(parents=True)
    summary = []
    for index, record in enumerate(samples):
        variant, level, difficulty = record['condition']
        condition = cache.condition(variant, level, difficulty)
        donor = samples[(index+1) % len(samples)]
        donor_condition = cache.condition(*donor['condition'])
        raw = condition['raw'].to(args.device)[None]
        valid = condition['valid'].to(args.device)[None]
        donor_raw = donor_condition['raw'].to(args.device)
        indices = torch.arange(raw.shape[1], device=args.device) % len(donor_raw)
        mismatched = donor_raw[indices][None] if len(samples) > 1 else torch.roll(raw, 4, 1)
        metadata = teacher.manifest['variants'][variant]
        timing = TimingMap.from_dict(metadata['timing'])
        boundary = audio_end(timing, int(metadata['num_samples']))
        source_rows = by_record[str(record['record_id'])]
        chart_end = max(int(row.get('song_end_tick', row['target_end_tick'])) for row in source_rows)
        result = dict(record_id=record['record_id'], variant=variant, level=level,
            source_difficulty_id=difficulty, mismatch_record_id=donor['record_id'], modes={})
        destination = output/str(record['record_id']).replace(':', '-')
        destination.mkdir()
        for name, tokens in (('clean', raw), ('zero', torch.zeros_like(raw)), ('mismatched', mismatched)):
            physical = MiaCodeInferenceState(timing, catalog, model.vocabulary, boundary.onset_tick,
                audio_tail_tick=boundary.tail_tick, audio_end_seconds=boundary.seconds)
            rows, _, windows = generate_baseline_rows(model=model, feature=None, timing=timing,
                level=float(level), cabinet=cabinets[str(record['record_id'])], difficulty_id=int(difficulty),
                seed=20260811, temperature=0., device=torch.device(args.device), physical_state=physical,
                bpm_audio=dict(num_samples=int(metadata['num_samples'])), trace=False,
                teacher_audio_cache=teacher, teacher_audio_variant=variant, style_mode='draft',
                _style_override=DraftSongStyle(tokens, valid, structure=model.draft_generator.structure))
            evidence = dict(events=dict(Counter(row['event'] for row in rows)), window_count=len(windows),
                tail_windows=[dict(start_tick=window['start_tick'], status=window['status'],
                    rows=window['rows'], moments=window['moments']) for window in windows if window['start_tick'] >= chart_end],
                physical=physical.report())
            write_json(destination/(name+'.json'), dict(rows=rows, windows=windows, **evidence))
            result['modes'][name] = evidence
        summary.append(result)
    write_json(output/'summary.json', dict(checkpoint_epoch=payload['epoch']+1,
        evidence='training-set fixed clean/zero/mismatched draft generations; no quality threshold',
        seed=20260811, samples=summary))
    print(json.dumps(dict(output=str(output), samples=len(summary)), allow_nan=False))


if __name__ == '__main__':
    main()
