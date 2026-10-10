"""Terminal fixed-song generations. Training-set evidence, with no pass threshold."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from maimai_baseline.ar_checkpoint import load_checkpoint
from maimai_baseline.ar_contract import TAIL_CONTRACT
from maimai_baseline.data.frozen_latent import FrozenLatentCache
from maimai_baseline.data.frozen_context import FrozenSongContext
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.inference.generation import generate_baseline_rows
from maimai_baseline.model.song_ar import SongStyleContext
from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_baseline.physical.miacode_inference import MiaCodeInferenceState
from maimai_baseline.time.audio_end import audio_end
from maimai_dataset.timing import TimingMap


def fixed_charts(charts, count=3):
    """Stable cached order, identity only, one chart per unique audio."""
    chosen, seen = [], set()
    for chart in charts:
        if chart['key'][2] == 0 and chart['audio_key'] not in seen:
            chosen.append(chart); seen.add(chart['audio_key'])
            if len(chosen) == count:
                break
    if not chosen:
        raise ValueError('no identity fixed samples in selected training data')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'latent-cache', 'song-context-cache', 'route-artifact',
                 'bpm-audio-manifest', 'physical-path-catalog', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--teacher-audio-cache', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('refusing to overwrite fixed-sample diagnostic')
    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, route_artifact=args.route_artifact,
                                    device=device, operation='inference')
    if getattr(model, 'prior_condition_contract', None) != TAIL_CONTRACT:
        raise ValueError('fixed samples require the new B tail contract')
    cache = FrozenLatentCache(args.latent_cache)
    context = FrozenSongContext(args.song_context_cache, cache, teacher_audio_cache=args.teacher_audio_cache)
    teacher = None
    if args.teacher_audio_cache:
        from maimai_baseline.data.teacher_audio import TeacherAudioCache
        teacher = TeacherAudioCache(args.teacher_audio_cache)
    window_root = Path(cache.manifest['decoder_data']['window_cache'])
    window_manifest = json.loads((window_root/'manifest.json').read_text('utf8'))
    records = [json.loads(line) for line in
        (Path(window_manifest['profile'])/'manifest.jsonl').read_text('utf8').splitlines() if line.strip()]
    cabinets = {str(row['record_id']): row['cabinet'] for row in records}
    catalog = CompactPathCatalog.load(args.physical_path_catalog, model.vocabulary)
    bpm = json.loads(args.bpm_audio_manifest.read_text('utf-8-sig'))
    output.mkdir(parents=True)
    summary = []
    for chart in fixed_charts(context.charts):
        info = bpm['variants'][chart['audio_key']]
        root = args.bpm_audio_manifest.parent
        feature = np.load(root/info['feature_path'], mmap_mode='r', allow_pickle=False)
        with np.load(root/info['coordinates_path'], allow_pickle=False) as archive:
            coordinates = {k: archive[k].copy() for k in archive.files}
        timing = TimingMap.from_dict(info['timing'])
        boundary = audio_end(timing, int(info['num_samples']))
        mu, valid = cache.full_chart(tuple(chart['key']))
        override = SongStyleContext(mu[None].to(device), valid[None].to(device))
        record = str(chart['key'][1])
        sample_root = output/record.replace(':', '-')
        sample_root.mkdir()
        sample = dict(record=record, audio_key=chart['audio_key'], target_blocks=int(valid.sum()),
            song_end_tick=chart['song_end_tick'], audio_onset_tick=boundary.onset_tick,
            audio_tail_tick=boundary.tail_tick, modes={})
        for name, latent_temperature, mixture_temperature, intervention in (
                ('gt-real-region', 1., 1., override), ('prior-default', 1., 1., None),
                ('prior-zero-temperatures', 0., 0., None)):
            physical = MiaCodeInferenceState(timing, catalog, model.vocabulary,
                boundary.onset_tick, audio_tail_tick=boundary.tail_tick, audio_end_seconds=boundary.seconds)
            rows, _, windows = generate_baseline_rows(model=model, feature=feature, timing=timing,
                level=float(chart['level']), cabinet=cabinets[record],
                difficulty_id=int(chart['difficulty_id']), seed=20260811, temperature=0.,
                device=device, physical_state=physical,
                bpm_audio=dict(coordinates=coordinates, num_samples=int(info['num_samples'])),
                style_mode='ar', latent_temperature=latent_temperature,
                mixture_temperature=mixture_temperature, trace=False, _style_override=intervention,
                teacher_audio_cache=teacher, teacher_audio_variant=chart['audio_key'] if teacher else None)
            # Rows are public fields and times; windows include the actual EOS status.
            tail = [w for w in windows if w['start_tick'] >= chart['song_end_tick']]
            observed = dict(events=dict(Counter(r['event'] for r in rows)), windows=len(windows),
                tail_windows=[dict(start_tick=w['start_tick'], status=w['status'], rows=w['rows'],
                    moments=w['moments']) for w in tail],
                physical=physical.report())
            write_json(sample_root/(name+'.json'), dict(rows=rows, windows=windows, **observed))
            sample['modes'][name] = observed
        summary.append(sample)
    write_json(output/'summary.json', dict(checkpoint_epoch=payload['epoch']+1, seed=20260811,
        evidence='training-set fixed generations; no quality threshold',
        gt_intervention='Decoder only on real cached mu; missing tail uses the same prior sample', samples=summary))
    print(json.dumps(dict(output=str(output), samples=len(summary)), allow_nan=False))


if __name__ == '__main__':
    main()
