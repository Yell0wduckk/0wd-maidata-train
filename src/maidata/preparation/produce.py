"""Explicit offline all-official production CLI. See docs/encoder-audio-cache.md."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .features import (AUX_KEYS, MEL_RECIPE, acoustic18, derive_grid, grid_coordinates,
                       linear_resample, mel_at_centers, read_wav, validate_stats)
from .timing import TempoMap, chart_timing, fields
from .cache import validate_grid


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def token(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


def save_arrays(path, arrays, *, grid=False, mel_for_validation=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    # Validate this newly produced payload once, including storage round-trip.
    with np.load(temp,allow_pickle=False) as archive:
        loaded = {k:archive[k] for k in archive.files}
    if grid:
        validate_grid(dict(loaded,mel_patches=mel_for_validation) if mel_for_validation is not None else loaded)
    for k,v in loaded.items():
        if not np.array_equal(v,arrays[k]) or not np.isfinite(v).all():
            raise ValueError(f'cache round-trip/nonfinite failure {path}:{k}')
    digest = file_hash(temp)
    os.replace(temp,path)
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size,
                fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})


def make_catalog(source, wav_root, stats_path, full_manifest, r):
    stats = read_json(stats_path)
    validate_stats(stats)
    stats_sha = file_hash(stats_path)  # small required fixed-stat identity, not audio hashing
    full = read_json(full_manifest)
    if not full['complete'] or full['stats_sha256']!=stats_sha:
        raise ValueError('full metadata/stat identity mismatch')
    wavs = jsonl(wav_root/'audio-manifest.jsonl')
    canonical = {row['canonical_song_id']:row for row in wavs
                 if row['relative_source_path'].replace('\\','/')==f"data/extracted/{row['canonical_song_id']}/track.mp3"}
    expected_sampling = dict(sample_rate=24000,n_fft=1024,hann_periodic=True,n_mels=128,
                             f_min=30.,f_max=12000.,phase='absolute-k/24',
                             sample_rounding='half-up-float64',boundary='reflect-waveform;outside-center-padding')
    for key,value in expected_sampling.items():
        if full['sampling'].get(key)!=value:
            raise ValueError(f'incompatible old full sampling: {key}')
    old_variants = {}
    for key,value in full['variants'].items():
        identity = value['identity']
        if identity['stats_sha256']!=stats_sha or identity['sampling']!=full['sampling']:
            raise ValueError(f'old full variant metadata mismatch: {key}')
        old_variants[token(dict(asset=identity['source_sha256'],timing=identity['timing']))] = dict(
            variant=key,feature_path=str(full_manifest.parent/value['feature_path']),
            feature_sha256=value['feature_sha256'],coordinates_path=str(full_manifest.parent/value['coordinates_path']),
            coordinates_sha256=value['coordinates_sha256'],shape=value['shape'],
            num_samples=value['num_samples'],original_sha256=identity['original_sha256'],
            producer_manifest=str(full_manifest),producer_variant=key)
    assets, variants, songs, errors = {}, {}, [], []
    for path in sorted(source.glob('*/maidata.txt')):
        song_id = path.parent.name
        try:
            f = fields(path.read_text(encoding='utf-8-sig'))
            wav = canonical[song_id]
            asset = wav['output_sha256']
            duration = wav['samples_per_channel']/wav['source_sample_rate']
            if asset not in assets:
                assets[asset] = dict(wav_path=str(wav_root/wav['output_relative_path']),
                    wav_sha256=asset, source_sha256=wav['source_sha256'],
                    source_asset_id=wav['asset_id'],duration_seconds=duration,
                    source_sample_rate=wav['source_sample_rate'],samples_per_channel=wav['samples_per_channel'],
                    target_p50_lufs=wav['target_p50_lufs'], aliases=[])
            assets[asset]['aliases'].append(song_id)
            song = dict(song_id=song_id,title=f.get('title',''),version=f.get('version',''),
                        source_path=str(path),asset=asset,charts={})
            for key, body in f.items():
                if not key.startswith('inote_') or not body.strip():
                    continue
                difficulty = key.split('_',1)[1]
                timing = chart_timing(body, f.get('wholebpm') or 120,
                                     f.get('first_'+difficulty,f.get('first',0)) or 0)
                identity = dict(asset=asset,timing=timing,samples_per_beat=r)
                variant = token(identity)
                if variant not in variants:
                    variants[variant] = dict(**identity, charts=[])
                variants[variant]['charts'].append(f'{song_id}:{difficulty}')
                song['charts'][difficulty] = variant
            if not song['charts']:
                if not f.get('wholebpm'):
                    raise ValueError('neither chart timing nor declared wholebpm')
                timing = chart_timing('',f['wholebpm'],f.get('first',0) or 0)
                identity = dict(asset=asset,timing=timing,samples_per_beat=r)
                variant = token(identity)
                variants.setdefault(variant,dict(**identity,charts=[]))
                song['audio_only_variant'] = variant
                song['source_note'] = 'no inote body; timing from declared wholebpm/first only'
            songs.append(song)
        except Exception as exc:
            errors.append(dict(song_id=song_id,error=str(exc)))
    if errors:
        raise ValueError(json.dumps(errors,ensure_ascii=False))
    for variant in variants.values():
        candidate = old_variants.get(token(dict(asset=variant['asset'],timing=variant['timing']))) if r==24 else None
        if candidate and candidate['original_sha256']!=assets[variant['asset']]['source_sha256']:
            candidate = dict(candidate,provenance_note='producer original MP3 metadata differs; actual normalized WAV SHA matches exactly')
        variant['reused_mel'] = candidate
    # Preserve the nine accepted legacy WAV variants rather than relabelling
    # their Mel as current canonical audio or rebuilding the existing full set.
    by_wav = {row['output_sha256']:row for row in wavs}
    existing_old = {v['reused_mel']['variant'] for v in variants.values() if v['reused_mel']}
    songs_by_id = {s['song_id']:s for s in songs}
    for old_id,value in full['variants'].items():
        if old_id in existing_old:
            continue
        asset = value['source_sha256']
        wav = by_wav[asset]
        if asset not in assets:
            assets[asset] = dict(wav_path=str(wav_root/wav['output_relative_path']),
                wav_sha256=asset,source_sha256=wav['source_sha256'],source_asset_id=wav['asset_id'],
                duration_seconds=wav['samples_per_channel']/wav['source_sample_rate'],
                source_sample_rate=wav['source_sample_rate'],samples_per_channel=wav['samples_per_channel'],
                target_p50_lufs=wav['target_p50_lufs'],aliases=value['song_ids'],historical=True)
        identity = dict(asset=asset,timing=value['identity']['timing'],samples_per_beat=r)
        variant_id = token(identity)
        variants[variant_id] = dict(**identity,charts=[],legacy_full_song_ids=value['song_ids'],
            reused_mel=old_variants[token(dict(asset=asset,timing=value['identity']['timing']))] if r==24 else None)
        for song_id in value['song_ids']:
            songs_by_id[song_id].setdefault('legacy_full_variants',[]).append(variant_id)
    return dict(source_root=str(source),wav_manifest=str(wav_root/'audio-manifest.jsonl'),
        fixed_stats_path=str(stats_path),fixed_stats_sha256=stats_sha,
        old_full_manifest=str(full_manifest),old_full_variant_count=len(full['variants']),
        old_full_reuse='direct beat-grid FFT; source/timing/stats identity then k/PCM/valid match',
        pilot_reuse='ineligible: uniform 10ms Mel cannot replace independent beat-grid FFT',
        mel_recipe=MEL_RECIPE,aux_keys=AUX_KEYS,aux_recipe=dict(reference='event-audio-v4-rhythm',
            sample_rate=22050,n_fft=2048,hop_length=441,rate_hz=50,resample='soxr_hq',
            normalization='per-song median/IQR for 15; relative/structure own scales',
            rhythm='whole-song endpoint search; independent endpoints at BPM jumps'),
        samples_per_beat=r,songs=songs,assets=assets,variants=variants,
        historical_wav_records=len(wavs)-len(canonical),stats=stats)


def prepare_job(args):
    output,asset_id,asset,variants,catalog = args
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    root = Path(output)
    base_path = root/'base'/f'{asset_id}.npz'
    base_meta_path = base_path.with_suffix('.json')
    base_key = token(dict(asset=asset_id,aux=catalog['aux_recipe']))
    y,sr = read_wav(asset['wav_path'])
    if sr!=asset['source_sample_rate'] or len(y)!=asset['samples_per_channel']:
        raise ValueError(f'WAV header contradicts accepted metadata: {asset_id}')
    pcm = linear_resample(y,sr,24000)
    if base_meta_path.exists():
        base_meta = read_json(base_meta_path)
        if base_meta['input_key']!=base_key or not base_path.is_file():
            raise ValueError(f'existing base mismatch {asset_id}')
        with np.load(base_path,allow_pickle=False) as archive:
            times,acoustic = archive['times_sec'],archive['acoustic18']
    else:
        times,acoustic = acoustic18(y,sr)
        if acoustic.shape!=(len(times),18):
            raise ValueError(f'base shape mismatch {asset_id}')
        base_meta = dict(input_key=base_key,
                         **save_arrays(base_path,dict(times_sec=times,acoustic18=acoustic)))
        write_json(base_meta_path,base_meta)
    records = []
    for variant_id,variant in variants:
        path = root/'grids'/f'{variant_id}.npz'
        meta_path = path.with_suffix('.json')
        key = token(dict(base=base_key,timing=variant['timing'],r=catalog['samples_per_beat'],
                         mel=catalog['mel_recipe'],stats=catalog['fixed_stats_sha256']))
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta['input_key']!=key or not path.is_file():
                raise ValueError(f'existing variant mismatch {variant_id}')
        else:
            tempo = TempoMap(variant['timing'])
            coordinates = grid_coordinates(tempo,asset['duration_seconds'],len(pcm),catalog['samples_per_beat'])
            mel,reused,new,unmatched,indices = grid_mel(pcm,coordinates,variant.get('reused_mel'),catalog['stats'])
            arrays = derive_grid(mel,coordinates,times,acoustic,tempo,asset['duration_seconds'])
            reference = variant.get('reused_mel')
            if reference:
                # Store only a view into the accepted producer's Mel plus any
                # actual missing positions; never copy/rebuild its complete Mel.
                arrays.pop('mel_patches')
                arrays['mel_source_indices'] = indices
                missing = coordinates['sample_valid'].reshape(-1)&(indices.reshape(-1)<0)
                arrays['mel_new_flat_indices'] = np.flatnonzero(missing)
                arrays['mel_new_values'] = mel.reshape(-1,128)[missing]
            meta = dict(input_key=key,variant=variant_id,asset=asset_id,reused_mel_positions=reused,
                        new_mel_positions=new,unmatched_old_positions=unmatched,
                        mel_reference=reference,
                        **save_arrays(path,arrays,grid=True,mel_for_validation=mel if reference else None))
            write_json(meta_path,meta)
        records.append(meta)
    return dict(asset=asset_id,base=base_meta,grids=records,seconds=time.perf_counter()-started)


def grid_mel(pcm, coordinates, candidate, stats):
    shape = coordinates['grid_k'].shape
    centers = coordinates['sample_indices'].reshape(-1)
    valid = coordinates['sample_valid'].reshape(-1)
    mel = np.zeros((len(centers),128),np.float32)
    reused = np.zeros(len(centers),bool)
    source_indices = np.full(len(centers),-1,np.int64)
    unmatched = 0
    if candidate:
        old = np.load(candidate['feature_path'],mmap_mode='r',allow_pickle=False)
        with np.load(candidate['coordinates_path'],allow_pickle=False) as archive:
            c = {k:archive[k] for k in archive.files}
        if (list(old.shape)!=candidate['shape'] or old.shape[0]!=128 or old.dtype!=np.float32
                or candidate['num_samples']!=len(pcm)
                or any(c[k].shape!=(old.shape[1],) for k in ('grid_k','seconds','samples','valid'))
                or c['valid'].dtype!=np.bool_ or np.any(np.diff(c['grid_k'])<=0)):
            raise ValueError('malformed or incompatible reused full Mel coordinates')
        k = coordinates['grid_k'].reshape(-1)
        indices = np.searchsorted(c['grid_k'],k)
        inside = indices<len(c['grid_k'])
        safe = np.minimum(indices,len(c['grid_k'])-1)
        match = inside & (c['grid_k'][safe]==k)
        exact = match & (c['samples'][safe]==centers) & (c['valid'][safe]==valid)
        exact &= np.abs(c['seconds'][safe]-coordinates['sample_times_sec'].reshape(-1))<1e-9
        reused = exact & valid
        values = np.asarray(old[:,safe[reused]].T)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite reused full Mel')
        mel[reused] = values
        source_indices[reused] = safe[reused]
        unmatched = int(np.count_nonzero(match & valid & ~exact))
    missing = valid & ~reused
    mel[missing] = mel_at_centers(pcm,centers[missing],np.ones(missing.sum(),bool),stats)
    return mel.reshape(*shape,128),int(reused.sum()),int(missing.sum()),unmatched,source_indices.reshape(shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('catalog','build'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--wav-root',type=Path)
    parser.add_argument('--stats',type=Path)
    parser.add_argument('--full-manifest',type=Path)
    parser.add_argument('--samples-per-beat',type=int,default=24)
    parser.add_argument('--workers',type=int,default=4,help='offline production processes, unrelated to training DataLoader')
    parser.add_argument('--limit',type=int,help='small real-input production test')
    parser.add_argument('--song-id',action='append',help='small real-input test, includes selected historical audio variants')
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.command=='catalog':
        if (root/'base').exists() or (root/'grids').exists():
            raise ValueError('production exists; refusing to replace its catalog')
        catalog = make_catalog(args.source.resolve(),args.wav_root.resolve(),args.stats.resolve(),args.full_manifest.resolve(),args.samples_per_beat)
        write_json(root/'catalog.json',catalog)
        duration = sum(a['duration_seconds'] for a in catalog['assets'].values())
        grid_samples = 0
        for v in catalog['variants'].values():
            duration_v = catalog['assets'][v['asset']]['duration_seconds']
            tempo = TempoMap(v['timing'])
            grid_samples += (int(np.ceil(tempo.to_beats(duration_v)))-int(np.floor(tempo.to_beats(0))))*args.samples_per_beat
        estimate = duration*50*(18*4+8)+grid_samples*(128*4+32*4+25)
        summary = dict(ids=len(catalog['songs']),versions=dict(Counter(s['version'] for s in catalog['songs'])),
            charts=sum(len(s['charts']) for s in catalog['songs']),unique_wav=len(catalog['assets']),
            timing_variants=len(catalog['variants']),mel_reusable_variants=sum(bool(v['reused_mel']) for v in catalog['variants'].values()),
            estimated_uncompressed_bytes=int(estimate),free_bytes=shutil.disk_usage(root).free,
            duration_hours=duration/3600,output=str(root))
        write_json(root/'coverage.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return
    catalog = read_json(root/'catalog.json')
    grouped = {key:[] for key in catalog['assets']}
    for key,value in catalog['variants'].items():
        grouped[value['asset']].append((key,value))
    selected = list(catalog['assets'].items())[:args.limit]
    if args.song_id:
        selected = [(key,value) for key,value in selected if set(value['aliases'])&set(args.song_id)]
    started = time.perf_counter()
    failures,done = [],[]
    common = {key:catalog[key] for key in ('fixed_stats_sha256','mel_recipe','aux_recipe','samples_per_beat','stats')}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_job,(str(root),key,value,grouped[key],common)):key for key,value in selected}
        for future in as_completed(futures):
            key = futures[future]
            try:
                record = future.result()
                done.append(record)
                with (root/'production.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(f"DONE {len(done)}/{len(selected)} {key[:12]} {record['seconds']:.2f}s",flush=True)
            except Exception as exc:
                failures.append(dict(asset=key,error=repr(exc)))
                print(f'ERROR {key} {exc!r}',flush=True)
    summary = dict(complete=not failures and len(selected)==len(catalog['assets']),
        assets=len(done),variants=sum(len(d['grids']) for d in done),failures=failures,
        wall_seconds=time.perf_counter()-started,workers=args.workers,
        mel_reused_variants=sum(g['reused_mel_positions']>0 for d in done for g in d['grids']),
        mel_reused_positions=sum(g['reused_mel_positions'] for d in done for g in d['grids']),
        mel_new_positions=sum(g['new_mel_positions'] for d in done for g in d['grids']),
        mel_unmatched_old_positions=sum(g['unmatched_old_positions'] for d in done for g in d['grids']),
        output_bytes=sum(d['base']['bytes']+sum(g['bytes'] for g in d['grids']) for d in done))
    write_json(root/('sample-summary.json' if args.limit or args.song_id else 'production-summary.json'),summary)
    print(json.dumps(summary),flush=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
