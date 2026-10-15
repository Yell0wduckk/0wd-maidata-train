"""Export a frozen stage-II Encoder's public P0 for cached Drafter training."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset

from maidata.model import ENCODER_KEYS, build_model, encoder_config
from maidata.decoder_training.p0_cache import collate, save_bank
from maidata.decoder_training.cache_schema import P0_SCHEMA
from maidata.draft.windows import song_coverage, assemble_full_p0
from maidata.preparation.cache import load_grid
from maidata.time import TimingMap
from .__main__ import load_config
from .checkpoint import atomic_save, digest_file, source_identity
from .data import worker_init
from .frozen_p0 import ENCODER_SCHEMA
from .pretrain import autocast
from .runner import write_json


def load_stage_two_encoder(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != 'maidata.decoder-training.v4'
            or payload.get('architecture') != 'maidata.decoder-causal-cont-tick384.v2'):
        raise ValueError('P0 export requires the stage-II v4 normalized-plan CONT v2 checkpoint')
    config = encoder_config(payload['recipe'])
    encoder = build_model(config)
    encoder.load_state_dict({key.removeprefix('encoder.'): value for key, value in payload['model_state'].items()
                             if key.startswith('encoder.')}, strict=True)
    encoder.requires_grad_(False).eval()
    # Hash this selected producer once at new-cache production, never at training startup.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    identity = dict(path=str(path), sha256=digest.hexdigest(), schema=payload['schema'],
        architecture=payload['architecture'], completed_epochs=payload.get('completed_epochs'),
        step=payload.get('step'), source=payload.get('source'), component='encoder')
    return encoder, config, identity


class AudioOnly(Dataset):
    def __init__(self, groups, config):
        self.groups, self.config = list(groups.items()), config

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        variant, meta = self.groups[index]
        c = self.config
        grid = load_grid(Path(c['pretrain_audio_cache']) / 'grids' / f'{variant}.npz',
                         mel_root=c['pretrain_mel_root'], validate_input=False)
        coverage = song_coverage(TimingMap.from_audio_timing(meta['timing'], validate_input=False),
            meta['audio_end_seconds'], first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        return variant, grid, coverage


@torch.inference_mode()
def export(config, checkpoint, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('P0 cache output must be new or empty')
    device = torch.device(config['pretrain_device'])
    if device.type == 'cuda' and (not torch.cuda.is_available()
            or (config['pretrain_precision'] == 'bf16' and not torch.cuda.is_bf16_supported())):
        raise ValueError('requested CUDA/bf16 runtime is unavailable')
    encoder, encoder_config, producer = load_stage_two_encoder(checkpoint)
    groups, splits = {}, {}
    for side in ('train', 'validation'):
        plan = json.loads((Path(config[f'pretrain_{side}_plan']) / 'plan.json').read_text(encoding='utf-8'))
        if plan['config']['target_beats'] != encoder_config['target_beats']:
            raise ValueError('Encoder target-window span differs from the selected plan')
        splits[side] = [g['audio_variant'] for g in plan['audio_groups']]
        for group in plan['audio_groups']:
            variant = group['audio_variant']
            if variant in groups:
                raise ValueError('train/validation audio groups overlap')
            groups[variant] = group
    c = dict(config, **encoder_config)
    encoder.to(device)
    # Accepted input grids are consumed normally, without catalog/payload rescans.
    loader = DataLoader(AudioOnly(groups, c), batch_size=1,
        collate_fn=partial(collate, config=c, validate_input=False), num_workers=c['pretrain_workers'],
        persistent_workers=c['pretrain_workers'] > 0, worker_init_fn=worker_init,
        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(c['pretrain_seed']))
    output.mkdir(parents=True, exist_ok=True)
    # Keep the frozen Encoder with the new cache so completed Drafter checkpoints
    # can embed it and remain usable by normal teacher export/inference loaders.
    snapshot = output / 'encoder.pt'
    frozen_config = json.loads(json.dumps(encoder_config))
    atomic_save(snapshot, dict(schema=ENCODER_SCHEMA, producer_sha256=producer['sha256'],
        encoder_config=frozen_config, model_state={k: v.cpu() for k, v in encoder.state_dict().items()}))
    report = dict(schema=P0_SCHEMA, complete=False, producer=producer, encoder_config=frozen_config,
        encoder=dict(path=snapshot.name, sha256=digest_file(snapshot), bytes=snapshot.stat().st_size),
        p0_rate=c['fpn_rates'][0], precision=c['pretrain_precision'], workers=c['pretrain_workers'],
        source=source_identity(), input_paths={k: c[k] for k in
            ('pretrain_audio_cache', 'pretrain_mel_root', 'pretrain_train_plan', 'pretrain_validation_plan')},
        splits=splits, groups={}, center_ownership='unique target centers; audio padding remains invalid',
        decoder_forward_calls=0, draft_forward_calls=0)
    write_json(output / 'manifest.json', report)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for raw in loader:
        batch = raw.to(device)
        with autocast(device, c):
            local = encoder(batch.songs, batch.windows)
            bank = assemble_full_p0(local.P0, batch.windows.target_start_beats, batch.windows.window_to_song,
                batch.coverages, p0_rate=c['fpn_rates'][0], layout=batch.p0_layout)
        for i, variant in enumerate(batch.variants):
            path = output / 'banks' / f'{variant}.pt'
            row = save_bank(path, bank, i, batch.coverages[i], variant=variant,
                            producer_sha256=producer['sha256'], p0_rate=c['fpn_rates'][0])
            row.update(path='banks/' + path.name, **{k: groups[variant][k] for k in
                       ('audio_asset', 'timing', 'audio_end_seconds')})
            if not report['groups']:
                restored = torch.load(path, map_location='cpu', weights_only=True)
                if not all(torch.equal(restored[k], getattr(bank, k)[i].cpu()) for k in ('values', 'beats', 'valid')):
                    raise ValueError('new P0 bank roundtrip differs')
                report['first_bank_roundtrip_exact'] = True
            report['groups'][variant] = row
        print(json.dumps(dict(completed=len(report['groups']), total=len(groups),
                              seconds=time.perf_counter()-started)), flush=True)
        write_json(output / 'manifest.json', report)
        del batch, bank, local
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        report.update(peak_allocated=torch.cuda.max_memory_allocated(device),
                      peak_reserved=torch.cuda.max_memory_reserved(device))
    report.update(complete=True, seconds=time.perf_counter()-started,
                  bytes=snapshot.stat().st_size + sum(row['bytes'] for row in report['groups'].values()))
    write_json(output / 'manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--decoder-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    export(load_config(args.config), args.decoder_checkpoint, args.output)


if __name__ == '__main__':
    main()
