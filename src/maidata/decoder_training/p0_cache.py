"""Export frozen stage-I Encoder public P0 banks; never start Decoder training."""
import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader

from maidata.entry import CONFIG
from maidata.audio.layout import prepare_encoder_layout
from maidata.draft.layout import prepare_p0
from maidata.draft.windows import song_coverage, audio_windows, assemble_full_p0
from maidata.preparation.cache import load_grid, song_inputs, window_inputs
from maidata.time import TimingMap
from maidata.training.data import worker_init
from maidata.training.transport import pack_batch
from maidata.training.checkpoint import digest_file
from .checkpoint import load_producer
from .network import DecoderTrainer
from .recipe import autocast
from .runner import write_json


SCHEMA = 'maidata.frozen-p0-bank.v1'


class AudioOnly(Dataset):
    def __init__(self, groups, config):
        self.groups, self.config = groups, config

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        variant, meta = self.groups[index]
        c = self.config
        grid = load_grid(Path(c['decoder_train_audio_cache'])/'grids'/f'{variant}.npz',
                         mel_root=c['decoder_train_mel_root'])
        if grid['mel_patches'].shape[1:] != (c['samples_per_beat'], c['mel_bins']):
            raise ValueError('audio grid/config representation differs')
        coverage = song_coverage(TimingMap.from_audio_timing(meta['timing']), meta['audio_end_seconds'],
                                first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
        if (list(coverage.starts(include_tail=False)) != meta['encoder_starts_beats']
                or coverage.first_beat+coverage.draft_beats != meta['draft_end_beat']):
            raise ValueError('incomplete audio or Draft coverage')
        return variant, grid, coverage


@dataclass
class AudioPacket:
    variants: list
    songs: object
    windows: object
    coverages: list
    p0_layout: dict


def collate(items, *, config):
    variants, grids, coverages = map(list, zip(*items))
    starts, mapping = audio_windows(coverages, device='cpu')
    songs = song_inputs(grids)
    windows = window_inputs(grids, starts.numpy(), mapping.numpy(), target_beats=config['target_beats'],
                            before_beats=config['audio_before_beats'], after_beats=config['audio_after_beats'])
    layout = prepare_encoder_layout(songs.beat_centers, songs.sample_valid.any(-1), windows,
        radii=config['stage_attention_radii'], rates=config['fpn_rates'], before=config['audio_before_beats'],
        context=config['audio_before_beats']+config['target_beats']+config['audio_after_beats'])
    windows.layout = layout
    p0 = prepare_p0(layout['grids'][0], layout['resample'][0]['valid'], starts, mapping, coverages,
                    p0_rate=config['fpn_rates'][0])
    return pack_batch(AudioPacket(variants, songs, windows, coverages, p0))


def save_bank(path, bank, index, coverage, *, variant, producer_sha256, p0_rate):
    count = coverage.audio_windows*coverage.target_beats
    payload = dict(schema=SCHEMA, audio_variant=variant, producer_sha256=producer_sha256,
        coverage=asdict(coverage), values=bank.values[index].detach().cpu(),
        beats=bank.beats[index].detach().cpu(), valid=bank.valid[index].detach().cpu())
    values, beats, valid = (payload[k] for k in ('values','beats','valid'))
    if (values.ndim != 2 or beats.shape != values.shape[:1] or valid.shape != beats.shape
            or valid.dtype != torch.bool or not torch.isfinite(values).all()
            or not torch.isfinite(beats).all() or not (beats[1:] > beats[:-1]).all()
            or len(beats) != count*p0_rate or count <= 0):
        raise ValueError('invalid exported P0 bank')
    expected = coverage.first_beat+(torch.arange(len(beats)).float()+.5)/p0_rate
    if not torch.allclose(beats.float(),expected,atol=1e-5,rtol=1e-6):
        raise ValueError('P0 target centers have wrong coordinates')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return dict(path=path.name, sha256=digest_file(path), bytes=path.stat().st_size,
                shape=list(values.shape), dtype=str(values.dtype), valid_tokens=int(valid.sum()),
                coverage=payload['coverage'])


@torch.no_grad()
def export(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('P0 cache output must be new or empty')
    device = torch.device(config['decoder_train_device'])
    if (device.type != 'cuda' or not torch.cuda.is_available()
            or config['decoder_train_precision'] != 'bf16' or not torch.cuda.is_bf16_supported()):
        raise ValueError('this export requires CUDA bf16')
    manifests = [json.loads((Path(config['decoder_train_'+k])/'plan.json').read_text())
                 for k in ('train_plan','validation_plan')]
    groups, splits = {}, {}
    for side, manifest in zip(('train','validation'), manifests):
        splits[side] = [g['audio_variant'] for g in manifest['audio_groups']]
        for g in manifest['audio_groups']:
            key = g['audio_variant']
            if key in groups:
                raise ValueError('train/validation audio groups overlap')
            groups[key] = g
    binding = {key:{name:digest_file(Path(config[key.replace('pretrain_','decoder_train_')])/name)
                   for name in ('catalog.json','production-summary.json')}
               for key in ('pretrain_chart_cache','pretrain_audio_cache')}
    model = DecoderTrainer(config).eval()
    producer = load_producer(model, config, binding)
    # D/P/M are not run or transferred to GPU. Only public Encoder outputs are exported.
    encoder = model.encoder.to(device).eval()
    generator = torch.Generator().manual_seed(config['decoder_train_seed'])
    workers = config['decoder_train_workers']
    loader = DataLoader(AudioOnly(list(groups.items()),config), batch_size=1,
        collate_fn=partial(collate,config=config),num_workers=workers,persistent_workers=workers>0,
        worker_init_fn=worker_init,pin_memory=True,generator=generator)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(schema=SCHEMA,complete=False,producer=producer,normalization=binding,
        precision=config['decoder_train_precision'],p0_rate=config['fpn_rates'][0],
        source_encoder=config.get('_export_source_commit'),splits=splits,groups={},
        workers=workers,center_ownership='unique target centers; audio padding remains invalid')
    write_json(output/'manifest.json',report)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for raw in loader:
        batch = raw.to(device)
        with autocast(device,config):
            local = encoder(batch.songs,batch.windows)
            bank = assemble_full_p0(local.P0,batch.windows.target_start_beats,batch.windows.window_to_song,
                batch.coverages,p0_rate=config['fpn_rates'][0],layout=batch.p0_layout)
        for i,variant in enumerate(batch.variants):
            path = output/'banks'/f'{variant}.pt'
            row = save_bank(path,bank,i,batch.coverages[i],variant=variant,producer_sha256=producer['sha256'],
                            p0_rate=config['fpn_rates'][0])
            row.update(path='banks/'+path.name,audio_asset=groups[variant]['audio_asset'],
                       timing=groups[variant]['timing'],audio_end_seconds=groups[variant]['audio_end_seconds'])
            if not report['groups']:
                restored = torch.load(path,map_location='cpu',weights_only=True)
                assert all(torch.equal(restored[k],getattr(bank,k)[i].cpu()) for k in ('values','beats','valid'))
                report['first_bank_roundtrip_exact'] = True
            report['groups'][variant] = row
        print(json.dumps(dict(completed=len(report['groups']),total=len(groups),variant=batch.variants[0],
                              seconds=time.perf_counter()-started)),flush=True)
        write_json(output/'manifest.json',report)
        del batch,bank,local
    torch.cuda.synchronize(device)
    report.update(complete=True,seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(device),
                  peak_reserved=torch.cuda.max_memory_reserved(device),bytes=sum(r['bytes'] for r in report['groups'].values()))
    write_json(output/'manifest.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    config = dict(CONFIG)
    values = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if set(values)-set(config):
        parser.error('unknown config keys')
    config.update(values)
    marker = Path(__file__).resolve().parents[3]/'SOURCE_COMMIT'
    config['_export_source_commit'] = marker.read_text().strip() if marker.exists() else None
    export(config,args.output)


if __name__ == '__main__':
    main()
