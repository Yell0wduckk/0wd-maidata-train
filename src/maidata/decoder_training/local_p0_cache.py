"""Export/read frozen P0 per original Decoder window, including full empty tails."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset

from maidata.entry import CONFIG
from maidata.model import build_model, encoder_config
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage
from maidata.preparation.cache import load_grid
from maidata.training.checkpoint import atomic_save, digest_file
from maidata.training.data import worker_init
from maidata.training.frozen_p0 import FrozenP0
from maidata.training.transport import pack_batch
from .data import collate_online_audio
from .recipe import autocast


SCHEMA = 'maidata.frozen-decoder-local-p0.v1'


class LocalP0:
    def __init__(self, root, source):
        self.root = Path(root)
        self.manifest = m = json.loads((self.root/'manifest.json').read_text('utf-8'))
        if (m.get('schema') != SCHEMA or m.get('complete') is not True
                or m['audio_producer'] != source.identity or encoder_config(m['encoder_config']) != source.encoder_config):
            raise ValueError('Decoder requires complete local-window P0 from its frozen Encoder')
        self.identity = dict(schema=SCHEMA, audio_producer=m['audio_producer'], precision=m['precision'],
                             bank_sha256={k:r['sha256'] for k,r in m['groups'].items()})

    def read(self, variant, starts):
        row = self.manifest['groups'][variant]
        bank = torch.load(self.root/row['path'],map_location='cpu',weights_only=True)
        if (bank['schema'] != SCHEMA or bank['audio_variant'] != variant
                or bank['producer_sha256'] != self.manifest['audio_producer']['producer_sha256']
                or bank['starts'] != row['starts']):
            raise ValueError('local P0 window ownership differs')
        lookup = {start:i for i,start in enumerate(bank['starts'])}
        indices = torch.tensor([lookup[start] for start in starts],dtype=torch.long)
        return (BeatMemory(*(bank[key].index_select(0,indices) for key in ('values','beats','valid'))),
                SongCoverage(**row['coverage']))


class LocalAudio(Dataset):
    def __init__(self, groups, config):
        self.groups, self.config = list(groups.items()), config

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        variant, starts = self.groups[index]
        c = self.config
        grid = load_grid(Path(c['decoder_train_audio_cache'])/'grids'/f'{variant}.npz',
                         mel_root=c['decoder_train_mel_root'],validate_input=False)
        return dict(variant=variant,grid=grid,windows=[dict(start=start) for start in starts])


def collate(items, *, config):
    online, _ = collate_online_audio(items,config)
    return pack_batch(online)


def selected_starts(config):
    groups = {}
    for side in ('train','validation'):
        root = Path(config[f'decoder_train_{side}_plan'])
        plan = json.loads((root/'plan.json').read_text('utf-8'))
        local = {g['audio_variant']:set() for g in plan['audio_groups']}
        for line in (root/plan['files']['windows']).read_text('utf-8').splitlines():
            row = json.loads(line)
            local[row['audio_variant']].add(row['start_beat'])
        if set(groups) & set(local):
            raise ValueError('local P0 train/validation audio groups overlap')
        groups.update({k:sorted(v) for k,v in local.items()})
    return groups


@torch.inference_mode()
def export(config, output):
    from .runner import write_json
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('local P0 output must be new or empty')
    source = FrozenP0(config['decoder_train_p0_cache'])
    c = dict(config,**source.encoder_config)
    device = torch.device(c['decoder_train_device'])
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            c['decoder_train_precision'] == 'bf16' and not torch.cuda.is_bf16_supported()):
        raise ValueError('requested CUDA/bf16 local P0 export runtime is unavailable')
    encoder = build_model(c)
    source.load_encoder(encoder)
    for module in encoder.modules():
        if hasattr(module,'validate_inputs'):
            module.validate_inputs = False
    encoder.to(device)
    groups = selected_starts(c)
    loader = DataLoader(LocalAudio(groups,c),batch_size=1,collate_fn=partial(collate,config=c),
        num_workers=c['decoder_train_workers'],persistent_workers=c['decoder_train_workers'] > 0,
        worker_init_fn=worker_init,pin_memory=device.type=='cuda',
        generator=torch.Generator().manual_seed(c['decoder_train_seed']))
    report = dict(schema=SCHEMA,complete=False,audio_producer=source.identity,encoder_config=source.encoder_config,
        precision=c['decoder_train_precision'],workers=c['decoder_train_workers'],audio_scope='window',
        groups={},decoder_forward_calls=0,draft_forward_calls=0)
    write_json(output/'manifest.json',report)
    started = time.perf_counter()
    for raw in loader:
        batch = raw.to(device)
        with autocast(device,c):
            bank = encoder(batch.songs,batch.windows).P0
        variant = batch.audio_keys[0][0]
        starts = [start for _,start in batch.audio_keys]
        payload = dict(schema=SCHEMA,audio_variant=variant,producer_sha256=source.identity['producer_sha256'],
            starts=starts,**{k:getattr(bank,k).cpu() for k in ('values','beats','valid')})
        values,beats,valid = (payload[k] for k in ('values','beats','valid'))
        count = int((c['audio_before_beats']+c['target_beats']+c['audio_after_beats'])*c['fpn_rates'][0])
        expected = torch.tensor(starts)[:,None]-c['audio_before_beats']+(torch.arange(count)+.5)/c['fpn_rates'][0]
        if (values.shape != (len(starts),count,c['d_model']) or beats.shape != values.shape[:2]
                or valid.shape != beats.shape or valid.dtype != torch.bool or not torch.isfinite(values).all()
                or not torch.allclose(beats,expected,atol=1e-5,rtol=1e-6)):
            raise ValueError('new local P0 output shape/coordinates/values differ')
        path = output/'banks'/f'{variant}.pt'
        atomic_save(path,payload)
        if not report['groups']:
            loaded = torch.load(path,map_location='cpu',weights_only=True)
            if not all(torch.equal(loaded[k],payload[k]) for k in ('values','beats','valid')):
                raise ValueError('new local P0 roundtrip differs')
            report['first_bank_roundtrip_exact'] = True
        report['groups'][variant] = dict(path='banks/'+path.name,sha256=digest_file(path),bytes=path.stat().st_size,
            starts=starts,coverage=source.manifest['groups'][variant]['coverage'],shape=list(values.shape))
        write_json(output/'manifest.json',report)
        print(json.dumps(dict(completed=len(report['groups']),total=len(groups),seconds=time.perf_counter()-started)),flush=True)
    report.update(complete=True,seconds=time.perf_counter()-started,
                  bytes=sum(r['bytes'] for r in report['groups'].values()))
    write_json(output/'manifest.json',report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    config = dict(CONFIG)
    values = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if set(values)-set(config):
        parser.error('unknown config keys')
    config.update(values)
    export(config,args.output)


if __name__ == '__main__':
    main()
