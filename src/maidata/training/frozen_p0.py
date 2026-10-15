"""Read accepted public P0 banks without Mel reads or Encoder forwards."""
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.decoder_training.cache_schema import P0_SCHEMA
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage
from maidata.model import ENCODER_KEYS, encoder_config


ENCODER_SCHEMA = 'maidata.frozen-p0-encoder.v1'


class FrozenP0:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        if m.get('schema') != P0_SCHEMA or m.get('complete') is not True or 'encoder' not in m:
            raise ValueError('cached Drafter requires a complete P0 export with its frozen Encoder snapshot')
        if set(m['encoder_config']) - {'fpn_p0_normalization'} != set(ENCODER_KEYS) - {'fpn_p0_normalization'}:
            raise ValueError('P0 Encoder configuration keys differ')
        self.encoder_config = encoder_config(m['encoder_config'])
        if set(self.encoder_config) != set(ENCODER_KEYS) or m['p0_rate'] != self.encoder_config['fpn_rates'][0]:
            raise ValueError('P0 Encoder configuration or output rate differs')
        # These are producer identities recorded at export, never recomputed by training.
        self.identity = dict(producer_sha256=m['producer']['sha256'], encoder_sha256=m['encoder']['sha256'],
            precision=m['precision'], p0_rate=m['p0_rate'],
            bank_sha256={key: row['sha256'] for key, row in m['groups'].items()})

    def load_encoder(self, encoder):
        payload = torch.load(self.root / self.manifest['encoder']['path'], map_location='cpu', weights_only=True)
        if (payload.get('schema') != ENCODER_SCHEMA or payload['producer_sha256'] != self.identity['producer_sha256']
                or encoder_config(payload['encoder_config']) != self.encoder_config):
            raise ValueError('P0 Encoder snapshot does not match its producer')
        encoder.load_state_dict(payload['model_state'], strict=True)
        encoder.requires_grad_(False).eval()

    def read(self, variant):
        row = self.manifest['groups'][variant]
        payload = torch.load(self.root / row['path'], map_location='cpu', weights_only=True)
        if (payload.get('schema') != P0_SCHEMA or payload['audio_variant'] != variant
                or payload['producer_sha256'] != self.identity['producer_sha256']
                or payload['coverage'] != row['coverage']):
            raise ValueError('P0 bank producer, audio variant or coverage differs')
        return BeatMemory(*(payload[k] for k in ('values', 'beats', 'valid'))), SongCoverage(**payload['coverage'])

    def check_plan(self, plan):
        # Metadata-only join to the already selected plan; no payload scans/hashes.
        for variant, group in plan.groups.items():
            row = self.manifest['groups'][variant]
            if (row['audio_asset'] != group['audio_asset'] or row['timing'] != group['timing']
                    or row['audio_end_seconds'] != group['audio_end_seconds']
                    or row['coverage']['first_beat'] != group['first_beat']
                    or row['coverage']['target_beats'] != plan.manifest['config']['target_beats']):
                raise ValueError('P0 metadata does not match the selected audio plan')


def resolve_audio_config(config):
    mode = config['pretrain_audio_mode']
    if mode not in ('joint', 'cached'):
        raise ValueError('pretrain_audio_mode must be joint or cached')
    root = config['pretrain_p0_cache']
    if mode == 'joint':
        if root is not None:
            raise ValueError('joint audio updates cannot consume a frozen P0 cache')
        return config
    if not root:
        raise ValueError('cached audio mode requires pretrain_p0_cache')
    cache = FrozenP0(root)
    if config['target_beats'] != cache.encoder_config['target_beats']:
        raise ValueError('P0 target-window span differs from the Drafter plan')
    # Only Encoder architecture comes from its producer. D/R/P retain their own recipe.
    return dict(config, **cache.encoder_config)


def collate_banks(items):
    return BeatMemory(*(pad_sequence([getattr(i['p0'], key) for i in items], batch_first=True)
                        for key in ('values', 'beats', 'valid')))
