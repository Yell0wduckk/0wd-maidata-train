"""Preserve occurrence membership, shuffle windows globally, and budget batches."""
from collections import defaultdict
import json
from pathlib import Path
import random

from maidata.sampling.plan import make_occurrences
from .batching import occurrence_batches
from .lengths import load_lengths
from .ordering import history_dropped


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class DecoderPlan:
    def __init__(self, root, config, *, training):
        self.root, self.training = Path(root), training
        self.manifest = p = read_json(self.root / 'plan.json')
        self.windows = [json.loads(s) for s in (self.root / p['files']['windows']).read_text('utf-8').splitlines()]
        self.lookup = {w['window_id']: w for w in self.windows}
        self.groups = {g['audio_variant']: g for g in p['audio_groups']}
        self.charts = {c['chart_id']: c for c in p['charts']}
        self._epoch, self._rows, self._order = None, None, None
        self._lengths = None

    def _original_occurrences(self, epoch):
        p = self.manifest
        return make_occurrences(self.windows, multiplier=p['config']['complex_rhythm_multiplier'],
                                seed=p['seed'], epoch=epoch)

    def occurrences(self, epoch):
        if epoch != self._epoch:
            rows = self._original_occurrences(epoch)
            if self.training:
                random.Random(f'decoder-windows:{self.manifest["seed"]}:{epoch}').shuffle(rows)
            self._rows = defaultdict(list)
            for index, row in enumerate(rows):
                row['sequence_index'] = index
                self._rows[row['audio_variant']].append(row)
            self._order = rows
            self._epoch = epoch
        return self._order

    def rows(self, variant, epoch, indices=None):
        self.occurrences(epoch)
        rows = self._rows[variant] if indices is None else [self._order[i] for i in indices]
        return [dict(self.lookup[r['window_id']], occurrence=r['occurrence'],
                     sequence_index=r['sequence_index']) for r in rows]

    def batches(self, epoch, config):
        rows = self.occurrences(epoch)
        if self._lengths is None:
            self._lengths = load_lengths(self, config)
        lengths = []
        for row in rows:
            target, history = self._lengths[row['window_id']]
            lengths.append(target + (0 if history_dropped(config, self.training, epoch, row) else history))
        centers = config['target_beats'] * config['fpn_rates'][0]
        if int(centers) != centers:
            raise ValueError('center P0 length must be integral')
        audio = {key: len(g['encoder_starts_beats']) * int(centers) for key, g in self.groups.items()}
        return occurrence_batches(rows, audio, lengths, config)
