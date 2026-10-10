"""Read the existing Decoder occurrences without resampling or splitting songs."""
from collections import defaultdict
import json
from pathlib import Path

from maidata.sampling.plan import CONFIG_KEYS, make_occurrences


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class DecoderPlan:
    def __init__(self, root, config, *, training):
        self.root, self.training = Path(root), training
        self.manifest = p = read_json(self.root / 'plan.json')
        if (p['schema'] != 'maidata.window-sampling-plan.v1' or p['policy']['views'] != ['original']
                or p['policy'].get('consumer') == 'encoder-draft-pretraining'
                or p['policy']['split'] != ('train' if training else 'validation')
                or p['seed'] != config['decoder_train_seed']):
            raise ValueError('Decoder requires the bound original Decoder split/seed')
        for key in CONFIG_KEYS:
            expected = 1.0 if key == 'complex_rhythm_multiplier' and not training else config[key]
            if p['config'][key] != expected:
                raise ValueError(f'Decoder plan/config differs: {key}')
        if config['target_stride_beats'] != config['target_beats']:
            raise ValueError('Decoder training currently requires disjoint target windows')
        self.windows = [json.loads(s) for s in (self.root / p['files']['windows']).read_text('utf-8').splitlines()]
        self.lookup = {w['window_id']: w for w in self.windows}
        self.groups = {g['audio_variant']: g for g in p['audio_groups']}
        self.charts = {c['chart_id']: c for c in p['charts']}
        if len(self.lookup) != len(self.windows) or len(self.windows) != p['totals']['base_windows']:
            raise ValueError('invalid unique base windows')
        self._epoch, self._rows = None, None
        saved = [json.loads(s) for s in (self.root / p['files']['occurrences']).read_text('utf-8').splitlines()]
        if self.occurrences(p['epoch']) != saved or list(self._rows) != list(self.groups):
            raise ValueError('saved occurrence order differs from bound sampling policy')
        for key, g in self.groups.items():
            rows = self._rows[key]
            if (len(rows) != g['occurrences'] or {r['chart_id'] for r in rows} != set(g['chart_ids'])
                    or rows[0]['sequence_index'] != g['occurrence_start']
                    or rows[-1]['sequence_index']+1 != g['occurrence_stop']):
                raise ValueError('incomplete audio group or changed group boundaries')

    def occurrences(self, epoch):
        if epoch != self._epoch:
            p = self.manifest
            rows = make_occurrences(self.windows, multiplier=p['config']['complex_rhythm_multiplier'],
                                    seed=p['seed'], epoch=epoch)
            self._rows = defaultdict(list)
            for row in rows:
                self._rows[row['audio_variant']].append(row)
            self._epoch = epoch
        return [r for rows in self._rows.values() for r in rows]

    def rows(self, variant, epoch):
        self.occurrences(epoch)
        return [dict(self.lookup[r['window_id']], occurrence=r['occurrence']) for r in self._rows[variant]]

    def batches(self, epoch, config):
        self.occurrences(epoch)
        order = list(self._rows)
        n, accumulate = (config[k] for k in ('decoder_train_audio_groups_per_batch',
                                            'decoder_train_gradient_accumulation_steps'))
        batches = [order[i:i+n] for i in range(0, len(order), n)]
        return [batches[i:i+accumulate] for i in range(0, len(batches), accumulate)]
