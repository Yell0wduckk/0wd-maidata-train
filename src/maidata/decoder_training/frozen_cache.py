"""Read accepted frozen public tensors; no network, raw audio, or cache building."""
from collections import defaultdict
import json
from pathlib import Path

import torch

from maidata.draft.windows import SongCoverage
from maidata.training.checkpoint import digest_file
from .cache_schema import P0_SCHEMA, DP_SCHEMA


class FrozenCache:
    def __init__(self, plans, config, binding):
        self.config = config
        self.roots = {k: Path(config['decoder_train_'+k+'_cache']) for k in ('p0', 'dp')}
        self.manifests = {k: json.loads((root/'manifest.json').read_text('utf-8')) for k, root in self.roots.items()}
        self.binding = {k: digest_file(root/'manifest.json') for k, root in self.roots.items()}
        p0, dp = self.manifests['p0'], self.manifests['dp']
        if (p0['schema'] != P0_SCHEMA or dp['schema'] != DP_SCHEMA
                or not p0['complete'] or not dp['complete']):
            raise ValueError('complete matching P0 and D/P caches are required')
        normalization = {k: binding[k] for k in ('pretrain_chart_cache', 'pretrain_audio_cache')}
        if (p0['producer'] != dp['producer'] or p0['normalization'] != normalization
                or dp['normalization'] != normalization or dp['p0_manifest_sha256'] != self.binding['p0']
                or p0['p0_rate'] != config['fpn_rates'][0] or p0['precision'] != dp['precision']):
            raise ValueError('frozen cache producer, P0 or normalization binding differs')
        self.producer = p0['producer']
        splits = {side: list(plan.groups) for side, plan in zip(('train', 'validation'), plans)}
        if p0['splits'] != splits or dp['splits'] != splits:
            raise ValueError('frozen cache split binding differs')
        expected_plans = {side: {name: hashes[name] for name in ('plan.json', plan.manifest['files']['windows'])}
                          for side, plan, hashes in zip(('train', 'validation'), plans, binding['plans'])}
        if dp['plans'] != expected_plans:
            raise ValueError('frozen D/P window plan binding differs')
        self.windows = defaultdict(dict)
        self.groups = {key: row for plan in plans for key, row in plan.groups.items()}
        for plan in plans:
            for row in plan.windows:
                self.windows[row['audio_variant']][row['window_id']] = (row['chart_id'], row['start_beat'])
        if set(p0['groups']) != set(self.groups) or set(dp['groups']) != set(self.groups):
            raise ValueError('frozen cache audio inventory differs')
        for variant, meta in self.groups.items():
            a, d = p0['groups'][variant], dp['groups'][variant]
            coverage = SongCoverage(**a['coverage'])
            if (a['audio_asset'] != meta['audio_asset'] or a['timing'] != meta['timing']
                    or a['audio_end_seconds'] != meta['audio_end_seconds']
                    or coverage.target_beats != config['target_beats']
                    or list(coverage.starts(include_tail=False)) != meta['encoder_starts_beats']
                    or coverage.first_beat+coverage.draft_beats != meta['draft_end_beat']
                    or a['shape'] != [coverage.audio_windows*coverage.target_beats*config['fpn_rates'][0], config['d_model']]):
                raise ValueError('frozen P0 audio geometry differs')
            mapping = d['chart_to_group']
            n = len(set(mapping.values()))
            if (set(mapping) != set(meta['chart_ids']) or set(mapping.values()) != set(range(n))
                    or d['z_shape'] != [n, coverage.draft_beats, config['draft_latent_dim']]
                    or d['windows'] != len(self.windows[variant])
                    or d['plan_shape'] != [len(self.windows[variant]), 12]):
                raise ValueError('frozen D/P chart or window geometry differs')

    def read(self, variant):
        """Ordinary reads check tensors/ownership; never rehash accepted banks."""
        p0, dp = [torch.load(self.roots[k]/self.manifests[k]['groups'][variant]['path'],
                            map_location='cpu', weights_only=True) for k in ('p0', 'dp')]
        a, d = (self.manifests[k]['groups'][variant] for k in ('p0', 'dp'))
        for payload, schema in ((p0, P0_SCHEMA), (dp, DP_SCHEMA)):
            if (payload['schema'] != schema or payload['audio_variant'] != variant
                    or payload['producer_sha256'] != self.producer['sha256']):
                raise ValueError('frozen bank producer or audio identity differs')
        if p0['coverage'] != a['coverage'] or dp['chart_to_group'] != d['chart_to_group']:
            raise ValueError('frozen bank coverage or chart ownership differs')
        _memory(p0, a['shape'], a['dtype'])
        _memory(dp['z'], d['z_shape'], d['z_dtype'])
        coverage = SongCoverage(**a['coverage'])
        expected = coverage.first_beat+(torch.arange(a['shape'][0]).float()+.5)/self.config['fpn_rates'][0]
        if (not torch.allclose(p0['beats'].float(), expected, atol=1e-5, rtol=1e-6)
                or int(p0['valid'].sum()) != a['valid_tokens']):
            raise ValueError('cached P0 coordinates or valid count differ')
        z_beats = coverage.first_beat+torch.arange(coverage.draft_beats).float()+.5
        if not torch.equal(dp['z']['beats'], z_beats.expand_as(dp['z']['beats'])) or not dp['z']['valid'].all():
            raise ValueError('cached Z must include full beat coverage and valid empty tail')
        p = dp['planner']
        ids = p['window_ids']
        size = len(ids)
        _tensor(p['plan_values'], d['plan_shape'], d['plan_dtype'])
        if (len(set(ids)) != size or set(ids) != set(self.windows[variant]) or len(p['chart_ids']) != size
                or p['groups'].shape != (size,) or p['groups'].dtype != torch.long
                or p['starts'].shape != (size,) or not p['starts'].is_floating_point()
                or p['valid'].shape != (size,) or p['valid'].dtype != torch.bool or not p['valid'].all()):
            raise ValueError('cached Planner window inventory or mask differs')
        for i, key in enumerate(ids):
            chart, start = self.windows[variant][key]
            if (p['chart_ids'][i] != chart or p['starts'][i] != start
                    or p['groups'][i] != dp['chart_to_group'][chart]):
                raise ValueError('cached Planner window/condition ownership differs')
        if len(dp['conditions']) != len(dp['chart_to_group']):
            raise ValueError('cached condition metadata is incomplete')
        dp['window_index'] = {key: i for i, key in enumerate(ids)}
        dp['chart_conditions'] = dict(zip(dp['chart_to_group'], dp['conditions'], strict=True))
        return p0, dp


def _tensor(tensor, shape, dtype):
    if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape or str(tensor.dtype) != dtype
            or not tensor.is_floating_point() or tensor.requires_grad or not torch.isfinite(tensor).all()):
        raise ValueError('invalid frozen tensor shape, dtype or values')


def _memory(memory, shape, dtype):
    _tensor(memory['values'], shape, dtype)
    beats, valid = memory['beats'], memory['valid']
    if (list(beats.shape) != shape[:-1] or not beats.is_floating_point() or beats.requires_grad or not torch.isfinite(beats).all()
            or valid.shape != beats.shape or valid.dtype != torch.bool):
        raise ValueError('invalid frozen memory coordinates or mask')
