"""Read accepted fixed Draft/Plan tensors; no P0 or producer execution."""
import json
from pathlib import Path
import torch
from .cache_schema import DP_SCHEMA, Z_REPRESENTATION
from maidata.draft.spacing import z_stride


def condition_config(config, cache):
    """M consumes the frozen producer's Z width and beat spacing."""
    width = getattr(cache, 'latent_dim', None)
    return dict(config, draft_latent_dim=width, draft_z_stride_beats=cache.z_stride_beats) \
        if config['decoder_z_cross_attention'] and width else dict(config)


class FrozenCache:
    def __init__(self, plans, config, binding):
        self.roots = ({'dp': Path(config['decoder_train_dp_cache'])}
                      if config['decoder_z_cross_attention'] or config['decoder_train_plan_source'] == 'predicted' else {})
        self.manifests = {k: json.loads((root/'manifest.json').read_text('utf-8')) for k, root in self.roots.items()}
        self.binding = {k: str(root/'manifest.json') for k, root in self.roots.items()}
        self.producer = self.manifests.get('dp', {}).get('producer', {})
        self.use_z = config['decoder_z_cross_attention']
        self.predicted_plan = config['decoder_train_plan_source'] == 'predicted'
        self.latent_dim = None
        if config.get('planner_span_beats', 16) != 16:
            raise ValueError('fixed conditions require the original single Plan16')
        if self.roots:
            manifest = self.manifests['dp']
            if (manifest.get('schema') != DP_SCHEMA
                    or manifest.get('z_representation') != Z_REPRESENTATION
                    or type(manifest.get('z_width')) is not int or manifest['z_width'] < 1):
                raise ValueError('fixed conditions require a new split Drafter Z/plan cache; old Z cannot be expanded')
            self.latent_dim = manifest['z_width']
            if (config['decoder_train_plan_source'] == 'predicted'
                    and manifest.get('plan_span_beats', 16) != 16):
                raise ValueError('fixed predicted conditions require the original single Plan16')
            # Existing split-D caches predate spacing and are exactly per-beat.
            self.z_stride_beats = z_stride(manifest.get('z_stride_beats', 1), target_beats=config['target_beats'])
            expected = self.producer.get('recipe', {}).get('draft_z_stride_beats', 1)
            if self.z_stride_beats != expected:
                raise ValueError('fixed Z spacing differs from its producer recipe')

    def read(self, variant):
        if not self.roots:
            return {}
        dp = torch.load(self.roots['dp']/self.manifests['dp']['groups'][variant]['path'],
                        map_location='cpu', weights_only=True)
        if (dp.get('schema') != DP_SCHEMA or dp.get('z_representation') != Z_REPRESENTATION
                or dp.get('z_width') != self.latent_dim
                or dp.get('z_stride_beats', 1) != self.z_stride_beats
                or (self.predicted_plan and dp.get('plan_span_beats', 16) != 16)
                or self.use_z and dp['z']['values'].shape[-1] != self.latent_dim):
            raise ValueError('fixed Draft representation/width differs; old Z cannot be expanded')
        dp['window_index'] = {key: i for i, key in enumerate(dp['planner']['window_ids'])}
        return dp
