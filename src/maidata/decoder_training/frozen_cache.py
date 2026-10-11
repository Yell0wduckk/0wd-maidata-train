"""Read accepted fixed Draft/Plan tensors; no P0 or producer execution."""
import json
from pathlib import Path
import torch


class FrozenCache:
    def __init__(self, plans, config, binding):
        self.roots = ({'dp': Path(config['decoder_train_dp_cache'])}
                      if config['decoder_z_cross_attention'] or config['decoder_train_plan_source'] == 'predicted' else {})
        self.manifests = {k: json.loads((root/'manifest.json').read_text('utf-8')) for k, root in self.roots.items()}
        self.binding = {k: str(root/'manifest.json') for k, root in self.roots.items()}
        self.producer = self.manifests.get('dp', {}).get('producer', {})

    def read(self, variant):
        if not self.roots:
            return {}
        dp = torch.load(self.roots['dp']/self.manifests['dp']['groups'][variant]['path'],
                        map_location='cpu', weights_only=True)
        dp['window_index'] = {key: i for i, key in enumerate(dp['planner']['window_ids'])}
        return dp
