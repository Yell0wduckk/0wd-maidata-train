"""Read accepted frozen public tensors; no validation or producer execution."""
import json
from pathlib import Path
import torch


class FrozenCache:
    def __init__(self, plans, config, binding):
        self.roots = {k: Path(config['decoder_train_'+k+'_cache']) for k in ('p0', 'dp')}
        self.manifests = {k: json.loads((root/'manifest.json').read_text('utf-8')) for k, root in self.roots.items()}
        self.binding = {k: str(root/'manifest.json') for k, root in self.roots.items()}
        self.producer = self.manifests['p0']['producer']

    def read(self, variant):
        p0, dp = [torch.load(self.roots[k]/self.manifests[k]['groups'][variant]['path'],
                            map_location='cpu', weights_only=True) for k in ('p0', 'dp')]
        dp['window_index'] = {key: i for i, key in enumerate(dp['planner']['window_ids'])}
        return p0, dp
