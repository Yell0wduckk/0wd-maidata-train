"""Read accepted path resources; never rebuild geometry or depend on old code."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

from maidata.chart.routes import catalog, path_for


def pad_bit(pad):
    if pad == 'C':
        return 1 << 16
    if len(pad) == 2 and pad[0] in 'ABDEK' and pad[1] in '12345678':
        return 1 << ({'A': 0, 'B': 8, 'D': 17, 'E': 25, 'K': 33}[pad[0]]+int(pad[1])-1)
    raise ValueError(f'invalid physical pad: {pad}')


@dataclass(frozen=True)
class PathGeometry:
    key: str
    length: float
    entries: tuple[tuple[float, int], ...]
    exempt: bool = False
    path_id: int = -1


class Geometry:
    def __init__(self, paths):
        self.paths = dict(paths)
        for key, value in self.paths.items():
            if key != value.key or not math.isfinite(value.length) or value.length <= 0:
                raise ValueError(f'invalid geometry: {key}')
            if any(not math.isfinite(p) or not 0 <= p <= 1 or m <= 0 for p, m in value.entries):
                raise ValueError(f'invalid path contacts: {key}')

    @classmethod
    def from_resources(cls, path_catalog_root, slide_data_path):
        root = Path(path_catalog_root)
        raw = json.loads((root/'paths.json').read_text('utf-8'))['paths']
        semantics = {v['key']: v for v in json.loads((root/'runtime-path-semantics.json').read_text('utf-8'))['records']}
        rules = {v['key']: v for v in json.loads((root/'runtime-path-rules.json').read_text('utf-8'))['records']}
        samples = json.loads(Path(slide_data_path).read_text('utf-8'))
        expected = {p for v in catalog() for p in v['paths_by_start']}
        if {p['key'] for p in raw} != expected or set(semantics) != expected or set(rules) != expected:
            raise ValueError('geometry does not cover the Route-71 catalog')
        paths = {}
        path_ids = {p['key']: int(p['path_id']) for p in raw}
        for key in expected:
            # Preserve the accepted Physical WiFi geometry alias, separately
            # from the directed path-coverage relation used by fast-slide.
            alias = key[0]+'-'+key[-1] if 'w' in key else key
            record, rule = semantics[alias], rules[alias]
            contacts = {}
            for contact in record['contacts']:
                contacts.setdefault(contact['pad'], []).append(float(contact['start_proportion']))
            entries, previous = [], 0.
            for i, area in enumerate(record['judge_sequence']):
                values = [v for pad in area for v in contacts.get(pad, ())]
                if i and not values:
                    raise ValueError(f'missing contact time: {key}, {area}')
                p = 0. if not i else min(1., max(previous, min(values)))
                entries.append((p, sum(pad_bit(pad) for pad in set(area))))
                previous = p
            # Only ordinary slides can be chained. The WiFi placeholder length
            # is unused by multi-segment time allocation.
            points = samples['slides'][key]['samples'] if 'w' not in key else ()
            length = max(1., math.fsum(math.hypot(b['x']-a['x'], b['y']-a['y']) for a, b in zip(points, points[1:])))
            paths[key] = PathGeometry(key, length, tuple(entries), bool(rule['is_l'] or rule['is_special_l']), path_ids[key])
        return cls(paths)

    def path(self, start_pad, route_id):
        return self.paths[path_for(start_pad, route_id)]

    @staticmethod
    def covers(previous, following):
        if previous == following:
            return True
        if 'w' not in previous or '-' not in following or previous[0] != following[0]:
            return False
        end = int(previous[-1])
        return int(following[-1]) in ((end-2) % 8+1, end, end % 8+1)
