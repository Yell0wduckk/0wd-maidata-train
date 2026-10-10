"""Route-71 and spatial transforms; no model or Physical imports."""
from functools import lru_cache
from importlib.resources import files
import json
import re

from .types import ChartError, BUTTON_PADS, TOUCH_PADS


@lru_cache(maxsize=1)
def catalog():
    return tuple(json.loads(files('maidata.chart').joinpath('assets/slide_variant_catalog.json').read_text('utf-8'))['variants'])


def path_for(start_pad, route_id):
    if start_pad not in BUTTON_PADS or type(route_id) is not int or not 0 <= route_id < 71:
        raise ChartError('invalid start/Route-71 pair')
    return catalog()[route_id]['paths_by_start'][int(start_pad[1])-1]


@lru_cache(maxsize=1)
def _inverse():
    return {p: int(v['variant_id']) for v in catalog() for p in v['paths_by_start']}


def route_for(path):
    try:
        return _inverse()[path]
    except KeyError as e:
        raise ChartError(f'unsupported Route-71 path: {path}') from e


def chain_paths(start_pad, routes):
    result = []
    for route in routes:
        path = path_for(start_pad, route)
        result.append(path)
        start_pad = 'A'+path[-1]
    return tuple(result)


def mirror_pad(pad):
    if pad == 'C':
        return pad
    if pad not in set(BUTTON_PADS+TOUCH_PADS):
        raise ChartError(f'invalid pad: {pad}')
    lane = int(pad[1])
    lane = (1-lane) % 8+1 if pad[0] in 'DE' else 9-lane
    return pad[0]+str(lane)


def mirror_route(start_pad, route):
    path = path_for(start_pad, route)
    # Route-71's < and > are absolute rotation directions; p/q and s/z swap.
    path = path.translate(str.maketrans('<>pqsz', '><qpzs'))
    path = re.sub(r'[1-8]', lambda m: str(9-int(m[0])), path)
    return route_for(path)
