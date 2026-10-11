"""Teacher-conditioned Physical supports, independent of model probabilities.

The prefix is always strictly before target_token_index. Kind is marginalized
over positions; pad uses teacher kind; route uses the completed preceding
segments; duration uses the entire teacher route. No future duration or action
is used to construct a candidate support.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path

from maidata.chart.codec import token_from_dict
from maidata.physical import PhysicalState
from maidata.physical.attribution import AttributionState
from maidata.physical.conflicts import Conflict
from maidata.physical.state import Decision
from maidata.physical.rules import CATEGORIES, CATEGORY_WEIGHTS, CATEGORY_DIVISOR
from maidata.time import TimingMap

SCHEMA = 'maidata.training-physical-original.v2'
ROUTE_COVERAGE = 'first-route-path-coverage.v1'


def build_supports(record, geometry):
    timing = TimingMap.from_ir(record['metadata'])
    state = PhysicalState(timing,geometry,audio_end_seconds=record['source']['audio_end_seconds'])
    tokens = [token_from_dict(t) for t in record['tokens']]
    state.inject_boundary(tokens[0],start_tick=record['start_tick'])
    attribution = AttributionState(state)
    stages = []
    for i, token in enumerate(tokens[1:],1):
        stage = attribution.stage_and_accept(token)
        stage['target_token_index'] = i
        stages.append(stage)
    stage = attribution.terminal_stage()
    stage['target_token_index'] = len(tokens)
    stages.append(stage)
    return dict(schema=SCHEMA,view='original',chart_id=record['source']['chart_id'],
                route_coverage_attribution=ROUTE_COVERAGE,
                source_schema=record['schema'],categories=list(CATEGORIES),
                category_weights=list(CATEGORY_WEIGHTS),category_divisor=CATEGORY_DIVISOR,
                stages=stages,token_count=len(tokens),boundary_target=False)


def _validate_decision(record):
    for key, witnesses in (('physical', 'conflicts'), ('penalty', 'new_conflicts')):
        value = record[key]
        if type(value) is not int or not 0 <= value < 16:
            raise ValueError('invalid Physical category mask')
        conflicts = frozenset(Conflict.from_record(row) for row in record[witnesses])
        if any(c.category not in CATEGORIES for c in conflicts):
            raise ValueError('unknown conflict category')
        mask = sum(1 << i for i,c in enumerate(CATEGORIES) if any(w.category == c for w in conflicts))
        if value != mask:
            raise ValueError('Physical mask/witness mismatch')
    if not {Conflict.from_record(c) for c in record['new_conflicts']} <= {Conflict.from_record(c) for c in record['conflicts']}:
        raise ValueError('new responsibility without raw witness')
    if any(type(record[k]) is not bool for k in ('deferred', 'penalty_deferred')):
        raise ValueError('invalid deferred state')


def validate(record):
    if record['schema'] != SCHEMA or record['view'] != 'original' or record['boundary_target']:
        raise ValueError('incompatible training Physical contract')
    if len(record['stages']) != record['token_count']:
        raise ValueError('missing teacher support stage')
    for i,s in enumerate(record['stages'],1):
        if s['target_token_index'] != i:
            raise ValueError('support target index mismatch')
        if set(s['types']) != set(s['grammar_types']):
            raise ValueError('missing prefix type support')
        for field in ('types', 'kind', 'pad'):
            for value in s.get(field, {}).values():
                _validate_decision(value)
        for value in s.get('route', []):
            _validate_decision(value)
        for kind in ('duration','moment'):
            if kind not in s:
                continue
            support = s[kind]
            lo, hi = support['domain']
            cursor = lo
            for a,b,d in support['intervals']:
                if a != cursor or b < a or b > hi or not 0 <= d['physical'] < 16:
                    raise ValueError('support interval coverage/mask mismatch')
                _validate_decision(d)
                cursor = b+1
            if cursor != max(lo,hi+1):
                raise ValueError('incomplete time support domain')
    return record


def save_supports(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data = json.dumps(record,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(gzip.compress(data,compresslevel=3,mtime=0))
    loaded = json.loads(gzip.decompress(temporary.read_bytes()))
    validate(loaded)
    if loaded != json.loads(data):
        raise ValueError('training support roundtrip mismatch')
    sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
    size = temporary.stat().st_size
    os.replace(temporary,path)
    return dict(sha256=sha,bytes=size)


def load_supports(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return validate(json.load(stream))


def candidate_decision(record, *, penalty=True):
    """Training defaults to new responsibility; hard masks explicitly use raw."""
    key = 'penalty' if penalty else 'physical'
    deferred = 'penalty_deferred' if penalty else 'deferred'
    return Decision(tuple(record['basic']),frozenset(c for i,c in enumerate(CATEGORIES)
                    if record[key] & (1 << i)),record[deferred])


def type_decision(stage, token_type, *, penalty=True):
    return candidate_decision(stage['types'][token_type], penalty=penalty)


def time_decision(stage, tick, *, window_end=None, censored=False, penalty=True):
    """Apply actual-window closure after reading window-independent supports."""
    if 'moment' in stage and (censored or window_end is not None and tick >= window_end):
        return Decision()
    support = stage.get('duration',stage.get('moment'))
    for lo,hi,record in support['intervals']:
        if lo <= tick <= hi:
            return candidate_decision(record, penalty=penalty)
    return Decision((support['outside_basic'],))


def duration_candidate_cells(stage, timing):
    from maidata.time import duration_cells
    lo,hi = stage['duration']['domain']
    if hi < lo:
        return ()
    kind = 'HOLD_DURATION' if stage['phase'] == 'hold' else 'SLIDE_DURATION'
    return tuple((cell,time_decision(stage,cell.tick)) for cell in
                 duration_cells(kind,lo,hi,timing=timing,onset_tick=stage['cursor_tick']))

