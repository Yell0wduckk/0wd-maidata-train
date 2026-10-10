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
from maidata.chart.types import Note, Route, Moment, HoldDuration, SlideDuration
from maidata.physical import PhysicalState
from maidata.physical.support import decision_record, duration_support, moment_support, interval_decision
from maidata.physical.rules import CATEGORIES, CATEGORY_WEIGHTS, CATEGORY_DIVISOR
from maidata.time import TimingMap

SCHEMA = 'maidata.training-physical-original.v1'


def build_supports(record, geometry):
    timing = TimingMap.from_ir(record['metadata'])
    state = PhysicalState(timing,geometry,audio_end_seconds=record['source']['audio_end_seconds'])
    tokens = [token_from_dict(t) for t in record['tokens']]
    state.inject_boundary(tokens[0],start_tick=record['start_tick'])
    stages = []
    for i, token in enumerate(tokens[1:],1):
        s = state.grammar.state
        stage = dict(target_token_index=i, cursor_tick=s.cursor, owner=s.owner,
                     phase=s.phase, grammar_types=list(state.grammar.allowed_types()))
        if isinstance(token,Note):
            stage['kind'] = {k:decision_record(d) for k,d in state.query_note_kinds().items()}
            stage['pad_kind'] = token.kind
            stage['pad'] = {p:decision_record(d) for p,d in state.query_note_positions(token.kind).items()}
        elif isinstance(token,Route):
            first = s.phase in ('route','track_done')
            stage['route'] = [decision_record(state.query(Route(r,0 if first else None))) for r in range(71)]
        elif isinstance(token,(HoldDuration,SlideDuration)):
            stage['duration'] = duration_support(state)
        elif isinstance(token,Moment):
            stage['moment'] = moment_support(state)
        stages.append(stage)
        state.accept_source(token)
    # The final real prefix still needs a terminal time query at training time.
    stages.append(dict(target_token_index=len(tokens),cursor_tick=state.grammar.state.cursor,
                       owner=state.grammar.state.owner,phase=state.grammar.state.phase,
                       grammar_types=list(state.grammar.allowed_types()),moment=moment_support(state)))
    return dict(schema=SCHEMA,view='original',chart_id=record['source']['chart_id'],
                source_schema=record['schema'],categories=list(CATEGORIES),
                category_weights=list(CATEGORY_WEIGHTS),category_divisor=CATEGORY_DIVISOR,
                stages=stages,token_count=len(tokens),boundary_target=False)


def validate(record):
    if record['schema'] != SCHEMA or record['view'] != 'original' or record['boundary_target']:
        raise ValueError('incompatible training Physical contract')
    if len(record['stages']) != record['token_count']:
        raise ValueError('missing teacher support stage')
    for i,s in enumerate(record['stages'],1):
        if s['target_token_index'] != i:
            raise ValueError('support target index mismatch')
        for kind in ('duration','moment'):
            if kind not in s:
                continue
            support = s[kind]
            lo, hi = support['domain']
            cursor = lo
            for a,b,d in support['intervals']:
                if a != cursor or b < a or b > hi or not 0 <= d['physical'] < 16:
                    raise ValueError('support interval coverage/mask mismatch')
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


def time_decision(stage, tick, *, window_end=None, censored=False):
    """Clip cached real-onset support using the consumer's actual window.

    Window-closing exact queries and censored tails have no NOTE-completion
    constraint. Probabilities and q are computed by the future training layer.
    """
    from maidata.physical.state import Decision
    if 'moment' in stage and (censored or window_end is not None and tick >= window_end):
        return Decision()
    return interval_decision(stage.get('duration',stage.get('moment')),tick)


def duration_candidate_cells(stage, timing):
    from maidata.time import duration_cells
    lo,hi = stage['duration']['domain']
    if hi < lo:
        return ()
    kind = 'HOLD_DURATION' if stage['phase'] == 'hold' else 'SLIDE_DURATION'
    return tuple((cell,time_decision(stage,cell.tick)) for cell in
                 duration_cells(kind,lo,hi,timing=timing,onset_tick=stage['cursor_tick']))

