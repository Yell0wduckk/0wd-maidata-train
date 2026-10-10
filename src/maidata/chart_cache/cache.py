"""Portable JSON cache storage and strict ordinary reads."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from maidata.chart.codec import EncodedChart, token_from_dict
from maidata.chart.grammar import Grammar

SCHEMA = 'maidata.full-chart-original.v2'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def validate(record):
    if record['schema'] != SCHEMA or record['view'] != 'original':
        raise ValueError('incompatible chart cache schema/view')
    encoded = EncodedChart(tuple(token_from_dict(t) for t in record['tokens']), record['metadata'], record['start_tick'])
    grammar = Grammar(encoded.timing)
    completions = []
    for i, token in enumerate(encoded.tokens):
        if i == 0:
            grammar.inject_boundary(token, start_tick=encoded.start_tick)
            c = None
        else:
            c = grammar.accept(token)
        mapping = record['token_map'][i]
        if mapping['token_index'] != i or mapping['tick'] != grammar.state.cursor:
            raise ValueError('token time/index mapping mismatch')
        if not math.isfinite(mapping['second']):
            raise ValueError('nonfinite token time')
        if c is not None:
            completions.append(c)
    grammar.finish()
    if len(completions) != len(record['actions']):
        raise ValueError('completion/action count mismatch')
    for c, row in zip(completions, record['actions']):
        if (c.owner, c.track_index, c.token_index) != (row['owner'], row['track_index'], row['token_index']):
            raise ValueError('action ownership mismatch')
        a = row['action']
        if c.owner != a['owner'] or c.tick != a['tick']:
            raise ValueError('action coordinates mismatch')
        if c.track is not None and len(c.track.routes) != len(a['motions']):
            raise ValueError('segment count mismatch')
        for m in a['motions']:
            if not (math.isfinite(m['start']) and math.isfinite(m['end']) and m['end'] > m['start']):
                raise ValueError('invalid segment time')
    return encoded


def load_chart(path, *, validate_input=True):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        record = json.load(stream)
    if validate_input:
        validate(record)
    return record


def save_chart(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    data = json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    with temporary.open('wb') as stream:
        stream.write(gzip.compress(data, compresslevel=3, mtime=0))
    # This new payload is validated once, before its atomic publication.
    with gzip.open(temporary, 'rt', encoding='utf-8') as stream:
        loaded = json.load(stream)
    validate(loaded)
    if loaded != json.loads(data):
        raise ValueError('chart storage round-trip mismatch')
    sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
    size = temporary.stat().st_size
    os.replace(temporary, path)
    return dict(sha256=sha, bytes=size)
