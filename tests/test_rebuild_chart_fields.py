import gzip
import json

import pytest

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.chart.codec import EncodedChart, decode_chart
from maidata.chart.routes import catalog
from maidata.chart_cache import produce
from maidata.chart_cache.cache import load_chart
from maidata.chart_cache.rebuild_fields import one
from maidata.physical import Geometry, PathGeometry


@pytest.mark.parametrize('legacy_success', [True, False])
def test_rebuild_fields_from_accepted_cache_or_wifi_failure(tmp_path, monkeypatch, legacy_success):
    geometry = Geometry({p: PathGeometry(p, 1., ((0., 1), (.5, 2)))
                         for r in catalog() for p in r['paths_by_start']})
    monkeypatch.setattr(produce, 'GEOMETRY', geometry)
    metadata = dict(schema='maimai.chart_ir.v1', tick_rate=180, first_seconds=0.,
                    controls=[dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1))])
    tokens = (Moment(0), Note.make('slide', 'A1'), Route(70, 0), SlideDuration(96))
    if not legacy_success:
        tokens += (Route(0, 1), SlideDuration(96))
    ir = decode_chart(EncodedChart(tokens, metadata), geometry)
    row = dict(chart_id='47:6', song_id='47', difficulty='6', path='charts/47/6.json.gz',
               timing=dict(bpm=[[0, 1, 120]], first_seconds=0), audio_end_seconds=10.)
    source, output = tmp_path/'source', tmp_path/'output'
    if legacy_success:
        old = produce.build_record(ir, row)
        old['schema'] = 'maidata.full-chart-original.v1'
        old['tokens'][1]['kind'] = 'wifi'
        old['tokens'].insert(0, dict(type='END', start_tick=0))
        old['token_map'].insert(0, dict(source=None))
        path = source/row['path']
    else:
        old = ir
        path = source/'failures/47-6.ir.json.gz'
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(json.dumps(old).encode()))
    status = dict(input_key='accepted', status='success' if legacy_success else 'unrepresentable')
    result = one(row, status, source, output, 'recipe')
    assert result['status'] == 'success'
    record = load_chart(output/row['path'])
    assert record['schema'] == 'maidata.full-chart-original.v2'
    assert record['tokens'][0] == dict(type='MOMENT', delta_ms=0)
    assert record['tokens'][1]['kind'] == 'slide'
    assert len(record['actions']) == (1 if legacy_success else 2)
    assert 'stages' not in record
    assert one(row, status, source, output, 'recipe')['resumed']


def test_rebuild_fields_preserves_unsupported_record(tmp_path):
    row = dict(chart_id='12:5', song_id='12', difficulty='5', path='charts/12/5.json.gz')
    status = dict(input_key='accepted', status='unrepresentable', reason='absolute duration')
    result = one(row, status, tmp_path/'source', tmp_path/'output', 'recipe')
    assert result['status'] == 'unrepresentable'
    assert result['reason'] == 'absolute duration'
    assert not (tmp_path/'output'/row['path']).exists()
