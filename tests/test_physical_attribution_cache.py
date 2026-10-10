from copy import deepcopy

import pytest

from maidata.chart import Moment, Note, Route, HoldDuration, SlideDuration, EncodedChart, decode_chart
from maidata.chart.routes import catalog
from maidata.physical import Geometry, PathGeometry
from maidata.chart_cache import produce
from maidata.chart_cache.cache import save_chart
from maidata.chart_cache.training import build_supports, save_supports, load_supports, validate, time_decision
from maidata.chart_cache.rebuild_supports import rebuild_one


@pytest.fixture
def source(monkeypatch):
    geometry = Geometry({p:PathGeometry(p,1.,((0.,1 << (int(p[0])-1)),))
                         for v in catalog() for p in v['paths_by_start']})
    monkeypatch.setattr(produce, 'GEOMETRY', geometry)
    metadata = dict(schema='maimai.chart_ir.v1',tick_rate=180,first_seconds=0,
                    controls=[dict(kind='bpm',bpm=60,position_beats=dict(num=0,den=1))])
    row = dict(chart_id='test:5',song_id='test',difficulty='5',path='charts/test/5.json.gz',
               timing=dict(bpm=[[0,1,60]],first_seconds=0),audio_end_seconds=8)
    def make(tokens):
        return produce.build_record(decode_chart(EncodedChart(tuple(tokens),metadata),geometry),row)
    return geometry,row,make


def test_v2_storage_separates_hard_masks_and_first_responsibility(source,tmp_path):
    geometry,row,make = source
    tokens = [Moment(0),Note.make('tap','A1'),Note.make('hold','A1'),HoldDuration(96)]
    fields = make(tokens)
    result = build_supports(fields,geometry)
    assert result['schema'] == 'maidata.training-physical-original.v2'
    assert result['stages'][1]['pad']['A1']['penalty'] == 2
    duration = result['stages'][2]
    assert time_decision(duration,96).allowed
    assert not time_decision(duration,96,penalty=False).allowed
    path = tmp_path/'supports.json.gz'
    save_supports(path,result)
    assert load_supports(path) == result
    bad = deepcopy(result); bad['schema'] = 'maidata.training-physical-original.v1'
    with pytest.raises(ValueError): validate(bad)
    bad = deepcopy(result); bad['stages'][0]['types'].pop('NOTE')
    with pytest.raises(ValueError): validate(bad)
    bad = deepcopy(result); bad['stages'][1]['pad']['A1']['new_conflicts'] = []
    with pytest.raises(ValueError): validate(bad)


def test_type_support_is_independent_of_gt_next_type_and_duration(source):
    geometry,row,make = source
    prefix = [Moment(0),Note.make('tap','A1'),Note.make('hold','A1'),HoldDuration(96)]
    a = build_supports(make(prefix+[Moment(1000),Note.make('tap','A2')]),geometry)
    b = build_supports(make(prefix+[Note.make('touch','C')]),geometry)
    assert a['stages'][3]['types'] == b['stages'][3]['types']
    prefix = [Moment(0),Note.make('slide','A1'),Route(0,0)]
    a = build_supports(make(prefix+[SlideDuration(96)]),geometry)
    b = build_supports(make(prefix+[SlideDuration(192)]),geometry)
    assert a['stages'][:2] == b['stages'][:2]
    assert a['stages'][2]['types'] == b['stages'][2]['types']
    assert a['stages'][2]['duration'] == b['stages'][2]['duration']


def test_rebuild_only_supports_leaves_field_bytes_and_owner_maps_unchanged(source,tmp_path):
    geometry,row,make = source
    fields = make([Moment(0),Note.make('tap','A1'),Note.make('hold','A1'),HoldDuration(96)])
    root, output = tmp_path/'fields', tmp_path/'physical'
    field_path = root/row['path']
    storage = save_chart(field_path,fields)
    original = field_path.read_bytes()
    result = rebuild_one(row,dict(status='success',**storage),root,output,'test-recipe')
    assert result['status'] == 'success'
    assert field_path.read_bytes() == original
    assert load_supports(output/row['path'])['source_sha256'] == storage['sha256']
    assert rebuild_one(row,dict(status='success',**storage),root,output,'test-recipe')['resumed']
    with pytest.raises(ValueError):
        rebuild_one(row,dict(status='success',**storage),root,output,'different-recipe')
