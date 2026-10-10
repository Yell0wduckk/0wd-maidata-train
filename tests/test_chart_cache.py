from copy import deepcopy
from dataclasses import replace

import pytest

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.chart.codec import decode_chart, EncodedChart
from maidata.chart.types import ChartError
from maidata.physical import Geometry, PathGeometry, PhysicalState, PhysicalError
from maidata.chart.routes import catalog
from maidata.time import TimingMap
from maidata.chart_cache import produce
from maidata.chart_cache.cache import load_chart, save_chart


@pytest.fixture
def geometry(monkeypatch):
    g = Geometry({p: PathGeometry(p, 1., ((0., 1), (.5, 2)))
                  for v in catalog() for p in v['paths_by_start']})
    monkeypatch.setattr(produce, 'GEOMETRY', g)
    return g


def test_teacher_retains_violation_and_candidate_still_rejects(geometry):
    s = PhysicalState(TimingMap([(0, 120)]), geometry, audio_end_seconds=100)
    s.inject_boundary(Moment(0), start_tick=0)
    for t in (Note.make('tap', 'A1'),):
        s.accept_source(t)
    saved = s.snapshot()
    with pytest.raises(PhysicalError):
        s.accept(Note.make('tap', 'A1'))
    assert s.snapshot() == saved
    c, d = s.accept_source(Note.make('tap', 'A1'))
    assert c.owner == 1 and 'overlap' in d.physical
    assert len(s.actions) == 2
    s.restore(saved)
    with pytest.raises(ChartError):
        s.accept_source(SlideDuration(96))
    assert s.snapshot() == saved


def test_cache_chain_branch_ownership_roundtrip(tmp_path, geometry):
    tokens = (Moment(0), Note.make('slide', 'A1'), Route(0, 0), Route(0),
              SlideDuration(192), Route(1, 1), SlideDuration(96),
              Moment(1000), Note.make('tap', 'A8', short_hold=1))
    metadata = dict(schema='maimai.chart_ir.v1', tick_rate=180, first_seconds=0.,
                    controls=[dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1))])
    ir = decode_chart(EncodedChart(tokens, metadata), geometry)
    row = dict(timing=dict(bpm=[[0, 1, 120]], first_seconds=0), audio_end_seconds=100)
    record = produce.build_record(ir, row)
    assert len(record['actions']) == 3 and record['note_count'] == 2
    assert [a['track_index'] for a in record['actions']] == [0, 1, None]
    motions = record['actions'][0]['action']['motions']
    assert motions[0]['end'] == motions[1]['start']
    assert record['decisions'][3]['deferred']
    path = tmp_path/'chart.json.gz'
    save_chart(path, record)
    loaded = load_chart(path)
    assert loaded['tokens'] == record['tokens']
    assert loaded['view'] == 'original'
    bad = deepcopy(record)
    bad['actions'][0]['owner'] = 9
    with pytest.raises(ValueError, match='ownership'):
        save_chart(tmp_path/'bad.json.gz', bad)


def test_teacher_defers_pending_chain_and_preserves_audio_error(geometry):
    s = PhysicalState(TimingMap([(0, 120)]), geometry, audio_end_seconds=.1)
    s.inject_boundary(Moment(0), start_tick=0)
    for t in (Note.make('slide', 'A1'), Route(0, 0), Route(1)):
        c, d = s.accept_source(t)
    assert c is None and d.deferred and not s.actions
    c, d = s.accept_source(SlideDuration(96))
    assert d.basic_errors == ('duration_outside_audio',)
    assert len(s.actions[0].motions) == 2
