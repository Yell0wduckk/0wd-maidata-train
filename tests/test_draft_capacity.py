"""Query/substep cost retains base windows across both selected data splits."""
import json

from maidata.chart.codec import token_to_dict
from maidata.chart.types import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.draft import capacity


def test_capacity_uses_selected_base_windows_and_validation_max(tmp_path, monkeypatch):
    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    selection = write_json('selection.json', {'charts': [
        {'chart_id': key, 'chart_cache_path': key + '.json.gz'} for key in ('a', 'b')]})
    plan = write_json('plan.json', dict(sources=dict(selection=str(selection), chart_cache=str(tmp_path)),
        config=dict(target_beats=4, target_stride_beats=4)))
    split = write_json('split.json', dict(train=dict(chart_ids=['a']), validation=dict(chart_ids=['b'])))
    windows = [dict(window_id=f'{key}:{i}', chart_id=key, start_tick=i * 384, end_tick=(i + 1) * 384)
               for key in ('a', 'b', 'not-selected') for i in range(2)]
    (tmp_path / 'windows.jsonl').write_text('\n'.join(json.dumps(row) for row in windows), encoding='utf-8')
    # Cost measurement reads base windows, preserving existing occurrences.
    (tmp_path / 'occurrences.jsonl').write_text('not used', encoding='utf-8')

    def record(key, tokens, ticks, owners):
        return dict(source=dict(chart_id=key), schema='maidata.full-chart-original.v2', view='original',
            start_tick=0, metadata=dict(first_seconds=0, render_end_beat=dict(num=4, den=1),
                controls=[dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1))]),
            tokens=[token_to_dict(t) for t in tokens],
            token_map=[dict(token_index=i, tick=t, owner=o) for i, (t, o) in enumerate(zip(ticks, owners))])

    records = {
        'a.json.gz': record('a', [Moment(0), Note.make('hold', 'A1'), HoldDuration(200)],
                            [0] * 3, [None, 0, 0]),
        'b.json.gz': record('b', [Moment(0), Note.make('tap', 'A1'), Moment(500),
                            Note.make('slide', 'A2'), Route(0, 0), Route(1), SlideDuration(192)],
                            [0, 0, 96, 96, 96, 96, 96], [None, 0, None, 1, 1, 1, 1]),
    }
    reads = []
    def load(path):
        reads.append(path.name)
        return records[path.name]
    monkeypatch.setattr(capacity, 'load_chart', load)
    report, rows = capacity.measure_capacity(plan, split)
    assert reads == ['a.json.gz', 'b.json.gz']
    assert report['schema'] == 'maidata.draft-query-cost.v3'
    assert report['max_query_rows'] == 9
    assert report['representation'] == 'beat-local-tick96'
    assert report['splits']['train']['max'] == 3
    assert report['splits']['validation']['max'] == 9
    assert report['all']['windows'] == 4 and report['all']['empty_windows'] == 2
    assert report['max_windows'][0]['split'] == 'validation'
    assert report['max_windows'][0]['query_roles'] == dict(MOMENT=2, NOTE_KIND=2, NOTE_FLAGS=2, ROUTE_ID=2, TRACK_BREAK=1)
    assert [row['query_rows'] for row in rows] == [3, 0, 9, 0]
    assert [row['query_cost'] for row in rows] == [7, 4, 13, 4]
    assert [row['moment_queries'] for row in rows] == [1, 0, 2, 0]
