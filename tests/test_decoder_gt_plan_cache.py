"""Selected-chart GT statistics are produced offline and read without replay."""
import gzip
import inspect
import json
import math
from unittest.mock import patch

import pytest
import torch

from maidata.chart.codec import token_to_dict
from maidata.chart.types import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.chart_cache.cache import load_chart
from maidata.decoder_training import gt_plan_cache as cache
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets


def record(song='42', difficulty=5):
    tokens = (Moment(0), Note.make('slide', 'A1'), Route(0, 0), Route(1), SlideDuration(192),
              Route(70, 1), SlideDuration(96), Note.make('tap', 'A1', short_hold=1),
              Moment(500), Note.make('touch', 'C', short_touch_hold=1),
              Moment(500), Note.make('hold', 'A5'), HoldDuration(200))
    ticks = [0]*8+[96]*2+[192]*3
    owners = [None, 0, 0, 0, 0, 0, 0, 1, None, 2, None, 3, 3]
    return dict(schema='maidata.full-chart-original.v2', view='original', start_tick=0,
        source=dict(song_id=song, chart_id=f'{song}:{difficulty}', audio_variant=song, difficulty=str(difficulty),
                    source_path='unavailable-original-path'),
        metadata=dict(first_seconds=.125, render_end_beat=dict(num=8, den=1),
            controls=[dict(kind='bpm', bpm=120, position_beats=dict(num=0, den=1)),
                      dict(kind='bpm', bpm=240, position_beats=dict(num=2, den=1))]),
        tokens=[token_to_dict(t) for t in tokens],
        token_map=[dict(token_index=i, tick=t, owner=o) for i, (t, o) in enumerate(zip(ticks, owners))])


def write_chart(root, value):
    key = value['source']['chart_id']
    relative = key.replace(':', '/')+'.json.gz'
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value).encode()))
    return relative


def rows(key, variant):
    return [dict(window_id=f'{key}:{start}', chart_id=key, audio_variant=variant,
                 start_tick=start*96, end_tick=(start+4)*96) for start in (0, 4, 8)]


def test_variant_export_reads_each_chart_once_and_preserves_exact_gt_semantics(tmp_path):
    root, output = tmp_path/'charts', tmp_path/'out'
    charts = {f'42:{d}': write_chart(root, record(difficulty=d)) for d in (5, 6)}
    # Interleave difficulties: row order must remain the plan's order.
    ordered = [row for pair in zip(rows('42:5', '42'), rows('42:6', '42')) for row in pair]
    cache.initialize(dict(decoder_train_chart_cache=str(root), decoder_train_source_root=str(tmp_path/'source')), output)
    with patch('maidata.chart_cache.cache.load_chart', wraps=load_chart) as read, \
         patch('maidata.preparation.cache.load_grid', side_effect=AssertionError('no audio')), \
         patch('maidata.chart_cache.training.load_supports', side_effect=AssertionError('no Physical')), \
         patch('maidata.model.build_model', side_effect=AssertionError('no network')), \
         patch('maidata.training.checkpoint.digest_file', side_effect=AssertionError('no old input hashing')):
        variant, meta = cache.export_variant(('42', ordered, charts))
    assert variant == '42' and meta['shape'] == [6, 12]
    assert read.call_count == 2 and all(c.kwargs == {'validate_input': False} for c in read.call_args_list)
    saved = torch.load(output/meta['path'], weights_only=True)
    assert saved['window_ids'] == [row['window_id'] for row in ordered]
    assert saved['valid'].dtype == torch.bool and saved['valid'].all()
    expected = torch.tensor([math.log1p(4)/math.log1p(64), (4/1.5)/16,
        1/3, 1/4, .5, 1/4, 3/4, 5/6, 0, 0, 0, 1], dtype=torch.float32)
    torch.testing.assert_close(saved['plan_values'][0], expected, rtol=1e-6, atol=1e-7)
    empty = torch.tensor([0.]*8+[1., 0., 0., 0.])
    for i in (2, 3, 4, 5):
        torch.testing.assert_close(saved['plan_values'][i], empty, rtol=0, atol=0)
    # A branched/chained slide remains one NOTE; short holds keep hold semantics.
    assert saved['plan_values'][0, 3] == .25 and saved['plan_values'][0, 6] == .75


def test_missing_end_uses_only_selected_source_inote_and_relocated_path(tmp_path):
    value = record()
    expected = MusicPlanTargets(from_record(value)).window(0, 384)
    del value['metadata']['render_end_beat']
    root, source, output = tmp_path/'charts', tmp_path/'source', tmp_path/'out'
    relative = write_chart(root, value)
    path = source/'42/maidata.txt'
    path.parent.mkdir(parents=True)
    path.write_text('&wholebpm=120\n&inote_5=(120){4}1,2,3,4,,,,,E\n&inote_6={invalid}\n', encoding='utf-8')
    cache.initialize(dict(decoder_train_chart_cache=str(root), decoder_train_source_root=str(source)), output)
    from maidata.draft.source_timing import read_source_end_beat
    with patch('maidata.draft.plan_targets.read_source_end_beat', wraps=read_source_end_beat) as read:
        _, meta = cache.export_variant(('42', rows('42:5', '42')[:1], {'42:5': relative}))
    assert read.call_count == 1 and read.call_args.args == (str(path), '5')
    saved = torch.load(output/meta['path'], weights_only=True)
    torch.testing.assert_close(saved['plan_values'][0, :8], torch.tensor(expected.continuous), rtol=0, atol=0)


def test_reader_loads_only_manifest_then_selected_bank_without_revalidation(tmp_path):
    (tmp_path/'banks').mkdir()
    payload = dict(plan_values=torch.tensor([[0.]*8+[1., 0., 0., 0.]], dtype=torch.float32),
                   valid=torch.ones(1, dtype=torch.bool), window_ids=['42:5:8'])
    torch.save(payload, tmp_path/'banks/42.pt')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(groups={'42': {'path': 'banks/42.pt'}})))
    with patch('torch.load', side_effect=AssertionError('startup reads manifest only')):
        reader = cache.GTPlanCache(tmp_path)
    assert reader.binding == str(tmp_path/'manifest.json')
    with patch('torch.isfinite', side_effect=AssertionError('accepted banks are not rescanned')), \
         patch('maidata.chart_cache.cache.load_chart', side_effect=AssertionError('no training chart decoding')):
        value = reader.read('42')
    assert value['window_index'] == {'42:5:8': 0} and value['window_ids'] == payload['window_ids']
    assert torch.equal(value['plan_values'], payload['plan_values']) and value['valid'].all()


def test_export_selected_train_and_validation_banks_without_other_assets(tmp_path):
    chart_root = tmp_path/'charts'
    config = dict(decoder_train_chart_cache=str(chart_root), decoder_train_source_root=str(tmp_path/'source'), target_beats=4)
    for side, song, difficulties in (('train', '42', (5, 6)), ('validation', '7', (5,))):
        root = tmp_path/side
        root.mkdir()
        charts, windows = [], []
        for difficulty in difficulties:
            value = record(song, difficulty)
            key = value['source']['chart_id']
            charts.append(dict(chart_id=key, chart_cache_path=write_chart(chart_root, value)))
            windows.extend(rows(key, song))
        (root/'plan.json').write_text(json.dumps(dict(charts=charts, audio_groups=[{'audio_variant': song}],
                                                      files={'windows': 'windows.jsonl'})))
        (root/'windows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in windows))
        config['decoder_train_'+side+'_plan'] = str(root)
    # A malformed unselected asset must never be enumerated or decoded.
    (chart_root/'unselected.json.gz').write_bytes(b'invalid')
    output = tmp_path/'gt'
    manifest = cache.export(config, output, workers=1)  # tiny CPU fixture only
    assert manifest['complete'] and manifest['schema'] == cache.SCHEMA
    assert manifest['continuous'] == list(cache.CONTINUOUS) and manifest['categories'] == list(cache.CATEGORIES)
    assert manifest['splits'] == dict(train=['42'], validation=['7'])
    reader = cache.GTPlanCache(output)
    assert len(reader.read('42')['window_ids']) == 6 and len(reader.read('7')['window_ids']) == 3
    assert reader.read('42')['valid'].all() and reader.read('7')['valid'].all()
    with pytest.raises(FileExistsError, match='new or empty'):
        cache.export({}, output)


def test_cli_and_export_keep_workers12_default(tmp_path):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(target_beats=4)))
    output = tmp_path/'out'
    with patch.object(cache, 'export') as export:
        cache.main(['--config', str(path), '--output', str(output)])
    assert export.call_args.kwargs == {'workers': 12}
    assert export.call_args.args[0]['target_beats'] == 4 and export.call_args.args[1] == output
    assert inspect.signature(cache.export).parameters['workers'].default == 12
