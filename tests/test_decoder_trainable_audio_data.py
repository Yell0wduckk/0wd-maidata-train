"""Online stage-II audio IO/layouts with fixed Draft/Plan and offline labels."""
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import torch

from maidata.audio.types import ScaleMemory
from maidata.decoder_training import data
from maidata.decoder_training.data import DecoderDataset, WindowBatchSampler, collate_audio, collate_online_audio
from maidata.decoder_training.frozen_cache import FrozenCache
from maidata.draft.windows import assemble_full_p0, song_coverage
from maidata.time import TimingMap
from test_decoder_training import config, encoder_config, training_item


def test_fixed_cache_reads_only_dp_without_a_p0_root(config, tmp_path, monkeypatch):
    item = training_item(0, config, monkeypatch)
    root = tmp_path/'dp'
    root.mkdir()
    torch.save(item['dp'], root/'0.pt')
    producer = {'sha256': 'fixed-stage-one'}
    (root/'manifest.json').write_text(json.dumps(dict(producer=producer, groups={'0': {'path': '0.pt'}})))
    c = dict(config, decoder_train_dp_cache=str(root), decoder_train_p0_cache=None)
    with patch('torch.load', side_effect=AssertionError('startup must not read payloads')):
        cache = FrozenCache([], c, {})
    assert set(cache.roots) == {'dp'} and set(cache.binding) == {'dp'}
    assert cache.producer == producer
    with patch('torch.isfinite', side_effect=AssertionError('no input revalidation')):
        result = cache.read('0')
    assert result['window_index'] == item['dp']['window_index']
    assert torch.equal(result['z']['values'], item['dp']['z']['values'])
    assert not (tmp_path/'p0').exists()


def test_dataset_reads_one_existing_grid_and_dp_per_selected_variant(config, tmp_path, monkeypatch):
    item = training_item(0, config, monkeypatch, remaster=True, training=False)
    root = tmp_path/'audio'
    (root/'grids').mkdir(parents=True)
    np.savez(root/'grids/0.npz', **item['grid'])
    rows = [dict(window_id=w['identity'][0], occurrence=w['identity'][1], chart_id=w['chart_id'],
                 start_beat=w['start'], sequence_index=i) for i, w in enumerate(item['windows'])]
    selected = (4, 0, 1)  # mixed chart + two occurrences of one identical window
    plan = SimpleNamespace(training=True, groups={'0': dict(timing=dict(bpm=[[0, 1, 120]], first_seconds=0.),
        audio_end_seconds=1.2)}, rows=lambda variant, epoch, indices: [rows[i] for i in indices])
    loaded = {}
    for w in item['windows']:
        loaded[w['identity'][0]] = dict(history=w['offset'], tokens=w['tokens'],
                                       next_types=w['next_types'], labels=w['labels'],metadata=dict(logical_lengths=[w['length']-w['offset'],w['offset']]))
    calls = dict(dp=0, windows=[])

    class Cache:
        def read(self, variant):
            calls['dp'] += 1
            assert variant == '0'
            return item['dp']

    class Windows:
        def read(self, identity):
            calls['windows'].append(identity)
            return loaded[identity]

    c = dict(config, decoder_train_audio_cache=str(root), decoder_train_mel_root=None,
             decoder_train_p0_cache=None)
    dataset = DecoderDataset(plan, c, {}, Cache(), Windows())
    with patch.object(data, 'load_grid', wraps=data.load_grid) as read:
        got = dataset[('0', 2, selected)]
    assert read.call_count == calls['dp'] == 1
    assert read.call_args.kwargs['validate_input'] is False
    assert len(calls['windows']) == len({rows[i]['window_id'] for i in selected}) == 2
    assert 'p0' not in got
    assert got['coverage'] == item['coverage']
    assert [w['sequence_index'] for w in got['windows']] == list(selected)
    assert [w['identity'] for w in got['windows']] == [(rows[i]['window_id'], rows[i]['occurrence']) for i in selected]


def test_online_placeholder_encodes_only_selected_windows_and_deduplicates_charts(config, monkeypatch):
    items = [training_item(i, config, monkeypatch, remaster=i == 0) for i in range(2)]
    from maidata.audio.layout import prepare_encoder_layout
    with patch('maidata.model.build_model', side_effect=AssertionError('worker cannot construct E')), \
            patch('maidata.audio.layout.prepare_encoder_layout', wraps=prepare_encoder_layout) as prepare:
        online, placeholder = collate_online_audio(items, config)
    assert prepare.call_count == 1
    assert len(online.songs.mel_patches) == 2
    assert online.windows.window_to_song.tolist() == [0, 0, 1, 1]
    assert online.windows.target_start_beats.tolist() == [0., 3., 0., 3.]
    assert online.audio_keys == [('0', 0), ('0', 3), ('1', 0), ('1', 3)]
    assert placeholder.values.shape == (4, 18, 0)
    layout = online.windows.layout
    torch.testing.assert_close(placeholder.beats, layout['grids'][0], rtol=0, atol=0)
    assert torch.equal(placeholder.valid, layout['resample'][0]['valid'])


def test_interleaved_occurrences_keep_order_and_fixed_dp_under_packed_transport(config, monkeypatch):
    items = [training_item(i, config, monkeypatch, remaster=i == 0) for i in range(2)]
    for i, item in enumerate(items):
        item['windows'] = [dict(w, sequence_index=2*j+i) for j, w in enumerate(item['windows'])]
        item.pop('p0')  # no offline P0 values are needed by the Collator
    expected = sorted([w for item in items for w in item['windows']], key=lambda w: w['sequence_index'])
    packet = collate_audio(items, config=config, packed=True)
    batch = packet.to('cpu')
    assert batch.audio_variants == ['0', '1'] and batch.audio_lengths == [36, 36]
    assert [w['identity'] for w in batch.windows] == [w['identity'] for w in expected]
    assert batch.conditions.audio.values.shape[-1] == 0
    assert 'training_chunks' not in batch.online_audio.windows.layout
    assert len(packet.context.buffers) <= 5
    assert batch.online_audio.songs.mel_patches.data_ptr() == packet.online_audio.songs.mel_patches.data_ptr()
    torch.testing.assert_close(batch.online_audio.songs.mel_patches[0], torch.from_numpy(items[0]['grid']['mel_patches']))
    conditions = batch.conditions
    for i, window in enumerate(expected):
        unique = conditions.occurrence_mapping[i]
        item = items[batch.online_audio.windows.window_to_song[conditions.audio_mapping[unique]]]
        row = item['dp']['window_index'][window['identity'][0]]
        torch.testing.assert_close(conditions.plan[unique], item['dp']['planner']['plan_values'][row], rtol=0, atol=0)
    assert not conditions.decoder_z.values.requires_grad and not conditions.plan.requires_grad
    assert sum(c.labels.counts['main'] for c in batch.chunks) > 0
    for chunk in batch.chunks:
        assert chunk.audio_layout['spans']
        assert chunk.audio_layout['compact_gather'].dtype == torch.long


def test_online_geometry_covers_unequal_songs_and_all_local_windows(config):
    timing = TimingMap([(0, 120)])

    def item(first, end, seed):
        coverage = song_coverage(timing, end/2, first_beat=first, target_beats=config['target_beats'])
        count = int(np.ceil(end))-first
        rate = config['samples_per_beat']
        beats = first+np.arange(count, dtype=np.float32)+.5
        positions = first+np.arange(count*rate).reshape(count, rate)/rate
        valid = (positions >= 0) & (positions < end)
        rng = np.random.default_rng(seed)
        mel = rng.normal(size=(count, rate, config['mel_bins'])).astype(np.float32)
        aux = rng.normal(size=(count, rate, 32)).astype(np.float32)
        mel[~valid], aux[~valid] = 0, 0
        grid = dict(first_beat=np.array(first), beat_centers=beats, sample_valid=valid,
                    mel_patches=mel, aux32_grid=aux, aux32_centers=aux.mean(1))
        return dict(variant=str(seed), grid=grid, coverage=coverage, windows=[dict(start=3. if seed == 0 else 0.)])

    # The long song has four coverage windows, but this batch requests only one.
    items = [item(-3, 8.4, 0), item(0, 2.4, 1)]
    online, placeholder = collate_online_audio(items, config)
    assert online.windows.target_start_beats.tolist() == [3., 0.]
    assert online.windows.window_to_song.tolist() == [0, 1]
    assert placeholder.values.shape == (2, 18, 0)
    assert placeholder.beats[0, 0] == 1 + .5/3
    assert online.songs.sample_valid[1, 3:].count_nonzero() == 0
    assert online.audio_keys == [('0', 3.), ('1', 0.)]


def test_random_window_sampler_deduplicates_variants_without_reordering(config):
    rows = [dict(audio_variant=v, sequence_index=i) for i, v in enumerate(('b', 'a', 'b', 'c', 'a'))]
    plan = SimpleNamespace(manifest={'epoch': 1}, batches=lambda epoch, config: [[rows]])
    sampler = WindowBatchSampler(plan, config)
    assert list(sampler) == [[('b', 1, (0, 2)), ('a', 1, (1, 4)), ('c', 1, (3,))]]


def test_loader_preserves_worker12_and_packed_collation(config):
    plan = SimpleNamespace(manifest={'epoch': 1}, groups={})
    with patch.object(data, 'DataLoader', side_effect=lambda *args, **kwargs: kwargs):
        loader = data.make_loader(plan, dict(config, decoder_train_workers=12), {}, object(), object())
    assert loader['num_workers'] == 12 and loader['persistent_workers']
    assert loader['collate_fn'].keywords['packed'] is True
    assert loader['worker_init_fn'] is data.worker_init
