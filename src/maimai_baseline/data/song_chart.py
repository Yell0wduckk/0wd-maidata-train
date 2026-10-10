"""Immutable complete GT ownership packs; runtime never assembles chart rows."""
from pathlib import Path
import json
import torch
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_dataset.integrity import file_sha256
from maimai_baseline.style_contract import STYLE_ID

SCHEMA = 'maimai.full_song_chart_inputs.v1'


def pack_rows(rows, end_tick, vocabulary):
    from maimai_dataset.encoding import TIME_FIELDS
    from maimai_dataset.modifiers import LEGACY_FIELDS, ROLES, row_bits
    from maimai_baseline.data.route import Route71Codec
    from maimai_baseline.contract import REMOVED_FIELDS
    from maimai_baseline.data.global_style import pack_reference_rows
    if end_tick < 0:
        raise ValueError('no valid nonnegative song coordinates')
    rows = [{'event': 'TARGET_START'}, *[r for r in rows if r['event'] in ('MOMENT', 'NOTE', 'SLIDE_BRANCH')]]
    ids = {k: [] for k in vocabulary.field_names}
    for row in rows:
        encoded, _ = vocabulary.encode_row({k:v for k,v in row.items() if k not in LEGACY_FIELDS})
        for k in ids: ids[k].append(encoded[k])
    ids = {k:torch.tensor(v, dtype=torch.long) for k,v in ids.items()}
    for role in ROLES: ids['modifier_'+role] = torch.tensor([row_bits(r)[role] for r in rows])
    ids['route'], _ = Route71Codec(vocabulary).encode_mapping(ids, target=False)
    for k in REMOVED_FIELDS: ids.pop(k)
    times = {k:torch.tensor([float(r.get(k, 0)) for r in rows]) for k in TIME_FIELDS}
    active = {k:torch.tensor([k in r for r in rows]) for k in TIME_FIELDS}
    tick = 0; anchors = []
    for row in rows:
        if row['event'] == 'MOMENT':
            tick += int(row['delta_tick'])
            if tick < 0 or tick > end_tick: raise ValueError('GT onset outside audio')
        anchors.append(tick)
    count = end_tick//384+1
    item = dict(input_ids=ids, input_times=times, input_time_active=active,
        row_anchor_tick=torch.tensor(anchors).float(), start_tick=0, end_tick=count*384, song_end_tick=end_tick)
    packed = pack_reference_rows([item], vocabulary, beats_per_token=4, slots=count)
    positions = torch.tensor([a/96 for a,r in zip(anchors, rows) if r['event']=='MOMENT'])
    packed['moment_absolute'] = positions
    packed['moment_delta'] = torch.diff(positions, prepend=torch.zeros(1)) if len(positions) else positions
    return packed


class SongChartCache:
    def __init__(self, root, *, verify=False):
        self.root = Path(root)
        m = json.loads((self.root/'manifest.json').read_text('utf8'))
        from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_SCHEMA, CompactSongChartCache
        if m.get('schema') == COMPACT_SCHEMA:
            self._compact = CompactSongChartCache(root, verify=verify)
            self.manifest = self._compact.manifest
            self.lookup = self._compact.lookup
            return
        self._compact = None
        if (m.get('schema') != SCHEMA or m.get('representation') != STYLE_ID or not m.get('ready')):
            raise ValueError('whole-song GT cache layout mismatch')
        self.manifest = m
        self.lookup = {tuple(c['key']):c for c in m['charts']}
        if len(self.lookup) != len(m['charts']): raise ValueError('duplicate full chart')
        if verify:
            for c in [*m['charts'],*m['audio_files'].values()]:
                if file_sha256(self.root/c['file']) != c['sha256']: raise ValueError('full chart payload mismatch')

    def get(self, key):
        if self._compact is not None:
            return self._compact.get(key)
        meta = self.lookup[tuple(key)]
        return torch.load(self.root/meta['file'], map_location='cpu', weights_only=True)


class SongChartProvider:
    def __init__(self, root, dataset, role, joint=False, song_audio_cache=None, latent_enabled=True, latent_cache=None, teacher_audio_cache=None, draft_cache=None):
        self.latent_enabled = latent_enabled
        self.song_audio_root=song_audio_cache
        self.song_audio=None
        self.joint = joint
        self.root, self.role = str(root), role
        self.lookup = {str(s['segment_id']):str(s['record_id']) for s in dataset.segments}
        self.cache = None
        self.latent_cache_root, self.latent_cache = latent_cache, None
        self.teacher_audio_cache = teacher_audio_cache
        if draft_cache is not None and latent_cache is not None:
            raise ValueError('draft cache and encoder latent cache are mutually exclusive')
        self.draft_cache_root, self.draft_cache = draft_cache, None

    def __call__(self, items):
        from maimai_baseline.time.style import style_geometry
        if self.draft_cache_root is not None:
            from maimai_baseline.data.draft_cache import AudioDraftCache
            if self.draft_cache is None:
                self.draft_cache = AudioDraftCache(self.draft_cache_root, fields=())
            return self.draft_cache.windows(items, self.role)
        if self.cache is None:
            if self.latent_cache_root is not None:
                from maimai_baseline.data.chart_encoder_pretrain import ChartOnlySongCache
                self.cache = ChartOnlySongCache(self.root)
            else:
                self.cache = SongChartCache(self.root)
        keys = [(self.role,self.lookup[str(i['segment_id'])],0 if i.get('spatial_view_id','identity')=='identity' else 1) for i in items]
        unique = list(dict.fromkeys(keys))
        if self.latent_cache_root is not None:
            from maimai_baseline.data.frozen_latent import FrozenLatentCache
            from maimai_baseline.data.song_audio import SongAudioCache
            if self.latent_cache is None:
                self.latent_cache = FrozenLatentCache(self.latent_cache_root, fields=())
            if self.song_audio is None and self.teacher_audio_cache is None:
                self.song_audio = SongAudioCache(self.song_audio_root)
            result = self.latent_cache.windows(items, self.role)
            variants = [self.cache.lookup[key]['variant'] for key in keys]
            starts = torch.tensor([i['start_tick'] for i in items])
            result.update(window_variants=variants, window_groups=torch.tensor([unique.index(k) for k in keys]),
                window_blocks=(starts / 384).long(), window_valid=result['latent_valid'], coverage=1.)
            if self.teacher_audio_cache is None:
                result['song_audio'] = self.song_audio.batch(variants, include_fusion=False)
            return result
        if not self.latent_enabled:
            from maimai_baseline.data.song_audio import SongAudioCache
            if self.song_audio is None:
                self.song_audio = SongAudioCache(self.song_audio_root)
            return dict(song_audio=self.song_audio.batch([self.cache.lookup[k]['variant'] for k in keys], include_fusion=False))
        starts = torch.tensor([i['start_tick'] for i in items])
        ends = torch.tensor([i['end_tick'] for i in items])
        boundaries = torch.tensor([self.cache.lookup[k]['song_end_tick'] for k in keys])
        if bool((starts % 384 != 0).any()): raise ValueError('unaligned four-beat chart window')
        if self.joint:
            variants=list(dict.fromkeys(self.cache.lookup[k]['variant'] for k in unique))
            audio_root = Path(self.cache.manifest.get('source_audio_root', self.cache.root))
            audio={v:torch.load(audio_root/self.cache.manifest['audio_files'][v]['file'],map_location='cpu',weights_only=True) for v in variants}
            return dict(audio=audio,window_variants=[self.cache.lookup[k]['variant'] for k in keys],
                window_starts=starts.long(),chart_variants={str(k):self.cache.lookup[k]['variant'] for k in unique})
        from maimai_baseline.data.song_audio import SongAudioCache
        if self.song_audio is None:self.song_audio=SongAudioCache(self.song_audio_root)
        audio=self.song_audio.batch([self.cache.lookup[k]["variant"] for k in keys], include_fusion=False)
        return dict(song_audio=audio,songs=[self.cache.get(k) for k in unique], window_groups=torch.tensor([unique.index(k) for k in keys]),
            window_blocks=(starts/384).long(), window_valid=style_geometry(starts, ends, boundaries)[1], coverage=1.)
