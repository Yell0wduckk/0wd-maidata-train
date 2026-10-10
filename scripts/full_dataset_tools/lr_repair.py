"""Regenerate mirror rows in the existing source-map order, including branches.

Inverse canonical sorting alone retains reparsed branch ordinals. Instead each
output row is the exact mirror of its mapped canonical source. No Physical
geometry or other spatial-view payload is rebuilt. Publishing a derived manifest
records new bindings and leaves all parent hashes and payloads untouched.
"""
from collections import defaultdict
import copy
import os
from pathlib import Path
import shutil

import numpy as np

from maimai_baseline.data.training_dataset import read, rows, inside, sha, identity, write
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.data.curriculum import MultiViewRouteStore, EpochPlanStore
from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
from maimai_baseline.physical.spatial import transform_mask
from maimai_baseline.physical.catalog import CompactPathCatalog, pad_bit
from maimai_dataset.chart_cache import materialize_chart_vectors, _tensor_at, _encoded
from maimai_dataset.vocabulary import CompactVocabulary
from full_dataset_tools.pipeline import transform_rows, dump, meta

POLICY = 'mirror-lr-canonical-content-in-existing-source-order-v1'
REPORT = 'preparation/reports/lr-repair-verify.json'
PROOF = 'provenance/lr-repair.json'


def mapped_rows(canonical, mapping):
    """Keep complete NOTE/branch groups and markers; reject unsafe permutations."""
    mapping = np.asarray(mapping, dtype=np.int64)
    if not np.array_equal(np.sort(mapping), np.arange(len(canonical))):
        raise ValueError('source mapping must be a full row bijection')
    moments, moment = [], -1
    for row in canonical:
        if row['event'] == 'MOMENT':
            moment += 1
        moments.append(moment)
    group = None
    for position, source in enumerate(mapping):
        if moments[int(source)] != moments[position]:
            raise ValueError('mapping moves a row across moments')
        row = canonical[int(source)]
        if row['event'] not in ('NOTE', 'SLIDE_BRANCH'):
            if source != position:
                raise ValueError('mapping moves a moment/structural marker')
            group = None
        elif row['event'] == 'NOTE':
            group = int(source)
        elif group is None or source != group + 1:
            raise ValueError('mapping splits or reorders a NOTE/branch group')
        else:
            group = int(source)
    mirrored = transform_rows(canonical, 1)
    return [mirrored[int(source)] for source in mapping]


class Inputs:
    def __init__(self, manifest, *, repaired=False):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.data = read(self.path)
        if self.data.get('schema') != 'maimai.training_dataset.v1':
            raise ValueError('expected prepared training dataset.json')
        if not repaired:
            preparation_only = self.data.get('repair_source_only') is True
            if not self.data.get('ready') and not preparation_only:
                raise ValueError('source dataset is not ready')
            # Full runtime stores its accepted report in the sibling reports/.
            receipt = (self.root / self.data['verification']).resolve()
            if not receipt.is_relative_to(self.root.parent):
                raise ValueError('verification report escapes dataset parent')
            accepted = read(receipt)
            if sha(receipt) != self.data['verification_sha256'] or not accepted.get('passed'):
                raise ValueError('source verification binding mismatch')
            if preparation_only and (accepted.get('repair_source_only') is not True
                                     or self.data.get('repair_required') != ['mirror-lr']):
                raise ValueError('repair-only source lacks its bound preparation receipt')
            if not self.data.get('files'):
                items = accepted.get('runtime_files', [])
                self.data['files'] = {m['path']: {'sha256': m['sha256'], 'bytes': m['bytes']} for m in items}
                if not items or len(self.data['files']) != len(items):
                    raise ValueError('accepted report lacks a unique runtime file inventory')
        self.paths = {k: inside(self.root, v) for k, v in self.data['inputs'].items()}
        self.profile = self.paths['dataset']
        self.spatial = self.paths['spatial_view_cache']
        self.sm = read(self.spatial / 'manifest.json')
        MultiViewRouteStore.load(self.spatial, profile_fingerprint=self.sm['profile_fingerprint'],
                                vocabulary_sha256=self.sm['vocabulary_sha256'],
                                selection_sha256=self.sm['selection_sha256'], verify_payloads=False)
        self.source = self.paths['physical_source_rows']
        self.maps = PhysicalSourceRowStore(
            self.source, profile_fingerprint=self.sm['profile_fingerprint'],
            vocabulary_sha256=self.sm['vocabulary_sha256'],
            chart_cache_manifest_sha256=sha(self.paths['chart_cache'] / 'manifest.json'),
            spatial_cache_manifest_sha256=sha(self.spatial / 'manifest.json'))
        self.vocab = CompactVocabulary.from_dict(read(self.profile / 'vocabulary.json'))
        self.codec = Route71Codec(self.vocab)
        self.catalog = CompactPathCatalog.load(self.paths['physical_path_catalog'], self.vocab)
        self.physical = PhysicalBitplaneStore(
            self.paths['physical_bitplane_cache'], profile_fingerprint=self.sm['profile_fingerprint'],
            chart_cache_manifest_sha256=sha(self.paths['chart_cache'] / 'manifest.json'),
            spatial_cache_manifest_sha256=sha(self.spatial / 'manifest.json'), catalog_identity=self.catalog.identity)
        self.physical.validate(full_hash=False, scan_contents=False)
        self.physical_records = {r['record_id']: r for r in self.physical.chart_records}
        expected_inputs = dict(profile_segments=sha(self.profile / 'segments.jsonl'),
                               chart_manifest=sha(self.paths['chart_cache'] / 'manifest.json'),
                               spatial_manifest=sha(self.spatial / 'manifest.json'),
                               bpm_manifest=sha(self.paths['bpm_audio_manifest']))
        for plan in self.data['epoch_plans'].values():
            if EpochPlanStore(inside(self.root, plan), verify_payloads=False).manifest.get('repack_inputs') != expected_inputs:
                raise ValueError('epoch plan input bindings mismatch')
        segments = {s['segment_id']: s for s in rows(self.profile / 'segments.jsonl')}
        cached = sorted(rows(inside(self.spatial, self.sm['segment_index']['path'])), key=lambda s: s['cache_index'])
        if [int(s['cache_index']) for s in cached] != list(range(len(cached))):
            raise ValueError('cache indices are not contiguous')
        self.groups = defaultdict(list)
        for i, item in enumerate(cached):
            seg = segments[item['segment_id']]
            self.groups[seg['record_id']].append((i, seg))
        if set(self.groups) != set(self.maps.indices['mirror-lr']):
            raise ValueError('chart/source-map coverage mismatch')
        if set(self.groups) != set(self.physical_records):
            raise ValueError('chart/Physical coverage mismatch')
        self.offsets = np.load(inside(self.spatial, self.sm['offsets']['path']), allow_pickle=False)

    def charts(self):
        for number, (rid, refs) in enumerate(self.groups.items(), 1):
            encoded = inside(self.profile, refs[0][1]['encoded_path'])
            document = read(encoded)
            if sha(encoded) != self.physical_records[rid]['encoded_sha256']:
                raise ValueError(f'canonical encoded/Physical identity mismatch: {rid}')
            canonical = document['rows']
            mapping = self.maps.rows('mirror-lr', rid).numpy()
            mirrored = mapped_rows(canonical, mapping)
            # A whole group must remain within its original moment/window slices.
            for _, segment in refs:
                for a, b in ((segment['context_row_start'], segment['target_row_start']),
                             (segment['target_row_start'], segment['target_row_end'])):
                    if not np.array_equal(np.sort(mapping[a:b]), np.arange(a, b)):
                        raise ValueError(f'mapping crosses a segment boundary: {rid}')
            # Independent Physical LUT/catalog check, including D/E and every branch.
            for row, source in zip(mirrored, mapping):
                original = canonical[int(source)]
                for field in ('pad', 'start_pad'):
                    if field in original and transform_mask(pad_bit(original[field]), 1) != pad_bit(row[field]):
                        raise ValueError(f'Physical mirror pad mismatch: {rid}/{source}')
                if 'shape' in original:
                    def path(r):
                        route = self.codec._factor_to_variant[(r['shape'], r['end_offset'], r.get('via_offset'))]
                        return self.catalog.path_id(int(r['start_pad'][1]), route)
                    old_path, new_path = path(original), path(row)
                    if int(self.catalog.path_ids_by_view[1, old_path]) != int(self.catalog.path_aliases[new_path]):
                        raise ValueError(f'Physical mirror path mismatch: {rid}/{source}')
            yield rid, refs, {**document, 'rows': mirrored}, mapping
            self.maps._rows.clear()
            if number % 100 == 0:
                print(f'mirror-lr: {number}/{len(self.groups)} charts', flush=True)


def audit(manifest):
    source = Inputs(manifest)
    counts = dict(charts=0, rows=0, mapped_positions=0, branch_rows=0)
    for _, _, doc, mapping in source.charts():
        counts['charts'] += 1
        counts['rows'] += len(mapping)
        counts['mapped_positions'] += int(np.count_nonzero(mapping != np.arange(len(mapping))))
        counts['branch_rows'] += sum(r['event'] == 'SLIDE_BRANCH' for r in doc['rows'])
    return dict(passed=True, policy=POLICY, source_manifest_sha256=sha(source.path),
                operation='mirror canonical content, then gather by existing source rows',
                scope='mapping structure and mirror semantics; not full payload verification', **counts)


def _signed(path, document, key='manifest_identity_sha256'):
    document.pop(key, None)
    excluded = {key, 'created_at', 'finalized_at'} if key == 'state_fingerprint' else {key}
    document[key] = identity({k: v for k, v in document.items() if k not in excluded})
    dump(path, document)


def _relative(root, path):
    return Path(path).relative_to(root).as_posix()


def _prepare_reuse(source, manifest):
    """Load a verified repaired dataset and validate its full-chart overlap."""
    if manifest is None:
        return None, set(), {}, None
    reuse = Inputs(manifest)
    if reuse.path == source.path:
        raise ValueError('reuse dataset must differ from the full source dataset')
    if not reuse.data.get('ready') or reuse.data.get('lr_repair', {}).get('policy') != POLICY:
        raise ValueError('reuse dataset must be a verified LR repair output')
    reused_ids = set(source.groups) & set(reuse.groups)
    if not reused_ids:
        raise ValueError('reuse dataset has no chart intersection with the full source')
    names = set(source.sm['views']['mirror-lr']['arrays'])
    if names != set(reuse.sm['views']['mirror-lr']['arrays']):
        raise ValueError('reuse and full mirror-lr array schemas differ')

    reuse_refs = {}
    for rid in sorted(reused_ids):
        full_refs = source.groups[rid]
        pilot_refs = {segment['segment_id']: (index, segment) for index, segment in reuse.groups[rid]}
        if len(pilot_refs) != len(reuse.groups[rid]):
            raise ValueError(f'reuse segment ids are not unique: {rid}')
        if {segment['segment_id'] for _, segment in full_refs} != set(pilot_refs):
            raise ValueError(f'reuse segment coverage differs: {rid}')
        full_doc = read(inside(source.profile, full_refs[0][1]['encoded_path']))
        pilot_doc = read(inside(reuse.profile, reuse.groups[rid][0][1]['encoded_path']))
        if full_doc['rows'] != pilot_doc['rows']:
            raise ValueError(f'reuse canonical rows differ: {rid}')
        full_mapping = source.maps.rows('mirror-lr', rid).numpy()
        pilot_mapping = reuse.maps.rows('mirror-lr', rid).numpy()
        if not np.array_equal(full_mapping, pilot_mapping):
            raise ValueError(f'reuse source-row mapping differs: {rid}')
        for index, segment in full_refs:
            pilot_index, _ = pilot_refs[segment['segment_id']]
            full_a, full_b = map(int, source.offsets[index:index + 2])
            pilot_a, pilot_b = map(int, reuse.offsets[pilot_index:pilot_index + 2])
            if full_b - full_a != pilot_b - pilot_a:
                raise ValueError(f'reuse segment length differs: {rid}/{segment["segment_id"]}')
        reuse_refs[rid] = pilot_refs

    reuse_info = dict(
        policy='reuse-verified-pilot-mirror-lr-by-record-v1',
        dataset_manifest_sha256=sha(reuse.path),
        reused_record_ids_sha256=identity(sorted(reused_ids)),
        reused_records=len(reused_ids),
        reused_segments=sum(len(source.groups[rid]) for rid in reused_ids),
        rebuilt_records=len(source.groups) - len(reused_ids),
    )
    return reuse, reused_ids, reuse_refs, reuse_info


def build(manifest, output, *, reuse='hardlink', reuse_manifest=None):
    source = Inputs(manifest)
    reuse_source, reused_ids, reuse_refs, reuse_info = _prepare_reuse(source, reuse_manifest)
    output = Path(output).resolve()
    if output == source.root or output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise ValueError('output must be separate from the source dataset')
    if reuse_source is not None:
        if output == reuse_source.root or reuse_source.root.is_relative_to(output):
            raise ValueError('output must not contain the reuse dataset')
        for path in reuse_source.paths.values():
            if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError('output overlaps the reuse dataset')
    if output.exists():
        raise FileExistsError(f'use a new output directory: {output}')
    if reuse not in ('hardlink', 'copy'):
        raise ValueError('unsupported reuse mode')
    changed = {_relative(source.root, source.spatial / 'manifest.json'),
               _relative(source.root, source.source / 'manifest.json'),
               _relative(source.root, source.paths['physical_bitplane_cache'] / 'manifest.json')}
    changed.update(_relative(source.root, inside(source.spatial, m['path']))
                   for m in source.sm['views']['mirror-lr']['arrays'].values())
    changed.update(str(Path(p) / 'manifest.json').replace('\\', '/') for p in source.data['epoch_plans'].values())
    changed.update((REPORT, PROOF, 'dataset.json'))
    output.mkdir(parents=True)
    data = copy.deepcopy(source.data)
    for key in ('build', 'build_sha256', 'build_status', 'verification_status',
                'repair_source_only', 'repair_required'):
        data.pop(key, None)
    data.update(ready=False, verification=REPORT, verification_sha256=None)
    write(output / 'dataset.json', data)
    inherited = {}
    for number, (name, info) in enumerate(source.data['files'].items(), 1):
        if name in changed:
            continue
        origin, target = inside(source.root, name), inside(output, name)
        if not origin.is_file() or origin.stat().st_size != info['bytes']:
            raise ValueError(f'missing/changed source file: {origin}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # All later writes use atomic replacement. Shared payloads are immutable.
        if reuse == 'hardlink':
            os.link(origin, target)
        else:
            shutil.copyfile(origin, target)
        inherited[name] = info
        if number % 5000 == 0:
            print(f'reused {number}/{len(source.data["files"])} files', flush=True)
    spatial = inside(output, data['inputs']['spatial_view_cache'])
    arrays = {}
    reuse_arrays = {}
    sm = copy.deepcopy(source.sm)
    staging = output / 'preparation/lr-profile'
    proof = {}
    try:
        for name, info in sm['views']['mirror-lr']['arrays'].items():
            path = inside(spatial, info['path']); path.parent.mkdir(parents=True, exist_ok=True)
            arrays[name] = np.lib.format.open_memmap(path, mode='w+', dtype=info['dtype'], shape=tuple(info['shape']))
        if reuse_source is not None:
            reuse_arrays = {
                name: np.load(inside(reuse_source.spatial, info['path']), mmap_mode='r', allow_pickle=False)
                for name, info in reuse_source.sm['views']['mirror-lr']['arrays'].items()
            }
        for rid, refs, doc, mapping in source.charts():
            if rid in reused_ids:
                for index, segment in refs:
                    reuse_index, _ = reuse_refs[rid][segment['segment_id']]
                    full_a, full_b = map(int, source.offsets[index:index + 2])
                    reuse_a, reuse_b = map(int, reuse_source.offsets[reuse_index:reuse_index + 2])
                    for name, array in arrays.items():
                        array[full_a:full_b] = reuse_arrays[name][reuse_a:reuse_b]
                proof[rid] = {'rows_sha256': identity(doc['rows']), 'row_count': len(mapping)}
                continue
            path = inside(staging, refs[0][1]['encoded_path'])
            write(path, doc)
            for i, segment in refs:
                values = source.codec.adapt_item(materialize_chart_vectors(profile_root=staging, segment=segment, vocabulary=source.vocab))
                a, b = map(int, source.offsets[i:i + 2])
                for name, array in arrays.items():
                    value = _tensor_at(values, name).numpy()
                    if value.shape != (b - a,):
                        raise ValueError(f'repaired segment length changed: {rid}/{name}')
                    array[a:b] = value
            proof[rid] = {'rows_sha256': identity(doc['rows']), 'row_count': len(mapping)}
            _encoded.cache_clear()
            path.unlink()  # Exact task-created staging file, never a source path.
        for name, array in arrays.items():
            array.flush()
            sm['views']['mirror-lr']['arrays'][name] = meta(spatial, inside(spatial, sm['views']['mirror-lr']['arrays'][name]['path']), array)
    finally:
        for array in arrays.values():
            array._mmap.close()
        for array in reuse_arrays.values():
            mmap = getattr(array, '_mmap', None)
            if mmap is not None:
                mmap.close()
        _encoded.cache_clear()
    binding = dict(policy=POLICY, source_dataset_sha256=sha(source.path),
                   parent_spatial_sha256=sha(source.spatial / 'manifest.json'),
                   parent_source_rows_sha256=sha(source.source / 'manifest.json'),
                   row_proof_sha256=identity(proof), repaired_views=['mirror-lr'],
                   source_map_payloads_unchanged=True, canonical_physical_payloads_unchanged=True)
    if reuse_info is not None:
        binding['reuse'] = reuse_info
    sm['lr_repair'] = binding
    _signed(spatial / 'manifest.json', sm)
    spatial_sha = sha(spatial / 'manifest.json')
    for kind, key in (('physical_source_rows', 'state_fingerprint'),
                      ('physical_bitplane_cache', 'manifest_identity_sha256')):
        old = read(source.paths[kind] / 'manifest.json')
        old.update(spatial_cache_manifest_sha256=spatial_sha, lr_repair=binding)
        _signed(inside(output, data['inputs'][kind]) / 'manifest.json', old, key)
    for plan_path in data['epoch_plans'].values():
        old = read(inside(source.root, plan_path) / 'manifest.json')
        if old.get('repack_inputs', {}).get('spatial_manifest') != binding['parent_spatial_sha256']:
            raise ValueError('source epoch plan is not bound to source spatial cache')
        parent = old['manifest_identity_sha256']
        old['repack_inputs']['spatial_manifest'] = spatial_sha
        old['lr_repair'] = {**binding, 'parent_plan_identity': parent, 'plan_payloads_unchanged': True}
        _signed(inside(output, plan_path) / 'manifest.json', old)
    provenance = dict(binding=binding, charts=proof, inherited_files=inherited)
    if reuse_info is not None:
        provenance['reuse'] = {**reuse_info, 'record_ids': sorted(reused_ids)}
    write(output / PROOF, provenance)
    # Inventory for verification: inherited bytes retain their original hashes.
    files = dict(inherited)
    for name in changed - {REPORT, 'dataset.json'}:
        path = inside(output, name)
        files[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    data.update(files=files, lr_repair=binding,
                preparation_status={'lr-repair-build': 'passed', 'lr-repair-verify': 'pending'})
    write(output / 'dataset.json', data)
    return dict(built=True, ready=False, dataset_manifest=str(output / 'dataset.json'),
                charts=len(proof), next='verify', policy=POLICY)


def verify(manifest):
    manifest = Path(manifest).resolve()
    data = read(manifest)
    if data.get('lr_repair', {}).get('policy') != POLICY:
        raise ValueError('dataset is not an lr repair output')
    data.update(ready=False, verification_sha256=None)
    write(manifest, data)
    source = Inputs(manifest, repaired=True)
    # Check immutable inherited bytes as well as every new array and manifest.
    for number, (name, info) in enumerate(source.data['files'].items(), 1):
        path = inside(source.root, name)
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError(f'payload hash mismatch: {path}')
        if number % 5000 == 0:
            print(f'verified {number}/{len(source.data["files"])} files', flush=True)
    proof = read(source.root / PROOF)
    if proof['binding'] != source.data['lr_repair'] or identity(proof['charts']) != proof['binding']['row_proof_sha256']:
        raise ValueError('repair provenance mismatch')
    arrays = {name: np.load(inside(source.spatial, m['path']), mmap_mode='r', allow_pickle=False)
              for name, m in source.sm['views']['mirror-lr']['arrays'].items()}
    # No training or geometry work: independently rematerialize only Chart tensors.
    staging = source.root / 'preparation/lr-verify-profile'
    count = 0
    try:
        for rid, refs, doc, mapping in source.charts():
            if proof['charts'][rid] != {'rows_sha256': identity(doc['rows']), 'row_count': len(mapping)}:
                raise ValueError(f'row proof mismatch: {rid}')
            path = inside(staging, refs[0][1]['encoded_path']); write(path, doc)
            for i, segment in refs:
                expected = source.codec.adapt_item(materialize_chart_vectors(profile_root=staging, segment=segment, vocabulary=source.vocab))
                a, b = map(int, source.offsets[i:i + 2])
                for name, array in arrays.items():
                    if not np.array_equal(array[a:b], _tensor_at(expected, name).numpy()):
                        raise ValueError(f'LR vector semantic mismatch: {rid}/{segment["segment_id"]}/{name}')
                count += 1
            _encoded.cache_clear(); path.unlink()
    finally:
        for array in arrays.values():
            array._mmap.close()
        _encoded.cache_clear()
    report = dict(passed=True, policy=POLICY, charts=len(proof['charts']), segments=count,
                  lr_repair=proof['binding'], scope='all LR vector fields, context/targets, Physical mirror LUT and paths; inherited payload hashes')
    write(source.root / REPORT, report)
    data = source.data
    data.update(ready=True, verification=REPORT, verification_sha256=sha(source.root / REPORT),
                preparation_status={'lr-repair-build': 'passed', 'lr-repair-verify': 'passed'})
    write(source.path, data)
    return report
