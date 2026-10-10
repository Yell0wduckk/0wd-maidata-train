"""Build current windows from frozen full-chart payloads; never extract audio or Physical."""
from __future__ import annotations
from bisect import bisect_left
from functools import lru_cache
from collections import defaultdict, Counter
import copy
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
from maimai_dataset.window import WINDOW, TARGET_TICKS, HISTORY_TICKS, CHART_TICKS_PER_BEAT, AUDIO_BEATS_BEFORE, AUDIO_BEATS_AFTER, training_window_count
from maimai_baseline.data.training_dataset import read, rows, write, write_rows, inside, copy_verified, inventory, check_inventory, identity, sha, stage, code_identity


def segment_chart(document, template):
    """Cover complete objects plus an empty tail; history remains unsupervised."""
    chunks = document['moment_chunks']
    ticks = [int(c['beat_num'] * CHART_TICKS_PER_BEAT // c['beat_den']) for c in chunks]
    from maimai_baseline.data.prepare import encoded_moments
    # Validate rational anchors and complete row coverage before using any offsets.
    encoded_moments(document, {'record_id': document['record_id'], 'ir_sha256': document['source_ir_sha256']})
    from maimai_dataset.timing import TimingMap
    from fractions import Fraction
    timing = TimingMap.from_dict(template['timing'])
    count = len(document['rows'])
    result = []
    window_count = training_window_count(ticks[-1], int(template['song_end_tick']))
    for number in range(window_count):
        start = number * TARGET_TICKS
        end, context = start + TARGET_TICKS, max(0, start - HISTORY_TICKS)
        first, stop, history = [bisect_left(ticks, t) for t in (start, end, context)]
        row = dict(template)
        row.update(segment_id=f"{document['record_id']}:{number:04d}", target_start_tick=start,
                   target_end_tick=end, context_start_tick=context,
                   context_row_start=chunks[history]['row_start'] if history < len(chunks) else count,
                   target_row_start=chunks[first]['row_start'] if first < len(chunks) else count,
                   target_row_end=chunks[stop]['row_start'] if stop < len(chunks) else count,
                   context_first_tick=ticks[history] if history < first else None,
                   target_first_tick=ticks[first] if first < stop else None, window=WINDOW)
        row['target_rows'] = row['target_row_end'] - row['target_row_start'] + 1
        row['audio_frame_start'] = int(np.floor(timing.beat_to_second(Fraction(start,96)-AUDIO_BEATS_BEFORE)*100))
        row['audio_frame_end'] = int(np.ceil(timing.beat_to_second(Fraction(end,96)+AUDIO_BEATS_AFTER)*100))
        result.append(row)
    if [i for s in result for i in range(s['target_row_start'],s['target_row_end'])] != list(range(count)):
        raise ValueError('canonical target row coverage changed')
    return result


def build_profile(root, audit):
    source = Path(audit['sources']['profile']); output = root/'profile'
    selected = set(audit['chart_ids'])
    records = [r for r in rows(source/'manifest.jsonl') if r['record_id'] in selected]
    templates = {}
    for s in rows(source/'segments.jsonl'):
        if s['record_id'] in selected: templates.setdefault(s['record_id'],s)
    segments, proof = [], {}
    for r in records:
        path = inside(source,r['encoded_path']); document = read(path)
        copy_verified(path,inside(output,r['encoded_path']),sha(path))
        generated = segment_chart(document,templates[r['record_id']]); segments.extend(generated)
        proof[r['record_id']] = {'encoded_sha256':sha(path), 'rows':len(document['rows']),
                                  'notes':sum(v['event']=='NOTE' for v in document['rows']), 'segments':len(generated),
                                  'timing_sha256':identity(templates[r['record_id']]['timing'])}
    write_rows(output/'manifest.jsonl',records); write_rows(output/'segments.jsonl',segments)
    copy_verified(source/'vocabulary.json',output/'vocabulary.json',sha(source/'vocabulary.json'))
    old_summary = read(source/'summary.json')
    summary = {k:old_summary[k] for k in ('schema','row_codec')}
    summary.update(window=WINDOW, profile='context16-pilot',
        source_profile_fingerprint=audit['source_profile_fingerprint'],
        profile_fingerprint=identity({'source':audit['source_profile_fingerprint'],
            'window':WINDOW,'manifest':sha(output/'manifest.jsonl'),'segments':sha(output/'segments.jsonl')}),
        chart_ticks_per_quarter_beat=CHART_TICKS_PER_BEAT, chart_ticks_per_whole=CHART_TICKS_PER_BEAT*4,
        target_beats=WINDOW['target_beats'], prefix_beats=WINDOW['history_beats'],
        audio_before_beats=AUDIO_BEATS_BEFORE, audio_after_beats=AUDIO_BEATS_AFTER,
        eligible_charts=len(records), input_charts=len(records), isolated_charts=0,
        training_rows=sum(v['rows'] for v in proof.values()), segment_count=len(segments),
        eligible_split_counts=dict(Counter(r['split'] for r in records)),
        eligible_source_difficulty_counts=dict(Counter(str(r['source_difficulty_id']) for r in records)),
        segment_split_counts=dict(Counter(s['split'] for s in segments)),
        segment_source_difficulty_counts=dict(Counter(str(s['source_difficulty_id']) for s in segments)),
        target_row_split_counts={role:sum(s['target_rows'] for s in segments if s['split']==role) for role in {s['split'] for s in segments}})
    write(output/'summary.json',summary)
    profile = {'schema':summary['schema'], 'row_codec':summary['row_codec'], 'profile':summary['profile'],
        'window':WINDOW, 'base_feature_stats':'../audio/feature-stats.json','base_manifest':'manifest.jsonl','base_dataset':'.',
        'summary_sha256':sha(output/'summary.json'),'vocabulary_sha256':sha(output/'vocabulary.json'),
        'source_profile_fingerprint':audit['source_profile_fingerprint']}
    write(output/'profile.json',profile)
    selection = {'schema':'maimai.small_sample_suite.v1', **audit['selection']}
    write(root/'selection.json',selection)
    write(root/'provenance/canonical-coverage.json',proof)
    return [output,root/'selection.json',root/'provenance/canonical-coverage.json']


def load_chart_pack(state_root, meta):
    index = rows(inside(state_root,meta['chart_index']['path']))
    with inside(state_root,meta['chart_pack']['path']).open('rb') as stream:
        for entry in index:
            stream.seek(entry['offset']); raw = zlib.decompress(stream.read(entry['length']))
            if len(raw)!=entry['uncompressed_length'] or hashlib.sha256(raw).hexdigest()!=entry['record_sha256']:
                raise ValueError('view chart pack record hash mismatch')
            value=json.loads(raw)
            if value['record_id']!=entry['record_id']: raise ValueError('view chart record identity mismatch')
            yield value


def metadata(root,path,array=None):
    value={'path':Path(path).relative_to(root).as_posix(),'sha256':sha(path)}
    if array is not None: value.update(dtype=array.dtype.name,shape=list(array.shape))
    return value


def build_spatial(root,audit):
    from maimai_baseline.contract import VIEW_NAMES, VIEW_CACHE_SCHEMA
    from maimai_baseline.data.route import Route71Codec
    from maimai_dataset.vocabulary import CompactVocabulary
    from maimai_dataset.chart_cache import materialize_chart_vectors, _tensor_at
    from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
    source=Path(audit['sources']['physical_views']); state=read(source/'manifest.json')
    old=read(Path(audit['sources']['spatial'])/'manifest.json')
    mapping=PhysicalSourceRowStore(source,profile_fingerprint=state['profile_fingerprint'],
        vocabulary_sha256=state['vocabulary_sha256'],chart_cache_manifest_sha256=state['canonical_cache_manifest_sha256'],
        spatial_cache_manifest_sha256=state['spatial_cache_manifest_sha256'])
    output=root/'spatial-views'
    if output.exists(): raise ValueError('incomplete spatial output: quarantine spatial-views before retry')
    output.mkdir()
    profile=root/'profile'; segments=rows(profile/'segments.jsonl'); chart=read(root/'chart-cache/manifest.json')
    offsets=np.load(root/'chart-cache/offsets.npy'); np.save(output/'offsets.npy',offsets)
    copy_verified(root/'chart-cache/segments.jsonl',output/'segments.jsonl',sha(root/'chart-cache/segments.jsonl'))
    vocab=CompactVocabulary.from_dict(read(profile/'vocabulary.json')); codec=Route71Codec(vocab)
    by_record=defaultdict(list)
    for i,s in enumerate(segments): by_record[s['record_id']].append((i,s))
    views={}; proof={}
    for view in VIEW_NAMES:
        target=root/'preparation/staging/view-profile'/view; arrays={}
        try:
            seen=set()
            for full in load_chart_pack(source,state['views'][view]):
                rid=full['record_id']
                if rid not in by_record: raise ValueError('unselected chart in Physical view pack')
                if rid in seen: raise ValueError('duplicate view chart')
                seen.add(rid)
                template=by_record[rid][0][1]; canonical=read(inside(profile,template['encoded_path']))
                view_rows=full['rows']; source_rows=mapping.rows(view,rid).numpy()
                if len(view_rows)!=len(canonical['rows']) or len(source_rows)!=len(view_rows):
                    raise ValueError('view/canonical row count mismatch')
                if view=='identity' and view_rows!=canonical['rows']: raise ValueError('Physical canonical rows differ from frozen encoded chart')
                ticks=[]; tick=0
                for i,r in enumerate(view_rows):
                    if r['event']=='MOMENT': tick+=r['delta_tick']
                    ticks.append(tick)
                    if full['row_map'][i]['tick']!=tick or r['event']!=canonical['rows'][int(source_rows[i])]['event']:
                        raise ValueError('view canonical event/timing mapping mismatch')
                canonical_ticks=[]; t=0
                for r in canonical['rows']:
                    if r['event']=='MOMENT': t+=r['delta_tick']
                    canonical_ticks.append(t)
                if any(ticks[i]!=canonical_ticks[int(source_rows[i])] for i in range(len(ticks))):
                    raise ValueError('spatial mapping changed event anchors')
                changed={**canonical,'rows':view_rows}
                write(inside(target,template['encoded_path']),changed)
                for i,s in by_record[rid]:
                    vectors=codec.adapt_item(materialize_chart_vectors(profile_root=target,segment=s,vocabulary=vocab))
                    if not arrays:
                        for name, oldmeta in old['views'][view]['arrays'].items():
                            path=output/'arrays'/view/f'{name}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                            arrays[name]=np.lib.format.open_memmap(path,mode='w+',dtype=oldmeta['dtype'],shape=(int(offsets[-1]),))
                    for name,a in arrays.items():
                        values=_tensor_at(vectors,name).numpy()
                        if values.shape!=(int(offsets[i+1]-offsets[i]),):raise ValueError('view window row bounds differ')
                        a[offsets[i]:offsets[i+1]]=values
                proof[f'{view}/{rid}']={'rows_sha256':identity(view_rows),'source_rows_sha256':hashlib.sha256(source_rows.astype('<i4').tobytes()).hexdigest()}
            if seen!=set(by_record): raise ValueError('incomplete spatial chart coverage')
            metas={}
            for name,a in arrays.items():
                a.flush();metas[name]=metadata(output,output/'arrays'/view/f'{name}.npy',a)
            views[view]={'arrays':metas,'operation':state['views'][view]['operation']}
        finally:
            for a in arrays.values():a._mmap.close()
    manifest={'schema':VIEW_CACHE_SCHEMA,'window':WINDOW,'profile_fingerprint':read(profile/'summary.json')['profile_fingerprint'],
        'profile_manifest_sha256':sha(profile/'manifest.jsonl'),'vocabulary_sha256':sha(profile/'vocabulary.json'),
        'canonical_cache_manifest_sha256':sha(root/'chart-cache/manifest.json'),'selection_sha256':chart['selection']['selection_sha256'],
        'segment_count':len(segments),'row_count':int(offsets[-1]),'segment_index':metadata(output,output/'segments.jsonl'),
        'offsets':metadata(output,output/'offsets.npy',offsets),'views':views,
        'route_informative_mass_by_class':old['route_informative_mass_by_class'],
        'reuse_provenance':{'source_manifest_sha256':sha(source/'manifest.json'),'row_proof_sha256':identity(proof)}}
    manifest['manifest_identity_sha256']=identity(manifest)
    # View order is a runtime contract; JSON sorting would change it.
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    write(root/'provenance/spatial-row-proof.json',proof)
    return [output,root/'provenance/spatial-row-proof.json']


def import_audio(root,audit):
    from maimai_baseline.time.grid import SAMPLING, INTEGRATION, identity
    from maimai_baseline.data.store import SCHEMA,GEOMETRY_SCHEMA,publish,window_arrays
    from maimai_dataset.timing import TimingMap
    old=read(Path(audit['sources']['audio'])/'bpm-audio-manifest.json')
    profile=root/'profile';chart=read(root/'chart-cache/manifest.json');out=root/'audio'
    material={'window':WINDOW,'source_manifest_sha256':sha(Path(audit['sources']['audio'])/'bpm-audio-manifest.json'),
        'profile_manifest_sha256':sha(profile/'manifest.jsonl'),'segments_sha256':sha(profile/'segments.jsonl'),
        'selection_sha256':chart['selection']['selection_sha256'],'stats_sha256':old['stats_sha256'],
        'base_policy':old['base_policy'],'audio_manifest_sha256':old['normalization_manifest_sha256']}
    preparation=identity(material);variants={};keys={}
    def receipt(path,binding,digest):
        if sha(path)!=digest: raise ValueError('imported payload hash mismatch')
        p=path.with_name(path.name+'.complete.json');value={'identity':binding,'sha256':digest}
        if p.exists() and read(p)!=value:raise ValueError('conflicting import receipt')
        write(p,value)
    for oldkey,value in old['variants'].items():
        v=copy.deepcopy(value);v['identity']['sampling']=SAMPLING
        key=identity(v['identity']);keys[oldkey]=key;variants[key]=v
        binding={'preparation':preparation,'variant':v['identity']}
        for kind in ('feature','coordinates'):receipt(inside(out,v[kind+'_path']),binding,v[kind+'_sha256'])
    receipt(out/'feature-stats.json',material,old['stats_sha256'])
    song_variants=defaultdict(list)
    for key,v in variants.items():
        for song in v['song_ids']:song_variants[str(song)].append(key)
    from maimai_baseline.time.grid import timing_identity
    mapping={};geometry={};cached={}
    for s in rows(profile/'segments.jsonl'):
        candidates=[key for key in song_variants[str(s['song_id'])] if variants[key]['identity']['timing']==timing_identity(TimingMap.from_dict(s['timing']))]
        if len(candidates)!=1:raise ValueError('ambiguous audio variant binding')
        key=candidates[0];v=variants[key];start=int(s['target_start_tick']);mapping[s['segment_id']]=key
        if key not in cached:
            features=np.load(inside(out,v['feature_path']),mmap_mode='r')
            with np.load(inside(out,v['coordinates_path']),allow_pickle=False) as a:coords={n:a[n] for n in a.files}
            cached[key]=(features,coords)
        path=root/'geometry'/f'{key}-{start}.npz'
        binding={'preparation':preparation,'variant':v['identity'],'start_tick':start}
        from maimai_baseline.data.store import completed
        saved=completed(path,binding)
        if saved is None:
            features,coords=cached[key]
            value=window_arrays(features,coords,TimingMap.from_dict(s['timing']),start,v['num_samples'])[4]
            saved=publish(path,value,binding)
        geometry[s['segment_id']]={'path':path.name,'sha256':saved['sha256']}
    g={'schema':GEOMETRY_SCHEMA,'window':WINDOW,'profile_fingerprint':read(profile/'summary.json')['profile_fingerprint'],
        'suite_sha256':material['selection_sha256'],'segment_count':len(mapping),'segments':geometry}
    gmeta=publish(root/'geometry/manifest.json',g,material)
    ameta=publish(out/'audit.json',{'inputs':material,'preparation_identity':preparation,'complete':True,'passed':True,
        'conflicts':[],'sampling':SAMPLING,'integration':INTEGRATION,'variants':variants},material)
    manifest={'schema':SCHEMA,'window':WINDOW,'complete':True,'sampling':SAMPLING,'integration':INTEGRATION,
        **{k:material[k] for k in ('profile_manifest_sha256','segments_sha256','selection_sha256','stats_sha256','base_policy')},
        'normalization_manifest_sha256':material['audio_manifest_sha256'],'preparation_identity':preparation,
        'audit_sha256':ameta['sha256'],'geometry_manifest_sha256':gmeta['sha256'],'geometry_root':'../geometry','stats_path':'feature-stats.json',
        'variants':variants,'segments':mapping,'song_ids':old['song_ids'],
        'feature_import':{'source_manifest_sha256':material['source_manifest_sha256'],'old_to_new_variant':keys,'payloads_unchanged':True}}
    publish(out/'bpm-audio-manifest.json',manifest,material)
    return [out,root/'geometry']


def rebind_physical(root,audit):
    from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
    source=root/'preparation/staging/physical-original';old=read(source/'manifest.json')
    proof=read(root/'provenance/canonical-coverage.json');state=read(Path(audit['sources']['physical_views'])/'manifest.json')
    if any(old[k]!=state[v] for k,v in [('profile_fingerprint','profile_fingerprint'),
        ('chart_cache_manifest_sha256','canonical_cache_manifest_sha256'),('spatial_cache_manifest_sha256','spatial_cache_manifest_sha256')]):
        raise ValueError('Physical/view provenance mismatch')
    store=PhysicalBitplaneStore(source);store.validate(full_hash=True,scan_contents=True)
    if {r['record_id'] for r in store.chart_records}!=set(proof):raise ValueError('Physical canonical chart coverage mismatch')
    segments=rows(root/'profile/segments.jsonl')
    templates={s['record_id']:s for s in segments}
    from maimai_dataset.timing import TimingMap
    for r in store.chart_records:
        i=r['chart_index'];n=int(store.offsets['source_offsets'][i+1]-store.offsets['source_offsets'][i])
        if n!=proof[r['record_id']]['rows'] or r['encoded_sha256']!=proof[r['record_id']]['encoded_sha256']:
            raise ValueError('Physical canonical encoded identity/row count mismatch')
        timing=TimingMap.from_dict(templates[r['record_id']]['timing'])
        begin,end=map(int,store.offsets['timing_offsets'][i:i+2])
        expected={'timing_ticks':[float(p.beat)*96 for p in timing.points],
                  'timing_seconds':[p.second for p in timing.points],'timing_bpms':[p.bpm for p in timing.points]}
        for key,value in expected.items():
            if not np.array_equal(store.arrays[key][begin:end],np.asarray(value,dtype=np.float64)):
                raise ValueError(f'Physical full-chart timing mismatch: {r["record_id"]}/{key}')
    out=root/'physical/bitplanes';out.mkdir(parents=True,exist_ok=True)
    for meta in [old['charts'],*old['arrays'].values(),*old['offsets'].values()]:
        copy_verified(inside(source,meta['path']),inside(out,meta['path']),meta['sha256'])
    indices={r['record_id']:r['chart_index'] for r in store.chart_records}
    segments=rows(root/'profile/segments.jsonl')
    write_rows(out/'segments.jsonl',[{'segment_id':s['segment_id'],'record_id':s['record_id'],'chart_index':indices[s['record_id']]} for s in segments])
    binding={'source_manifest_sha256':sha(source/'manifest.json'),'canonical_proof_sha256':sha(root/'provenance/canonical-coverage.json'),
             'spatial_proof_sha256':sha(root/'provenance/spatial-row-proof.json'),'window':WINDOW}
    manifest={**old,'window':WINDOW,'reuse_provenance':binding,'profile_fingerprint':read(root/'profile/summary.json')['profile_fingerprint'],
        'chart_cache_manifest_sha256':sha(root/'chart-cache/manifest.json'),'spatial_cache_manifest_sha256':sha(root/'spatial-views/manifest.json'),
        'profile_manifest_sha256':sha(root/'profile/manifest.jsonl'),'segments_sha256':sha(root/'profile/segments.jsonl'),
        'selection_sha256':read(root/'chart-cache/manifest.json')['selection']['selection_sha256'],
        'segments':metadata(out,out/'segments.jsonl'),'segment_count':len(segments)}
    manifest.pop('manifest_identity_sha256',None);manifest['manifest_identity_sha256']=identity(manifest);write(out/'manifest.json',manifest)
    sm={k:state[k] for k in ('schema','source_row_mapping','vocabulary_sha256','chart_count')}
    sm.update(window=WINDOW,reuse_provenance=binding,profile_fingerprint=manifest['profile_fingerprint'],
        canonical_cache_manifest_sha256=manifest['chart_cache_manifest_sha256'],spatial_cache_manifest_sha256=manifest['spatial_cache_manifest_sha256'],
        views={v:{k:m[k] for k in ('source_row_index','source_row_pack')} for v,m in state['views'].items()})
    sm['state_fingerprint']=identity(sm);write(root/'physical/source-rows/manifest.json',sm)
    return [out,root/'physical/source-rows']


class CurriculumSource:
    def __init__(self,root,epochs=75,full_selection=False):
        from maimai_baseline.contract import SEED,EPOCH_PLAN_MODE
        self.mode=EPOCH_PLAN_MODE;self.available_epochs=epochs;self.seed=SEED
        self.full_selection=bool(full_selection)
        if self.full_selection and read(root/'selection.json').get('schema')!='maimai.full_train_selection.v1':
            raise ValueError('full curriculum requires explicit full train selection')
        selected=set(read(root/'selection.json')['train_song_ids'])
        self.segments=[s for s in rows(root/'profile/segments.jsonl') if s['song_id'] in selected]
        self.charts=defaultdict(set);self.by_chart=defaultdict(list)
        audio=read(root/'audio/bpm-audio-manifest.json')
        self.audio_keys=[(audio['segments'][s['segment_id']],s['target_start_tick']) for s in self.segments]
        for i,s in enumerate(self.segments):
            key=(s['song_id'],int(s['source_difficulty_id']));self.by_chart[key].append(i);self.charts[key[0]].add(key[1])
        self.manifest={'epochs':epochs,'seed':SEED,'token_budget':2048,'window':WINDOW,'segment_count':len(self.segments),
            'segment_order_sha256':identity([s['segment_id'] for s in self.segments]),'curriculum':'existing-chart-slot-policy',
            'profile_segments_sha256':sha(root/'profile/segments.jsonl')}
        if self.full_selection:self.manifest['curriculum']='full-selection-existing-chart-slot-policy-v1'
        from maimai_baseline.contract import spatial_sampling_identity
        self.manifest["spatial_sampling"] = spatial_sampling_identity()
        self.manifest['manifest_identity_sha256']=identity(self.manifest)
    @lru_cache(maxsize=1)
    def load_epoch(self,index):
        from maimai_baseline.data.curriculum import curriculum_slots,assign_spatial_views,VirtualSegmentRef,_stable_order
        from maimai_baseline.contract import VIEW_NAMES
        slots=curriculum_slots(self.charts,seed=self.seed,epoch=index+1,full_selection=self.full_selection)
        assigned=assign_spatial_views(slots,seed=self.seed,epoch=index+1)
        groups=defaultdict(list)
        for slot in slots:
            for i in self.by_chart[slot.chart_key]:groups[self.audio_keys[i]].append(VirtualSegmentRef(i,VIEW_NAMES.index(assigned[slot]),slot.occurrence))
        groups=[(key,tuple(refs),sum(max(1,self.segments[r.canonical_index]['target_rows']) for r in refs)) for key,refs in groups.items()]
        ordered=_stable_order(groups,self.seed,index,'audio-group-order')
        refs=tuple(r for _,members,_ in ordered for r in members)
        return (refs,)
    def audit(self,index):
        from maimai_baseline.data.repack import reference_hash
        from maimai_baseline.data.curriculum import curriculum_slots,assign_spatial_views
        from maimai_baseline.contract import DIFFICULTY_POOLS
        slots=curriculum_slots(self.charts,seed=self.seed,epoch=index+1,full_selection=self.full_selection)
        views=assign_spatial_views(slots,seed=self.seed,epoch=index+1)
        counts=defaultdict(Counter)
        for slot in slots:counts[DIFFICULTY_POOLS[slot.difficulty_id]][views[slot]]+=1
        refs=self.load_epoch(index)[0]
        return {'plan_sha256':reference_hash(refs),'chart_slots':len(slots),'segment_references':len(refs),
            'audio_groups':len({self.audio_keys[r.canonical_index] for r in refs}),
            'difficulty_slot_counts':dict(Counter(str(s.difficulty_id) for s in slots)),
            'view_counts_by_pool':{p:dict(v) for p,v in counts.items()}}


def plan_inputs(root):
    return {k:sha(root/p) for k,p in [('profile_segments','profile/segments.jsonl'),('chart_manifest','chart-cache/manifest.json'),
        ('spatial_manifest','spatial-views/manifest.json'),('bpm_manifest','audio/bpm-audio-manifest.json')]}


def plan_metrics(root,source):
    from maimai_baseline.contract import VIEW_NAMES
    lengths=[s['target_rows'] for s in source.segments];notes={};full={}
    for i,s in enumerate(source.segments):
        rid=s['record_id']
        if rid not in full:full[rid]=read(inside(root/'profile',s['encoded_path']))['rows']
        count=sum(r['event']=='NOTE' for r in full[rid][s['target_row_start']:s['target_row_end']])
        for v in range(len(VIEW_NAMES)):notes[i,v]=count
    return dict(lengths=lengths,notes=notes,audio_keys=source.audio_keys)


def build(root,audit):
    def migrate():
        for item in audit['migrations']:copy_verified(item['source'],inside(root,item['destination']),item['sha256'])
        write(root/'provenance/migration.json',audit['migrations'])
        provenance = [root/'provenance/migration.json']
        for key,name in [('audio','bpm-audio-manifest.json'),('audio','audit.json'),('spatial','manifest.json'),('physical_views','manifest.json')]:
            path=Path(audit['sources'][key])/name
            copy_verified(path,root/'provenance'/f'{key}-{name}',sha(path))
            provenance.append(root/'provenance'/f'{key}-{name}')
        return provenance
    stage(root,'copy',audit,migrate)
    # Copy receipts alone never excuse a changed/missing payload.
    for m in audit['migrations']:
        if sha(inside(root,m['destination']))!=m['sha256']:raise ValueError('migrated payload changed')
    stage(root,'profile',audit,lambda:build_profile(root,audit))
    from maimai_dataset.chart_cache import build_chart_vector_cache
    def chart():
        build_chart_vector_cache(profile_root=root/'profile',suite_path=root/'selection.json',output=root/'chart-cache')
        return [root/'chart-cache']
    stage(root,'chart',audit,chart)
    stage(root,'spatial',audit,lambda:build_spatial(root,audit))
    stage(root,'audio',audit,lambda:import_audio(root,audit))
    stage(root,'physical',audit,lambda:rebind_physical(root,audit))
    def plans():
        from maimai_baseline.data.repack import build_plans
        source=CurriculumSource(root)
        build_plans(source,root/'epoch-plans',**plan_metrics(root,source),input_identity=plan_inputs(root))
        return [root/'epoch-plans']
    stage(root,'plans',audit,plans)
    return {'built':True,'ready':False,'next':'verify'}


PATHS={'dataset':'profile','chart_cache':'chart-cache','interpolation_geometry':'geometry',
    'bpm_audio_manifest':'audio/bpm-audio-manifest.json','spatial_view_cache':'spatial-views','physical_path_catalog':'physical/path-catalog',
    'physical_source_rows':'physical/source-rows','physical_bitplane_cache':'physical/bitplanes','route_artifact':'route-artifact'}


def runtime_args(root,budget):
    from maimai_baseline.cli import parser
    argv=['verify-inputs','--training-recipe',str(budget),'--device','cpu','--num-workers','0','--audio-policy','normalized-p50',
        '--output',str(root/f'preparation/reports/runtime-{budget}')]
    for k,v in {**PATHS,'epoch_plan_cache':f'epoch-plans/{budget}'}.items():argv.extend(['--'+k.replace('_','-'),str(inside(root,v))])
    return parser().parse_args(argv)


def verify(root,audit):
    from maimai_baseline.training.verification import input_files
    from maimai_baseline.training.inputs import _prepare_inputs,_loaders
    from maimai_baseline.data.store import BpmFeatureStore
    from maimai_baseline.data.curriculum import EpochPlanStore
    from maimai_baseline.data.repack import verify_plan
    # Verify all completed stage receipts, including encoded full-chart bytes.
    files={}
    for name in ('copy','profile','chart','spatial','audio','physical','plans'):
        report=read(root/f'preparation/reports/{name}.json')
        if report['audit_identity'] != audit['identity']:raise ValueError('build receipt identity mismatch')
        check_inventory(root,report['files']);files.update(report['files'])
    source=CurriculumSource(root);metrics=plan_metrics(root,source);batches={}
    for budget in (3072,4096):
        args=runtime_args(root,budget)
        for path,digest in input_files(args):
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('runtime input escapes dataset root')
            actual=sha(path)
            if digest is not None and actual!=digest:raise ValueError(f'runtime payload hash mismatch: {path}')
            files[path.relative_to(root).as_posix()]={'sha256':actual,'bytes':path.stat().st_size}
        inputs=_prepare_inputs(args)
        if budget==3072:
            BpmFeatureStore(root/'audio/bpm-audio-manifest.json',profile=root/'profile',stats=inputs.feature_stats,
                song_ids=set(audit['selection']['train_song_ids']+audit['selection']['validation_song_ids']),
                selection_sha256=inputs.chart_store.manifest['selection']['selection_sha256'],geometry_root=root/'geometry',
                expected_segment_ids=set(inputs.chart_store.segment_records),verify_payloads=True)
            inputs.physical_store.validate(full_hash=True,scan_contents=True)
            proof=read(root/'provenance/canonical-coverage.json')
            inputs.physical_source_rows.validate_records({k:v['rows'] for k,v in proof.items()})
        verify_plan(source,EpochPlanStore(root/f'epoch-plans/{budget}'),**metrics)
        train,_,validation=_loaders(inputs,args)
        first=next(iter(train));val=next(iter(validation))
        batches[str(budget)]={'train_audio_shape':list(first['audio'].shape),'validation_audio_shape':list(val['audio'].shape),
            'train_rows':int(first['target_ids']['event'].ne(-100).sum())}
    result={'passed':True,'audit_identity':audit['identity'],'source_identity':code_identity(),'files':files,
        'bytes':sum(m['bytes'] for m in files.values()),'first_batches':batches,'window':WINDOW,'scope':'local-cpu-data-only'}
    write(root/'preparation/reports/verify.json',result)
    dataset={'schema':'maimai.training_dataset.v1','window':WINDOW,'ready':True,'migration_finalized':False,
        'audit_identity':audit['identity'],'source_identity':code_identity(),'selection':'selection.json','inputs':PATHS,
        'epoch_plans':{str(b):f'epoch-plans/{b}' for b in (3072,4096)},'verification':'preparation/reports/verify.json',
        'verification_sha256':sha(root/'preparation/reports/verify.json'),'charts':audit['charts'],'files':files}
    write(root/'dataset.json',dataset)
    return result
