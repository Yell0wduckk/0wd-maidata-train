"""Disk-shared interval descriptors and exhaustive small-chord tables.

The build API consumes explicit chart records. Runtime loading never opens the
old bitplanes, and neither the disk identity nor the LRU contains training order.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
from .intervals import Action, prune_history
from .stage_query import StageResult, physical_query_key
from .shared_cache import CACHE_IDENTITY, MomentRelations, MomentTable, state_key
from maimai_dataset.timing import TimingMap


def key_text(value):
    return json.dumps(value,separators=(',',':'),ensure_ascii=True)


def build_chart(root,*,chart,view,timing,source_moments,source_actions,moment_ticks,actions,
                catalog,vocabulary,audio_end_seconds):
    """Build only explicitly selected new chart/view outputs; caller binds inputs."""
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    arrays=[]
    offset=0
    tables=[]
    visible=tuple(a for a in actions if a.moment<0)
    for moment,tick in enumerate(moment_ticks):
        base=prune_history(visible,timing.beat_to_second(float(tick)/96))
        groups={}
        for a in actions:
            if a.moment==moment:
                groups.setdefault(a.owner,[]).append(a)
        relations=MomentRelations(base,tuple(tuple(b) for b in groups.values()),str(chart),str(view),moment,
                                 'causal-chart-prefix',float(tick))
        table=MomentTable(relations,timing=timing,catalog=catalog,vocabulary=vocabulary,
                          audio_end_seconds=audio_end_seconds).build()
        entries={}
        for key,queries in table.states.items():
            records={}
            for query,result in queries.items():
                shape=list(result.bad.shape)
                size=result.bad.size
                arrays.append(result.bad.reshape(-1))
                meta=asdict(result);meta.pop('bad')
                records[key_text(query)]=dict(meta,offset=offset,shape=shape)
                offset+=size
            entries[key_text(key)]=records
        tables.append(dict(entries=entries,order_count=table.order_count,
                           base=[actions.index(a) for a in base],
                           base_handoffs=[dict(handoff_release=a.handoff_release,
                               handoff_incoming=a.handoff_incoming,handoff_outgoing=a.handoff_outgoing) for a in base],
                           blocks=[[actions.index(a) for a in b] for b in relations.blocks]))
        visible=(*base,*(a for block in groups.values() for a in block))
    np.save(root/'supports.npy',np.concatenate(arrays) if arrays else np.empty(0,dtype=bool),allow_pickle=False)
    metadata=dict(contract=CACHE_IDENTITY,chart=str(chart),view=str(view),timing=timing.to_dict(),audio_end_seconds=float(audio_end_seconds),
        actions=[asdict(a) for a in actions],source_moments=list(map(int,source_moments)),
        source_actions=list(map(int,source_actions)),moment_ticks=list(map(float,moment_ticks)),tables=tables)
    (root/'chart.json').write_text(json.dumps(metadata,separators=(',',':')),encoding='utf-8')
    return dict(chart=str(chart),view=str(view),path=root.name,source_row_count=len(source_moments),
                orders=sum(t['order_count'] for t in tables),states=sum(len(t['entries']) for t in tables))


class ChartIntervals:
    def __init__(self,root,catalog,vocabulary):
        self.data=json.loads((Path(root)/'chart.json').read_text('utf-8'))
        if self.data.get('contract')!=CACHE_IDENTITY:
            raise ValueError('Physical chart stage cache semantics mismatch')
        self.arrays=np.load(Path(root)/'supports.npy',mmap_mode='r',allow_pickle=False)
        if self.arrays.dtype!=np.bool_ or self.arrays.ndim!=1:
            raise ValueError('Physical packed payload shape/dtype mismatch')
        self.actions=tuple(Action(**dict(a,entries=tuple(tuple(x) for x in a['entries']))) for a in self.data['actions'])
        self.timing=TimingMap.from_dict(self.data['timing'])
        self.catalog,self.vocabulary=catalog,vocabulary
        self.loaded={}

    def action_for_source(self,source):
        if not 0<=source<len(self.data['source_actions']):return None
        index=self.data['source_actions'][source]
        return self.actions[index] if index>=0 else None

    def prior(self,moment):
        table=self.data['tables'][moment]
        return tuple(replace(self.actions[i],**facts) for i,facts in zip(table['base'],table['base_handoffs'],strict=True))

    def context(self,moment,prefix):
        return QueryContext(self,moment,tuple(prefix))

    def query(self,moment,visible_sources,*query):
        prefix=tuple(a for s in visible_sources for a in [self.action_for_source(s)] if a is not None and a.moment==moment)
        return self.context(moment,prefix).query(*query)

    def close(self):
        # Results are mmap views: release internal references before the mapping.
        self.loaded.clear()
        self.arrays._mmap.close()


class QueryContext:
    """One immutable visible prefix; never retained for large-chord subsets."""
    def __init__(self,chart,moment,prefix):
        self.chart,self.moment=chart,moment
        self.large=len(chart.data['tables'][moment]['blocks'])>=4
        self.engine=None
        if self.large:
            from .inference_state import PhysicalInferenceState
            self.engine=PhysicalInferenceState(chart.timing,chart.catalog,chart.vocabulary,2**31-1,
                audio_tail_tick=2**31-1,audio_end_seconds=chart.data['audio_end_seconds'])
            self.engine.actions=list(chart.prior(moment))+list(prefix)
        else:
            self.prefix_key=key_text(state_key(prefix))
            self.entries=chart.data['tables'][moment]['entries'][self.prefix_key]

    def query(self,*query):
        chart=self.chart
        if self.large:
            return self.engine.query(chart.data['moment_ticks'][self.moment],self.moment,*query)
        q=key_text(physical_query_key(chart.catalog,*query))
        key=(self.moment,self.prefix_key,q)
        if key not in chart.loaded:
            e=self.entries[q]
            meta={k:v for k,v in e.items() if k not in ('offset','shape')}
            meta['values']=tuple(meta['values'])
            offset=int(e['offset']);shape=tuple(e['shape']);size=int(np.prod(shape))
            if offset<0 or offset+size>chart.arrays.size or len(shape)<2 or shape[0]!=4:
                raise ValueError('Physical packed entry bounds invalid')
            chart.loaded[key]=StageResult(**meta,bad=chart.arrays[offset:offset+size].reshape(shape))
        return chart.loaded[key]


class IntervalPhysicalStore:
    def __init__(self,root,*,catalog,vocabulary,allow_partial=False,**expected):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text('utf-8'))
        if self.manifest['contract']!=CACHE_IDENTITY:
            raise ValueError('Physical interval cache semantics mismatch')
        self.catalog,self.vocabulary=catalog,vocabulary
        self.chart_records=tuple(self.manifest['charts'])
        self.segment_records={r['segment_id']:r for r in self.manifest['segments']}
        self.records={(int(r['chart_index']),int(r['view_index'])):r for r in self.manifest['views']}
        self._loaded=OrderedDict()

    def handle(self,segment_id):
        return int(self.segment_records[str(segment_id)]['chart_index'])

    def chart(self,index,view):
        key=(int(index),int(view))
        if key not in self._loaded:
            self._loaded[key]=ChartIntervals(self.root/self.records[key]['path'],self.catalog,self.vocabulary)
            while len(self._loaded)>4:
                _,old=self._loaded.popitem(last=False)
                old.close()
        self._loaded.move_to_end(key)
        return self._loaded[key]

    def __getstate__(self):
        return dict(self.__dict__,_loaded=OrderedDict())
