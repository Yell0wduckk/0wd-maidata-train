"""Moment state plus only the teacher-forced stage queries actually consumed."""
from __future__ import annotations
from itertools import permutations
from dataclasses import dataclass
from .inference_state import PhysicalInferenceState
from .intervals import IDENTITY
from .stage_query import query_key, physical_query_key
from maimai_dataset.modifiers import normalize_row

CACHE_IDENTITY={'schema':'maimai.shared_physical_moments.v5','rules':IDENTITY,
    'representation':'packed-causal-stage-results-v3',
    'orders':{'1':1,'2':2,'3':6,'4+':'visible-prefix-merge'},
    'key':['chart','view','moment','history-context','visible-actions','owner','stage','preceding-fields'],
    'forbidden_keys':['epoch','occurrence','batch'],
    'physical_flags':['short_hold','short_touch_hold']}

@dataclass(frozen=True)
class MomentRelations:
    base: tuple
    blocks: tuple[tuple,...]
    chart: str
    view: str
    moment: int
    context: str
    tick: float

def state_key(actions):
    owner=next((a.owner for a in reversed(actions) if a.row['event']=='NOTE'),-1)
    from .intervals import is_touch
    keys=tuple(sorted((a.owner,a.tick,a.path,a.wifi,a.pad,a.end_pad,a.press_start,a.press_end,
        a.launch,a.actual_end,a.release,a.row['event'],is_touch(a),a.short,
        a.handoff_incoming,a.handoff_outgoing,a.handoff_release if a.handoff_release is not None else a.release) for a in actions))
    return keys,owner

def action_queries(action):
    """The source target chooses query conditions, never future-stage answers."""
    yield ('event',)
    if action is None:
        return
    row=normalize_row(action.source_row or action.row)
    kind=row.get('kind') if row['event']=='NOTE' else None
    if kind is not None:
        yield ('kind',)
        yield ('position',kind)
    position=action.pad
    flags=row.get('flags',())
    if action.track:
        yield query_key('route',kind,position,flags)
    if action.track or kind in ('hold','touch_hold'):
        yield query_key('duration',kind,position,flags,action.route_class)

class MomentTable:
    def __init__(self,relations,*,timing,catalog,vocabulary,audio_end_seconds):
        self.relations=relations
        self.timing,self.catalog,self.vocabulary=timing,catalog,vocabulary
        self.audio_end_seconds=audio_end_seconds
        self.states={}
        self.order_count=0

    def engine(self,prefix):
        r=self.relations
        state=PhysicalInferenceState(self.timing,self.catalog,self.vocabulary,2**31-1,
            audio_tail_tick=2**31-1,audio_end_seconds=self.audio_end_seconds)
        state.actions=list(r.base)+list(prefix)
        return state

    def build(self):
        if len(self.relations.blocks)>3:
            return self
        for order in permutations(self.relations.blocks):
            self.order_count+=1
            prefix=[]
            for block in order:
                for action in block:
                    self._insert(prefix,action)
                    prefix.append(action)
            self._insert(prefix,None)
        return self

    def _insert(self,prefix,action):
        entries=self.states.setdefault(state_key(prefix),{})
        engine=None
        for query in action_queries(action):
            key=physical_query_key(self.catalog,*query)
            if key not in entries:
                engine=engine or self.engine(prefix)
                entries[key]=engine.query(self.relations.tick,self.relations.moment,*key)

    def query(self,prefix,*query):
        key=physical_query_key(self.catalog,*query)
        if len(self.relations.blocks)<=3:
            return self.states[state_key(prefix)][key]
        return self.engine(prefix).query(self.relations.tick,self.relations.moment,*key)

    @property
    def identity(self):
        r=self.relations
        return r.chart,r.view,r.moment,r.context
