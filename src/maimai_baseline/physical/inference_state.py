"""Generated-prefix interface to the shared exact interval rule engine."""
from __future__ import annotations
import time
from .intervals import Action, describe, reasons, IDENTITY
from maimai_dataset.event_time import seconds, maximum_duration

POLICY = 'physical-row-stages-interval-v1'
KINDS = ('tap','touch','hold','touch_hold','slide','wifi')

class PhysicalDecodeError(ValueError):
    def __init__(self,message,**context):
        super().__init__(f'{message}: {context}')
        self.context=context

class PhysicalInferenceState:
    def __init__(self,timing,catalog,vocabulary,song_end_tick,*,audio_tail_tick,audio_end_seconds=None):
        self.timing,self.catalog,self.vocabulary=timing,catalog,vocabulary
        self.song_end_tick,self.audio_tail_tick=int(song_end_tick),int(audio_tail_tick)
        self.audio_end_seconds=seconds(timing,audio_tail_tick) if audio_end_seconds is None else float(audio_end_seconds)
        self.actions=[]
        self.version=0
        self._cache={}
        self._stage_engine=None
        self._engine_key=None
        self._interval_context=None
        self._time_tick=None
        self._time_values={}
        self.cache_hits=self.rollbacks=0
        self.check_seconds=0.
        self.check_same_path=True
        self._value_table={k:tuple(str(vocabulary.fields[f].decode(i)) for i in vocabulary.allowed_value_ids('NOTE',f,note_kind=k))
            for k in KINDS for f in ['start_pad' if k in ('slide','wifi') else 'pad']}

    def second(self,tick):
        return seconds(self.timing,tick)

    def owner(self,moment):
        for a in reversed(self.actions):
            if a.moment==moment and a.row['event']=='NOTE':
                return a if a.track else None
        return None

    def branch_available(self, moment):
        owner = self.owner(moment)
        return owner is not None and not owner.wifi and not any(
            a.owner == owner.owner and a.row['event'] == 'SLIDE_BRANCH' for a in self.actions)

    def _candidate(self,row,tick,moment,route=None,*,project=True):
        row=dict(row)
        branch=row['event']=='SLIDE_BRANCH'
        owner=self.owner(moment) if branch else None
        if branch:
            if owner is None:
                raise ValueError('branch has no NOTE owner')
            row['start_pad']=owner.pad
        if self._time_tick!=tick:
            self._time_values={};self._time_tick=tick
        return describe(row,tick=tick,moment=moment,owner=owner.owner if branch else max((a.owner for a in self.actions),default=-1)+1,timing=self.timing,catalog=self.catalog,route=route,project=project,time_values=self._time_values)

    def _reasons(self,row,tick,moment,route=None):
        candidate=self._candidate(row,tick,moment,route,project=False)
        from .query_context import IntervalQueryContext
        key=(self.version,len(self.actions),float(tick),int(moment),self.check_same_path)
        if self._interval_context is None or self._interval_context[0]!=key:
            self._interval_context=(key,IntervalQueryContext(self.actions,candidate.press_start,self.catalog,self.check_same_path))
        return self._interval_context[1].reasons(candidate)

    def _safe_upper(self,row,tick,moment,route,field,lower,upper):
        if upper<lower:
            return upper
        lo,hi=lower,upper+1
        while lo<hi:
            mid=(lo+hi)//2
            if 'multitouch' in self._reasons(dict(row,**{field:mid}),tick,moment,route):
                hi=mid
            else:
                lo=mid+1
        return lo-1

    def local_query(self, tick, moment, stage, *args, **kwargs):
        from .local_query import query
        return query(self, tick, moment, stage, *args, **kwargs)

    def can_start(self,tick,moment):
        return bool(self.query(tick,moment,'event').allowed[0])

    def query(self,tick,moment,stage,kind=None,position=None,flags=(),route=None):
        from .stage_query import StageEngine, query_key
        key=(self.version,float(tick),int(moment),self.check_same_path,query_key(stage,kind,position,flags,route))
        if key not in self._cache:
            started=time.perf_counter()
            engine_key=(self.version,float(tick),int(moment),self.check_same_path)
            if self._engine_key!=engine_key:
                self._stage_engine=StageEngine(self,tick,moment)
                self._engine_key=engine_key
            self._cache[key]=self._stage_engine.query(stage,kind,position,flags,route)
            self.check_seconds+=time.perf_counter()-started
        else:
            self.cache_hits+=1
        return self._cache[key]

    def validate(self,row,tick,moment,route=None):
        from maimai_dataset.modifiers import materialize_row
        actual=materialize_row(row)
        slide=actual['event']=='SLIDE_BRANCH' or actual.get('kind') in ('slide','wifi')
        field='branch_duration_tick' if slide else 'hold_duration_tick' if actual.get('kind') in ('hold','touch_hold') else None
        if field and not 1<=int(actual.get(field,0))<=maximum_duration(self.timing,tick,self.audio_end_seconds,slide=slide):
            return False
        try:
            from .intervals import is_touch, touch_repeat
            candidate = self._candidate(row,tick,moment,route,project=False)
            if is_touch(candidate) and touch_repeat(self.actions,candidate.pad,candidate.press_start):
                return False
            return not self._reasons(row,tick,moment,route)
        except (ValueError,KeyError,IndexError):
            return False

    def commit(self,row,tick,moment,route=None,*,check=True):
        if check and not self.validate(row,tick,moment,route):
            raise PhysicalDecodeError('Physical candidate failed final check',row=row,tick=tick,moment=moment)
        self.actions.append(self._candidate(row,tick,moment,route))
        self.version+=1
        self._cache.clear()
        self._local_context=None
        self._stage_engine=self._interval_context=None
        self._engine_key=None

    def restore(self,count):
        if not 0<=count<=len(self.actions):
            raise ValueError('invalid Physical action cursor')
        del self.actions[count:]
        self.version+=1
        self._cache.clear()
        self._local_context=None
        self._stage_engine=self._interval_context=None
        self._engine_key=None

    def report(self):
        from .local_query import POLICY as local_policy
        return dict(policy=POLICY,backend='exact-intervals',rules=dict(IDENTITY),catalog=dict(self.catalog.identity),committed_actions=len(self.actions),check_seconds=self.check_seconds,cache_hits=self.cache_hits,rollbacks=self.rollbacks,forced_window_closures=0,
            local_policy=local_policy,
            local_stage_seconds=dict(getattr(self,'local_seconds',{})),
            local_counts=dict(getattr(self,'local_counts',{})))
