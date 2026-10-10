"""Bounded, per-decision interval preparation shared by all field stages."""
from collections import defaultdict
from .intervals import EPS, geometry_reasons, track_weight, branch_errors, is_touch, handoff_releases, prune_history


def matching(edges):
    matched={}
    def visit(i,seen):
        for group in edges.get(i,()):
            if group in seen:continue
            seen.add(group)
            if group not in matched or visit(matched[group],seen):
                matched[group]=i;return True
        return False
    return sum(visit(i,set()) for i in edges)


class IntervalQueryContext:
    def __init__(self,history,tick_second,catalog,check_same_path):
        self.history=prune_history(history,tick_second)
        self.catalog,self.check_same_path=catalog,check_same_path
        self.geometry={};self.occupancy={}
        points=sorted({p for a in self.history for p in (a.press_start,a.press_end,
            *((a.launch,a.release) if a.track else ()))})
        points=[float('-inf'),*points,float('inf')]
        self.raw_panels=[]
        for lo,hi in zip(points,points[1:]):
            if hi<=lo+EPS:continue
            presses=tuple(a for a in self.history if a.press_start<=lo+EPS and a.press_end>lo+EPS)
            groups=defaultdict(list)
            for a in self.history:
                if a.track and a.launch<=lo+EPS and a.release>lo+EPS:groups[a.owner].append(a)
            self.raw_panels.append((lo,hi,presses,dict(groups)))
        self.track_releases = handoff_releases(self.history)
        self.handoffs = {}
        self.panel_cache = {}
        self.panels = self._panels(self.track_releases)

    def _panels(self, releases):
        key=tuple((id(a),releases[id(a)]) for a in self.history if a.track)
        if key in self.panel_cache:return self.panel_cache[key]
        panels=[]
        for left,right,presses,raw_groups in self.raw_panels:
            cuts=sorted({left,right,*(p for _,p in key if left<p<right)})
            ordinary=tuple(a for a in presses if not is_touch(a))
            touches=tuple(a for a in presses if is_touch(a))
            for lo,hi in zip(cuts,cuts[1:]):
                if hi<=lo+EPS:continue
                groups={g:tuple(a for a in tracks if releases[id(a)]>lo+EPS) for g,tracks in raw_groups.items()}
                groups={g:tracks for g,tracks in groups.items() if tracks}
                edges={i:tuple(g for g,tracks in groups.items() if any(a.pad==t.pad and abs(a.press_start-t.launch)<=EPS for t in tracks))
                    for i,a in enumerate(ordinary) if a.short}
                if touches:
                    edges['touch-pool']=tuple(sorted({g for a in touches if a.short for g,tracks in groups.items()
                        if any(a.pad==t.pad and abs(a.press_start-t.launch)<=EPS for t in tracks)}))
                demand=len(ordinary)+bool(touches)+sum(track_weight(tracks) for tracks in groups.values())
                panels.append((lo,hi,ordinary,touches,groups,edges,demand,demand-matching(edges)))
        self.panel_cache[key]=panels
        return panels

    def overloaded(self,c):
        # Equivalent short Hold/Tap heads share occupancy, but track endpoints
        # and actual ends must remain distinct even with identical release times.
        key=(c.owner,c.pad,is_touch(c),c.short,c.press_start,c.press_end,c.launch,c.release,
             c.actual_end,c.end_pad,c.path,c.wifi)
        if key in self.occupancy:return self.occupancy[key]
        spans=[(c.press_start,c.press_end)]+([(c.launch,c.release)] if c.track else [])
        points=[c.press_start,c.press_end]+([c.launch,c.release] if c.track else [])
        releases = self.track_releases
        if c.track:
            # In the chronological prefix there are no later track launches:
            # the incoming handoff is independent of the trial duration.
            future=any(a.track and a.launch>c.launch+EPS for a in self.history)
            handoff_key=(c.owner,c.pad,c.end_pad,c.path,c.wifi,c.launch,
                c.actual_end if future else None,c.release if future else None)
            if handoff_key not in self.handoffs:
                full=handoff_releases((*self.history,c))
                self.handoffs[handoff_key]=({id(a):full[id(a)] for a in self.history if a.track},full[id(c)])
            previous,candidate_release=self.handoffs[handoff_key]
            releases={**previous,id(c):candidate_release if future else c.release}
        panels=self._panels(releases) if c.track else self.panels
        for left,right,ordinary,touches,groups,base_edges,base_demand,_ in panels:
            cuts=sorted({left,right,*(p for p in points if left<p<right)})
            for lo,hi in zip(cuts,cuts[1:]):
                if hi<=lo+EPS or not any(lo<end-EPS and hi>start+EPS for start,end in spans):continue
                press=c.press_start<=lo+EPS and c.press_end>lo+EPS
                track=c.track and c.launch<=lo+EPS and releases[id(c)]>lo+EPS
                demand=base_demand
                edges=dict(base_edges)
                if track:
                    demand+=2 if c.wifi else 1
                    for i,a in enumerate(ordinary):
                        if a.short and a.pad==c.pad and abs(a.press_start-c.launch)<=EPS:
                            edges[i]=tuple(sorted(set((*edges.get(i,()),c.owner))))
                    if any(a.short and a.pad==c.pad and abs(a.press_start-c.launch)<=EPS for a in touches):
                        edges['touch-pool']=tuple(sorted(set((*edges.get('touch-pool',()),c.owner))))
                if press:
                    touch=is_touch(c)
                    demand+=int(not touch or not touches)
                    if c.short:
                        options={g for g,tracks in groups.items() if any(c.pad==a.pad and abs(c.press_start-a.launch)<=EPS for a in tracks)}
                        if track and abs(c.press_start-c.launch)<=EPS:options.add(c.owner)
                        if touch:edges['touch-pool']=tuple(sorted(set((*edges.get('touch-pool',()),*options))))
                        else:edges[len(ordinary)]=tuple(sorted(options))
                if demand-matching(edges)>2:
                    self.occupancy[key]=True;return True
        self.occupancy[key]=False;return False

    def reasons(self,c):
        # Candidate duration does not affect pre-existing position/path bans.
        key=(c.row['event'],is_touch(c),c.pad,c.path,c.press_start,c.launch)
        if key not in self.geometry:
            self.geometry[key]=frozenset(geometry_reasons(self.history,c,self.catalog,check_same_path=self.check_same_path))
        result=set(self.geometry[key]) | branch_errors(self.history, c)
        if self.overloaded(c):result.add('multitouch')
        return result
