"""Transactional adapter from generated rows to the standalone slide runtime."""
from dataclasses import asdict
import time

from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from maimai_baseline.physical.inference_state import PhysicalInferenceState
from maimai_baseline.inference.reporting import trace_enabled
from maimai_baseline.inference.reporting import count
from maimai_baseline.physical.miacode_runtime import Marker
from maimai_baseline.physical.miacode_runtime import incremental_replay
from maimai_baseline.physical.miacode_runtime import new_violations
from maimai_baseline.physical.miacode_runtime import resources

POLICY = "physical-v5-miacode-slide-retry-v2"


class MiaCodeInferenceState(PhysicalInferenceState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data, self.runtime_identity = resources()
        for p in self.catalog.raw_paths:
            key = p["key"]
            if key not in data["slides"] and key not in data["wifi"]:
                raise ValueError(f"MiaCode resource lacks catalog path {key}")
        self.markers = []
        self._baseline = []
        self._snapshot = None
        self._trial_states = {}
        self._trial_cache = {}
        self.runtime_seconds = 0.
        self.runtime_checks = self.runtime_cache_hits = 0
        self.bypass_next = False
        self.bypass_first = False
        self.bypasses = 0

    def query(self, *args, **kwargs):
        return super().query(*args, **kwargs)

    def marker(self, row, tick, moment, route):
        from maimai_dataset.modifiers import materialize_row
        row = materialize_row(row)
        branch = row["event"] == "SLIDE_BRANCH"
        owner = self.owner(moment) if branch else None
        kind = ("wifi" if route == 70 else "slide") if branch else row["kind"]
        slide = kind in ("slide", "wifi")
        second = self.second(tick)
        launch = self.second(tick+CHART_TICKS_PER_BEAT) if slide else 0.
        duration = int(row.get("branch_duration_tick" if slide else "hold_duration_tick", 0))
        from maimai_dataset.event_time import duration_end
        end = duration_end(self.timing,tick,duration,slide=slide)
        pad = row["start_pad" if slide else "pad"]
        path = self.catalog.raw_paths[self.catalog.path_id(int(pad[1:]), route)]["key"] if slide else ""
        owner_id = owner.owner if branch else len(self.markers)
        return Marker(len(self.markers), kind, pad, second, end, launch, path,
                      owner=owner_id, head=not branch and row.get("head_kind", "star") != "none",
                      protected="ex" in str(row.get("flags", "")),
                      head_protected="ex" in str(row.get("head_flags", "")), moment=moment)

    def check(self, row, tick, moment, route=None):
        marker = self.marker(row, tick, moment, route)
        key = (self.version, marker)
        self.runtime_checks += 1
        if key in self._trial_cache:
            self.runtime_cache_hits += 1
        if key not in self._trial_cache:
            if len(self._trial_cache) >= 16:
                oldest = next(iter(self._trial_cache))
                del self._trial_cache[oldest]
                self._trial_states.pop(oldest, None)
            start = time.perf_counter()
            diagnostics, states, snapshot, work = incremental_replay(tuple(self.markers)+(marker,), self._snapshot)
            self._trial_states[key] = diagnostics, snapshot
            bad = new_violations(self._baseline, diagnostics)
            old_keys = {d.key for d in self._baseline}
            warnings = [d for d in diagnostics if d.severity == "Warning" and
                        (d.key not in old_keys or marker.id in (d.affected, d.cause))]
            interesting = {marker.id} | {v for d in bad+warnings for v in (d.affected, d.cause)}
            self._trial_cache[key] = (not bad, dict(
                violations=[asdict(d) for d in bad],
                warnings=[asdict(d) for d in warnings], checked_slide_count=len(states),
                judgments=[s for s in states if s["id"] in interesting] if trace_enabled() else [], replay=work))
            self.runtime_seconds += time.perf_counter()-start
            count("miacode_replay_ticks",work["replay_ticks"])
            count("miacode_outer_events",work["outer_events"])
        return self._trial_cache[key]

    def commit(self, row, tick, moment, route=None):
        marker = self.marker(row, tick, moment, route)
        bypass = self.bypass_next
        # A bypass is explicit and only suppresses physical checks, never parsing.
        if not bypass and not self.check(row, tick, moment, route)[0]:
            raise ValueError("MiaCode candidate was committed without acceptance")
        super().commit(row, tick, moment, route, check=not bypass)
        self.markers.append(marker)
        trial = self._trial_states.get((self.version-1, marker))
        if trial is None:
            ds, _, snapshot, _ = incremental_replay(tuple(self.markers), self._snapshot)
            trial = ds, snapshot
        self._baseline, self._snapshot = trial
        self.bypasses += int(bypass)
        self.bypass_next = self.bypass_first = False
        self._trial_cache.clear()
        self._trial_states.clear()

    def restore(self, count):
        super().restore(count)
        del self.markers[count:]
        self._baseline, _, self._snapshot, _ = incremental_replay(tuple(self.markers))
        self._trial_cache.clear()
        self._trial_states.clear()
        self.bypass_next = self.bypass_first = False

    def report(self):
        result = super().report()
        result.update(policy=POLICY, same_path_hard_mask=self.check_same_path,
                      slide_timing_policy="miacode-unshifted-critical-v1",
                      runtime=self.runtime_identity, runtime_seconds=self.runtime_seconds,
                      runtime_checks=self.runtime_checks, runtime_cache_hits=self.runtime_cache_hits,
                      physical_bypasses=self.bypasses, runtime_checkpoint=bool(self._snapshot and self._snapshot[1]))
        return result
