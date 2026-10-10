"""Exact beat controls; global waits/releases and written-BPM motion lengths."""
from bisect import bisect_right
from fractions import Fraction
import math

TICKS_PER_BEAT = 96


class TimingMap:
    def __init__(self, points, first_seconds=0., *, validate_input=True):
        self.points = tuple((Fraction(b), float(v)) for b, v in points)
        self.first_seconds = float(first_seconds)
        self.validate_input = validate_input
        if validate_input and (not self.points or self.points[0][0] != 0 or not math.isfinite(self.first_seconds)
                or any(not math.isfinite(v) or v <= 0 for _, v in self.points)
                or any(a[0] >= b[0] for a, b in zip(self.points, self.points[1:]))):
            raise ValueError('invalid timing controls')
        self.beats = tuple(b for b, _ in self.points)
        seconds = [self.first_seconds]
        for (a, v), (b, _) in zip(self.points, self.points[1:]):
            seconds.append(seconds[-1] + float(b-a)*60/v)
        self.seconds = tuple(seconds)

    @classmethod
    def from_audio_timing(cls, metadata, *, validate_input=True):
        return cls([(Fraction(n, d), v) for n, d, v in metadata['bpm']], metadata['first_seconds'], validate_input=validate_input)

    @classmethod
    def from_ir(cls, ir, *, validate_input=True):
        points = {}
        for c in ir['controls']:
            if c['kind'] == 'bpm':
                b = c['position_beats']
                points[Fraction(b['num'], b['den'])] = c['bpm']
        return cls(sorted(points.items()), ir['first_seconds'], validate_input=validate_input)

    def bpm_at_beat(self, beat):
        return self.points[max(0, bisect_right(self.beats, beat)-1)][1]

    def beat_to_second(self, beat):
        i = max(0, bisect_right(self.beats, beat)-1)
        return self.seconds[i] + float(beat-self.beats[i])*60/self.points[i][1]

    def second_to_beat(self, second):
        if self.validate_input and not math.isfinite(second):
            raise ValueError('non-finite second')
        i = max(0, bisect_right(self.seconds, second)-1)
        return float(self.beats[i]) + (second-self.seconds[i])*self.points[i][1]/60

    def second(self, tick):
        return self.beat_to_second(Fraction(tick, 96) if type(tick) is int else tick/96)

    def written_tick_seconds(self, tick):
        return 60/(96*self.bpm_at_beat(tick/96))

    def delta_ms(self, previous_tick, tick):
        return 1000*(self.second(tick)-self.second(previous_tick))

    def tick_after(self, tick, delta_ms):
        raw = self.second_to_beat(self.second(tick)+delta_ms/1000)*96
        nearest = math.floor(raw+.5)
        if abs(raw-nearest) > 1e-6:
            raise ValueError('MOMENT is not on the 96-tick beat grid')
        return nearest


def advance_ticks(timing, second, ticks):
    return timing.beat_to_second(timing.second_to_beat(second)+ticks/96)


def duration_end(timing, tick, duration, *, slide=False):
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('invalid duration')
    return timing.second(tick+(96 if slide else 0)) + duration*timing.written_tick_seconds(tick)


def split_duration(start, end, lengths):
    if not lengths or any(not math.isfinite(x) or x <= 0 for x in lengths) or end <= start:
        raise ValueError('invalid chain lengths/time')
    total, cursor = math.fsum(lengths), start
    result = []
    consumed = 0.
    for i, length in enumerate(lengths):
        consumed += length
        stop = end if i == len(lengths)-1 else start+(end-start)*consumed/total
        result.append((cursor, stop))
        cursor = stop
    return tuple(result)
