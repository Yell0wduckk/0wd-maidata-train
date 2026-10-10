"""Read only timing controls from simai; note content never bounds audio."""
from fractions import Fraction
import re

import numpy as np


def fields(text):
    matches = list(re.finditer(r"^&([^=\r\n]+)=", text, re.M))
    return {m[1]: text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def chart_timing(body, initial_bpm, first):
    # Square brackets describe note durations, not the chart clock.
    body = re.sub(r"\[[^\]]*\]", "", body)
    beat, step = Fraction(0), Fraction(1)
    bpm = float(initial_bpm)
    points = {Fraction(0): bpm}
    for token in re.finditer(r"\(([^()]*)\)|\{([^{}]*)\}|(,)", body):
        if token[1] is not None:
            bpm = float(token[1])
            if not np.isfinite(bpm) or bpm <= 0:
                raise ValueError(f"invalid BPM {token[1]}")
            points[beat] = bpm
        elif token[2] is not None:
            value = token[2].strip()
            if value.startswith('#'):
                step = Fraction(value[1:]) * Fraction(str(bpm)) / 60
            else:
                step = 4 / Fraction(value)
            if step <= 0:
                raise ValueError(f"invalid subdivision {value}")
        else:
            beat += step
    # Repeated declarations of the active BPM do not create new variants.
    compact = []
    for position, tempo in sorted(points.items()):
        if not compact or tempo != compact[-1][2]:
            compact.append([position.numerator, position.denominator, tempo])
    result = {'first_seconds': float(first), 'bpm': compact}
    TempoMap(result)
    return result


class TempoMap:
    def __init__(self, timing):
        self.first = float(timing['first_seconds'])
        self.rational_beats = [Fraction(n,d) for n,d,_ in timing['bpm']]
        self.beats = np.array([n/d for n, d, _ in timing['bpm']], dtype=np.float64)
        self.bpms = np.array([v for _, _, v in timing['bpm']], dtype=np.float64)
        if (not np.isfinite(self.first) or len(self.beats) == 0 or self.beats[0] != 0
                or not np.isfinite(self.beats).all() or not np.isfinite(self.bpms).all()
                or np.any(self.bpms <= 0) or np.any(np.diff(self.beats) <= 0)):
            raise ValueError('invalid timing map')
        seconds = [self.first]
        for i in range(1,len(self.beats)):
            seconds.append(seconds[-1]+float(self.rational_beats[i]-self.rational_beats[i-1])*60/self.bpms[i-1])
        self.seconds = np.array(seconds,np.float64)

    def grid_seconds(self, k, r):
        # Rational beat subtraction reproduces the accepted grid's scalar clock
        # before float64 seconds and half-up PCM quantization.
        indices = np.maximum(0,np.searchsorted(self.beats,np.asarray(k)/r,side='right')-1)
        return np.array([self.seconds[i]+float(Fraction(int(n),r)-self.rational_beats[i])*60/self.bpms[i]
                         for n,i in zip(np.asarray(k).flat,indices.flat)],np.float64).reshape(np.shape(k))

    def to_seconds(self, beats):
        beats = np.asarray(beats, dtype=np.float64)
        i = np.maximum(0, np.searchsorted(self.beats, beats, side='right') - 1)
        return self.seconds[i] + (beats - self.beats[i]) * 60 / self.bpms[i]

    def to_beats(self, seconds):
        seconds = np.asarray(seconds, dtype=np.float64)
        i = np.maximum(0, np.searchsorted(self.seconds, seconds, side='right') - 1)
        return self.beats[i] + (seconds - self.seconds[i]) * self.bpms[i] / 60

    def tempo(self, beats):
        return self.bpms[np.maximum(0, np.searchsorted(self.beats, beats, side='right') - 1)]
