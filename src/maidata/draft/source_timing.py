"""Planner-only source timing fallback; no note, network or cache production."""
from fractions import Fraction
from pathlib import Path
import re


def _cells(body):
    start, depth = 0, 0
    for i, char in enumerate(body):
        if char == '[':
            depth += 1
        elif char == ']':
            depth = max(0, depth - 1)
        elif char == ',' and depth == 0:
            yield body[start:i]
            start = i + 1
    yield body[start:]


def inote_end_beat(body, *, initial_bpm, validate_input=True):
    """First E, otherwise the last timing cell's exclusive end (simai semantics).

    Empty cells, including the one after a trailing comma, consume a step.
    Duration brackets are opaque; E1 is a touch, not a terminal. Fractional and
    seconds grids also follow the current input producer's timing convention.
    """
    if not body.strip():
        return Fraction(0)
    beat, step, bpm = Fraction(0), Fraction(1), Fraction(str(initial_bpm))
    if validate_input and (bpm <= 0):
        raise ValueError('source BPM must be positive')
    for cell in _cells(body):
        rest = cell.strip()
        while rest:
            if rest[0] in '({':
                closing = ')' if rest[0] == '(' else '}'
                end = rest.find(closing)
                if validate_input and (end < 0):
                    raise ValueError('unterminated source timing directive')
                value = rest[1:end].strip()
                if rest[0] == '(':
                    bpm = Fraction(value)
                    if validate_input and (bpm <= 0):
                        raise ValueError('source BPM must be positive')
                else:
                    step = (Fraction(value[1:]) * bpm / 60 if value.startswith('#')
                            else 4 / Fraction(value))
                    if validate_input and (step <= 0):
                        raise ValueError('source grid step must be positive')
                rest = rest[end + 1:].strip()
            elif rest.startswith('||'):
                meter = re.match(r'\|\|\s*\d+\s*/\s*\d+', rest)
                if validate_input and (meter is None):
                    raise ValueError('invalid source meter directive')
                rest = rest[meter.end():].strip()
            else:
                break
        if rest.upper() == 'E':
            return beat
        beat += step
    return beat


def read_source_end_beat(source_path, difficulty, *, validate_input=True):
    """Read only the requested difficulty when accepted metadata lacks its end."""
    if validate_input and (not source_path or difficulty is None):
        raise ValueError('source final timing-grid beat requires source_path and difficulty, or source_end_beat')
    text = Path(source_path).read_text(encoding='utf-8-sig')
    matches = list(re.finditer(r'^&([^=\r\n]+)=', text, re.M))
    fields = {m[1]: text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
              for i, m in enumerate(matches)}
    key = f'inote_{difficulty}'
    if validate_input and (key not in fields):
        raise ValueError(f'source final timing-grid beat requires {key}')
    return inote_end_beat(fields[key], initial_bpm=fields.get('wholebpm') or fields.get('bpm') or 120, validate_input=validate_input)
