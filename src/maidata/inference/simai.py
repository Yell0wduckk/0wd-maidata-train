"""Bounded simai spacing adapted from the original baseline's 1/96-beat onset grid."""
from fractions import Fraction
import math
import re

SPACING_POLICY = "over-four-beats-4-remainder-96-tick-384-v2"
_INTERVAL = re.compile(r"\{(\d+)\}([^,]*)(,+)")


def bounded_render_document(document):
    """Align only export E; reject unsupported event timing before native rendering."""
    result = dict(document)
    for item in (*document.get("moments", ()), *document.get("controls", ())):
        value = item["position_beats"]
        beat = Fraction(value["num"], value["den"])
        if (beat * 96).denominator != 1:
            raise ValueError("simai event position is outside the 1/96-beat grid; refusing to move it")
    if "render_end_beat" in document:
        value = document["render_end_beat"]
        beat = Fraction(value["num"], value["den"])
        end = Fraction(math.ceil(beat * 96), 96)
        result["render_end_beat"] = {"num": end.numerator, "den": end.denominator}
    return result


def compact_spacing(chart):
    """Preserve intervals up to four beats; compact only longer intervals."""
    def replace(match):
        subdivision = int(match[1])
        if not 0 < subdivision <= 384:
            raise ValueError(f"simai subdivision exceeds export limit: {subdivision}")
        ticks = Fraction(len(match[3]) * 384, subdivision)
        if ticks.denominator != 1:
            raise ValueError("simai interval is outside the 1/96-beat grid")
        if ticks <= 4 * 96:
            return match[0]
        whole, remainder = divmod(ticks.numerator, 96)
        coarse, fine = divmod(remainder, 4)
        parts = []
        body = match[2]
        for division, count in ((4, whole), (96, coarse), (384, fine)):
            if count:
                parts.append(f"{{{division}}}" + body + "," * count)
                body = ""
        return "".join(parts)

    result = _INTERVAL.sub(replace, chart)
    if any(int(n) > 384 for n in re.findall(r"\{(\d+)\}", result)):
        raise ValueError("simai subdivision exceeds 384")
    return result


def four_beat_lines(chart, *, start_tick=0):
    """Split top-level timing intervals at exact four-beat boundaries.

    Commas and braces inside duration/slide expressions are never interpreted.
    This changes whitespace/subdivision spelling only, never musical time.
    start_tick (1/96 beat) places a spliced fragment on the chart's own bars.
    """
    # Meter directives consume the rest of their line in simai. Their newline
    # is syntax; ordinary layout newlines may be removed before reflowing.
    text = ''.join(part.replace('\r', '') if part.startswith('||') else ''.join(part.splitlines())
                   for part in re.split(r'(\|\|[^\r\n]*(?:\r?\n|$))', chart))
    out, pending, tick, subdivision, depth, i = [], [], start_tick % 384, 4, 0, 0
    active = None

    def emit(body, division):
        nonlocal active
        # Input subdivisions and output subdivisions differ when an interval
        # straddles a bar. Restore only when the next interval actually needs it.
        for line in body.splitlines(keepends=True):
            if active != division:
                out.append(f'{{{division}}}')
                active = division
            out.append(line)
            if line.endswith('\n'):
                active = None

    def advance(size):
        nonlocal tick, active
        tick += size
        if tick % 384 == 0:
            out.append('\n')
            active = None

    while i < len(text):
        c = text[i]
        if depth == 0 and text.startswith('||', i):
            end = text.find('\n', i)
            end = len(text) if end < 0 else end + 1
            pending.append(text[i:end]); i = end; continue
        if c == '[': depth += 1
        elif c == ']': depth -= 1
        if c == '{' and depth == 0:
            end = text.index('}', i)
            subdivision = int(text[i+1:end])
            if not 0 < subdivision <= 384 or 384 % subdivision:
                raise ValueError('four-beat layout requires a 1/96-beat subdivision')
            i=end+1; continue
        if c == ',' and depth == 0:
            duration = 384 // subdivision
            body = ''.join(pending)
            if tick % 384 + duration <= 384:
                emit(body + ',', subdivision)
                advance(duration)
            else:
                remaining = duration
                while remaining:
                    size = min(remaining, 384 - tick % 384)
                    unit = math.gcd(size, 384)
                    # A subdivision must precede the note, not follow it.
                    emit(body + ',' * (size // unit), 384 // unit)
                    body = ''
                    advance(size); remaining -= size
            pending = []
        else:
            pending.append(c)
        i += 1
    emit(''.join(pending), subdivision)
    return ''.join(out)


def persistent_subdivisions(chart):
    """Deduplicate within a line; retain explicit subdivisions at line starts."""
    active = None
    def replace(match):
        nonlocal active
        if match[0].startswith('||') or match[0] == '\n':
            active = None
            return match[0]
        if match[0].startswith('['):
            return match[0]
        subdivision = int(match[1])
        if subdivision == active:
            return ''
        active = subdivision
        return match[0]
    return re.sub(r'\|\|[^\n]*(?:\n|$)|\[[^\]]*\]|\n|\{(\d+)\}', replace, chart)
