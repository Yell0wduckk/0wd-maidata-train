"""Bounded simai spacing adapted from the original baseline's 1/96-beat onset grid."""
from fractions import Fraction
import math
import re

SPACING_POLICY = "over-four-beats-4-remainder-96-tick-384-v2"
_INTERVAL = re.compile(r"\{(\d+)\}([^,\n]*)(,+)")


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


def four_beat_lines(chart):
    """Split top-level timing intervals at exact four-beat boundaries.

    Commas and braces inside duration/slide expressions are never interpreted.
    This changes whitespace/subdivision spelling only, never musical time.
    """
    text = ''.join(chart.splitlines())
    out, pending, tick, subdivision, depth, i = [], [], 0, 4, 0, 0
    while i < len(text):
        c = text[i]
        if c == '[': depth += 1
        elif c == ']': depth -= 1
        if c == '{' and depth == 0:
            end = text.index('}', i)
            subdivision = int(text[i+1:end])
            if not 0 < subdivision <= 384 or 384 % subdivision:
                raise ValueError('four-beat layout requires a 1/96-beat subdivision')
            pending.append(text[i:end+1]); i=end+1; continue
        if c == ',' and depth == 0:
            duration = 384 // subdivision
            if tick % 384 + duration <= 384:
                out.extend(pending); out.append(','); tick += duration
                if tick % 384 == 0: out.append('\n')
            else:
                remaining = duration
                body = re.sub(r'\{\d+\}', '', ''.join(pending))
                while remaining:
                    size = min(remaining, 384 - tick % 384)
                    unit = math.gcd(size, 384)
                    # A subdivision must precede the note, not follow it.
                    out.append('{' + str(384 // unit) + '}' + body + ',' * (size // unit))
                    body = ''
                    tick += size; remaining -= size
                    if tick % 384 == 0: out.append('\n')
                out.append('{' + str(subdivision) + '}')
            pending = []
        else:
            pending.append(c)
        i += 1
    return ''.join(out + pending)
