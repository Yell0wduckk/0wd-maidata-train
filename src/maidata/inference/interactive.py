"""python -m maidata.inference.interactive: a resident joint decoder for editor requests.

The process loads only M and reuses the audio and Draft/Plan caches written by
``preencode`` / ``predraft`` (it never runs E, D or P). It then answers JSON-line
requests on stdin with JSON-line events on stdout:

- ``{"op": "chart", "revision": r, "ir": <maimai.chart_ir.v1>}`` replaces the
  editor chart that later requests condition on.
- ``{"op": "suggest", "id": i, "revision": r, "mode": "continue"|"range",
  "start_tick": a, "end_tick": b, "plan_source": "predicted"|"none", ...}`` generates onsets in ``[a, b)`` after the
  chart's content before ``a``. A newer suggest cancels the running one.
- ``{"op": "cancel", "id": i}`` and ``{"op": "close"}``.

Self-attention K/V is kept per target window. A request reuses the longest
unchanged expanded-row prefix of that window and computes only the remaining
rows in one causal pass, so unchanged history is never recomputed and an edit
recomputes exactly its suffix.
"""
import argparse
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import json
import math
import sys
import threading
import time


class Cancelled(Exception):
    pass


def parser():
    from .precompute import add_runtime, add_audio
    p = argparse.ArgumentParser(description='Resident joint decoder for MiaCode continuation and range generation.')
    add_runtime(p)
    add_audio(p)
    p.add_argument('--difficulty', type=int, choices=range(1, 8), required=True)
    p.add_argument('--level', type=float, required=True)
    p.add_argument('--geometry', required=True)
    p.add_argument('--slide-data', required=True)
    p.add_argument('--window-contexts', type=int, default=4, help='Target windows whose self K/V stay resident')
    return p


@dataclass
class WindowContext:
    memory: object
    rows: tuple = ()     # expanded (input, beat, slide_beat) rows
    entries: int = 0     # cached rows; entry j was completed with family(rows[j+1])
    cache: object = None
    plan_source: str = 'predicted'


def cache_length(cache):
    return 0 if cache is None else cache.layers[0].valid.shape[1]


def slice_cache(cache, length):
    """Views of the first rows only; K/V are immutable, so nothing is copied."""
    from maidata.decoder.types import KV, DecoderCache
    if cache is None or length == 0:
        return None
    if length == cache_length(cache):
        return cache
    return DecoderCache(tuple(KV(kv.key[:, :, :length], kv.value[:, :, :length], kv.valid[:, :length],
                                 kv.query_layout) for kv in cache.layers), cache.memory)


def expanded_rows(tokens, beats, timing, boundary_index):
    from maidata.decoder.sequence import expand_inputs
    seq = expand_inputs(tokens, beats, timing, boundary_index=boundary_index)
    return tuple(zip(seq['tokens'], seq['beats'], seq['slide_beats']))


def extend_cache(decoder, context, rows, *, device):
    """Make context.cache cover rows[:-1]; the last row is fed by decode_window.

    An entry's back-layer K/V depends on the type of the row after it, so entry j
    is reusable iff rows[:j+1] match and the following row keeps its family.
    """
    import torch
    from maidata.chart.types import GENERATED_TYPES
    from maidata.decoder.batching import pack_tokens
    from maidata.decoder.sequence import family
    target = len(rows) - 1
    keep, limit = 0, min(target, context.entries)
    while (keep < limit and context.rows[keep] == rows[keep]
           and family(context.rows[keep+1][0]) == family(rows[keep+1][0])):
        keep += 1
    cache = slice_cache(context.cache, keep)
    if keep < target:
        part = rows[keep:target]
        batch = pack_tokens([[r[0] for r in part]], [[r[1] for r in part]], device=device,
                            slide_positions=[[r[2] for r in part]])
        types = torch.tensor([[GENERATED_TYPES.index(family(r[0])) for r in rows[keep+1:target+1]]], device=device)
        cache = decoder.complete(decoder.begin(batch, context.memory, cache=cache), types).cache
    context.rows, context.entries, context.cache = tuple(rows), target, cache
    return keep, target - keep


def chart_groups(ir, timing, duration):
    """Real onset groups of an editor chart; notes the model cannot represent are skipped."""
    from maidata.chart import encode_chart
    from maidata.chart.types import ChartError
    from maidata.decoder.windows import real_onsets
    from maidata.time import TimingMap
    if ir.get('schema') != 'maimai.chart_ir.v1':
        raise ValueError('expected maimai.chart_ir.v1')
    clock = TimingMap.from_ir(ir)
    if (clock.points != timing.points
            or not math.isclose(clock.first_seconds, timing.first_seconds, rel_tol=0, abs_tol=1e-9)):
        raise ValueError('Chart BPM/first changed since this session opened; restart the session')
    skipped = 0
    try:
        encoded = encode_chart(ir)
    except ChartError:
        moments = []
        for moment in ir['moments']:
            groups = []
            for group in moment['groups']:
                notes = []
                for note in group['notes']:
                    try:
                        encode_chart(dict(ir, moments=[dict(moment, groups=[dict(group, notes=[note])])]))
                        notes.append(note)
                    except ChartError:
                        skipped += 1
                if notes:
                    groups.append(dict(group, notes=notes))
            if groups:
                moments.append(dict(moment, groups=groups))
        encoded = encode_chart(dict(ir, moments=moments))
    groups = real_onsets(encoded.tokens, timing, sequence_start_tick=encoded.start_tick)
    kept = tuple(g for g in groups if g.tick >= 0 and timing.second(g.tick) < duration)
    skipped += sum(len([t for t in g.content if t.type == 'NOTE']) for g in groups if g not in kept)
    return rebase(kept, timing), skipped


def rebase(groups, timing):
    """Each real onset's delta is measured from the previous kept real onset."""
    result, previous = [], None
    for g in groups:
        anchor = previous if previous is not None else min(0, g.tick)
        result.append(replace(g, delta_ms=timing.delta_ms(anchor, g.tick)))
        previous = g.tick
    return tuple(result)


def window_tokens(sequence, start_tick, timing):
    """Window inputs and their absolute coordinates, as in decoder.windows.build_window."""
    from maidata.chart.types import Moment
    cursor, beats = start_tick, [start_tick / 96]
    for token in sequence.tokens[1:]:
        if isinstance(token, Moment):
            cursor = timing.tick_after(cursor, token.delta_ms)
        beats.append(cursor / 96)
    return sequence.tokens, tuple(beats)


def continuation_limits(start, end, request, *, audio_end_tick, following_tick=None):
    """Independent output bounds on the global 96-tick beat grid."""
    limits = [(end, 'requested_end'), (audio_end_tick, 'audio_end')]
    if following_tick is not None:
        limits.append((following_tick, 'next_note'))
    for key, default, unit, reason in (
            ('max_beats', 4, 96, 'beat_limit'),
            ('max_four_beat_blocks', 1, 384, 'four_beat_limit')):
        value = request.get(key, default)
        if type(value) is not int or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer; 0 disables')
        if value:
            limits.append((start // unit * unit + value * unit, reason))
    # Prefer named bounds over a duplicate caller-provided end.
    tick = min(t for t, _ in limits)
    reason = next(r for t, r in reversed(limits) if t == tick)
    return tick, reason


class Interactive:
    def __init__(self, args, *, emit):
        import torch
        from maidata.physical import Geometry
        from maidata.model import decoder_time_distributions
        from .cache import JointCache, draft_identity, identity_key
        from .checkpoint import JointCheckpoint
        from .precompute import read_timing, requested_audio_identity
        self.emit = emit
        self.device = torch.device(args.device)
        self.precision = args.precision
        self.timing_data, self.timing = read_timing(args)
        checkpoint = JointCheckpoint(args.joint_checkpoint)
        cache = JointCache(args.cache_dir, checkpoint, device=self.device, precision=args.precision)
        encoded = cache.find_audio(requested_audio_identity(checkpoint, args, self.timing, self.device))
        if encoded is None:
            raise FileNotFoundError('No matching audio cache; run audio preencoding for this audio/timing/checkpoint first')
        draft = cache.root/'draft'/(identity_key(draft_identity(checkpoint, encoded.key, difficulty=args.difficulty,
            level=args.level, device=self.device, precision=args.precision))+'.pt')
        if not draft.is_file():
            raise FileNotFoundError('No matching Draft/Plan cache; run Draft/Plan predecoding for this difficulty/level first')
        models = checkpoint.models(device=self.device, components=('decoder',))
        self.decoder, self.config = models.decoder, models.config
        self.duration = encoded.duration
        with self.autocast():
            self.conditions = cache.draft(encoded, difficulty=args.difficulty, level=args.level)
            self.audio_kv = self.decoder.prepare_audio(self.conditions.audio)
        self.geometry = Geometry.from_resources(args.geometry, args.slide_data)
        self.distributions = decoder_time_distributions(self.config)
        self.window_ticks = self.config['target_beats'] * 96
        self.contexts = OrderedDict()
        self.max_contexts = max(1, args.window_contexts)
        self.rule_starts = {}
        self.groups, self.skipped, self.revision = (), 0, None

    def autocast(self):
        import contextlib
        import torch
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == 'bf16'))
        return stack

    def ready(self):
        return dict(windows=len(self.conditions.starts), target_beats=self.config['target_beats'],
                    duration_seconds=self.duration, history_beats=self.config['chart_history_beats'])

    def set_chart(self, revision, ir):
        self.groups, self.skipped = chart_groups(ir, self.timing, self.duration)
        self.revision = revision

    def context(self, index, plan_source='predicted'):
        import torch
        from maidata.draft.windows import select_windows
        if plan_source not in ('predicted', 'none'):
            raise ValueError('interactive plan_source must be predicted or none')
        if index in self.contexts and self.contexts[index].plan_source == plan_source:
            self.contexts.move_to_end(index)
            return self.contexts[index]
        c, conditions = self.config, self.conditions
        start = conditions.starts[index]
        with self.autocast():
            selected = select_windows(conditions.z, torch.tensor([0], device=self.device),
                torch.tensor([start], device=self.device, dtype=torch.float32), torch.tensor([True], device=self.device),
                target_beats=c['target_beats'], before_beats=c['decoder_z_before_beats'],
                after_beats=c['decoder_z_after_beats'], z_stride_beats=c['draft_z_stride_beats']
                ).memory if c['decoder_z_cross_attention'] else conditions.z
            audio_index = index if conditions.audio_scope == 'window' else 0
            plans, valid = conditions.plans[index:index+1], conditions.plan_valid[index:index+1]
            if plan_source == 'none':
                plans, valid = torch.zeros_like(plans), torch.zeros_like(valid)
            memory = self.decoder.prepare_memory(self.audio_kv, torch.tensor([audio_index], device=self.device),
                selected, plans, valid)
        # Self K/V depends on Plan as well as chart rows. Never reuse it after
        # changing the conditioning, and retain the predicted Draft cache intact.
        context = self.contexts[index] = WindowContext(memory, plan_source=plan_source)
        self.contexts.move_to_end(index)
        while len(self.contexts) > self.max_contexts:
            self.contexts.popitem(last=False)
        return context

    def window_sequence(self, groups, index, floor, end):
        from maidata.chart.windows import window_from_groups
        start = index * self.window_ticks
        return window_from_groups(tuple((g.tick, g.content) for g in groups if g.tick < floor), self.timing,
            start_tick=start, end_tick=end, audio_after_beats=self.config['audio_after_beats'],
            audio_end_seconds=self.duration, history_present=True)

    def rules(self, groups, index, floor, end):
        """Rule state after the chart before floor, resumed from the latest valid window start."""
        from .tail_collision import GenerationPhysicalState
        state = GenerationPhysicalState(self.timing, self.geometry, audio_end_seconds=self.duration)
        first = 0
        for k in range(index, 0, -1):
            saved = self.rule_starts.get(k)
            if saved is not None and saved[0] == tuple(g for g in groups if g.tick < k * self.window_ticks):
                state.restore(saved[1])
                first = k
                break
        for k in range(first, index + 1):
            last = k == index
            window_end = end if last else (k + 1) * self.window_ticks
            sequence = self.window_sequence(groups, k, floor if last else window_end, window_end)
            state.inject_boundary(sequence.tokens[0], start_tick=k * self.window_ticks,
                                  history_present=True, window_end=window_end)
            for token in sequence.tokens[1:]:
                state.accept_source(token)
            if not last:
                self.rule_starts[k + 1] = (tuple(g for g in groups if g.tick < window_end), state.snapshot())
        return state, sequence

    def suggest(self, request, cancelled):
        from maidata.chart import token_to_dict
        from maidata.chart.types import Moment
        from maidata.decoder.session import DecodeSession
        from maidata.decoder.windows import history_prefix, real_onsets
        from .generation import decode_window, repetition_history
        from .options import GenerationOptions
        from .render import render_group, render_span
        from .sampling import Sampler
        if request.get('revision') != self.revision:
            raise ValueError('suggest refers to a chart revision this session has not received')
        plan_source = request.get('plan_source', 'predicted')
        if plan_source not in ('predicted', 'none'):
            raise ValueError('interactive plan_source must be predicted or none')
        mode = request.get('mode', 'continue')
        start, end = int(request['start_tick']), int(request['end_tick'])
        if mode not in ('continue', 'range') or start < 0 or end <= start:
            raise ValueError('invalid suggest mode or tick range')
        budget = max(1, int(request.get('max_tokens', 32))) if mode == 'continue' else None
        audio_end = math.ceil(self.timing.second_to_beat(self.duration) * 96)
        limit_reason = 'requested_end'
        if mode == 'continue':
            following = next((g.tick for g in self.groups if g.tick >= start), None)
            end, limit_reason = continuation_limits(start, int(request['end_tick']), request,
                audio_end_tick=audio_end, following_tick=following)
        defaults = GenerationOptions()
        options = GenerationOptions(temperature=float(request.get('temperature', defaults.temperature)),
            top_k=int(request.get('top_k', defaults.top_k)), top_p=float(request.get('top_p', defaults.top_p)),
            seed=int(request.get('seed', defaults.seed)),
            repetition_window=request.get('repetition_window', defaults.repetition_window),
            repetition_penalty=float(request.get('repetition_penalty', defaults.repetition_penalty)))
        sampler = Sampler(temperature=options.temperature, seed=options.seed, top_k=options.top_k,
                          top_p=options.top_p, physical_hard_mask=options.physical_hard_mask,
                          repetition_window=options.repetition_window, repetition_penalty=options.repetition_penalty)
        groups = [g for g in self.groups if g.tick < start]
        repeat_prefix = repetition_history(groups, options.repetition_window)
        generated, produced, reused, computed, bypass, repairs = [], 0, 0, 0, 0, []
        stop_reason = limit_reason
        floor = start
        while floor < end:
            index = floor // self.window_ticks
            if index >= len(self.conditions.starts):
                break
            window_start, window_end = index * self.window_ticks, (index + 1) * self.window_ticks
            if cancelled.is_set():
                raise Cancelled()
            state, sequence = self.rules(groups, index, floor, window_end)
            history, positions, boundary, _ = history_prefix(groups, self.timing, start_tick=window_start,
                history_beats=self.config['chart_history_beats'], dropped=False)
            current, beats = window_tokens(sequence, window_start, self.timing)
            tokens, coordinates = history + current, positions + beats
            rows = expanded_rows(tokens, coordinates, self.timing, len(history))
            context = self.context(index, plan_source)
            session = DecodeSession(state)
            origin = state.grammar.state.cursor

            def stop(accepted, rules):
                if cancelled.is_set():
                    raise Cancelled()
                return (budget is not None and produced + len(accepted) >= budget
                        and rules.grammar.state.phase in ('done', 'track_done'))

            with self.autocast():
                kept, fresh = extend_cache(self.decoder, context, rows, device=self.device)
                session.cache = context.cache
                sampler.floor_tick, sampler.physical_hard_mask = floor, options.physical_hard_mask
                new, report = decode_window(self.decoder, context.memory, session, sequence.tokens[0], sampler,
                    self.distributions, self.timing, start_tick=window_start, end_tick=window_end,
                    cap_tick=window_end + self.config['audio_after_beats'] * 96, device=self.device,
                    options=options, window=index + 1, first_token=tokens[-1], stop=stop,
                    repetition_prefix=repeat_prefix, onset_end_tick=end if end <= window_end else None)
            reused, computed = reused + kept, computed + fresh
            bypass += len(report['physical_bypass'])
            repairs.extend(report['tail_repairs'])
            if new:
                cursor, more = origin, []
                for token in new:
                    if isinstance(token, Moment):
                        cursor = self.timing.tick_after(cursor, token.delta_ms)
                    more.append(cursor / 96)
                all_rows = expanded_rows(tokens + tuple(new), coordinates + tuple(more), self.timing, len(history))
                entries = min(len(all_rows) - 1, cache_length(session.cache))
                context.rows, context.entries = all_rows, entries
                context.cache = slice_cache(session.cache, entries)
                onsets = list(real_onsets((Moment(0), *new), self.timing, sequence_start_tick=origin))
                if onsets:
                    previous = groups[-1].tick if groups else min(0, onsets[0].tick)
                    onsets[0] = replace(onsets[0], delta_ms=self.timing.delta_ms(previous, onsets[0].tick))
                groups.extend(onsets)
                generated.extend(onsets)
                produced += len(new)
                repeat_prefix = (repeat_prefix + tuple(new))[-options.repetition_window:]
            if report.get('stopped') or budget is not None and produced >= budget:
                if report.get('stop_reason') != 'onset_limit':
                    stop_reason = 'token_limit'
                break
            floor = window_end
        if mode == 'range':
            span = (start, end)
        elif generated:
            last = generated[-1].tick
            following = sorted(int(b) for b in request.get('boundaries', ()) if int(b) > last)
            span = (start, min(end, following[0] if following else (last // 96 + 1) * 96))
        else:
            # A silent prefix is still a valid continuation: accepting its
            # commas advances the caret so short limits cannot trap an empty chart.
            span = (start, end) if end > start else None
        controls = [dict(kind='bpm', bpm=bpm, position_beats=dict(num=b.numerator, den=b.denominator))
                    for b, bpm in self.timing.points]
        text = render_span(generated, *span, controls) if span else ''
        from .__main__ import ir_metadata
        from .generation import chart_from_groups
        from maidata.chart import decode_chart
        fragment_ir = decode_chart(chart_from_groups(generated, self.timing,
            ir_metadata(self.timing, self.duration)), self.geometry)
        return dict(id=request.get('id'), revision=self.revision, mode=mode, start_tick=start,
            end_tick=span[1] if span else start, text=text, ir_fragment=fragment_ir,
            onsets=[dict(tick=g.tick, text=render_group(g),
                         **({'content': [token_to_dict(t) for t in g.content]} if request.get('content') else {}))
                    for g in generated],
            tokens=produced, reused_rows=reused, computed_rows=computed,
            skipped_notes=self.skipped, physical_bypass=bypass, seed=options.seed,
            limit_tick=end, stop_reason=stop_reason, plan_source=plan_source, tail_repairs=repairs,
            repetition_window=options.repetition_window, repetition_penalty=options.repetition_penalty)


class Server:
    """Main thread reads requests; one worker runs the model. Newer suggests supersede."""
    def __init__(self, session, emit):
        self.session, self.emit = session, emit
        self.pending = deque()
        self.condition = threading.Condition()
        self.running = None  # (id, cancelled event)

    def submit(self, request):
        op = request.get('op')
        with self.condition:
            if op == 'suggest':
                if self.running is not None:
                    self.running[1].set()
                for old in [r for r in self.pending if r.get('op') == 'suggest']:
                    self.pending.remove(old)
                    self.emit('cancelled', id=old.get('id'))
            elif op == 'cancel':
                if self.running is not None and self.running[0] == request.get('id'):
                    self.running[1].set()
                for old in [r for r in self.pending if r.get('op') == 'suggest' and r.get('id') == request.get('id')]:
                    self.pending.remove(old)
                    self.emit('cancelled', id=old.get('id'))
                return
            self.pending.append(request)
            self.condition.notify()

    def work(self):
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                request = self.pending.popleft()
                cancelled = threading.Event()
                self.running = (request.get('id'), cancelled)
            op = request.get('op')
            if op == 'close':
                return
            began = time.monotonic()
            try:
                if op == 'chart':
                    self.session.set_chart(request.get('revision'), request['ir'])
                    self.emit('chart_ready', revision=self.session.revision, onsets=len(self.session.groups),
                              skipped_notes=self.session.skipped)
                elif op == 'suggest':
                    result = self.session.suggest(request, cancelled)
                    self.emit('result', elapsed_ms=round(1000*(time.monotonic()-began)), **result)
                else:
                    raise ValueError(f'unknown op: {op}')
            except Cancelled:
                self.emit('cancelled', id=request.get('id'))
            except Exception as exc:  # the session survives a bad request
                self.emit('error', id=request.get('id'), op=op, message=str(exc))
            finally:
                with self.condition:
                    self.running = None


def main(argv=None, *, stdin=None, stdout=None):
    import io
    args = parser().parse_args(argv)
    stdin = stdin or io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    stdout = stdout or sys.stdout
    lock = threading.Lock()

    def emit(event, **fields):
        with lock:
            stdout.write(json.dumps(dict(event=event, **fields), ensure_ascii=True, allow_nan=False) + '\n')
            stdout.flush()

    try:
        session = Interactive(args, emit=emit)
    except Exception as exc:
        emit('error', id=None, op='open', message=str(exc), fatal=True)
        raise SystemExit(1)
    emit('ready', **session.ready())
    server = Server(session, emit)
    worker = threading.Thread(target=server.work, daemon=True)
    worker.start()
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            emit('error', id=None, op=None, message=f'invalid request JSON: {exc}')
            continue
        server.submit(request)
        if request.get('op') == 'close':
            break
    else:
        server.submit(dict(op='close'))
    worker.join()


if __name__ == '__main__':
    main()
