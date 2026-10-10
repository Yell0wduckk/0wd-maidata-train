"""Audio coverage, public P0 assembly and latent selection; no model ownership."""
from dataclasses import dataclass
import math

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import BeatMemory, WindowLatents, validate_memory


@dataclass(frozen=True)
class SongCoverage:
    first_beat: int
    audio_windows: int
    target_beats: int
    span_beats: float

    @property
    def draft_beats(self):
        return (self.audio_windows + 1) * self.target_beats

    def starts(self, *, include_tail):
        return tuple(self.first_beat + i * self.target_beats
                     for i in range(self.audio_windows + int(include_tail)))


def song_coverage(timing, audio_end_seconds, *, first_beat, target_beats):
    """Use the cache's first beat, real audio duration, and one complete tail window."""
    if (type(first_beat) is not int or type(target_beats) is not int or target_beats < 1
            or not math.isfinite(audio_end_seconds) or audio_end_seconds <= 0):
        raise ValueError('invalid audio coverage inputs')
    start, end = timing.second_to_beat(0), timing.second_to_beat(audio_end_seconds)
    if first_beat != math.floor(start):
        raise ValueError('first_beat must match the audio cache grid origin')
    return SongCoverage(first_beat, math.ceil((end - first_beat) / target_beats),
                        target_beats, end - start)


def audio_windows(coverages, *, device):
    """Encoder coverage only; the extra target tail has no real P0 to encode."""
    starts, mapping = [], []
    for song, coverage in enumerate(coverages):
        row = coverage.starts(include_tail=False)
        starts.extend(row)
        mapping.extend([song] * len(row))
    return (torch.tensor(starts, dtype=torch.float32, device=device),
            torch.tensor(mapping, dtype=torch.long, device=device))


def assemble_full_p0(local, starts, window_to_audio, coverages, *, p0_rate):
    """Consume any public values/beats/valid object, without importing Encoder.

    Each audio coverage window must occur exactly once. Only its target center is
    selected; padding remains invalid and all three tensors use identical indices.
    """
    validate_memory(local)
    if (starts.shape != (len(local.values),) or window_to_audio.shape != starts.shape
            or window_to_audio.dtype != torch.long or not coverages
            or ((window_to_audio < 0) | (window_to_audio >= len(coverages))).any()
            or not math.isfinite(p0_rate) or p0_rate <= 0):
        raise ValueError('invalid P0 window ownership')
    values, beats, valid = [], [], []
    for song, coverage in enumerate(coverages):
        indices = (window_to_audio == song).nonzero().flatten()
        order = indices[torch.argsort(starts[indices])]
        expected_starts = starts.new_tensor(coverage.starts(include_tail=False))
        if not torch.equal(starts[order], expected_starts):
            raise ValueError('P0 coverage must include each target center exactly once')
        count = coverage.target_beats * p0_rate
        if not float(count).is_integer():
            raise ValueError('P0 rate must give an integer center length')
        chunks, positions, masks = [], [], []
        for index in order:
            start = starts[index]
            selected = (local.beats[index] >= start) & (local.beats[index] < start + coverage.target_beats)
            expected = start + (torch.arange(int(count), device=start.device).float() + .5) / p0_rate
            actual = local.beats[index, selected]
            if actual.shape != expected.shape or not torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=0):
                raise ValueError('P0 target center coordinates do not match the configured rate')
            chunks.append(local.values[index, selected])
            positions.append(actual)
            masks.append(local.valid[index, selected])
        values.append(torch.cat(chunks))
        beats.append(torch.cat(positions))
        valid.append(torch.cat(masks))
    v = pad_sequence(valid, batch_first=True)
    return BeatMemory(pad_sequence(values, batch_first=True).masked_fill(~v[..., None], 0),
                      pad_sequence(beats, batch_first=True), v)


def draft_queries(coverages, audio_mapping):
    if (audio_mapping.ndim != 1 or audio_mapping.dtype != torch.long or not len(audio_mapping)
            or ((audio_mapping < 0) | (audio_mapping >= len(coverages))).any()):
        raise ValueError('invalid Draft audio mapping')
    rows, valid = [], []
    for index in audio_mapping.tolist():
        c = coverages[index]
        rows.append(c.first_beat + torch.arange(c.draft_beats, device=audio_mapping.device).float() + .5)
        valid.append(torch.ones(c.draft_beats, device=audio_mapping.device, dtype=torch.bool))
    return pad_sequence(rows, batch_first=True), pad_sequence(valid, batch_first=True)


def select_windows(z, window_groups, starts, window_valid, *, target_beats, before_beats, after_beats):
    """Separate calls select R, P or Decoder contexts; no query regeneration."""
    validate_memory(z)
    if (any(type(v) is not int or v < 0 for v in (target_beats, before_beats, after_beats))
            or target_beats == 0 or window_groups.ndim != 1 or not len(window_groups)
            or window_groups.dtype != torch.long or starts.shape != window_groups.shape
            or window_valid.shape != starts.shape or window_valid.dtype != torch.bool
            or not torch.isfinite(starts).all() or not torch.equal(starts, starts.round())
            or ((window_groups < 0) | (window_groups >= len(z.values))).any()):
        raise ValueError('invalid latent window geometry/mapping')
    count = before_beats + target_beats + after_beats
    desired = starts[:, None] - before_beats + torch.arange(count, device=starts.device).float() + .5
    out, mask = [], []
    for w, group in enumerate(window_groups.tolist()):
        source_indices = z.valid[group].nonzero().flatten()
        source_beats = z.beats[group, source_indices]
        if len(source_beats) and (not torch.equal(source_beats - .5, (source_beats - .5).round())
                                 or not torch.equal(source_beats[1:] - source_beats[:-1],
                                                    torch.ones_like(source_beats[1:]))):
            raise ValueError('Z must have contiguous absolute one-beat centers')
        if len(source_indices):
            indices = torch.searchsorted(source_beats.contiguous(), desired[w])
            safe = indices.clamp_max(len(source_indices) - 1)
            ok = (indices < len(source_indices)) & (source_beats[safe] == desired[w]) & window_valid[w]
            row = z.values[group, source_indices[safe]].masked_fill(~ok[..., None], 0)
        else:
            ok = torch.zeros(count, dtype=torch.bool, device=z.values.device)
            row = z.values[group, :1].expand(count, -1).masked_fill(~ok[..., None], 0)
        if window_valid[w] and not ok[before_beats:before_beats + target_beats].all():
            raise ValueError('valid target window is outside the full Draft coverage')
        out.append(row)
        mask.append(ok)
    return WindowLatents(BeatMemory(torch.stack(out), desired, torch.stack(mask)),
                         starts, target_beats, window_valid)
