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


def assemble_full_p0(local, starts, window_to_audio, coverages, *, p0_rate, layout=None):
    """Consume any public values/beats/valid object, without importing Encoder.

    Each audio coverage window must occur exactly once. Only its target center is
    selected; padding remains invalid and all three tensors use identical indices.
    """
    validate_memory(local)
    from .layout import prepare_p0, to_device
    if layout is None:
        layout = to_device(prepare_p0(local.beats.detach().cpu(), local.valid.cpu(),
            starts.detach().cpu(), window_to_audio.cpu(), coverages, p0_rate=p0_rate), local.values.device)
    if tuple(local.beats.shape) != layout['shape']:
        raise ValueError('P0 layout shape mismatch')
    indices = layout['indices']
    valid = local.valid.flatten()[indices] & layout['present']
    values = local.values.flatten(0, 1)[indices].masked_fill(~valid[..., None], 0)
    beats = local.beats.flatten()[indices].masked_fill(~layout['present'], 0)
    return BeatMemory(values, beats, valid)



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


def select_windows(z, window_groups, starts, window_valid, *, target_beats, before_beats, after_beats, layout=None):
    """Separate calls select R, P or Decoder contexts; no query regeneration."""
    validate_memory(z)
    from .layout import prepare_selection, to_device
    if layout is None:
        layout = to_device(prepare_selection(z.beats.detach().cpu(), z.valid.cpu(), window_groups.cpu(),
            starts.detach().cpu(), window_valid.cpu(), target_beats=target_beats,
            before_beats=before_beats, after_beats=after_beats), z.values.device)
    if tuple(z.beats.shape) != layout['shape']:
        raise ValueError('Z layout shape mismatch')
    values = z.values.flatten(0, 1)[layout['indices']].masked_fill(~layout['valid'][..., None], 0)
    return WindowLatents(BeatMemory(values, layout['beats'], layout['valid']),
                         starts, target_beats, window_valid)
