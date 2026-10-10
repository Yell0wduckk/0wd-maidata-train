"""Style support follows the decoder's candidate-center boundary convention."""
import torch
from maimai_baseline.style_contract import STYLE_BEATS, STYLE_SLOTS


def style_geometry(start_tick, end_tick, song_end_tick, *, beats_per_token=STYLE_BEATS, slots=STYLE_SLOTS):
    """Return local centers, valid slots and actual duration in beats.

    Target windows are half-open; song_end_tick is an inclusive event center.
    A terminal onset exactly on a slot boundary belongs to the following slot,
    including slot zero of a terminal window whose duration is zero.
    """
    offsets = torch.arange(slots, device=start_tick.device, dtype=torch.float32) * beats_per_token
    starts, ends = start_tick.float(), end_tick.float()
    song_ends = torch.as_tensor(song_end_tick, device=starts.device).float()
    slot_starts = starts[:, None] + offsets[None] * 96.
    valid = (slot_starts < ends[:, None]) & (slot_starts <= song_ends[..., None])
    span = ((torch.minimum(ends, song_ends) - starts) / 96.).clamp(0., slots * beats_per_token)
    centers = (offsets + beats_per_token / 2)[None].expand(len(starts), -1)
    return centers, valid, span
