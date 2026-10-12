"""Inference-only beat-zero view of accepted audio grids; never rewrite caches."""
import numpy as np

from maidata.preparation.timing import TempoMap


def beat_zero_grid(grid, timing):
    """Drop pre-chart beats, or prepend invalid zero beats before the WAV.

    Cached features retain their original timing and values. As with all grid
    inference, the cache must have been produced for the requested BPM/first.
    """
    first = int(grid['first_beat'])
    if first == 0:
        return grid
    count = max(1, first + len(grid['beat_centers']))
    source = np.arange(count) - first
    present = (source >= 0) & (source < len(grid['beat_centers']))
    result = dict(grid)
    for key in ('mel_patches', 'sample_valid', 'aux32_grid', 'aux32_centers'):
        original = grid[key]
        value = np.zeros((count, *original.shape[1:]), dtype=original.dtype)
        value[present] = original[source[present]]
        result[key] = value
    r = grid['mel_patches'].shape[1]
    result['first_beat'] = np.array(0, dtype=np.int64)
    result['beat_centers'] = np.arange(count, dtype=np.float32) + .5
    result['grid_k'] = np.arange(count*r, dtype=np.int64).reshape(count, r)
    # Preserve cached coordinates; only synthesize coordinates for missing beats.
    tempo = TempoMap(dict(first_seconds=timing.first_seconds,
        bpm=[[b.numerator, b.denominator, v] for b, v in timing.points]))
    missing_times = tempo.grid_seconds(result['grid_k'][~present], r)
    for key in ('sample_times_sec', 'sample_indices'):
        original = grid[key]
        value = np.zeros((count, *original.shape[1:]), dtype=original.dtype)
        value[present] = original[source[present]]
        value[~present] = (missing_times if key == 'sample_times_sec'
                          else np.floor(missing_times*24000+.5).astype(np.int64))
        result[key] = value
    return result
