"""Z block geometry; defaults are owned by the public CONFIG."""


def z_stride(value=None, *, target_beats=None):
    if value is None:
        from maidata.entry import CONFIG
        value = CONFIG['draft_z_stride_beats']
    if type(value) is not int or value not in (1, 2, 4, 8, 16):
        raise ValueError('draft_z_stride_beats must be one of 1, 2, 4, 8, 16')
    if target_beats is not None and target_beats % value:
        raise ValueError('target_beats must be divisible by draft_z_stride_beats')
    return value


def integer_offsets(lower, upper, z_stride_beats):
    """Inclusive bounds on the integer Z grid, relative to a coverage/window start."""
    stride = z_stride(z_stride_beats)
    if stride not in (1, 2):
        raise ValueError('integer D centers require stride one or two')
    if type(lower) is not int or type(upper) is not int or upper < lower:
        raise ValueError('integer Z bounds must be ordered integer beats')
    first = -((-lower)//stride)*stride
    return tuple(range(first, upper+1, stride))


def integer_z_count(draft_beats, z_stride_beats):
    """Include the complete-tail endpoint without changing the anchor phase."""
    if type(draft_beats) is not int or draft_beats < 0:
        raise ValueError('integer Draft coverage must be a nonnegative beat count')
    stride = z_stride(z_stride_beats, target_beats=draft_beats)
    integer_offsets(0, 0, stride)
    return draft_beats//stride+1
