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
