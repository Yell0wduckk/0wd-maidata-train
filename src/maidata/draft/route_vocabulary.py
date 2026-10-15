"""Draft-only route targets; chart and main Decoder IDs stay Route-71."""

ROUTE71 = 'route71'
AXIS15_37 = 'axis15-37'

# Ordered by the minimum original Route-71 ID. Reflection fixes the start and
# opposite pads: at start 1, 2<->8, 3<->7, 4<->6, with direction symbols swapped.
AXIS15_GROUPS = (
    (0, 4), (1, 3), (2,), (5, 10), (6, 9), (7, 8),
    (11, 16), (12, 20), (13, 19), (14, 18), (15, 17),
    (21, 29), (22, 33), (23, 32), (24, 31), (25, 30),
    (26, 34), (27, 36), (28, 35),
    (37, 47), (38, 46), (39, 45), (40, 52), (41, 51),
    (42, 50), (43, 49), (44, 48), (53, 54),
    (55,), (56, 61), (57, 60), (58, 59),
    (62, 69), (63, 68), (64, 67), (65, 66), (70,),
)
ROUTE71_TO_AXIS15 = tuple(next(i for i, group in enumerate(AXIS15_GROUPS)
                             if route in group) for route in range(71))


def route_groups(vocabulary):
    if vocabulary == AXIS15_37:
        return AXIS15_GROUPS
    if vocabulary == ROUTE71:
        return tuple((i,) for i in range(71))
    raise ValueError('reconstruction_route_vocabulary must be route71 or axis15-37')


def route_class_count(vocabulary):
    return len(route_groups(vocabulary))


def route_targets(labels, vocabulary):
    """Adapt original Route-71 labels, preserving ignored slots and supports."""
    route_class_count(vocabulary)
    if vocabulary == ROUTE71:
        return labels
    mapping = labels.new_tensor(ROUTE71_TO_AXIS15)
    return mapping[labels.clamp_min(0)].masked_fill(labels == -100, -100)
