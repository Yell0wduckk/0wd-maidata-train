"""Reconstruction-only heads derived from public field vocabularies."""
from maidata.chart.types import GENERATED_TYPES, KINDS, BUTTON_PADS, TOUCH_PADS, FLAGS

ROW_TYPES = (*GENERATED_TYPES, 'SEGMENT_END')
FIELD_SIZES = {'kind': len(KINDS), 'route_id': 71, 'track_break': 2,
               **dict.fromkeys(FLAGS, 2)}
TIME_FIELDS = ('moment_position_tick', 'hold_duration_tick', 'slide_duration_tick')
QUERY_ROLES = GENERATED_TYPES
COUNT_CLASSES = 97
