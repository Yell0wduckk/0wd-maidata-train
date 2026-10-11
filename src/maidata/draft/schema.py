"""Reconstruction-only heads derived from public field vocabularies."""
from maidata.chart.types import KINDS, FLAGS

FIELD_SIZES = {'kind': len(KINDS), 'route_id': 71, 'track_break': 2,
               **dict.fromkeys(FLAGS, 2)}
QUERY_ROLES = ('MOMENT', 'NOTE_KIND', 'NOTE_FLAGS', 'ROUTE_ID', 'TRACK_BREAK', 'COUNT')
MOMENT_CLASSES = 96
POSITIVE_COUNT_CLASSES = 96
