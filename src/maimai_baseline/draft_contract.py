"""Representation-specific draft geometry, independent of legacy chart latents."""
from collections.abc import Mapping
from dataclasses import dataclass


LEGACY_STYLE_ID = 'native-audio-draft128-v1'
BEAT_STYLE_ID = 'native-audio-draft128-beat-v1'


@dataclass(frozen=True)
class DraftGeometry:
    beats_per_token: int
    target_slots: int
    condition_slots: int
    first_slot: int

    @property
    def centers(self):
        return tuple((self.first_slot + i + .5) * self.beats_per_token
                     for i in range(self.condition_slots))


def geometry(value=BEAT_STYLE_ID):
    """Resolve a style, model config or saved generator structure explicitly."""
    if isinstance(value, Mapping):
        name = value.get('chart_style', value.get('name'))
        if name is None:
            count = value.get('beats_per_token')
            name = BEAT_STYLE_ID if count == 1 else LEGACY_STYLE_ID if count == 4 else None
    else:
        name = value if isinstance(value, str) else getattr(value, 'chart_style', None)
    if name == BEAT_STYLE_ID:
        return DraftGeometry(1, 16, 48, -16)
    if name == LEGACY_STYLE_ID:
        return DraftGeometry(4, 4, 4, 0)
    raise ValueError('unsupported draft representation geometry')
