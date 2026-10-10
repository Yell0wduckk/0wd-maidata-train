"""Explicit specialized supervision settings, independent of batch packing."""
from dataclasses import dataclass, asdict
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class SpecializedSupervision:
    enabled: bool = False
    hold_multiplier: float = 1.2
    slide_multiplier: float = 1.2

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("specialized supervision enabled must be boolean")
        for value in (self.hold_multiplier, self.slide_multiplier):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("specialized supervision multipliers must be finite and positive")

    def identity(self):
        return {**asdict(self), "effective_hold_multiplier": self.hold_multiplier if self.enabled else 1.0,
                "effective_slide_multiplier": self.slide_multiplier if self.enabled else 1.0}

    @classmethod
    def from_identity(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("missing specialized supervision configuration")
        try:
            result = cls(**{key: value[key] for key in ("enabled", "hold_multiplier", "slide_multiplier")})
        except (KeyError, TypeError) as error:
            raise ValueError("invalid specialized supervision configuration") from error
        if result.identity() != dict(value):
            raise ValueError("specialized supervision identity mismatch")
        return result


def from_args(args):
    get = args.get if isinstance(args, Mapping) else lambda name, default: getattr(args, name, default)
    return SpecializedSupervision(get("specialized_supervision", False),
                                  get("hold_loss_multiplier", 1.2), get("slide_loss_multiplier", 1.2))
