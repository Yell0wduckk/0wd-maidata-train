"""Operational generation defaults shared by the CLI and Python API."""
from dataclasses import dataclass
import secrets


def resolve_seed(seed):
    """Resolve the random sentinel once, without changing global RNG state."""
    return secrets.randbelow(2**31) if seed == -1 else seed


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = .4
    seed: int = 20261010
    max_tokens_per_window: int = 512
    max_tokens_per_moment: int = 20
    max_backtracks_per_window: int = 128
    rollback_tokens: int = 16
    physical_hard_mask: bool = True
    trace: bool = False
    top_k: int = 0
    top_p: float = 1.
    max_slide_judge_areas: int = 40
    repetition_window: int = 32
    repetition_penalty: float = 1.

    pad_repetition_window: int | None = None
    pad_repetition_penalty: float | None = None
    route_repetition_window: int | None = None
    route_repetition_penalty: float | None = None

    @property
    def repetition_settings(self):
        return {f'{field}_repetition_{suffix}': getattr(self, f'{field}_repetition_{suffix}')
                for field in ('pad', 'route') for suffix in ('window', 'penalty')}

    @property
    def repetition_history_limit(self):
        return max(self.pad_repetition_window, self.route_repetition_window)

    def __post_init__(self):
        validate_slide_limit(self.max_slide_judge_areas)
        # Freeze the actual seed for the whole run and its provenance report.
        object.__setattr__(self, 'seed', resolve_seed(self.seed))
        for name, value in resolve_repetition(self.repetition_window, self.repetition_penalty,
                                              **self.repetition_settings).items():
            object.__setattr__(self, name, value)


def validate_slide_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError('max_slide_judge_areas must be a positive integer')


def resolve_repetition(repetition_window=32, repetition_penalty=1., **settings):
    """Explicit field settings override legacy shared values, including zero."""
    resolved = dict(repetition_window=repetition_window, repetition_penalty=repetition_penalty)
    for field in ('pad', 'route'):
        for suffix, fallback in (('window', repetition_window), ('penalty', repetition_penalty)):
            key = f'{field}_repetition_{suffix}'
            resolved[key] = fallback if settings.get(key) is None else settings[key]
    for key, value in resolved.items():
        if key.endswith('window'):
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        elif not 0 <= value < float('inf'):
            raise ValueError(f'{key} must be finite and nonnegative')
    return {key: value for key, value in resolved.items() if key.startswith(('pad_', 'route_'))}
