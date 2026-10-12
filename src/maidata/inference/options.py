"""Operational generation defaults shared by the CLI and Python API."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.
    seed: int = 20261010
    max_tokens_per_window: int = 512
    max_tokens_per_moment: int = 20
    max_backtracks_per_window: int = 128
    rollback_tokens: int = 16
    physical_hard_mask: bool = True
    trace: bool = False
    top_k: int = 0
    top_p: float = 1.
