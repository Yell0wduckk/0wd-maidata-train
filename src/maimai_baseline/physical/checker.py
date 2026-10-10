from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from contextlib import contextmanager
from typing import Any, Mapping

import torch

from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_baseline.physical.spatial import region_byte_lut


_EXTENSION: Any | None = None
_EXTENSION_ERROR: BaseException | None = None
_LOCK = threading.Lock()
_PENDING_TIMINGS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_PENDING_PERMUTE_TIMINGS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_PENDING_OPPORTUNITY_TIMINGS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_PENDING_COUNT_TIMINGS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_COUNT_DETAILS = {name: [] for name in ("count_context", "count_build", "count_index", "count_query", "count_duration", "count_guard")}
_TIMING_HISTORY: dict[str, list[float]] = {
    **{name: [] for name in _COUNT_DETAILS},
    "count_matrix_analysis": [],
    "view_permutation": [],
    "opportunity_preparation": [],
    "checker_kernel": [],
}
_TIMING_SKIP = 0
_TIMING_TAKE: int | None = None
_TIMING_PRODUCTION_CALLS = 0
_TIMING_ACCEPT_CURRENT = False


def _extension_name(sources: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.read_bytes())
    return "maimai_physical_bitplane_" + digest.hexdigest()[:12]


def load_cuda_extension() -> Any:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError(
            "CUDA Physical extension previously failed to load"
        ) from _EXTENSION_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Physical extension requires an available CUDA device")
    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        try:
            from torch.utils.cpp_extension import load

            root = Path(__file__).resolve().parent / "csrc"
            sources = (root / "physical_bitplane.cpp", root / "physical_v4.cu", root / "count_guard.cu", root / "count_queries.cu")
            _EXTENSION = load(
                name=_extension_name(sources),
                sources=[str(path) for path in sources],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=False,
            )
        except BaseException as error:
            _EXTENSION_ERROR = error
            raise RuntimeError(
                "failed to compile/load the CUDA Physical bitplane extension; "
                "no host-checker fallback is permitted"
            ) from error
    return _EXTENSION


def _cuda(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if value.device.type != "cuda":
        raise ValueError("CUDA Physical received a host tensor after batch transfer")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(
            f"CUDA Physical tensor dtype mismatch: {value.dtype} != {dtype}"
        )
    return value.contiguous()


def analyze_cuda_v4(
    batch: Mapping[str, Any],
    delta_query_slots: torch.Tensor,
    catalog: CompactPathCatalog,
    *, delta_opportunities: torch.Tensor | None = None, geometry_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analyze full illegal sets without selecting a model candidate."""

    required = (
        "physical_chart_indices",
        "physical_view_indices",
        "physical_batch_pair_indices",
        "physical_pair_chart_indices",
        "physical_pair_view_indices",
        "physical_pair_plane_offsets",
        "physical_opportunity_plan",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError("CUDA Physical v4 resident batch lacks " + ", ".join(missing))
    extension = load_cuda_extension()
    paths = catalog.device_tensors(delta_query_slots.device)
    from maimai_baseline.physical.resident import active_device_cache

    resident = active_device_cache(delta_query_slots.device)
    cache = resident.tensors
    measure = bool(batch.get("physical_cuda_timing", False))
    permute_start = torch.cuda.Event(enable_timing=True) if measure else None
    permute_end = torch.cuda.Event(enable_timing=True) if measure else None
    if permute_start is not None:
        permute_start.record()
    planes = resident.materialization_buffer(
        int(batch["physical_pair_plane_value_count"])
    )
    extension.materialize_planes_v2(
        _cuda(cache["plane_low_bytes"], dtype=torch.uint8),
        _cuda(cache["plane_high_bits"], dtype=torch.uint8),
        _cuda(cache["plane_offsets"], dtype=torch.long),
        planes,
        _cuda(batch["physical_pair_plane_offsets"], dtype=torch.long),
        _cuda(batch["physical_pair_chart_indices"], dtype=torch.long),
        _cuda(batch["physical_pair_view_indices"], dtype=torch.long),
        region_byte_lut(delta_query_slots.device),
    )
    if permute_end is not None:
        permute_end.record()
        _PENDING_PERMUTE_TIMINGS.append((permute_start, permute_end))

    start_event = torch.cuda.Event(enable_timing=True) if measure else None
    end_event = torch.cuda.Event(enable_timing=True) if measure else None
    if start_event is not None:
        start_event.record()
    if delta_opportunities is None:
        delta_opportunities = batch["physical_opportunity_plan"]["values"]
    result = extension.analyze_resident_v4(
        planes,
        _cuda(batch["physical_pair_plane_offsets"], dtype=torch.long),
        _cuda(cache["slot_origins"], dtype=torch.long),
        _cuda(cache["slot_counts"], dtype=torch.long),
        _cuda(cache["action_values"], dtype=torch.int32),
        _cuda(cache["action_masks_by_view"], dtype=torch.long),
        _cuda(cache["action_floating"], dtype=torch.float64),
        _cuda(cache["source_action"], dtype=torch.long),
        _cuda(cache["source_offsets"], dtype=torch.long),
        _cuda(cache["moment_slots"], dtype=torch.long),
        _cuda(cache["moment_offsets"], dtype=torch.long),
        _cuda(cache["carry_offsets"], dtype=torch.long),
        _cuda(cache["carry_indices"], dtype=torch.long),
        _cuda(batch["physical_opportunity_plan"]["values"], dtype=torch.long),
        _cuda(batch["physical_input_source_row_ids"], dtype=torch.long),
        _cuda(batch["local_block_id"], dtype=torch.long),
        _cuda(batch["physical_chart_indices"], dtype=torch.long),
        _cuda(batch["physical_batch_pair_indices"], dtype=torch.long),
        _cuda(batch["physical_view_indices"], dtype=torch.long),
        _cuda(delta_query_slots, dtype=torch.long),
        _cuda(delta_opportunities, dtype=torch.long),
        _cuda(cache["point_ticks"], dtype=torch.float64),
        _cuda(cache["point_seconds"], dtype=torch.float64),
        _cuda(cache["point_bpms"], dtype=torch.float64),
        _cuda(cache["point_padding_mask"], dtype=torch.bool),
        _cuda(paths["path_offsets"], dtype=torch.long),
        _cuda(paths["path_masks_by_view"], dtype=torch.long),
        _cuda(paths["path_enter_proportions"], dtype=torch.float64),
        _cuda(paths["path_critical_proportions"], dtype=torch.float64),
        _cuda(paths["path_flags"], dtype=torch.long),
        _cuda(paths["path_ids_by_view"], dtype=torch.long),
        geometry_only,
    )
    if end_event is not None:
        end_event.record()
        _PENDING_TIMINGS.append((start_event, end_event))
    return tuple(result)  # type: ignore[return-value]


def _consume(
    pending: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    *,
    history: str,
    accept_current: bool | None = None,
) -> list[float]:
    values = list(pending)
    pending.clear()
    result: list[float] = []
    for start, end in values:
        seconds = float(start.elapsed_time(end)) / 1_000.0
        result.append(seconds)
        accepted = _TIMING_ACCEPT_CURRENT if accept_current is None else accept_current
        if accepted:
            _TIMING_HISTORY[history].append(seconds)
    return result


def consume_cuda_timing_seconds() -> list[float]:
    """Return completed opt-in checker timings after the caller synchronized."""
    global _TIMING_PRODUCTION_CALLS, _TIMING_ACCEPT_CURRENT
    ordinal = _TIMING_PRODUCTION_CALLS
    _TIMING_PRODUCTION_CALLS += 1
    _TIMING_ACCEPT_CURRENT = ordinal >= _TIMING_SKIP and (
        _TIMING_TAKE is None or ordinal < _TIMING_SKIP + _TIMING_TAKE
    )
    _consume(_PENDING_COUNT_TIMINGS, history="count_matrix_analysis")
    for name, pending in _COUNT_DETAILS.items():
        _consume(pending, history=name)
    _consume(_PENDING_PERMUTE_TIMINGS, history="view_permutation")
    _consume(_PENDING_OPPORTUNITY_TIMINGS, history="opportunity_preparation")
    return _consume(_PENDING_TIMINGS, history="checker_kernel")




def record_opportunity_timing(
    start: torch.cuda.Event, end: torch.cuda.Event
) -> None:
    _PENDING_OPPORTUNITY_TIMINGS.append((start, end))




def configure_timing_collection(*, skip: int, take: int | None) -> None:
    global _TIMING_SKIP, _TIMING_TAKE, _TIMING_PRODUCTION_CALLS
    _TIMING_SKIP = max(0, int(skip))
    _TIMING_TAKE = None if take is None else max(0, int(take))
    _TIMING_PRODUCTION_CALLS = 0
    for values in _TIMING_HISTORY.values():
        values.clear()
    _PENDING_TIMINGS.clear()
    _PENDING_COUNT_TIMINGS.clear()
    for pending in _COUNT_DETAILS.values():
        pending.clear()
    _PENDING_PERMUTE_TIMINGS.clear()
    _PENDING_OPPORTUNITY_TIMINGS.clear()


def timing_breakdown() -> dict[str, list[float]]:
    return {name: list(values) for name, values in _TIMING_HISTORY.items()}


def record_count_timing(start: torch.cuda.Event, end: torch.cuda.Event) -> None:
    _PENDING_COUNT_TIMINGS.append((start,end))


@contextmanager
def count_phase(batch, name):
    """Opt-in events only; no production per-phase synchronization."""
    enabled = bool(batch.get("physical_cuda_timing", False))
    if not enabled:
        yield
        return
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _COUNT_DETAILS[name].append((start, end))
