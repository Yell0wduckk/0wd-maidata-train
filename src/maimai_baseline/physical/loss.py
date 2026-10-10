"""L3 and conditional-stage Physical objective for the current interval engine."""
from typing import Any, Mapping
from maimai_baseline.loss.common import L3LossResult
from maimai_baseline.loss.l3 import l3_loss
from maimai_baseline.loss.supervision import SpecializedSupervision
from maimai_baseline.physical.contract import PHYSICAL_WARMUP_STEPS
from maimai_dataset.vocabulary import CompactVocabulary


def physical_bad_mass_loss(
    chart: L3LossResult,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *,
    runtime_cache: Mapping[str, Any],
    warmup_coefficient: float,
    allow_cpu_reference: bool = False,
    collect_metrics: bool = True,
) -> L3LossResult:
    from maimai_baseline.physical.stage_loss import compute
    return compute(
        chart,
        outputs,
        batch,
        vocabulary,
        runtime_cache=runtime_cache,
        warmup_coefficient=warmup_coefficient,
        collect_metrics=collect_metrics,
    )


def physical_warmup(step: int) -> float:
    if step < 0:
        raise ValueError("negative Physical optimizer step")
    return min(1.0, step / (PHYSICAL_WARMUP_STEPS - 1))


def baseline_physical_loss(
    outputs, batch, vocabulary, *, warmup_coefficient=1.0, allow_cpu_reference=False, collect_metrics=True,
    supervision: SpecializedSupervision = SpecializedSupervision()
):
    runtime_cache: dict[str, Any] = {}
    capture = {"position_cache": batch["l3_position_cache"]} if "l3_position_cache" in batch else {}
    chart = l3_loss(outputs, batch, vocabulary, runtime_cache=runtime_cache, collect_metrics=collect_metrics,
                    supervision=supervision, **capture)
    return physical_bad_mass_loss(
        chart,
        outputs,
        batch,
        vocabulary,
        runtime_cache=runtime_cache,
        warmup_coefficient=warmup_coefficient,
        allow_cpu_reference=allow_cpu_reference,
        collect_metrics=collect_metrics,
    )
