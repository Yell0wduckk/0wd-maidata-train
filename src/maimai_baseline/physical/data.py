from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from maimai_baseline.data.dataset import BaselineDataset
from maimai_baseline.data.batch import collate_baseline_segments
from maimai_baseline.data.curriculum import VirtualSegmentRef
from maimai_baseline.physical.interval_store import IntervalPhysicalStore
from maimai_baseline.physical.source_rows import PhysicalSourceRowStore


def source_row_metadata(
    segment: Mapping[str, Any], length: int, *, view_to_canonical: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = int(segment["target_row_start"]), int(segment["target_row_end"])
    rows = [
        -2,
        *range(int(segment["context_row_start"]), start),
        -1,
        *range(start, end),
    ]
    if len(rows) != length:
        raise ValueError(f"{segment['segment_id']} Physical source-row length mismatch")
    # Empty leading windows can use chart EOF as context_row_start while the
    # target interval is [0, 0). range(context_start, start) is then empty.
    if not (0 <= int(segment["context_row_start"]) <= view_to_canonical.numel()
            and 0 <= start <= end <= view_to_canonical.numel()):
        raise ValueError(f"{segment['segment_id']} Physical source-row segment bounds invalid")
    inputs = torch.tensor(rows)
    active = inputs.ge(0)
    inputs[active] = view_to_canonical.index_select(0, inputs[active])
    # SEGMENT_END is a boundary, including at chart EOF; do not look it up as
    # a viewed action. Complete-window boundaries remain canonical row offsets.
    return inputs, torch.cat((inputs[1:], torch.tensor([end])))


class PhysicalBaselineDataset(BaselineDataset):
    def __init__(
        self, *args: Any, physical_store: IntervalPhysicalStore,
        physical_source_rows: PhysicalSourceRowStore, **kwargs: Any
    ):
        self.physical_store = physical_store
        self.physical_source_rows = physical_source_rows
        super().__init__(*args, **kwargs)
        for segment in self.segments:
            physical_store.handle(str(segment["segment_id"]))
            record = f"{segment['song_id']}:{segment.get('source_difficulty_id', segment['difficulty_id'])}"
            for view in self.spatial_store.arrays:
                mapping = physical_source_rows.rows(view, record)
                if int(segment["target_row_end"]) > mapping.numel():
                    raise ValueError(f"{segment['segment_id']} Physical source-row segment incomplete")

    def _view_raw_item(self, reference: VirtualSegmentRef) -> dict[str, Any]:
        result = super()._view_raw_item(reference)
        segment = self.segments[int(reference.canonical_index)]
        inputs, targets = source_row_metadata(
            segment, int(result["row_anchor_tick"].shape[0]),
            view_to_canonical=self.physical_source_rows.rows(
                reference.view,
                f"{segment['song_id']}:{segment.get('source_difficulty_id', segment['difficulty_id'])}",
            ),
        )
        result["_physical_input_source_row_ids"] = inputs
        result["_physical_target_source_row_ids"] = targets
        result["_physical_chart_index"] = self.physical_store.handle(
            str(segment["segment_id"])
        )
        result["_physical_view_index"] = int(reference.view_id)
        return result


def collate_physical_segments(
    items: Sequence[Mapping[str, Any]],
    *,
    physical_store: IntervalPhysicalStore,
    vocabulary,
    audio_loader: Any = None,
    chart_style: bool = False,
    global_reference_provider=None,
    cached_style_targets: bool = False,
    timing_root=None,
    chart_only=False,
) -> dict[str, Any]:
    batch = collate_baseline_segments(items, vocabulary=vocabulary, audio_loader=audio_loader,
                                      timing_root=timing_root, chart_only=chart_only)
    if cached_style_targets:
        batch["style_cache_keys"] = [(str(item["segment_id"]), str(item["spatial_view_id"])) for item in items]
    width = int(batch["row_padding_mask"].shape[1])
    for name in ("physical_input_source_row_ids", "physical_target_source_row_ids"):
        values = torch.full((len(items), width), -3, dtype=torch.long)
        for index, item in enumerate(items):
            source = item["_" + name]
            values[index, : source.numel()] = source
        batch[name] = values
    batch['physical_chart_indices']=torch.tensor([i['_physical_chart_index'] for i in items])
    batch['physical_view_indices']=torch.tensor([i['_physical_view_index'] for i in items])
    from maimai_baseline.physical.stage_plan import prepare
    batch.update(prepare(batch,physical_store,vocabulary))
    if chart_style and global_reference_provider is None:
        from maimai_baseline.data.style import prepare_style_inputs
        batch["style_inputs"] = prepare_style_inputs(batch, vocabulary)
    if global_reference_provider is not None:
        batch["global_style_inputs"] = global_reference_provider(items)
    return batch
