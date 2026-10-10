"""Training datasets and epoch-plan sampling."""

from __future__ import annotations
from maimai_dataset.window import WINDOW_POSITIONS
from collections.abc import Iterable, Mapping
from maimai_baseline.contract import AUGMENTATION_SEED
from maimai_baseline.data.augmentation import prepare_augmented_item
from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, VirtualSegmentRef
from maimai_baseline.data.route import Route71Codec
from maimai_dataset.dataset import CompactSegmentDataset
from pathlib import Path
from torch.utils.data import Sampler
from typing import Any
import multiprocessing


class RandomMomentAugmentedDataset(CompactSegmentDataset):
    def __init__(
        self,
        *args: Any,
        augmentation_seed: int = AUGMENTATION_SEED,
        **kwargs: Any,
    ) -> None:
        self.augmentation_seed = int(augmentation_seed)
        self._shared_epoch = multiprocessing.Value("q", 0, lock=True)
        self._shared_enabled = multiprocessing.Value("i", 1, lock=True)
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def set_augmentation_enabled(self, enabled: bool) -> None:
        with self._shared_enabled.get_lock():
            self._shared_enabled.value = int(bool(enabled))

    @property
    def augmentation_epoch(self) -> int:
        return int(self._shared_epoch.value)

    @property
    def augmentation_enabled(self) -> bool:
        return bool(self._shared_enabled.value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return prepare_augmented_item(
            super().__getitem__(index),
            self.vocabulary,
            seed=self.augmentation_seed,
            epoch=self.augmentation_epoch,
            enabled=self.augmentation_enabled,
        )


# Spatial-view dataset and epoch-plan sampler


class SpatialRoute71Dataset(RandomMomentAugmentedDataset):
    """Canonical validation plus epoch-indexed whole-chart training views."""

    def __init__(
        self,
        *args: Any,
        spatial_store: MultiViewRouteStore,
        augmentation_seed: int = AUGMENTATION_SEED,
        **kwargs: Any,
    ) -> None:
        self.spatial_store = spatial_store
        super().__init__(*args, augmentation_seed=augmentation_seed, **kwargs)
        self.route_codec = Route71Codec(self.vocabulary)
        canonical = set(self.chart_store.segment_indices) if self.chart_store else set()
        selected = {str(segment["segment_id"]) for segment in self.segments}
        missing = sorted(selected - set(self.spatial_store.segment_indices))
        if missing:
            raise ValueError(f"spatial cache lacks selected segments: {missing[:8]}")
        if canonical and not selected.issubset(canonical):
            raise ValueError("dataset selection escaped the canonical chart cache")

    def _view_raw_item(self, reference: VirtualSegmentRef) -> dict[str, Any]:
        index = int(reference.canonical_index)
        if not 0 <= index < len(self.segments):
            raise IndexError(index)
        segment = self.segments[index]
        vectors = self.spatial_store.record(str(segment["segment_id"]), reference.view)
        result = {
            "segment_id": str(segment["segment_id"]),
            "song_id": str(segment["song_id"]),
            "audio_group_key": str(segment["audio_group_key"]),
            **vectors,
            "difficulty_id": int(
                segment.get("condition_difficulty_id", segment["difficulty_id"])
            ),
            "source_difficulty_id": int(
                segment.get("source_difficulty_id", segment["difficulty_id"])
            ),
            "cabinet_id": 2 if segment["cabinet"] == "DX" else 1,
            "level": float(segment.get("level_value") or 0.0),
            "start_tick": float(segment["target_start_tick"]),
            "end_tick": float(segment["target_end_tick"]),
            "song_end_tick": int(segment["song_end_tick"]),
            "spatial_view_id": str(reference.view),
            "occurrence": int(reference.occurrence),
        }
        if not self.defer_audio:
            result.update(self.audio_item(str(segment["audio_group_key"])))
        return result

    def __getitem__(self, index: int | VirtualSegmentRef) -> dict[str, Any]:
        if isinstance(index, VirtualSegmentRef):
            prepared = prepare_augmented_item(
                self._view_raw_item(index),
                self.vocabulary,
                seed=self.augmentation_seed,
                epoch=self.augmentation_epoch,
                enabled=self.augmentation_enabled,
            )
            return self._attach_route_informative_mass(prepared)
        # Validation always uses the exact identity view and canonical order.
        identity = VirtualSegmentRef(
            canonical_index=int(index), view_id=0, occurrence=0
        )
        prepared = prepare_augmented_item(
            self._view_raw_item(identity),
            self.vocabulary,
            seed=self.augmentation_seed,
            epoch=self.augmentation_epoch,
            enabled=False,
        )
        return self._attach_route_informative_mass(prepared)

    def _attach_route_informative_mass(
        self, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        """Align Route-71 weights with the final decoder-row layout.

        The multiview cache stores weights for the uncropped sequence.  Both
        augmented training items and canonical validation items pass through
        the history-context crop, so the weights must be reconstructed from the final
        ``target_ids["route"]`` instead of retaining the cache-length tensor.
        """

        route_target = prepared["target_ids"]["route"]
        active = route_target.ne(-100)
        route_mass = prepared["row_anchor_tick"].new_zeros(route_target.shape)
        route_mass[active] = self.spatial_store.route_mass_by_class.index_select(
            0, route_target[active]
        )
        prepared["route_informative_mass"] = route_mass
        return prepared


class EpochPlanBatchSampler(Sampler[list[VirtualSegmentRef]]):
    def __init__(
        self,
        store: EpochPlanStore,
        dataset: SpatialRoute71Dataset,
        *,
        batching_manifest: dict[str, Any],
    ) -> None:
        self.store = store
        self.dataset = dataset
        self.batching_manifest = batching_manifest
        self.epoch = 0
        self._batches: tuple[tuple[VirtualSegmentRef, ...], ...] | None = None

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch or self._batches is None:
            self.epoch = epoch
            self._batches = self.store.load_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.batch_target_rows = tuple(sum(max(1, int(self.dataset.target_lengths[ref.canonical_index])) for ref in batch) for batch in self._batches)
        if any(n <= 0 for n in self.batch_target_rows):
            raise ValueError("epoch plan contains an empty batch")
        audit = dict(self.store.audit(epoch))
        if "batch_target_rows" in audit and list(self.batch_target_rows) != audit["batch_target_rows"]:
            raise ValueError("epoch plan target row counts mismatch")
        self.batching_manifest["epoch_plan"] = {
            key: audit[key]
            for key in (
                "mode",
                "epoch",
                "chart_slots",
                "segment_references",
                "audio_groups",
                "natural_batch_count",
                "fixed_batch_count",
                "difficulty_slot_counts",
                "view_counts_by_pool",
                "plan_sha256",
                "batch_count", "target_rows", "note_rows",
            ) if key in audit
        }

    def _current(self) -> tuple[tuple[VirtualSegmentRef, ...], ...]:
        if self._batches is None:
            self.set_epoch(self.epoch)
        assert self._batches is not None
        return self._batches

    def __iter__(self) -> Iterable[list[VirtualSegmentRef]]:
        for batch in self._current():
            yield list(batch)

    def __len__(self) -> int:
        return self.store.batch_count(self.epoch)

    @property
    def singleton_oversize(self) -> int:
        return 0

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.batching_manifest


# Public BPM96 baseline dataset and collate


def _segment_index(index: int | VirtualSegmentRef) -> int:
    return int(index.canonical_index if isinstance(index, VirtualSegmentRef) else index)


class BaselineDataset(SpatialRoute71Dataset):
    """Attach timing metadata while leaving every cached chart tensor untouched."""

    def configure_bpm_audio(self, store: Any, policy: str) -> None:
        from maimai_baseline.data.audio import exact_level
        from maimai_baseline.data.audio import training_audio_policy
        from maimai_baseline.time.grid import identity
        policy = training_audio_policy(policy)
        self.bpm_audio_store = store
        self.audio_policy = policy
        self.runtime_segment_audio_keys = []
        self.bpm_runtime_groups = {}
        self.bpm_base_keys = {}
        for segment in self.segments:
            level = exact_level(segment["level_value"])
            if level != exact_level(store.levels[str(segment["record_id"])]):
                raise ValueError(f"BPM profile level mismatch: {segment['segment_id']}")
            key = identity({"variant": store.segments[str(segment["segment_id"])],
                            "start_tick": int(segment["target_start_tick"]), "gain_db": 0.0})
            # Keep the established zero-gain identity material for runtime grouping.
            self.bpm_runtime_groups[key] = segment
            self.bpm_base_keys[str(segment["audio_group_key"])] = key
            self.runtime_segment_audio_keys.append(key)
        self.audio_frame_lengths = [WINDOW_POSITIONS] * len(self.segments)


    def feature_source(self, segment: Mapping[str, Any]) -> Path:
        if hasattr(self, "bpm_audio_store"):
            store = self.bpm_audio_store
            variant = store.variants[store.segments[str(segment["segment_id"])]]
            return store.root / variant["feature_path"]
        return super().feature_source(segment)

    def audio_item(self, group_key: str) -> dict[str, Any]:
        if not hasattr(self, "bpm_audio_store"):
            return super().audio_item(group_key)
        key = group_key if group_key in self.bpm_runtime_groups else self.bpm_base_keys[group_key]
        result = self.bpm_audio_store.window(self.bpm_runtime_groups[key])
        result["audio_group_key"] = key
        return result

    def audio_sharing_report(self, batches: Iterable[Iterable[Any]]) -> list[dict[str, int]]:
        reports = []
        for index, batch in enumerate(batches):
            batch = tuple(batch)
            segments = [self.segments[_segment_index(ref)] for ref in batch]
            def base_identity(s):
                if hasattr(self, "bpm_audio_store"):
                    return (self.bpm_audio_store.segments[str(s["segment_id"])], int(s["target_start_tick"]))
                return str(s["audio_group_key"])
            base = {base_identity(s) for s in segments}
            keys = getattr(self, "runtime_segment_audio_keys", None)
            actual = base if keys is None else {keys[_segment_index(ref)] for ref in batch}
            reports.append({"batch": index, "chart_windows": len(segments),
                            "base_audio_windows": len(base),
                            "actual_audio_windows": len(actual)})
        return reports

    def __getitem__(self, index: int | VirtualSegmentRef) -> dict[str, Any]:
        result = super().__getitem__(index)
        segment = self.segments[_segment_index(index)]
        result["_delta10ms_timing"] = dict(segment["timing"])
        result["_delta10ms_audio_frame_start"] = int(segment["audio_frame_start"])
        if hasattr(self, "bpm_audio_store"):
            result["_bpm96"] = True
            store = self.bpm_audio_store
            result["_prepared_timing"] = store.prepared_timings[str(segment["record_id"])]
            variant = store.variants[store.segments[str(segment["segment_id"])]]
            result["audio_num_samples"] = int(variant["num_samples"])
        if hasattr(self, "runtime_segment_audio_keys"):
            result["audio_group_key"] = self.runtime_segment_audio_keys[_segment_index(index)]
            if not self.defer_audio:
                result.update(self.audio_item(result["audio_group_key"]))
        return result
