from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.array_store import ReadOnlyArrayStore
from maimai_baseline.physical.contract import BITPLANE_CACHE_SCHEMA
from maimai_baseline.physical.contract import BITPLANE_CACHE_CONTRACT
from maimai_baseline.physical.contract import PLANE_COUNT
from maimai_baseline.physical.contract import VALID_REGION_MASK
from maimai_baseline.physical.contract import CACHE_ACTION_COLUMNS
from maimai_baseline.physical.contract import ACTION_FLOAT_COLUMNS


def _sha256_file(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


class PhysicalBitplaneStore(ReadOnlyArrayStore):
    def __init__(
        self,
        root: Path,
        *,
        profile_fingerprint: str | None = None,
        chart_cache_manifest_sha256: str | None = None,
        spatial_cache_manifest_sha256: str | None = None,
        catalog_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text("utf-8-sig"))
        schema = str(self.manifest.get("schema", ""))
        if schema != BITPLANE_CACHE_SCHEMA:
            if "physical_plan_cache" in schema:
                raise DatasetError(
                    "epoch/batch Physical plan cache v1 is invalid; build a Physical bitplane cache"
                )
            raise DatasetError(
                f"unsupported Physical bitplane cache schema: {schema!r}"
            )
        if self.manifest.get("contract") != BITPLANE_CACHE_CONTRACT:
            raise DatasetError("Physical bitplane cache contract mismatch")
        identity = str(self.manifest.get("manifest_identity_sha256", ""))
        payload = dict(self.manifest)
        payload.pop("manifest_identity_sha256", None)
        if canonical_json_sha256(payload) != identity:
            raise DatasetError("Physical bitplane cache manifest identity mismatch")
        checks = {
            "profile_fingerprint": profile_fingerprint,
            "chart_cache_manifest_sha256": chart_cache_manifest_sha256,
            "spatial_cache_manifest_sha256": spatial_cache_manifest_sha256,
        }
        for name, expected in checks.items():
            if expected is not None and str(self.manifest.get(name)) != str(expected):
                raise DatasetError(f"Physical bitplane cache {name} mismatch")
        if catalog_identity is not None and dict(self.manifest["path_catalog"]) != dict(
            catalog_identity
        ):
            raise DatasetError("Physical bitplane cache path catalog mismatch")
        self.arrays = {
            name: np.load(
                self.root / str(meta["path"]), mmap_mode="r", allow_pickle=False
            )
            for name, meta in self.manifest["arrays"].items()
        }
        self.offsets = {
            name: np.load(
                self.root / str(meta["path"]), mmap_mode="r", allow_pickle=False
            )
            for name, meta in self.manifest["offsets"].items()
        }
        self.chart_records = tuple(load_jsonl(self.root / "charts.jsonl"))
        self.segment_records = {
            str(value["segment_id"]): value
            for value in load_jsonl(self.root / "segments.jsonl")
        }
        self.record_indices = {
            str(value["record_id"]): int(value["chart_index"])
            for value in self.chart_records
        }

    def handle(self, segment_id: str) -> int:
        try:
            return int(self.segment_records[str(segment_id)]["chart_index"])
        except KeyError as error:
            raise DatasetError(
                f"Physical bitplane cache lacks segment {segment_id}"
            ) from error

    def chart_bytes(self, chart_index: int) -> memoryview:
        begin = int(self.offsets["plane_offsets"][chart_index])
        end = int(self.offsets["plane_offsets"][chart_index + 1])
        return memoryview(self.arrays["planes"][begin:end])

    def describe_items(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return only the compact chart/view indirection needed by production.

        The canonical cache stays mmap-backed here and resident on the CUDA
        device.  No chart payload is copied into a DataLoader batch.
        """

        pairs: list[tuple[int, int]] = []
        pair_index: dict[tuple[int, int], int] = {}
        batch_pairs: list[int] = []
        batch_charts: list[int] = []
        batch_views: list[int] = []
        pair_plane_offsets = [0]
        slot_counts = self.offsets["slot_counts"]
        for item in items:
            chart = int(item["_physical_chart_index"])
            view = int(item["_physical_view_index"])
            pair = (chart, view)
            index = pair_index.get(pair)
            if index is None:
                index = len(pairs)
                pair_index[pair] = index
                pairs.append(pair)
                pair_plane_offsets.append(
                    pair_plane_offsets[-1] + PLANE_COUNT * int(slot_counts[chart])
                )
            batch_pairs.append(index)
            batch_charts.append(chart)
            batch_views.append(view)
        return {
            "physical_chart_indices": torch.tensor(batch_charts, dtype=torch.long),
            "physical_view_indices": torch.tensor(batch_views, dtype=torch.long),
            "physical_batch_pair_indices": torch.tensor(batch_pairs, dtype=torch.long),
            "physical_pair_chart_indices": torch.tensor(
                [chart for chart, _ in pairs], dtype=torch.long
            ),
            "physical_pair_view_indices": torch.tensor(
                [view for _, view in pairs], dtype=torch.long
            ),
            "physical_pair_plane_offsets": torch.tensor(
                pair_plane_offsets, dtype=torch.long
            ),
            "physical_pair_plane_value_count": int(pair_plane_offsets[-1]),
        }

    def pack_items(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Test/oracle compatibility pack; production uses :meth:`describe_items`."""

        pairs: list[tuple[int, int]] = []
        pair_index: dict[tuple[int, int], int] = {}
        batch_charts: list[int] = []
        for item in items:
            pair = (
                int(item["_physical_chart_index"]),
                int(item["_physical_view_index"]),
            )
            if pair not in pair_index:
                pair_index[pair] = len(pairs)
                pairs.append(pair)
            batch_charts.append(pair_index[pair])

        packed_planes: list[np.ndarray] = []
        packed_actions: list[np.ndarray] = []
        packed_floating: list[np.ndarray] = []
        packed_source_action: list[np.ndarray] = []
        packed_source_moment: list[np.ndarray] = []
        packed_moment_rows: list[np.ndarray] = []
        packed_moment_ticks: list[np.ndarray] = []
        packed_moment_slots: list[np.ndarray] = []
        packed_carry_indices: list[np.ndarray] = []
        packed_timing_ticks: list[np.ndarray] = []
        packed_timing_seconds: list[np.ndarray] = []
        packed_timing_bpms: list[np.ndarray] = []
        plane_offsets = [0]
        action_offsets = [0]
        source_offsets = [0]
        moment_offsets = [0]
        carry_offsets = [0]
        timing_offsets = [0]
        origins: list[int] = []
        counts: list[int] = []

        def section(name: str, offsets_name: str, chart: int) -> np.ndarray:
            offsets = self.offsets[offsets_name]
            return np.array(
                self.arrays[name][int(offsets[chart]) : int(offsets[chart + 1])],
                copy=True,
            )

        for chart, _view in pairs:
            planes = section("planes", "plane_offsets", chart)
            actions = section("action_values", "action_offsets", chart)
            floating = section("action_floating", "action_offsets", chart)
            source_action = section("source_action", "source_offsets", chart)
            source_moment = section("source_moment", "source_offsets", chart)
            moment_rows = section("moment_rows", "moment_offsets", chart)
            moment_ticks = section("moment_ticks", "moment_offsets", chart)
            moment_slots = section("moment_slots", "moment_offsets", chart)
            local_carry_offsets = section(
                "carry_offsets", "carry_offsets_offsets", chart
            ).astype(np.int64, copy=False)
            local_carry_indices = section(
                "carry_indices", "carry_index_offsets", chart
            ).astype(np.int64, copy=False)
            timing_ticks = section("timing_ticks", "timing_offsets", chart)
            timing_seconds = section("timing_seconds", "timing_offsets", chart)
            timing_bpms = section("timing_bpms", "timing_offsets", chart)
            action_base = action_offsets[-1]
            packed_planes.append(planes)
            packed_actions.append(actions)
            packed_floating.append(floating)
            packed_source_action.append(
                np.where(source_action >= 0, source_action + action_base, -1)
            )
            packed_source_moment.append(source_moment)
            packed_moment_rows.append(moment_rows)
            packed_moment_ticks.append(moment_ticks)
            packed_moment_slots.append(moment_slots)
            packed_carry_indices.append(local_carry_indices + action_base)
            # The local offsets are cumulative. Append their successive deltas
            # to the one global CSR used by the packed charts.
            for count in np.diff(local_carry_offsets).tolist():
                carry_offsets.append(carry_offsets[-1] + int(count))
            packed_timing_ticks.append(timing_ticks)
            packed_timing_seconds.append(timing_seconds)
            packed_timing_bpms.append(timing_bpms)
            plane_offsets.append(plane_offsets[-1] + int(planes.shape[0]))
            action_offsets.append(action_offsets[-1] + int(actions.shape[0]))
            source_offsets.append(source_offsets[-1] + int(source_action.shape[0]))
            moment_offsets.append(moment_offsets[-1] + int(moment_rows.shape[0]))
            timing_offsets.append(timing_offsets[-1] + int(timing_ticks.shape[0]))
            origins.append(int(self.offsets["slot_origins"][chart]))
            counts.append(int(self.offsets["slot_counts"][chart]))

        def concatenate(
            values: Sequence[np.ndarray],
            dtype: np.dtype[Any],
            tail: tuple[int, ...] = (),
        ) -> torch.Tensor:
            if values:
                array = np.concatenate(values, axis=0).astype(dtype, copy=False)
            else:
                array = np.empty((0, *tail), dtype=dtype)
            return torch.from_numpy(np.ascontiguousarray(array))

        timing_width = max(
            (int(value.shape[0]) for value in packed_timing_ticks), default=1
        )
        timing_point_ticks = torch.zeros(
            (len(pairs), timing_width), dtype=torch.float64
        )
        timing_point_seconds = torch.zeros_like(timing_point_ticks)
        timing_point_bpms = torch.ones_like(timing_point_ticks)
        timing_point_padding = torch.ones((len(pairs), timing_width), dtype=torch.bool)
        for index, (ticks, seconds, bpms) in enumerate(
            zip(
                packed_timing_ticks,
                packed_timing_seconds,
                packed_timing_bpms,
                strict=True,
            )
        ):
            count = int(ticks.shape[0])
            timing_point_ticks[index, :count] = torch.from_numpy(ticks)
            timing_point_seconds[index, :count] = torch.from_numpy(seconds)
            timing_point_bpms[index, :count] = torch.from_numpy(bpms)
            timing_point_padding[index, :count] = False

        return {
            "planes": concatenate(packed_planes, np.dtype(np.int64)),
            "plane_offsets": torch.tensor(plane_offsets, dtype=torch.long),
            "slot_origins": torch.tensor(origins, dtype=torch.long),
            "slot_counts": torch.tensor(counts, dtype=torch.long),
            "views": torch.tensor([view for _, view in pairs], dtype=torch.long),
            "batch_charts": torch.tensor(batch_charts, dtype=torch.long),
            "action_values": concatenate(
                packed_actions, np.dtype(np.int64), (len(CACHE_ACTION_COLUMNS),)
            ),
            "action_floating": concatenate(
                packed_floating, np.dtype(np.float64), (len(ACTION_FLOAT_COLUMNS),)
            ),
            "action_offsets": torch.tensor(action_offsets, dtype=torch.long),
            "source_action": concatenate(packed_source_action, np.dtype(np.int64)),
            "source_moment": concatenate(packed_source_moment, np.dtype(np.int64)),
            "source_offsets": torch.tensor(source_offsets, dtype=torch.long),
            "moment_rows": concatenate(packed_moment_rows, np.dtype(np.int64)),
            "moment_ticks": concatenate(packed_moment_ticks, np.dtype(np.int64)),
            "moment_slots": concatenate(packed_moment_slots, np.dtype(np.int64)),
            "moment_offsets": torch.tensor(moment_offsets, dtype=torch.long),
            "carry_offsets": torch.tensor(carry_offsets, dtype=torch.long),
            "carry_indices": concatenate(packed_carry_indices, np.dtype(np.int64)),
            "timing_ticks": concatenate(packed_timing_ticks, np.dtype(np.float64)),
            "timing_seconds": concatenate(packed_timing_seconds, np.dtype(np.float64)),
            "timing_bpms": concatenate(packed_timing_bpms, np.dtype(np.float64)),
            "timing_offsets": torch.tensor(timing_offsets, dtype=torch.long),
            "point_ticks": timing_point_ticks,
            "point_seconds": timing_point_seconds,
            "point_bpms": timing_point_bpms,
            "point_padding_mask": timing_point_padding,
            "layout": "canonical-chart-major-readonly-causal-v2",
        }

    def validate(self, *, full_hash: bool = False, scan_contents: bool = True) -> dict[str, Any]:
        for forbidden in ("epoch", "occurrence", "batch"):
            if forbidden in self.manifest:
                raise DatasetError(f"Physical bitplane manifest contains {forbidden}")
        expected_charts = int(self.manifest["chart_count"])
        expected_segments = int(self.manifest["segment_count"])
        if len(self.chart_records) != expected_charts:
            raise DatasetError("Physical bitplane chart index count mismatch")
        if len(self.segment_records) != expected_segments:
            raise DatasetError("Physical bitplane segment index count mismatch")
        if len(self.record_indices) != expected_charts:
            raise DatasetError("Physical bitplane chart identities are not unique")
        if set(self.record_indices.values()) != set(range(expected_charts)) or any(
            not 0 <= int(record["chart_index"]) < expected_charts
            for record in self.segment_records.values()
        ):
            raise DatasetError("Physical cache chart index bounds mismatch")
        for name, metadata in {
            **self.manifest["arrays"],
            **self.manifest["offsets"],
        }.items():
            value = self.arrays.get(name, self.offsets.get(name))
            if value is None:
                raise DatasetError(f"Physical cache array is missing: {name}")
            if list(value.shape) != [int(part) for part in metadata["shape"]]:
                raise DatasetError(f"Physical cache array shape mismatch: {name}")
            if str(value.dtype) != str(metadata["dtype"]):
                raise DatasetError(f"Physical cache array dtype mismatch: {name}")
        for name in (
            "plane_offsets",
            "slot_offsets",
            "action_offsets",
            "source_offsets",
            "moment_offsets",
            "carry_offsets_offsets",
            "carry_index_offsets",
            "timing_offsets",
        ):
            offsets = self.offsets[name]
            if offsets.shape != (expected_charts + 1,) or int(offsets[0]) != 0:
                raise DatasetError(f"Physical cache offset layout mismatch: {name}")
            if np.any(offsets[1:] < offsets[:-1]):
                raise DatasetError(f"Physical cache offsets are not monotonic: {name}")
        for name, offset_name in (
            ("action_values", "action_offsets"), ("action_floating", "action_offsets"),
            ("source_action", "source_offsets"), ("source_moment", "source_offsets"),
            ("moment_rows", "moment_offsets"), ("moment_ticks", "moment_offsets"),
            ("moment_slots", "moment_offsets"), ("carry_offsets", "carry_offsets_offsets"),
            ("carry_indices", "carry_index_offsets"), ("timing_ticks", "timing_offsets"),
            ("timing_seconds", "timing_offsets"), ("timing_bpms", "timing_offsets"),
        ):
            if int(self.offsets[offset_name][-1]) != self.arrays[name].shape[0]:
                raise DatasetError(f"Physical cache payload length mismatch: {name}")
        if int(self.offsets["plane_offsets"][-1]) != int(
            self.arrays["planes"].shape[0]
        ):
            raise DatasetError("Physical cache plane payload length mismatch")
        if not np.array_equal(
            np.diff(self.offsets["plane_offsets"]),
            PLANE_COUNT * np.asarray(self.offsets["slot_counts"]),
        ):
            raise DatasetError("Physical cache is not five planes per chart")
        high_mask = np.int64(~VALID_REGION_MASK)
        high_clear = True
        planes = self.arrays["planes"]
        chunk = 8 << 20
        for begin in range(0, int(planes.shape[0]) if scan_contents else 0, chunk):
            if np.bitwise_and(planes[begin : begin + chunk], high_mask).any():
                high_clear = False
                break
        if not high_clear:
            raise DatasetError("Physical bitplane cache uses bits above region 40")
        if full_hash:
            for name in ("charts", "segments"):
                metadata = self.manifest[name]
                if _sha256_file(self.root / str(metadata["path"])) != str(
                    metadata["sha256"]
                ):
                    raise DatasetError(f"Physical cache index hash mismatch: {name}")
            for group in ("arrays", "offsets"):
                for name, metadata in self.manifest[group].items():
                    if _sha256_file(self.root / str(metadata["path"])) != str(
                        metadata["sha256"]
                    ):
                        raise DatasetError(f"Physical cache hash mismatch: {name}")
        return {
            "schema": "maimai.route71_delta10ms_physical_bitplane_cache_audit.v2",
            "chart_count": len(self.chart_records),
            "segment_count": len(self.segment_records),
            "high_23_bits_clear": high_clear if scan_contents else None,
            "full_hash": bool(full_hash),
            "manifest_identity_sha256": self.manifest["manifest_identity_sha256"],
        }
