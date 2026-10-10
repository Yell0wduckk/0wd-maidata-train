from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from maimai_dataset.vocabulary import CompactVocabulary
from maimai_baseline.data.route import Route71Codec

from maimai_baseline.physical.contract import VALID_REGION_MASK
from maimai_baseline.physical.spatial import transform_mask


def pad_bit(pad: str) -> int:
    value = str(pad)
    if len(value) == 2 and value[1].isdigit():
        lane = int(value[1])
        if not 1 <= lane <= 8:
            return 0
        offsets = {"A": 0, "B": 8, "D": 17, "E": 25, "K": 33}
        if value[0] in offsets:
            return 1 << (offsets[value[0]] + lane - 1)
    if value == "C":
        return 1 << 16
    return 0


def pads_mask(pads: tuple[str, ...] | list[str]) -> int:
    result = 0
    for pad in pads:
        result |= pad_bit(str(pad))
    if result & ~VALID_REGION_MASK:
        raise ValueError("physical path uses a region outside the low 41 bits")
    return result


def _sha256(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


@dataclass
class CompactPathCatalog:
    raw_paths: tuple[Mapping[str, Any], ...]
    route_path_ids: torch.Tensor
    path_offsets: torch.Tensor
    path_masks: torch.Tensor
    path_enter_proportions: torch.Tensor
    path_critical_proportions: torch.Tensor
    path_flags: torch.Tensor
    path_aliases: torch.Tensor
    path_ids_by_view: torch.Tensor
    identity: Mapping[str, Any]
    _device_cache: dict[str, Mapping[str, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def load(
        cls,
        root: Path,
        vocabulary: CompactVocabulary,
    ) -> "CompactPathCatalog":
        root = Path(root).expanduser().resolve()
        paths_path = root / "paths.json"
        semantics_path = root / "runtime-path-semantics.json"
        rules_path = root / "runtime-path-rules.json"
        for path in (paths_path, semantics_path, rules_path):
            if not path.is_file():
                raise FileNotFoundError(f"Physical path resource is missing: {path}")
        path_payload = json.loads(paths_path.read_text("utf-8"))
        semantics_payload = json.loads(semantics_path.read_text("utf-8"))
        rules_payload = json.loads(rules_path.read_text("utf-8"))
        raw = tuple(path_payload.get("paths", ()))
        if len(raw) != 568:
            raise ValueError(
                f"Physical path resource must contain 568 paths, got {len(raw)}"
            )
        if [int(value["path_id"]) for value in raw] != list(range(568)):
            raise ValueError("Physical path ids must be ordered 0..567")
        semantics = {
            int(value["path_id"]): value
            for value in semantics_payload.get("records", ())
        }
        rules = {
            int(value["path_id"]): value for value in rules_payload.get("records", ())
        }
        if set(semantics) != set(range(568)) or set(rules) != set(range(568)):
            raise ValueError("Physical path semantics/rules do not cover all 568 paths")

        straight_by_endpoints = {
            (int(value["start_lane"]), int(value["end_lane"])): int(value["path_id"])
            for value in raw
            if str(value["kind"]) == "slide" and "-" in str(value["key"])
        }
        aliases: list[int] = []
        for value in raw:
            path_id = int(value["path_id"])
            if str(value["kind"]) != "wifi":
                aliases.append(path_id)
                continue
            endpoint = (int(value["start_lane"]), int(value["end_lane"]))
            if endpoint not in straight_by_endpoints:
                raise ValueError(f"WiFi path has no straight alias: {value['key']}")
            aliases.append(straight_by_endpoints[endpoint])

        offsets = [0]
        masks: list[int] = []
        proportions: list[float] = []
        critical: list[float] = []
        flags: list[int] = []
        for original_id, alias_id in enumerate(aliases):
            record = semantics[alias_id]
            sequence = tuple(
                tuple(str(pad) for pad in area)
                for area in record.get("judge_sequence", ())
            )
            if not sequence:
                raise ValueError(
                    f"Physical path {original_id} has no v1 judge sequence"
                )
            contacts_by_pad: dict[str, list[float]] = {}
            for contact in record.get("contacts", ()):
                contacts_by_pad.setdefault(str(contact["pad"]), []).append(
                    float(contact["start_proportion"])
                )
            previous = 0.0
            for area_index, area in enumerate(sequence):
                mask = pads_mask(list(area))
                if not mask:
                    raise ValueError(
                        f"Physical path {original_id} contains an empty area"
                    )
                starts = [
                    start for pad in area for start in contacts_by_pad.get(str(pad), ())
                ]
                if area_index == 0:
                    proportion = 0.0
                elif starts:
                    proportion = min(starts)
                else:
                    raise ValueError(
                        f"Physical path {original_id} area has no contact timing: {area}"
                    )
                proportion = min(1.0, max(previous, float(proportion)))
                masks.append(mask)
                proportions.append(proportion)
                previous = proportion
            offsets.append(len(masks))
            rule = rules[alias_id]
            critical.append(float(rule["critical_proportion"]))
            flags.append(
                int(bool(rule.get("is_l"))) | (int(bool(rule.get("is_special_l"))) << 1)
            )

        codec = Route71Codec(vocabulary)
        by_factor: dict[tuple[int, int, int], dict[int, int]] = {}
        for value in raw:
            fields = value["field_ids"]
            key = (
                int(fields["shape"]),
                int(fields["end_offset"]),
                int(fields.get("via_offset", -100)),
            )
            lane = int(value["start_lane"])
            if lane not in range(1, 9) or int(fields["start_pad"]) != vocabulary.fields[
                "start_pad"
            ].encode(f"A{lane}"):
                raise ValueError("Physical path/start vocabulary mismatch")
            family = by_factor.setdefault(key, {})
            if lane in family:
                raise ValueError("duplicate Physical path/start pair")
            family[lane] = int(value["path_id"])
        route_rows: list[tuple[int, ...]] = []
        for route in range(71):
            restored = codec.restore_factor_ids(route)
            key = (
                int(restored["shape"]),
                int(restored["end_offset"]),
                int(restored.get("via_offset", -100)),
            )
            template = by_factor.get(key)
            if template is None:
                raise ValueError(f"Route-71 path template is missing at class {route}")
            if set(template) != set(range(1, 9)):
                raise ValueError("Physical route is not closed under rotation")
            route_rows.append(tuple(template[lane] for lane in range(1, 9)))
        route_table = (
            torch.tensor(route_rows, dtype=torch.long).transpose(0, 1).contiguous()
        )
        if (
            route_table.shape != (8, 71)
            or int(torch.unique(route_table).numel()) != 568
        ):
            raise ValueError("Route-71/path lookup must be a 568-path bijection")

        # Spatially augmented decoder rows use viewed start pads and Route-71
        # classes, while cached actions retain canonical path ids.  Judge-mask
        # sequence plus transformed start lane is a bijection for the 560
        # non-WiFi paths; WiFi deliberately resolves through its straight alias.
        from maimai_baseline.physical.spatial import LANE_PERMUTATIONS

        by_signature: dict[tuple[int, tuple[int, ...]], int] = {}
        for path_id in range(560):
            begin, end = offsets[path_id], offsets[path_id + 1]
            signature = (
                int(raw[path_id]["start_lane"]),
                tuple(int(value) for value in masks[begin:end]),
            )
            if signature in by_signature:
                raise ValueError("Physical path/view signature is not unique")
            by_signature[signature] = path_id
        path_views: list[list[int]] = []
        for view, permutation in enumerate(LANE_PERMUTATIONS):
            viewed: list[int] = []
            for path_id, alias_id in enumerate(aliases):
                begin, end = offsets[alias_id], offsets[alias_id + 1]
                start_lane = permutation[int(raw[alias_id]["start_lane"]) - 1] + 1
                signature = (
                    start_lane,
                    tuple(
                        transform_mask(int(value), view) for value in masks[begin:end]
                    ),
                )
                if signature not in by_signature:
                    raise ValueError(
                        f"Physical path {path_id} has no spatial-view mapping for view {view}"
                    )
                viewed.append(by_signature[signature])
            path_views.append(viewed)

        return cls(
            raw_paths=raw,
            route_path_ids=route_table,
            path_offsets=torch.tensor(offsets, dtype=torch.long),
            path_masks=torch.tensor(masks, dtype=torch.long),
            path_enter_proportions=torch.tensor(proportions, dtype=torch.float64),
            path_critical_proportions=torch.tensor(critical, dtype=torch.float64),
            path_flags=torch.tensor(flags, dtype=torch.long),
            path_aliases=torch.tensor(aliases, dtype=torch.long),
            path_ids_by_view=torch.tensor(path_views, dtype=torch.long),
            identity={
                "schema": "maimai.physical_bitplane_path_catalog.v1",
                "paths_sha256": _sha256(paths_path),
                "semantics_sha256": _sha256(semantics_path),
                "rules_sha256": _sha256(rules_path),
                "semantics_fingerprint": semantics_payload.get("fingerprint"),
                "rules_fingerprint": rules_payload.get("fingerprint"),
                "wifi": "same-endpoint-straight-slide-v1",
            },
        )

    def path_id(self, start_lane: int, route_class: int) -> int:
        if not 1 <= int(start_lane) <= 8 or not 0 <= int(route_class) < 71:
            return -1
        return int(self.route_path_ids[int(start_lane) - 1, int(route_class)])

    def path_key(self, path: int) -> str:
        # The immutable Route-71 catalog also supports compact CPU fixtures
        # without raw geometry records. Cache once, outside candidate sweeps.
        if self.raw_paths:
            return str(self.raw_paths[path]['key'])
        if not hasattr(self, '_path_keys'):
            variants = json.loads((Path(__file__).resolve().parents[1] /
                'assets/slide_variant_catalog.json').read_text('utf-8'))['variants']
            self._path_keys = {self.path_id(lane, route): value['paths_by_start'][lane-1]
                for route, value in enumerate(variants) for lane in range(1, 9)}
        return self._path_keys[path]

    def covers(self, previous: int, following: int) -> bool:
        """Directed full-path coverage; geometry aliases are not this relation."""
        old, new = self.path_key(previous), self.path_key(following)
        if old == new:
            return True
        if 'w' not in old or '-' not in new or old[0] != new[0]:
            return False
        end = int(old[-1])
        return int(new[-1]) in ((end-2) % 8+1, end, end % 8+1)

    def device_tensors(self, device: torch.device | str) -> Mapping[str, torch.Tensor]:
        key = str(torch.device(device))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                "route_path_ids": self.route_path_ids.to(device=device),
                "path_offsets": self.path_offsets.to(device=device),
                "path_masks": self.path_masks.to(device=device),
                "path_masks_by_view": torch.stack(
                    [
                        torch.tensor(
                            [
                                transform_mask(int(mask), view)
                                for mask in self.path_masks
                            ],
                            dtype=torch.long,
                        )
                        for view in range(4)
                    ],
                    dim=0,
                ).to(device=device),
                "path_enter_proportions": self.path_enter_proportions.to(device=device),
                "path_critical_proportions": self.path_critical_proportions.to(
                    device=device
                ),
                "path_flags": self.path_flags.to(device=device),
                "path_aliases": self.path_aliases.to(device=device),
                "path_ids_by_view": self.path_ids_by_view.to(device=device),
            }
            route_path = cached["route_path_ids"].reshape(-1)
            aliases = cached["path_aliases"].index_select(0, route_path).reshape(8, 71)
            cached["route_alias_word"] = torch.div(aliases, 64, rounding_mode="floor")
            cached["route_alias_bit"] = aliases.remainder(64)
            cached["route_candidate_allowed"] = cached["path_flags"].index_select(
                0, route_path).reshape(8, 71).bitwise_and(3).eq(0)
            self._device_cache[key] = cached
        return cached


__all__ = ["CompactPathCatalog", "pad_bit", "pads_mask"]
