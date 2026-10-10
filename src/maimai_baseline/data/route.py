"""Route-71 row encoding and vocabulary mapping."""

from __future__ import annotations
from collections.abc import Mapping
from maimai_baseline.contract import REMOVED_FIELDS, ROUTE_CLASSES
from maimai_dataset.encoding import factorize_path, restore_path
from maimai_dataset.vocabulary import CompactVocabulary
from pathlib import Path
from typing import Any
import json
import torch


# Route-71 data encoding


BASIC_CATALOG_RELATIVE_PATH = Path(
    "assets/slide_variant_catalog.json"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class Route71Codec:
    def __init__(
        self,
        vocabulary: CompactVocabulary,
        catalog_path: Path | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        path = (
            Path(catalog_path)
            if catalog_path is not None
            else (_project_root() / BASIC_CATALOG_RELATIVE_PATH)
        ).resolve()
        payload = json.loads(path.read_text("utf-8"))
        variants = payload["variants"]
        if [int(value["variant_id"]) for value in variants] != list(
            range(ROUTE_CLASSES)
        ):
            raise RuntimeError("basic Slide variant catalog is not B000-B070 ordered")
        self.catalog_path = path
        self.catalog_identity = str(payload["identity"])
        self.paths = tuple(str(value["canonical_path"]) for value in variants)
        self._factor_to_variant: dict[tuple[str, int, int | None], int] = {}
        for variant in variants:
            factored = factorize_path(str(variant["canonical_path"]), start_pad="A1")
            key = (
                str(factored["shape"]),
                int(factored["end_offset"]),
                (
                    None
                    if "via_offset" not in factored
                    else int(factored["via_offset"])
                ),
            )
            if key in self._factor_to_variant:
                raise RuntimeError(f"duplicate basic route factorization: {key}")
            self._factor_to_variant[key] = int(variant["variant_id"])
        if len(self._factor_to_variant) != ROUTE_CLASSES:
            raise RuntimeError("basic route factorization does not contain 71 classes")

    def _decode_factor(
        self, values: Mapping[str, torch.Tensor], index: int, *, target: bool
    ) -> tuple[str, int, int | None] | None:
        inactive = -100 if target else 0
        shape_id = int(values["shape"][index])
        end_id = int(values["end_offset"][index])
        via_id = int(values["via_offset"][index])
        if shape_id == inactive and end_id == inactive and via_id == inactive:
            return None
        if target:
            if shape_id < 0 or end_id < 0:
                raise ValueError("partial active route target")
        elif shape_id == 0 or end_id == 0:
            raise ValueError("partial active route input")
        shape = self.vocabulary.fields["shape"].decode(shape_id)
        end = self.vocabulary.fields["end_offset"].decode(end_id)
        via = (
            None
            if via_id == inactive
            else self.vocabulary.fields["via_offset"].decode(via_id)
        )
        if not isinstance(shape, str) or not isinstance(end, int):
            raise ValueError("route factor contains OOV/NA shape or end")
        if via is not None and not isinstance(via, int):
            raise ValueError("route factor contains OOV/NA via")
        return shape, end, via

    def encode_mapping(
        self, values: Mapping[str, torch.Tensor], *, target: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(values["shape"].shape[0])
        fill = -100 if target else 0
        route = torch.full((length,), fill, dtype=torch.long)
        mass = torch.zeros((length,), dtype=torch.float32)
        event = values["event"]
        kind = values.get("kind")
        for index in range(length):
            factor = self._decode_factor(values, index, target=target)
            if factor is None:
                continue
            variant = self._factor_to_variant.get(factor)
            if variant is None:
                shape, end, via = factor
                candidate = restore_path(
                    start_pad="A1", shape=shape, end_offset=end, via_offset=via
                )
                raise ValueError(
                    f"compound/OOV Slide route is not legal in pilot Route-71: {candidate}"
                )
            route[index] = variant if target else variant + 1
            if target:
                event_name = self.vocabulary.fields["event"].decode(int(event[index]))
                kind_name = (
                    None
                    if kind is None
                    else self.vocabulary.fields["kind"].decode(int(kind[index]))
                )
                informative = 0
                for field in REMOVED_FIELDS:
                    if int(values[field][index]) < 0:
                        continue
                    allowed = self.vocabulary.allowed_value_ids(
                        str(event_name),
                        field,
                        note_kind=(str(kind_name) if event_name == "NOTE" else None),
                    )
                    informative += int(len(allowed) > 1)
                mass[index] = float(informative)
                if event_name != "NOTE" and informative < 2:
                    raise ValueError("active branch route lost informative slot mass")
        return route, mass

    def adapt_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(item)
        result["input_ids"] = dict(item["input_ids"])
        result["target_ids"] = dict(item["target_ids"])
        route_input, _ = self.encode_mapping(result["input_ids"], target=False)
        route_target, route_mass = self.encode_mapping(result["target_ids"], target=True)
        for field in REMOVED_FIELDS:
            result["input_ids"].pop(field)
            result["target_ids"].pop(field)
        result["input_ids"]["route"] = route_input
        result["target_ids"]["route"] = route_target
        result["route_informative_mass"] = route_mass
        return result

    def restore_factor_ids(self, route_class: int) -> dict[str, int]:
        if not 0 <= int(route_class) < ROUTE_CLASSES:
            raise ValueError(f"invalid Route-71 class: {route_class}")
        factored = factorize_path(self.paths[int(route_class)], start_pad="A1")
        result = {
            "shape": self.vocabulary.fields["shape"].encode(factored["shape"]),
            "end_offset": self.vocabulary.fields["end_offset"].encode(
                factored["end_offset"]
            ),
            "via_offset": -100,
        }
        if "via_offset" in factored:
            result["via_offset"] = self.vocabulary.fields["via_offset"].encode(
                factored["via_offset"]
            )
        return result
