from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .encoding import TIME_FIELDS
from .codec import EVENTS, ROW_CODEC, VOCABULARY_SCHEMA


NA_TOKEN = "<NA>"
OOV_TOKEN = "<OOV>"


def scalar_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scalar_value(key: str) -> Any:
    return json.loads(key)


@dataclass(frozen=True)
class FieldVocabulary:
    values: tuple[str, ...]

    @property
    def index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.values)}

    def encode(self, value: Any) -> int:
        return self.index.get(scalar_key(value), 1)

    def decode(self, index: int) -> Any | None:
        if index < 2 or index >= len(self.values):
            return None
        return scalar_value(self.values[index])


class CompactVocabulary:
    """Vocabulary for the only supported v2-test1 baseline row format."""

    SCHEMA = VOCABULARY_SCHEMA

    def __init__(
        self,
        *,
        fields: Mapping[str, Sequence[str]],
        applicability: Mapping[str, Sequence[str]],
        event_value_constraints: Mapping[str, Mapping[str, Sequence[str]]],
        note_kind_value_constraints: Mapping[str, Mapping[str, Sequence[str]]],
        time_values: Mapping[str, Sequence[int]],
        schema: str = SCHEMA,
        row_codec: str | None = None,
    ) -> None:
        if schema != VOCABULARY_SCHEMA:
            raise ValueError(f"unsupported baseline vocabulary schema: {schema!r}")
        if row_codec != ROW_CODEC:
            raise ValueError(
                "baseline vocabulary row codec mismatch: "
                f"expected={ROW_CODEC!r}, got={row_codec!r}"
            )
        self.schema = str(schema)
        self.row_codec = str(row_codec)
        self.events = tuple(EVENTS)
        self.fields = {
            str(name): FieldVocabulary(tuple(str(value) for value in values))
            for name, values in fields.items()
        }
        if "event" not in self.fields:
            raise ValueError("baseline vocabulary requires event")
        self.field_names = tuple(self.fields)
        self.applicability = {
            str(event): frozenset(str(field) for field in values)
            for event, values in applicability.items()
        }
        self.event_value_constraints = {
            str(event): {
                str(field): tuple(str(value) for value in values)
                for field, values in by_field.items()
            }
            for event, by_field in event_value_constraints.items()
        }
        self.note_kind_value_constraints = {
            str(kind): {
                str(field): tuple(str(value) for value in values)
                for field, values in by_field.items()
            }
            for kind, by_field in note_kind_value_constraints.items()
        }
        self.time_values = {
            field: tuple(sorted({int(value) for value in time_values.get(field, ())}))
            for field in TIME_FIELDS
        }
        self.time_indices = {
            field: {value: index for index, value in enumerate(values)}
            for field, values in self.time_values.items()
        }
        self._event_ids = {
            event: self.fields["event"].encode(event) for event in self.events
        }
        if any(value == 1 for value in self._event_ids.values()):
            raise ValueError("baseline vocabulary is missing a required event")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompactVocabulary":
        return cls(
            fields=value["fields"],
            applicability=value["applicability"],
            event_value_constraints=value.get("event_value_constraints", {}),
            note_kind_value_constraints=value.get("note_kind_value_constraints", {}),
            time_values=value.get("time_values", {}),
            schema=str(value.get("schema")),
            row_codec=value.get("row_codec"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_codec": self.row_codec,
            "fields": {
                name: list(vocabulary.values) for name, vocabulary in self.fields.items()
            },
            "applicability": {
                event: sorted(fields) for event, fields in self.applicability.items()
            },
            "event_value_constraints": {
                event: {field: list(values) for field, values in by_field.items()}
                for event, by_field in self.event_value_constraints.items()
            },
            "note_kind_value_constraints": {
                kind: {field: list(values) for field, values in by_field.items()}
                for kind, by_field in self.note_kind_value_constraints.items()
            },
            "time_values": {
                field: list(values) for field, values in self.time_values.items()
            },
        }

    def encode_row(self, row: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, bool]]:
        encoded: dict[str, int] = {}
        active: dict[str, bool] = {}
        for field, vocabulary in self.fields.items():
            if field not in row:
                encoded[field] = 0
                active[field] = False
                continue
            encoded[field] = vocabulary.encode(row[field])
            active[field] = encoded[field] >= 2
        return encoded, active

    def decode_row(self, encoded: Mapping[str, int]) -> dict[str, Any]:
        event = self.fields["event"].decode(int(encoded.get("event", 0)))
        if not isinstance(event, str):
            raise ValueError("baseline row has no valid event")
        row: dict[str, Any] = {"event": event}
        for field, index in encoded.items():
            if field == "event" or field not in self.fields:
                continue
            value = self.fields[field].decode(int(index))
            if value is not None and field in self.applicability.get(event, ()):
                row[field] = value
        return row

    def event_id(self, event: str) -> int:
        return self._event_ids[event]

    def allowed_next_events(self, previous_event: str) -> tuple[str, ...]:
        transitions = {
            "SEGMENT_START": ("MOMENT", "TARGET_START"),
            "TARGET_START": ("MOMENT", "SEGMENT_END"),
            "SEGMENT_END": (),
            "MOMENT": ("NOTE",),
            "NOTE": ("NOTE", "SLIDE_BRANCH", "MOMENT", "TARGET_START", "SEGMENT_END"),
            "SLIDE_BRANCH": (
                "SLIDE_BRANCH",
                "NOTE",
                "MOMENT",
                "TARGET_START",
                "SEGMENT_END",
            ),
        }
        return transitions[previous_event]

    def allowed_next_event_ids(self, previous_id: int) -> tuple[int, ...]:
        previous = self.fields["event"].decode(int(previous_id))
        if not isinstance(previous, str):
            return tuple(self._event_ids.values())
        return tuple(self.event_id(event) for event in self.allowed_next_events(previous))

    def allowed_value_ids(
        self,
        event: str,
        field: str,
        *,
        note_kind: str | None = None,
    ) -> tuple[int, ...]:
        if field not in self.fields:
            return ()
        allowed: Sequence[str] | None = None
        if event == "NOTE" and note_kind is not None:
            allowed = self.note_kind_value_constraints.get(note_kind, {}).get(field)
        if allowed is None:
            allowed = self.event_value_constraints.get(event, {}).get(field)
        if allowed is None:
            return tuple(range(2, len(self.fields[field].values)))
        index = self.fields[field].index
        return tuple(index[value] for value in allowed if value in index and index[value] >= 2)


__all__ = ["CompactVocabulary", "FieldVocabulary", "NA_TOKEN", "OOV_TOKEN"]
