"""Preparation and audit helpers for the explicit full train-only dataset.

This module is deliberately separate from the historical pilot preparation
flow.  It can prepare the chart and BPM inputs from the frozen full selection,
but it never downloads or invents Physical data.  A missing full spatial or
Physical artifact is reported as a blocker and keeps the runtime manifest
``ready`` flag false.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import platform
import shutil
import sys

import numpy as np

from maimai_baseline.data.audio import TARGET_P50, extract_at_centers, validate_stats
from maimai_baseline.data.store import (
    GEOMETRY_SCHEMA,
    SCHEMA as BPM_SCHEMA,
    completed,
    contained,
    publish,
    window_arrays,
)
from maimai_baseline.source_identity import baseline_source_identity
from maimai_baseline.time.grid import INTEGRATION, SAMPLING, audio_grid, identity, timing_identity
from maimai_dataset.chart_cache import build_chart_vector_cache
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.timing import TimingMap
from maimai_dataset.window import WINDOW


CONFIG_SCHEMA = "maimai.full_dataset_preparation_spec.v1"
SELECTION_SCHEMA = "maimai.full_train_selection.v1"
BUILD_SCHEMA = "maimai.full_dataset_build.v1"
VERIFY_SCHEMA = "maimai.full_dataset_verification.v1"
DATASET_SCHEMA = "maimai.training_dataset.v1"


class FullDatasetError(ValueError):
    """A deterministic preparation or provenance failure."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FullDatasetError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FullDatasetError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    if path.exists():
        try:
            if json.loads(path.read_text(encoding="utf-8-sig")) == value:
                return
        except Exception:
            pass
        if not replace:
            raise FullDatasetError(f"refusing to overwrite conflicting artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    if path.exists():
        if file_sha256(path) != file_sha256(temporary):
            temporary.unlink()
            raise FullDatasetError(f"refusing to overwrite conflicting JSONL artifact: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def _sha(path: Path) -> str:
    return file_sha256(Path(path))


def _path_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FullDatasetError(f"source payload is missing: {source}")
    if _sha(source) != expected_sha256:
        raise FullDatasetError(f"source payload changed after audit: {source}")
    if destination.exists():
        if not destination.is_file() or _sha(destination) != expected_sha256:
            raise FullDatasetError(f"destination payload conflicts with audited source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    if _sha(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise FullDatasetError(f"copied payload hash mismatch: {destination}")
    temporary.replace(destination)


def _canonical_without(value: Mapping[str, Any], key: str) -> str:
    return canonical_json_sha256({k: v for k, v in value.items() if k != key})


def _runtime_environment() -> dict[str, Any]:
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
    }


@dataclass(frozen=True)
class FullDatasetSpec:
    config_path: Path
    root: Path
    runtime: Path
    reports: Path
    workspace_root: Path
    config: dict[str, Any]
    selection: dict[str, Any]
    source_profile: Path
    base_dataset: Path
    normalized_audio: Path
    legacy_selection: Path

    @classmethod
    def load(cls, config_path: Path) -> "FullDatasetSpec":
        config_path = Path(config_path).expanduser().resolve()
        config = _read_json(config_path)
        if config.get("schema") != CONFIG_SCHEMA:
            raise FullDatasetError(f"unsupported full dataset config schema: {config.get('schema')!r}")
        root = config_path.parent
        selection_path = contained(root, config.get("selection", "selection.json"))
        selection = _read_json(selection_path)
        if selection.get("schema") != SELECTION_SCHEMA:
            raise FullDatasetError(f"unsupported full selection schema: {selection.get('schema')!r}")
        workspace_root = (root / str(config.get("workspace_root", "../../../.."))).resolve()
        sources = config.get("sources")
        if not isinstance(sources, dict):
            raise FullDatasetError("full dataset config has no sources object")
        source_profile = (workspace_root / str(sources["profile"])).resolve()
        base_dataset = (workspace_root / str(sources["base_dataset"])).resolve()
        normalized_audio = (workspace_root / str(sources["normalized_audio"])).resolve()
        legacy_selection = contained(root, selection.get("provenance", {}).get("historical_selection", "provenance/legacy-selection.json"))
        for path in (source_profile, base_dataset, normalized_audio, legacy_selection):
            if not path.exists():
                raise FullDatasetError(f"configured source is missing: {path}")
        if config.get("window") != WINDOW:
            raise FullDatasetError("full dataset config window does not match current baseline window")
        if config.get("audio_policy") != "normalized-p50":
            raise FullDatasetError("full dataset preparation requires normalized-p50")
        if config.get("training_level_gain") is not False:
            raise FullDatasetError("full dataset training must disable level gain")
        if config.get("historical_selection_policy", "require-record-level-evidence") not in {
            "require-record-level-evidence",
            "current-selection-authoritative",
        }:
            raise FullDatasetError("unsupported historical selection policy")
        if selection.get("validation_song_ids") != []:
            raise FullDatasetError("full train-only selection must declare an empty validation_song_ids list")
        actual_selection_sha = _sha(selection_path)
        declared_selection_sha = selection.get("selection_sha256")
        if declared_selection_sha and _canonical_without(selection, "selection_sha256") != declared_selection_sha:
            raise FullDatasetError("declared full selection content SHA-256 does not match its contents")
        expected_selection_sha = config.get("selection_sha256")
        if expected_selection_sha and expected_selection_sha != actual_selection_sha:
            raise FullDatasetError("full config/selection SHA-256 mismatch")
        return cls(
            config_path=config_path,
            root=root,
            runtime=contained(root, config.get("output_root", "runtime")),
            reports=contained(root, "reports"),
            workspace_root=workspace_root,
            config=config,
            selection=selection,
            source_profile=source_profile,
            base_dataset=base_dataset,
            normalized_audio=normalized_audio,
            legacy_selection=legacy_selection,
        )

    @property
    def selection_sha256(self) -> str:
        return _sha(self.root / str(self.config.get("selection", "selection.json")))

    @property
    def selected_record_ids(self) -> list[str]:
        return [str(value) for value in self.selection.get("record_ids", [])]

    @property
    def selected_song_ids(self) -> list[str]:
        return [str(value) for value in self.selection.get("train_song_ids", [])]

    @property
    def epoch_plan_budgets(self) -> tuple[int, ...]:
        values = self.config.get("epoch_plan_budgets", [3072, 4096])
        if not isinstance(values, list) or not values:
            raise FullDatasetError("full dataset config must declare epoch_plan_budgets")
        budgets = tuple(int(value) for value in values)
        if len(set(budgets)) != len(budgets) or any(value not in (3072, 4096) for value in budgets):
            raise FullDatasetError(f"unsupported or duplicate epoch plan budgets: {budgets!r}")
        return budgets


def _source_records(spec: FullDatasetSpec) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path = spec.source_profile / "manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in by_id:
            raise FullDatasetError(f"duplicate or missing source record_id: {record_id!r}")
        by_id[record_id] = row
    return rows, by_id


def _source_templates(spec: FullDatasetSpec, selected: set[str]) -> dict[str, dict[str, Any]]:
    path = spec.source_profile / "segments.jsonl"
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", ""))
            if record_id in selected:
                result.setdefault(record_id, row)
    return result


def _normalization_rows(spec: FullDatasetSpec) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    path = spec.normalized_audio / "audio-manifest.jsonl"
    rows = _read_jsonl(path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    output_paths: set[str] = set()
    for row in rows:
        if row.get("schema") != "maimai.audio_fixed_gain_p50.v1":
            raise FullDatasetError(f"unsupported normalized audio row schema: {row.get('schema')!r}")
        output = str(row.get("output_relative_path", ""))
        if not output or output in output_paths:
            raise FullDatasetError(f"duplicate normalized audio output path: {output!r}")
        output_paths.add(output)
        canonical_song = row.get("canonical_song_id")
        if canonical_song is None:
            canonical_song = ""
        key = (str(canonical_song), str(row.get("source_sha256", "")))
        by_key[key] = row
    return rows, by_key


def _selected_audio(spec: FullDatasetSpec, records: Sequence[Mapping[str, Any]], norm_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for record in records:
        song = str(record["song_id"])
        source_sha = str(record["audio"]["file_sha256"])
        row = norm_by_key.get((song, source_sha))
        if row is None:
            # Historical trimmed assets intentionally have no canonical_song_id
            # in the normalization manifest.  Their source SHA is still the
            # unambiguous binding and must be matched to the selected song.
            matches = [
                candidate
                for (canonical_song, digest), candidate in norm_by_key.items()
                if digest == source_sha and canonical_song in ("", song)
            ]
            if len(matches) == 1:
                row = matches[0]
                mapping = "source-sha-fallback"
            elif len(matches) > 1:
                failures.append({
                    "song_id": song,
                    "record_id": str(record["record_id"]),
                    "kind": "normalized-audio-mapping-ambiguous",
                    "candidate_count": len(matches),
                })
                continue
        else:
            mapping = "exact-song-source-sha"
        if row is None:
            # A small number of preserved profile rows contain stale derived
            # audio metadata while the canonical normalized manifest points
            # to the current, hash-verifiable source file.  Bind by the
            # explicit canonical song id only when it is unique; the audit
            # records the metadata drift and _audit_record verifies the actual
            # source against the normalized manifest SHA.
            canonical_matches = [
                candidate
                for (canonical_song, _digest), candidate in norm_by_key.items()
                if canonical_song == song
            ]
            if len(canonical_matches) == 1:
                row = canonical_matches[0]
                mapping = "canonical-song-fallback"
            elif len(canonical_matches) > 1:
                failures.append({
                    "song_id": song,
                    "record_id": str(record["record_id"]),
                    "kind": "normalized-audio-mapping-ambiguous",
                    "candidate_count": len(canonical_matches),
                })
                continue
        if row is None:
            failures.append({"song_id": song, "record_id": str(record["record_id"]), "kind": "normalized-audio-mapping"})
            continue
        row = dict(row)
        row["source_mapping"] = mapping
        row["profile_source_sha256"] = source_sha
        previous = result.get(song)
        if previous is not None and previous.get("output_sha256") != row.get("output_sha256"):
            failures.append({"song_id": song, "record_id": str(record["record_id"]), "kind": "inconsistent-song-audio"})
        else:
            result[song] = dict(row)
    return result, failures


def _check_path(path: Path, expected_sha: str | None, *, hash_payloads: bool, kind: str, failures: list[dict[str, Any]]) -> str | None:
    if not path.is_file():
        failures.append({"path": str(path), "kind": kind, "error": "missing"})
        return None
    if not hash_payloads:
        return expected_sha
    try:
        actual = _sha(path)
    except Exception as error:
        failures.append({"path": str(path), "kind": kind, "error": str(error)})
        return None
    if expected_sha is not None and actual != expected_sha:
        failures.append({"path": str(path), "kind": kind, "expected_sha256": expected_sha, "actual_sha256": actual})
    return actual


def _audit_record(
    spec: FullDatasetSpec,
    record: Mapping[str, Any],
    template: Mapping[str, Any] | None,
    audio: Mapping[str, Any] | None,
    *,
    hash_payloads: bool,
    failures: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    audio_hashes: dict[str, str],
) -> dict[str, Any]:
    record_id = str(record["record_id"])
    encoded_path = contained(spec.source_profile, record["encoded_path"])
    ir_path = contained(spec.base_dataset, record["ir_path"])
    timing_path = contained(spec.base_dataset, record["timing_path"])
    encoded_sha = _check_path(encoded_path, None, hash_payloads=hash_payloads, kind="encoded", failures=failures)
    ir_file_sha = _check_path(ir_path, None, hash_payloads=hash_payloads, kind="ir", failures=failures)
    timing_file_sha = _check_path(timing_path, None, hash_payloads=hash_payloads, kind="timing", failures=failures)
    semantic: dict[str, Any] = {"encoded": False, "ir": False, "timing": False, "audio": False, "normalized_audio": False}
    document: dict[str, Any] | None = None
    if encoded_path.is_file():
        try:
            document = _read_json(encoded_path)
            from maimai_baseline.data.prepare import encoded_moments
            encoded_moments(document, record)
            semantic["encoded"] = True
        except Exception as error:
            failures.append({"record_id": record_id, "kind": "encoded-semantics", "error": str(error)})
    ir_document: dict[str, Any] | None = None
    if ir_path.is_file():
        try:
            ir_document = _read_json(ir_path)
            if canonical_json_sha256(ir_document) != str(record["ir_sha256"]):
                raise FullDatasetError("canonical IR SHA-256 mismatch")
            from maimai_chart_ir.models import validate_chart_ir
            validate_chart_ir(ir_document)
            semantic["ir"] = True
        except Exception as error:
            failures.append({"record_id": record_id, "kind": "ir-semantics", "error": str(error)})
    if timing_path.is_file() and template is not None:
        try:
            timing_document = _read_json(timing_path)
            actual_timing = TimingMap.from_dict(timing_document)
            template_timing = TimingMap.from_dict(template["timing"])
            if timing_identity(actual_timing) != timing_identity(template_timing):
                raise FullDatasetError("source timing and profile timing differ")
            if ir_document is not None:
                from maimai_dataset.timing import TimingMap as IRTimingMap
                if timing_identity(IRTimingMap.from_document(ir_document)) != timing_identity(actual_timing):
                    raise FullDatasetError("source IR and timing file differ")
            semantic["timing"] = True
        except Exception as error:
            failures.append({"record_id": record_id, "kind": "timing-semantics", "error": str(error)})
    audio_hash = None
    wav_hash = None
    audio_path = normalized_path = None
    profile_audio_sha = None
    normalized_source_sha = None
    if audio is not None:
        audio_path = (spec.workspace_root / str(audio["relative_source_path"])).resolve()
        normalized_path = contained(spec.normalized_audio, audio["output_relative_path"])
        profile_audio_sha = str(record["audio"]["file_sha256"])
        normalized_source_sha = str(audio["source_sha256"])
        if profile_audio_sha != normalized_source_sha:
            notes.append({
                "record_id": record_id,
                "song_id": str(record["song_id"]),
                "kind": "source-profile-audio-metadata-drift",
                "profile_sha256": profile_audio_sha,
                "normalized_source_sha256": normalized_source_sha,
                "mapping": audio.get("source_mapping"),
            })
        if str(audio_path) not in audio_hashes:
            actual = _check_path(audio_path, normalized_source_sha, hash_payloads=hash_payloads, kind="source-audio", failures=failures)
            if actual is not None:
                audio_hashes[str(audio_path)] = actual
        audio_hash = audio_hashes.get(str(audio_path), normalized_source_sha)
        wav_key = str(normalized_path)
        if wav_key not in audio_hashes:
            actual = _check_path(normalized_path, str(audio["output_sha256"]), hash_payloads=hash_payloads, kind="normalized-audio", failures=failures)
            if actual is not None:
                audio_hashes[wav_key] = actual
        wav_hash = audio_hashes.get(wav_key, str(audio["output_sha256"]))
        semantic["audio"] = audio_path.is_file() and (not hash_payloads or audio_hash == normalized_source_sha)
        semantic["normalized_audio"] = normalized_path.is_file() and (not hash_payloads or wav_hash == str(audio["output_sha256"]))
    return {
        "record_id": record_id,
        "song_id": str(record["song_id"]),
        "source_split": str(record.get("split", "")),
        "difficulty_id": int(record["source_difficulty_id"]),
        "version": str(record.get("version", "")),
        "encoded_path": str(encoded_path),
        "encoded_file_sha256": encoded_sha,
        "ir_path": str(ir_path),
        "ir_file_sha256": ir_file_sha,
        "ir_canonical_sha256": str(record.get("ir_sha256", "")),
        "timing_path": str(timing_path),
        "timing_file_sha256": timing_file_sha,
        "audio_path": None if audio_path is None else str(audio_path),
        "audio_file_sha256": audio_hash,
        "profile_audio_sha256": profile_audio_sha,
        "normalized_source_sha256": normalized_source_sha,
        "normalized_audio_path": None if normalized_path is None else str(normalized_path),
        "normalized_audio_sha256": wav_hash,
        "semantic": semantic,
    }


def audit_sources(spec: FullDatasetSpec, *, hash_payloads: bool = True) -> dict[str, Any]:
    """Audit and freeze the current full selection without changing it."""
    rows, by_id = _source_records(spec)
    selected_ids = spec.selected_record_ids
    failures: list[dict[str, Any]] = []
    if len(selected_ids) != len(set(selected_ids)):
        failures.append({"kind": "selection-duplicates"})
    missing_ids = [record_id for record_id in selected_ids if record_id not in by_id]
    if missing_ids:
        failures.append({"kind": "selection-records-missing", "record_ids": missing_ids[:32], "count": len(missing_ids)})
    selected_records = [by_id[record_id] for record_id in selected_ids if record_id in by_id]
    selected_songs = set(spec.selected_song_ids)
    if len(spec.selected_song_ids) != len(selected_songs):
        failures.append({"kind": "selection-song-duplicates"})
    if {str(row["song_id"]) for row in selected_records} != selected_songs:
        failures.append({"kind": "selection-song-coverage"})
    config_acceptance = spec.config.get("acceptance", {})
    if len(selected_records) != int(config_acceptance.get("candidate_charts", len(selected_records))):
        failures.append({"kind": "selection-chart-count", "actual": len(selected_records), "expected": config_acceptance.get("candidate_charts")})
    if len(selected_songs) != int(config_acceptance.get("songs", len(selected_songs))):
        failures.append({"kind": "selection-song-count", "actual": len(selected_songs), "expected": config_acceptance.get("songs")})
    forbidden_songs = {str(value) for value in config_acceptance.get("forbidden_song_ids", [])}
    forbidden_records = {str(value) for value in config_acceptance.get("forbidden_record_ids", [])}
    forbidden_versions = {str(value) for value in config_acceptance.get("forbidden_versions", [])}
    excluded: list[dict[str, Any]] = []
    for row in selected_records:
        rid = str(row["record_id"])
        if str(row["song_id"]) in forbidden_songs or rid in forbidden_records or str(row.get("version")) in forbidden_versions:
            excluded.append({"record_id": rid, "song_id": str(row["song_id"]), "version": row.get("version")})
        if row.get("status") != "eligible" or row.get("base_status") not in (None, "eligible"):
            failures.append({"record_id": rid, "kind": "not-eligible", "status": row.get("status"), "base_status": row.get("base_status")})
    if excluded:
        failures.append({"kind": "forbidden-records-selected", "records": excluded})
    template_by_id = _source_templates(spec, set(selected_ids))
    template_missing = sorted(set(selected_ids) - set(template_by_id))
    if template_missing:
        failures.append({"kind": "source-segment-template-missing", "record_ids": template_missing[:32], "count": len(template_missing)})
    normalized_rows, norm_by_key = _normalization_rows(spec)
    audio_by_song, audio_failures = _selected_audio(spec, selected_records, norm_by_key)
    failures.extend(audio_failures)
    audio_manifest_path = spec.normalized_audio / "audio-manifest.jsonl"
    normalization_summary_path = spec.normalized_audio / "normalization-summary.json"
    try:
        normalization_summary = _read_json(normalization_summary_path)
        if normalization_summary.get("audio_manifest_sha256") != _sha(audio_manifest_path):
            failures.append({"kind": "normalization-manifest-identity"})
        if normalization_summary.get("target_p50_lufs") != TARGET_P50:
            failures.append({"kind": "normalization-target", "actual": normalization_summary.get("target_p50_lufs"), "expected": TARGET_P50})
    except Exception as error:
        normalization_summary = {}
        failures.append({"kind": "normalization-summary", "error": str(error)})
    audio_hashes: dict[str, str] = {}
    notes: list[dict[str, Any]] = []
    record_hashes: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(selected_records, 1):
        record_hashes[str(row["record_id"])] = _audit_record(
            spec, row, template_by_id.get(str(row["record_id"])), audio_by_song.get(str(row["song_id"])),
            hash_payloads=hash_payloads, failures=failures, audio_hashes=audio_hashes,
            notes=notes,
        )
        if index % 250 == 0 or index == len(selected_records):
            print(f"full source audit {index}/{len(selected_records)}", file=sys.stderr, flush=True)
    source_summary_path = spec.source_profile / "summary.json"
    source_manifest_path = spec.source_profile / "manifest.jsonl"
    source_summary = _read_json(source_summary_path)
    source_identity = {
        "profile_manifest_sha256": _sha(source_manifest_path),
        "profile_summary_sha256": _sha(source_summary_path),
        "profile_fingerprint": source_summary.get("profile_fingerprint"),
        "expected_profile_manifest_sha256": spec.selection.get("provenance", {}).get("source_profile_manifest_sha256"),
        "expected_profile_summary_sha256": spec.selection.get("provenance", {}).get("source_profile_summary_sha256"),
    }
    if source_identity["expected_profile_manifest_sha256"] and source_identity["profile_manifest_sha256"] != source_identity["expected_profile_manifest_sha256"]:
        failures.append({"kind": "source-profile-manifest-identity"})
    if source_identity["expected_profile_summary_sha256"] and source_identity["profile_summary_sha256"] != source_identity["expected_profile_summary_sha256"]:
        failures.append({"kind": "source-profile-summary-identity"})
    legacy = _read_json(spec.legacy_selection)
    historical_train = {str(value) for value in spec.selection.get("provenance", {}).get("historical_train_song_ids", [])}
    historical_validation = {str(value) for value in spec.selection.get("provenance", {}).get("historical_validation_song_ids", [])}
    historical_songs = historical_train | historical_validation
    legacy_counts = legacy.get("counts", {})
    historical_count = int(legacy_counts.get("train_charts", 0)) + int(legacy_counts.get("validation_charts", 0))
    historical_record_ids = legacy.get("record_ids")
    history_record_level_evidence = isinstance(historical_record_ids, list)
    history_policy = str(spec.config.get("historical_selection_policy", "require-record-level-evidence"))
    history_resolved = history_record_level_evidence
    same_song_set = historical_songs == selected_songs
    count_is_authorized = len(selected_records) == int(config_acceptance.get("candidate_charts", len(selected_records)))
    history_accepted = history_resolved or (
        history_policy == "current-selection-authoritative"
        and same_song_set
        and count_is_authorized
    )
    if not history_resolved:
        history_note = {
            "kind": "historical-record-diff-unresolved",
            "reason": "legacy selection contains song lists and counts but no record_ids",
            "delta": len(selected_records) - historical_count,
        }
        if history_accepted:
            notes.append({
                **history_note,
                "kind": "historical-record-diff-accepted",
                "policy": history_policy,
                "accepted_current_chart_count": len(selected_records),
            })
        else:
            failures.append(history_note)
    history = {
        "historical_chart_count": historical_count,
        "current_selected_chart_count": len(selected_records),
        "delta": len(selected_records) - historical_count,
        "historical_record_level_evidence": history_record_level_evidence,
        "resolved": history_resolved,
        "accepted": history_accepted,
        "policy": history_policy,
        "same_song_set": same_song_set,
        "historical_song_count": len(historical_songs),
        "current_song_count": len(selected_songs),
        "explanation": (
            "current 4147-record selection accepted as authoritative by explicit preparation policy; legacy record_ids are unavailable"
            if history_accepted and not history_resolved
            else "record-level comparison performed"
            if history_resolved
            else "record-level reconciliation remains unresolved because the preserved legacy selection has no record_ids"
        ),
    }
    if not same_song_set:
        failures.append({"kind": "historical-song-set-diff", "historical_only": sorted(historical_songs - selected_songs), "current_only": sorted(selected_songs - historical_songs)})
    semantic_counts = Counter()
    for value in record_hashes.values():
        semantic_counts.update({key: int(flag) for key, flag in value["semantic"].items()})
    integrity_passed = not [
        failure for failure in failures
        if failure.get("kind") != "historical-record-diff-unresolved"
    ]
    identity_material = {
        "schema": "maimai.full_source_audit_identity.v1",
        "selection_sha256": spec.selection_sha256,
        "source_identity": source_identity,
        "record_hashes_sha256": canonical_json_sha256(record_hashes),
        "audio_manifest_sha256": _sha(audio_manifest_path),
        "normalization_summary_sha256": _sha(normalization_summary_path),
        "hash_payloads": hash_payloads,
        "historical_selection_policy": history_policy,
    }
    report = {
        "schema": "maimai.full_source_audit.v1",
        "complete": True,
        "passed": bool(integrity_passed and history["accepted"]),
        "source_integrity_passed": integrity_passed,
        "selection_frozen": True,
        "selection_sha256": spec.selection_sha256,
        "config_sha256": _sha(spec.config_path),
        "source_profile": str(spec.source_profile),
        "source_identity": source_identity,
        "source_summary": {
            "input_charts": source_summary.get("input_charts"),
            "eligible_charts": source_summary.get("eligible_charts"),
            "isolated_charts": source_summary.get("isolated_charts"),
            "eligible_split_counts": source_summary.get("eligible_split_counts"),
        },
        "selected": {
            "song_count": len(selected_songs),
            "chart_count": len(selected_records),
            "record_ids_sha256": canonical_json_sha256(selected_ids),
            "source_split_counts": dict(Counter(str(row.get("split", "")) for row in selected_records)),
            "difficulty_counts": dict(Counter(str(row["source_difficulty_id"]) for row in selected_records)),
            "version_counts": dict(Counter(str(row.get("version", "")) for row in selected_records)),
            "all_runtime_split": "train",
            "validation_song_count": 0,
        },
        "filters": {
            "forbidden_song_ids": sorted(forbidden_songs),
            "forbidden_record_ids": sorted(forbidden_records),
            "forbidden_versions": sorted(forbidden_versions),
            "selected_forbidden_records": excluded,
        },
        "audio": {
            "normalization_root": str(spec.normalized_audio),
            "manifest_sha256": _sha(audio_manifest_path),
            "normalization_summary_sha256": _sha(normalization_summary_path),
            "target_p50_lufs": normalization_summary.get("target_p50_lufs"),
            "selected_song_count": len(audio_by_song),
            "selected_audio_sources": audio_by_song,
            "hash_payloads": hash_payloads,
        },
        "semantic_counts": dict(semantic_counts),
        "record_hashes": record_hashes,
        "notes": notes,
        "historical_comparison": history,
        "failures": failures,
        "identity_material": identity_material,
        "audit_identity": canonical_json_sha256(identity_material),
        "environment": _runtime_environment(),
    }
    _write_json(spec.reports / "source-audit.json", report, replace=True)
    return report


def _load_audit(spec: FullDatasetSpec) -> dict[str, Any]:
    path = spec.reports / "source-audit.json"
    if not path.is_file():
        raise FullDatasetError(f"source audit is missing: {path}")
    report = _read_json(path)
    if (report.get("schema") != "maimai.full_source_audit.v1"
            or report.get("selection_sha256") != spec.selection_sha256
            or report.get("config_sha256") != _sha(spec.config_path)):
        raise FullDatasetError("source audit schema or selection identity mismatch")
    return report




def build_profile(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize a fresh 16-beat profile, remapping every source split to train."""
    _, by_id = _source_records(spec)
    selected_ids = spec.selected_record_ids
    templates = _source_templates(spec, set(selected_ids))
    profile_root = spec.runtime / "profile"
    from maimai_baseline.data.prepare import encoded_moments
    from maimai_baseline.data.training_build import segment_chart
    runtime_records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    for index, record_id in enumerate(selected_ids, 1):
        record = dict(by_id[record_id])
        record["source_split"] = str(record.get("split", ""))
        record["split"] = "train"
        source_path = contained(spec.source_profile, record["encoded_path"])
        expected = audit["record_hashes"][record_id]["encoded_file_sha256"]
        if expected is None:
            expected = _sha(source_path)
        _copy_verified(source_path, contained(profile_root, record["encoded_path"]), str(expected))
        document = _read_json(source_path)
        encoded_moments(document, record)
        template = dict(templates[record_id])
        template["source_split"] = str(template.get("split", record["source_split"]))
        template["split"] = "train"
        generated = segment_chart(document, template)
        for segment in generated:
            segment["source_split"] = record["source_split"]
            segment["split"] = "train"
        runtime_records.append(record)
        segments.extend(generated)
        coverage[record_id] = {
            "source_split": record["source_split"],
            "encoded_sha256": str(expected),
            "rows": len(document["rows"]),
            "notes": sum(row.get("event") == "NOTE" for row in document["rows"]),
            "segments": len(generated),
            "target_row_count": sum(int(row["target_row_end"]) - int(row["target_row_start"]) for row in generated),
            "timing_sha256": identity(template["timing"]),
        }
        if index % 250 == 0 or index == len(selected_ids):
            print(f"full 16-beat profile {index}/{len(selected_ids)}", file=sys.stderr, flush=True)
    if {str(row["record_id"]) for row in runtime_records} != set(selected_ids):
        raise FullDatasetError("runtime profile record coverage differs from frozen selection")
    _write_jsonl(profile_root / "manifest.jsonl", runtime_records)
    _write_jsonl(profile_root / "segments.jsonl", segments)
    source_vocabulary = spec.source_profile / "vocabulary.json"
    _copy_verified(source_vocabulary, profile_root / "vocabulary.json", _sha(source_vocabulary))
    profile_manifest_sha = _sha(profile_root / "manifest.jsonl")
    segments_sha = _sha(profile_root / "segments.jsonl")
    source_summary = _read_json(spec.source_profile / "summary.json")
    summary = {
        "schema": source_summary["schema"],
        "row_codec": source_summary["row_codec"],
        "profile": "context16-full-train",
        "window": WINDOW,
        "source_profile": source_summary.get("profile"),
        "source_profile_fingerprint": source_summary.get("profile_fingerprint"),
        "profile_fingerprint": identity({
            "source_profile_fingerprint": source_summary.get("profile_fingerprint"),
            "selection_sha256": spec.selection_sha256,
            "window": WINDOW,
            "profile_manifest_sha256": profile_manifest_sha,
            "segments_sha256": segments_sha,
            "split_remap": spec.selection.get("source_split_remap"),
        }),
        "profile_manifest_sha256": profile_manifest_sha,
        "segments_sha256": segments_sha,
        "chart_ticks_per_quarter_beat": 96,
        "chart_ticks_per_whole": 384,
        "target_beats": WINDOW["target_beats"],
        "prefix_beats": WINDOW["history_beats"],
        "audio_before_beats": WINDOW["audio_before_beats"],
        "audio_after_beats": WINDOW["audio_after_beats"],
        "eligible_charts": len(runtime_records),
        "input_charts": len(runtime_records),
        "isolated_charts": 0,
        "training_rows": sum(value["rows"] for value in coverage.values()),
        "segment_count": len(segments),
        "eligible_split_counts": dict(Counter(str(row["source_split"]) for row in runtime_records)),
        "segment_split_counts": {"train": len(segments)},
        "eligible_source_difficulty_counts": dict(Counter(str(row["source_difficulty_id"]) for row in runtime_records)),
        "segment_source_difficulty_counts": {"train": dict(Counter(str(row["source_difficulty_id"]) for row in runtime_records))},
        "target_row_split_counts": {"train": sum(value["target_row_count"] for value in coverage.values())},
        "selection_sha256": spec.selection_sha256,
        "source_split_remap": spec.selection.get("source_split_remap"),
    }
    _write_json(profile_root / "summary.json", summary)
    profile = {
        "schema": summary["schema"],
        "row_codec": summary["row_codec"],
        "profile": summary["profile"],
        "window": WINDOW,
        "base_feature_stats": "../audio/feature-stats.json",
        "base_manifest": "manifest.jsonl",
        "base_dataset": ".",
        "summary_sha256": _sha(profile_root / "summary.json"),
        "vocabulary_sha256": _sha(profile_root / "vocabulary.json"),
        "source_profile_fingerprint": source_summary.get("profile_fingerprint"),
        "selection_sha256": spec.selection_sha256,
    }
    _write_json(profile_root / "profile.json", profile)
    _write_json(spec.runtime / "selection.json", spec.selection)
    _write_json(spec.runtime / "provenance" / "canonical-coverage.json", coverage)
    return {
        "status": "built",
        "profile": str(profile_root),
        "profile_fingerprint": summary["profile_fingerprint"],
        "profile_manifest_sha256": profile_manifest_sha,
        "segments_sha256": segments_sha,
        "chart_count": len(runtime_records),
        "segment_count": len(segments),
        "bytes": _path_size(profile_root),
    }


def build_chart_cache(spec: FullDatasetSpec) -> dict[str, Any]:
    profile_root = spec.runtime / "profile"
    cache_root = spec.runtime / "chart-cache"
    summary = _read_json(profile_root / "summary.json")
    if cache_root.exists():
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FullDatasetError(f"incomplete chart cache exists; quarantine before retry: {cache_root}")
        manifest = _read_json(manifest_path)
        if manifest.get("profile_fingerprint") != summary["profile_fingerprint"]:
            raise FullDatasetError("existing full chart cache has a different profile fingerprint")
    else:
        build_chart_vector_cache(profile_root=profile_root, suite_path=spec.runtime / "selection.json", output=cache_root)
    manifest = _read_json(cache_root / "manifest.json")
    if manifest.get("selection", {}).get("validation_enabled") is not False:
        raise FullDatasetError("full chart cache did not publish validation_enabled=false")
    if manifest.get("selection", {}).get("validation_song_ids") != []:
        raise FullDatasetError("full chart cache published non-empty validation selection")
    return {
        "status": "built" if cache_root.exists() else "missing",
        "path": str(cache_root),
        "manifest_sha256": _sha(cache_root / "manifest.json"),
        "segment_count": int(manifest["segment_count"]),
        "song_counts": manifest.get("song_counts"),
        "bytes": _path_size(cache_root),
    }


def _full_audio_material(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    profile = spec.runtime / "profile"
    chart = _read_json(spec.runtime / "chart-cache" / "manifest.json")
    stats_path = spec.base_dataset / "feature-stats.json"
    return {
        "schema": "maimai.full_bpm_audio_preparation.v1",
        "profile_manifest_sha256": _sha(profile / "manifest.jsonl"),
        "segments_sha256": _sha(profile / "segments.jsonl"),
        "chart_cache_manifest_sha256": _sha(spec.runtime / "chart-cache" / "manifest.json"),
        "selection_sha256": str(chart["selection"]["selection_sha256"]),
        "stats_sha256": _sha(stats_path),
        "audio_manifest_sha256": str(audit["audio"]["manifest_sha256"]),
        "base_policy": "normalized-p50",
        "sampling": SAMPLING,
        "integration": INTEGRATION,
        "source_profile_fingerprint": _read_json(profile / "summary.json")["profile_fingerprint"],
    }


def _variant_inputs(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records = {str(row["record_id"]): row for row in _read_jsonl(spec.runtime / "profile" / "manifest.jsonl")}
    segments = _read_jsonl(spec.runtime / "profile" / "segments.jsonl")
    audio_sources = {str(song): value for song, value in audit["audio"]["selected_audio_sources"].items()}
    variants: dict[str, dict[str, Any]] = {}
    variant_by_record: dict[str, str] = {}
    refs: list[dict[str, Any]] = []
    for segment in segments:
        rid = str(segment["record_id"])
        record = records[rid]
        key = variant_by_record.get(rid)
        if key is None:
            audio = audio_sources[str(segment["song_id"])]
            # Every segment for a record is produced from the same timing map.
            # Cache this conversion: rebuilding it for all 93k windows made
            # the full-data preparation needlessly memory- and CPU-heavy.
            timing = TimingMap.from_dict(segment["timing"])
            original_sha = str(record["audio"]["file_sha256"])
            source_sha = str(audio["output_sha256"])
            variant_identity = {
                "original_sha256": original_sha,
                "timing": timing_identity(timing),
                "source_sha256": source_sha,
                "base_policy": "normalized-p50",
                "stats_sha256": _sha(spec.base_dataset / "feature-stats.json"),
                "sampling": SAMPLING,
            }
            key = identity(variant_identity)
            variant_by_record[rid] = key
            if key not in variants:
                expected_samples = max(1, round(int(audio["samples_per_channel"]) * 24000 / int(audio["source_sample_rate"])))
                variants[key] = {
                    "identity": variant_identity,
                    "source_path": str(contained(spec.normalized_audio, audio["output_relative_path"])),
                    "source_sha256": source_sha,
                    "original_path": str((spec.workspace_root / str(audio["relative_source_path"])).resolve()),
                    "timing": timing.to_dict(),
                    "song_ids": [],
                    "old_frames": int(record["audio"]["feature_frames"]),
                    "num_samples": expected_samples,
                    "audio": {
                        "source_p50_lufs": audio.get("source_p50_lufs"),
                        "normalization_gain_db": audio.get("gain_db"),
                        "target_p50_lufs": audio.get("target_p50_lufs"),
                    },
                }
        if str(segment["song_id"]) not in variants[key]["song_ids"]:
            variants[key]["song_ids"].append(str(segment["song_id"]))
        refs.append({"segment": segment, "variant": key})
    return variants, refs


def _decode_existing_normalized_waveform(path: Path) -> tuple[Any, dict[str, Any]]:
    """Decode an existing P50 WAV without requiring the optional TorchCodec DLL.

    The full-data contract consumes already-normalized WAV assets.  SoundFile
    reads those PCM files directly, while the resampling operation is the same
    stable float32 half-pixel operation used by the canonical audio module.
    This is an audio-preparation fallback only; it is not a Physical/geometry
    CPU fallback.
    """
    import soundfile as sf
    import torch
    from maimai_dataset.audio import resample_linear_float32

    samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] not in (1, 2):
        raise FullDatasetError(f"normalized audio must decode to mono/stereo PCM: {path}")
    if not np.isfinite(samples).all():
        raise FullDatasetError(f"normalized audio contains non-finite PCM: {path}")
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T)).mean(dim=0, keepdim=True)
    source_rate = int(source_rate)
    if source_rate <= 0:
        raise FullDatasetError(f"normalized audio has no valid sample rate: {path}")
    if source_rate != 24000:
        output_samples = max(1, round(waveform.shape[1] * 24000 / source_rate))
        waveform = resample_linear_float32(waveform, output_samples)
    source_hash = _sha(path)
    return waveform.contiguous(), {
        "source_sha256": source_hash,
        "source_p50_lufs": None,
        "normalization_gain_db": 0.0,
        "source_sample_rate": source_rate,
        "num_samples": int(waveform.shape[1]),
        "sample_rate": 24000,
        "duration_seconds": float(waveform.shape[1] / 24000),
    }


def build_audio(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Build complete normalized-p50 BPM features and 16-beat geometry."""
    stats_path = spec.base_dataset / "feature-stats.json"
    stats = _read_json(stats_path)
    validate_stats(stats)
    material = _full_audio_material(spec, audit)
    preparation_identity = identity(material)
    output = spec.runtime / "audio"
    geometry_root = spec.runtime / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)
    variants, refs = _variant_inputs(spec, audit)
    audit_path = output / "audit.json"
    if audit_path.exists():
        audio_audit = _read_json(audit_path)
        if audio_audit.get("inputs") != material or audio_audit.get("preparation_identity") != preparation_identity:
            raise FullDatasetError("existing BPM audit has a different preparation identity")
        variants = {str(key): dict(value) for key, value in audio_audit["variants"].items()}
    else:
        audio_audit = {
            "schema": "maimai.bpm96_audit.v1",
            "complete": True,
            "passed": True,
            "sampling": SAMPLING,
            "integration": INTEGRATION,
            "preparation_identity": preparation_identity,
            "inputs": material,
            "variants": variants,
            "source_audio_policy": "normalized-p50-WAV-existing-and-hash-verified",
            "feature_generation": "canonical-extract-at-absolute-BPM96-centers-v1",
            "conflicts": [],
        }
        publish(audit_path, audio_audit, material)
    stats_receipt = publish(output / "feature-stats.json", stats_path.read_bytes(), material)
    geometry: dict[str, dict[str, Any]] = {}
    refs_by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_variant[ref["variant"]].append(ref["segment"])
    for index, (key, variant) in enumerate(variants.items(), 1):
        t = TimingMap.from_dict(variant["timing"])
        fpath = output / f"features/{key}.npy"
        cpath = output / f"coordinates/{key}.npz"
        artifact_identity = {"preparation": preparation_identity, "variant": variant["identity"]}
        feature_receipt = completed(fpath, artifact_identity)
        coord_receipt = completed(cpath, artifact_identity)
        if feature_receipt is None or coord_receipt is None:
            wave, decoded = _decode_existing_normalized_waveform(Path(variant["source_path"]))
            if decoded["source_sha256"] != variant["source_sha256"]:
                raise FullDatasetError(f"normalized audio changed after audit: {key}")
            if wave.shape[1] != int(variant["num_samples"]):
                raise FullDatasetError(
                    f"normalized audio sample count differs for {key}: {wave.shape[1]} != {variant['num_samples']}"
                )
            coords = audio_grid(t, int(variant["num_samples"]))
            features = extract_at_centers(wave, coords, stats)
            publish(fpath, features, artifact_identity)
            publish(cpath, coords, artifact_identity)
        else:
            features = np.load(fpath, mmap_mode="r", allow_pickle=False)
            with np.load(cpath, allow_pickle=False) as archive:
                coords = {name: archive[name] for name in archive.files}
        if features.dtype != np.float32 or features.shape != (128, len(coords["grid_k"])) or not np.isfinite(features).all():
            raise FullDatasetError(f"invalid full BPM feature payload: {key}")
        variant["feature_path"] = f"features/{key}.npy"
        variant["coordinates_path"] = f"coordinates/{key}.npz"
        variant["feature_sha256"] = _sha(fpath)
        variant["coordinates_sha256"] = _sha(cpath)
        variant["dtype"] = "float32"
        variant["shape"] = list(features.shape)
        variant["new_frames"] = int(features.shape[1])
        windows: dict[int, dict[str, Any]] = {}
        for segment in refs_by_variant[key]:
            start = int(segment["target_start_tick"])
            if start not in windows:
                geometry_path = geometry_root / f"{key}-{start}.npz"
                geometry_identity = {"preparation": preparation_identity, "variant": variant["identity"], "start_tick": start}
                receipt = completed(geometry_path, geometry_identity)
                if receipt is None:
                    arrays = window_arrays(features, coords, t, start, int(variant["num_samples"]))[4]
                    receipt = publish(geometry_path, arrays, geometry_identity)
                windows[start] = {"path": f"{key}-{start}.npz", "sha256": receipt["sha256"]}
            geometry[str(segment["segment_id"])] = windows[start]
        print(f"full BPM features {index}/{len(variants)}", file=sys.stderr, flush=True)
    geometry_manifest = {
        "schema": GEOMETRY_SCHEMA,
        "window": WINDOW,
        "profile_fingerprint": _read_json(spec.runtime / "profile" / "summary.json")["profile_fingerprint"],
        "suite_sha256": material["selection_sha256"],
        "segment_count": len(geometry),
        "segments": geometry,
    }
    geometry_receipt = publish(geometry_root / "manifest.json", geometry_manifest, material)
    manifest = {
        "schema": BPM_SCHEMA,
        "window": WINDOW,
        "complete": True,
        "sampling": SAMPLING,
        "integration": INTEGRATION,
        "base_policy": "normalized-p50",
        "stats_path": "feature-stats.json",
        "stats_sha256": material["stats_sha256"],
        "selection_sha256": material["selection_sha256"],
        "song_ids": sorted({str(value) for value in spec.selected_song_ids}),
        "profile_manifest_sha256": material["profile_manifest_sha256"],
        "segments_sha256": material["segments_sha256"],
        "preparation_identity": preparation_identity,
        "normalization_manifest_sha256": material["audio_manifest_sha256"],
        "geometry_manifest_sha256": geometry_receipt["sha256"],
        "geometry_root": "../geometry",
        "variants": variants,
        "segments": {str(ref["segment"]["segment_id"]): ref["variant"] for ref in refs},
        "audit_sha256": _sha(audit_path),
        "feature_generation": "from-existing-normalized-p50-WAV;stats-from-current-full-source",
    }
    manifest_receipt = publish(output / "bpm-audio-manifest.json", manifest, material)
    return {
        "status": "built",
        "path": str(output),
        "manifest_sha256": manifest_receipt["sha256"],
        "audit_sha256": _sha(audit_path),
        "geometry_manifest_sha256": geometry_receipt["sha256"],
        "variant_count": len(variants),
        "segment_count": len(refs),
        "bytes": _path_size(output) + _path_size(geometry_root),
        "preparation_identity": preparation_identity,
    }


def build_epoch_plans(spec: FullDatasetSpec) -> dict[str, Any]:
    """Build only the explicitly requested repacked epoch-plan budgets.

    Plan references are meaningful only after the final spatial cache exists,
    so this function deliberately refuses to create a plan with a fabricated
    spatial identity.  The existing curriculum/repack implementation remains
    the canonical deterministic planner.
    """
    spatial_manifest = spec.runtime / "spatial-views" / "manifest.json"
    if not spatial_manifest.is_file():
        raise FullDatasetError("full spatial cache is required before building epoch plans")
    output = spec.runtime / "epoch-plans"
    if output.exists():
        raise FullDatasetError(f"epoch-plan output already exists; quarantine before retry: {output}")
    from maimai_baseline.data.repack import build_plans
    from maimai_baseline.data.training_build import CurriculumSource, plan_inputs, plan_metrics

    source = CurriculumSource(
        spec.runtime,
        epochs=int(spec.config.get("epoch_plan_epochs", 75)),
        full_selection=True,
    )
    metrics = plan_metrics(spec.runtime, source)
    reports = build_plans(
        source,
        output,
        **metrics,
        input_identity=plan_inputs(spec.runtime),
        budgets=spec.epoch_plan_budgets,
    )
    return {
        "status": "built",
        "budgets": list(spec.epoch_plan_budgets),
        "epochs": source.available_epochs,
        "reports": reports,
        "bytes": _path_size(output),
    }


def verify_epoch_plans(spec: FullDatasetSpec) -> dict[str, Any]:
    from maimai_baseline.data.curriculum import EpochPlanStore
    from maimai_baseline.data.repack import verify_plan
    from maimai_baseline.data.training_build import CurriculumSource, plan_inputs, plan_metrics
    source = CurriculumSource(spec.runtime, epochs=int(spec.config.get("epoch_plan_epochs", 75)), full_selection=True)
    expected = plan_inputs(spec.runtime)
    metrics = plan_metrics(spec.runtime, source)
    reports = []
    for budget in spec.epoch_plan_budgets:
        store = EpochPlanStore(spec.runtime / "epoch-plans" / str(budget), verify_payloads=True)
        if store.manifest.get("repack_inputs") != expected or store.manifest.get("token_budget") != budget:
            raise FullDatasetError("full epoch plan input/budget identity mismatch")
        reports.append(verify_plan(source, store, **metrics))
    return {"passed": True, "budgets": list(spec.epoch_plan_budgets), "reports": reports}


def _inventory_candidates(spec: FullDatasetSpec) -> list[dict[str, Any]]:
    roots = [
        spec.workspace_root / "artifacts" / "route71-spatial-augmentation-v1",
        spec.runtime,
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(root.rglob("manifest.json"))
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            value = _read_json(path)
        except Exception:
            continue
        if value.get("schema") not in {"maimai.physical_state.multiview.v2", "maimai.physical_state.multiview.v1", "maimai.multiview_route71_cache.v1", "maimai.physical_bitplane_cache.v2"}:
            continue
        candidates.append({
            "path": str(path),
            "sha256": _sha(path),
            "schema": value.get("schema"),
            "profile_fingerprint": value.get("profile_fingerprint"),
            "selection_sha256": value.get("selection_sha256"),
            "chart_count": value.get("chart_count"),
            "segment_count": value.get("segment_count"),
            "runtime_consumable": value.get("runtime_consumable"),
            "full_profile_build_permitted": value.get("full_profile_build_permitted"),
            "integration_status": value.get("integration_status"),
        })
    return sorted(candidates, key=lambda row: row["path"])


def _spatial_physical_status(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    ready_paths = ("spatial-views/manifest.json", "physical/source-rows/manifest.json", "physical/bitplanes/manifest.json")
    if all((spec.runtime / name).is_file() for name in ready_paths):
        from maimai_baseline.data.full_physical import verify_full_physical
        try:
            verified = verify_full_physical(spec.runtime, full_hash=True)
            return {**verified, "full_identity_candidates": [{"path": str(spec.runtime), "verified": True}],
                    "reason": "full canonical inputs verified through current readers"}
        except Exception as error:
            return {"status": "blocked", "passed": False, "full_identity_candidates": [], "reason": str(error)}
    expected = len(spec.selected_record_ids)
    candidates = _inventory_candidates(spec)
    full_candidates = [
        row for row in candidates
        if row.get("chart_count") == expected and row.get("profile_fingerprint") == _read_json(spec.runtime / "profile" / "summary.json").get("profile_fingerprint")
    ]
    return {
        "status": "blocked" if not full_candidates else "requires_explicit_rebind_review",
        "expected_chart_count": expected,
        "expected_profile_fingerprint": _read_json(spec.runtime / "profile" / "summary.json").get("profile_fingerprint"),
        "candidates": candidates,
        "full_identity_candidates": full_candidates,
        "reason": "no full-profile canonical v2 four-view/source-row/bitplane artifact is available locally" if not full_candidates else "matching artifact exists but this preparation tool intentionally does not auto-rebind Physical data",
        "remote_or_cuda_actions": "not-run",
    }


def _write_dataset_manifest(spec: FullDatasetSpec, *, build_report: Mapping[str, Any] | None = None, verify_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    runtime = spec.runtime
    if build_report is None and (spec.reports / "build.json").is_file():
        build_report = _read_json(spec.reports / "build.json")
    inputs = {
        "dataset": "profile",
        "chart_cache": "chart-cache",
        "interpolation_geometry": "geometry",
        "bpm_audio_manifest": "audio/bpm-audio-manifest.json",
        "spatial_view_cache": "spatial-views",
        "physical_path_catalog": "physical/path-catalog",
        "physical_source_rows": "physical/source-rows",
        "physical_bitplane_cache": "physical/bitplanes",
        "route_artifact": "route-artifact",
    }
    value: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "window": WINDOW,
        "ready": bool(verify_report and verify_report.get("passed") is True),
        "migration_finalized": False,
        "selection": "selection.json",
        "selection_sha256": spec.selection_sha256,
        "validation": {"enabled": False, "policy": "disabled", "song_count": 0},
        "inputs": inputs,
        "epoch_plans": {str(budget): f"epoch-plans/{budget}" for budget in spec.epoch_plan_budgets},
        "epoch_plan_budgets": list(spec.epoch_plan_budgets),
        "build": "../reports/build.json",
        "verification": "../reports/verify.json",
        "scope": "all eligible source difficulties; train-only; no validation loss",
        "build_status": None if build_report is None else build_report.get("status"),
        "verification_status": None if verify_report is None else verify_report.get("status"),
    }
    if build_report is not None:
        value["build_sha256"] = _sha(spec.reports / "build.json") if (spec.reports / "build.json").is_file() else None
    if verify_report is not None and (spec.reports / "verify.json").is_file():
        value["verification_sha256"] = _sha(spec.reports / "verify.json")
    dataset_path = runtime / "dataset.json"
    if dataset_path.is_file():
        existing = _read_json(dataset_path)
        if existing.get("selection_sha256") != spec.selection_sha256:
            raise FullDatasetError("existing runtime dataset manifest belongs to a different selection")
    # This manifest is the derived stage pointer: build and verify runs must
    # update its status and report hashes atomically after preserving the
    # selection identity check above.
    _write_json(dataset_path, value, replace=True)
    return value


def build(spec: FullDatasetSpec, *, audio: bool = True) -> dict[str, Any]:
    audit = _load_audit(spec)
    stages: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    if not audit.get("source_integrity_passed"):
        raise FullDatasetError("source audit has integrity failures; refusing to build from changed source inputs")
    try:
        stages["profile"] = build_profile(spec, audit)
    except Exception as error:
        stages["profile"] = {"status": "failed", "error": str(error)}
        blockers.append(f"profile: {error}")
    if stages.get("profile", {}).get("status") == "built":
        try:
            stages["chart-cache"] = build_chart_cache(spec)
        except Exception as error:
            stages["chart-cache"] = {"status": "failed", "error": str(error)}
            blockers.append(f"chart-cache: {error}")
    else:
        stages["chart-cache"] = {"status": "blocked", "reason": "profile did not build"}
    if audio and stages.get("chart-cache", {}).get("status") == "built":
        try:
            stages["audio"] = build_audio(spec, audit)
        except Exception as error:
            stages["audio"] = {"status": "failed", "error": str(error)}
            blockers.append(f"audio: {error}")
    else:
        stages["audio"] = {"status": "not-run", "reason": "audio generation disabled or chart-cache unavailable"}
    spatial = _spatial_physical_status(spec, audit) if (spec.runtime / "profile" / "summary.json").is_file() else {"status": "blocked", "reason": "profile unavailable"}
    physical_status = "verified" if spatial.get("passed") else "blocked"
    stages["spatial-views"] = {"status": physical_status, "reason": spatial["reason"], "inventory": spatial}
    stages["physical"] = {"status": physical_status, "reason": spatial["reason"], "inventory": spatial}
    if not spatial.get("passed"):
        blockers.append("full spatial views and Physical bitplanes are not available for this profile")
    if (spec.runtime / "spatial-views" / "manifest.json").is_file():
        try:
            stages["plans"] = build_epoch_plans(spec)
        except Exception as error:
            stages["plans"] = {"status": "failed", "budgets": list(spec.epoch_plan_budgets), "error": str(error)}
            blockers.append(f"plans: {error}")
    else:
        stages["plans"] = {
            "status": "blocked",
            "budgets": list(spec.epoch_plan_budgets),
            "reason": "plans must bind the final spatial manifest; no full spatial cache is available",
        }
        blockers.append(f"epoch plans ({','.join(map(str, spec.epoch_plan_budgets))}) are not published until final spatial identity is available")
    if not audit["historical_comparison"].get("accepted", audit["historical_comparison"].get("resolved", False)):
        blockers.append("historical 4142 to current 4147 record-level difference remains unresolved")
    status = "partial_blocked" if blockers else "built"
    report = {
        "schema": BUILD_SCHEMA,
        "status": status,
        "ready": False,
        "audit_identity": audit["audit_identity"],
        "selection_sha256": spec.selection_sha256,
        "source_identity": baseline_source_identity(),
        "environment": _runtime_environment(),
        "stages": stages,
        "physical_inventory": spatial,
        "blockers": blockers,
        "training": {"authorized": False, "scheduler_epochs": None, "stop_after_epochs": None},
    }
    _write_json(spec.reports / "build.json", report, replace=True)
    _write_dataset_manifest(spec, build_report=report)
    return report


def _verify_profile(spec: FullDatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    profile = spec.runtime / "profile"
    records = _read_jsonl(profile / "manifest.jsonl")
    segments = _read_jsonl(profile / "segments.jsonl")
    expected_ids = spec.selected_record_ids
    actual_ids = [str(row["record_id"]) for row in records]
    failures: list[dict[str, Any]] = []
    policy_path = spec.root / "pilot-adoption-policy.json"
    receipt_path = spec.runtime / "provenance/pilot-adoption.json"
    adoption = _read_json(receipt_path) if receipt_path.is_file() else None
    if adoption is not None:
        policy = _read_json(policy_path)
        if adoption.get("policy_sha256") != _sha(policy_path) or set(adoption["proof"]) != set(policy["records"]):
            raise FullDatasetError("pilot adoption policy/receipt mismatch")
        for rid, hashes in policy["records"].items():
            if any(adoption["proof"][rid].get(k) != v for k, v in hashes.items()):
                raise FullDatasetError(f"pilot adoption proof mismatch: {rid}")
            if hashes["full_encoded_sha256"] != audit["record_hashes"][rid]["encoded_file_sha256"]:
                raise FullDatasetError(f"pilot adoption original source mismatch: {rid}")
    if actual_ids != expected_ids:
        failures.append({"kind": "profile-record-order-or-coverage", "expected_count": len(expected_ids), "actual_count": len(actual_ids)})
    if any(row.get("split") != "train" or row.get("record_id") in {"11010:6"} or str(row.get("song_id")) == "854" for row in records):
        failures.append({"kind": "profile-forbidden-or-nontrain-record"})
    by_record = defaultdict(list)
    for segment in segments:
        by_record[str(segment["record_id"])].append(segment)
        if segment.get("split") != "train" or str(segment.get("song_id")) == "854" or str(segment.get("record_id")) == "11010:6":
            failures.append({"kind": "segment-forbidden-or-nontrain", "segment_id": segment.get("segment_id")})
    for record in records:
        rid = str(record["record_id"])
        source = contained(profile, record["encoded_path"])
        document = _read_json(source)
        expected_hash = (adoption["proof"][rid]["pilot_encoded_sha256"] if adoption and rid in adoption["proof"]
                         else audit["record_hashes"][rid]["encoded_file_sha256"])
        if _sha(source) != expected_hash:
            failures.append({"record_id": rid, "kind": "encoded-source-hash-mismatch"})
        rows_count = len(document["rows"])
        ranges = sorted((int(s["target_row_start"]), int(s["target_row_end"])) for s in by_record[rid])
        cursor = 0
        for start, end in ranges:
            if start != cursor or end < start:
                failures.append({"record_id": rid, "kind": "target-row-gap-or-overlap", "range": [start, end], "cursor": cursor})
                break
            cursor = end
        if cursor != rows_count:
            failures.append({"record_id": rid, "kind": "target-row-incomplete", "covered": cursor, "rows": rows_count})
    summary = _read_json(profile / "summary.json")
    return {"passed": not failures, "record_count": len(records), "segment_count": len(segments), "profile_fingerprint": summary.get("profile_fingerprint"), "failures": failures}


def verify(spec: FullDatasetSpec) -> dict[str, Any]:
    audit = _load_audit(spec)
    failures: list[dict[str, Any]] = []
    profile_result: dict[str, Any] = {"passed": False, "status": "missing"}
    chart_result: dict[str, Any] = {"passed": False, "status": "missing"}
    audio_result: dict[str, Any] = {"passed": False, "status": "missing"}
    if (spec.runtime / "profile" / "manifest.jsonl").is_file():
        try:
            profile_result = _verify_profile(spec, audit)
            if not profile_result["passed"]:
                failures.extend(profile_result["failures"])
        except Exception as error:
            profile_result = {"passed": False, "status": "failed", "error": str(error)}
            failures.append({"kind": "profile-verification", "error": str(error)})
    else:
        failures.append({"kind": "profile-missing"})
    chart_path = spec.runtime / "chart-cache" / "manifest.json"
    if chart_path.is_file() and profile_result.get("passed"):
        try:
            from maimai_dataset.chart_cache import ChartVectorStore
            summary = _read_json(spec.runtime / "profile" / "summary.json")
            vocab_sha = _sha(spec.runtime / "profile" / "vocabulary.json")
            store = ChartVectorStore.load(spec.runtime / "chart-cache", profile_fingerprint=summary["profile_fingerprint"], vocabulary_sha256=vocab_sha, verify_payloads=True)
            chart_result = {"passed": True, "segment_count": len(store.segment_indices), "validation_enabled": store.manifest["selection"].get("validation_enabled") is True, "selection": store.manifest["selection"]}
            if chart_result["validation_enabled"] or store.song_ids("validation"):
                raise FullDatasetError("full chart cache unexpectedly contains validation")
        except Exception as error:
            chart_result = {"passed": False, "status": "failed", "error": str(error)}
            failures.append({"kind": "chart-cache-verification", "error": str(error)})
    else:
        failures.append({"kind": "chart-cache-missing-or-profile-failed"})
    runtime_files: list[dict[str, Any]] = []
    runtime_hashes: dict[str, str] = {}
    if spec.runtime.exists():
        for path in sorted(p for p in spec.runtime.rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
            relative = path.relative_to(spec.runtime).as_posix()
            # dataset.json is a mutable stage pointer whose contents include
            # the hashes of the reports written by this verify call.  Keep it
            # out of the immutable runtime artifact inventory to avoid a
            # self-referential/stale hash in verify.json.
            if relative == "dataset.json":
                continue
            digest = _sha(path)
            runtime_hashes[relative] = digest
            runtime_files.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    audio_manifest = spec.runtime / "audio" / "bpm-audio-manifest.json"
    if audio_manifest.is_file() and chart_result.get("passed"):
        try:
            from maimai_baseline.data.store import BpmFeatureStore, _validate_geometry_structure, window_arrays
            stats = _read_json(spec.runtime / "audio" / "feature-stats.json")
            chart_segments = _read_jsonl(spec.runtime / "chart-cache" / "segments.jsonl")
            profile_segments = _read_jsonl(spec.runtime / "profile" / "segments.jsonl")
            store = BpmFeatureStore(
                audio_manifest,
                profile=spec.runtime / "profile",
                stats=stats,
                song_ids=set(spec.selected_song_ids),
                selection_sha256=_read_json(spec.runtime / "chart-cache" / "manifest.json")["selection"]["selection_sha256"],
                geometry_root=spec.runtime / "geometry",
                expected_segment_ids={str(segment["segment_id"]) for segment in chart_segments},
                verify_payloads=False,
            )
            artifact_failures: list[dict[str, Any]] = []
            for key, variant in store.variants.items():
                for field, expected in (("feature_path", variant["feature_sha256"]), ("coordinates_path", variant["coordinates_sha256"])):
                    path = contained(store.root, variant[field])
                    relative = path.relative_to(spec.runtime).as_posix()
                    actual = runtime_hashes.get(relative)
                    if actual != expected:
                        artifact_failures.append({"kind": "audio-artifact-hash", "variant": key, "path": relative, "expected": expected, "actual": actual})
            starts_by_segment = {str(segment["segment_id"]): int(segment["target_start_tick"]) for segment in profile_segments}
            geometry_failures = 0
            for segment_id, row in store.geometry["segments"].items():
                path = contained(store.geometry_root, row["path"])
                relative = path.relative_to(spec.runtime).as_posix()
                actual = runtime_hashes.get(relative)
                if actual != row["sha256"]:
                    artifact_failures.append({"kind": "audio-geometry-hash", "segment_id": segment_id, "path": relative, "expected": row["sha256"], "actual": actual})
                    continue
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        _validate_geometry_structure({name: archive[name] for name in archive.files}, starts_by_segment[segment_id])
                except Exception as error:
                    geometry_failures += 1
                    artifact_failures.append({"kind": "audio-geometry-structure", "segment_id": segment_id, "path": relative, "error": str(error)})
            # Recompute only the first and last window of each variant.  The
            # full payloads and all geometry files are already hash-checked;
            # these deterministic samples add semantic coverage without
            # repeating 93k * 528 BPM lookups during every verify.
            samples_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for segment in profile_segments:
                key = store.segments[str(segment["segment_id"])]
                if len(samples_by_variant[key]) < 2:
                    samples_by_variant[key].append(segment)
                else:
                    samples_by_variant[key][1] = segment
            sampled = 0
            for key, samples in samples_by_variant.items():
                variant = store.variants[key]
                for segment in samples:
                    arrays = window_arrays(
                        store._features[key], store._coordinates[key], TimingMap.from_dict(segment["timing"]),
                        int(segment["target_start_tick"]), int(variant["num_samples"]), include_geometry=True,
                    )
                    row = store.geometry["segments"][str(segment["segment_id"])]
                    path = contained(store.geometry_root, row["path"])
                    with np.load(path, allow_pickle=False) as archive:
                        actual_geometry = {name: archive[name] for name in archive.files}
                    if set(actual_geometry) != set(arrays[4]) or any(not np.array_equal(actual_geometry[name], value) for name, value in arrays[4].items()):
                        artifact_failures.append({"kind": "audio-geometry-semantic-sample", "segment_id": segment["segment_id"]})
                    sampled += 1
            if artifact_failures:
                raise FullDatasetError(f"{len(artifact_failures)} BPM artifact verification checks failed")
            audio_result = {
                "passed": True,
                "variant_count": len(store.variants),
                "segment_count": len(store.segments),
                "policy": store.policy,
                "payload_hashes_checked": len(runtime_hashes),
                "geometry_files_checked": len(store.geometry["segments"]),
                "geometry_structure_failures": geometry_failures,
                "semantic_sampled_segment_count": sampled,
            }
        except Exception as error:
            audio_result = {"passed": False, "status": "failed", "error": str(error)}
            failures.append({"kind": "audio-verification", "error": str(error)})
    else:
        failures.append({"kind": "audio-missing-or-chart-failed"})
    spatial_status = _spatial_physical_status(spec, audit) if (spec.runtime / "profile" / "summary.json").is_file() else {"status": "blocked"}
    if spatial_status.get("passed") is not True:
        failures.append({"kind": "full-spatial-physical-missing", "inventory": spatial_status})
    else:
        try:
            from maimai_baseline.model.route import load_route_artifacts
            load_route_artifacts(spec.runtime / "route-artifact", verify_payloads=True)
        except Exception as error:
            failures.append({"kind": "route-artifact-verification", "error": str(error)})
    plan_budgets = spec.epoch_plan_budgets
    plans_present = all((spec.runtime / "epoch-plans" / str(budget) / "manifest.json").is_file() for budget in plan_budgets)
    plan_status = {"passed": False, "required_budgets": list(plan_budgets)}
    if not plans_present:
        failures.append({"kind": "epoch-plans-missing"})
    else:
        try:
            plan_status = verify_epoch_plans(spec)
        except Exception as error:
            failures.append({"kind": "epoch-plans-verification", "error": str(error)})
            plan_status["error"] = str(error)
    if not audit.get("passed"):
        failures.append({"kind": "source-audit-not-passed", "failures": audit.get("failures", []), "historical_comparison": audit.get("historical_comparison")})
    result = {
        "schema": VERIFY_SCHEMA,
        "status": "passed" if not failures else "blocked",
        "passed": not failures,
        "ready": not failures,
        "audit_identity": audit["audit_identity"],
        "selection_sha256": spec.selection_sha256,
        "profile": profile_result,
        "chart_cache": chart_result,
        "audio": audio_result,
        "spatial_physical": spatial_status,
        "plans": plan_status,
        "validation": {"enabled": False, "loader_constructed": False, "validation_loss": None, "reason": "explicit train-only contract"},
        "failures": failures,
        "runtime_files": runtime_files,
        "environment": _runtime_environment(),
    }
    _write_json(spec.reports / "verify.json", result, replace=True)
    _write_dataset_manifest(spec, verify_report=result)
    return result


def run(config_path: Path, stage: str, *, hash_payloads: bool = True, audio: bool = True) -> dict[str, Any]:
    spec = FullDatasetSpec.load(config_path)
    if stage == "audit":
        return audit_sources(spec, hash_payloads=hash_payloads)
    if stage == "build":
        return build(spec, audio=audio)
    if stage == "verify":
        return verify(spec)
    raise FullDatasetError(f"unknown full dataset stage: {stage}")


__all__ = [
    "BUILD_SCHEMA",
    "CONFIG_SCHEMA",
    "DATASET_SCHEMA",
    "FullDatasetError",
    "FullDatasetSpec",
    "SELECTION_SCHEMA",
    "VERIFY_SCHEMA",
    "audit_sources",
    "build",
    "build_audio",
    "build_chart_cache",
    "build_epoch_plans",
    "build_profile",
    "run",
    "verify",
]
