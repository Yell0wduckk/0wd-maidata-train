from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from maimai_chart_ir.models import validate_chart_ir


DATASET_SCHEMA = "maimai.dataset_manifest.v1"
CORPUS_RECORD_SCHEMA = "maimai.chart_ir.corpus_record.v1"
CORPUS_SUMMARY_SCHEMA = "maimai.chart_ir.corpus_summary.v1"
SPLIT_SEED = 20260811
_SPECIAL_ID = re.compile(r"\d{6}")


class DatasetError(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise DatasetError(f"{path}:{line_number}: JSONL record must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def split_for_group(group_hash: str, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0{group_hash}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _level(raw: Any) -> tuple[str, float | None]:
    text = str(raw or "").strip()
    try:
        return text, float(text.rstrip("+"))
    except ValueError:
        return text, None


def _audio_dict(info: Any) -> dict[str, Any] | None:
    if info is None:
        return None
    if isinstance(info, Mapping):
        return dict(info)
    if hasattr(info, "to_dict"):
        return dict(info.to_dict())
    raise DatasetError(f"unsupported audio info type: {type(info).__name__}")


def dataset_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    stable = []
    for record in sorted(records, key=lambda item: str(item.get("record_id", item.get("song_id", "")))):
        stable.append({
            key: value
            for key, value in record.items()
            if key not in {"feature_path", "ir_path", "corpus_root"}
        })
    return canonical_json_sha256(stable)


def build_dataset_manifest(
    corpus_rows: Sequence[Mapping[str, Any]],
    *,
    audio_by_song: Mapping[str, Any],
    ir_paths: Mapping[str, str],
    feature_paths: Mapping[str, str],
    timing_paths: Mapping[str, str] | None = None,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    summary_row = next((row for row in corpus_rows if row.get("schema") == CORPUS_SUMMARY_SCHEMA), None)
    source_records = [row for row in corpus_rows if row.get("schema") == CORPUS_RECORD_SCHEMA]
    if summary_row is None:
        raise DatasetError("corpus JSONL is missing maimai.chart_ir.corpus_summary.v1")

    song_splits: dict[str, str] = {}
    pcm_splits: dict[str, str] = {}
    for source in source_records:
        song_id = str(source.get("song_id", ""))
        if not song_id:
            raise DatasetError("corpus record is missing song_id")
        if _SPECIAL_ID.fullmatch(song_id):
            raise DatasetError(f"six-digit special chart leaked into regular corpus: {song_id}")
        audio = _audio_dict(audio_by_song.get(song_id))
        pcm_hash = str(audio.get("pcm_sha256", "")) if audio else ""
        split = split_for_group(pcm_hash, split_seed) if pcm_hash else "none"
        if song_id in song_splits and song_splits[song_id] != split:
            raise DatasetError(f"song {song_id} spans multiple splits")
        song_splits[song_id] = split
        if pcm_hash:
            if pcm_hash in pcm_splits and pcm_splits[pcm_hash] != split:
                raise DatasetError(f"PCM group {pcm_hash} spans multiple splits")
            pcm_splits[pcm_hash] = split

        metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
        cabinet = str(metadata.get("cabinet", "UNKNOWN")).strip().upper() or "UNKNOWN"
        common = {
            "schema": DATASET_SCHEMA,
            "song_id": song_id,
            "source_path": str(source.get("source_path", "")),
            "maidata_sha256": str(source.get("maidata_sha256", "")),
            "audio_path": str(source.get("audio_path", "")),
            "audio": audio,
            "feature_path": feature_paths.get(song_id),
            "title": str(metadata.get("title", "")),
            "artist": str(metadata.get("artist", "")),
            "version": str(metadata.get("version", "")),
            "cabinet": cabinet,
            "split": split,
        }
        if source.get("status") == "no_chart":
            records.append({
                **common,
                "record_id": f"{song_id}:no_chart",
                "status": "no_chart",
                "difficulty_id": None,
                "difficulty": None,
                "isolation_reasons": ["no_chart"],
                "ir_path": None,
                "ir_sha256": None,
            })
            continue

        difficulty_id = int(source["difficulty_id"])
        record_id = f"{song_id}:{difficulty_id}"
        level_text, level_value = _level(source.get("level"))
        document = source.get("document")
        if not isinstance(document, Mapping):
            reasons = sorted(set(
                str(reason) for reason in source.get("isolation_reasons", ["simai:parse_failure"])
            ))
            records.append({
                **common,
                "record_id": record_id,
                "status": "isolated",
                "difficulty_id": difficulty_id,
                "difficulty": str(source.get("difficulty", "")),
                "level": level_text,
                "level_value": level_value,
                "designer": str(source.get("designer", "")),
                "isolation_reasons": reasons,
                "error": str(source.get("error", "")),
                "ir_path": None,
                "ir_sha256": None,
                "timing_path": None,
            })
            continue
        validate_chart_ir(document)
        reasons = list(document["eligibility"]["isolation_reasons"])
        status = str(source.get("status", "isolated"))
        if source.get("simai_roundtrip_ok") is False:
            reasons.append("simai:roundtrip_failure")
        if not audio:
            status = "isolated"
            reasons.append("audio:decode_failure")
        if cabinet not in {"SD", "DX"}:
            status = "isolated"
            reasons.append("metadata:unknown_cabinet")
        reasons = sorted(set(reasons))
        if reasons:
            status = "isolated"
        ir_sha = canonical_json_sha256(document)
        records.append({
            **common,
            "record_id": record_id,
            "status": status,
            "difficulty_id": difficulty_id,
            "difficulty": str(source.get("difficulty", document.get("difficulty", ""))),
            "level": level_text,
            "level_value": level_value,
            "designer": str(source.get("designer", "")),
            "isolation_reasons": reasons,
            "ir_path": ir_paths.get(record_id),
            "ir_sha256": ir_sha,
            "timing_path": (timing_paths or {}).get(record_id),
        })

    chart_records = [record for record in records if record["difficulty_id"] is not None]
    chart_songs = {record["song_id"] for record in chart_records}
    status_counts = Counter(record["status"] for record in records)
    split_counts = Counter(record["split"] for record in chart_records if record["status"] == "eligible")
    difficulty_counts = Counter(str(record["difficulty_id"]) for record in chart_records)
    isolation_reason_counts = Counter(
        reason
        for record in records if record["status"] == "isolated"
        for reason in record.get("isolation_reasons", [])
    )
    cabinet_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"songs": 0, "charts": 0})
    for cabinet in {record["cabinet"] for record in chart_records}:
        selected = [record for record in chart_records if record["cabinet"] == cabinet]
        cabinet_counts[cabinet] = {
            "songs": len({record["song_id"] for record in selected}),
            "charts": len(selected),
        }
    summary = {
        "schema": "maimai.dataset_summary.v1",
        "split_seed": split_seed,
        "source_summary": dict(summary_row),
        "regular_songs": int(summary_row.get("discovered_songs", 0)),
        "chart_bearing_songs": len(chart_songs),
        "candidate_charts": len(chart_records),
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_split_counts": dict(sorted(split_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "cabinet_counts": dict(sorted(cabinet_counts.items())),
        "isolation_reason_counts": dict(sorted(isolation_reason_counts.items())),
        "pcm_groups": len(pcm_splits),
        "six_digit_id_leaks": 0,
        "song_split_conflicts": 0,
        "pcm_split_conflicts": 0,
    }
    summary["dataset_fingerprint"] = dataset_fingerprint(records)
    return records, summary
