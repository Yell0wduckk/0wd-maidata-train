"""Chart materialization and semantic round-trip verification."""
from __future__ import annotations

from maimai_chart_ir.models import ChartIRError, compare_chart_ir_semantics, model_projection
from maimai_dataset.materialize import normalize_slide_duration_ticks
from maimai_dataset.timing import TimingMap
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import os
import shutil
import subprocess
from maimai_baseline.inference.simai import bounded_render_document, compact_spacing, SPACING_POLICY
from maimai_baseline.inference.simai import four_beat_lines
from maimai_baseline.runtime_versions import require_compatible, current_version, policy


ALLOWED_ROUNDTRIP_MISMATCHES = frozenset({"unordered_item_count"})


CHART_IR_TIMEOUT_SECONDS = 30


def resolve_chart_ir_executable(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    candidates = (
        [requested / "miacode_chart_ir.exe", requested / "miacode_chart_ir"]
        if requested.is_dir()
        else [
            requested.with_name("miacode_chart_ir.exe"),
            requested.with_name("miacode_chart_ir"),
        ]
        if requested.name.casefold() in {"miacode.exe", "miacode"}
        else [requested]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"miacode_chart_ir executable does not exist: {requested}")


def _run_chart_ir(
    executable: Path, arguments: list[str], payload: bytes, operation: str
) -> bytes:
    if not executable.is_file():
        raise FileNotFoundError(f"miacode_chart_ir executable does not exist: {executable}")
    try:
        process = subprocess.run(
            [str(executable), *arguments],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=CHART_IR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ChartIRError(f"MiaCode {operation} timed out") from error
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise ChartIRError(f"MiaCode {operation} failed ({process.returncode}): {message}")
    return process.stdout


def render_chart(executable: Path, document: Mapping[str, Any]) -> str:
    document = bounded_render_document(document)
    return four_beat_lines(compact_spacing(_run_chart_ir(
        executable,
        ["render", "--input", "-", "--output", "-"],
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        "render",
    ).decode("utf-8")))


def reparse_chart(executable: Path, chart: str, *, first_seconds: float) -> dict[str, Any]:
    raw = _run_chart_ir(
        executable,
        ["dump", "--file", "-", "--first", format(first_seconds, ".12g"), "--output", "-"],
        chart.encode("utf-8"),
        "canonical simai reparse",
    )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChartIRError(f"MiaCode reparse returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ChartIRError("MiaCode reparse did not return a chart IR object")
    document = normalize_slide_duration_ticks(document)
    model_projection(document)
    return document


def maidata_text(
    chart: str,
    *,
    timing: TimingMap,
    difficulty_id: int,
    level: float,
    cabinet: str,
    title: str,
    artist: str,
    designer: str,
) -> str:
    clean = lambda value: value.replace("\r", " ").replace("\n", " ").strip()
    return "\n".join(
        [
            f"&title={clean(title)}",
            f"&artist={clean(artist)}",
            f"&first={format(timing.first_seconds, '.12g')}",
            f"&wholebpm={format(timing.points[0].bpm, '.12g')}",
            f"&cabinet={cabinet}",
            "&shortid=generated",
            f"&lv_{difficulty_id}={format(level, '.4g')}",
            f"&des_{difficulty_id}={clean(designer)}",
            f"&inote_{difficulty_id}={chart.strip()}",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_roundtrip(comparison: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = comparison.get("semantic_mismatches", ())
    kinds = {str(item.get("kind")) for item in mismatches}
    accepted = bool(not comparison.get("equal", False)) and bool(kinds) and (
        kinds <= ALLOWED_ROUNDTRIP_MISMATCHES
    )
    if not comparison.get("equal", False) and not accepted:
        raise ValueError(
            "MiaCode round-trip changed Route-71 chart semantics: "
            f"kinds={sorted(kinds)}, comparison={dict(comparison)}"
        )
    return {
        "equal": bool(comparison.get("equal", False)),
        "accepted_known_canonicalization": accepted,
        "allowed_mismatch_kinds": sorted(ALLOWED_ROUNDTRIP_MISMATCHES),
        "semantic_mismatch_count": int(comparison.get("semantic_mismatch_count", 0)),
        "semantic_mismatch_kinds": sorted(kinds),
        "audit_difference_count": int(comparison.get("audit_difference_count", 0)),
    }


def _install_audio(source: Path, destination: Path, *, force: bool) -> tuple[str, str]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source audio is missing: {source}")
    source_hash = _sha256(source)
    if destination.exists():
        if _sha256(destination) == source_hash:
            return "existing-identical", source_hash
        if not force:
            raise ValueError(f"existing packaged audio differs: {destination}")
        destination.unlink()
    link_type = "hardlink"
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        link_type = "copy"
    if destination.stat().st_size != source.stat().st_size:
        raise IOError("packaged audio size mismatch")
    if _sha256(destination) != source_hash:
        raise IOError("packaged audio SHA-256 mismatch")
    return link_type, source_hash


def materialize_baseline_inference(
    *,
    inference_dir: Path,
    timing_path: Path,
    audio_path: Path,
    miacode_exe: Path,
    title: str | None,
    artist: str,
    designer: str,
    force: bool = False,
) -> dict[str, Any]:
    inference_dir = Path(inference_dir).resolve()
    generation_path = inference_dir / "generation.json"
    ir_path = inference_dir / "model.ir.json"
    rows_path = inference_dir / "model.rows.json"
    if not generation_path.is_file() or not ir_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(
            f"baseline inference directory is incomplete: {inference_dir}"
        )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    require_compatible(generation, 'materialize')
    expected_audio = (generation.get("audio_preparation") or {}).get("source_sha256")
    if expected_audio is not None and _sha256(Path(audio_path)) != expected_audio:
        raise ValueError("packaged source audio differs from inference input")
    if generation.get("timing_sha256") is not None and _sha256(Path(timing_path)) != generation["timing_sha256"]:
        raise ValueError("materialization timing differs from inference input")
    document = json.loads(ir_path.read_text(encoding="utf-8"))
    timing = TimingMap.from_dict(
        json.loads(Path(timing_path).read_text(encoding="utf-8-sig"))
    )
    executable = resolve_chart_ir_executable(Path(miacode_exe))
    chart = render_chart(executable, document)
    reparsed = reparse_chart(executable, chart, first_seconds=timing.first_seconds)
    comparison = compare_chart_ir_semantics(document, reparsed, max_differences=32)
    roundtrip = validate_roundtrip(comparison)

    maidata_path = inference_dir / "maidata.txt"
    materialization_path = inference_dir / "materialization.json"
    if not force and (maidata_path.exists() or materialization_path.exists()):
        raise ValueError(
            "materialization outputs already exist; pass --force to replace them"
        )
    difficulty_id = int(generation["source_difficulty_id"])
    maidata_path.write_text(
        maidata_text(
            chart,
            timing=timing,
            difficulty_id=difficulty_id,
            level=float(generation["level"]),
            cabinet=str(generation["cabinet"]),
            title=title or str(generation.get("song_id", "generated")),
            artist=artist,
            designer=designer,
        ),
        encoding="utf-8",
    )
    packaged_audio = inference_dir / ("track" + Path(audio_path).suffix.lower())
    link_type, audio_hash = _install_audio(
        Path(audio_path), packaged_audio, force=force
    )
    result = {
        "schema": "maimai.baseline_materialization.v1",
        "runtime_version": generation['runtime_version'],
        "execution_version": current_version(),
        "materializer_version": policy()['components']['materializer'],
        "architecture": generation["architecture"],
        "generation_schema": generation["schema"],
        "generation_sha256": _sha256(generation_path),
        "model_rows_sha256": _sha256(rows_path),
        "model_ir_sha256": _sha256(ir_path),
        "checkpoint_sha256": generation.get("checkpoint_sha256"),
        "route_artifact": generation["route_artifact"],
        "inference_source": generation.get("inference_source"),
        "song_id": generation["song_id"],
        "source_difficulty_id": difficulty_id,
        "maidata": maidata_path.name,
        "audio": packaged_audio.name,
        "audio_source": str(Path(audio_path).resolve()),
        "audio_bytes": packaged_audio.stat().st_size,
        "audio_sha256": audio_hash,
        "audio_link_type": link_type,
        "miacode_chart_ir_exe": str(executable),
        "miacode_chart_ir_sha256": _sha256(executable),
        "semantic_roundtrip": roundtrip,
        "spacing_policy": SPACING_POLICY,
        "max_subdivision": 384,
        "render_end_beat": reparsed.get("render_end_beat"),
    }
    _json(materialization_path, result)
    return result
