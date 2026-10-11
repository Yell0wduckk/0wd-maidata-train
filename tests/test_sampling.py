"""Sampling ownership, complete audio coverage and deterministic extra exposure."""
from collections import Counter
from copy import deepcopy
import gzip
import json
import math

import pytest

from maidata.entry import CONFIG
from maidata.sampling import build_plan, chart_windows, make_occurrences, partition_plan, write_plan


def record(ticks=(), *, kinds=None, duration=40, bpm=60, first=0,
           chart_id="1:5", variant="audio-one", controls=None):
    source = dict(chart_id=chart_id, song_id=chart_id.split(":")[0],
                  difficulty=chart_id.split(":")[1], audio_variant=variant,
                  audio_asset="asset-one", path=f"charts/{chart_id.replace(':', '/')}.json.gz",
                  version="maimai DX FESTiVAL", audio_end_seconds=duration,
                  timing=dict(first_seconds=first, bpm=controls or [[0, 1, bpm]]))
    tokens = [dict(type="MOMENT", delta_ms=0)]
    mapping = [dict(token_index=0, tick=0)]
    for i, tick in enumerate(ticks):
        tokens.append(dict(type="NOTE", kind=kinds[i] if kinds else "tap", pad="A1"))
        mapping.append(dict(token_index=len(tokens) - 1, tick=tick))
    return dict(schema="maidata.full-chart-original.v2", view="original", source=source,
                start_tick=0, tokens=tokens, token_map=mapping,
                physical_categories={"overlap": 1}, basic_errors={})


def config(**changes):
    return dict(CONFIG, **changes)


def test_complete_audio_and_empty_tail_survive_last_note():
    windows = chart_windows(record([300 * 96], duration=350), config=config())
    assert len(windows) == 23
    assert [w["start_beat"] for w in windows] == list(range(0, 353, 16))
    assert windows[-2]["start_beat"] == 336
    assert windows[-1]["start_beat"] == 352
    assert windows[-1]["end_beat"] == 368
    assert [w["extra_tail"] for w in windows] == [False] * 22 + [True]
    assert all(w["unique_note_onsets"] == 0 for w in windows[-4:])
    assert windows[-1]["U"] == 372
    assert windows[-1]["audio_end_beat"] == 372


@pytest.mark.parametrize("end, starts", [
    (32.0, [0, 16, 32]),
    (math.nextafter(32.0, math.inf), [0, 16, 32, 48]),
])
def test_exact_audio_end_and_real_fraction_beyond_boundary(end, starts):
    windows = chart_windows(record(duration=end), config=config())
    assert [w["start_beat"] for w in windows] == starts
    assert windows[-1]["extra_tail"]


def test_negative_first_beat_keeps_padding_history_and_full_bpm_map():
    windows = chart_windows(record([0], duration=1, bpm=120, first=.25),
                            config=config(target_beats=4, target_stride_beats=4))
    assert [w["start_beat"] for w in windows] == [-1, 3]
    assert windows[0]["first_beat"] == -1
    assert windows[0]["audio_start_beat"] == -5
    assert windows[0]["history_start_tick"] == -5 * 96
    assert windows[0]["history_end_tick"] == -96
    changing = chart_windows(record(duration=10, controls=[[0, 1, 60], [8, 1, 120]]),
                             config=config(target_beats=4, target_stride_beats=4))
    assert [w["start_beat"] for w in changing] == [0, 4, 8, 12]


def test_overlapping_supervision_has_separate_complete_tail():
    windows = chart_windows(record(duration=31),
                            config=config(target_beats=16, target_stride_beats=10))
    assert [w["start_beat"] for w in windows] == [0, 10, 20, 36]
    assert windows[-1]["end_beat"] == 52
    assert windows[-1]["extra_tail"]
    with pytest.raises(ValueError):
        chart_windows(record(), config=config(target_stride_beats=17))
    with pytest.raises(ValueError):
        chart_windows(record(), config=config(target_stride_beats=0))


def test_complex_at_least_three_distinct_subquarter_gaps_and_no_pseudo_onsets():
    r = record([0, 12, 12, 36, 84, 180], kinds=["tap", "tap", "tap", "hold", "slide", "tap"])
    # Boundary/query MOMENTs and contents at their owner's onset are not new onsets.
    for kind, tick in [("MOMENT", 5), ("ROUTE", 84), ("SLIDE_DURATION", 84), ("MOMENT", 9999)]:
        r["tokens"].append(dict(type=kind))
        r["token_map"].append(dict(token_index=len(r["tokens"]) - 1, tick=tick))
    first = chart_windows(r, config=config())[0]
    assert first["unique_note_onsets"] == 5
    assert first["subquarter_delta_ticks"] == [12, 24, 48]
    assert first["subquarter_delta_kinds"] == 3
    assert first["complex"] and first["slide"]
    simpler = chart_windows(record([0, 12, 36, 132]), config=config())[0]
    assert simpler["subquarter_delta_ticks"] == [12, 24]
    assert not simpler["complex"]


def test_half_open_ownership_excludes_cross_window_gap_and_zero_history():
    windows = chart_windows(record([73, 96, 100, 110, 130, 160], duration=3),
        config=config(target_beats=1, target_stride_beats=1, chart_history_beats=0))
    assert windows[0]["unique_note_onsets"] == 1
    assert windows[0]["subquarter_delta_ticks"] == []
    assert windows[1]["unique_note_onsets"] == 5
    assert windows[1]["subquarter_delta_ticks"] == [4, 10, 20, 30]
    assert 23 not in windows[1]["subquarter_delta_ticks"]
    assert all(w["history_start_tick"] == w["history_end_tick"] for w in windows)


def test_incompatible_view_or_lost_real_onset_is_an_error():
    bad = record([0])
    bad["view"] = "mirror"
    with pytest.raises(ValueError):
        chart_windows(bad, config=config())
    with pytest.raises(ValueError):
        chart_windows(record([-96]), config=config())
    with pytest.raises(ValueError):
        chart_windows(record([32 * 96], duration=32), config=config())


def exposure_windows():
    result = []
    for chart_id, variant in [("1:5", "same"), ("1:6", "same"), ("2:5", "other")]:
        r = record([0, 12, 36, 84], duration=20, chart_id=chart_id, variant=variant)
        result.extend(chart_windows(r, config=config()))
    return result


def test_complex_two_total_exposures_preserves_every_base_and_audio_group():
    windows = exposure_windows()
    windows[1]["slide"] = True  # Slide-only gets no additional exposure.
    original = deepcopy(windows)
    occurrences = make_occurrences(windows, multiplier=2., seed=42, epoch=1)
    assert windows == original
    counts = Counter(row["window_id"] for row in occurrences)
    assert counts == {w["window_id"]: 2 if w["complex"] else 1 for w in windows}
    for w in windows:
        local = [r["occurrence"] for r in occurrences if r["window_id"] == w["window_id"]]
        assert sorted(local) == list(range(counts[w["window_id"]]))
    runs = [r["audio_variant"] for i, r in enumerate(occurrences)
            if i == 0 or r["audio_variant"] != occurrences[i - 1]["audio_variant"]]
    assert len(runs) == len(set(runs)) == 2
    assert occurrences == make_occurrences(windows, multiplier=2., seed=42, epoch=1)
    assert occurrences != make_occurrences(windows, multiplier=2., seed=42, epoch=2)


def test_fractional_extra_selection_is_without_replacement_and_keeps_all_bases():
    windows = exposure_windows()
    occurrences = make_occurrences(windows, multiplier=1.5, seed=4, epoch=1)
    # Three complex base windows: half-up(3 * .5) = two additional instances.
    assert len(occurrences) == len(windows) + 2
    counts = Counter(r["window_id"] for r in occurrences)
    assert set(counts) == {w["window_id"] for w in windows}
    assert sorted(counts.values()).count(2) == 2
    assert max(counts.values()) == 2
    assert len(make_occurrences(windows, multiplier=1., seed=4, epoch=1)) == len(windows)


def test_selected_cache_to_grouped_plan_roundtrip_and_metadata_binding(tmp_path):
    cache = tmp_path / "cache"
    sources, selected = [], []
    for chart_id in ("1:5", "1:6"):
        r = record([0, 12, 36, 84], duration=20, chart_id=chart_id)
        r["metadata"] = dict(first_seconds=0, controls=[dict(kind="bpm", bpm=60,
            position_beats=dict(num=0, den=1))])
        source = r["source"]
        sources.append(source)
        payload = cache / source["path"]
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(gzip.compress(json.dumps(r).encode()))
        selected.append(dict(chart_id=chart_id, song_id="1", difficulty=source["difficulty"],
            selection_index=1, chart_index=len(selected) + 1,
            audio_variant=source["audio_variant"], audio_asset=source["audio_asset"],
            audio_end_seconds=20, chart_cache_path=source["path"], cache_status="success"))
    # The unrelated catalog entry has no payload; building a subset must not touch it.
    catalog = dict(schema="maidata.full-chart-original.v2", views=["original"],
                   charts=[*sources, dict(chart_id="999:5", path="missing.json.gz")])
    (cache / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (cache / "production-summary.json").write_text(json.dumps(dict(results=[
        dict(chart_id=s["chart_id"], status="success", view="original") for s in sources
    ])), encoding="utf-8")
    selection = dict(name="synthetic", selected_chart_ids=[s["chart_id"] for s in sources],
        charts=selected, sources=dict(chart_catalog=str(cache / "catalog.json"),
                                     chart_schema="maidata.full-chart-original.v2"))
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    plan = build_plan(path, config=config(), seed=9, epoch=1)
    assert plan["totals"]["charts"] == 2
    assert plan["totals"]["audio_groups"] == 1
    assert plan["totals"]["base_windows"] == 6
    assert plan["totals"]["occurrences"] == 8
    assert [r["selection_index"] for r in plan["charts"]] == [1, 1]
    assert [r["chart_index"] for r in plan["charts"]] == [1, 2]
    group = plan["audio_groups"][0]
    assert group["encoder_starts_beats"] == [0, 16]
    assert group["draft_end_beat"] == group["supervision_end_beat"] == 48
    assert group["occurrence_start"] == 0 and group["occurrence_stop"] == 8
    paths = write_plan(plan, tmp_path / "output")
    manifest = json.loads(paths["plan"].read_text(encoding="utf-8"))
    for key in ("windows", "occurrences"):
        assert key not in manifest
        assert manifest["files"][key] == paths[key].name
        rows = [json.loads(line) for line in paths[key].read_text(encoding="utf-8").splitlines()]
        assert rows == plan[key]
    # Relocating a cache uses an explicit override rather than rewriting selection identity.
    selection["sources"]["chart_catalog"] = str(tmp_path / "old-root" / "catalog.json")
    path.write_text(json.dumps(selection), encoding="utf-8")
    assert build_plan(path, config=config(), chart_cache=cache)["totals"]["charts"] == 2
    selection["charts"][0]["audio_variant"] = "wrong"
    path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        build_plan(path, config=config(), chart_cache=cache)


@pytest.fixture
def split_inputs(tmp_path):
    cache = tmp_path / "cache"
    sources, selected, payloads = [], [], {}
    for index, chart_id in enumerate(("1:5", "1:6", "2:5"), 1):
        song = chart_id.split(":")[0]
        r = record([0, 12, 36, 84], duration=20, chart_id=chart_id,
                   variant=f"audio-{song}")
        r["source"]["audio_asset"] = f"asset-{song}"
        r["metadata"] = dict(first_seconds=0, controls=[dict(kind="bpm", bpm=60,
            position_beats=dict(num=0, den=1))])
        source = r["source"]
        sources.append(source)
        path = cache / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(r).encode()))
        payloads[chart_id] = path
        selected.append(dict(chart_id=chart_id, song_id=song, difficulty=source["difficulty"],
            selection_index=int(song), chart_index=index, audio_variant=source["audio_variant"],
            audio_asset=source["audio_asset"], audio_end_seconds=20,
            chart_cache_path=source["path"], cache_status="success"))
    (cache / "catalog.json").write_text(json.dumps(dict(
        schema="maidata.full-chart-original.v2", views=["original"], charts=sources)), encoding="utf-8")
    (cache / "production-summary.json").write_text(json.dumps(dict(results=[
        dict(chart_id=r["chart_id"], status="success", view="original") for r in sources
    ])), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(name="split-fixture",
        selected_chart_ids=[r["chart_id"] for r in sources], charts=selected,
        sources=dict(chart_catalog=str(cache / "catalog.json"),
                     chart_schema="maidata.full-chart-original.v2"))), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps(dict(schema="maidata.encoder-draft-split.v1",
        source_selection="selection.json",
        train=dict(song_ids=["1"], chart_ids=["1:5", "1:6"],
                   audio_assets=["asset-1"], audio_variants=["audio-1"]),
        validation=dict(song_ids=["2"], chart_ids=["2:5"],
                        audio_assets=["asset-2"], audio_variants=["audio-2"]))), encoding="utf-8")
    return selection, split, cache, payloads


def test_partition_build_reads_only_requested_side_and_validation_is_one_pass(split_inputs):
    selection, split, _, payloads = split_inputs
    validation_bytes = payloads["2:5"].read_bytes()
    payloads["2:5"].unlink()
    train = build_plan(selection, config=config(), split=split, partition="train")
    assert train["totals"]["charts"] == 2
    assert train["totals"]["occurrences"] == 8
    payloads["2:5"].write_bytes(validation_bytes)
    payloads["1:5"].unlink()
    payloads["1:6"].unlink()
    validation = build_plan(selection, config=config(complex_rhythm_multiplier=3.5),
                            split=split, partition="validation")
    assert validation["totals"]["base_windows"] == validation["totals"]["occurrences"] == 3
    assert validation["totals"]["complex_windows"] == 1
    assert validation["config"]["complex_rhythm_multiplier"] == 1.0
    assert all(row["occurrence"] == 0 for row in validation["occurrences"])
    with pytest.raises(ValueError, match="explicit split"):
        build_plan(selection, config=config(), partition="train")


def test_partition_derivation_needs_only_existing_plan_and_split(split_inputs):
    selection, split, cache, payloads = split_inputs
    plan = build_plan(selection, config=config(), seed=7, epoch=2)
    original = deepcopy(plan)
    # Moving/reusing a generated all-plan does not reread its source payloads or metadata.
    for path in [selection, cache / "catalog.json", cache / "production-summary.json", *payloads.values()]:
        path.unlink()
    train = partition_plan(plan, split, partition="train")
    validation = partition_plan(plan, split, partition="validation")
    assert plan == original
    assert {c["chart_id"] for c in train["charts"]} == {"1:5", "1:6"}
    assert {c["chart_id"] for c in validation["charts"]} == {"2:5"}
    assert train["totals"]["occurrences"] == 8
    assert validation["totals"]["occurrences"] == 3
    assert {w["window_id"] for w in train["windows"]} | {
        w["window_id"] for w in validation["windows"]} == {w["window_id"] for w in plan["windows"]}
    for result in (train, validation):
        group = result["audio_groups"][0]
        assert group["chart_ids"] == [c["chart_id"] for c in result["charts"]]
        assert group["occurrence_start"] == 0
        assert group["occurrence_stop"] == len(result["occurrences"])
        assert result["sources"]["split"] == str(split)
    assert partition_plan(plan, split, partition="all")["occurrences"] == plan["occurrences"]
    with pytest.raises(ValueError, match="complete all-selection"):
        partition_plan(train, split, partition="validation")
    # Even separately named songs cannot leak the same existing audio asset across sides.
    bad = json.loads(split.read_text(encoding="utf-8"))
    bad["validation"]["audio_assets"] = ["asset-1"]
    split.write_text(json.dumps(bad), encoding="utf-8")
    for group in plan["audio_groups"]:
        if group["audio_variant"] == "audio-2":
            group["audio_asset"] = "asset-1"
    with pytest.raises(ValueError, match="audio_assets overlap"):
        partition_plan(plan, split, partition="train")
