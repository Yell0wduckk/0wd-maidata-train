from __future__ import annotations


from typing import Final


BITPLANE_CACHE_SCHEMA: Final = "maimai.route71_short_notes_physical_bitplane_cache.v3"


LOSS_SCHEME: Final = "group-budget-physical-row-stages-v12"


PHYSICAL_COEFFICIENT: Final = 1.0


PHYSICAL_CATEGORIES: Final = (
    "multitouch",
    "overlap",
    "slide_too_fast",
    "tap_on_slide",
)


PHYSICAL_CATEGORY_WEIGHTS: Final = (1.0, 1.0, 4.0, 1.0)


SLOT_SECONDS: Final = 0.01


VALID_REGION_BITS: Final = 41


VALID_REGION_MASK: Final = (1 << VALID_REGION_BITS) - 1


ACTIVE_REGION_BITS: Final = 33


ACTIVE_REGION_MASK: Final = (1 << ACTIVE_REGION_BITS) - 1


PLANE_COUNT: Final = 5


PRESS_STATE: Final = 0


TOUCH_STATE: Final = 1


SLIDE_LAUNCH_GUARD: Final = 2


SLIDE_OCCUPANCY: Final = 3


SLIDE_ENTRY: Final = 4


KIND_TAP: Final = 0


KIND_TOUCH: Final = 1


KIND_HOLD: Final = 2


KIND_TOUCH_HOLD: Final = 3


KIND_SLIDE: Final = 4


KIND_WIFI: Final = 5


KIND_NONE: Final = -1


EVENT_NOTE: Final = 0


EVENT_SLIDE_BRANCH: Final = 1


CACHE_ACTION_COLUMNS: Final = (
    "moment_index",
    "source_row",
    "event_code",
    "kind_code",
    "region_mask",
    "path_id",
    "head_tick",
    "duration_tick",
    "head_slot",
    "hold_end_slot",
    "launch_slot",
    "slide_end_slot",
)


ACTION_FLOAT_COLUMNS: Final = (
    "head_units",
    "trace_start_units",
    "motion_units",
    "too_fast_deadline_units",
)


OPPORTUNITY_COLUMNS: Final = (
    "batch_index",
    "moment_order",
    "semantic_row",
    "delta_row",
    "source_row",
    "moment_row",
    "opportunity_kind",
    "anchor_tick",
    "conditioned_event_bit",
    "owner_start_lane",
    "insertion_event_bits",
)


ILLEGAL_SET_COLUMNS: Final = (
    "multitouch_event",
    "multitouch_kind",
    "multitouch_pad",
    "multitouch_start_pad",
    "overlap_pad",
    "overlap_start_pad",
    "tap_on_slide_pad",
    "tap_on_slide_start_pad",
    "duration_first_bad_slot",
)


PATH_BITSET_WORDS: Final = (568 + 63) // 64


BITPLANE_CACHE_CONTRACT: Final = {
    "schema": "maimai.route71_delta10ms_physical_bitplane.v2",
    "architecture": (
        "v6-window-conformer-hierarchical-moment2-route71-dualdist-"
        "delta10ms-physical-bitplane-v2"
    ),
    "loss_scheme": "group-budget-route71-delta10ms-physical-bitplane-v1",
    "base_loss_scheme": "group-budget-v1",
    "physical_coefficient": 0.05,
    "quantum_ms": 10,
    "storage": "five-plane-int64-carrier-uint64-low41",
    "cache": "canonical-chart-major-short-notes-five-plane-v3",
    "short_notes": "normalize-0-or-1-to-materialized-1-tick-preserve-source-row",
    "prefix": "decoder-row-dynamic-causal-boundary-v2",
    "execution": "readonly-cache-one-opportunity-per-thread-v2",
    "cpu_oracle": "readonly-cache-causal-reference-v2",
    "candidate_selection": "decoder-greedy-modal-route71-delta10ms-v1",
    "categories": list(PHYSICAL_CATEGORIES),
    "wifi": "same-endpoint-straight-slide-v1",
    "slide_head_tap": False,
    "jump_regions": False,
    "outer_keys_active": False,
}


CONTRACT: Final = {
    "schema":"maimai.baseline_physical.row_stages.v9",
    "objective":"conditional-stage-illegal-mass-before-hard-mask",
    "loss_scheme":LOSS_SCHEME,"base_loss_scheme":"dual-decoder-field-budget-v6",
    "physical_coefficient":PHYSICAL_COEFFICIENT,
    "physical_category_weights":dict(zip(PHYSICAL_CATEGORIES,PHYSICAL_CATEGORY_WEIGHTS,strict=True)),
    "physical_category_reduction":"weighted-sum-divided-by-fixed-four-v1",
    "categories":list(PHYSICAL_CATEGORIES),"quantum_ms":None,
    "storage":"causal-stage-small-chord-tables-and-large-chord-interval-relations",
    "execution":"worker-interval-merge-device-conditional-probabilities",
    "short_ticks":2,"release_ticks":22,"head_guard_last_tick":70,"path_pass_seconds":.2,
    "touch_repeat_seconds": .25,
    "touch_repeat_responsibility": "pad-only-all-touch-onsets",
    "same_path_wait": "exact-path-inclusive-launch-all-shapes",
    "slide_grouping":"note-owner-additive-tracks;ordinary-one;wifi-two;max-one-branch",
    "duration":"written-bpm-global-wait-and-release",
    "startup_credit":"one-short-or-touch-pool-per-group;long-hold-independent",
    "touch_hands":"all-active-one-pool;short-startup-cancels-pool",
    "short_hold":"tap-touch-equivalent-no-release",
    "touch_on_slide":"excluded",
    "path_coverage":"directed-wifi-three-straights",
    "handoff":"ordinary-end-to-start-one-to-one-0..22-global-ticks",
    "handoff_history":"lock-past-matches-before-pruning",
    "cache_identity":"maimai.shared_physical_moments.v5",
    "candidate_selection":"none",
    "time_stage_categories":["multitouch"],
    "stage_keys":"only-known-preceding-fields;physical-equivalent-flags-and-routes-share",
}


PHYSICAL_WARMUP_STEPS = 2500
OPPORTUNITY_NEXT_ONSET_NOTE = 0
OPPORTUNITY_NEXT_ONSET_BRANCH = 1
OPPORTUNITY_INSERT_SAME_ONSET = 2
PHYSICAL_EVENT_NAMES = (
    "SEGMENT_START",
    "MOMENT",
    "NOTE",
    "SLIDE_BRANCH",
    "TARGET_START",
    "SEGMENT_END",
)
CONTRACT["overlap"] = "half-open-exact-press-intervals"
CONTRACT["warmup_steps"] = PHYSICAL_WARMUP_STEPS
