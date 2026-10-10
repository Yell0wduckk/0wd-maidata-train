#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kActionWidth = 12;
constexpr int kOpportunityWidth = 11;
constexpr int kPress = 0;
constexpr int kTouch = 1;
constexpr int kLaunch = 2;
constexpr int kOccupancy = 3;
constexpr int kEntry = 4;
constexpr int kEventNote = 0;
constexpr int kKindTouch = 1;
constexpr int kKindHold = 2;
constexpr int kKindTouchHold = 3;
constexpr int kMaxPrefix = 64;
constexpr unsigned long long kValidMask = (1ULL << 41) - 1ULL;
constexpr unsigned long long kActiveMask = (1ULL << 33) - 1ULL;

__device__ __forceinline__ long long ll_max(long long left, long long right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double double_max(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ long long half_up(double value) {
  return static_cast<long long>(floor(value + 0.5 + 1.0e-8));
}

__device__ __forceinline__ unsigned long long permute_mask(
    unsigned long long value, int view, const unsigned char* lut) {
  if (view == 0) return value & kValidMask;
  unsigned long long result = value & (1ULL << 16);
  constexpr int offsets[5] = {0, 8, 17, 25, 33};
#pragma unroll
  for (int index = 0; index < 5; ++index) {
    const int offset = offsets[index];
    const unsigned char byte = static_cast<unsigned char>((value >> offset) & 255ULL);
    const int family = (index == 2 || index == 3) ? 1 : 0;
    result |= static_cast<unsigned long long>(lut[(view * 2 + family) * 256 + byte]) << offset;
  }
  return result & kValidMask;
}


__global__ void action_mask_views_kernel(
    const unsigned long long* canonical,
    unsigned long long* output,
    long long count,
    const unsigned char* lut) {
  const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count * 4) return;
  const int view = static_cast<int>(index / count);
  const long long action = index - static_cast<long long>(view) * count;
  output[index] = permute_mask(canonical[action], view, lut);
}

__global__ void materialize_planes_kernel(
    const unsigned char* canonical_low_bytes,
    const unsigned char* canonical_high_bits,
    long long canonical_value_count,
    const long long* canonical_offsets,
    unsigned long long* output,
    const long long* output_offsets,
    const long long* pair_charts,
    const long long* pair_views,
    const unsigned char* lut) {
  const long long pair = blockIdx.x;
  const long long chart = pair_charts[pair];
  const int view = static_cast<int>(pair_views[pair]);
  const long long source = canonical_offsets[chart];
  const long long target = output_offsets[pair];
  const long long count = canonical_offsets[chart + 1] - source;
  for (long long index = threadIdx.x; index < count; index += blockDim.x) {
    const long long packed_index = source + index;
    unsigned long long value = 0;
#pragma unroll
    for (int byte = 0; byte < 5; ++byte) {
      value |= static_cast<unsigned long long>(
          canonical_low_bytes[static_cast<long long>(byte) * canonical_value_count + packed_index])
          << (byte * 8);
    }
    value |= static_cast<unsigned long long>(
        (canonical_high_bits[packed_index >> 3] >> (packed_index & 7)) & 1U) << 40;
    output[target + index] = permute_mask(value, view, lut);
  }
}

__device__ __forceinline__ double tick_second(
    long long tick,
    long long chart,
    const double* point_ticks,
    const double* point_seconds,
    const double* point_bpms,
    const bool* point_padding,
    long long timing_width) {
  long long selected = 0;
  const long long base = chart * timing_width;
  for (long long index = 1; index < timing_width; ++index) {
    if (point_padding[base + index] || point_ticks[base + index] > static_cast<double>(tick)) break;
    selected = index;
  }
  return point_seconds[base + selected] +
      (static_cast<double>(tick) - point_ticks[base + selected]) * 60.0 /
      (point_bpms[base + selected] * 96.0);
}

struct Geometry {
  long long head;
  long long hold_end;
  long long launch;
  long long slide_end;
  double head_units;
  double trace_units;
  double motion_units;
  double deadline_units;
};

struct ActionRow {
  const long long* wide;
  const int* compact;

  __device__ __forceinline__ long long operator[](int column) const {
    return compact != nullptr
        ? static_cast<long long>(compact[column])
        : wide[column];
  }
};

struct ActionTable {
  const long long* wide;
  const int* compact;

  __device__ __forceinline__ ActionRow row(long long index) const {
    return ActionRow{
        wide == nullptr ? nullptr : wide + index * kActionWidth,
        compact == nullptr ? nullptr : compact + index * kActionWidth};
  }
};

__device__ __forceinline__ Geometry static_geometry(
    const ActionRow& action, const double* floating) {
  return Geometry{action[8], action[9], action[10], action[11],
                  floating[0], floating[1], floating[2], floating[3]};
}

__device__ __forceinline__ Geometry rebased_geometry(
    const ActionRow& action,
    long long head_tick,
    long long chart,
    const double* point_ticks,
    const double* point_seconds,
    const double* point_bpms,
    const bool* point_padding,
    long long timing_width,
    const double* critical) {
  const double head_second = tick_second(
      head_tick, chart, point_ticks, point_seconds, point_bpms, point_padding, timing_width);
  const long long head_slot = static_cast<long long>(floor(head_second * 100.0 + 1.0e-8));
  Geometry result{head_slot, head_slot + 1, -1, -1, head_second * 100.0, 0.0, 0.0, 0.0};
  const long long duration = action[7];
  if (action[3] == kKindHold || action[3] == kKindTouchHold) {
    const double end_second = tick_second(
        head_tick + duration, chart, point_ticks, point_seconds, point_bpms,
        point_padding, timing_width);
    result.hold_end = head_slot + ll_max(
        1LL, static_cast<long long>(ceil(
            double_max(0.0, end_second - head_second) * 100.0 - 1.0e-9)));
  }
  if (action[5] >= 0) {
    const double trace_second = tick_second(
        head_tick + 96, chart, point_ticks, point_seconds, point_bpms,
        point_padding, timing_width);
    const double end_second = tick_second(
        head_tick + 96 + ll_max(1LL, duration), chart, point_ticks, point_seconds,
        point_bpms, point_padding, timing_width);
    const double motion_second = double_max(0.01, end_second - trace_second);
    result.trace_units = trace_second * 100.0;
    result.motion_units = motion_second * 100.0;
    result.launch = half_up(result.trace_units);
    result.slide_end = ll_max(result.launch + 1, half_up(end_second * 100.0));
    const double proportion = critical[action[5]];
    result.deadline_units =
        (trace_second + proportion * motion_second -
         (42.0 / 180.0 + (1.0 - proportion) * motion_second / 4.0)) * 100.0;
  }
  return result;
}

__device__ __forceinline__ long long path_boundary(
    const double* proportions,
    long long area,
    const Geometry& geometry,
    long long path_begin) {
  if (area == path_begin) return geometry.launch;
  return ll_max(
      geometry.launch,
      half_up(geometry.trace_units + proportions[area] * geometry.motion_units));
}

__device__ __forceinline__ unsigned long long action_state(
    const ActionRow& action,
    unsigned long long region_mask,
    const Geometry& geometry,
    long long slot,
    int plane,
    const long long* path_offsets,
    const long long* path_masks,
    const double* proportions) {
  if (plane == kPress || plane == kTouch) {
    if (action[2] != kEventNote) return 0;
    bool active = slot == geometry.head;
    if (action[3] == kKindHold || action[3] == kKindTouchHold) {
      active = geometry.head <= slot && slot < geometry.hold_end;
    }
    if (!active) return 0;
    const bool touch = action[3] == kKindTouch || action[3] == kKindTouchHold;
    return ((plane == kTouch) == touch)
        ? region_mask : 0ULL;
  }
  const long long path = action[5];
  if (path < 0) return 0;
  const long long begin = path_offsets[path];
  const long long end = path_offsets[path + 1];
  if (plane == kLaunch) {
    return geometry.launch - 5 <= slot && slot <= geometry.launch
        ? static_cast<unsigned long long>(path_masks[begin]) : 0ULL;
  }
  unsigned long long result = 0;
  for (long long area = begin; area < end; ++area) {
    const long long boundary = path_boundary(proportions, area, geometry, begin);
    if (plane == kEntry) {
      if (area > begin && boundary == slot) {
        result |= static_cast<unsigned long long>(path_masks[area]);
      }
      continue;
    }
    const long long start = ll_max(geometry.launch + 1, boundary);
    const long long stop = area + 1 < end
        ? path_boundary(proportions, area + 1, geometry, begin)
        : geometry.slide_end + 1;
    if (start <= slot && slot < stop) {
      result |= static_cast<unsigned long long>(path_masks[area]);
    }
  }
  return result;
}

struct Context {
  long long chart;
  long long plane_chart;
  int view;
  long long moment;
  long long boundary;
  long long carry_begin;
  long long carry_end;
  long long prefix_count;
  long long prefix[kMaxPrefix];
};

__device__ __forceinline__ unsigned long long dense_read(
    const unsigned long long* planes,
    const long long* plane_offsets,
    const long long* origins,
    const long long* counts,
    long long plane_chart,
    long long chart,
    int plane,
    long long slot) {
  const long long local = slot - origins[chart];
  if (local < 0 || local >= counts[chart]) return 0;
  return planes[plane_offsets[plane_chart] + static_cast<long long>(plane) * counts[chart] + local];
}


__device__ __forceinline__ unsigned long long entry_in_range(
    const ActionRow& action,
    const Geometry& geometry,
    long long first_slot,
    long long stop_slot,
    const long long* path_offsets,
    const long long* path_masks,
    const double* proportions) {
  const long long path = action[5];
  if (path < 0 || first_slot >= stop_slot) return 0;
  const long long begin = path_offsets[path];
  const long long end = path_offsets[path + 1];
  unsigned long long result = 0;
  for (long long area = begin + 1; area < end; ++area) {
    const long long boundary = path_boundary(proportions, area, geometry, begin);
    if (first_slot <= boundary && boundary < stop_slot) {
      result |= static_cast<unsigned long long>(path_masks[area]);
    }
  }
  return result;
}

constexpr int kIllegalWidthV4 = 9;
constexpr int kPathWordsV4 = 9;

struct CurrentStateV4 {
  unsigned long long press;
  unsigned long long touch;
  unsigned long long launch;
  unsigned long long occupancy;
};

__device__ __forceinline__ CurrentStateV4 or_state_v4(
    const CurrentStateV4& left, const CurrentStateV4& right) {
  return CurrentStateV4{left.press | right.press, left.touch | right.touch,
                        left.launch | right.launch, left.occupancy | right.occupancy};
}

__device__ __forceinline__ int hand_count_v4(const CurrentStateV4& value) {
  const unsigned long long press = value.press & kActiveMask;
  const unsigned long long touch = value.touch & kActiveMask;
  return __popcll(press | (value.launch & (~touch) & kValidMask)) +
      __popcll(value.occupancy & kValidMask) + static_cast<int>(touch != 0);
}

__device__ __forceinline__ CurrentStateV4 add_note_v4(
    CurrentStateV4 value, unsigned long long mask, bool touch_kind) {
  if (touch_kind) value.touch |= mask;
  else value.press |= mask;
  return value;
}

__device__ __forceinline__ Context build_context_v4(
    const long long* opportunity,
    long long semantic_row,
    const long long* source_action,
    const long long* source_offsets,
    const long long* moment_slots,
    const long long* moment_offsets,
    const long long* carry_offsets,
    const long long* input_sources,
    const long long* local_blocks,
    long long row_width,
    const long long* batch_charts,
    const long long* batch_plane_charts,
    const long long* batch_views) {
  const long long batch = opportunity[0];
  const long long chart = batch_charts[batch];
  Context context{};
  context.chart = chart;
  context.plane_chart = batch_plane_charts[batch];
  context.view = static_cast<int>(batch_views[batch]);
  context.moment = opportunity[1];
  const long long global_moment = moment_offsets[chart] + context.moment;
  context.boundary = moment_slots[global_moment];
  context.carry_begin = carry_offsets[global_moment];
  context.carry_end = carry_offsets[global_moment + 1];
  context.prefix_count = 0;
  if (semantic_row < 0 || semantic_row >= row_width) return context;
  const long long block = local_blocks[batch * row_width + semantic_row];
  const long long source_count = source_offsets[chart + 1] - source_offsets[chart];
  for (long long row = 0; row <= semantic_row; ++row) {
    if (local_blocks[batch * row_width + row] != block) continue;
    const long long source = input_sources[batch * row_width + row];
    if (source < 0 || source >= source_count) continue;
    const long long action = source_action[source_offsets[chart] + source];
    if (action < 0) continue;
    bool duplicate = false;
    for (long long old = 0; old < context.prefix_count; ++old) {
      duplicate = duplicate || context.prefix[old] == action;
    }
    if (!duplicate && context.prefix_count < kMaxPrefix) {
      context.prefix[context.prefix_count++] = action;
    }
  }
  return context;
}

__device__ __forceinline__ CurrentStateV4 causal_state_v4(
    long long slot,
    const Context& context,
    const unsigned long long* planes,
    const long long* plane_offsets,
    const long long* origins,
    const long long* counts,
    const ActionTable& actions,
    const long long* action_masks_by_view,
    long long action_count,
    const double* action_floating,
    const long long* carry_indices,
    const long long* path_offsets,
    const long long* path_masks_by_view,
    long long mask_count,
    const double* proportions) {
  CurrentStateV4 result{0, 0, 0, 0};
  if (slot < context.boundary) {
    result.press = dense_read(planes, plane_offsets, origins, counts,
                              context.plane_chart, context.chart, kPress, slot);
    result.touch = dense_read(planes, plane_offsets, origins, counts,
                              context.plane_chart, context.chart, kTouch, slot);
    result.launch = dense_read(planes, plane_offsets, origins, counts,
                               context.plane_chart, context.chart, kLaunch, slot);
    result.occupancy = dense_read(planes, plane_offsets, origins, counts,
                                  context.plane_chart, context.chart, kOccupancy, slot);
    return result;
  }
  const long long* masks = path_masks_by_view +
      static_cast<long long>(context.view) * mask_count;
  for (long long cursor = context.carry_begin; cursor < context.carry_end; ++cursor) {
    const long long index = carry_indices[cursor];
    const ActionRow action = actions.row(index);
    const Geometry geometry = static_geometry(action, action_floating + index * 4);
    const unsigned long long region_mask = static_cast<unsigned long long>(
        action_masks_by_view[static_cast<long long>(context.view) * action_count + index]);
    result.press |= action_state(action, region_mask, geometry, slot, kPress,
                                 path_offsets, masks, proportions);
    result.touch |= action_state(action, region_mask, geometry, slot, kTouch,
                                 path_offsets, masks, proportions);
    result.launch |= action_state(action, region_mask, geometry, slot, kLaunch,
                                  path_offsets, masks, proportions);
    result.occupancy |= action_state(action, region_mask, geometry, slot, kOccupancy,
                                     path_offsets, masks, proportions);
  }
  return result;
}

__device__ __forceinline__ CurrentStateV4 prefix_state_v4(
    long long slot,
    long long head_tick,
    const Context& context,
    const ActionTable& actions,
    const long long* action_masks_by_view,
    long long action_count,
    const long long* path_offsets,
    const long long* path_masks_by_view,
    long long mask_count,
    const double* proportions,
    const double* critical,
    const double* point_ticks,
    const double* point_seconds,
    const double* point_bpms,
    const bool* point_padding,
    long long timing_width) {
  CurrentStateV4 result{0, 0, 0, 0};
  const long long* masks = path_masks_by_view +
      static_cast<long long>(context.view) * mask_count;
  for (long long cursor = 0; cursor < context.prefix_count; ++cursor) {
    const long long index = context.prefix[cursor];
    const ActionRow action = actions.row(index);
    const Geometry geometry = rebased_geometry(
        action, head_tick, context.chart, point_ticks, point_seconds, point_bpms,
        point_padding, timing_width, critical);
    const unsigned long long region_mask = static_cast<unsigned long long>(
        action_masks_by_view[static_cast<long long>(context.view) * action_count + index]);
    result.press |= action_state(action, region_mask, geometry, slot, kPress,
                                 path_offsets, masks, proportions);
    result.touch |= action_state(action, region_mask, geometry, slot, kTouch,
                                 path_offsets, masks, proportions);
    result.launch |= action_state(action, region_mask, geometry, slot, kLaunch,
                                  path_offsets, masks, proportions);
    result.occupancy |= action_state(action, region_mask, geometry, slot, kOccupancy,
                                     path_offsets, masks, proportions);
  }
  return result;
}

__device__ __forceinline__ unsigned long long historical_entry_v4(
    long long slot,
    long long head_tick,
    const Context& context,
    const unsigned long long* planes,
    const long long* plane_offsets,
    const long long* origins,
    const long long* counts,
    const ActionTable& actions,
    const long long* action_masks_by_view,
    long long action_count,
    const double* action_floating,
    const long long* carry_indices,
    const long long* path_offsets,
    const long long* path_masks_by_view,
    long long mask_count,
    const double* proportions,
    const double* critical,
    const double* point_ticks,
    const double* point_seconds,
    const double* point_bpms,
    const bool* point_padding,
    long long timing_width) {
  unsigned long long result = 0;
  const long long first = slot - 18;
  const long long dense_stop = slot < context.boundary ? slot : context.boundary;
  for (long long at = first; at < dense_stop; ++at) {
    result |= dense_read(planes, plane_offsets, origins, counts, context.plane_chart,
                         context.chart, kEntry, at);
  }
  const long long dynamic_first = ll_max(first, context.boundary);
  const long long* masks = path_masks_by_view +
      static_cast<long long>(context.view) * mask_count;
  if (dynamic_first < slot) {
    for (long long cursor = context.carry_begin; cursor < context.carry_end; ++cursor) {
      const long long index = carry_indices[cursor];
      const ActionRow action = actions.row(index);
      result |= entry_in_range(action, static_geometry(action, action_floating + index * 4),
                               dynamic_first, slot, path_offsets, masks, proportions);
    }
  }
  for (long long cursor = 0; cursor < context.prefix_count; ++cursor) {
    const long long index = context.prefix[cursor];
    const ActionRow action = actions.row(index);
    const Geometry geometry = rebased_geometry(
        action, head_tick, context.chart, point_ticks, point_seconds, point_bpms,
        point_padding, timing_width, critical);
    result |= entry_in_range(action, geometry, first, slot,
                             path_offsets, masks, proportions);
  }
  return result;
}

template <bool DeltaOnly>
__global__ void analyze_v4_kernel(
    const unsigned long long* planes,
    const long long* plane_offsets,
    const long long* origins,
    const long long* counts,
    const int* actions_compact,
    const long long* action_masks_by_view,
    long long action_count,
    const double* action_floating,
    const long long* source_action,
    const long long* source_offsets,
    const long long* moment_slots,
    const long long* moment_offsets,
    const long long* carry_offsets,
    const long long* carry_indices,
    const long long* opportunities,
    const long long* input_sources,
    const long long* local_blocks,
    long long row_width,
    const long long* batch_charts,
    const long long* batch_plane_charts,
    const long long* batch_views,
    const long long* delta_query_slots,
    long long delta_width,
    const double* point_ticks,
    const double* point_seconds,
    const double* point_bpms,
    const bool* point_padding,
    long long timing_width,
    const long long* path_offsets,
    const long long* path_masks_by_view,
    long long mask_count,
    const double* proportions,
    const double* critical,
    const long long* flags,
    const long long* path_ids_by_view,
    long long* illegal,
    unsigned long long* active_paths,
    bool* delta_bad,
    bool geometry_only,
    long long opportunity_count) {
  const long long op_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (op_index >= opportunity_count) return;
  const long long* opportunity = opportunities + op_index * kOpportunityWidth;
  const long long batch = opportunity[0];
  const long long chart = batch_charts[batch];
  const ActionTable actions{nullptr, actions_compact};
  if constexpr (DeltaOnly) {
    Context delta_context = build_context_v4(
        opportunity, opportunity[3], source_action, source_offsets, moment_slots,
        moment_offsets, carry_offsets, input_sources, local_blocks, row_width,
        batch_charts, batch_plane_charts, batch_views);
    for (long long offset = 0; offset < delta_width; ++offset) {
      const long long query = delta_query_slots[op_index * delta_width + offset];
      bool bad = false;
      if (query >= 0) {
        const CurrentStateV4 value = causal_state_v4(
            query, delta_context, planes, plane_offsets, origins, counts, actions,
            action_masks_by_view, action_count, action_floating, carry_indices,
            path_offsets, path_masks_by_view, mask_count, proportions);
        bad = hand_count_v4(value) >= 2;
      }
      delta_bad[op_index * delta_width + offset] = bad;
    }

    return;
  }
  long long* out = illegal + op_index * kIllegalWidthV4;
#pragma unroll
  for (int column = 0; column < kIllegalWidthV4; ++column) out[column] = 0;
  out[8] = -1;
  unsigned long long* out_paths = active_paths + op_index * kPathWordsV4;
#pragma unroll
  for (int word = 0; word < kPathWordsV4; ++word) out_paths[word] = 0;
  Context context = build_context_v4(
      opportunity, opportunity[2], source_action, source_offsets, moment_slots,
      moment_offsets, carry_offsets, input_sources, local_blocks, row_width,
      batch_charts, batch_plane_charts, batch_views);
  const long long head_tick = opportunity[7];
  const long long head_slot = context.boundary;

  const CurrentStateV4 causal = causal_state_v4(
      head_slot, context, planes, plane_offsets, origins, counts, actions,
      action_masks_by_view, action_count, action_floating, carry_indices,
      path_offsets, path_masks_by_view, mask_count, proportions);
  const CurrentStateV4 prefix = prefix_state_v4(
      head_slot, head_tick, context, actions, action_masks_by_view, action_count,
      path_offsets, path_masks_by_view, mask_count, proportions, critical,
      point_ticks, point_seconds, point_bpms, point_padding, timing_width);
  const CurrentStateV4 state = or_state_v4(causal, prefix);
  const int base_count = hand_count_v4(state);
  int prefix_notes = 0;
  for (long long cursor = 0; cursor < context.prefix_count; ++cursor) {
    prefix_notes += actions.row(context.prefix[cursor])[2] == kEventNote;
  }

  if (opportunity[6] == 2) {  // OPPORTUNITY_INSERT_SAME_ONSET
    if (prefix_notes > 0 && base_count >= 2) {
      bool has_safe_non_touch_position = false;
      for (int position = 0; position < 33; ++position) {
        has_safe_non_touch_position = has_safe_non_touch_position ||
            hand_count_v4(add_note_v4(state, 1ULL << position, false)) <= 2;
      }
      if ((state.touch & kActiveMask) != 0 && !has_safe_non_touch_position) {
        out[1] |= (1LL << 0) | (1LL << kKindHold) |
            (1LL << 4) | (1LL << 5);
      } else if ((state.touch & kActiveMask) == 0 && !has_safe_non_touch_position) {
        out[0] = 1;
      }
    }
    return;
  }

  const long long source = opportunity[4];
  const long long source_count = source_offsets[chart + 1] - source_offsets[chart];
  if (source < 0 || source >= source_count) return;
  const long long target_index = source_action[source_offsets[chart] + source];
  if (target_index < 0) return;
  const ActionRow target = actions.row(target_index);
  const Geometry target_geometry = rebased_geometry(
      target, head_tick, chart, point_ticks, point_seconds, point_bpms,
      point_padding, timing_width, critical);
  const long long target_path = target[5];

  if (target_path >= 0 && !(flags[target_path] & 3LL)) {
    const long long candidate_launch = target_geometry.launch;
    for (long long cursor = context.carry_begin; cursor < context.carry_end; ++cursor) {
      const long long index = carry_indices[cursor];
      const ActionRow old = actions.row(index);
      if (old[5] < 0 || (flags[old[5]] & 3LL)) continue;
      const Geometry geometry = static_geometry(old, action_floating + index * 4);
      if (geometry.launch - 5 > candidate_launch || geometry.slide_end < head_slot) continue;
      const long long viewed = path_ids_by_view[static_cast<long long>(context.view) * 568 + old[5]];
      out_paths[viewed >> 6] |= 1ULL << (viewed & 63);
    }
    for (long long cursor = 0; cursor < context.prefix_count; ++cursor) {
      const long long index = context.prefix[cursor];
      const ActionRow old = actions.row(index);
      if (old[5] < 0 || (flags[old[5]] & 3LL)) continue;
      const Geometry geometry = rebased_geometry(
          old, head_tick, chart, point_ticks, point_seconds, point_bpms,
          point_padding, timing_width, critical);
      if (geometry.launch - 5 > candidate_launch || geometry.slide_end < head_slot) continue;
      const long long viewed = path_ids_by_view[static_cast<long long>(context.view) * 568 + old[5]];
      out_paths[viewed >> 6] |= 1ULL << (viewed & 63);
    }
  }

  if (target[2] != kEventNote) return;
  const long long kind = target[3];
  const bool touch_kind = kind == kKindTouch || kind == kKindTouchHold;
  const bool start_kind = kind == 4 || kind == 5;
  if (!geometry_only) {
  CurrentStateV4 without_guard = state;
  without_guard.launch = 0;
  unsigned long long bad_position = 0;
  const int positions = start_kind ? 8 : 33;
  for (int position = 0; position < positions; ++position) {
    const unsigned long long mask = 1ULL << position;
    const int with_count = hand_count_v4(add_note_v4(state, mask, touch_kind));
    const int without_count = hand_count_v4(add_note_v4(without_guard, mask, touch_kind));
    if (with_count > 2 && without_count <= 2) bad_position |= mask;
  }
  out[start_kind ? 3 : 2] = static_cast<long long>(bad_position);

  }

  const CurrentStateV4 previous = causal_state_v4(
      head_slot - 1, context, planes, plane_offsets, origins, counts, actions,
      action_masks_by_view, action_count, action_floating, carry_indices,
      path_offsets, path_masks_by_view, mask_count, proportions);
  unsigned long long same_slot_history = 0;
  // Earlier canonical source rows only: never reveal current/future GT.
  for (long long prior_source = opportunity[5] - 1; prior_source >= 0; --prior_source) {
    if (prior_source >= source_count) continue;
    const long long index = source_action[source_offsets[chart] + prior_source];
    if (index < 0) continue;
    const ActionRow prior = actions.row(index);
    if (prior[8] < head_slot) break;
    if (prior[0] >= context.moment || prior[8] != head_slot || prior[2] != kEventNote) continue;
    if ((prior[3] == kKindHold || prior[3] == kKindTouchHold) && prior[9] <= head_slot) continue;
    same_slot_history |= static_cast<unsigned long long>(
        action_masks_by_view[static_cast<long long>(context.view) * action_count + index]);
  }
  const unsigned long long overlap =
      (previous.press | previous.touch | prefix.press | prefix.touch | same_slot_history) & kActiveMask;
  out[start_kind ? 5 : 4] = static_cast<long long>(overlap & (start_kind ? 0xFFULL : kActiveMask));
  const unsigned long long entries = historical_entry_v4(
      head_slot, head_tick, context, planes, plane_offsets, origins, counts,
      actions, action_masks_by_view, action_count, action_floating, carry_indices,
      path_offsets, path_masks_by_view, mask_count, proportions, critical,
      point_ticks, point_seconds, point_bpms, point_padding, timing_width);
  out[start_kind ? 7 : 6] = static_cast<long long>(
      entries & (start_kind ? 0xFFULL : kActiveMask));

  if (geometry_only) return;

  const unsigned long long target_mask = static_cast<unsigned long long>(
      action_masks_by_view[static_cast<long long>(context.view) * action_count + target_index]);
  if (prefix_notes > 0 && base_count >= 2 &&
      hand_count_v4(add_note_v4(state, target_mask, touch_kind)) > 2) {
    if ((state.touch & kActiveMask) != 0) {
      out[1] |= (1LL << 0) | (1LL << kKindHold) |
          (1LL << 4) | (1LL << 5);
    } else {
      out[0] = 1;
    }
  }
  if (kind == kKindHold || kind == kKindTouchHold) {
    // Nothing decoder-visible can create a new collision after every carry and
    // current-moment prefix action has ended.  Bounding the scan here avoids a
    // per-hold walk to the end of the song while preserving the first bad slot.
    long long dynamic_stop = head_slot + 1;
    for (long long cursor = context.carry_begin; cursor < context.carry_end; ++cursor) {
      const long long index = carry_indices[cursor];
      const Geometry geometry = static_geometry(
          actions.row(index), action_floating + index * 4);
      dynamic_stop = ll_max(dynamic_stop, ll_max(geometry.hold_end,
                                                 geometry.slide_end + 1));
    }
    for (long long cursor = 0; cursor < context.prefix_count; ++cursor) {
      const long long index = context.prefix[cursor];
      const Geometry geometry = rebased_geometry(
          actions.row(index), head_tick, chart, point_ticks, point_seconds,
          point_bpms, point_padding, timing_width, critical);
      dynamic_stop = ll_max(dynamic_stop, ll_max(geometry.hold_end,
                                                 geometry.slide_end + 1));
    }
    for (long long slot = head_slot + 1; slot < dynamic_stop; ++slot) {
      const CurrentStateV4 future = or_state_v4(
          causal_state_v4(slot, context, planes, plane_offsets, origins, counts,
                          actions, action_masks_by_view, action_count,
                          action_floating, carry_indices, path_offsets,
                          path_masks_by_view, mask_count, proportions),
          prefix_state_v4(slot, head_tick, context, actions,
                          action_masks_by_view, action_count, path_offsets,
                          path_masks_by_view, mask_count, proportions, critical,
                          point_ticks, point_seconds, point_bpms, point_padding,
                          timing_width));
      const int before = hand_count_v4(future);
      const int after = hand_count_v4(add_note_v4(future, target_mask, touch_kind));
      if (before <= 2 && after > 2) {
        out[8] = slot;
        break;
      }
    }
  }

  if ((kind == 4 || kind == 5) && target_path >= 0) {
    const long long slot = target_geometry.launch + 1;
    CurrentStateV4 future = or_state_v4(
        causal_state_v4(slot, context, planes, plane_offsets, origins, counts,
                        actions, action_masks_by_view, action_count,
                        action_floating, carry_indices, path_offsets,
                        path_masks_by_view, mask_count, proportions),
        prefix_state_v4(slot, head_tick, context, actions,
                        action_masks_by_view, action_count, path_offsets,
                        path_masks_by_view, mask_count, proportions, critical,
                        point_ticks, point_seconds, point_bpms, point_padding,
                        timing_width));
    const int before = hand_count_v4(future);
    const long long* masks = path_masks_by_view +
        static_cast<long long>(context.view) * mask_count;
    future.occupancy |= action_state(
        target, target_mask, target_geometry, slot, kOccupancy,
        path_offsets, masks, proportions);
    if (before <= 2 && hand_count_v4(future) > 2) out[1] |= 1LL << kind;
  }
}

}  // namespace

torch::Tensor physical_action_mask_views_cuda_v2(
    torch::Tensor canonical_masks,
    torch::Tensor lane_lut) {
  const c10::cuda::CUDAGuard guard(canonical_masks.device());
  const long long count = canonical_masks.numel();
  auto output = torch::empty({4, count}, canonical_masks.options());
  if (count > 0) {
    const int threads = 256;
    const int blocks = static_cast<int>((count * 4 + threads - 1) / threads);
    action_mask_views_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned long long*>(canonical_masks.const_data_ptr()),
        reinterpret_cast<unsigned long long*>(output.mutable_data_ptr()), count,
        static_cast<const unsigned char*>(lane_lut.const_data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

void physical_materialize_planes_cuda_v2(
    torch::Tensor canonical_plane_low_bytes,
    torch::Tensor canonical_plane_high_bits,
    torch::Tensor canonical_plane_offsets,
    torch::Tensor output_planes,
    torch::Tensor output_plane_offsets,
    torch::Tensor pair_charts,
    torch::Tensor pair_views,
    torch::Tensor lane_lut) {
  const c10::cuda::CUDAGuard guard(canonical_plane_low_bytes.device());
  const int pairs = static_cast<int>(pair_charts.numel());
  if (pairs > 0) {
    const long long canonical_value_count = canonical_plane_low_bytes.numel() / 5;
    materialize_planes_kernel<<<pairs, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<const unsigned char*>(canonical_plane_low_bytes.const_data_ptr()),
        static_cast<const unsigned char*>(canonical_plane_high_bits.const_data_ptr()),
        canonical_value_count,
        static_cast<const long long*>(canonical_plane_offsets.const_data_ptr()),
        reinterpret_cast<unsigned long long*>(output_planes.mutable_data_ptr()),
        static_cast<const long long*>(output_plane_offsets.const_data_ptr()),
        static_cast<const long long*>(pair_charts.const_data_ptr()),
        static_cast<const long long*>(pair_views.const_data_ptr()),
        static_cast<const unsigned char*>(lane_lut.const_data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

std::vector<torch::Tensor> physical_analyze_resident_cuda_v4(
    torch::Tensor planes,
    torch::Tensor plane_offsets,
    torch::Tensor slot_origins,
    torch::Tensor slot_counts,
    torch::Tensor action_values,
    torch::Tensor action_masks_by_view,
    torch::Tensor action_floating,
    torch::Tensor source_action,
    torch::Tensor source_offsets,
    torch::Tensor moment_slots,
    torch::Tensor moment_offsets,
    torch::Tensor carry_offsets,
    torch::Tensor carry_indices,
    torch::Tensor opportunity_values,
    torch::Tensor input_sources,
    torch::Tensor local_blocks,
    torch::Tensor batch_charts,
    torch::Tensor batch_plane_charts,
    torch::Tensor batch_views,
    torch::Tensor delta_query_slots,
    torch::Tensor delta_opportunities,
    torch::Tensor point_ticks,
    torch::Tensor point_seconds,
    torch::Tensor point_bpms,
    torch::Tensor point_padding,
    torch::Tensor path_offsets,
    torch::Tensor path_masks_by_view,
    torch::Tensor proportions,
    torch::Tensor critical,
    torch::Tensor flags,
    torch::Tensor path_ids_by_view, bool geometry_only) {
  const c10::cuda::CUDAGuard guard(planes.device());
  const long long count = opportunity_values.size(0);
  const long long delta_width = delta_query_slots.size(1);
  const long long query_count = delta_opportunities.size(0);
  auto illegal = torch::empty({count, kIllegalWidthV4},
                              planes.options().dtype(torch::kInt64));
  auto active_paths = torch::empty({count, kPathWordsV4},
                                   planes.options().dtype(torch::kInt64));
  auto delta_bad = torch::empty({query_count, delta_width},
                                planes.options().dtype(torch::kBool));
  if (count > 0) {
    const int threads = 16;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    analyze_v4_kernel<false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned long long*>(planes.const_data_ptr()),
        static_cast<const long long*>(plane_offsets.const_data_ptr()),
        static_cast<const long long*>(slot_origins.const_data_ptr()),
        static_cast<const long long*>(slot_counts.const_data_ptr()),
        static_cast<const int*>(action_values.const_data_ptr()),
        static_cast<const long long*>(action_masks_by_view.const_data_ptr()),
        action_values.size(0),
        static_cast<const double*>(action_floating.const_data_ptr()),
        static_cast<const long long*>(source_action.const_data_ptr()),
        static_cast<const long long*>(source_offsets.const_data_ptr()),
        static_cast<const long long*>(moment_slots.const_data_ptr()),
        static_cast<const long long*>(moment_offsets.const_data_ptr()),
        static_cast<const long long*>(carry_offsets.const_data_ptr()),
        static_cast<const long long*>(carry_indices.const_data_ptr()),
        static_cast<const long long*>(opportunity_values.const_data_ptr()),
        static_cast<const long long*>(input_sources.const_data_ptr()),
        static_cast<const long long*>(local_blocks.const_data_ptr()),
        input_sources.size(1),
        static_cast<const long long*>(batch_charts.const_data_ptr()),
        static_cast<const long long*>(batch_plane_charts.const_data_ptr()),
        static_cast<const long long*>(batch_views.const_data_ptr()),
        static_cast<const long long*>(delta_query_slots.const_data_ptr()),
        delta_width,
        static_cast<const double*>(point_ticks.const_data_ptr()),
        static_cast<const double*>(point_seconds.const_data_ptr()),
        static_cast<const double*>(point_bpms.const_data_ptr()),
        static_cast<const bool*>(point_padding.const_data_ptr()),
        point_ticks.size(1),
        static_cast<const long long*>(path_offsets.const_data_ptr()),
        static_cast<const long long*>(path_masks_by_view.const_data_ptr()),
        path_masks_by_view.size(1),
        static_cast<const double*>(proportions.const_data_ptr()),
        static_cast<const double*>(critical.const_data_ptr()),
        static_cast<const long long*>(flags.const_data_ptr()),
        static_cast<const long long*>(path_ids_by_view.const_data_ptr()),
        static_cast<long long*>(illegal.mutable_data_ptr()),
        reinterpret_cast<unsigned long long*>(active_paths.mutable_data_ptr()),
        static_cast<bool*>(delta_bad.mutable_data_ptr()),
        geometry_only,
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (query_count > 0 && !geometry_only) {
    const int threads = 16;
    const int blocks = static_cast<int>((query_count + threads - 1) / threads);
    analyze_v4_kernel<true><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned long long*>(planes.const_data_ptr()),
        static_cast<const long long*>(plane_offsets.const_data_ptr()),
        static_cast<const long long*>(slot_origins.const_data_ptr()),
        static_cast<const long long*>(slot_counts.const_data_ptr()),
        static_cast<const int*>(action_values.const_data_ptr()),
        static_cast<const long long*>(action_masks_by_view.const_data_ptr()),
        action_values.size(0),
        static_cast<const double*>(action_floating.const_data_ptr()),
        static_cast<const long long*>(source_action.const_data_ptr()),
        static_cast<const long long*>(source_offsets.const_data_ptr()),
        static_cast<const long long*>(moment_slots.const_data_ptr()),
        static_cast<const long long*>(moment_offsets.const_data_ptr()),
        static_cast<const long long*>(carry_offsets.const_data_ptr()),
        static_cast<const long long*>(carry_indices.const_data_ptr()),
        static_cast<const long long*>(delta_opportunities.const_data_ptr()),
        static_cast<const long long*>(input_sources.const_data_ptr()),
        static_cast<const long long*>(local_blocks.const_data_ptr()),
        input_sources.size(1),
        static_cast<const long long*>(batch_charts.const_data_ptr()),
        static_cast<const long long*>(batch_plane_charts.const_data_ptr()),
        static_cast<const long long*>(batch_views.const_data_ptr()),
        static_cast<const long long*>(delta_query_slots.const_data_ptr()),
        delta_width,
        static_cast<const double*>(point_ticks.const_data_ptr()),
        static_cast<const double*>(point_seconds.const_data_ptr()),
        static_cast<const double*>(point_bpms.const_data_ptr()),
        static_cast<const bool*>(point_padding.const_data_ptr()),
        point_ticks.size(1),
        static_cast<const long long*>(path_offsets.const_data_ptr()),
        static_cast<const long long*>(path_masks_by_view.const_data_ptr()),
        path_masks_by_view.size(1),
        static_cast<const double*>(proportions.const_data_ptr()),
        static_cast<const double*>(critical.const_data_ptr()),
        static_cast<const long long*>(flags.const_data_ptr()),
        static_cast<const long long*>(path_ids_by_view.const_data_ptr()),
        static_cast<long long*>(illegal.mutable_data_ptr()),
        reinterpret_cast<unsigned long long*>(active_paths.mutable_data_ptr()),
        static_cast<bool*>(delta_bad.mutable_data_ptr()),
        geometry_only,
        query_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {illegal, active_paths, delta_bad};
}
