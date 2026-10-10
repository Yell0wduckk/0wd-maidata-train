#include <torch/extension.h>

#include <vector>

torch::Tensor physical_action_mask_views_cuda_v2(
    torch::Tensor canonical_masks,
    torch::Tensor lane_lut);

void physical_materialize_planes_cuda_v2(
    torch::Tensor canonical_plane_low_bytes,
    torch::Tensor canonical_plane_high_bits,
    torch::Tensor canonical_plane_offsets,
    torch::Tensor output_planes,
    torch::Tensor output_plane_offsets,
    torch::Tensor pair_charts,
    torch::Tensor pair_views,
    torch::Tensor lane_lut);


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
    torch::Tensor path_enter_proportions,
    torch::Tensor path_critical_proportions,
    torch::Tensor path_flags,
    torch::Tensor path_ids_by_view, bool geometry_only);

namespace {

void require_cuda_contiguous(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void require_int64(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.scalar_type() == torch::kInt64, name, " must be int64");
}

void require_float64(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.scalar_type() == torch::kFloat64, name, " must be float64");
}

}  // namespace

torch::Tensor action_mask_views_v2(
    torch::Tensor canonical_masks,
    torch::Tensor lane_lut) {
  require_cuda_contiguous(canonical_masks, "canonical_masks");
  require_cuda_contiguous(lane_lut, "lane_lut");
  require_int64(canonical_masks, "canonical_masks");
  TORCH_CHECK(canonical_masks.dim() == 1, "canonical_masks must be one-dimensional");
  TORCH_CHECK(lane_lut.scalar_type() == torch::kUInt8 && lane_lut.dim() == 3 &&
                  lane_lut.size(0) == 4 && lane_lut.size(1) == 2 && lane_lut.size(2) == 256,
              "lane_lut must be uint8[4,2,256]");
  return physical_action_mask_views_cuda_v2(canonical_masks, lane_lut);
}

void materialize_planes_v2(
    torch::Tensor canonical_plane_low_bytes,
    torch::Tensor canonical_plane_high_bits,
    torch::Tensor canonical_plane_offsets,
    torch::Tensor output_planes,
    torch::Tensor output_plane_offsets,
    torch::Tensor pair_charts,
    torch::Tensor pair_views,
    torch::Tensor lane_lut) {
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&canonical_plane_low_bytes, "canonical_plane_low_bytes"},
           {&canonical_plane_high_bits, "canonical_plane_high_bits"},
           {&canonical_plane_offsets, "canonical_plane_offsets"},
           {&output_planes, "output_planes"},
           {&output_plane_offsets, "output_plane_offsets"},
           {&pair_charts, "pair_charts"}, {&pair_views, "pair_views"},
           {&lane_lut, "lane_lut"}}) {
    require_cuda_contiguous(*item.first, item.second);
  }
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&canonical_plane_offsets, "canonical_plane_offsets"},
           {&output_planes, "output_planes"},
           {&output_plane_offsets, "output_plane_offsets"},
           {&pair_charts, "pair_charts"}, {&pair_views, "pair_views"}}) {
    require_int64(*item.first, item.second);
  }
  TORCH_CHECK(canonical_plane_low_bytes.scalar_type() == torch::kUInt8 &&
                  canonical_plane_high_bits.scalar_type() == torch::kUInt8,
              "packed canonical planes must be uint8");
  TORCH_CHECK(canonical_plane_low_bytes.numel() % 5 == 0,
              "packed canonical low plane bytes must have five byte planes");
  TORCH_CHECK(pair_charts.sizes() == pair_views.sizes(),
              "pair chart/view tensors mismatch");
  TORCH_CHECK(output_plane_offsets.numel() == pair_charts.numel() + 1,
              "output plane offsets mismatch");
  TORCH_CHECK(lane_lut.scalar_type() == torch::kUInt8 && lane_lut.dim() == 3 &&
                  lane_lut.size(0) == 4 && lane_lut.size(1) == 2 && lane_lut.size(2) == 256,
              "lane_lut must be uint8[4,2,256]");
  physical_materialize_planes_cuda_v2(
      canonical_plane_low_bytes, canonical_plane_high_bits,
      canonical_plane_offsets, output_planes,
      output_plane_offsets, pair_charts, pair_views, lane_lut);
}

std::vector<torch::Tensor> analyze_resident_v4(
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
    torch::Tensor path_enter_proportions,
    torch::Tensor path_critical_proportions,
    torch::Tensor path_flags,
    torch::Tensor path_ids_by_view, bool geometry_only) {
  const std::vector<std::pair<const torch::Tensor*, const char*>> tensors = {
      {&planes, "planes"}, {&plane_offsets, "plane_offsets"},
      {&slot_origins, "slot_origins"}, {&slot_counts, "slot_counts"},
      {&action_values, "action_values"},
      {&action_masks_by_view, "action_masks_by_view"},
      {&action_floating, "action_floating"},
      {&source_action, "source_action"}, {&source_offsets, "source_offsets"},
      {&moment_slots, "moment_slots"}, {&moment_offsets, "moment_offsets"},
      {&carry_offsets, "carry_offsets"}, {&carry_indices, "carry_indices"},
      {&opportunity_values, "opportunity_values"},
      {&input_sources, "input_sources"}, {&local_blocks, "local_blocks"},
      {&batch_charts, "batch_charts"},
      {&batch_plane_charts, "batch_plane_charts"},
      {&batch_views, "batch_views"}, {&delta_query_slots, "delta_query_slots"},
      {&delta_opportunities, "delta_opportunities"},
      {&point_ticks, "point_ticks"}, {&point_seconds, "point_seconds"},
      {&point_bpms, "point_bpms"}, {&point_padding, "point_padding"},
      {&path_offsets, "path_offsets"},
      {&path_masks_by_view, "path_masks_by_view"},
      {&path_enter_proportions, "path_enter_proportions"},
      {&path_critical_proportions, "path_critical_proportions"},
      {&path_flags, "path_flags"}, {&path_ids_by_view, "path_ids_by_view"}};
  for (const auto& item : tensors) require_cuda_contiguous(*item.first, item.second);
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&planes, "planes"}, {&plane_offsets, "plane_offsets"},
           {&slot_origins, "slot_origins"}, {&slot_counts, "slot_counts"},
           {&action_masks_by_view, "action_masks_by_view"},
           {&source_action, "source_action"}, {&source_offsets, "source_offsets"},
           {&moment_slots, "moment_slots"}, {&moment_offsets, "moment_offsets"},
           {&carry_offsets, "carry_offsets"}, {&carry_indices, "carry_indices"},
           {&opportunity_values, "opportunity_values"},
           {&input_sources, "input_sources"}, {&local_blocks, "local_blocks"},
           {&batch_charts, "batch_charts"},
           {&batch_plane_charts, "batch_plane_charts"},
           {&batch_views, "batch_views"}, {&delta_query_slots, "delta_query_slots"},
      {&delta_opportunities, "delta_opportunities"},
           {&path_offsets, "path_offsets"}, {&path_masks_by_view, "path_masks_by_view"},
           {&path_flags, "path_flags"}, {&path_ids_by_view, "path_ids_by_view"}}) {
    require_int64(*item.first, item.second);
  }
  TORCH_CHECK(action_values.scalar_type() == torch::kInt32,
              "resident action_values must be int32");
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&action_floating, "action_floating"}, {&point_ticks, "point_ticks"},
           {&point_seconds, "point_seconds"}, {&point_bpms, "point_bpms"},
           {&path_enter_proportions, "path_enter_proportions"},
           {&path_critical_proportions, "path_critical_proportions"}}) {
    require_float64(*item.first, item.second);
  }
  TORCH_CHECK(point_padding.scalar_type() == torch::kBool,
              "point_padding must be bool");
  TORCH_CHECK(opportunity_values.dim() == 2 && opportunity_values.size(1) == 11,
              "invalid v4 opportunity layout");
  TORCH_CHECK(delta_query_slots.dim() == 2 &&
                  delta_opportunities.dim() == 2 && delta_opportunities.size(1) == 11 &&
                  delta_query_slots.size(0) == delta_opportunities.size(0),
              "invalid v4 delta query layout");
  TORCH_CHECK(path_ids_by_view.dim() == 2 && path_ids_by_view.size(0) == 4 &&
                  path_ids_by_view.size(1) == 568,
               "path_ids_by_view must be int64[4,568]");
  TORCH_CHECK(input_sources.sizes() == local_blocks.sizes(),
              "input_sources/local_blocks mismatch");
  return physical_analyze_resident_cuda_v4(
      planes, plane_offsets, slot_origins, slot_counts, action_values,
      action_masks_by_view, action_floating, source_action, source_offsets,
      moment_slots, moment_offsets, carry_offsets, carry_indices,
      opportunity_values, input_sources, local_blocks, batch_charts,
      batch_plane_charts, batch_views, delta_query_slots, delta_opportunities, point_ticks,
      point_seconds, point_bpms, point_padding, path_offsets,
      path_masks_by_view, path_enter_proportions,
      path_critical_proportions, path_flags, path_ids_by_view, geometry_only);
}

torch::Tensor physical_count_guard_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor physical_count_constant_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor physical_count_context_cuda(torch::Tensor, torch::Tensor, std::vector<torch::Tensor>);
std::vector<torch::Tensor> physical_count_duration_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>);
std::vector<torch::Tensor> physical_count_index_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void physical_count_query_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("count_constant", &physical_count_constant_cuda);
  module.def("count_context", &physical_count_context_cuda);
  module.def("count_duration", &physical_count_duration_cuda);
  module.def("count_index", &physical_count_index_cuda);
  module.def("count_query", &physical_count_query_cuda);
  module.def("count_guard", &physical_count_guard_cuda);
  module.def("analyze_resident_v4", &analyze_resident_v4);
  module.def("action_mask_views_v2", &action_mask_views_v2);
  module.def("materialize_planes_v2", &materialize_planes_v2);
}
