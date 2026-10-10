#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void guard_kernel(const long long* intervals, const long long* offsets,
    const long long* opportunities, const long long* heads,
    const unsigned char* counts, long long* result, long long n) {
  const long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long b = opportunities[i*11], row = opportunities[i*11+2];
  unsigned long long occupied = 0, guard = 0;
  for (long long j = offsets[b]; j < offsets[b+1]; ++j) {
    const long long* a = intervals + j*6;
    if (a[1] > row || heads[i] < a[2] || heads[i] >= a[3]) continue;
    if (a[5]) guard |= static_cast<unsigned long long>(a[4]);
    else occupied |= static_cast<unsigned long long>(a[4]);
  }
  unsigned long long mask = 0;
  const int base = static_cast<int>(counts[i]) + 1;
  if (base <= 2) {
    for (int pad=0; pad<33; ++pad) {
      if (base + __popcll(guard & ~occupied & ~(1ULL << pad)) > 2)
        mask |= 1ULL << pad;
    }
  }
  result[i] = static_cast<long long>(mask);
}
}

torch::Tensor physical_count_guard_cuda(torch::Tensor intervals, torch::Tensor offsets,
    torch::Tensor opportunities, torch::Tensor heads, torch::Tensor counts) {
  const c10::cuda::CUDAGuard device_guard(opportunities.device());
  for (const auto& x : {intervals, offsets, opportunities, heads}) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kInt64,
                "count guard descriptors must be contiguous CUDA int64");
    TORCH_CHECK(x.device()==opportunities.device(), "count guard device mismatch");
  }
  TORCH_CHECK(intervals.dim()==2 && intervals.size(1)==6, "count guard interval shape");
  TORCH_CHECK(opportunities.dim()==2 && opportunities.size(1)==11, "count guard opportunity shape");
  TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type()==torch::kUInt8 &&
              counts.device()==opportunities.device() && counts.numel()==opportunities.size(0) &&
              heads.numel()==counts.numel(), "count guard count shape/device/type");
  auto result = torch::zeros({opportunities.size(0)}, opportunities.options());
  if (result.numel()) {
    guard_kernel<<<(result.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const long long*>(intervals.const_data_ptr()), static_cast<const long long*>(offsets.const_data_ptr()),
      static_cast<const long long*>(opportunities.const_data_ptr()), static_cast<const long long*>(heads.const_data_ptr()),
      counts.data_ptr<unsigned char>(), static_cast<long long*>(result.mutable_data_ptr()), result.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}
