#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

namespace {
using I = int64_t;
constexpr int TILE = 256;
void check(const torch::Tensor& x, const torch::Tensor& base, at::ScalarType type) {
  TORCH_CHECK(x.is_cuda() && x.device() == base.device() && x.is_contiguous(), "count tensor must be contiguous on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == type, "count tensor dtype mismatch");
}

// One classification per chart. Conservative exact continuity check; duplicate
// timing entries are allowed but discontinuities never take the affine path.
__global__ void constant_kernel(const double* ticks, const double* secs, const double* bpm,
    const bool* pad, bool* flags, I charts, I width) {
  I c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= charts) return;
  I p = c*width;
  bool same = !pad[p];
  for (I j=1; j<width && !pad[p+j]; ++j) {
    double expected = secs[p] + (ticks[p+j]-ticks[p])*60.0/(bpm[p]*96.0);
    same &= bpm[p+j] == bpm[p] && secs[p+j] == expected;
  }
  flags[c] = same;
}

struct Timing {
  const double *ticks, *secs, *bpm;
  const bool *pad, *constant;
  const I* charts;
  I width;
  __device__ I at(I b, double value, bool seconds) const {
    I p=charts[b]*width;
    if (constant[charts[b]]) return p;
    const double* points=seconds ? secs : ticks;
    // Match the padded timing table's rightmost <= selection.
    I lo=0, hi=width;
    while (lo<hi) {
      I mid=(lo+hi)/2;
      if (!pad[p+mid] && points[p+mid]<=value) lo=mid+1;
      else hi=mid;
    }
    return p + (lo ? lo-1 : 0);
  }
  __device__ double second(I b, double tick) const {
    I p=at(b,tick,false);
    return __dadd_rn(secs[p], __ddiv_rn(__dmul_rn(tick-ticks[p],60.0),bpm[p]*96.0));
  }
  __device__ double tick(I b, double second) const {
    I p=at(b,second,true);
    return ticks[p] + (second-secs[p])*bpm[p]*(96.0/60.0);
  }
};
Timing timing(const std::vector<torch::Tensor>& t, const torch::Tensor& base) {
  TORCH_CHECK(t.size()==6, "count timing requires six tensors");
  for (int i=0;i<3;++i) check(t[i],base,torch::kFloat64);
  check(t[3],base,torch::kBool); check(t[4],base,torch::kBool); check(t[5],base,torch::kInt64);
  TORCH_CHECK(t[0].dim()==2 && t[0].size(1)>0 && t[0].sizes()==t[1].sizes() && t[0].sizes()==t[2].sizes() && t[0].sizes()==t[3].sizes(), "invalid count timing shape");
  TORCH_CHECK(t[4].numel()==t[0].size(0), "invalid constant BPM flags");
  return {t[0].data_ptr<double>(),t[1].data_ptr<double>(),t[2].data_ptr<double>(),t[3].data_ptr<bool>(),t[4].data_ptr<bool>(),t[5].data_ptr<I>(),t[0].size(1)};
}
__global__ void context_kernel(const I* ticks, const I* batches, double* context, I n, Timing t) {
  I i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  double head=t.second(batches[i],ticks[i]);
  context[i*3]=head;
  context[i*3+1]=floor(__dadd_rn(__dmul_rn(head,100.0),1e-7));
  context[i*3+2]=floor(__dadd_rn(__dadd_rn(__dmul_rn(t.second(batches[i],ticks[i]+96),100.0),.5),1e-8));
}
__device__ I endpoint(I tick, I duration, bool slide, I b, const double* ctx, Timing t) {
  if (!slide) {
    double length=fmax(t.second(b,tick+duration)-ctx[0],0.0);
    return I(ctx[1])+max(I(ceil(__dadd_rn(__dmul_rn(length,100.0),-1e-9))),I(1))+1;
  }
  I end=I(floor(__dadd_rn(__dadd_rn(__dmul_rn(t.second(b,tick+96+duration),100.0),.5),1e-8)));
  return max(end,I(ctx[2])+1)+2;
}
__global__ void duration_kernel(const I* ticks, const I* batches, const bool* slides,
    const double* context, const I* bad, const I* lower, const I* upper,
    I* threshold, bool* unavoidable, I n, Timing t) {
  I i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  I lo=lower[i], hi=upper[i]+1, b=batches[i], tick=ticks[i];
  const double* ctx=context+i*3;
  unavoidable[i]=false;
  if (bad[i]==INT_MAX) { threshold[i]=hi; return; }
  bool slide=slides[i];
  if (endpoint(tick,lo,slide,b,ctx,t)>bad[i]) {
    unavoidable[i]=true; threshold[i]=lo; return;
  }
  if (endpoint(tick,upper[i],slide,b,ctx,t)<=bad[i]) { threshold[i]=hi; return; }
  double seconds = slide ? (double(bad[i]-2)+.5-1e-8)/100.0
      : ctx[0]+(double(bad[i]-I(ctx[1])-1)+1e-9)/100.0;
  double raw=t.tick(b,seconds);
  I candidate=slide ? I(ceil(raw))-tick-96 : I(floor(raw))+1-tick;
  candidate=max(lo,min(upper[i],candidate));
  // Exact legacy endpoint verification also handles near-integer inversions.
  if (endpoint(tick,candidate,slide,b,ctx,t)>bad[i] &&
      (candidate==lo || endpoint(tick,candidate-1,slide,b,ctx,t)<=bad[i])) {
    threshold[i]=candidate; return;
  }
  // Rare numerical boundary correction stays inside this one CUDA launch.
  while (lo<hi) {
    I mid=lo+(hi-lo)/2;
    if (endpoint(tick,mid,slide,b,ctx,t)>bad[i]) hi=mid;
    else lo=min(mid+1,hi);
  }
  threshold[i]=lo;
}

__global__ void point_index_kernel(const I* op, const I* reps, const I* delta, const I* heads,
    const I* ranges, const I* offsets, I* locations, I* histogram, I n, I q, I width) {
  I i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n+q*width) return;
  I b,row,slot;
  if (i<n) { b=op[i*11]; row=op[i*11+2]; slot=heads[i]; }
  else { I j=i-n,r=j/width; b=reps[r*11]; row=max(reps[r*11+3],I(0)); slot=delta[j]; }
  I lo=ranges[b*2],hi=ranges[b*2+1];
  locations[i*3]=-1;
  if (slot<lo || slot>=hi || (i>=n && slot<0)) return;
  I tile=offsets[b]+(slot-lo)/TILE;
  locations[i*3]=tile; locations[i*3+1]=row; locations[i*3+2]=(slot-lo)%TILE;
  atomicAdd(reinterpret_cast<unsigned long long*>(histogram+tile),1ULL);
}
__global__ void point_scatter_kernel(const I* loc, I* cursor, I* ids, I n) {
  I i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n || loc[i*3]<0) return;
  I dest=atomicAdd(reinterpret_cast<unsigned long long*>(cursor+loc[i*3]),1ULL);
  ids[dest]=i;
}
__global__ void point_query_kernel(const uint8_t* counts, const I* loc, const I* offsets,
    const I* ids, uint8_t* head, bool* delta, I n, I rows, I begin) {
  I tile=begin+blockIdx.x;
  for (I j=offsets[tile]+threadIdx.x; j<offsets[tile+1]; j+=blockDim.x) {
    I i=ids[j];
    uint8_t value=counts[blockIdx.x*rows*TILE+loc[i*3+1]*TILE+loc[i*3+2]];
    if (i<n) head[i]=value; else delta[i-n]=value>=2;
  }
}
__global__ void first_query_kernel(const uint8_t* counts, const I* tiles, const I* offsets,
    const I* op, const I* starts, I* first, I rows, I begin, I end) {
  I i=blockIdx.x,b=op[i*11],row=op[i*11+2];
  I a=max(offsets[b],begin),z=min(offsets[b+1],end);
  int value=INT_MAX;
  for (I tile=a;tile<z;++tile) {
    I slot=tiles[tile*3+1]+threadIdx.x;
    if (threadIdx.x<tiles[tile*3+2] && slot>=starts[i] &&
        counts[(tile-begin)*rows*TILE+row*TILE+threadIdx.x]>=2)
      value=min(value,int(slot));
  }
  __shared__ int reduction[TILE];
  reduction[threadIdx.x]=value; __syncthreads();
  for (int stride=TILE/2;stride;stride/=2) {
    if (threadIdx.x<stride) reduction[threadIdx.x]=min(reduction[threadIdx.x],reduction[threadIdx.x+stride]);
    __syncthreads();
  }
  if (threadIdx.x==0) first[i]=min(first[i],I(reduction[0]));
}
}

torch::Tensor physical_count_constant_cuda(torch::Tensor ticks, torch::Tensor seconds,
    torch::Tensor bpms, torch::Tensor padding) {
  const c10::cuda::CUDAGuard guard(ticks.device());
  check(ticks,ticks,torch::kFloat64); check(seconds,ticks,torch::kFloat64);
  check(bpms,ticks,torch::kFloat64); check(padding,ticks,torch::kBool);
  TORCH_CHECK(ticks.dim()==2 && ticks.size(1)>0 && ticks.sizes()==seconds.sizes() && ticks.sizes()==bpms.sizes() && ticks.sizes()==padding.sizes(), "invalid timing arrays");
  auto result=torch::empty({ticks.size(0)},padding.options());
  if (ticks.size(0)) constant_kernel<<<(ticks.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<double>(),seconds.data_ptr<double>(),bpms.data_ptr<double>(),padding.data_ptr<bool>(),result.data_ptr<bool>(),ticks.size(0),ticks.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return result;
}
torch::Tensor physical_count_context_cuda(torch::Tensor ticks, torch::Tensor batches, std::vector<torch::Tensor> ts) {
  const c10::cuda::CUDAGuard guard(ticks.device());
  check(ticks,ticks,torch::kInt64); check(batches,ticks,torch::kInt64);
  TORCH_CHECK(ticks.dim()==1 && batches.sizes()==ticks.sizes(), "invalid count context inputs");
  auto t=timing(ts,ticks); auto result=torch::empty({ticks.numel(),3},ticks.options().dtype(torch::kFloat64));
  if (ticks.numel()) context_kernel<<<(ticks.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<I>(),batches.data_ptr<I>(),result.data_ptr<double>(),ticks.numel(),t);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return result;
}
std::vector<torch::Tensor> physical_count_duration_cuda(torch::Tensor ticks, torch::Tensor batches,
    torch::Tensor slides, torch::Tensor context, torch::Tensor bad, torch::Tensor lower,
    torch::Tensor upper, std::vector<torch::Tensor> ts) {
  const c10::cuda::CUDAGuard guard(ticks.device());
  for (auto x:{ticks,batches,bad,lower,upper}) { check(x,ticks,torch::kInt64); TORCH_CHECK(x.dim()==1 && x.numel()==ticks.numel(),"invalid duration vector"); }
  check(slides,ticks,torch::kBool); check(context,ticks,torch::kFloat64);
  TORCH_CHECK(slides.sizes()==ticks.sizes() && context.dim()==2 && context.size(0)==ticks.numel() && context.size(1)==3,"invalid duration context");
  auto t=timing(ts,ticks); auto result=torch::empty_like(ticks), unavoidable=torch::empty_like(slides);
  if (ticks.numel()) duration_kernel<<<(ticks.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<I>(),batches.data_ptr<I>(),slides.data_ptr<bool>(),context.data_ptr<double>(),bad.data_ptr<I>(),lower.data_ptr<I>(),upper.data_ptr<I>(),result.data_ptr<I>(),unavoidable.data_ptr<bool>(),ticks.numel(),t);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {result,unavoidable};
}
std::vector<torch::Tensor> physical_count_index_cuda(torch::Tensor op, torch::Tensor reps,
    torch::Tensor delta, torch::Tensor heads, torch::Tensor ranges, torch::Tensor tile_offsets, I tile_count) {
  const c10::cuda::CUDAGuard guard(op.device());
  for (auto x:{op,reps,delta,heads,ranges,tile_offsets}) check(x,op,torch::kInt64);
  TORCH_CHECK(op.dim()==2 && op.size(1)==11 && reps.dim()==2 && reps.size(1)==11 && delta.dim()==2 && delta.size(0)==reps.size(0) && heads.numel()==op.size(0),"invalid point queries");
  I total=op.size(0)+delta.numel();
  auto loc=torch::empty({total,3},op.options()), hist=torch::zeros({tile_count+1},op.options());
  if (total) point_index_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(op.data_ptr<I>(),reps.data_ptr<I>(),delta.data_ptr<I>(),heads.data_ptr<I>(),ranges.data_ptr<I>(),tile_offsets.data_ptr<I>(),loc.data_ptr<I>(),hist.data_ptr<I>(),op.size(0),reps.size(0),delta.size(1));
  auto offsets=hist.cumsum(0)-hist;
  auto cursor=offsets.clone(),ids=torch::empty({total},op.options());
  if (total) point_scatter_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(loc.data_ptr<I>(),cursor.data_ptr<I>(),ids.data_ptr<I>(),total);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {loc,offsets,ids};
}
void physical_count_query_cuda(torch::Tensor counts, torch::Tensor tiles, torch::Tensor tile_offsets,
    torch::Tensor op, torch::Tensor starts, std::vector<torch::Tensor> index,
    torch::Tensor head, torch::Tensor delta, torch::Tensor first, I begin, I end) {
  const c10::cuda::CUDAGuard guard(op.device());
  check(counts,op,torch::kUInt8); check(head,op,torch::kUInt8); check(delta,op,torch::kBool);
  for (auto x:{tiles,tile_offsets,op,starts,first}) check(x,op,torch::kInt64);
  TORCH_CHECK(index.size()==3,"invalid point index"); for (auto x:index) check(x,op,torch::kInt64);
  TORCH_CHECK(counts.dim()==3 && counts.size(0)==end-begin && counts.size(2)==TILE && begin>=0 && end<=tiles.size(0),"invalid count tile group");
  if (end>begin) point_query_kernel<<<end-begin,256,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<uint8_t>(),index[0].data_ptr<I>(),index[1].data_ptr<I>(),index[2].data_ptr<I>(),head.data_ptr<uint8_t>(),delta.data_ptr<bool>(),op.size(0),counts.size(1),begin);
  if (op.size(0)) first_query_kernel<<<op.size(0),TILE,0,at::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<uint8_t>(),tiles.data_ptr<I>(),tile_offsets.data_ptr<I>(),op.data_ptr<I>(),starts.data_ptr<I>(),first.data_ptr<I>(),counts.size(1),begin,end);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
