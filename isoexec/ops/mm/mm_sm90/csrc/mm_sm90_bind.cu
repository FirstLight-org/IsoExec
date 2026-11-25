// Torch binding of the sm90 segment-tree GEMM: the configuration table, one launch entry point, and
// the combine kernel that evaluates the remaining tree levels of a split execution.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdlib>
#include <string>

#include "mm_sm90_config.cuh"
#include "mm_sm90_table.cuh"

ISOEXEC_MM_SM90_ALL(ISOEXEC_MM_SM90_DECLARE)

namespace isoexec_mm_sm90 {

struct Entry {
  ConfigInfo info;
  RunFn run;
  int (*stages)();
};

#define ISOEXEC_MM_SM90_ENTRY(name, BM, BN, CM, CN, S, ED, F32, LV, SW, ST) \
  Entry{ConfigInfo{#name, BM, BN, 64, CM, CN, ISOEXEC_MM_SM90_SN_##S, F32, LV, 0, SW}, &run_##name, &stages_##name},

static const Entry kTable[] = {ISOEXEC_MM_SM90_ALL(ISOEXEC_MM_SM90_ENTRY)};
static constexpr int kNumConfigs = sizeof(kTable) / sizeof(kTable[0]);

// ISOEXEC_MM_SM90_PDL (default on): launch with programmatic stream serialization, so the GEMM's prologue
// overlaps its predecessor's tail and a PDL-aware successor (the pik owner reduction) can be scheduled before
// the GEMM retires. Schedule only: every kernel schedule calls griddepcontrol.wait before its first global
// access (CUTLASS_ENABLE_GDC_FOR_SM90), so no operand is read or written early.
static bool pdl_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("ISOEXEC_MM_SM90_PDL");
    if (e == nullptr || *e == '\0') return true;
    const std::string v(e);
    return !(v == "0" || v == "false" || v == "no" || v == "off");
  }();
  return on;
}

static int device_sm_count() {
  int dev = 0, n = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n;
}

// ------------------------------------------------------------------------------------------ combine
// out[e] = tree over s of p[s][e]: the balanced adjacent-pair tree (pik.plan.combine_order), every add
// fp32, left operand the lower segment index; bf16 inputs widened exactly; one optional bf16 round.
template <int S, typename TIn, typename TOut>
__global__ void combine_kernel(const TIn* __restrict__ p, TOut* __restrict__ out, int64_t n, int64_t plane) {
  int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) return;
  float v[S];
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if constexpr (std::is_same_v<TIn, float>) {
      v[s] = p[s * plane + e];
    } else {
      v[s] = __bfloat162float(p[s * plane + e]);
    }
  }
#pragma unroll
  for (int stride = 1; stride < S; stride *= 2) {
#pragma unroll
    for (int i = 0; i < S; i += 2 * stride) v[i] = __fadd_rn(v[i], v[i + stride]);
  }
  if constexpr (std::is_same_v<TOut, float>) {
    out[e] = v[0];
  } else {
    out[e] = __float2bfloat16_rn(v[0]);
  }
}

template <typename TIn, typename TOut>
static void launch_combine(const void* p, void* out, int S, int64_t n, int64_t plane, cudaStream_t st) {
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  auto P = static_cast<const TIn*>(p);
  auto O = static_cast<TOut*>(out);
  switch (S) {
    case 1: combine_kernel<1><<<blocks, threads, 0, st>>>(P, O, n, plane); break;
    case 2: combine_kernel<2><<<blocks, threads, 0, st>>>(P, O, n, plane); break;
    case 4: combine_kernel<4><<<blocks, threads, 0, st>>>(P, O, n, plane); break;
    case 8: combine_kernel<8><<<blocks, threads, 0, st>>>(P, O, n, plane); break;
    default: TORCH_CHECK(false, "combine: S must be 1, 2, 4 or 8, not ", S);
  }
}

}  // namespace isoexec_mm_sm90

using namespace isoexec_mm_sm90;

static int64_t num_configs() { return kNumConfigs; }

static py::dict config_info(int64_t i) {
  TORCH_CHECK(i >= 0 && i < kNumConfigs, "config index out of range");
  auto const& e = kTable[i];
  py::dict d;
  d["name"] = e.info.name;
  d["block_m"] = e.info.block_m;
  d["block_n"] = e.info.block_n;
  d["block_k"] = e.info.block_k;
  d["cluster_m"] = e.info.cluster_m;
  d["cluster_n"] = e.info.cluster_n;
  d["schedule"] = e.info.schedule;
  d["out_fp32"] = e.info.out_fp32;
  d["levels"] = e.info.levels;
  d["swap"] = e.info.swap;
  d["stages"] = e.stages();
  return d;
}

// d is written in place. a: [M, >= L*kseg] bf16 with unit inner stride; w: [N, >= L*kseg] bf16 likewise;
// batch l reads columns [l*kseg_off, ...) where kseg_off = a_batch / b_batch elements.
static void run(int64_t cid, torch::Tensor const& a, torch::Tensor const& w, torch::Tensor& d, int64_t M,
                int64_t N, int64_t kseg, int64_t L, int64_t ldd, int64_t a_batch, int64_t b_batch, int64_t d_batch,
                int64_t seg_k_tiles, bool round_leaf, int64_t sm_count, int64_t swizzle, int64_t raster) {
  TORCH_CHECK(cid >= 0 && cid < kNumConfigs, "config index out of range");
  auto const& e = kTable[cid];
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 operands only");
  TORCH_CHECK(d.scalar_type() == (e.info.out_fp32 ? at::kFloat : at::kBFloat16), "output dtype does not match ",
              e.info.name);
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1, "unit inner stride required");
  const at::cuda::CUDAGuard guard(a.device());
  LaunchArgs la{a.data_ptr(), w.data_ptr(), d.data_ptr(), static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(kseg), static_cast<int>(L), a.stride(0), w.stride(0), ldd, a_batch, b_batch,
                d_batch, static_cast<int>(seg_k_tiles), round_leaf ? 1 : 0,
                static_cast<int>(sm_count > 0 ? sm_count : device_sm_count()), static_cast<int>(swizzle),
                static_cast<int>(raster), at::cuda::getCurrentCUDAStream().stream(), pdl_enabled() ? 1 : 0};
  int rc = e.run(la);
  TORCH_CHECK(rc == 0, "mm_sm90 config ", e.info.name, " refused the launch (code ", rc, ") for M=", M, " N=", N,
              " kseg=", kseg, " L=", L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void combine(torch::Tensor const& p, torch::Tensor& out, int64_t S, int64_t n, int64_t plane) {
  const at::cuda::CUDAGuard guard(p.device());
  auto st = at::cuda::getCurrentCUDAStream().stream();
  bool in_bf16 = p.scalar_type() == at::kBFloat16;
  bool out_bf16 = out.scalar_type() == at::kBFloat16;
  TORCH_CHECK(in_bf16 || p.scalar_type() == at::kFloat, "combine: fp32 or bf16 partials");
  TORCH_CHECK(out_bf16 || out.scalar_type() == at::kFloat, "combine: fp32 or bf16 out");
  if (n == 0) return;
  if (in_bf16 && out_bf16) launch_combine<__nv_bfloat16, __nv_bfloat16>(p.data_ptr(), out.data_ptr(), S, n, plane, st);
  else if (in_bf16) launch_combine<__nv_bfloat16, float>(p.data_ptr(), out.data_ptr(), S, n, plane, st);
  else if (out_bf16) launch_combine<float, __nv_bfloat16>(p.data_ptr(), out.data_ptr(), S, n, plane, st);
  else launch_combine<float, float>(p.data_ptr(), out.data_ptr(), S, n, plane, st);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_configs", &num_configs);
  m.def("config_info", &config_info);
  m.def("run", &run, py::arg("cid"), py::arg("a"), py::arg("w"), py::arg("d"), py::arg("M"), py::arg("N"),
        py::arg("kseg"), py::arg("L"), py::arg("ldd"), py::arg("a_batch"), py::arg("b_batch"), py::arg("d_batch"),
        py::arg("seg_k_tiles"), py::arg("round_leaf"), py::arg("sm_count"), py::arg("swizzle") = 1,
        py::arg("raster") = 0);
  m.def("combine", &combine);
  m.def("sm_count", &device_sm_count);
}
