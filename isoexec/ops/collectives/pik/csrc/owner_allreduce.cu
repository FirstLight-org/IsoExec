// Owner tree all-reduce: pik's fixed leaf tree as ONE launch per row-parallel site.
//
// Arithmetic (the only float-touching code is `reduce_vec`):
//   every output element e is   round_out( tree(op[0], op[1], ..., op[W*C-1]) )
//   where op[r*C + j] = float(plane j of rank r at element e), the tree is the balanced binary tree over
//   W*C operands in ascending index order with the LEFT operand carrying the lower indices (exactly
//   pik.plan.combine_order / codegen.tree_source), every add is an fp32 __fadd_rn (no contraction is
//   possible: there is no multiply), and round_out is either the identity (fp32 root) or one RNE
//   fp32->bf16 conversion (bf16 root). Operands are bf16 LEAVES (C = leaves per rank) or fp32 internal
//   NODES (C = 1, each node the producer's own subtree of the same tree); a node is the same fp32 value
//   wherever it is computed, so leaf wire and node wire are the same expression.
//
// Launch structure (rank-uniform: grid, block, mode are pure functions of (n, world, C, dtypes, env)):
//   mode 0  LOCAL     no barriers; operands are local tensors (NCCL transport, single-GPU simulation).
//   mode 1  ONE-SHOT  barrier; every rank evaluates every element reading all peers; barrier.
//   mode 3  TWO-SHOT PUSH: as mode 2 but owners push roots into every rank's tmp; each rank copies its own.
//   mode 2  TWO-SHOT  barrier; rank r evaluates its owned slice [r*chunk, (r+1)*chunk) and stores the
//                     root into its symmetric `tmp` and its local `out`; barrier; every rank pulls the
//                     other owners' roots from their `tmp` into `out` (an all-gather of FINISHED roots).
//
// Barriers are per-block handshakes over a FIXED, small grid (grid-stride loops), never one per 1024
// elements: block b of every rank walks the same element subsequence in every phase, so block b's
// handshake with block b of each peer orders exactly the data block b exchanges. Thread t < W publishes
// with st.release.sys into rank t's flag array and spins with ld.acquire.sys (a plain load, not an RMW)
// on its own array. Epochs are per-block monotone counters in device memory bumped by the kernel itself,
// so CUDA-graph replays re-run the increment (graph-safe), compared wrap-safely with (int)(v - e) >= 0.
//
// Buffer-reuse safety without double buffering:
//   * input planes: a rank's next producer kernel writes them only after this kernel retired on that
//     rank, which needs every block past barrier 1, which needs every peer block past its reads of them;
//   * tmp (two-shot): a rank overwrites it in phase 1 of call i+1 only after barrier 0 of call i+1, i.e.
//     after every peer has started call i+1 -- stream order means that peer's call i (whose phase 2 read
//     this tmp) has retired.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace pik_owner {

constexpr int kMaxWorld = 16;
constexpr int kRegions = 2;

struct Args {
  const char* in[kMaxWorld];  // rank r's operand planes (plane j at + j*plane_stride elements)
  char* tmp[kMaxWorld];       // rank r's two-shot root staging (root dtype, global element coordinate)
  uint32_t* flags[kMaxWorld]; // rank r's flag array: [kRegions][max_blocks][kMaxWorld]
  uint32_t* epoch;            // LOCAL per-block epoch counters [max_blocks]
  char* out;                  // LOCAL destination (root dtype)
  int64_t n;                  // elements (mode 1/2: global; mode 0: count)
  int64_t plane_stride;       // elements between one rank's operand planes
  int64_t chunk;              // two-shot owned-slice length, a multiple of VEC
  int64_t base;               // mode 0: operand element of out[0]
  int rank;
  int max_blocks;
  int mode;
};

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }  // exact
__device__ __forceinline__ float to_f(float x) { return x; }

template <typename T>
__device__ __forceinline__ T from_f(float x);
template <>
__device__ __forceinline__ float from_f<float>(float x) { return x; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

// L2-coherent loads (ld.global.cg): peer data and tmp are produced by other agents; never serve them from L1.
template <typename T, int VEC>
__device__ __forceinline__ void load_pack(const T* p, T (&d)[VEC]) {
  constexpr int B = int(sizeof(T)) * VEC;
  if constexpr (B % 16 == 0) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < B / 16; ++i) {
      uint4 u = __ldcg(q + i);
      memcpy(reinterpret_cast<char*>(d) + 16 * i, &u, 16);
    }
  } else if constexpr (sizeof(T) == 4) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      float f = __ldcg(reinterpret_cast<const float*>(p) + i);
      memcpy(&d[i], &f, 4);
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      unsigned short s = __ldcg(reinterpret_cast<const unsigned short*>(p) + i);
      memcpy(&d[i], &s, 2);
    }
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void store_pack(T* p, const T (&d)[VEC]) {
  constexpr int B = int(sizeof(T)) * VEC;
  if constexpr (B % 16 == 0) {
    uint4* q = reinterpret_cast<uint4*>(p);
#pragma unroll
    for (int i = 0; i < B / 16; ++i) {
      uint4 u;
      memcpy(&u, reinterpret_cast<const char*>(d) + 16 * i, 16);
      q[i] = u;
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) p[i] = d[i];
  }
}

// The fixed tree: balanced, ascending, left operand = lower indices. Same recursion as codegen._emit.
template <int LO, int HI, int N>
__device__ __forceinline__ float tree(const float (&v)[N]) {
  if constexpr (HI - LO == 1) {
    return v[LO];
  } else {
    constexpr int MID = (LO + HI) / 2;
    const float a = tree<LO, MID, N>(v);
    const float b = tree<MID, HI, N>(v);
    return __fadd_rn(a, b);
  }
}

// THE arithmetic. Every mode calls exactly this for every element it produces.
template <int W, int C, typename Tin, typename Tout, int VEC>
__device__ __forceinline__ void reduce_vec(const Args& a, int64_t e, Tout (&res)[VEC]) {
  constexpr int NOP = W * C;
  Tin raw[NOP][VEC];
#pragma unroll
  for (int r = 0; r < W; ++r) {
#pragma unroll
    for (int j = 0; j < C; ++j) {
      load_pack<Tin, VEC>(reinterpret_cast<const Tin*>(a.in[r]) + j * a.plane_stride + e, raw[r * C + j]);
    }
  }
#pragma unroll
  for (int l = 0; l < VEC; ++l) {
    float v[NOP];
#pragma unroll
    for (int i = 0; i < NOP; ++i) v[i] = to_f(raw[i][l]);
    res[l] = from_f<Tout>(tree<0, NOP, NOP>(v));
  }
}

template <int W>
__device__ __forceinline__ void block_barrier(const Args& a, int region, uint32_t e) {
  __syncthreads();  // every thread's prior accesses precede the release below (cumulativity)
  if (threadIdx.x < W) {
    const int64_t row = (int64_t(region) * a.max_blocks + blockIdx.x) * kMaxWorld;
    st_release_sys(a.flags[threadIdx.x] + row + a.rank, e);
    const uint32_t* mine = a.flags[a.rank] + row + threadIdx.x;
    while (int32_t(ld_acquire_sys(mine) - e) < 0) {
    }
  }
  __syncthreads();  // every thread's later accesses follow the acquires above
}

template <int W, int C, typename Tin, typename Tout, int VEC>
__global__ void __launch_bounds__(512) owner_ar_kernel(const __grid_constant__ Args a) {
  const int64_t gstride = int64_t(gridDim.x) * blockDim.x;
  const int64_t t0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  Tout* out = reinterpret_cast<Tout*>(a.out);

  if (a.mode == 0) {
    const int64_t nv = a.n / VEC;  // host guarantees VEC | n
    for (int64_t v = t0; v < nv; v += gstride) {
      Tout r[VEC];
      reduce_vec<W, C, Tin, Tout, VEC>(a, a.base + v * VEC, r);
      store_pack<Tout, VEC>(out + v * VEC, r);
    }
    return;
  }

  const uint32_t e = a.epoch[blockIdx.x] + 1u;  // read by every thread before thread 0 bumps it (after a barrier)
  block_barrier<W>(a, 0, e);                     // every peer's operand planes are complete and visible

  if (a.mode == 1) {
    const int64_t nv = a.n / VEC;
    for (int64_t v = t0; v < nv; v += gstride) {
      Tout r[VEC];
      reduce_vec<W, C, Tin, Tout, VEC>(a, v * VEC, r);
      store_pack<Tout, VEC>(out + v * VEC, r);
    }
    block_barrier<W>(a, 1, e);  // no peer still reads this rank's planes once any rank retires
  } else if (a.mode == 3) {
    // TWO-SHOT PUSH (the legacy schedule): the owner stores each finished root into EVERY rank's tmp
    // (remote stores are fire-and-forget over NVLink), barrier, then each rank copies its own tmp to out.
    // Block b's barrier releases exactly the roots block b pushed, and block b copies exactly those.
    const int64_t cv = a.chunk / VEC;
    const int64_t lo = int64_t(a.rank) * a.chunk;
    for (int64_t v = t0; v < cv; v += gstride) {
      const int64_t g = lo + v * VEC;
      if (g < a.n) {
        Tout r[VEC];
        reduce_vec<W, C, Tin, Tout, VEC>(a, g, r);
#pragma unroll
        for (int p = 0; p < W; ++p) store_pack<Tout, VEC>(reinterpret_cast<Tout*>(a.tmp[p]) + g, r);
        store_pack<Tout, VEC>(out + g, r);
      }
    }
    block_barrier<W>(a, 1, e);
    const Tout* mine = reinterpret_cast<const Tout*>(a.tmp[a.rank]);
    for (int64_t v = t0; v < cv; v += gstride) {
      Tout got[W][VEC];
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const int64_t g = int64_t(r) * a.chunk + v * VEC;
        if (r != a.rank && g < a.n) load_pack<Tout, VEC>(mine + g, got[r]);
      }
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const int64_t g = int64_t(r) * a.chunk + v * VEC;
        if (r != a.rank && g < a.n) store_pack<Tout, VEC>(out + g, got[r]);
      }
    }
  } else {
    const int64_t cv = a.chunk / VEC;
    Tout* mine = reinterpret_cast<Tout*>(a.tmp[a.rank]);
    const int64_t lo = int64_t(a.rank) * a.chunk;
    for (int64_t v = t0; v < cv; v += gstride) {
      const int64_t g = lo + v * VEC;
      if (g < a.n) {
        Tout r[VEC];
        reduce_vec<W, C, Tin, Tout, VEC>(a, g, r);
        store_pack<Tout, VEC>(mine + g, r);
        store_pack<Tout, VEC>(out + g, r);
      }
    }
    block_barrier<W>(a, 1, e);  // owned roots published; every peer is done reading this rank's planes
    for (int64_t v = t0; v < cv; v += gstride) {
      Tout got[W][VEC];
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const int64_t g = int64_t(r) * a.chunk + v * VEC;
        if (r != a.rank && g < a.n) load_pack<Tout, VEC>(reinterpret_cast<const Tout*>(a.tmp[r]) + g, got[r]);
      }
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const int64_t g = int64_t(r) * a.chunk + v * VEC;
        if (r != a.rank && g < a.n) store_pack<Tout, VEC>(out + g, got[r]);
      }
    }
  }
  if (threadIdx.x == 0) a.epoch[blockIdx.x] = e;
}

template <int W, int C, typename Tin, typename Tout>
void launch_vec(const Args& a, int vec, int blocks, int threads, cudaStream_t s) {
  if (vec == 8) {
    owner_ar_kernel<W, C, Tin, Tout, 8><<<blocks, threads, 0, s>>>(a);
  } else {
    TORCH_CHECK(vec == 1, "pik owner all-reduce: vec must be 1 or 8, got ", vec);
    owner_ar_kernel<W, C, Tin, Tout, 1><<<blocks, threads, 0, s>>>(a);
  }
}

template <int W, int C, typename Tin>
void launch_out(const Args& a, bool out_bf16, int vec, int blocks, int threads, cudaStream_t s) {
  if (out_bf16) {
    launch_vec<W, C, Tin, __nv_bfloat16>(a, vec, blocks, threads, s);
  } else {
    launch_vec<W, C, Tin, float>(a, vec, blocks, threads, s);
  }
}

template <int W>
void launch_world(const Args& a, int C, bool in_bf16, bool out_bf16, int vec, int blocks, int threads,
                  cudaStream_t s) {
  if (!in_bf16) {
    TORCH_CHECK(C == 1, "pik owner all-reduce: fp32 operands are internal nodes, one plane per rank");
    launch_out<W, 1, float>(a, out_bf16, vec, blocks, threads, s);
    return;
  }
  switch (C) {
    case 1: launch_out<W, 1, __nv_bfloat16>(a, out_bf16, vec, blocks, threads, s); return;
    case 2: launch_out<W, 2, __nv_bfloat16>(a, out_bf16, vec, blocks, threads, s); return;
    case 4: launch_out<W, 4, __nv_bfloat16>(a, out_bf16, vec, blocks, threads, s); return;
    case 8:
      if constexpr (W <= 8) {
        launch_out<W, 8, __nv_bfloat16>(a, out_bf16, vec, blocks, threads, s);
        return;
      }
      break;
    default: break;
  }
  TORCH_CHECK(false, "pik owner all-reduce: unsupported (world, planes) = (", W, ", ", C, ")");
}

void owner_allreduce(std::vector<int64_t> in_ptrs, std::vector<int64_t> tmp_ptrs, std::vector<int64_t> flag_ptrs,
                     c10::optional<torch::Tensor> epoch, torch::Tensor out, int64_t n, int64_t plane_stride,
                     int64_t chunk, int64_t base, int64_t rank, int64_t max_blocks, int64_t mode, int64_t planes,
                     bool in_bf16, int64_t vec, int64_t blocks, int64_t threads) {
  const int W = int(in_ptrs.size());
  TORCH_CHECK(W >= 1 && W <= kMaxWorld, "pik owner all-reduce: world out of range: ", W);
  TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be a contiguous CUDA tensor");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat32,
              "root dtype must be bf16 or fp32");
  TORCH_CHECK(mode >= 0 && mode <= 3, "mode must be 0..3");
  TORCH_CHECK(blocks >= 1 && threads >= 32 && threads <= 512 && threads % 32 == 0, "bad launch geometry");
  TORCH_CHECK(threads >= W, "a block needs at least `world` threads for its barrier lanes");
  TORCH_CHECK(n >= 0 && n % vec == 0 && plane_stride % vec == 0 && chunk % vec == 0 && base % vec == 0,
              "vec must divide n, plane_stride, chunk and base");
  Args a;
  std::memset(&a, 0, sizeof(a));
  for (int r = 0; r < W; ++r) a.in[r] = reinterpret_cast<const char*>(in_ptrs[r]);
  if (mode != 0) {
    TORCH_CHECK(int(flag_ptrs.size()) == W, "one flag array per rank");
    TORCH_CHECK(epoch.has_value() && epoch->is_cuda(), "barrier modes need the local epoch tensor");
    TORCH_CHECK(blocks <= max_blocks && epoch->numel() >= max_blocks, "grid exceeds the flag capacity");
    TORCH_CHECK(rank >= 0 && rank < W, "rank out of range");
    for (int r = 0; r < W; ++r) a.flags[r] = reinterpret_cast<uint32_t*>(flag_ptrs[r]);
    a.epoch = reinterpret_cast<uint32_t*>(epoch->data_ptr());
    if (mode >= 2) {
      TORCH_CHECK(int(tmp_ptrs.size()) == W, "one tmp buffer per rank");
      for (int r = 0; r < W; ++r) a.tmp[r] = reinterpret_cast<char*>(tmp_ptrs[r]);
    }
  }
  a.out = reinterpret_cast<char*>(out.data_ptr());
  a.n = n;
  a.plane_stride = plane_stride;
  a.chunk = chunk;
  a.base = base;
  a.rank = int(rank);
  a.max_blocks = int(max_blocks);
  a.mode = int(mode);
  if (n == 0 && mode == 0) return;

  const at::cuda::OptionalCUDAGuard guard(out.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const bool out_bf16 = out.scalar_type() == torch::kBFloat16;
  const int C = int(planes);
  switch (W) {
    case 1: launch_world<1>(a, C, in_bf16, out_bf16, int(vec), int(blocks), int(threads), s); break;
    case 2: launch_world<2>(a, C, in_bf16, out_bf16, int(vec), int(blocks), int(threads), s); break;
    case 4: launch_world<4>(a, C, in_bf16, out_bf16, int(vec), int(blocks), int(threads), s); break;
    case 8: launch_world<8>(a, C, in_bf16, out_bf16, int(vec), int(blocks), int(threads), s); break;
    case 16: launch_world<16>(a, C, in_bf16, out_bf16, int(vec), int(blocks), int(threads), s); break;
    default: TORCH_CHECK(false, "pik owner all-reduce: world must be a power of two <= 16, got ", W);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pik_owner

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("owner_allreduce", &pik_owner::owner_allreduce, "pik owner tree all-reduce (one launch)",
        py::arg("in_ptrs"), py::arg("tmp_ptrs"), py::arg("flag_ptrs"), py::arg("epoch"), py::arg("out"),
        py::arg("n"), py::arg("plane_stride"), py::arg("chunk"), py::arg("base"), py::arg("rank"),
        py::arg("max_blocks"), py::arg("mode"), py::arg("planes"), py::arg("in_bf16"), py::arg("vec"),
        py::arg("blocks"), py::arg("threads"));
}
