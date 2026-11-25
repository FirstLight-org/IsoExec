// Owner tree all-reduce, Lamport protocol, with an optional fused residual-add + RMSNorm epilogue.
//
// ARITHMETIC (identical to owner_allreduce.cu's `reduce_vec` and to norms_sm90's `add_rms_fwd`; nothing here
// chooses a rounding):
//   root[e] = round_root( tree(op[0], ..., op[W*C-1]) )      op[r*C + j] = float(plane j of rank r at e)
//     the balanced binary tree over W*C operands in ascending index order, LEFT operand = lower indices, every
//     add an fp32 __fadd_rn (pik.plan.combine_order / codegen.tree_source), round_root = identity (fp32 root)
//     or one RNE fp32->bf16 conversion (bf16 root).
//   EPI_ADD_RMS (bf16 root only), per row of width N = 256*K (norms recipe v1, L = 32 lanes):
//     h[c]   = rn_bf16( add.rn(float(root[c]), float(residual[c])) )
//     lane l chain: acc = +0; for k in [0,K), j in [0,8): acc = add.rn(acc, mul.rn(v, v)), v = float(h[k*256 + 8l + j])
//     butterfly s = 16..1: acc = add.rn(acc, shfl_xor(acc, s));  rs = frsqrt_rn(add.rn(div.rn(acc, N), eps))
//     y[c]   = rn_bf16( mul.rn(mul.rn(v, rs), g) ),  g = w (MODE 1) | add.rn(1, w) (MODE 2)
//   which is exactly what `aten.add` of the bf16 root and the residual followed by `add_rms_fwd` computes.
//
// PROTOCOL (Lamport: the data is the flag). Every receive buffer is pre-filled with an all-ones byte pattern,
// the SENTINEL: bf16 0xFFFF / fp32 0xFFFFFFFF, both NaNs with a payload. A consumer spins on each 16-byte
// vector until no element equals the sentinel; a producer writes data with volatile (relaxed.sys) stores over
// NVLink. Element stores are atomic, so a torn 16-byte vector is simply still-pending elements.
//   * Wire planes (leaves / nodes) may contain any value. A plane element equal to the sentinel is a NaN; it is
//     sent as the canonical NaN instead. Exact: every operand passes through at least one __fadd_rn (W*C >= 2),
//     and an fp32 add with a NaN operand returns the canonical NaN whatever the payload, so the root's bits do
//     not depend on the payload (the battery checks this on raw NaN patterns).
//   * Roots and h are results of __fadd_rn / RNE conversion, whose only NaN is canonical (0x7FFFFFFF,
//     bf16 0x7FFF), never the sentinel: they are sent unmodified.
// Buffers rotate over 3 generations: call k receives into generation k%3 and re-fills generation (k+2)%3 (the
// one call k-1 consumed) with the sentinel. A peer writes generation g again only in call k+2, which it reaches
// only after this rank finished call k+1, so after this kernel (and its re-fill) retired. The generation counter
// and the per-generation extents live in device memory and are advanced by the kernel itself: CUDA-graph
// replays re-run the advance (graph-replay safe), and every rank advances in lockstep because every launch
// decision is rank-uniform.
//
// MODES  0 LOCAL     operands are W local tensors (single-GPU simulation / NCCL-received planes); no protocol.
//        1 ONE-SHOT  every rank pushes all its planes to every peer, every rank reduces every row.
//        2 TWO-SHOT  row r is owned by rank r % W: ranks push the owner's rows to it, the owner reduces (and
//                    joins the residual), pushes the finished root / h row to every rank's root/h area; with the
//                    norm epilogue every rank then normalizes every row from its own area (the norm is per row,
//                    so it is the same program wherever it runs), staging each row in shared memory.
//
// PDL: launched with programmatic stream serialization; nothing is read or written before griddepcontrol.wait,
// and the kernel triggers its dependents once its data movement is done (their prologue overlaps its tail).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace pik_lamport {

constexpr int kMaxWorld = 16;
constexpr int kGens = 3;
constexpr int kLanes = 32;  // norms recipe v1 lanes per row for N % 256 == 0

enum Epi : int { EPI_ROOT = 0, EPI_ADD_RMS = 1 };

struct Args {
  const char* in[kMaxWorld];  // mode 0: rank r's planes; modes 1/2: in[0] = this rank's LOCAL planes
  char* recv1[kMaxWorld];     // rank r's wire receive area (3 generations of gen1 bytes)
  char* recv2[kMaxWorld];     // rank r's root / h receive area (3 generations of gen2 bytes)
  unsigned long long* ctl;    // LOCAL: [0] generation counter, [1] retired blocks, [2..4] ext1, [5..7] ext2
  const __nv_bfloat16* resid; // EPI_ADD_RMS: residual [M, N], row stride sr
  const void* w;              // EPI_ADD_RMS: gamma [N], bf16 or fp32
  char* out;                  // root [M, N] (EPI_ROOT) or h [M, N] (EPI_ADD_RMS), contiguous
  __nv_bfloat16* y;           // EPI_ADD_RMS: normed [M, N], contiguous
  int64_t M, N;
  int64_t plane_stride;  // elements between one rank's planes (mode 0) / this rank's planes (modes 1, 2)
  int64_t sr;
  int64_t gen1, gen2;  // bytes per generation
  int rank, mode, norm_mode, w_fp32;
  float eps;
};

// ---------------------------------------------------------------------------------------- memory primitives

__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_volatile16(void* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// A 16-byte vector still holds (part of) the sentinel. bf16: any halfword 0xFFFF; fp32: any word 0xFFFFFFFF.
template <typename T>
__device__ __forceinline__ bool pending(uint4 v);
template <>
__device__ __forceinline__ bool pending<__nv_bfloat16>(uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  bool p = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) p |= ((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu);
  return p;
}
template <>
__device__ __forceinline__ bool pending<float>(uint4 v) {
  return (v.x == 0xFFFFFFFFu) | (v.y == 0xFFFFFFFFu) | (v.z == 0xFFFFFFFFu) | (v.w == 0xFFFFFFFFu);
}

// Wire element equal to the sentinel -> the canonical NaN (exact: see the header).
template <typename T>
__device__ __forceinline__ uint4 desentinel(uint4 v);
template <>
__device__ __forceinline__ uint4 desentinel<__nv_bfloat16>(uint4 v) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t lo = w[i] & 0xFFFFu, hi = w[i] >> 16;
    lo = lo == 0xFFFFu ? 0x7FFFu : lo;
    hi = hi == 0xFFFFu ? 0x7FFFu : hi;
    w[i] = lo | (hi << 16);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
template <>
__device__ __forceinline__ uint4 desentinel<float>(uint4 v) {
  v.x = v.x == 0xFFFFFFFFu ? 0x7FFFFFFFu : v.x;
  v.y = v.y == 0xFFFFFFFFu ? 0x7FFFFFFFu : v.y;
  v.z = v.z == 0xFFFFFFFFu ? 0x7FFFFFFFu : v.z;
  v.w = v.w == 0xFFFFFFFFu ? 0x7FFFFFFFu : v.w;
  return v;
}

__device__ __forceinline__ uint4 poll16(const void* p, bool bf16) {
  uint4 v;
  do {
    v = ld_volatile16(p);
  } while (bf16 ? pending<__nv_bfloat16>(v) : pending<float>(v));
  return v;
}

// 8 consecutive elements of T as exact fp32 (bf16 widening is exact).
template <typename T>
__device__ __forceinline__ void widen8(float (&f)[8], const uint4 (&u)[sizeof(T) / 2]);
template <>
__device__ __forceinline__ void widen8<__nv_bfloat16>(float (&f)[8], const uint4 (&u)[1]) {
  const uint32_t w[4] = {u[0].x, u[0].y, u[0].z, u[0].w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}
template <>
__device__ __forceinline__ void widen8<float>(float (&f)[8], const uint4 (&u)[2]) {
  f[0] = __uint_as_float(u[0].x); f[1] = __uint_as_float(u[0].y); f[2] = __uint_as_float(u[0].z);
  f[3] = __uint_as_float(u[0].w); f[4] = __uint_as_float(u[1].x); f[5] = __uint_as_float(u[1].y);
  f[6] = __uint_as_float(u[1].z); f[7] = __uint_as_float(u[1].w);
}

// RNE fp32 -> bf16 pair, low = a (cvt.rn.bf16x2.f32: the norms kernels' instruction).
__device__ __forceinline__ uint32_t rn_pair(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}

template <typename T>
__device__ __forceinline__ void pack8(uint4 (&u)[sizeof(T) / 2], const float (&f)[8]);
template <>
__device__ __forceinline__ void pack8<__nv_bfloat16>(uint4 (&u)[1], const float (&f)[8]) {
  u[0] = make_uint4(rn_pair(f[0], f[1]), rn_pair(f[2], f[3]), rn_pair(f[4], f[5]), rn_pair(f[6], f[7]));
}
template <>
__device__ __forceinline__ void pack8<float>(uint4 (&u)[2], const float (&f)[8]) {
  u[0] = make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
  u[1] = make_uint4(__float_as_uint(f[4]), __float_as_uint(f[5]), __float_as_uint(f[6]), __float_as_uint(f[7]));
}

// The fixed tree: balanced, ascending, left operand = lower indices. Same recursion as codegen._emit and
// owner_allreduce.cu's `tree`.
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

// ------------------------------------------------------------------------------------------------ layout

// Rows per sender slot of the wire receive area (two-shot: the owner's rows only; one-shot: all rows).
__host__ __device__ __forceinline__ int64_t slot_rows(int64_t M, int W, int mode) {
  return mode == 2 ? (M + W - 1) / W : M;
}
__host__ __device__ __forceinline__ int64_t align16(int64_t b) { return (b + 15) & ~int64_t(15); }
template <typename Tin, int C>
__host__ __device__ __forceinline__ int64_t slot_bytes(int64_t M, int64_t N, int W, int mode) {
  return align16(slot_rows(M, W, mode) * C * N * int64_t(sizeof(Tin)));
}

// --------------------------------------------------------------------------------------------- the kernel

template <int W, int C, typename Tin, typename Tout, int EPI>
struct Kern {
  static constexpr int NOP = W * C;
  static constexpr int UIN = int(sizeof(Tin)) / 2;   // 16-byte vectors per 8 wire elements
  static constexpr int UOUT = int(sizeof(Tout)) / 2;  // 16-byte vectors per 8 root elements
  static constexpr bool IN_BF16 = sizeof(Tin) == 2;

  // Every operand vector of 8 columns of one row: local planes read directly, peers' planes from this rank's
  // receive area. All loads are issued before any is waited on (one round trip, not W*C), then only vectors
  // still holding the sentinel are re-polled. Schedule only: the values are the same whichever load lands.
  __device__ static void reduce8(const Args& a, char* rx, int64_t slot, int64_t li, int64_t row, int64_t c,
                                 uint4 (&res)[UOUT]) {
    uint4 u[NOP][UIN];
#pragma unroll
    for (int r = 0; r < W; ++r) {
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (a.mode == 0 || r == a.rank) {
          const Tin* p = reinterpret_cast<const Tin*>(a.mode == 0 ? a.in[r] : a.in[0]) + j * a.plane_stride +
                         row * a.N + c;
#pragma unroll
          for (int q = 0; q < UIN; ++q) u[r * C + j][q] = reinterpret_cast<const uint4*>(p)[q];
        } else {
          const char* p = rx + r * slot + ((li * C + j) * a.N + c) * int64_t(sizeof(Tin));
#pragma unroll
          for (int q = 0; q < UIN; ++q) u[r * C + j][q] = ld_volatile16(p + 16 * q);
        }
      }
    }
    if (a.mode != 0) {
#pragma unroll
      for (int r = 0; r < W; ++r) {
        if (r == a.rank) continue;
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const char* p = rx + r * slot + ((li * C + j) * a.N + c) * int64_t(sizeof(Tin));
#pragma unroll
          for (int q = 0; q < UIN; ++q)
            while (IN_BF16 ? pending<__nv_bfloat16>(u[r * C + j][q]) : pending<float>(u[r * C + j][q]))
              u[r * C + j][q] = ld_volatile16(p + 16 * q);
        }
      }
    }
    float v[NOP][8];
#pragma unroll
    for (int i = 0; i < NOP; ++i) widen8<Tin>(v[i], u[i]);
    float root[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      float t[NOP];
#pragma unroll
      for (int i = 0; i < NOP; ++i) t[i] = v[i][l];
      root[l] = tree<0, NOP, NOP>(t);
    }
    if constexpr (EPI == EPI_ADD_RMS) {
      // root rounded once to bf16 (the row-parallel output), then the residual join of add_rms_fwd.
      uint4 rp[1];
      pack8<__nv_bfloat16>(rp, root);
      float x[8], r8[8];
      widen8<__nv_bfloat16>(x, rp);
      const uint4 ru[1] = {*reinterpret_cast<const uint4*>(a.resid + row * a.sr + c)};
      widen8<__nv_bfloat16>(r8, ru);
#pragma unroll
      for (int l = 0; l < 8; ++l) x[l] = __fadd_rn(x[l], r8[l]);
      pack8<__nv_bfloat16>(res, x);
    } else {
      pack8<Tout>(res, root);
    }
  }

  // add_rms_fwd's per-row program for one row, by one warp, on the row's finished h at `src` (shared memory
  // or `out`). Chunks are loaded kB at a time ahead of the chain; the chain order is unchanged (k ascending,
  // j ascending), as are the fold, rstd and output product.
  static constexpr int kB = 8;
  __device__ static void norm_row(const Args& a, const __nv_bfloat16* src, int64_t row, int lane) {
    const int K = int(a.N / (8 * kLanes));
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += kB) {
      uint4 hb[kB];
#pragma unroll
      for (int i = 0; i < kB; ++i) {
        if (k0 + i < K) {
          hb[i] = *reinterpret_cast<const uint4*>(src + (k0 + i) * 8 * kLanes + 8 * lane);
        }
      }
#pragma unroll
      for (int i = 0; i < kB; ++i) {
        if (k0 + i < K) {
          const uint4 u[1] = {hb[i]};
          float v[8];
          widen8<__nv_bfloat16>(v, u);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc = __fadd_rn(acc, __fmul_rn(v[j], v[j]));
        }
      }
    }
#pragma unroll
    for (int s = kLanes >> 1; s > 0; s >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));
    const float rs = __frsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(a.N)), a.eps));
    __nv_bfloat16* yr = a.y + row * a.N;
    for (int k0 = 0; k0 < K; k0 += kB) {
      uint4 hb[kB], gb[kB][2];
#pragma unroll
      for (int i = 0; i < kB; ++i) {
        if (k0 + i < K) {
          const int64_t c = int64_t(k0 + i) * 8 * kLanes + 8 * lane;
          hb[i] = *reinterpret_cast<const uint4*>(src + c);
          if (a.w_fp32) {
            gb[i][0] = reinterpret_cast<const uint4*>(static_cast<const float*>(a.w) + c)[0];
            gb[i][1] = reinterpret_cast<const uint4*>(static_cast<const float*>(a.w) + c)[1];
          } else {
            gb[i][0] = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.w) + c);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kB; ++i) {
        if (k0 + i < K) {
          const int64_t c = int64_t(k0 + i) * 8 * kLanes + 8 * lane;
          const uint4 u[1] = {hb[i]};
          float v[8], g[8], o[8];
          widen8<__nv_bfloat16>(v, u);
          if (a.w_fp32) {
            const uint4 gu[2] = {gb[i][0], gb[i][1]};
            widen8<float>(g, gu);
          } else {
            const uint4 gu[1] = {gb[i][0]};
            widen8<__nv_bfloat16>(g, gu);
          }
          if (a.norm_mode == 2) {
#pragma unroll
            for (int j = 0; j < 8; ++j) g[j] = __fadd_rn(1.f, g[j]);
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) o[j] = __fmul_rn(__fmul_rn(v[j], rs), g[j]);
          uint4 ou[1];
          pack8<__nv_bfloat16>(ou, o);
          *reinterpret_cast<uint4*>(yr + c) = ou[0];
        }
      }
    }
  }

  // Fused protocol modes, phase A: this rank's owned rows reduced and joined (h), thread-parallel over the
  // grid, each finished vector pushed to the root/h receive area of every rank that runs the row's norm
  // (two-shot: all ranks, this one included; one-shot: this one).
  __device__ static void owned_h(const Args& a, char* rx, int64_t slot, int gen) {
    const int64_t vecs = a.N / 8;
    const bool all = a.mode != 2;
    const int64_t own_cnt = all ? a.M : (a.M > a.rank ? (a.M - a.rank + W - 1) / W : 0);
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, nth = int64_t(gridDim.x) * blockDim.x;
    for (int64_t t = tid; t < own_cnt * vecs; t += nth) {
      const int64_t idx = t / vecs, c = (t % vecs) * 8;
      const int64_t row = all ? idx : idx * W + a.rank;
      uint4 res[UOUT];
      reduce8(a, rx, slot, idx, row, c, res);
      const int64_t off = gen * a.gen2 + (row * a.N + c) * 2;
      if (a.mode == 2) {
#pragma unroll
        for (int p = 0; p < W; ++p) st_volatile16(a.recv2[p] + off, res[0]);
      } else {
        st_volatile16(a.recv2[a.rank] + off, res[0]);
      }
    }
  }

  // Phase B: every row's norm on this rank, from its own root/h receive area. A block takes `rpg` rows at a
  // time: all its threads poll the rows' h vectors (one round trip per vector, all in flight together) into
  // shared memory and `out`, then one warp per row runs the chain from shared memory.
  static constexpr int kSmemBytes = 48 * 1024;
  __device__ static void norm_all(const Args& a, const char* r2) {
    extern __shared__ uint4 sm[];
    const int warps = blockDim.x / 32, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t vecs = a.N / 8;
    int64_t cap = kSmemBytes / (a.N * 2);
    cap = cap < warps ? cap : warps;
    int64_t rpg = (a.M + gridDim.x - 1) / gridDim.x;
    rpg = rpg < 1 ? 1 : (rpg > cap ? cap : rpg);
    for (int64_t g0 = int64_t(blockIdx.x) * rpg; g0 < a.M; g0 += int64_t(gridDim.x) * rpg) {
      const int64_t gn = a.M - g0 < rpg ? a.M - g0 : rpg;
      for (int64_t t = threadIdx.x; t < gn * vecs; t += blockDim.x) {
        const int64_t off = (g0 * vecs + t) * 16;  // byte offset of this vector in the [M, N] bf16 layout
        uint4 v = ld_volatile16(r2 + off);
        while (pending<__nv_bfloat16>(v)) v = ld_volatile16(r2 + off);
        sm[t] = v;
        reinterpret_cast<uint4*>(a.out + off)[0] = v;
      }
      __syncthreads();
      if (warp < gn) norm_row(a, reinterpret_cast<const __nv_bfloat16*>(sm + warp * vecs), g0 + warp, lane);
      __syncthreads();
    }
  }

  // Rows [r0, r0 + R) (a block's group): `owned` rows are reduced here, others are copied from recv2. Then
  // (EPI_ADD_RMS) one warp per row runs the norm on the stored h.
  __device__ static void rows(const Args& a, char* rx, int64_t slot, char* r2, int gen, bool owned_pass) {
    const int64_t vecs = a.N / 8;  // 8-element vectors per row
    const int warps = blockDim.x / 32, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const bool all_owned = a.mode != 2;
    // Rows this pass handles, enumerated 0..cnt-1: owned rows (row = i*W + rank, or every row) or the rows
    // owned by peers (two-shot gather).
    const int64_t own_cnt = all_owned ? a.M : (a.M > a.rank ? (a.M - a.rank + W - 1) / W : 0);
    const int64_t cnt = owned_pass ? own_cnt : a.M - own_cnt;
    // Rows per block group: as few as spread the rows over the whole grid (a row's vectors are shared by all
    // threads of the block), at most one per warp (the norm runs one warp per row).
    int64_t rpg = (cnt + gridDim.x - 1) / gridDim.x;
    rpg = rpg < 1 ? 1 : (rpg > warps ? warps : rpg);
    for (int64_t g0 = int64_t(blockIdx.x) * rpg; g0 < cnt; g0 += int64_t(gridDim.x) * rpg) {
      const int64_t gn = cnt - g0 < rpg ? cnt - g0 : rpg;
      for (int64_t t = threadIdx.x; t < gn * vecs; t += blockDim.x) {
        const int64_t idx = g0 + t / vecs, c = (t % vecs) * 8;
        int64_t row, li;
        if (owned_pass) {
          row = all_owned ? idx : idx * W + a.rank;
          li = idx;
        } else {  // idx-th row not owned by this rank, ascending
          const int64_t q = idx / (W - 1), rr = idx % (W - 1);
          row = q * W + (rr < a.rank ? rr : rr + 1);
          li = 0;
        }
        Tout* orow = reinterpret_cast<Tout*>(a.out) + row * a.N + c;
        uint4 res[UOUT];
        if (owned_pass) {
          reduce8(a, rx, slot, li, row, c, res);
          if (a.mode == 2) {
#pragma unroll
            for (int p = 0; p < W; ++p) {
              if (p == a.rank) continue;
              char* dst = a.recv2[p] + gen * a.gen2 + (row * a.N + c) * int64_t(sizeof(Tout));
#pragma unroll
              for (int q = 0; q < UOUT; ++q) st_volatile16(dst + 16 * q, res[q]);
            }
          }
        } else {
          const char* src = r2 + (row * a.N + c) * int64_t(sizeof(Tout));
#pragma unroll
          for (int q = 0; q < UOUT; ++q) res[q] = poll16(src + 16 * q, sizeof(Tout) == 2);
        }
#pragma unroll
        for (int q = 0; q < UOUT; ++q) reinterpret_cast<uint4*>(orow)[q] = res[q];
      }
      if constexpr (EPI == EPI_ADD_RMS) {
        // Reached only by mode 0 (lamport_local, the single-GPU simulation): every row is owned. The protocol
        // modes fuse the norm in owned_h + norm_all instead.
        __syncthreads();  // the group's h rows are complete and visible to every warp of the block
        if (warp < gn) {
          const int64_t row = g0 + warp;
          norm_row(a, reinterpret_cast<const __nv_bfloat16*>(a.out) + row * a.N, row, lane);
        }
        __syncthreads();
      }
    }
  }
};

template <int W, int C, typename Tin, typename Tout, int EPI>
__global__ void __launch_bounds__(512) lamport_kernel(const __grid_constant__ Args a) {
  using K = Kern<W, C, Tin, Tout, EPI>;
  pdl_wait();
  if (a.mode == 0) {
    K::rows(a, nullptr, 0, nullptr, 0, true);
    return;
  }
  const unsigned long long call = __ldcg(a.ctl);  // every block reads it before the last block advances it
  const int gen = int(call % kGens), prev = int((call + kGens - 1) % kGens);
  const int64_t slot = slot_bytes<Tin, C>(a.M, a.N, W, a.mode);
  char* rx = a.recv1[a.rank] + gen * a.gen1;
  char* r2 = a.recv2[a.rank] + gen * a.gen2;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, nth = int64_t(gridDim.x) * blockDim.x;

  // 1. Re-fill the generation call k-1 consumed (this rank's own areas) with the sentinel.
  {
    const uint4 ones = make_uint4(~0u, ~0u, ~0u, ~0u);
    uint4* p1 = reinterpret_cast<uint4*>(a.recv1[a.rank] + prev * a.gen1);
    const int64_t n1 = int64_t(__ldcg(a.ctl + 2 + prev)) / 16;
    for (int64_t i = tid; i < n1; i += nth) p1[i] = ones;
    uint4* p2 = reinterpret_cast<uint4*>(a.recv2[a.rank] + prev * a.gen2);
    const int64_t n2 = int64_t(__ldcg(a.ctl + 5 + prev)) / 16;
    for (int64_t i = tid; i < n2; i += nth) p2[i] = ones;
  }
  if (tid == 0) {
    a.ctl[2 + gen] = (unsigned long long)(W * slot);
    a.ctl[5 + gen] = (a.mode == 2 || EPI == EPI_ADD_RMS) ? (unsigned long long)(a.M * a.N * int64_t(sizeof(Tout)))
                                                         : 0ull;
  }

  // 2. Push this rank's wire planes: one-shot to every peer, two-shot to each row's owner.
  {
    const int64_t vecs = a.N / 8, per_row = C * vecs;
    const Tin* in = reinterpret_cast<const Tin*>(a.in[0]);
    for (int64_t t = tid; t < a.M * per_row; t += nth) {
      const int64_t row = t / per_row, rem = t % per_row, j = rem / vecs, c = (rem % vecs) * 8;
      const int owner = int(row % W);
      if (a.mode == 2 && owner == a.rank) continue;
      uint4 u[K::UIN];
      const Tin* src = in + j * a.plane_stride + row * a.N + c;
#pragma unroll
      for (int q = 0; q < K::UIN; ++q) u[q] = desentinel<Tin>(reinterpret_cast<const uint4*>(src)[q]);
      const int64_t li = a.mode == 2 ? row / W : row;
      const int64_t off = gen * a.gen1 + a.rank * slot + ((li * C + j) * a.N + c) * int64_t(sizeof(Tin));
      if (a.mode == 2) {
#pragma unroll
        for (int q = 0; q < K::UIN; ++q) st_volatile16(a.recv1[owner] + off + 16 * q, u[q]);
      } else {
#pragma unroll
        for (int p = 0; p < W; ++p) {
          if (p == a.rank) continue;
#pragma unroll
          for (int q = 0; q < K::UIN; ++q) st_volatile16(a.recv1[p] + off + 16 * q, u[q]);
        }
      }
    }
  }

  // 3. Owned rows: reduce (+ epilogue), store locally, two-shot: push the finished rows to every peer.
  if constexpr (EPI == EPI_ADD_RMS) {
    // 3. Owned rows reduced and joined, pushed as h.  4. Every row's norm from the receive area.
    K::owned_h(a, rx, slot, gen);
    K::norm_all(a, r2);
  } else {
    // 3. Owned rows: reduce, store locally, two-shot: push the finished rows to every peer.
    K::rows(a, rx, slot, r2, gen, true);
    // 4. Two-shot: the peers' finished rows.
    if (a.mode == 2) K::rows(a, rx, slot, r2, gen, false);
  }

  // Dependents may start their prologue now (not earlier: an early trigger lets the next persistent GEMM
  // claim SMs while this kernel still moves data -- measured slower).
  pdl_trigger();
  // 5. Advance the generation once every block has read it.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned long long done = atomicAdd(&a.ctl[1], 1ull);
    if (done == gridDim.x - 1) {
      a.ctl[1] = 0ull;
      a.ctl[0] = call + 1ull;
    }
  }
}

// ------------------------------------------------------------------------------------------------- launch

template <int W, int C, typename Tin, typename Tout, int EPI>
void launch(const Args& a, int blocks, int threads, bool pdl, cudaStream_t s) {
  cudaLaunchConfig_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = EPI == EPI_ADD_RMS ? Kern<W, C, Tin, Tout, EPI>::kSmemBytes : 0;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, lamport_kernel<W, C, Tin, Tout, EPI>, a));
}

template <int W, int C, typename Tin>
void by_out(const Args& a, bool out_bf16, int epi, int blocks, int threads, bool pdl, cudaStream_t s) {
  if (epi == EPI_ADD_RMS) {
    TORCH_CHECK(out_bf16, "pik lamport: the add+RMSNorm epilogue needs a bf16 root");
    launch<W, C, Tin, __nv_bfloat16, EPI_ADD_RMS>(a, blocks, threads, pdl, s);
  } else if (out_bf16) {
    launch<W, C, Tin, __nv_bfloat16, EPI_ROOT>(a, blocks, threads, pdl, s);
  } else {
    launch<W, C, Tin, float, EPI_ROOT>(a, blocks, threads, pdl, s);
  }
}

template <int W>
void by_planes(const Args& a, int C, bool in_bf16, bool out_bf16, int epi, int blocks, int threads, bool pdl,
               cudaStream_t s) {
  if (!in_bf16) {
    TORCH_CHECK(C == 1, "pik lamport: fp32 operands are internal nodes, one plane per rank");
    by_out<W, 1, float>(a, out_bf16, epi, blocks, threads, pdl, s);
    return;
  }
  if (C == 1) return by_out<W, 1, __nv_bfloat16>(a, out_bf16, epi, blocks, threads, pdl, s);
  if constexpr (W <= 8) {
    if (C == 2) return by_out<W, 2, __nv_bfloat16>(a, out_bf16, epi, blocks, threads, pdl, s);
  }
  TORCH_CHECK(false, "pik lamport: unsupported (world, planes) = (", W, ", ", C, ")");
}

int64_t wire_slot_bytes(int64_t M, int64_t N, int64_t W, int64_t C, int64_t elt, int64_t mode) {
  return align16(slot_rows(M, int(W), int(mode)) * C * N * elt);
}

void lamport_allreduce(std::vector<int64_t> in_ptrs, std::vector<int64_t> recv1, std::vector<int64_t> recv2,
                       c10::optional<torch::Tensor> ctl, c10::optional<torch::Tensor> resid,
                       c10::optional<torch::Tensor> weight, torch::Tensor out, c10::optional<torch::Tensor> y,
                       int64_t M, int64_t N, int64_t plane_stride, int64_t gen1, int64_t gen2, int64_t rank,
                       int64_t world, int64_t mode, int64_t planes, bool in_bf16, int64_t norm_mode, double eps,
                       int64_t blocks, int64_t threads, bool pdl) {
  const int W = int(world);
  TORCH_CHECK(W >= 2 && W <= kMaxWorld && (W & (W - 1)) == 0, "pik lamport: world must be a power of two in [2, 16]");
  TORCH_CHECK(mode >= 0 && mode <= 2, "pik lamport: mode must be 0..2");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.numel() == M * N, "out must be a contiguous [M, N]");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat32, "root dtype");
  TORCH_CHECK(N % 8 == 0 && plane_stride % 8 == 0, "pik lamport: N and the plane stride must be multiples of 8");
  TORCH_CHECK(blocks >= 1 && threads >= 32 && threads <= 512 && threads % 32 == 0, "bad launch geometry");
  TORCH_CHECK(int64_t(in_ptrs.size()) == (mode == 0 ? W : 1), "pik lamport: operand pointer count");
  const int epi = norm_mode > 0 ? EPI_ADD_RMS : EPI_ROOT;
  Args a;
  std::memset(&a, 0, sizeof(a));
  for (size_t r = 0; r < in_ptrs.size(); ++r) {
    a.in[r] = reinterpret_cast<const char*>(in_ptrs[r]);
    TORCH_CHECK(in_ptrs[r] % 16 == 0, "pik lamport: operand planes must be 16-byte aligned");
  }
  if (mode != 0) {
    TORCH_CHECK(int64_t(recv1.size()) == W && int64_t(recv2.size()) == W, "one receive area per rank");
    TORCH_CHECK(ctl.has_value() && ctl->is_cuda() && ctl->numel() >= 8 && ctl->element_size() == 8, "ctl");
    TORCH_CHECK(rank >= 0 && rank < W, "rank out of range");
    for (int r = 0; r < W; ++r) {
      a.recv1[r] = reinterpret_cast<char*>(recv1[r]);
      a.recv2[r] = reinterpret_cast<char*>(recv2[r]);
    }
    a.ctl = reinterpret_cast<unsigned long long*>(ctl->data_ptr());
    const int64_t elt = in_bf16 ? 2 : 4;
    TORCH_CHECK(W * wire_slot_bytes(M, N, W, planes, elt, mode) <= gen1, "pik lamport: wire exceeds generation");
    TORCH_CHECK((mode != 2 && norm_mode == 0) || M * N * int64_t(out.element_size()) <= gen2,
                "pik lamport: roots exceed generation");
  }
  if (epi == EPI_ADD_RMS) {
    TORCH_CHECK(N % (8 * kLanes) == 0, "pik lamport: the fused norm needs N % 256 == 0 (recipe v1, L = 32)");
    TORCH_CHECK(N * 2 <= 48 * 1024, "pik lamport: the fused norm stages a row in 48 KiB of shared memory");
    TORCH_CHECK(resid.has_value() && resid->scalar_type() == torch::kBFloat16 && resid->stride(-1) == 1 &&
                    resid->size(0) == M && resid->size(1) == N && (resid->stride(0) % 8) == 0 &&
                    (reinterpret_cast<uintptr_t>(resid->data_ptr()) % 16) == 0,
                "pik lamport: residual must be a bf16 [M, N] with 16-byte aligned rows");
    TORCH_CHECK(weight.has_value() && weight->is_contiguous() && weight->numel() == N &&
                    (weight->scalar_type() == torch::kBFloat16 || weight->scalar_type() == torch::kFloat32),
                "pik lamport: weight must be a contiguous [N] bf16/fp32");
    TORCH_CHECK(y.has_value() && y->is_contiguous() && y->numel() == M * N && y->scalar_type() == torch::kBFloat16,
                "pik lamport: y must be a contiguous bf16 [M, N]");
    TORCH_CHECK(norm_mode == 1 || norm_mode == 2, "norm mode must be 1 (w) or 2 (1 + w)");
    a.resid = reinterpret_cast<const __nv_bfloat16*>(resid->data_ptr());
    a.sr = resid->stride(0);
    a.w = weight->data_ptr();
    a.w_fp32 = weight->scalar_type() == torch::kFloat32;
    a.y = reinterpret_cast<__nv_bfloat16*>(y->data_ptr());
    a.norm_mode = int(norm_mode);
    a.eps = float(eps);
  }
  a.out = reinterpret_cast<char*>(out.data_ptr());
  a.M = M;
  a.N = N;
  a.plane_stride = plane_stride;
  a.gen1 = gen1;
  a.gen2 = gen2;
  a.rank = int(rank);
  a.mode = int(mode);
  if (M == 0 && mode == 0) return;
  const at::cuda::OptionalCUDAGuard guard(out.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const bool out_bf16 = out.scalar_type() == torch::kBFloat16;
  const int C = int(planes);
  switch (W) {
    case 2: by_planes<2>(a, C, in_bf16, out_bf16, epi, int(blocks), int(threads), pdl, s); break;
    case 4: by_planes<4>(a, C, in_bf16, out_bf16, epi, int(blocks), int(threads), pdl, s); break;
    case 8: by_planes<8>(a, C, in_bf16, out_bf16, epi, int(blocks), int(threads), pdl, s); break;
    case 16: by_planes<16>(a, C, in_bf16, out_bf16, epi, int(blocks), int(threads), pdl, s); break;
    default: TORCH_CHECK(false, "pik lamport: world");
  }
}

}  // namespace pik_lamport

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lamport_allreduce", &pik_lamport::lamport_allreduce, "pik owner tree all-reduce, Lamport protocol",
        py::arg("in_ptrs"), py::arg("recv1"), py::arg("recv2"), py::arg("ctl"), py::arg("resid"), py::arg("weight"),
        py::arg("out"), py::arg("y"), py::arg("M"), py::arg("N"), py::arg("plane_stride"), py::arg("gen1"),
        py::arg("gen2"), py::arg("rank"), py::arg("world"), py::arg("mode"), py::arg("planes"), py::arg("in_bf16"),
        py::arg("norm_mode"), py::arg("eps"), py::arg("blocks"), py::arg("threads"), py::arg("pdl"));
  m.def("wire_slot_bytes", &pik_lamport::wire_slot_bytes);
}
