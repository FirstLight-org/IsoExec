// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 Engram, native SM100. See ../README.md for the declared numerical program.
// Compiled with --fmad=false --ftz=false; every float operation below is spelled out in the order
// the README declares. No float atomics anywhere (integer CUB primitives only in table_grad).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace engram {

constexpr int kMaxNgram = 8;

// ------------------------------------------------------------------------------------------
// 1. tokenizer-compressed n-gram hashing (integer-exact; mirrors EngramHash / NgramHashState)
// ------------------------------------------------------------------------------------------
// ids [B,T] i64, token_map [V] i64, mask [B,T] bool|null, state [B,N-1] i64|null (compressed ids of the
// N-1 positions before this call, oldest first, -1 = dead/absent). primes [L,N-1,H], offsets [L,(N-1)H],
// mult [L,N]. out [B,T,L,(N-1)H].
// The history id of position q < 0 of row b from the engine's RAW lookback (``raw [B, N-1]`` int32 token ids, newest
// first, -1 none): the compressed id tmap[raw] (-1 for none), the value engine glue used to build as the int64
// oldest-first ``state`` (flip, map through tmap, -1 kept): oldest-first slot (N - 1) + q is newest-first slot -q - 1.
// A raw id outside the vocabulary raises the error word (the torch glue's index would fault) and reads as -1.
__device__ __forceinline__ int64_t raw_history(const int* raw, int64_t b, int64_t q, int N, const int64_t* tmap,
                                               int64_t V, int* err) {
  const int64_t r = raw[b * (N - 1) + (-q - 1)];
  if (r < 0) return -1;
  if (r >= V) { *err = 1; return -1; }
  return tmap[r];
}

__global__ void hash_kernel(const int64_t* __restrict__ ids, const int64_t* __restrict__ tmap,
                            const bool* __restrict__ mask, const int64_t* __restrict__ state,
                            const int64_t* __restrict__ primes, const int64_t* __restrict__ offsets,
                            const int64_t* __restrict__ mult, int64_t* __restrict__ out,
                            int* __restrict__ err, int64_t B, int64_t T, int64_t V, int L, int N, int H,
                            int64_t pad_id, const uint64_t* __restrict__ magic, const int* __restrict__ raw,
                            const int64_t* __restrict__ seg) {
  // One thread per output column (b, t, l, i, h): the 64-bit modulo is the expensive step, so the (N-1) H
  // columns of a (b, t, l) run in parallel instead of in one thread's loop. Integer arithmetic only: the same
  // tok / rolling / modulo as the per-(b, t, l) loop, column by column (bit-identical by construction).
  const int cols = (N - 1) * H;
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B * T * L * cols) return;
  const int col = (int)(gid % cols);
  const int64_t btl = gid / cols;
  const int l = (int)(btl % L);
  const int64_t bt = btl / L;
  const int64_t b = bt / T, t = bt % T;
  const int i = col / H + 1, h = col % H;   // n-gram order i + 1 uses tokens t, t-1, .., t-i
  uint64_t rolling = 0;
  bool blocked = false;
  for (int s = 0; s <= i; ++s) {
    int64_t q = t - s;
    int64_t c;
    if (seg != nullptr && q >= 0 && q < seg[b * T + t]) {
      c = -1;              // before this token's packed sequence: an empty history, as a call of its own
    } else if (q >= 0) {
      int64_t raw = ids[b * T + q];
      if (raw < 0 || raw >= V) {
        if (h == 0 && i == N - 1 && l == 0) *err = 1;
        c = -1;
      } else if (mask != nullptr && !mask[b * T + q]) {
        c = -1;
      } else {
        c = tmap[raw];
      }
    } else {
      c = raw != nullptr ? raw_history(raw, b, q, N, tmap, V, err)
                         : (state != nullptr) ? state[b * (N - 1) + (N - 1) + q] : -1;
    }
    blocked = blocked || (c == -1);
    const int64_t tok = blocked ? pad_id : c;
    // rolling xor of tok[s] * mult[l,s]; the product fits int64 by the multiplier bound.
    rolling = s == 0 ? (uint64_t)(tok * mult[l * N]) : (rolling ^ (uint64_t)(tok * mult[l * N + s]));
  }
  const int64_t pi = ((int64_t)l * (N - 1) + (i - 1)) * H + h;
  const uint64_t p = (uint64_t)primes[pi];
  uint64_t r;
  if (magic != nullptr) {
    // rolling mod p without the emulated 64-bit division: magic = floor((2^64 - 1) / p) (host), the quotient estimate
    // q = mulhi(rolling, magic) is at most 2 below the true quotient, so r = rolling - q p lands in [0, 3p) and at
    // most two exact corrections give the remainder: integer arithmetic, the same value as rolling % p.
    const uint64_t q = __umul64hi(rolling, magic[pi]);
    r = rolling - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
  } else {
    r = rolling % p;
  }
  out[((b * T + t) * L + l) * cols + col] = (int64_t)r + offsets[(int64_t)l * cols + col];
}

// Remainder of rolling mod p (magic = floor((2^64 - 1) / p) per prime, or nullptr for the plain modulo).
__device__ __forceinline__ uint64_t hash_mod(uint64_t rolling, uint64_t p, const uint64_t* magic, int64_t pi) {
  if (magic != nullptr) {
    // q = mulhi(rolling, magic) is at most 2 below the true quotient, so r = rolling - q p lands in [0, 3p) and at
    // most two exact corrections give the remainder: integer arithmetic, the same value as rolling % p.
    const uint64_t q = __umul64hi(rolling, magic[pi]);
    uint64_t r = rolling - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return r;
  }
  return rolling % p;
}

// Tiled form of hash_kernel (the same integers): a block owns kHashTileTokens consecutive tokens (b, t) of the
// flattened [B*T] axis. Phase 1: one thread per (token, s) forms the compressed id of position t - s once (instead
// of once per output column). Phase 2: one thread per (token, l) forms the N - 1 rolling values (prefix xor with the
// prefix blocking, exactly hash_kernel's loop). Phase 3: one thread per output column reduces its rolling value by
// its prime and adds the offset; the block's columns are one contiguous, coalesced range of `out`.
constexpr int kHashTileTokens = 32;
constexpr int kHashTileThreads = 256;
constexpr int64_t kHashTileMinTokens = 4096;

__global__ void __launch_bounds__(kHashTileThreads)
hash_tile_kernel(const int64_t* __restrict__ ids, const int64_t* __restrict__ tmap,
                 const bool* __restrict__ mask, const int64_t* __restrict__ state,
                 const int64_t* __restrict__ primes, const int64_t* __restrict__ offsets,
                 const int64_t* __restrict__ mult, int64_t* __restrict__ out,
                 int* __restrict__ err, int64_t B, int64_t T, int64_t V, int L, int N, int H,
                 int64_t pad_id, const uint64_t* __restrict__ magic, const int* __restrict__ raw,
                 const int64_t* __restrict__ seg) {
  extern __shared__ int64_t hash_smem[];
  int64_t* cids = hash_smem;                                   // [kHashTileTokens][N]
  uint64_t* roll = reinterpret_cast<uint64_t*>(hash_smem + kHashTileTokens * N);   // [kHashTileTokens][L][N-1]
  const int cols = (N - 1) * H;
  const int64_t BT = B * T;
  const int64_t bt0 = (int64_t)blockIdx.x * kHashTileTokens;
  const int tokens = (int)min((int64_t)kHashTileTokens, BT - bt0);
  for (int j = threadIdx.x; j < tokens * N; j += blockDim.x) {
    const int tok = j / N, s = j % N;
    const int64_t bt = bt0 + tok, b = bt / T, t = bt % T, q = t - s;
    int64_t c;
    if (seg != nullptr && q >= 0 && q < seg[b * T + t]) {
      c = -1;
    } else if (q >= 0) {
      const int64_t raw = ids[b * T + q];
      if (raw < 0 || raw >= V) {
        *err = 1;
        c = -1;
      } else if (mask != nullptr && !mask[b * T + q]) {
        c = -1;
      } else {
        c = tmap[raw];
      }
    } else {
      c = raw != nullptr ? raw_history(raw, b, q, N, tmap, V, err)
                         : (state != nullptr) ? state[b * (N - 1) + (N - 1) + q] : -1;
    }
    cids[j] = c;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < tokens * L; j += blockDim.x) {
    const int tok = j / L, l = j % L;
    uint64_t rolling = 0;
    bool blocked = false;
    for (int s = 0; s < N; ++s) {
      const int64_t c = cids[tok * N + s];
      blocked = blocked || (c == -1);
      const int64_t tk = blocked ? pad_id : c;
      // rolling xor of tok[s] * mult[l,s]; the product fits int64 by the multiplier bound.
      rolling = s == 0 ? (uint64_t)(tk * mult[l * N]) : (rolling ^ (uint64_t)(tk * mult[l * N + s]));
      if (s > 0) roll[j * (N - 1) + (s - 1)] = rolling;
    }
  }
  __syncthreads();
  const int per_token = L * cols;
  int64_t* dst = out + bt0 * per_token;
  for (int j = threadIdx.x; j < tokens * per_token; j += blockDim.x) {
    const int tok = j / per_token, rem = j % per_token;
    const int l = rem / cols, col = rem % cols;
    const int i = col / H + 1, h = col % H;
    const int64_t pi = ((int64_t)l * (N - 1) + (i - 1)) * H + h;
    const uint64_t r = hash_mod(roll[(tok * L + l) * (N - 1) + (i - 1)], (uint64_t)primes[pi], magic, pi);
    dst[j] = (int64_t)r + offsets[(int64_t)l * cols + col];
  }
}

__global__ void state_kernel(const int64_t* __restrict__ ids, const int64_t* __restrict__ tmap,
                             const bool* __restrict__ mask, const int64_t* __restrict__ old_state,
                             int64_t* __restrict__ new_state, int64_t B, int64_t T, int64_t V, int N) {
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B * (N - 1)) return;
  int64_t b = gid / (N - 1);
  int j = (int)(gid % (N - 1));
  int64_t pos = T - (N - 1) + j;
  int64_t c;
  if (pos >= 0) {
    int64_t raw = ids[b * T + pos];
    c = (raw < 0 || raw >= V || (mask != nullptr && !mask[b * T + pos])) ? -1 : tmap[raw];
  } else {
    c = old_state != nullptr ? old_state[b * (N - 1) + (T + j)] : -1;
  }
  new_state[gid] = c;
}

void hash_cuda(const at::Tensor& ids, const at::Tensor& tmap, const c10::optional<at::Tensor>& mask,
               const c10::optional<at::Tensor>& state, const at::Tensor& primes, const at::Tensor& offsets,
               const at::Tensor& mult, int64_t pad_id, at::Tensor out, at::Tensor err,
               const c10::optional<at::Tensor>& new_state, const c10::optional<at::Tensor>& magic,
               const c10::optional<at::Tensor>& raw_state, const c10::optional<at::Tensor>& seg_start) {
  c10::cuda::CUDAGuard g(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t B = ids.size(0), T = ids.size(1);
  int L = (int)primes.size(0), N = (int)primes.size(1) + 1, H = (int)primes.size(2);
  TORCH_CHECK(N >= 2 && N <= kMaxNgram, "max_ngram_size must be in [2,", kMaxNgram, "]");
  int64_t total = B * T * L * (int64_t)((N - 1) * H);
  const int* raw = nullptr;
  if (raw_state) {
    TORCH_CHECK(!state && !new_state, "raw_state replaces state and returns no new state");
    TORCH_CHECK(raw_state->scalar_type() == at::kInt && raw_state->is_contiguous() && raw_state->dim() == 2 &&
                raw_state->size(0) == B && raw_state->size(1) == N - 1, "raw_state: int32 [B, N-1], newest first");
    raw = raw_state->data_ptr<int>();
  }
  const int64_t* seg = nullptr;
  if (seg_start) {
    TORCH_CHECK(!state && !raw_state && !new_state, "seg_start hashes packed sequences from an empty history");
    TORCH_CHECK(seg_start->scalar_type() == at::kLong && seg_start->is_contiguous() && seg_start->numel() == B * T,
                "seg_start: int64 [B*T], each token's sequence start");
    seg = seg_start->data_ptr<int64_t>();
  }
  const size_t tile_smem = sizeof(int64_t) * (size_t)kHashTileTokens * (N + (size_t)L * (N - 1));
  // The token tile pays off once there are enough tokens to fill the GPU with tiles (r21: 8192 rows 5.9 vs 10.1 us; at
  // <= 2048 rows the per-column kernel is as fast or faster). A pure function of the shape; the same integers either way.
  if (total > 0 && B * T >= kHashTileMinTokens && tile_smem <= 48 * 1024) {
    const int64_t BT = B * T;
    hash_tile_kernel<<<(unsigned)((BT + kHashTileTokens - 1) / kHashTileTokens), kHashTileThreads, tile_smem, stream>>>(
        ids.data_ptr<int64_t>(), tmap.data_ptr<int64_t>(), mask ? mask->data_ptr<bool>() : nullptr,
        state ? state->data_ptr<int64_t>() : nullptr, primes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        mult.data_ptr<int64_t>(), out.data_ptr<int64_t>(), err.data_ptr<int>(), B, T, tmap.numel(), L, N, H,
        pad_id, magic ? reinterpret_cast<const uint64_t*>(magic->data_ptr<int64_t>()) : nullptr, raw, seg);
  } else if (total > 0) {
    hash_kernel<<<(unsigned)((total + 127) / 128), 128, 0, stream>>>(
        ids.data_ptr<int64_t>(), tmap.data_ptr<int64_t>(), mask ? mask->data_ptr<bool>() : nullptr,
        state ? state->data_ptr<int64_t>() : nullptr, primes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        mult.data_ptr<int64_t>(), out.data_ptr<int64_t>(), err.data_ptr<int>(), B, T, tmap.numel(), L, N, H,
        pad_id, magic ? reinterpret_cast<const uint64_t*>(magic->data_ptr<int64_t>()) : nullptr, raw, seg);
  }
  if (new_state) {
    int64_t n = B * (N - 1);
    state_kernel<<<(unsigned)((n + 127) / 128), 128, 0, stream>>>(
        ids.data_ptr<int64_t>(), tmap.data_ptr<int64_t>(), mask ? mask->data_ptr<bool>() : nullptr,
        state ? state->data_ptr<int64_t>() : nullptr, new_state->data_ptr<int64_t>(), B, T, tmap.numel(), N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------
// 2. row lookup: FP8 e4m3 rows * UE8M0 scale per 32 (exact decode), or dense fp32 master rows
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
  if (e == 255) return __uint_as_float(0x7fc00000u);       // NaN encoding
  if (e == 0) return __uint_as_float(0x00400000u);         // 2^-127 (fp32 subnormal), exact
  return __uint_as_float((uint32_t)e << 23);               // 2^(e-127)
}
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t v) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(v, __NV_E4M3);
  return __half2float(__half(h));
}
template <typename O> __device__ __forceinline__ O cast_out(float v);
template <> __device__ __forceinline__ __nv_bfloat16 cast_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ float cast_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __half cast_out<__half>(float v) { return __float2half_rn(v); }

// one warp per output row; lane owns 8 consecutive elements per 256-wide chunk
template <typename O> struct Pack8;
template <> struct Pack8<__nv_bfloat16> { using T = uint4; };
template <> struct Pack8<__half> { using T = uint4; };
template <> struct Pack8<float> { using T = struct { float4 a, b; }; };

// One warp per output row; lane owns 8 consecutive elements per 256-wide chunk; the 8 outputs leave as one
// vector store (the same values, one instruction instead of eight).
template <typename O>
__global__ void lookup_fp8_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                                  const int64_t* __restrict__ idx, O* __restrict__ out,
                                  int* __restrict__ err, int64_t N, int64_t R, int D) {
  int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= N) return;
  int lane = threadIdx.x & 31;
  int64_t r = idx[row];
  bool bad = (r < 0 || r >= R);
  if (bad && lane == 0) *err = 1;
  for (int c = lane * 8; c < D; c += 256) {
    typename Pack8<O>::T pk;
    O* vals = reinterpret_cast<O*>(&pk);
    if (bad) {
#pragma unroll
      for (int i = 0; i < 8; ++i) vals[i] = cast_out<O>(0.f);
    } else {
      uint2 raw = *reinterpret_cast<const uint2*>(w + r * D + c);
      float s = e8m0_to_float(sc[r * (D / 32) + c / 32]);
      const uint8_t* p = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) vals[i] = cast_out<O>(__fmul_rn(fp8e4m3_to_float(p[i]), s));
    }
    *reinterpret_cast<typename Pack8<O>::T*>(out + row * D + c) = pk;
  }
}

template <typename O>
__global__ void lookup_master_kernel(const float* __restrict__ w, const int64_t* __restrict__ idx,
                                     O* __restrict__ out, int* __restrict__ err, int64_t N, int64_t R, int D) {
  int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= N) return;
  int lane = threadIdx.x & 31;
  int64_t r = idx[row];
  bool bad = (r < 0 || r >= R);
  if (bad && lane == 0) *err = 1;
  for (int c = lane * 4; c < D; c += 128) {
    float4 v = bad ? make_float4(0.f, 0.f, 0.f, 0.f) : *reinterpret_cast<const float4*>(w + r * D + c);
    out[row * D + c + 0] = cast_out<O>(v.x);
    out[row * D + c + 1] = cast_out<O>(v.y);
    out[row * D + c + 2] = cast_out<O>(v.z);
    out[row * D + c + 3] = cast_out<O>(v.w);
  }
}

void lookup_fp8_cuda(const at::Tensor& w, const at::Tensor& sc, const at::Tensor& idx, at::Tensor out,
                     at::Tensor err) {
  c10::cuda::CUDAGuard g(w.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = idx.numel(), R = w.size(0);
  int D = (int)w.size(1);
  TORCH_CHECK(D % 8 == 0 && D % 32 == 0, "row width must be a multiple of 32");
  if (N == 0) return;
  unsigned blocks = (unsigned)((N + 7) / 8);
  auto* wp = reinterpret_cast<const uint8_t*>(w.data_ptr());
  auto* sp = reinterpret_cast<const uint8_t*>(sc.data_ptr());
  if (out.scalar_type() == at::kBFloat16)
    lookup_fp8_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(wp, sp, idx.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), N, R, D);
  else if (out.scalar_type() == at::kFloat)
    lookup_fp8_kernel<float><<<blocks, 256, 0, stream>>>(wp, sp, idx.data_ptr<int64_t>(),
        out.data_ptr<float>(), err.data_ptr<int>(), N, R, D);
  else if (out.scalar_type() == at::kHalf)
    lookup_fp8_kernel<__half><<<blocks, 256, 0, stream>>>(wp, sp, idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out.data_ptr()), err.data_ptr<int>(), N, R, D);
  else TORCH_CHECK(false, "out dtype must be bf16, fp16 or fp32");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------
// 2b. fused lookup + checkpoint activation quantisation (the exact input form of the checkpoint FP8 GEMMs)
// ------------------------------------------------------------------------------------------
// The Engram `wkv` projection quantises its bf16 input (the looked-up rows) with the 1x32 UE8M0 activation program of
// quant/checkpoint_sm100 (`quant<32, FP8, UE8M0>`: amax floored at 1e-4, scale = amax * float(1/448) rounded UP to a power
// of two, code = e4m3(clamp(div_rn(x, scale), +-448))). Looked-up rows are D (multiple of 32) wide, so every 32-group of
// the quantiser lies inside one row: the fused kernel decodes (fp8 * 2^e, one narrowing to bf16 -- the bf16 that the
// unfused lookup would have stored) and quantises in registers, writing E4M3 codes [N, D] and UE8M0 scales [N, D/32]
// directly in the layout `mm/checkpoint_tc_sm100` consumes. No bf16 intermediate, no second pass; bitwise equal to
// quantize(fp8_rowwise_lookup(...)).
constexpr float kActFloor = 1.0e-4f;                           // quant/checkpoint_sm100 floor (activations)
constexpr float kTableFloor = 448.0f * 1.1754943508222875e-38f;  // 448 * 2^-126: a table scale is never subnormal

// 8 values per lane, four consecutive lanes = one 32-group. Instruction-for-instruction the quantiser's program.
template <int FLOOR_KIND>
__device__ __forceinline__ void ue8m0_quant_group(const float (&v)[8], uint8_t (&code)[8], uint8_t& sbyte) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, FLOOR_KIND == 0 ? kActFloor : kTableFloor);
  float scale = __fmul_rn(amax, 1.0f / 448.0f);
  const uint32_t b = __float_as_uint(scale);
  const uint32_t exponent = ((b >> 23) & 255) + ((b & 0x7fffff) != 0);
  scale = __uint_as_float(exponent << 23);
  sbyte = (uint8_t)exponent;
  // scale = 2^(exponent-127) is an exact power of two, so for 1 <= exponent <= 253 (1/scale = 2^(127-exponent) is an exact
  // normal float) x / scale and x * (1/scale) are the same real number and round identically (overflow and subnormal results
  // included): the multiply replaces the (slow, correctly rounded) division bit for bit. Other exponents (scale 0, inf,
  // a subnormal-range reciprocal) take the division.
  if (exponent >= 1 && exponent <= 253) {
    const float inv = __uint_as_float((254u - exponent) << 23);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = fminf(448.0f, fmaxf(-448.0f, __fmul_rn(v[i], inv)));
      code[i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = fminf(448.0f, fmaxf(-448.0f, __fdiv_rn(v[i], scale)));
      code[i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
  }
}

// kRowsPerWarp independent rows per warp: all index, scale and data loads of the rows are issued before any is consumed
// (the gather is latency bound; one row per warp leaves a single 256-byte request in flight per warp).
constexpr int kLookupRows = 4;
__global__ void lookup_fp8_quant_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                                        const int64_t* __restrict__ idx, uint8_t* __restrict__ codes,
                                        uint8_t* __restrict__ qs, int* __restrict__ err, int64_t N, int64_t R, int D) {
  const int64_t row0 = ((int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * kLookupRows;
  if (row0 >= N) return;
  const int lane = threadIdx.x & 31;
  int64_t r[kLookupRows];
  bool bad[kLookupRows], live[kLookupRows];
#pragma unroll
  for (int i = 0; i < kLookupRows; ++i) {
    live[i] = row0 + i < N;
    r[i] = live[i] ? idx[row0 + i] : 0;
    bad[i] = r[i] < 0 || r[i] >= R;
    if (live[i] && bad[i] && lane == 0) *err = 1;
  }
  for (int c0 = 0; c0 < D; c0 += 256) {
    const int c = c0 + lane * 8;
    const bool act = c < D;
    uint2 raw[kLookupRows];
    uint8_t eb[kLookupRows];
#pragma unroll
    for (int i = 0; i < kLookupRows; ++i) {
      const bool ld = live[i] && !bad[i] && act;
      raw[i] = ld ? *reinterpret_cast<const uint2*>(w + r[i] * D + c) : make_uint2(0u, 0u);
      eb[i] = ld ? sc[r[i] * (D / 32) + c / 32] : (uint8_t)127;
    }
#pragma unroll
    for (int i = 0; i < kLookupRows; ++i) {
      float v[8];
      if (bad[i] || !live[i] || !act) {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = 0.f;
      } else {
        const float s = e8m0_to_float(eb[i]);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&raw[i]);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(fp8e4m3_to_float(p[k]), s)));
      }
      uint8_t code[8], sb;
      ue8m0_quant_group<0>(v, code, sb);      // all lanes participate (warp shuffles); stores are predicated below
      if (live[i] && act) {
        *reinterpret_cast<uint2*>(codes + (row0 + i) * D + c) = *reinterpret_cast<const uint2*>(code);
        if ((lane & 3) == 0) qs[(row0 + i) * (D / 32) + c / 32] = sb;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------
// 2c. master (fp32) -> FP8 + UE8M0 table encode (after optimiser steps)
// ------------------------------------------------------------------------------------------
// The same 1x32 UE8M0 program applied to fp32 values with the table floor. Pure per-group function of its 32 inputs (a
// max, one multiply, an exponent increment, correctly rounded divisions and the hardware RN-even e4m3 conversion): no
// cross-group or cross-thread order, so the encode is run-to-run, device and schedule independent.
// Work item i (one warp): master row s = src ? src[i] : i is encoded into table row d = dst_map ? dst_map[s] : s. `src`
// (optional) is a list of master rows -- e.g. the unique touched rows of an optimiser step, of which only the first
// *nrows (device scalar) are live; `dst_map` (optional) maps master rows of an overlay to rows of the FP8 table.
__global__ void quantize_table_kernel(const float* __restrict__ master, const int64_t* __restrict__ src_list,
                                      const int64_t* __restrict__ dst_map, const int32_t* __restrict__ nrows,
                                      uint8_t* __restrict__ w, uint8_t* __restrict__ sc, int* __restrict__ err,
                                      int64_t N, int64_t M, int64_t R, int D) {
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= N) return;
  if (nrows && item >= *nrows) return;
  const int lane = threadIdx.x & 31;
  const int64_t src = src_list ? src_list[item] : item;
  if (src < 0 || src >= M) { if (lane == 0) *err = 1; return; }
  const int64_t dst = dst_map ? dst_map[src] : src;
  if (dst < 0 || dst >= R) { if (lane == 0) *err = 1; return; }
  for (int c0 = 0; c0 < D; c0 += 256) {
    const int c = c0 + lane * 8;
    const bool act = c < D;
    float v[8];
    if (!act) {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = 0.f;
    } else {
      const float4 a = *reinterpret_cast<const float4*>(master + src * D + c);
      const float4 b = *reinterpret_cast<const float4*>(master + src * D + c + 4);
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
    uint8_t code[8], sb;
    ue8m0_quant_group<1>(v, code, sb);
    if (act) {
      *reinterpret_cast<uint2*>(w + dst * D + c) = *reinterpret_cast<const uint2*>(code);
      if ((lane & 3) == 0) sc[dst * (D / 32) + c / 32] = sb;
    }
  }
}

void lookup_fp8_quant_cuda(const at::Tensor& w, const at::Tensor& sc, const at::Tensor& idx, at::Tensor codes,
                           at::Tensor qs, at::Tensor err) {
  c10::cuda::CUDAGuard g(w.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = idx.numel(), R = w.size(0);
  int D = (int)w.size(1);
  TORCH_CHECK(D % 32 == 0, "row width must be a multiple of 32");
  if (N == 0) return;
  lookup_fp8_quant_kernel<<<(unsigned)((N + 8 * kLookupRows - 1) / (8 * kLookupRows)), 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(w.data_ptr()), reinterpret_cast<const uint8_t*>(sc.data_ptr()),
      idx.data_ptr<int64_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()), reinterpret_cast<uint8_t*>(qs.data_ptr()),
      err.data_ptr<int>(), N, R, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_table_cuda(const at::Tensor& master, const c10::optional<at::Tensor>& src,
                         const c10::optional<at::Tensor>& dst_map, const c10::optional<at::Tensor>& nrows,
                         at::Tensor w, at::Tensor sc, at::Tensor err) {
  c10::cuda::CUDAGuard g(master.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int D = (int)master.size(1);
  const int64_t R = w.size(0);
  TORCH_CHECK(D % 32 == 0 && D % 4 == 0, "row width must be a multiple of 32");
  const int64_t M = master.size(0), N = src.has_value() ? src->numel() : M;
  if (N == 0) return;
  quantize_table_kernel<<<(unsigned)((N + 7) / 8), 256, 0, stream>>>(
      master.data_ptr<float>(), src.has_value() ? src->data_ptr<int64_t>() : nullptr,
      dst_map.has_value() ? dst_map->data_ptr<int64_t>() : nullptr,
      nrows.has_value() ? nrows->data_ptr<int32_t>() : nullptr, reinterpret_cast<uint8_t*>(w.data_ptr()),
      reinterpret_cast<uint8_t*>(sc.data_ptr()), err.data_ptr<int>(), N, M, R, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lookup_master_cuda(const at::Tensor& w, const at::Tensor& idx, at::Tensor out, at::Tensor err) {
  c10::cuda::CUDAGuard g(w.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = idx.numel(), R = w.size(0);
  int D = (int)w.size(1);
  TORCH_CHECK(D % 4 == 0, "row width must be a multiple of 4");
  if (N == 0) return;
  unsigned blocks = (unsigned)((N + 7) / 8);
  if (out.scalar_type() == at::kBFloat16)
    lookup_master_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(w.data_ptr<float>(), idx.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), N, R, D);
  else if (out.scalar_type() == at::kFloat)
    lookup_master_kernel<float><<<blocks, 256, 0, stream>>>(w.data_ptr<float>(), idx.data_ptr<int64_t>(),
        out.data_ptr<float>(), err.data_ptr<int>(), N, R, D);
  else if (out.scalar_type() == at::kHalf)
    lookup_master_kernel<__half><<<blocks, 256, 0, stream>>>(w.data_ptr<float>(), idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out.data_ptr()), err.data_ptr<int>(), N, R, D);
  else TORCH_CHECK(false, "out dtype must be bf16, fp16 or fp32");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------
// 3. gate + four-stream residual. Block = one (token, stream), 128 threads.
//    Reduction program over the width D (D % 8 == 0, C = D/8 chunks of 8):
//      thread j owns chunks j, j+128, ... ; accumulates serially in chunk order, 8 elements in
//      ascending order inside a chunk (fp32, separate mul and add, start 0);
//      then a warp butterfly (xor 16,8,4,2,1) and a fixed warp merge ((w0+w1)+(w2+w3)).
//    The tree depends only on D, never on token count / grid / occupancy.
// ------------------------------------------------------------------------------------------
constexpr int kGateThreads = 128;
//: Rows (token x stream) up to which the gate forward runs the row-in-registers kernel (placement only).
constexpr int64_t kGateRegRows = 256;

__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ld(const float* p) { return *p; }
template <typename W> __device__ __forceinline__ void st(W* p, float v);
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ void st<float>(float* p, float v) { *p = v; }

__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float (&v)[8]) {
  uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(b[i]);
}
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const float (&v)[8]) {
  uint4 raw;
  __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = __float2bfloat16_rn(v[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}

template <int K>
__device__ __forceinline__ void block_tree(float (&v)[K], float* smem) {
  // smem: [K][4]
#pragma unroll
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v[k] = __fadd_rn(v[k], __shfl_xor_sync(0xffffffffu, v[k], o));
  }
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) smem[k * 4 + warp] = v[k];
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k)
    v[k] = __fadd_rn(__fadd_rn(smem[k * 4 + 0], smem[k * 4 + 1]), __fadd_rn(smem[k * 4 + 2], smem[k * 4 + 3]));
  __syncthreads();
}

struct GateScalars {
  float rstd, S, dot, gate, rh, rk;
};

__device__ __forceinline__ GateScalars gate_finish(const float (&acc)[4], int D, float eps, float scale, bool masked);

// Reads x[t,s,:], key[t,s,:], q/k weights[s,:]; returns the reduced scalars (identical in every thread).
template <typename W>
__device__ GateScalars gate_scalars(const __nv_bfloat16* __restrict__ xr, const __nv_bfloat16* __restrict__ kr,
                                    const W* __restrict__ qw, const W* __restrict__ kw, int D, float eps,
                                    float scale, bool masked, float* smem, float* extra_dg,
                                    const __nv_bfloat16* __restrict__ dor, const __nv_bfloat16* __restrict__ vr) {
  float acc[4] = {0.f, 0.f, 0.f, 0.f};  // sum h^2, sum k^2, sum h*w*k, sum dout*v
  int C = D / 8;
  for (int c = threadIdx.x; c < C; c += kGateThreads) {
    float H[8], K[8], Q[8], KW[8], DO[8], VV[8];
    ld8(xr + c * 8, H);
    ld8(kr + c * 8, K);
    ld8(qw + c * 8, Q);
    ld8(kw + c * 8, KW);
    if (dor != nullptr) {
      ld8(dor + c * 8, DO);
      ld8(vr + c * 8, VV);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float h = H[i], k = K[i];
      float w = __fmul_rn(Q[i], KW[i]);
      acc[0] = __fadd_rn(acc[0], __fmul_rn(h, h));
      acc[1] = __fadd_rn(acc[1], __fmul_rn(k, k));
      acc[2] = __fadd_rn(acc[2], __fmul_rn(__fmul_rn(h, w), k));
      if (dor != nullptr) acc[3] = __fadd_rn(acc[3], __fmul_rn(DO[i], VV[i]));
    }
  }
  block_tree<4>(acc, smem);
  if (extra_dg) *extra_dg = acc[3];
  return gate_finish(acc, D, eps, scale, masked);
}

__device__ __forceinline__ GateScalars gate_finish(const float (&acc)[4], int D, float eps, float scale, bool masked) {
  GateScalars r;
  float fD = (float)D;
  float mh = __fdiv_rn(acc[0], fD), mk = __fdiv_rn(acc[1], fD);
  r.rh = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mh, eps)));
  r.rk = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mk, eps)));
  r.rstd = __fmul_rn(r.rh, r.rk);
  r.S = acc[2];
  r.dot = __fmul_rn(__fmul_rn(acc[2], r.rstd), scale);
  float a = fmaxf(fabsf(r.dot), 1e-6f);
  float u = copysignf(__fsqrt_rn(a), r.dot);
  float g = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-u)));
  r.gate = masked ? 0.f : g;
  return r;
}

template <typename W>
__global__ void __launch_bounds__(kGateThreads)
gate_fwd_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ key,
                const __nv_bfloat16* __restrict__ val, const W* __restrict__ qw, const W* __restrict__ kw,
                const bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, float* __restrict__ gate_out,
                int S, int D, float eps, float scale, int64_t kts, int64_t kss, int64_t vts) {
  __shared__ float smem[16];
  int64_t ts = blockIdx.x;  // token * S + stream
  int64_t t = ts / S;
  int s = (int)(ts % S);
  const __nv_bfloat16* xr = x + ts * D;
  const __nv_bfloat16* kr = key + t * kts + s * kss;   // key / value may be strided views (last dim contiguous)
  bool masked = mask != nullptr && !mask[t];
  GateScalars r = gate_scalars<W>(xr, kr, qw + (int64_t)s * D, kw + (int64_t)s * D, D, eps, scale, masked, smem,
                                  nullptr, nullptr, nullptr);
  if (threadIdx.x == 0 && gate_out) gate_out[ts] = r.gate;
  const __nv_bfloat16* vr = val + t * vts;
  for (int c = threadIdx.x; c < D / 8; c += kGateThreads) {
    float H[8], VV[8], O[8];
    ld8(xr + c * 8, H);
    ld8(vr + c * 8, VV);
#pragma unroll
    for (int i = 0; i < 8; ++i) O[i] = __fadd_rn(H[i], __fmul_rn(r.gate, VV[i]));
    st8(out + ts * D + c * 8, O);
  }
}

// The forward with the row held in registers (D / 8 <= NC * kGateThreads): the SAME per-thread chunk order
// (c = tid, tid + 128, ...), the same per-element accumulation, the same block_tree and gate_finish as gate_scalars,
// so the gate and the output are bit-identical to gate_fwd_kernel; the value row is loaded with the first pass and
// x is not read twice (one dependent global round trip fewer per row).
template <typename W, int NC>
__global__ void __launch_bounds__(kGateThreads)
gate_fwd_reg_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ key,
                    const __nv_bfloat16* __restrict__ val, const W* __restrict__ qw, const W* __restrict__ kw,
                    const bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, float* __restrict__ gate_out,
                    int S, int D, float eps, float scale, int64_t kts, int64_t kss, int64_t vts) {
  __shared__ float smem[16];
  int64_t ts = blockIdx.x;
  int64_t t = ts / S;
  int s = (int)(ts % S);
  const __nv_bfloat16* xr = x + ts * D;
  const __nv_bfloat16* kr = key + t * kts + s * kss;
  const __nv_bfloat16* vr = val + t * vts;
  const W* qws = qw + (int64_t)s * D;
  const W* kws = kw + (int64_t)s * D;
  bool masked = mask != nullptr && !mask[t];
  const int C = D / 8;
  float Hc[NC][8], Vc[NC][8];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = threadIdx.x + j * kGateThreads;
    if (c < C) {
      float K[8], Q[8], KW[8];
      ld8(xr + c * 8, Hc[j]);
      ld8(kr + c * 8, K);
      ld8(qws + c * 8, Q);
      ld8(kws + c * 8, KW);
      ld8(vr + c * 8, Vc[j]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float h = Hc[j][i], k = K[i];
        float w = __fmul_rn(Q[i], KW[i]);
        acc[0] = __fadd_rn(acc[0], __fmul_rn(h, h));
        acc[1] = __fadd_rn(acc[1], __fmul_rn(k, k));
        acc[2] = __fadd_rn(acc[2], __fmul_rn(__fmul_rn(h, w), k));
      }
    }
  }
  block_tree<4>(acc, smem);
  GateScalars r = gate_finish(acc, D, eps, scale, masked);
  if (threadIdx.x == 0 && gate_out) gate_out[ts] = r.gate;
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = threadIdx.x + j * kGateThreads;
    if (c < C) {
      float O[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) O[i] = __fadd_rn(Hc[j][i], __fmul_rn(r.gate, Vc[j][i]));
      st8(out + ts * D + c * 8, O);
    }
  }
}

// The forward with the five operand rows brought into shared memory by bulk copies (one elected thread issues
// cp.async.bulk for x, key, value, q and k weight rows; one mbarrier with the transaction bytes): the latency-bound
// small-batch form, whose time is the memory round trip, not arithmetic. Then gate_scalars' program exactly
// (per-thread chunks c = tid, tid + 128, ..; per-element accumulation; block_tree; gate_finish) from shared memory:
// bit-identical to gate_fwd_kernel. Requires 16-byte aligned rows (checked by the host).
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes),
                  "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

template <typename W>
__global__ void __launch_bounds__(kGateThreads)
gate_fwd_bulk_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ key,
                     const __nv_bfloat16* __restrict__ val, const W* __restrict__ qw, const W* __restrict__ kw,
                     const bool* __restrict__ mask, __nv_bfloat16* __restrict__ out, float* __restrict__ gate_out,
                     int S, int D, float eps, float scale, int64_t kts, int64_t kss, int64_t vts) {
  extern __shared__ __align__(128) unsigned char dsm[];
  __shared__ float smem[16];
  __shared__ __align__(8) uint64_t bar;
  int64_t ts = blockIdx.x;
  int64_t t = ts / S;
  int s = (int)(ts % S);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(dsm);
  __nv_bfloat16* ks = xs + D;
  __nv_bfloat16* vs = ks + D;
  W* qs = reinterpret_cast<W*>(vs + D);
  W* kws = qs + D;
  const uint32_t rb = (uint32_t)D * 2, wb = (uint32_t)D * (uint32_t)sizeof(W);
  if (threadIdx.x == 0) {
    const uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(3 * rb + 2 * wb) : "memory");
    bulk_g2s(xs, x + ts * D, rb, &bar);
    bulk_g2s(ks, key + t * kts + s * kss, rb, &bar);
    bulk_g2s(vs, val + t * vts, rb, &bar);
    bulk_g2s(qs, qw + (int64_t)s * D, wb, &bar);
    bulk_g2s(kws, kw + (int64_t)s * D, wb, &bar);
  }
  __syncthreads();
  {
    const uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar);
    uint32_t done = 0;
    while (!done) {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(b) : "memory");
    }
  }
  bool masked = mask != nullptr && !mask[t];
  GateScalars r = gate_scalars<W>(xs, ks, qs, kws, D, eps, scale, masked, smem, nullptr, nullptr, nullptr);
  if (threadIdx.x == 0 && gate_out) gate_out[ts] = r.gate;
  for (int c = threadIdx.x; c < D / 8; c += kGateThreads) {
    float H[8], VV[8], O[8];
    ld8(xs + c * 8, H);
    ld8(vs + c * 8, VV);
#pragma unroll
    for (int i = 0; i < 8; ++i) O[i] = __fadd_rn(H[i], __fmul_rn(r.gate, VV[i]));
    st8(out + ts * D + c * 8, O);
  }
}

// backward A: per (token, stream). Recomputes the forward scalars with the identical reduction.
template <typename W>
__global__ void __launch_bounds__(kGateThreads)
gate_bwd_a_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ key,
                  const __nv_bfloat16* __restrict__ val, const W* __restrict__ qw, const W* __restrict__ kw,
                  const bool* __restrict__ mask, const __nv_bfloat16* __restrict__ dout,
                  __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dkey,
                  float* __restrict__ gate_out, float* __restrict__ dS_out, int S, int D, float eps,
                  float scale, int64_t kts, int64_t kss, int64_t vts) {
  __shared__ float smem[16];
  int64_t ts = blockIdx.x;
  int64_t t = ts / S;
  int s = (int)(ts % S);
  const __nv_bfloat16* xr = x + ts * D;
  const __nv_bfloat16* kr = key + t * kts + s * kss;
  const __nv_bfloat16* dor = dout + ts * D;
  const W* qr = qw + (int64_t)s * D;
  const W* kwr = kw + (int64_t)s * D;
  bool masked = mask != nullptr && !mask[t];
  float dg;
  GateScalars r = gate_scalars<W>(xr, kr, qr, kwr, D, eps, scale, masked, smem, &dg, dor, val + t * vts);
  float ddot = 0.f;
  if (!masked) {
    float a = fabsf(r.dot);
    float du = (a >= 1e-6f) ? __fdiv_rn(0.5f, __fsqrt_rn(a)) : 0.f;
    float g = r.gate;
    ddot = __fmul_rn(__fmul_rn(dg, __fmul_rn(g, __fsub_rn(1.f, g))), du);
  }
  float dS = __fmul_rn(__fmul_rn(ddot, r.rstd), scale);
  float dS_ = __fmul_rn(ddot, r.S);  // d wrt (rh*rk) times S, before scale
  float drh = __fmul_rn(__fmul_rn(dS_, r.rk), scale);
  float drk = __fmul_rn(__fmul_rn(dS_, r.rh), scale);
  float dmh = __fmul_rn(drh, __fmul_rn(-0.5f, __fmul_rn(__fmul_rn(r.rh, r.rh), r.rh)));
  float dmk = __fmul_rn(drk, __fmul_rn(-0.5f, __fmul_rn(__fmul_rn(r.rk, r.rk), r.rk)));
  float ch = __fmul_rn(dmh, __fdiv_rn(2.f, (float)D));
  float ck = __fmul_rn(dmk, __fdiv_rn(2.f, (float)D));
  if (threadIdx.x == 0) {
    gate_out[ts] = r.gate;
    dS_out[ts] = dS;
  }
  for (int c = threadIdx.x; c < D / 8; c += kGateThreads) {
    float H[8], K[8], Q[8], KW[8], DO[8], DX[8], DK[8];
    ld8(xr + c * 8, H);
    ld8(kr + c * 8, K);
    ld8(qr + c * 8, Q);
    ld8(kwr + c * 8, KW);
    ld8(dor + c * 8, DO);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float h = H[i], k = K[i];
      float w = __fmul_rn(Q[i], KW[i]);
      DX[i] = __fadd_rn(__fadd_rn(DO[i], __fmul_rn(dS, __fmul_rn(w, k))), __fmul_rn(ch, h));
      DK[i] = __fadd_rn(__fmul_rn(dS, __fmul_rn(h, w)), __fmul_rn(ck, k));
    }
    st8(dx + ts * D + c * 8, DX);
    st8(dkey + ts * D + c * 8, DK);
  }
}

// dvalue[t,d] = ((g0*do0 + g1*do1) + g2*do2) + g3*do3 : ascending stream order (8 columns per thread)
__global__ void gate_dv_kernel(const __nv_bfloat16* __restrict__ dout, const float* __restrict__ gate,
                               __nv_bfloat16* __restrict__ dv, int64_t T, int S, int D) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (token, 8-column group)
  int G = D / 8;
  if (i >= T * G) return;
  int64_t t = i / G;
  int c = (int)(i % G);
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int s = 0; s < S; ++s) {
    float DO[8];
    ld8(dout + (t * S + s) * D + c * 8, DO);
    float g = gate[t * S + s];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = __fadd_rn(acc[j], __fmul_rn(g, DO[j]));
  }
  st8(dv + t * D + c * 8, acc);
}

// weight-gradient partials: ws[seg, s*D+d] = sum over tokens of segment (ascending) dS[t,s]*(h*key)
// (8 adjacent columns per thread; every column's order is token-ascending and independent of the vector width)
constexpr int kSegT = 64;
__global__ void gate_dw_seg_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ key,
                                   const float* __restrict__ dS, float* __restrict__ ws, int64_t T, int S,
                                   int D, int64_t kts, int64_t kss) {
  int64_t grp = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (s*D + d) / 8
  if (grp * 8 >= (int64_t)S * D) return;
  int64_t col = grp * 8;
  int s = (int)(col / D);
  int d = (int)(col % D);
  int64_t seg = blockIdx.y;
  int64_t t0 = seg * kSegT, t1 = min(T, t0 + kSegT);
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int64_t t = t0; t < t1; ++t) {
    float H[8], K[8];
    ld8(x + (t * S + s) * D + d, H);
    ld8(key + t * kts + s * kss + d, K);
    float g = dS[t * S + s];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = __fadd_rn(acc[j], __fmul_rn(g, __fmul_rn(H[j], K[j])));
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) ws[seg * (int64_t)S * D + col + j] = acc[j];
}

// balanced pairwise tree over segments, odd tail carried; then dq = dw*k_weight, dk = dw*q_weight
template <typename W>
__global__ void gate_dw_tree_kernel(float* __restrict__ ws, const W* __restrict__ qw, const W* __restrict__ kw,
                                    W* __restrict__ dq, W* __restrict__ dk, int64_t nseg, int64_t cols) {
  int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) return;
  for (int64_t stride = 1; stride < nseg; stride *= 2)
    for (int64_t i = 0; i + stride < nseg; i += 2 * stride)
      ws[i * cols + col] = __fadd_rn(ws[i * cols + col], ws[(i + stride) * cols + col]);
  float dw = nseg > 0 ? ws[col] : 0.f;
  st<W>(dq + col, __fmul_rn(dw, ld(kw + col)));
  st<W>(dk + col, __fmul_rn(dw, ld(qw + col)));
}

template <typename W>
static const W* wptr(const at::Tensor& t) {
  return reinterpret_cast<const W*>(t.data_ptr());
}

#define DISPATCH_W(wt, FN, ...)                                                                     \
  do {                                                                                              \
    if ((wt) == at::kBFloat16) { using Wt = __nv_bfloat16; FN<Wt>(__VA_ARGS__); }                   \
    else if ((wt) == at::kFloat) { using Wt = float; FN<Wt>(__VA_ARGS__); }                         \
    else TORCH_CHECK(false, "q/k weights must be bf16 or fp32");                                    \
  } while (0)

template <typename W>
static void gate_fwd_launch(const at::Tensor& x, const at::Tensor& key, const at::Tensor& val,
                            const at::Tensor& qw, const at::Tensor& kw, const c10::optional<at::Tensor>& mask,
                            at::Tensor out, at::Tensor gate, double eps, double scale) {
  int S = (int)x.size(-2), D = (int)x.size(-1);
  int64_t n = x.numel() / D;
  const int64_t kts = key.stride(-3), kss = key.stride(-2), vts = val.stride(-2);
  if (n == 0) return;
  // Kernel choice is placement only (both run gate_scalars' program). The row-in-registers kernel holds 2 x NC x 8
  // floats per thread: it wins while the launch is latency-bound (few rows) and loses occupancy above that
  // (measured, B200: M = 1 / 32 rows of 4 streams 3.6 / 3.8 us vs 3.9 / 4.3; M = 128 7.2 vs 5.5 with NC = 8).
  const size_t bulk_smem = (size_t)D * (3 * 2 + 2 * sizeof(W));
  const bool aligned16 = ((reinterpret_cast<uintptr_t>(x.data_ptr()) | reinterpret_cast<uintptr_t>(key.data_ptr()) |
                           reinterpret_cast<uintptr_t>(val.data_ptr()) | reinterpret_cast<uintptr_t>(qw.data_ptr()) |
                           reinterpret_cast<uintptr_t>(kw.data_ptr())) & 15) == 0 &&
                         ((kts | kss | vts) * 2) % 16 == 0 && (D * 2) % 16 == 0;
  if (n <= kGateRegRows && aligned16 && bulk_smem <= 200 * 1024 && qw.is_contiguous() && kw.is_contiguous()) {
    auto kern = gate_fwd_bulk_kernel<W>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bulk_smem));
    kern<<<(unsigned)n, kGateThreads, bulk_smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()), wptr<W>(qw), wptr<W>(kw),
        mask ? mask->data_ptr<bool>() : nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        gate.data_ptr<float>(), S, D, (float)eps, (float)scale, kts, kss, vts);
    return;
  }
  if (n <= kGateRegRows && D / 8 <= 5 * kGateThreads) {
    gate_fwd_reg_kernel<W, 5><<<(unsigned)n, kGateThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()), wptr<W>(qw), wptr<W>(kw),
        mask ? mask->data_ptr<bool>() : nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        gate.data_ptr<float>(), S, D, (float)eps, (float)scale, kts, kss, vts);
    return;
  }
  gate_fwd_kernel<W><<<(unsigned)n, kGateThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()), wptr<W>(qw), wptr<W>(kw),
      mask ? mask->data_ptr<bool>() : nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      gate.data_ptr<float>(), S, D, (float)eps, (float)scale, kts, kss, vts);
}

void gate_fwd_cuda(const at::Tensor& x, const at::Tensor& key, const at::Tensor& val, const at::Tensor& qw,
                   const at::Tensor& kw, const c10::optional<at::Tensor>& mask, at::Tensor out, at::Tensor gate,
                   double eps, double scale) {
  c10::cuda::CUDAGuard g(x.device());
  TORCH_CHECK(x.size(-1) % 8 == 0, "width must be a multiple of 8");
  DISPATCH_W(qw.scalar_type(), gate_fwd_launch, x, key, val, qw, kw, mask, out, gate, eps, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename W>
static void gate_bwd_launch(const at::Tensor& x, const at::Tensor& key, const at::Tensor& val,
                            const at::Tensor& qw, const at::Tensor& kw, const c10::optional<at::Tensor>& mask,
                            const at::Tensor& dout, at::Tensor dx, at::Tensor dkey, at::Tensor dval,
                            at::Tensor dq, at::Tensor dk, at::Tensor gate, at::Tensor dS, at::Tensor ws,
                            double eps, double scale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int S = (int)x.size(-2), D = (int)x.size(-1);
  int64_t T = x.numel() / ((int64_t)S * D);
  if (T == 0) { dq.zero_(); dk.zero_(); return; }
  const int64_t kts = key.stride(-3), kss = key.stride(-2), vts = val.stride(-2);
  auto bf = [](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  auto bfm = [](at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  gate_bwd_a_kernel<W><<<(unsigned)(T * S), kGateThreads, 0, stream>>>(
      bf(x), bf(key), bf(val), wptr<W>(qw), wptr<W>(kw), mask ? mask->data_ptr<bool>() : nullptr, bf(dout),
      bfm(dx), bfm(dkey), gate.data_ptr<float>(), dS.data_ptr<float>(), S, D, (float)eps, (float)scale, kts, kss, vts);
  gate_dv_kernel<<<(unsigned)((T * (D / 8) + 255) / 256), 256, 0, stream>>>(bf(dout), gate.data_ptr<float>(),
                                                                      bfm(dval), T, S, D);
  int64_t nseg = (T + kSegT - 1) / kSegT, cols = (int64_t)S * D;
  dim3 grid((unsigned)((cols / 8 + 127) / 128), (unsigned)nseg);
  gate_dw_seg_kernel<<<grid, 128, 0, stream>>>(bf(x), bf(key), dS.data_ptr<float>(), ws.data_ptr<float>(), T, S, D, kts, kss);
  gate_dw_tree_kernel<W><<<(unsigned)((cols + 127) / 128), 128, 0, stream>>>(
      ws.data_ptr<float>(), wptr<W>(qw), wptr<W>(kw), reinterpret_cast<W*>(dq.data_ptr()),
      reinterpret_cast<W*>(dk.data_ptr()), nseg, cols);
}

void gate_bwd_cuda(const at::Tensor& x, const at::Tensor& key, const at::Tensor& val, const at::Tensor& qw,
                   const at::Tensor& kw, const c10::optional<at::Tensor>& mask, const at::Tensor& dout,
                   at::Tensor dx, at::Tensor dkey, at::Tensor dval, at::Tensor dq, at::Tensor dk,
                   at::Tensor gate, at::Tensor dS, at::Tensor ws, double eps, double scale) {
  c10::cuda::CUDAGuard g(x.device());
  TORCH_CHECK(x.size(-1) % 8 == 0, "width must be a multiple of 8");
  DISPATCH_W(qw.scalar_type(), gate_bwd_launch, x, key, val, qw, kw, mask, dout, dx, dkey, dval, dq, dk, gate,
             dS, ws, eps, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------
// 4. table-row gradient, duplicate-index reduction without atomics.
//    sort (stable LSD radix, u32 key, i32 position) -> run-length encode -> exclusive sum -> per-run reduce.
//    Per-run order: 8 warp-stripes; stripe w sums sorted entries w, w+8, w+16, ... (ascending original
//    position, since the sort is stable) sequentially in fp32; stripes merged ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)).
// ------------------------------------------------------------------------------------------
__global__ void idx_to_u32_kernel(const int64_t* __restrict__ idx, uint32_t* __restrict__ keys,
                                  int32_t* __restrict__ pos, int* __restrict__ err, int64_t N, int64_t R) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int64_t r = idx[i];
  if (r < 0 || r >= R) { *err = 1; r = 0; }
  keys[i] = (uint32_t)r;
  pos[i] = (int32_t)i;
}

template <typename G>
__device__ __forceinline__ float gld(const G* p) { return ld(p); }

// 8 consecutive elements (16B for bf16, 2 x 16B for fp32; callers guarantee alignment: D % 8 == 0)
__device__ __forceinline__ void gld8(const __nv_bfloat16* p, float (&v)[8]) {
  uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(b[i]);
}
__device__ __forceinline__ void gld4(const __nv_bfloat16* p, float (&v)[4]) {
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(b[i]);
}
__device__ __forceinline__ void gld4(const float* p, float (&v)[4]) {
  float4 a = *reinterpret_cast<const float4*>(p);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
}
__device__ __forceinline__ void gld8(const float* p, float (&v)[8]) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// Both kernels implement the same declared program (stripe w = entries w, w+8, ... in sorted order; merge
// ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))); they differ only in which hardware executes a stripe, so their
// outputs are bit-identical. Short runs (cnt <= kShortRun, the common case): one warp per run, the eight
// stripes live in registers. Long runs: one block per run, one warp per stripe, merged through smem.
constexpr int kShortRun = 64;

// Singleton runs (cnt == 1, the overwhelmingly common case for hashed buckets): the program degenerates to
// p0 = 0 + x, p1..p7 = 0, merged by the same tree. Four runs per warp keep four independent gathers in flight.
template <typename G>
__global__ void __launch_bounds__(256)
run_reduce_single_kernel(const G* __restrict__ grad, const int32_t* __restrict__ pos,
                         const uint32_t* __restrict__ ukeys, const int32_t* __restrict__ counts,
                         const int32_t* __restrict__ offsets, const int32_t* __restrict__ nruns,
                         float* __restrict__ gout, int64_t* __restrict__ rows_out, int D) {
  int lane = threadIdx.x & 31;
  int u0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 4;
  int total = nruns[0];
  int cnt[4], p[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int u = u0 + k;
    cnt[k] = u < total ? counts[u] : 0;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) p[k] = cnt[k] == 1 ? pos[offsets[u0 + k]] : 0;
  float v[4][2][4];
  bool on0 = lane * 4 < D, on1 = 128 + lane * 4 < D;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const G* row = grad + (int64_t)p[k] * D + lane * 4;
    if (cnt[k] == 1 && on0) gld4(row, v[k][0]);
    if (cnt[k] == 1 && on1) gld4(row + 128, v[k][1]);
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    if (cnt[k] != 1) continue;
    int u = u0 + k;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (h == 0 ? on0 : on1) {
        float r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float p0 = __fadd_rn(0.f, v[k][h][i]);
          r[i] = __fadd_rn(__fadd_rn(__fadd_rn(p0, 0.f), 0.f), 0.f);
        }
        *reinterpret_cast<float4*>(gout + (int64_t)u * D + h * 128 + lane * 4) = make_float4(r[0], r[1], r[2], r[3]);
      }
    }
    if (lane == 0) rows_out[u] = (int64_t)ukeys[u];
  }
}

template <typename G>
__global__ void __launch_bounds__(256)
run_reduce_short_kernel(const G* __restrict__ grad, const int32_t* __restrict__ pos,
                        const uint32_t* __restrict__ ukeys, const int32_t* __restrict__ counts,
                        const int32_t* __restrict__ offsets, const int32_t* __restrict__ nruns,
                        float* __restrict__ gout, int64_t* __restrict__ rows_out, int D) {
  // block j owns runs [256 j, 256 j + 256); warps take the flagged (2 <= cnt <= kShortRun) ones in turn
  __shared__ int flags[256];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int total = nruns[0];
  int base = blockIdx.x * 256;
  if (base >= total) return;
  int u0 = base + threadIdx.x;
  int c0 = u0 < total ? counts[u0] : 0;
  int f = (c0 >= 2 && c0 <= kShortRun) ? 1 : 0;
  flags[threadIdx.x] = f;
  if (__syncthreads_count(f) == 0) return;
  for (int r = warp; r < 256; r += 8) {
  if (!flags[r]) continue;
  int u = base + r;
  int cnt = counts[u];
  int off = offsets[u];
  // lane owns columns [4 lane, 4 lane + 4) and [128 + 4 lane, ...) : contiguous 4-wide vectors across the warp
  float acc[8][2][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][h][i] = 0.f;
  bool on0 = lane * 4 < D, on1 = 128 + lane * 4 < D;
  for (int e0 = 0; e0 < cnt; e0 += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (e0 + j < cnt) {
        const G* row = grad + (int64_t)pos[off + e0 + j] * D + lane * 4;
        float v[4];
        if (on0) {
          gld4(row, v);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[j][0][i] = __fadd_rn(acc[j][0][i], v[i]);
        }
        if (on1) {
          gld4(row + 128, v);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[j][1][i] = __fadd_rn(acc[j][1][i], v[i]);
        }
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (h == 0 ? on0 : on1) {
      float r[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        r[i] = __fadd_rn(__fadd_rn(__fadd_rn(acc[0][h][i], acc[1][h][i]), __fadd_rn(acc[2][h][i], acc[3][h][i])),
                         __fadd_rn(__fadd_rn(acc[4][h][i], acc[5][h][i]), __fadd_rn(acc[6][h][i], acc[7][h][i])));
      *reinterpret_cast<float4*>(gout + (int64_t)u * D + h * 128 + lane * 4) = make_float4(r[0], r[1], r[2], r[3]);
    }
  }
  if (lane == 0) rows_out[u] = (int64_t)ukeys[u];
  }
}

template <typename G>
__device__ __forceinline__ void reduce_long_run(int u, const G* __restrict__ grad, const int32_t* __restrict__ pos,
                                                const uint32_t* __restrict__ ukeys, const int32_t* __restrict__ counts,
                                                const int32_t* __restrict__ offsets, float* __restrict__ gout,
                                                int64_t* __restrict__ rows_out, int D, float (*part)[512 + 8]) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int off = offsets[u], cnt = counts[u];
  int groups = D / 8;                       // 8-wide groups; lane handles groups lane, lane+32 (D <= 512)
  float acc[2][8];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;
  for (int e = warp; e < cnt; e += 8) {
    const G* row = grad + (int64_t)pos[off + e] * D;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int gi = lane + 32 * j;
      if (gi < groups) {
        float v[8];
        gld8(row + gi * 8, v);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[j][i] = __fadd_rn(acc[j][i], v[i]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int gi = lane + 32 * j;
    if (gi < groups)
#pragma unroll
      for (int i = 0; i < 8; ++i) part[warp][gi * 8 + i] = acc[j][i];
  }
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += 256) {
    float a = __fadd_rn(__fadd_rn(part[0][d], part[1][d]), __fadd_rn(part[2][d], part[3][d]));
    float b = __fadd_rn(__fadd_rn(part[4][d], part[5][d]), __fadd_rn(part[6][d], part[7][d]));
    gout[(int64_t)u * D + d] = __fadd_rn(a, b);
  }
  if (threadIdx.x == 0) rows_out[u] = (int64_t)ukeys[u];
  __syncthreads();
}

// all_long == 0: block j owns runs [256 j, 256 j + 256) and handles those with cnt > kShortRun;
// all_long == 1 (D > 256, no warp path): grid-stride over every run.
template <typename G>
__global__ void __launch_bounds__(256)
run_reduce_long_kernel(const G* __restrict__ grad, const int32_t* __restrict__ pos,
                       const uint32_t* __restrict__ ukeys, const int32_t* __restrict__ counts,
                       const int32_t* __restrict__ offsets, const int32_t* __restrict__ nruns,
                       float* __restrict__ gout, int64_t* __restrict__ rows_out, int D, int all_long) {
  __shared__ float part[8][512 + 8];
  __shared__ int flags[256];
  int total = nruns[0];
  if (all_long) {
    for (int u = blockIdx.x; u < total; u += gridDim.x)
      reduce_long_run<G>(u, grad, pos, ukeys, counts, offsets, gout, rows_out, D, part);
    return;
  }
  int base = blockIdx.x * 256;
  if (base >= total) return;
  int u0 = base + threadIdx.x;
  int f = (u0 < total && counts[u0] > kShortRun) ? 1 : 0;
  flags[threadIdx.x] = f;
  if (__syncthreads_count(f) == 0) return;
  for (int r = 0; r < 256; ++r)
    if (flags[r]) reduce_long_run<G>(base + r, grad, pos, ukeys, counts, offsets, gout, rows_out, D, part);
}

template <typename G>
static void run_reduce(const at::Tensor& grad, const at::Tensor& pos_out, const uint32_t* uk,
                       const at::Tensor& counts, const at::Tensor& offsets, const at::Tensor& nruns,
                       at::Tensor& g_out, at::Tensor& rows_out, int64_t N, int D, cudaStream_t stream) {
  const G* gp = reinterpret_cast<const G*>(grad.data_ptr());
  bool warp_ok = D <= 256;
  if (warp_ok)
    run_reduce_single_kernel<G><<<(unsigned)((N + 31) / 32), 256, 0, stream>>>(
        gp, pos_out.data_ptr<int32_t>(), uk, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
        nruns.data_ptr<int32_t>(), g_out.data_ptr<float>(), rows_out.data_ptr<int64_t>(), D);
  if (warp_ok)
    run_reduce_short_kernel<G><<<(unsigned)((N + 255) / 256), 256, 0, stream>>>(
        gp, pos_out.data_ptr<int32_t>(), uk, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
        nruns.data_ptr<int32_t>(), g_out.data_ptr<float>(), rows_out.data_ptr<int64_t>(), D);
  unsigned blocks = warp_ok ? (unsigned)((N + 255) / 256) : (unsigned)N;
  run_reduce_long_kernel<G><<<blocks, 256, 0, stream>>>(
      gp, pos_out.data_ptr<int32_t>(), uk, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      nruns.data_ptr<int32_t>(), g_out.data_ptr<float>(), rows_out.data_ptr<int64_t>(), D, warp_ok ? 0 : 1);
}

// dense[rows[u], :] += g[u, :]   (rows unique => one writer per row, no atomics)
__global__ void scatter_rows_kernel(float* __restrict__ dense, const int64_t* __restrict__ rows,
                                    const float* __restrict__ g, const int32_t* __restrict__ nruns, int D) {
  int u = blockIdx.x;
  if (u >= nruns[0]) return;
  float* dst = dense + rows[u] * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    dst[d] = __fadd_rn(dst[d], g[(int64_t)u * D + d]);
}

static at::Tensor scratch(int64_t bytes, const at::Device& dev) {
  return at::empty({std::max<int64_t>(bytes, 1)}, at::TensorOptions().dtype(at::kByte).device(dev));
}

// returns number of bytes of temp storage needed? No: allocate inside via torch's caching allocator.
void table_grad_cuda(const at::Tensor& idx, const at::Tensor& grad, int64_t num_rows, at::Tensor rows_out,
                     at::Tensor g_out, at::Tensor nruns, at::Tensor err) {
  c10::cuda::CUDAGuard g(idx.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = idx.numel();
  int D = (int)grad.size(-1);
  TORCH_CHECK(D % 8 == 0 && D <= 512, "row width must be a multiple of 8 and <= 512");
  TORCH_CHECK(num_rows >= 1 && num_rows <= (int64_t)0xffffffffLL, "table rows must fit u32");
  nruns.zero_();
  if (N == 0) return;
  auto dev = idx.device();
  auto i32 = at::TensorOptions().dtype(at::kInt).device(dev);
  auto u32 = at::TensorOptions().dtype(at::kInt).device(dev);  // stored as 4-byte words
  at::Tensor keys_in = at::empty({N}, u32), keys_out = at::empty({N}, u32);
  at::Tensor pos_in = at::empty({N}, i32), pos_out = at::empty({N}, i32);
  at::Tensor ukeys = at::empty({N}, u32), counts = at::empty({N}, i32), offsets = at::empty({N}, i32);
  idx_to_u32_kernel<<<(unsigned)((N + 255) / 256), 256, 0, stream>>>(
      idx.data_ptr<int64_t>(), reinterpret_cast<uint32_t*>(keys_in.data_ptr()), pos_in.data_ptr<int32_t>(),
      err.data_ptr<int>(), N, num_rows);
  int bits = 1;
  while (bits < 32 && ((uint64_t)1 << bits) < (uint64_t)num_rows) ++bits;
  size_t b1 = 0, b2 = 0, b3 = 0;
  auto* k_in = reinterpret_cast<uint32_t*>(keys_in.data_ptr());
  auto* k_out = reinterpret_cast<uint32_t*>(keys_out.data_ptr());
  auto* uk = reinterpret_cast<uint32_t*>(ukeys.data_ptr());
  cub::DeviceRadixSort::SortPairs(nullptr, b1, k_in, k_out, pos_in.data_ptr<int32_t>(), pos_out.data_ptr<int32_t>(),
                                  (int)N, 0, bits, stream);
  cub::DeviceRunLengthEncode::Encode(nullptr, b2, k_out, uk, counts.data_ptr<int32_t>(), nruns.data_ptr<int32_t>(),
                                     (int)N, stream);
  cub::DeviceScan::ExclusiveSum(nullptr, b3, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), (int)N, stream);
  at::Tensor tmp = scratch((int64_t)std::max({b1, b2, b3}), dev);
  b1 = b2 = b3 = (size_t)tmp.numel();
  cub::DeviceRadixSort::SortPairs(tmp.data_ptr(), b1, k_in, k_out, pos_in.data_ptr<int32_t>(),
                                  pos_out.data_ptr<int32_t>(), (int)N, 0, bits, stream);
  cub::DeviceRunLengthEncode::Encode(tmp.data_ptr(), b2, k_out, uk, counts.data_ptr<int32_t>(),
                                     nruns.data_ptr<int32_t>(), (int)N, stream);
  cub::DeviceScan::ExclusiveSum(tmp.data_ptr(), b3, counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), (int)N,
                                stream);
  if (grad.scalar_type() == at::kBFloat16)
    run_reduce<__nv_bfloat16>(grad, pos_out, uk, counts, offsets, nruns, g_out, rows_out, N, D, stream);
  else if (grad.scalar_type() == at::kFloat)
    run_reduce<float>(grad, pos_out, uk, counts, offsets, nruns, g_out, rows_out, N, D, stream);
  else TORCH_CHECK(false, "grad must be bf16 or fp32");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_rows_cuda(at::Tensor dense, const at::Tensor& rows, const at::Tensor& g, const at::Tensor& nruns) {
  c10::cuda::CUDAGuard gd(dense.device());
  int64_t N = rows.numel();
  if (N == 0) return;
  scatter_rows_kernel<<<(unsigned)N, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      dense.data_ptr<float>(), rows.data_ptr<int64_t>(), g.data_ptr<float>(), nruns.data_ptr<int32_t>(),
      (int)dense.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace engram
