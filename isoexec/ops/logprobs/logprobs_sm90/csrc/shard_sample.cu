// SPDX-License-Identifier: Apache-2.0
// Vocab-parallel sampling head: a second extension built from this directory (logprobs_sm90/_shard_extension.py).
// It includes logprobs.cu so the leaf states and the row merge are that file's code, not a copy.
//
// The gathered-vocabulary path all-gathers the [rows, V] bf16 logits, widens them to fp32 and runs vLLM's
// Gumbel-max sampler, the full-row log-softmax (leaf8 recipe) and vLLM's rank count over the whole row on
// every rank. This head works on each rank's contiguous vocab shard and exchanges O(rows):
//
//   shard_pass   one read of the shard: the recipe's leaf states (leaf_state_body, unchanged) and, per
//                leaf, the Gumbel-max candidate (value, global column, raw logit)
//   all-gather   5 * rows * leaves_local fp32 per rank (host, torch.distributed)
//   shard_finish the recipe's row merge over all G leaves in leaf order (merge_leaves), the winning
//                candidate, its logprob ((v - M) - L32, the full-row writer's expression) and this
//                shard's count of columns >= the winner's raw logit (vLLM's _ranks_kernel predicate)
//   all-reduce   the integer counts (host)
//
// Bits. The Gumbel value of a column is vLLM's (v1/worker/gpu/sample/gumbel.py, Triton 3.8,
// _gumbel_sample_kernel at USE_FP64=False, APPLY_TEMPERATURE=False, no logits cache): murmur3 of
// (seed, pos, global column), the two-halves uniform, _log1p_neg_stable and libdevice logf as Triton
// compiles them (fma.rn.ftz in logf, flush-free elsewhere), each instruction written explicitly below.
// vLLM's argmax is "largest value, lowest column on ties" (Triton tie_break_left inside a 1024-column
// block, then torch.argmax's first maximum across blocks); for non-NaN values that is a total order,
// so the per-leaf candidates combined in any order give vLLM's token. A row whose values contain NaN
// has no order-free argmax (Triton's tree decides); this path then picks among the non-NaN columns.
// A row that is NaN in every column has no candidate at all: it takes ``nan_col``, the column vLLM's
// gather path returns for an all-NaN row of this vocabulary (probed once at engine init, see
// runtimes/vllm/vllm_patches.prepare_v2_shard_head), with a NaN raw logit (logprob NaN, rank count 0, as
// the gathered path computes them).
// The leaf states and the merge are the recipe's own code, so the logprob is the full-row writer's
// value at the sampled column (and the trainer's sampled path value).

#include "logprobs.cu"

namespace logprobs_sm90 {
namespace {

struct GumbelBest {
  float v;    // Gumbel value (or the raw logit at temperature 0)
  int idx;    // global column
  float raw;  // raw logit (fp32 widened)
};

__device__ __forceinline__ bool gumbel_better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

__device__ __forceinline__ uint32_t murmur_mix(uint32_t h, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}

__device__ __forceinline__ uint32_t murmur_fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}

__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// libdevice __nv_logf under Triton's ftz reflect, as Triton 3.8 lowers it for sm_90.
__device__ __forceinline__ float tl_logf(float a) {
  float const neg = -a;
  bool const den = neg > -1.17549435e-38f;  // a < FLT_MIN: scale into the normal range
  float const m = den ? __fmul_rn(neg, -8388608.f) : a;
  float const i0 = den ? -23.f : 0.f;
  int const mb = __float_as_int(m);
  int const e = (mb - 0x3F2AAAAB) & int(0xFF800000u);
  float const i = fma_ftz(float(e), 1.1920928955078125e-07f, i0);
  float const f = __fadd_rn(__int_as_float(mb - e), -1.f);
  float r = fma_ftz(__int_as_float(0xBE055027), f, __int_as_float(0x3E1039F6));
  r = fma_ftz(r, f, __int_as_float(0xBDF8CDCC));
  r = fma_ftz(r, f, __int_as_float(0x3E0F2955));
  r = fma_ftz(r, f, __int_as_float(0xBE2AD8B9));
  r = fma_ftz(r, f, __int_as_float(0x3E4CED0B));
  r = fma_ftz(r, f, __int_as_float(0xBE7FFF22));
  r = fma_ftz(r, f, __int_as_float(0x3EAAAA78));
  r = fma_ftz(r, f, __int_as_float(0xBF000000));
  r = __fmul_rn(f, r);
  r = fma_ftz(r, f, f);
  r = fma_ftz(i, __int_as_float(0x3F317218), r);
  if (!(uint32_t(mb) < 0x7F800000u))
    r = fma_ftz(m, __int_as_float(0x7F800000), __int_as_float(0x7F800000));
  return a == 0.f ? -__builtin_huge_valf() : r;
}

// murmur3_uniform32: the uint32 split into exact 16-bit halves, hi * 2^-16 + (lo + 0.5) * 2^-32.
__device__ __forceinline__ float uniform32(uint32_t r) {
  float const hi = float(r >> 16), lo = float(r & 0xFFFFu);
  return __fmaf_rn(hi, 1.52587890625e-05f, __fmul_rn(__fadd_rn(lo, 0.5f), 2.3283064365386963e-10f));
}

// x + gumbel_noise(u) = x - log(-_log1p_neg_stable(u)).
__device__ __forceinline__ float gumbel_value(float x, uint32_t r) {
  float const u = uniform32(r);
  float l;
  if (u < 0.25f) {
    float p = __fmaf_rn(u, 0.125f, __int_as_float(0x3E124925));
    p = __fmaf_rn(u, p, __int_as_float(0x3E2AAAAB));
    p = __fmaf_rn(u, p, __int_as_float(0x3E4CCCCD));
    p = __fmaf_rn(u, p, 0.25f);
    p = __fmaf_rn(u, p, __int_as_float(0x3EAAAAAB));
    p = __fmaf_rn(u, p, 0.5f);
    p = __fmaf_rn(u, p, 1.f);
    l = __fmul_rn(p, -u);
  } else {
    l = tl_logf(fmaxf(__fsub_rn(1.f, u), 5.960464477539063e-08f));
  }
  return __fsub_rn(x, tl_logf(-l));
}

// Per-row Gumbel stream prefix: murmur3 over (seed lo, seed hi, pos lo); the column is mixed last.
__device__ __forceinline__ uint32_t gumbel_prefix(int64_t seed, int64_t pos) {
  uint32_t h = 0u;
  h = murmur_mix(h, uint32_t(uint64_t(seed) & 0xFFFFFFFFu));
  h = murmur_mix(h, uint32_t((uint64_t(seed) >> 32) & 0xFFFFFFFFu));
  h = murmur_mix(h, uint32_t(uint64_t(pos) & 0xFFFFFFFFu));
  return h;
}

__device__ __forceinline__ uint32_t gumbel_bits(uint32_t prefix, uint32_t col) {
  return murmur_fmix(murmur_mix(prefix, col) ^ 16u);
}

__device__ __forceinline__ void warp_best(GumbelBest &b) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float v = __shfl_xor_sync(0xffffffffu, b.v, o);
    int i = __shfl_xor_sync(0xffffffffu, b.idx, o);
    float r = __shfl_xor_sync(0xffffffffu, b.raw, o);
    if (gumbel_better(v, i, b.v, b.idx)) b = {v, i, r};
  }
}

// One CTA per (row, local leaf): the leaf state (unchanged recipe body) plus the leaf's Gumbel-max
// candidate. ``part`` is one fp32 buffer (the all-gather payload): M [rows][leaves_local], then S
// [rows][leaves_local], then the candidates [rows][leaves_local][3] = {value, column bits, raw}.
template <class T, bool TEMP>
__global__ __launch_bounds__(LANES) void shard_pass(const T *__restrict__ x, int64_t stride, int L, int leaves_local,
                                                    int64_t col0, const int *__restrict__ req_of_row,
                                                    const float *__restrict__ req_temp,
                                                    const int64_t *__restrict__ req_seed,
                                                    const int64_t *__restrict__ row_pos, float *__restrict__ part,
                                                    const float *__restrict__ temp) {
  int const row = blockIdx.x, g = blockIdx.y, rows = gridDim.x;
  int const req = req_of_row[row];
  bool const valid = req >= 0;
  float const rt = valid ? req_temp[req] : 0.f;
  bool const noisy = rt != 0.f;
  uint32_t const prefix = gumbel_prefix(valid ? req_seed[req] : 0, row_pos[row]);
  int const base = int(col0 + int64_t(g) * L);
  GumbelBest best{-__builtin_huge_valf(), 0x7fffffff, 0.f};
  int64_t const n = int64_t(rows) * leaves_local;
  leaf_state_body<T, TEMP, false>(x, stride, L, part, part + n, leaves_local, temp, nullptr, [&](float raw, int c) {
    int const col = base + c;
    float const v = noisy ? gumbel_value(raw, gumbel_bits(prefix, uint32_t(col))) : raw;
    if (gumbel_better(v, col, best.v, best.idx)) best = {v, col, raw};
  });
  __shared__ GumbelBest wb[LANES / 32];
  warp_best(best);
  if ((threadIdx.x & 31) == 0) wb[threadIdx.x >> 5] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    GumbelBest b = wb[0];
    for (int w = 1; w < LANES / 32; ++w)
      if (gumbel_better(wb[w].v, wb[w].idx, b.v, b.idx)) b = wb[w];
    float *slot = part + 2 * n + (int64_t(row) * leaves_local + g) * 3;
    slot[0] = b.v;
    slot[1] = __int_as_float(b.idx);
    slot[2] = b.raw;
  }
}

// One CTA per row: merge all G leaves (gathered: world shard_pass payloads, rank-major = leaf order),
// pick the sampled column, its logprob, and this shard's count of raw logits >= the sampled raw logit.
template <class T, bool TEMP>
__global__ __launch_bounds__(256) void shard_finish(const float *__restrict__ gathered, int world, int leaves_local,
                                                    int rows, const T *__restrict__ x, int64_t stride, int width,
                                                    int64_t *__restrict__ sampled, float *__restrict__ logprob,
                                                    int64_t *__restrict__ count, const float *__restrict__ temp,
                                                    int64_t nan_col) {
  int const row = blockIdx.x;
  __shared__ float s_raw;
  __shared__ int s_cnt[256 / 32];
  if (threadIdx.x == 0) {
    float mg[G], sg[G];
    GumbelBest b{-__builtin_huge_valf(), 0x7fffffff, 0.f};
    for (int gg = 0; gg < G; ++gg) {
      int const r = gg / leaves_local, l = gg % leaves_local;
      int64_t const n = int64_t(rows) * leaves_local;
      const float *part = gathered + int64_t(r) * 5 * n;
      int64_t const k = int64_t(row) * leaves_local + l;
      mg[gg] = part[k];
      sg[gg] = part[n + k];
      const float *p = part + 2 * n + k * 3;
      int const idx = __float_as_int(p[1]);
      if (gumbel_better(p[0], idx, b.v, b.idx)) b = {p[0], idx, p[2]};
    }
    if (b.idx == 0x7fffffff) {
      // Every column NaN (no leaf produced a candidate): the gathered path's token for such a row.
      b.idx = int(nan_col);
      b.raw = __int_as_float(0x7fffffff);
    }
    float M;
    double ls;
    merge_leaves(mg, sg, M, ls);
    float const L32 = float(ls);
    float t = 1.f;
    if constexpr (TEMP)
      t = *temp;
    float const v = TEMP ? __fdiv_rn(b.raw, t) : b.raw;
    sampled[row] = b.idx;
    logprob[row] = __fsub_rn(__fsub_rn(v, M), L32);
    s_raw = b.raw;
  }
  __syncthreads();
  float const xt = s_raw;
  const auto *xs = reinterpret_cast<const pack8<T> *>(x + int64_t(row) * stride);
  int c = 0;
  for (int q = threadIdx.x; q < width / STRIP; q += 256) {
    pack8<T> p = xs[q];
#pragma unroll
    for (int j = 0; j < STRIP; ++j)
      c += widen(p.v[j]) >= xt;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    c += __shfl_xor_sync(0xffffffffu, c, o);
  if ((threadIdx.x & 31) == 0) s_cnt[threadIdx.x >> 5] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int tot = 0;
    for (int w = 0; w < 256 / 32; ++w)
      tot += s_cnt[w];
    count[row] = tot;
  }
}

// Conformance probes (tests only): the Gumbel value of explicit (x, seed, pos, column) tuples, and of
// x = 0 over a contiguous range of raw uint32 hash outputs.
__global__ void gumbel_probe(const float *__restrict__ x, const int64_t *__restrict__ seed,
                             const int64_t *__restrict__ pos, const int64_t *__restrict__ col, float *__restrict__ out,
                             int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gumbel_value(x[i], gumbel_bits(gumbel_prefix(seed[i], pos[i]), uint32_t(col[i])));
}

__global__ void gumbel_range_probe(uint32_t first, float *__restrict__ out, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gumbel_value(0.f, first + uint32_t(i));
}

}  // namespace
}  // namespace logprobs_sm90

using namespace logprobs_sm90;

// Vocab-parallel sampling head, pass 1. x [rows, V_local] holds `leaves_local` whole leaves starting at
// global column col0; part = fp32 [5 * rows * leaves_local].
void shard_pass_cuda(const at::Tensor &x, int64_t leaves_local, int64_t col0, const at::Tensor &req_of_row,
                     const at::Tensor &req_temp, const at::Tensor &req_seed, const at::Tensor &row_pos,
                     at::Tensor part, const c10::optional<at::Tensor> &temperature) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK(leaves_local > 0 && V % (leaves_local * STRIP) == 0, "logprobs_sm90: shard leaf geometry");
  TORCH_CHECK(col0 >= 0 && col0 + V <= (int64_t(1) << 31) - 1, "logprobs_sm90: shard columns exceed int32");
  TORCH_CHECK(req_of_row.dtype() == at::kInt && req_of_row.numel() >= rows && req_of_row.is_contiguous(),
              "logprobs_sm90: req_of_row must be contiguous int32 [rows]");
  TORCH_CHECK(req_temp.dtype() == at::kFloat && req_seed.dtype() == at::kLong && row_pos.dtype() == at::kLong &&
                  row_pos.numel() >= rows && row_pos.is_contiguous() && req_temp.is_contiguous() &&
                  req_seed.is_contiguous(),
              "logprobs_sm90: req_temp fp32, req_seed int64, row_pos int64 [rows], contiguous");
  TORCH_CHECK(part.dtype() == at::kFloat && part.is_contiguous() && part.numel() >= 5 * int64_t(rows) * leaves_local,
              "logprobs_sm90: part must be contiguous fp32 [5 * rows * leaves_local]");
  int L = V / leaves_local;
  const float *tp = temp_ptr(temperature);
  dim3 grid(rows, leaves_local);
  auto stream = at::cuda::getCurrentCUDAStream();
  bool xbf = x.dtype() == at::kBFloat16;
#define PASS(T, TEMP)                                                                                \
  return shard_pass<T, TEMP><<<grid, LANES, 0, stream>>>(                                            \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), L, int(leaves_local), col0,             \
      req_of_row.data_ptr<int>(), req_temp.data_ptr<float>(), req_seed.data_ptr<int64_t>(),           \
      row_pos.data_ptr<int64_t>(), part.data_ptr<float>(), tp);
  if (xbf && tp) PASS(bf16, true)
  if (xbf && !tp) PASS(bf16, false)
  if (!xbf && tp) PASS(float, true)
  PASS(float, false)
#undef PASS
}

// Pass 2: gathered = the world pass-1 payloads, rank-major. Writes sampled (int64), logprob (fp32) and
// this shard's rank count (int64; the host sums it over the group).
void shard_finish_cuda(const at::Tensor &gathered, int64_t world, int64_t leaves_local, const at::Tensor &x,
                       at::Tensor sampled, at::Tensor logprob, at::Tensor count,
                       const c10::optional<at::Tensor> &temperature, int64_t nan_col) {
  check_rows(x);
  const at::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), V = x.size(1);
  if (rows == 0)
    return;
  TORCH_CHECK(world * leaves_local == G, "logprobs_sm90: world * leaves_local must be the recipe's G leaves");
  TORCH_CHECK(nan_col >= 0 && nan_col < world * int64_t(V), "logprobs_sm90: nan_col outside the vocabulary");
  TORCH_CHECK(V % STRIP == 0, "logprobs_sm90: shard width must be a multiple of the strip");
  TORCH_CHECK(gathered.dtype() == at::kFloat && gathered.is_contiguous() &&
                  gathered.numel() >= world * 5 * int64_t(rows) * leaves_local,
              "logprobs_sm90: gathered payload size");
  TORCH_CHECK(sampled.dtype() == at::kLong && count.dtype() == at::kLong && logprob.dtype() == at::kFloat &&
                  sampled.is_contiguous() && count.is_contiguous() && logprob.is_contiguous(),
              "logprobs_sm90: sampled/count int64, logprob fp32, contiguous");
  const float *tp = temp_ptr(temperature);
  bool xbf = x.dtype() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
#define FIN(T, TEMP)                                                                                 \
  return shard_finish<T, TEMP><<<rows, 256, 0, stream>>>(                                            \
      gathered.data_ptr<float>(), int(world), int(leaves_local), rows,                                \
      reinterpret_cast<const T *>(x.data_ptr()), x.stride(0), V, sampled.data_ptr<int64_t>(),         \
      logprob.data_ptr<float>(), count.data_ptr<int64_t>(), tp, nan_col);
  if (xbf && tp) FIN(bf16, true)
  if (xbf && !tp) FIN(bf16, false)
  if (!xbf && tp) FIN(float, true)
  FIN(float, false)
#undef FIN
}

at::Tensor gumbel_probe_cuda(const at::Tensor &x, const at::Tensor &seed, const at::Tensor &pos, const at::Tensor &col) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dtype() == at::kFloat && seed.dtype() == at::kLong && pos.dtype() == at::kLong &&
                  col.dtype() == at::kLong,
              "gumbel_probe: x fp32, seed/pos/col int64");
  auto xc = x.contiguous(), sc = seed.contiguous(), pc = pos.contiguous(), cc = col.contiguous();
  auto out = at::empty_like(xc);
  int64_t n = xc.numel();
  if (n)
    gumbel_probe<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        xc.data_ptr<float>(), sc.data_ptr<int64_t>(), pc.data_ptr<int64_t>(), cc.data_ptr<int64_t>(),
        out.data_ptr<float>(), n);
  return out;
}

void gumbel_range_probe_cuda(int64_t first, at::Tensor out) {
  const at::cuda::CUDAGuard guard(out.device());
  TORCH_CHECK(out.dtype() == at::kFloat && out.is_contiguous(), "gumbel_range_probe: fp32 out");
  int64_t n = out.numel();
  if (n)
    gumbel_range_probe<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(uint32_t(first),
                                                                                         out.data_ptr<float>(), n);
}
