// SPDX-License-Identifier: BSD-3-Clause
// Torch-facing entry point: encode the tensor maps, build the plan on the device, run the leaf, run
// the combine -- or, for a decode-shaped head-dim-256 call, run the decode leaf (`decode_d256.cu`).
// Nothing here chooses a numerical parameter: `seg` is handed in by the caller from the package's
// deployment-fixed table, and every other decision (the leaf, waves, granularity, grid capacity) is
// execution.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <tuple>
#include <vector>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <cmath>

#include "params.h"
#include "launch.h"

//: D256 constants, fitted with the quantised makespan on D256 decode and prefill shapes at runs 1 to
//: 16 (r1 on the plain leaf, r > 1 on the folding one) against the RANKING regret, with the plain and
//: the folding leaf priced separately per tile and per item. One price for both leaves could not
//: rank them: on decode the plain leaf wins at every k while long prefills fold 2x faster.
#ifndef GQA_D256_C
#define GQA_D256_C 0.6769
#endif
#ifndef GQA_D256_F
#define GQA_D256_F 16.9699
#endif
#ifndef GQA_D256_E
#define GQA_D256_E 5.8701
#endif
#ifndef GQA_D256_P
#define GQA_D256_P 5.9508
#endif
//: Per-tile and per-item costs of the FOLDING leaf (a run longer than one segment); a run-1 plan,
//: which takes the plain leaf, is charged `GQA_D256_C` and `GQA_D256_F`.
#ifndef GQA_D256_CF
#define GQA_D256_CF 1.6760
#endif
//: A folded D256 plan must beat run 1 by this fraction of its predicted cost to be taken. The model's
//: level rms is 13 %, and on decode the plain leaf won every measured row, while the folds that win
//: (long prefills) win by 20 to 50 %: at margin 0 the model took a fold that its own prediction
//: barely preferred and that measured slower. The margin changes no choice on the fitted rows.
#ifndef GQA_D256_FOLD_MARGIN
#define GQA_D256_FOLD_MARGIN 0.05
#endif
#ifndef GQA_D256_FF
#define GQA_D256_FF 11.9864
#endif

namespace gqa_sm100 {

#ifdef GQA_PLAN_BINDING
//: Test-only binding, compiled out by default: run the SINGLE-block plan instead of the
//: multi-block one, and keep the last call's plan tensors so the two can be compared bit for bit.
static bool g_plan_single = false;
static std::vector<at::Tensor> g_last_plan;
void set_plan_single(bool single) { g_plan_single = single; }
std::vector<at::Tensor> last_plan() { return g_last_plan; }
#endif

#define GQA_CHECK(cond, msg) TORCH_CHECK((cond), "gqa_sm100: " msg)

namespace {

struct MapStore {
    alignas(64) unsigned char bytes[128];
    CUtensorMap* map() { return reinterpret_cast<CUtensorMap*>(bytes); }
};

int host_ceil_div(int a, int b) { return (a + b - 1) / b; }

//: THE PLAN COST MODEL, and the only thing that arbitrates (wave count, fold run).
//:
//: It replaces two separate heuristics -- FA4's widened `q_stage` rule and the fold's
//: fill-the-machine loop -- with one quantity, because the two were arbitrating the SAME trade from
//: opposite ends and each missed rows the other could see. Every
//: quantity it reads is EXECUTION: the SM count, the shape, SEG and the deployment's own maxima. It
//: cannot move a bit, and the relations force every (waves, run) over one input and require
//: identical output.
//:
//: The form is a makespan: the CTA-microseconds the plan costs, spread over the SMs, but never less
//: than the longest single item, plus the launch floor and the combine's own floor when anything
//: publishes.
//:
//:     work    = tiles * C_w  +  items * F_w  +  spills * E_w  +  published * P
//:     cost    = FLOOR + [COMB if published] + max(work / SMs, longest item)
//:
//: `C_1` and `C_2` are MEASURED, not fitted: they are the per-KV-tile slopes of a tile-count sweep at
//: one and two waves (0.908 and 1.535 us per tile per CTA). The other five were fitted once by least
//: squares on log ratio over 14 rows x 10 configurations measured in one process, and the fit is
//: stable to three digits across restarts.
//:
//: `E_w` is the spill's EXPOSED cost and it is indexed by the WAVE COUNT rather than by the address
//: space the pending node lives in -- 7.52 us at one wave against 3.21 at two. That is the model's
//: one substantive claim: at two waves one wave's spill and its merges run while the OTHER wave's
//: MMA and softmax proceed, and at one wave there is no second stream to hide them behind. A model
//: with a single spill constant fits almost as well in rms and gets `q8192/k8192` wrong by 14 %,
//: which is how the term was identified.
struct PlanCost {
    static constexpr double C[2] = {0.908, 1.535};   // us per KV tile per CTA, by wave count
    static constexpr double F[2] = {7.73, 9.90};     // us per item, prologue and epilogue
    static constexpr double E[2] = {7.52, 3.21};     // us per spill, exposed
    static constexpr double P = 4.56;                // us per published (O, L, M) node
    static constexpr double FLOOR = 5.09;            // launch floor of the plan and the leaf
    static constexpr double COMB = 5.81;             // the combine's own floor, when it runs
};

//: The D256 constants. A D256 item is a CTA PAIR (two SMs), its per-tile cost is the
//: pair's, a spill moves a 128 KiB node per CTA (both CTAs in parallel), and there is no wave count.
struct PlanCost256 {
    static constexpr double C = GQA_D256_C;          // us per KV tile per pair
    static constexpr double F = GQA_D256_F;          // us per item
    static constexpr double E = GQA_D256_E;          // us per exposed spill
    static constexpr double P = GQA_D256_P;          // us per published node
    static constexpr double FF = GQA_D256_FF;        // us per item of the FOLDING leaf (run > 1)
    static constexpr double CF = GQA_D256_CF;        // us per KV tile per pair, FOLDING leaf
};

//: Spills the streaming binary counter performs over a run of `n` segments with `tmem` TMEM output
//: tiles. The same allocator `leaf_d128.cu::build_segment_plan` runs, counted rather than recorded.
int fold_spill_count(int n, int tmem) {
    bool used[2] = {}, held[FOLD_BACKING] = {};
    int slot[FOLD_STACK + 1], size[FOLD_STACK + 1], depth = 0, spills = 0;
    for (int j = 0; j < n; ++j) {
        int a = 0;
        while (a < tmem && used[a]) ++a;
        if (a >= tmem) {
            int d = 0;
            while (d < depth && slot[d] < 0) ++d;
            int b = 0;
            while (b < FOLD_BACKING && held[b]) ++b;
            if (d >= depth || b >= FOLD_BACKING) return -1;     // the run does not fit
            a = slot[d];
            slot[d] = -1 - b;
            held[b] = true;
            used[a] = false;
            ++spills;
        }
        used[a] = true;
        if (depth >= FOLD_STACK) return -1;
        slot[depth] = a;
        size[depth] = 1;
        ++depth;
        while (depth > 1 && (size[depth - 2] == size[depth - 1] || j == n - 1)) {
            const int left = slot[depth - 2], right = slot[depth - 1];
            if (left < 0) { held[-1 - left] = false; slot[depth - 2] = right; }
            else { used[right] = false; }
            size[depth - 2] += size[depth - 1];
            --depth;
        }
    }
    return spills;
}

struct PlanEstimate {
    double items = 0, tiles = 0, spills = 0, published = 0, longest_tiles = 0;
    //: Query tiles that publish at all, which is what bounds the combine's LIVE work. It sizes the
    //: combine's grid and nothing else: the combine grid-strides, so the grid is a performance choice
    //: and an inaccurate estimate cannot make it wrong.
    double pub_tiles = 0;
    bool feasible = true;
};

//: What the plan would emit at this (wave count, run), from the HOST-VISIBLE shape alone: `B`, `T`,
//: the deployment's `max_seqlen_q` and `max_seqlen_k`, SEG and the head geometry. Nothing here reads
//: a device tensor, so the launch stays CUDA-graph capturable and the estimate is the same on every
//: run of the same shape.
//:
//: It walks the query tiles of ONE synthetic request of the deployment's maximum length -- which is
//: what carries the causal structure, and therefore the per-item tile counts and the straddling
//: tiles that cannot fold -- and then scales the totals so the item count matches the `n_tiles` the
//: grid is actually sized by. A ragged batch is approximated by that same shape, exactly as the two
//: heuristics this replaces already approximated it.
PlanEstimate estimate_plan(int w, int run, int B, int T, int max_q, int max_k, int h_kv,
                           int ratio, int seg, int n_tiles, bool pair = false) {
    PlanEstimate e;
    //: A D256 item is a 256-row tile with ONE TMEM output tile.
    const int tt = pair ? 256 / ratio : (128 * w) / ratio;
    const int tmem = pair ? 1 : ((w == 1) ? 2 : 1);
    const int synthetic = std::max(1, host_ceil_div(max_q, tt));
    const int walk = std::min(synthetic, 2048);
    const double scale = double(n_tiles) / double(walk);
    for (int i = 0; i < walk; ++i) {
        //: SUBSAMPLED, not truncated. Under causal masking tile 0 is the shortest and the last is the
        //: longest, so walking the first `walk` of them and scaling up would undercount both the tile
        //: total and `longest_tiles` -- badly, at ratio 16 with a long context, which is where the cap
        //: binds. Striding over the whole range keeps the distribution and keeps the last tile in.
        const int t = (walk == synthetic) ? i : int(int64_t(i) * synthetic / walk);
        const int tok0 = t * tt;
        const int last = std::min(tok0 + tt - 1, max_q - 1);
        const int border = max_k - max_q + last + 1;
        if (border <= 0) continue;
        const int n = segment_count(border, seg);
        const int n_first = segment_count(std::max(1, max_k - max_q + tok0 + 1), seg);
        const int effective = (n_first == n) ? run : 1;
        const bool whole = n <= effective;
        for (int k = 0; k < n;) {
            const int z = node_size_at(k, n, effective);
            int tiles = 0;
            for (int j = 0; j < z; ++j) {
                const int begin = (k + j) * seg;
                tiles += std::max(1, host_ceil_div(std::min(begin + seg, border) - begin, KEY_TILE));
            }
            const int sp = fold_spill_count(z, tmem);
            if (sp < 0) { e.feasible = false; return e; }
            e.items += 1;
            e.tiles += tiles;
            e.spills += double(sp) * (pair ? 1 : w);   // each wave (or CTA pair) folds its own stack
            if (!whole) e.published += w;
            if (!whole && k == 0) e.pub_tiles += 1;
            e.longest_tiles = std::max(e.longest_tiles, double(tiles));
            k += z;
        }
    }
    e.items *= scale * h_kv;
    e.tiles *= scale * h_kv;
    e.spills *= scale * h_kv;
    e.published *= scale * h_kv;
    e.pub_tiles *= scale * h_kv;
    (void)B; (void)T;
    return e;
}

//: `multiseg` is `max_n > 1`, which is when `forward` LAUNCHES the combine -- not when a candidate
//: publishes. Charging `COMB` on `published` instead would be a 5.8 us credit that whole-fold
//: candidates do not earn, because the combine runs either way.
//: The makespan of `items` items of total `work` on `slots` CTA slots, never below the longest item
//: -- and WAVE-QUANTISED: a plan whose item count sits just above a multiple of the slots
//: pays a whole extra wave, which `max(work / slots, longest)` does not see (it mis-ranked the
//: D256 decode runs by up to 13 %). Execution only.
double makespan(double work, double items, int slots, double longest) {
    double span = std::max(work / double(slots), longest);
    if (items > 0) {
        const double waves = std::ceil(items / double(slots) - 1e-9);
        span = std::max(span, waves * work / items);
    }
    return span;
}

double plan_cost(const PlanEstimate& e, int w, int sms, bool multiseg) {
    if (!e.feasible || e.items <= 0) return 1e30;
    const int i = w - 1;
    const double work = e.tiles * PlanCost::C[i] + e.items * PlanCost::F[i]
                      + e.spills * PlanCost::E[i] + e.published * PlanCost::P;
    const double longest = e.longest_tiles * PlanCost::C[i] + PlanCost::F[i];
    //: D128 keeps the UNQUANTISED makespan: refitted (25 rows x 10 configs), the quantised form
    //: mis-ranks B1 q2048/k2048 (w2r2 43.3 us against w1r4 30.9) while the plain form chooses
    //: exactly what the shipped constants choose.
    return PlanCost::FLOOR + (multiseg ? PlanCost::COMB : 0.0) + std::max(work / double(sms), longest);
}

//: The estimate depends only on host scalars, so it is computed once per distinct shape. A decode
//: call is 10 us end to end and the walk is thousands of integer steps; without this the heuristic
//: would cost more than the kernel it is choosing for.
struct PlanKey {
    int b, t, max_q, max_k, h_kv, ratio, seg, sms, waves_override, run_override, pair;
    bool operator==(const PlanKey& o) const {
        return b == o.b && t == o.t && max_q == o.max_q && max_k == o.max_k && h_kv == o.h_kv
            && ratio == o.ratio && seg == o.seg && sms == o.sms
            && waves_override == o.waves_override && run_override == o.run_override
            && pair == o.pair;
    }
};
struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        size_t h = 1469598103934665603ull;
        for (int v : {k.b, k.t, k.max_q, k.max_k, k.h_kv, k.ratio, k.seg, k.sms,
                      k.waves_override, k.run_override, k.pair})
            h = (h ^ size_t(uint32_t(v))) * 1099511628211ull;
        return h;
    }
};

struct PlanChoice {
    int waves = 1;
    int run = 1;
    //: Query tiles the chosen plan is expected to publish from, which sizes the combine's grid.
    int pub_tiles = 0;
};

//: Choose `(waves, max_run)`. Pure execution; the relations force every alternative and require
//: identical bits.
PlanChoice choose_plan(const PlanKey& key) {
    static std::mutex lock;
    static std::unordered_map<PlanKey, PlanChoice, PlanKeyHash> memo;
    {
        std::lock_guard<std::mutex> guard(lock);
        const auto found = memo.find(key);
        if (found != memo.end()) return found->second;
    }
    PlanChoice best{1, 1, 0};
    if (key.pair) {
        //: D256: a CTA pair per item, ONE TMEM output tile, the fold's pending nodes
        //: in global backing slots. One candidate axis, the run; the pair occupies two SMs.
        double best_cost = 1e300;
        const int max_n = std::max(1, host_ceil_div(key.max_k, key.seg));
        int forced_run = 0;
        if (key.run_override > 0) {
            forced_run = 1;
            while (forced_run * 2 <= std::min(key.run_override, FOLD_MAX_RUN)) forced_run *= 2;
        }
        const int tt = 256 / key.ratio;
        const int ta = host_ceil_div(key.t, tt) + key.b;
        const int tb = host_ceil_div(key.max_q, tt) * std::max(key.b, 1);
        const int n_tiles = std::max(1, std::min(ta, tb));
        for (int run = 1; run <= FOLD_MAX_RUN; run *= 2) {
            if (forced_run > 0) {
                if (run != forced_run) continue;
            } else if (run > 1 && run >= 2 * max_n) {
                continue;
            }
            const PlanEstimate e = estimate_plan(1, run, key.b, key.t, key.max_q, key.max_k,
                                                 key.h_kv, key.ratio, key.seg, n_tiles, true);
            if (!e.feasible || e.items <= 0) continue;
            const double f_item = run > 1 ? PlanCost256::FF : PlanCost256::F;
            const double c_tile = run > 1 ? PlanCost256::CF : PlanCost256::C;
            const double work = e.tiles * c_tile + e.items * f_item
                              + e.spills * PlanCost256::E + e.published * PlanCost256::P;
            const double longest = e.longest_tiles * c_tile + f_item;
            const double cost = (PlanCost::FLOOR + (max_n > 1 ? PlanCost::COMB : 0.0)
                                 + makespan(work, e.items, std::max(1, key.sms / 2), longest))
                              * (run > 1 ? 1.0 + GQA_D256_FOLD_MARGIN : 1.0);
            if (cost < best_cost) {
                best_cost = cost;
                best = {1, run, int(e.pub_tiles + 0.5)};
            }
        }
        if (forced_run > 0) best.run = forced_run;
    } else {
        double best_cost = 1e300;
        const int max_n = std::max(1, host_ceil_div(key.max_k, key.seg));
        //: An override names ONE candidate and the model is not consulted for it. The run is rounded
        //: DOWN to a power of two, because `node_size_at` names canonical tree nodes only for a
        //: power-of-two run at an aligned start; a caller that asks for 3 gets 2 rather than a plan
        //: that cuts the tree somewhere that is not a node.
        int forced_run = 0;
        if (key.run_override > 0) {
            forced_run = 1;
            while (forced_run * 2 <= std::min(key.run_override, FOLD_MAX_RUN)) forced_run *= 2;
        }
        for (int w = 1; w <= 2; ++w) {
            if (key.waves_override > 0 && w != key.waves_override) continue;
            const int tt = (128 * w) / key.ratio;
            const int ta = host_ceil_div(key.t, tt) + key.b;
            const int tb = host_ceil_div(key.max_q, tt) * std::max(key.b, 1);
            const int n_tiles = std::max(1, std::min(ta, tb));
            for (int run = 1; run <= FOLD_MAX_RUN; run *= 2) {
                if (forced_run > 0) {
                    if (run != forced_run) continue;
                } else if (run > 1 && run >= 2 * max_n) {
                    //: A run past the deepest tree the deployment can have is the SAME plan as that
                    //: tree, so it is not a separate candidate -- and evaluating it would let two
                    //: spellings of one plan tie on cost and make the choice order-dependent.
                    continue;
                }
                const PlanEstimate e = estimate_plan(w, run, key.b, key.t, key.max_q, key.max_k,
                                                     key.h_kv, key.ratio, key.seg, n_tiles);
                const double cost = plan_cost(e, w, key.sms, max_n > 1);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = {w, run, int(e.pub_tiles + 0.5)};
                }
            }
        }
        if (key.waves_override > 0) best.waves = key.waves_override;
        if (forced_run > 0) best.run = forced_run;
    }
    std::lock_guard<std::mutex> guard(lock);
    memo[key] = best;
    return best;
}

}  // namespace

//: decode_d256.cu: the decode leaf and its combine, for a call `forward` has checked and routed.
void decode_d256_forward(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                         const at::Tensor& cu_seqlens_q, const at::Tensor& seqused_k,
                         const c10::optional<at::Tensor>& block_table,
                         const c10::optional<at::Tensor>& cu_seqlens_k, double softmax_scale, int seg,
                         int max_seqlen_q, int max_seqlen_k, int h_kv, int page, int kv_tokens,
                         const at::Tensor& out, const at::Tensor& lse, int split_override);

std::tuple<at::Tensor, at::Tensor> forward(
        const at::Tensor& q,              // bf16 [T, h_q, D]
        const at::Tensor& k_cache,        // bf16 [blocks, page, h_kv, D] or [T_kv, h_kv, D]
        const at::Tensor& v_cache,
        const at::Tensor& cu_seqlens_q,   // int32 [B+1]
        const at::Tensor& seqused_k,      // int32 [B]
        const c10::optional<at::Tensor>& block_table,   // int32 [B, max_blocks]
        const c10::optional<at::Tensor>& cu_seqlens_k,  // int32 [B+1]
        double softmax_scale,
        int64_t seg,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        int64_t granularity,
        int64_t waves_override,
        int64_t max_run_override,
        int64_t leaf,
        int64_t decode_split,
        const c10::optional<at::Tensor>& out_opt) {  // bf16 [T, h_q, D], contiguous: written in place
    GQA_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16, "q must be a bf16 CUDA tensor");
    GQA_CHECK(q.dim() == 3, "q must be [T, h_q, D]");
    GQA_CHECK(q.is_contiguous(), "q must be contiguous");
    GQA_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
              "the KV cache must be bf16");
    //: K and V may be strided views (an engine's interleaved K|V pages): the feature axis must be
    //: dense, and the token and head strides 16-byte multiples, which is what the TMA map encodes.
    GQA_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1, "the KV cache feature axis must be dense");
    GQA_CHECK(cu_seqlens_q.scalar_type() == at::kInt && seqused_k.scalar_type() == at::kInt,
              "cu_seqlens_q and seqused_k must be int32");

    const int T = int(q.size(0));
    const int h_q = int(q.size(1));
    const int dim = int(q.size(2));
    GQA_CHECK(dim == 128 || dim == 256,
              "only head dims 128 and 256 are compiled in this package");
    //: Head dim 256 runs a CTA PAIR per work item: each CTA owns 128 packed query rows and all 256
    //: output features, so the query tile is 256 rows and TMEM is 100 % full. There is no wave
    //: count, and a run longer than one segment takes the folding leaf (`leaf_d256_fold.cu`).
    const bool pair = (dim == 256);
    const int B = int(seqused_k.numel());
    GQA_CHECK(int(cu_seqlens_q.numel()) == B + 1, "cu_seqlens_q must hold B+1 entries");

    int h_kv = 0, page = 0, kv_tokens = 0;
    if (block_table.has_value()) {
        GQA_CHECK(k_cache.dim() == 4, "a paged cache must be [blocks, page, h_kv, D]");
        page = int(k_cache.size(1));
        h_kv = int(k_cache.size(2));
        GQA_CHECK(int(k_cache.size(3)) == dim, "cache head dim must match q");
        GQA_CHECK(page % KEY_TILE == 0,
                  "the page size must be a multiple of the KV tile width (128) so a tile never "
                  "crosses a page");
        kv_tokens = int(k_cache.size(0)) * page;
        //: Blocks flatten into one token axis (key slot = block * page + offset), so a block must
        //: start exactly `page` token strides after the previous one, in K and in V alike.
        GQA_CHECK(k_cache.stride(0) == int64_t(page) * k_cache.stride(1)
                      && v_cache.stride(0) == int64_t(page) * v_cache.stride(1),
                  "a paged cache's blocks must be page token-strides apart (the [blocks, page, h_kv, D] "
                  "view of a token-major layout)");
        GQA_CHECK(block_table->scalar_type() == at::kInt, "block_table must be int32");
        GQA_CHECK(int(block_table->size(0)) == B, "block_table must have B rows");
    } else {
        GQA_CHECK(k_cache.dim() == 3, "a packed cache must be [T_kv, h_kv, D]");
        GQA_CHECK(cu_seqlens_k.has_value(), "a packed cache needs cu_seqlens_k");
        h_kv = int(k_cache.size(1));
        GQA_CHECK(int(k_cache.size(2)) == dim, "cache head dim must match q");
        kv_tokens = int(k_cache.size(0));
    }
    GQA_CHECK(h_q % h_kv == 0, "h_q must be a multiple of h_kv");
    const int ratio = h_q / h_kv;
    GQA_CHECK(ratio <= 16, "h_q / h_kv must be at most 16");
    GQA_CHECK(seg > 0 && seg % KEY_TILE == 0, "seg must be a positive multiple of 128");
    if (out_opt.has_value())
        GQA_CHECK(out_opt->scalar_type() == at::kBFloat16 && out_opt->is_contiguous()
                      && out_opt->device() == q.device() && out_opt->numel() == q.numel(),
                  "out must be a contiguous bf16 tensor on q's device with q's element count");
    const auto opts_f = at::TensorOptions().dtype(at::kFloat).device(q.device());
    auto out = out_opt.has_value() ? out_opt->view({T, h_q, dim}) : at::empty({T, h_q, dim}, q.options());
    auto lse = at::empty({T, h_q}, opts_f);

    //: THE LEAF. A head-dim-256 call whose packed query rows per (request, KV head) fit the decode
    //: leaf's 16-row tile -- `max_seqlen_q * ratio <= DECODE_ROWS`: decode, and speculative decode up
    //: to `16 / ratio` tokens per request -- runs the decode leaf (`decode_d256.cu`), which executes
    //: the same leaf program swap-AB on one CTA per (request, KV head) and needs no plan. Every other
    //: call runs the plan below. The rule reads the call's shape only (never B or the lengths), and
    //: it is execution: the relations force both leaves and require identical bits. A general-plan
    //: override (granularity, run) names a plan of the general leaf, so it takes the general leaf.
    GQA_CHECK(leaf == LEAF_AUTO || leaf == LEAF_GENERAL || leaf == LEAF_DECODE, "unknown leaf choice");
    const bool decode_shaped = pair && max_seqlen_q >= 1 && max_seqlen_q * ratio <= DECODE_ROWS;
    GQA_CHECK(leaf != LEAF_DECODE || decode_shaped,
              "the decode leaf serves head dim 256 with max_seqlen_q * (h_q / h_kv) <= 16 only");
    const bool take_decode = leaf == LEAF_DECODE
            || (leaf == LEAF_AUTO && decode_shaped && granularity == GRAN_AUTO && max_run_override <= 0);
    //: The general leaf packs 128-row tiles of whole tokens, so it needs a ratio that divides 128; the
    //: decode leaf also serves 3 and 6 (one launch reads each KV head once for all its query heads).
    GQA_CHECK(128 % ratio == 0 || (take_decode && (ratio == 3 || ratio == 6)),
              "h_q / h_kv must divide 128, or be 3 or 6 on a decode-shaped call (the decode leaf)");
    if (take_decode) {
        GQA_CHECK(waves_override <= 0,
                  "head dim 256 runs one CTA pair per item and has no wave count to override");
        const c10::cuda::CUDAGuard guard(q.device());
        decode_d256_forward(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, cu_seqlens_k,
                            softmax_scale, int(seg), int(max_seqlen_q), int(max_seqlen_k), h_kv, page,
                            kv_tokens, out, lse, int(decode_split));
        return std::make_tuple(out, lse);
    }

    // Execution: the wave-count trade. Two waves give the MMA the OTHER wave's PV and QK to issue
    // while a softmax warpgroup works, which is worth real time; at WAVES == 1 TMEM has room for a
    // worker to fold an aligned run of segments in place instead of publishing every one of them,
    // and publishing fp32 partials costs more. At two waves the fold's pending nodes live in global
    // backing slots. `choose_plan` weighs these.
    const int max_n_probe = std::max(1, host_ceil_div(int(max_seqlen_k), int(seg)));
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    //: ONE RULE for (wave count, fold run), and it is `choose_plan`'s cost model above.
    //:
    //: What it replaces: FA4's `q_stage` rule widened twice, plus a separate loop that halved the
    //: run until the item count reached the SM count. Those two arbitrated the same trade from
    //: opposite ends -- one asked "can two waves fill the machine", the other "can this run still
    //: fill it" -- and each left rows the other could see: the wave rule's item-count clause missed
    //: the production prefill row, and the fill loop's hard `< sms` forbade `q512/k2048`'s and
    //: `q128/k8192`'s best plans. The cost model's regret against each row's own best plan is
    //: measurably lower than the two rules'.
    //:
    //: It is also what makes the two-wave FOLD reachable at all: the pending node lives in an
    //: L2-resident global slot, and the best plan for the longest rows is BOTH two waves and a
    //: fold, which no clause over item counts expresses.
    const PlanKey key{B, T, int(max_seqlen_q), int(max_seqlen_k), h_kv, ratio, int(seg), sms,
                      int(waves_override), int(max_run_override), pair ? 1 : 0};
    if (pair)
        GQA_CHECK(waves_override <= 0,
                  "head dim 256 runs one CTA pair per item and has no wave count to override");
    const PlanChoice chosen = choose_plan(key);
    const int waves = chosen.waves;
    GQA_CHECK(waves == 1 || waves == 2, "waves must be 1 or 2");
    const int tile_rows = pair ? 256 : 128 * waves;
    const int tile_tokens = tile_rows / ratio;
    const int max_n = max_n_probe;
    // The reach of the fold. Two score slots and two fp32 output tiles fill TMEM's 512 columns at
    // D128: at one wave BOTH output tiles are one worker's merge stack, at two waves each is a wave's
    // accumulator and the stack has one. Every pending node beyond those lives in a backing slot --
    // the 64 KiB shared node slot at one wave, and otherwise an L2-resident global slot per (resident
    // CTA, wave) -- so the run is not capped by TMEM at either wave count. At D256 the fp32
    // accumulator alone is 256 columns and a 128-row x 256-feature fp32 node is 128 KiB, so the
    // folding leaf keeps one node in TMEM and every further pending node in a global backing slot.
    const int tmem_tiles = pair ? 1 : (waves == 1 ? 2 : 1);
    const int shared_backing = (!pair && waves == 1) ? 1 : 0;
    const int max_run = chosen.run;
    GQA_CHECK(max_run >= 1 && (max_run & (max_run - 1)) == 0,
              "the aligned run must be a power of two: node_size_at names canonical tree nodes only "
              "for a power-of-two run at an aligned start");
    // The backing slots this (wave count, run) needs, and how many of them have to be global. The
    // count is the counter's stack depth less the TMEM tiles it has: `floor(log2(run)) + 1 - tmem`.
    const int fold_backing = fold_backing_slots(tmem_tiles, max_run);
    const int fold_globals = std::max(0, fold_backing - shared_backing);
    GQA_CHECK(fold_backing <= FOLD_BACKING, "the fold's backing slots exceed FOLD_BACKING");
    // A tile whose rows AGREE on their segment count publishes ceil(n / max_run) nodes; one whose rows
    // straddle a segment boundary must run as single segments and publishes up to n. The capacity has
    // to cover both, or the plan drops that tile and its output is never written, which is a silently
    // wrong answer. A tile straddles a boundary only where one falls inside its token range, and
    // boundaries are SEG apart, so per request there are at most ceil(max_seqlen_q / seg) + 1 of them.
    // That bound is what sizes the grid as well, and an over-large grid is not free: every CTA past the
    // live count still launches, reads the count and exits.
    const int tiles_a = host_ceil_div(T, tile_tokens) + B;
    const int tiles_b = host_ceil_div(int(max_seqlen_q), tile_tokens) * std::max(B, 1);
    const int n_tiles = std::max(1, std::min(tiles_a, tiles_b));
    //: The run the PLAN will actually emit, which is NOT always the run the cost model chose: a
    //: granularity override forces one segment per item or aligned pairs, and the capacity has to
    //: cover THAT. Sizing it from `max_run` alone would be wrong -- `GRAN_SINGLE` over a tile with
    //: 16 uniform segments needs 16 items where `max_run = 16` reserves one, the plan would overflow
    //: its item and slot capacity, and the combine would read an uninitialised `slot_base` out of
    //: `at::empty` memory. `rel_execution_plans` covers this case (q192/k8192 at `GRAN_SINGLE`),
    //: and `plan.cu` additionally writes a REFUSING entry on overflow so a wrong bound is a missing
    //: output rather than a wild read.
    const int plan_run = int(granularity) == GRAN_SINGLE ? 1
                       : int(granularity) == GRAN_PAIRS ? std::min(2, max_run)
                       : max_run;
    //: Nodes a UNIFORM tile publishes, and it is NOT `ceil(n / run)`. `node_size_at` emits full runs
    //: while they fit and then decomposes the remainder into DESCENDING POWERS OF TWO, so a tile with
    //: `n` segments publishes `floor(n / run) + popcount(n mod run)` nodes -- three at `n = 13,
    //: run = 8`, where `ceil(13 / 8)` is two. A `ceil` bound is therefore too small for every
    //: run of four or more whose remainder has two or more bits set, which is most of them.
    //:
    //: `popcount(n mod run) <= log2(run)`, so `ceil(max_n / run) + log2(run)` covers every `n` the
    //: deployment can have. It costs at most `log2(run) * n_tiles * h_kv` items of workspace.
    //:
    //: Example: `q2048/k8192` at `waves=1, max_run=8` needs 384 items per KV head, where a `ceil`
    //: bound gives 336. `rel_execution_plans` forces exactly that plan, and runs with
    //: `ISOEXEC_GQA_SM100_CHECK` set so `OVERFLOW_FLAG` is read.
    int plan_run_log = 0;
    while ((1 << plan_run_log) < plan_run) ++plan_run_log;
    const int64_t uniform_nodes = int64_t(n_tiles) * h_kv
                                * (std::max(1, host_ceil_div(max_n, plan_run)) + plan_run_log);
    //: STRADDLING TILES, and the bound is zero for a decode. A tile straddles a segment boundary only
    //: when its rows DISAGREE on their segment count, which needs the tile to hold more than one token;
    //: at `max_seqlen_q == 1` every tile is one token of one request and every tile is uniform. It is
    //: also bounded by the tiles a request HAS, which for a short request is far below
    //: `ceil(max_seqlen_q / seg) + 1`.
    //:
    //: This term sizes the GRID as well as the workspace, and an over-large grid is not free: at
    //: `B = 256, k = 16384, SEG = 2048` it reserved 4096 items where 256 are live and 1792 CTAs launched
    //: only to read `counts` and exit -- 2 to 5 % of the leaf on decode rows, on top of the
    //: workspace. It is the same consideration as the combine's capacity-sized grid.
    const int tiles_per_request = std::max(1, host_ceil_div(int(max_seqlen_q), tile_tokens));
    const int64_t straddle_tiles = max_seqlen_q <= 1 ? 0
        : std::min<int64_t>(int64_t(host_ceil_div(int(max_seqlen_q), int(seg)) + 1),
                            int64_t(tiles_per_request));
    const int64_t straddling = straddle_tiles * std::max(B, 1) * h_kv * max_n;
    const int64_t item_cap = std::min(uniform_nodes + straddling,
                                      int64_t(n_tiles) * h_kv * max_n);
    const int64_t combine_cap = int64_t(n_tiles) * h_kv;
    const int64_t slot_cap = item_cap * tile_rows;
    GQA_CHECK(slot_cap * (dim + 2) * 4 < (int64_t(48) << 30),
              "the partial workspace this (shape, seg) needs exceeds 48 GiB; raise seg");

    const auto opts_i = at::TensorOptions().dtype(at::kInt).device(q.device());
    auto counts = at::empty({NUM_COUNTS}, opts_i);   // the plan kernel zeroes it
    auto items = at::empty({item_cap * (int64_t(sizeof(Item)) / 4)}, opts_i);
    auto ctiles = at::empty({combine_cap * (int64_t(sizeof(CombineTile)) / 4)}, opts_i);
    auto partial_o = at::empty({slot_cap, dim}, opts_f);
    auto partial_l = at::empty({slot_cap}, opts_f);
    auto partial_m = at::empty({slot_cap}, opts_f);
    //: The fold's GLOBAL backing slots, one set per (SM, wave). Indexed by SM ID and not by
    //: `blockIdx.x`, because the leaf's shared memory admits one block per SM: the workspace is then
    //: the machine's size and not the grid's -- at `q8192/k8192` 28 MB at one wave against 290 MB --
    //: which is what lets it stay in L2 instead of streaming through HBM. `launch_leaf_d128` checks
    //: that occupancy at launch and refuses if it is ever not one.
    const int64_t fold_slots_total = int64_t(fold_globals) * sms * (pair ? 1 : waves);
    auto fold_node = fold_globals > 0
        ? at::empty({fold_slots_total, int64_t(128) * dim}, opts_f)
        : at::empty({0}, opts_f);

    auto stream = at::cuda::getCurrentCUDAStream();

    PlanParams pp{};
    pp.b = B;
    pp.h_q = h_q;
    pp.h_kv = h_kv;
    pp.head_ratio = ratio;
    pp.tile_tokens = tile_tokens;
    pp.tile_rows = tile_rows;
    pp.seg = int(seg);
    pp.granularity = int(granularity);
    pp.max_run = max_run;
    auto log2_or_minus1 = [](int v) {
        if (v <= 0 || (v & (v - 1)) != 0) return -1;
        int s = 0;
        while ((1 << s) < v) ++s;
        return s;
    };
    pp.seg_shift = log2_or_minus1(int(seg));
    pp.tt_shift = log2_or_minus1(tile_tokens);
    pp.hkv_shift = log2_or_minus1(h_kv);
    pp.max_q = int(max_seqlen_q);
    pp.item_cap = int(item_cap);
    pp.combine_cap = int(combine_cap);
    pp.slot_cap = int(slot_cap);
    pp.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    pp.seqused_k = seqused_k.data_ptr<int>();
    pp.items = reinterpret_cast<Item*>(items.data_ptr<int>());
    pp.combine_tiles = reinterpret_cast<CombineTile*>(ctiles.data_ptr<int>());
    pp.counts = counts.data_ptr<int>();
    pp.page = page;
    pp.block_table_stride = block_table.has_value() ? block_table->stride(0) : 0;
    pp.block_table = block_table.has_value() ? block_table->data_ptr<int>() : nullptr;
    pp.cu_seqlens_k = cu_seqlens_k.has_value() ? cu_seqlens_k->data_ptr<int>() : nullptr;
    //: Nonzero: the fast plan kernels (one block each); zero is the test binding's reference walk.
    const int plan_blocks = 1;
#ifdef GQA_PLAN_BINDING
    C10_CUDA_CHECK(launch_plan(pp, g_plan_single ? 0 : plan_blocks, stream));
    g_last_plan = {items, ctiles, counts};
#else
    C10_CUDA_CHECK(launch_plan(pp, plan_blocks, stream));
#endif

    MapStore mq, mk, mv;
    // The Q box is ONE wave: 128 packed rows = (128 / ratio) tokens x ratio heads. The CTA issues one
    // box per wave, so the box must not carry the whole tile's tokens.
    // At D256 the box is still ONE CTA's 128 packed rows, because each CTA of the pair issues its
    // own Q copy for its own 128 rows.
    const int wave_tokens = 128 / ratio;
    CUresult r = encode_map(mq.map(), q.data_ptr(), T, h_q, dim, wave_tokens, ratio);
    GQA_CHECK(r == CUDA_SUCCESS, "encoding the Q tensor map failed");
    // The K box is the number of keys ONE CTA loads. At D128 that is the whole 128-key tile; at D256
    // the pair splits the tile's keys, 64 to each CTA, because the `cta_group::2` QK takes its
    // N = 128 keys as 64 from each CTA's B operand. The V box stays the whole 128 keys: PV splits
    // the FEATURE axis instead, which the tensor map's feature-chunk coordinate selects.
    const int key_box = pair ? 64 : KEY_TILE;
    //: The token stride is the one the flattened token axis steps by (stride 1 of a paged view,
    //: stride 0 of a packed one); the head stride is stride -2 of either.
    const int tok_dim = block_table.has_value() ? 1 : 0;
    r = encode_map(mk.map(), k_cache.data_ptr(), kv_tokens, h_kv, dim, key_box, 1,
                   k_cache.stride(tok_dim), k_cache.stride(-2));
    GQA_CHECK(r == CUDA_SUCCESS, "encoding the K tensor map failed (token and head strides must be "
                                 "multiples of 8 elements)");
    r = encode_map(mv.map(), v_cache.data_ptr(), kv_tokens, h_kv, dim, KEY_TILE, 1,
                   v_cache.stride(tok_dim), v_cache.stride(-2));
    GQA_CHECK(r == CUDA_SUCCESS, "encoding the V tensor map failed");

    LeafParams lp{};
    lp.h_q = h_q;
    lp.h_kv = h_kv;
    lp.seg = int(seg);
    lp.page = page;
    lp.block_table_stride = block_table.has_value() ? block_table->stride(0) : 0;
    lp.scale_log2 = float(softmax_scale) * 1.4426950408889634f;
    lp.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    lp.seqused_k = seqused_k.data_ptr<int>();
    lp.block_table = block_table.has_value() ? block_table->data_ptr<int>() : nullptr;
    lp.cu_seqlens_k = cu_seqlens_k.has_value() ? cu_seqlens_k->data_ptr<int>() : nullptr;
    lp.items = reinterpret_cast<const Item*>(items.data_ptr<int>());
    lp.counts = counts.data_ptr<int>();
    lp.out = out.data_ptr();
    lp.lse = lse.data_ptr<float>();
    lp.partial_o = partial_o.data_ptr<float>();
    lp.partial_l = partial_l.data_ptr<float>();
    lp.partial_m = partial_m.data_ptr<float>();
    lp.fold_node = fold_globals > 0 ? fold_node.data_ptr<float>() : nullptr;
    lp.fold_globals = fold_globals;
    lp.fold_sms = sms;
    if (pair && max_run > 1) {
        //: A run of segments per item: the folding leaf. The plan never emits a run
        //: longer than `max_run`, so a run-1 plan only ever hands the plain leaf single segments.
        C10_CUDA_CHECK(launch_leaf_d256_fold(lp, *mq.map(), *mk.map(), *mv.map(), ratio,
                                             int(item_cap), stream));
    } else if (pair) {
        C10_CUDA_CHECK(launch_leaf_d256(lp, *mq.map(), *mk.map(), *mv.map(), ratio,
                                        int(item_cap), stream));
    } else {
        C10_CUDA_CHECK(launch_leaf_d128(lp, *mq.map(), *mk.map(), *mv.map(), waves, ratio,
                                        int(item_cap), stream));
    }

    CombineParams cp{};
    cp.h_q = h_q;
    cp.dim = dim;
    cp.head_ratio = ratio;
    cp.tile_tokens = tile_tokens;
    cp.tile_rows = tile_rows;
    cp.seg = int(seg);
    cp.scale_log2 = lp.scale_log2;
    cp.combine_tiles = reinterpret_cast<const CombineTile*>(ctiles.data_ptr<int>());
    cp.counts = counts.data_ptr<int>();
    cp.partial_o = lp.partial_o;
    cp.partial_l = lp.partial_l;
    cp.partial_m = lp.partial_m;
    cp.out = out.data_ptr();
    cp.lse = lse.data_ptr<float>();
    // No row can have more than one segment, so no tile can publish: the combine has nothing to do
    // and its launch would be pure floor. On a decode row the launch floor is most of the call.
    if (max_n > 1) {
        //: The combine runs ONE WARP per (token, query head) that has more than one segment, so
        //: `T * h_q` bounds its live warps and `T * h_q / 4` its live blocks. Below one live block
        //: per SM there is no occupancy to spend, and the four-node load group -- four partial-O
        //: loads in flight instead of a chain of one round trip per node -- is free; above it the
        //: group's 4 x EPT extra registers cost blocks per SM, and it measured 5 to 31 % WORSE on
        //: every row that fills the machine. Execution: it may read the SM
        //: count, and the relations force both and require identical bits.
        const bool group_loads = int64_t(T) * h_q <= int64_t(sms) * 4;
        const int live_rows = std::min(tile_rows,
                                       std::min(int(max_seqlen_q), tile_tokens) * ratio);
        //: Query tiles the combine is expected to have work for. It sizes the grid and nothing else:
        //: the combine grid-strides over `(entry, row group)` units, so an inaccurate estimate costs
        //: performance and cannot cost correctness.
        //:
        //: Why the estimate and not just a residency: the two are right on OPPOSITE kinds of
        //: shape. Where the fold publishes nothing -- `q8192/k8192` at two waves and run 16, where no
        //: tile straddles a segment boundary -- a capacity-sized grid launched 32768 blocks to read
        //: `counts` and exit, 18.3 us for no work. Where the combine is the dominant cost and every
        //: unit is real -- multi-segment D256 rows, where TMEM holds only one output tile --
        //: capping the grid at one residency instead COST 8 to 23 % of the combine, because 2368
        //: blocks each walking fourteen units serially leave a tail the scheduler cannot fill, while
        //: 32768 independent blocks queue perfectly. Sizing by the expected work takes both.
        const int pub_tiles = chosen.pub_tiles;
        C10_CUDA_CHECK(launch_combine(cp, dim, max_n, int(combine_cap), live_rows, sms, pub_tiles,
                                      group_loads, stream));
    }

    // The plan cannot report a capacity overflow without a synchronisation, and a synchronisation per
    // call would both cost a benchmark and break CUDA-graph capture. The bound above is chosen so an
    // overflow is impossible; this reads the flag anyway when the batteries ask, so a future bound
    // that is wrong fails loudly instead of leaving a tile of the output unwritten.
    if (std::getenv("ISOEXEC_GQA_SM100_CHECK") != nullptr
            && c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None) {
        const auto host_counts = counts.to(at::kCPU);
        const int* c = host_counts.data_ptr<int>();
        GQA_CHECK(c[OVERFLOW_FLAG] == 0,
                  "the plan overflowed its item/slot/combine capacity: part of the output was never "
                  "written. The host-side bound in host.cu is wrong for this shape.");
        GQA_CHECK(c[MAX_N] <= max_n,
                  "a row needs more segments than max_seqlen_k/seg allows: max_seqlen_k is wrong");
    }
    return std::make_tuple(out, lse);
}

}  // namespace gqa_sm100
