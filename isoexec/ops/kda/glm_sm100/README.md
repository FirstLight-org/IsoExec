# GLM Kimi Delta Attention on SM100 (`kda.glm`, impl `sm100_kda_v1`)

## Schedule

The recurrence reducer has no merge, so rule 1 of `isoexec/ops/schedule_policy.py` applies: segments of the exact
block, anchored at absolute position 0 of the sequence, one continued chain. `isoexec/ops/kda/schedule.py` holds the
reducer (`CHUNK`: built blocks 16, 32, 64; the block is the largest, 64) and turns `schedule_policy.no_merge(64)` into
the compiler definition `-DKDA_CL=64`; `csrc/common.cuh` has no chunk length of its own and does not compile without
it. `C = 64` in the kernels and `spec.CHUNK` are the storage height of a chunk tile, not the schedule. The loaded
module reports the value it was compiled with (`chunk_len()`), and `_extension.py` refuses a mismatch.
`isoexec/ops/kda/tests/test_schedule_policy_cpu.py` and `test_schedule_invariance_gpu.py` are the gates.

The engine paths (`ops.prefill`, `ops.decode_step`, `engine.forward` with a pool) and the trainer forward
(`ops.forward` of whole sequences, `engine.forward(cu_seqlens=...)`) run this schedule. The dense-state functional
forms (`ops.forward(initial_state=...)` and `ops.decode`) count chunks from the start of the call, so they equal the
schedule only when every call starts at a multiple of 64 (`tests/test_relations_gpu.py`,
`test_unaligned_continuation_is_within_tolerance`); the engine and the trainer do not use them for continuation.

A second reducer exists (`kda/glm_rec_sm100`, per token, block 1). A deployment that wants it selects it explicitly
(`schedule.select(schedule.TOKEN)`); its manifest differs and its state pool is refused by this package's engine.

On this tree the package is the operator only: there is no GLM model, `isoexec.nn.GLMKimiDeltaAttention` or serving
adapter, and the sections below that name them describe the tree the package was written on.

Status (2026-10-08): **implemented** — chunked prefill (varlen, initial/final fp32 state), backward
(q, k, v, log-decay, beta, initial state, including gradients through the returned final state), the GLM
layer composition bound as `backend="sm100"` on the public `isoexec.nn` modules (reference default
unchanged), and **engine records anchored at absolute position 0**: rollout (chunked prefill split
anywhere + one-token decode steps, any batch composition) is bit for bit the trainer's single
whole-sequence forward, outputs and final state. Every RL row is below FLA, including the exact engine
decode (0.97 / 0.84 / 0.82x FLA's fused recurrent decode at B = 256 / 512 / 1024; table below).

```python
from isoexec.ops.kda.glm_sm100 import ops
o, ht = ops.forward(q, k, v, log_decay, beta, cu_seqlens, h0, output_final_state=True)   # packed [T, H, 128]
o, h1 = ops.decode(q1, k1, v1, g1, b1, h, out_state=h)                                  # one token / sequence
dq, dk, dv, dg, dbeta, dh0 = ops.backward(q, k, v, log_decay, beta, cu_seqlens, h0, do, dht)

st = ops.KdaState(slots, H)                                    # engine records (csrc/record.cuh)
o = ops.prefill(q, k, v, g, b, cu, st, slot_in, slot_out)     # continue records anywhere (negative = fresh)
o1 = ops.decode_step(q1, k1, v1, g1, b1, st, slots)            # one token per sequence, incremental
ht = ops.record_state(st, slots)                               # == forward's final state, bitwise

from isoexec.nn import GLMKimiDeltaRule, GLMKimiDeltaAttention
GLMKimiDeltaRule(backend="sm100")(q, k, v, log_decay, beta, h0, return_state=True)       # [B, T, H, 128], autograd
GLMKimiDeltaAttention(cfg, backend="sm100", dtype=torch.bfloat16)(x)                     # full layer, autograd
```

## Contract (unchanged)

Core inputs: Q/K/log-decay `[batch, time, heads, key_channels]`, V `[batch, time, heads, value_channels]`,
beta `[batch, time, heads]`, optional FP32 initial state `[batch, heads, key_channels, value_channels]`. Q/K
use FP32 L2 normalization with epsilon `1e-6`; Q additionally scales by `key_channels**-0.5`. Per token:

```
A = state * exp(log_decay)[..., :, None]
delta = beta[..., None] * (V - sum(A * K[..., :, None], key_axis))
state = A + K[..., :, None] * delta[..., None, :]
output = sum(state * Q[..., :, None], key_axis)
```

The state stays FP32; outputs cast to the query dtype; decay is per key channel. Oracles: the eager
equation `isoexec.nn.GLMKimiDeltaRule(backend="reference")` (`reference.recurrence` is the same equation in
float64, differentiable) and the pinned upstream chunk oracle
`tests/upstream_kda.py` (vendored from HuggingFace Transformers, Apache-2.0; see NOTICE). The temporary `backend="upstream"` FLA adapter is
untouched. Geometry: GLM-5.3-Flash `linear_num_heads = 64`, `linear_head_dim = 128` (published_config);
the kernels take any head count with K = V = 128, bf16 q/k/v, fp32 or bf16 log decay and beta.

## Numerical recipe (`csrc/recipe.cuh` is the contract; version 1)

Chunks of **C = 64 tokens anchored at absolute sequence position 0** (a functional call starts at
position 0; an engine record carries its position), 16-row sub-blocks, fp32 state. Built with the shared SM100 profile (`sm_100a`, `--fmad=false --ftz=false`). Contractions are
`mma.sync.m16n8k16` bf16 chains with ascending k16 steps (the gdn_cpr_sm100 primitives and their gated
facts; `csrc/common.cuh` restates them so this kernel's digest covers its own headers).

* Token prep: k^ = rn(k·rsqrt(FOLD32(k²)+1e-6)), q^ likewise × 128^-1/2 (gdn's FOLD32), G = chunk-local
  cumsum of g·log2e (fp32, ascending), E(x) = `ex2.approx.f32` (one MUFU; a fixed function of its input).
* Off-diagonal 16×16 blocks (i-block < j-block): KK/KQ by mma on decay-factored bf16 operands referenced to
  the j-block's first row n: `rn(k_j E(G_j−G_n))`, `rn(k_i E(G_n−G_i))` — every exponent ≤ 0, so nothing
  overflows for any log decay ≤ 0 (no FLA `safe_gate` domain restriction; tested at g = −60/token).
* Diagonal blocks: the decayed key is carried as an ascending product of per-token factors,
  `T_ii = k_i`, `T_ji = T_{j−1,i} · E(G_j − G_{j−1})`, and `DIAG(a_j,k_i) = T8` over channel eighths of fma
  chains `fma(a_jc, T_ji,c, acc)`. One multiply per (pair, channel) instead of one exponential; never
  overflows.
* X = (I+L)⁻¹ with gdn_cpr's block orders (T4 / T16 trees); `u = rn(X β v)`, `w = rn(X β kg)`,
  `d = rn(u − w S^)`, `o = rn(Mq d; seed qg S^)`, `S' = kd^T d; seed S·E(G_L)` with `S^ = rn(S)`.
* **Chain bounds are a function of the row block** (`CHAIN64^b(j)`: k-steps 0..b(j) only), so a one-row
  decode issues one k-step with no relation to argue.

Execution (free; never enters the bits): `intra` (CTA per chunk×head, 105 KB smem, 2/SM) publishes the
state-independent rows (w, u, qg, kd, Mq, E(G_L)) to a token-indexed workspace; `state` walks each
(sequence, head, 64-column value slice) through its chunks with the fp32 state in mma accumulators
(value columns never meet, so the slice width is free). `intra` / `state` are instantiated twice
(`REC`): calls without engine records run the instantiation with no record or carry paths (every chunk
starts at row 0); the arithmetic of every row is the same. Functional decode: `decode_prep` (warp per row)
+ `decode_state` (persistent grid, next item's 32 KB state slice and rows prefetched by cp.async).

## Engine records (`csrc/record.cuh`, `ops.KdaState`)

Per (slot, head), 189,456 bytes (`record_layout.py` mirrors `csrc/record.cuh`; the legacy record was 220,944): the
boundary state S_c as three **state planes** (SH = rn(S_c) bf16 `[v][k]`, the operand image every step reads; SL =
the low 16 bits of S_c's fp32 pattern; SC = one carry bit per element, mma-fragment order, so that
`S_c = ((SH - SC) << 16) | SL` bit for bit), and the open chunk's rows [0, pos mod 64): k^, v
(transposed), d (transposed), kg (transposed), G, X rows, the open block's P rows and decay-factored
earlier keys ki^b, beta, pos. **Prefill** reads a continuing sequence's open rows into its first chunk and
re-evaluates the chunk from them (the recipe is row-separable, so those rows get the bits they had);
phase 1 never writes a record byte (it hands the open rows to a carry buffer the state pass copies), so a
call may read and write one slot. **Decode** (`csrc/decode_rec.cu`) evaluates row j = pos mod 64 from the
record through the same recipe functions and mma placements, appends it, and closes the chunk into S_{c+1}
at row 63. `record_state` materializes the forward's final-state formula from a record.

Engine decode execution (free; never enters the bits), two launches a step:
* `decode_row_kernel` — everything of row j that does not read S: CTA of 128 threads per (sequence,
  head), 41 KB shared memory so **five rows are in flight per SM** (the step is a chain of short dependent
  phases; concurrency, not bandwidth, is what it needs). Record rows stream in by cp.async in two groups
  (k^ / G / beta first; X / P / ki^b behind them, landing while row j's FOLD32 / gates / delta run); X rows
  are held lower-triangular packed (X_im = +0 for m > i is never read); kg rows load after the X phase into
  the dead P / ki^b space. Block entry computes ki^b eight channels a thread-step. One-live-row mma A
  operands are built from the live row's words (the same registers ldmatrix would give: placement facts
  unchanged). Hands w_j, qg_j, xu_j, Mq_j. and pos to the state pass; appends the row to the record.
* `decode_state_rec_kernel` — the part that reads S: persistent grid over (sequence, head, **32-column
  value slice**), 6 CTAs / SM, the next item's S^ slice, d / v slice rows and hand-off rows
  double-buffered by cp.async, the next-but-one item's (slot, pos) loaded a step ahead. Value columns are
  independent in the delta rule (d_j, u_j, o_j and S' columns are per value column; the K-axis sums stay
  whole inside each column), so the split needs no cross-CTA reduction and every element's chains are the
  ones the trainer evaluates. At a close it applies S_{c+1} = CHAIN64(kd^T, d; S·E(G_63)) to its slice.

**Record size audit (189,456 B per (slot, head); vLLM's fp32 state is 65,536 B; the record this replaces was 220,944 B).**

| Section | Bytes | Why it must be stored |
|---|---|---|
| SH = rn(S) bf16 | 32,768 | read every step (the B operand image of the d / o-seed chains) |
| SL, SC (low 16 bits, carry bit) | 32,768 + 2,048 | with SH they ARE the fp32 state, bit for bit: read by a close and by a record read only |
| k^ rows (bf16; kd^T at a close) | 16,384 | every later row's chains read them; the close folds them |
| v rows (bf16) | 16,384 | every later row's u chain |
| d rows [0, 63) (bf16) | 16,128 | o chains and the close (row 63 is the closing step's own, from shared memory) |
| kg rows [0, 63) (bf16) | 16,128 | every later row's w chain (row 63 is never read) |
| ki^b (bf16) | 12,288 | read every step of blocks 1-3 (recomputable: `KDA_REC_RECOMPUTE_KI`, below) |
| G rows (fp32) | 32,768 | cumulative decay; block entries, the close and continuations read it |
| X packed rows [0, 63) (fp32) | 8,448 | earlier rows of the solve |
| P (fp32) | 3,072 | the open block's solve |
| beta and pos | 272 | |

Every row is stored in its source precision. G, X and P are genuinely fp32 intermediates; nothing is
held at a wider type than its producer rounds it to. No open row is needed only by the chunk-close fold:
each one is also read by later rows of the same chunk. The state costs 33 bits an element (32 + the carry),
the minimum that keeps both the fp32 value and the bf16 image without a conversion in the step.

What remains is recomputable only at a price:
- ki^b from k^ and G (`csrc/record_profile.cuh`, `KDA_REC_RECOMPUTE_KI`): the record shrinks to 177,168 B; every
  step of blocks 1-3 pays the block-entry computation (up to 48 rows x 128 exponentials a head).
- kg from k^ and G: 3x the bytes read for those rows plus an exponential per element (not built).
- Any of the open rows from the raw inputs: a 64-row intra recompute per token.
- SH from the planes would read the state at twice the bytes per step, at fewer CTAs per SM (+15-25% estimated).

For vLLM the unit of capacity is the pool block, not the record.

## Relations (what is bitwise, what is tolerance)

| Relation | Kind | Gate |
|---|---|---|
| **rollout == trainer**: random prefill splits + decode steps over random subsets of a batch, == one whole-sequence `forward`, every output row and the final state (18 rollouts, 1-200 tokens, all chunk offsets and closes) | **bitwise** | `test_rollout_equals_whole_sequence_forward_bitwise` |
| records left by different splits continue identically (decode + prefill, outputs and state) | **bitwise** | `test_records_left_by_different_splits_continue_identically` |
| initial state at position 0, slot moves, zero-token moves | **bitwise** | `test_initial_state_and_slot_moves` |
| run-to-run (engine path), NULL decode lanes touch nothing | **bitwise** | `test_null_decode_lanes_and_run_to_run` |
| run-to-run: forward, functional decode, backward | **bitwise** | `test_forward_and_backward_are_run_to_run_bitwise` |
| batch invariance (composition, order, size 1..37 seqs, H 4 / 64) | **bitwise** | `test_a_sequence_is_bitwise_independent_of_its_batch` |
| functional `ops.decode` (dense state) == one-token `forward` call | **bitwise** | `test_decode_equals_one_token_forward_bitwise` |
| functional continuation with a dense state split at multiples of 64 == whole | **bitwise** | `test_chunk_aligned_continuation_equals_whole_bitwise` |
| functional continuation with a dense state split elsewhere | tolerance | `test_unaligned_continuation_is_within_tolerance` |

The dense-state functional API (`forward(initial_state=...)`, `ops.decode`) cannot carry an open chunk, so
its unaligned continuations are a different program by construction; the engine records are the path for
rollout.

## Determinism evidence

* SASS scan (`tests/test_sass_cpu.py`): **zero ATOM/ATOMS/ATOMG/RED instructions in every kernel of the extension** (24 SASS functions, including both `REC` instantiations of the chunk kernels, the engine decode, the record materializer and the o_proj leaf tree)
  (forward, decode, backward, layer stages). Every output element has one writer; weight-gradient
  reductions (dt_bias, A_log, norm weight) use fixed 256-row segments folded in ascending order.
* Battery: `python -m pytest isoexec/ops/kda/glm_sm100/tests isoexec/ops/kda/tests -q` on an SM100 device
  (CPU contract, SASS, GPU accuracy, relations, engine records). On the tree the package was written on,
  with its engine / TP layer and public layer tests, the battery gave **84 passed, 1 skipped**; the skip
  was the two-process NCCL test, which needs two GPUs and passed on two.

## Accuracy vs float64 oracles (measured; tests gate fwd 1e-2/2.5e-2, bwd 2e-2/5e-2 rel-L2/max-rel)

| Case | rel-L2 | max-rel |
|---|---|---|
| forward o / state, GLM gate, packed [300,64,17] | 3.7e-3 / 2.3e-3 | 5.3e-3 / 4.2e-3 |
| forward o / state, weak decay (g≈−0.05), T=1000 | 4.6e-3 / 3.5e-3 | 5.9e-3 / 5.6e-3 |
| forward o / state, unbounded decay (g down to −60) | 3.7e-3 / 2.4e-3 | 6.0e-3 / 3.9e-3 |
| backward dq, dk, dv, dg, dbeta, dh0 (through o and final state) | 3.1e-3, 3.3e-3, 3.4e-3, 3.4e-3, 3.0e-3, 2.1e-3 | ≤ 4.9e-3 |
| decode 40-step stream o / state | 3.9e-3 / 2.7e-3 | 9.8e-3 / 3.7e-3 |
| full GLM layer (bf16) vs float64 reference module: y, dx, all parameter grads | 5.8e-3, 7.1e-3, ≤ 1.6e-2 | — |
| pinned upstream oracle (`upstream_kda.py`) | within the forward gate | |

## Performance vs FLA 0.4.2 (temporary baseline)

One B200, provider order alternating per row,
CUPTI per-iteration kernel-time sum, median of 10 (20 for engine decode; spread < 1%, engine decode
B=256 3%: its positions advance through block entries and closes). H = 64, K = V = 128,
bf16 q/k/v, fp32 log decay (GLM bounded range), bf16 beta, fp32 h0, final state returned. FLA:
`chunk_kda` (prefill, train via autograd incl. state grads) and `fused_recurrent_kda` (decode; FLA
`chunk_kda` decode is 1724/3379 µs at B 256/512 and fails at B 1024). Engine decode runs records at
positions 640..703, so every open-chunk row and the chunk closes are in the median.

| Row | ours µs | FLA µs | ours/FLA |
|---|---|---|---|
| prefill (trainer `forward`) T=4096 / 8192 / 16384 | 673.5 / 1321.2 / 2583.8 | 850.9 / 1611.1 / 3161.8 | 0.79 / 0.82 / 0.82 |
| prefill (rollout, engine record) T=4096 / 8192 / 16384 | 709.4 / 1397.4 / 2743.6 | 851.6 / 1610.5 / 3163.1 | 0.83 / 0.87 / 0.87 |
| **decode (engine, exact) B=256** / 512 / 1024 | **390.7** / 754.5 / 1475.7 | **402.8** / 896.3 / 1804.6 | **0.97** / 0.84 / 0.82 |
| decode (functional dense one-token call) B=256 / 512 / 1024 | 387.1 / 758.7 / 1491.1 | 401.9 / 894.2 / 1800.9 | 0.96 / 0.85 / 0.83 |
| fwd+bwd T=4096 / 8192 / 16384 | 3148.9 / 6192.1 / 12272.7 | 3619.4 / 7077.7 / 14051.6 | 0.87 / 0.87 / 0.87 |

Round history: round 2 had the exact engine decode at 1.79 / 1.60 / 1.58x and
the record-aware chunk kernels costing 5-8% on the trainer forward; round 3 recovers the latter with the
`REC` instantiations (trainer forward back to round 1's 0.79-0.82) and brings the engine decode under FLA.
Engine decode split at B=256 (kernel µs): row kernel ~185, state pass ~192. Both are latency bound, not
bandwidth bound (~1.4 GB a step at B=256 against a measured 6.5-7 TB/s): the state pass reads S^ only
(537 MB) in 171 µs at j = 0. Tried and rejected (slower): programmatic dependent launch between the two
kernels, 3- and 4-stage state pipelines, a producer warp for the state prefetch, and one mma carrying both
the w and qg rows (a single dependent chain per warp).

## Backward design (`csrc/bwd_state.cu`, `csrc/bwd.cu`)

Gradient of the mathematical function (roundings as identities), no atomics, run-to-run deterministic,
not claimed batch-invariant. The forward kernels recompute w/u/qg/kd/Mq/E(G_L) and save S^ per chunk, d, X,
KK, xu, kg. `bwd_state`: reverse pass per (sequence, head, value slice): `dd = Mq^T dO + kd rn(dS)`,
`dS_c = dS·E(G_L) + qg^T dO − w^T rn(dd)`. `bwd_inter`: `dw, dqg, dkd, dMq, Σ_v S^ dS^`. `bwd_wy`:
`dY, dv, dkg`, then `dL = −X^T (dY β) X^T` by mma with hi/lo-split bf16 operands, `dKK, dbeta`.
`bwd_qkg`: decay-weighted product backward (off-diagonal by mma through the block references, diagonal by
the same per-token product chains), gate terms, reverse cumsum to dg, l2-norm backward. Intermediates that
are only ever bf16 mma operands or element-wise addends travel as bf16.

## Layer composition (`integration.py`)

`backend="sm100"` on `GLMKimiDeltaAttention` runs: the module's projections (GEMMs stay the `mm` family's),
Q/K/V causal width-4 SiLU convolutions on `gdn.conv_sm100` (its equation is GLM's: fp32 taps oldest to
current, SiLU, one bf16 rounding; bf16 activations with fp32 weights are supported in both directions),
the bounded forget gate `lb·sigmoid(e^{A_log}(f + dt_bias))` (fp32), sigmoid beta (bf16 boundary), the KDA
core, and the sigmoid-gated RMS output (`csrc/layer.cu`, fwd + bwd, deterministic segmented weight
gradients). Outside this package only `isoexec/ops/kda/__init__.py` names it; the GLM model module that
accepts `backend="sm100"` is not in this tree.

## Tensor parallelism and the engine layer (`engine.py`)

`engine.forward(shard, x, batch, pool)` is the inference composition of one GLM KDA layer on packed tokens
`[T, hidden]`, for engines (vLLM / SGLang) and Megatron-style TP. Mutable cache execution is
inference-only. Functional whole-sequence execution has a new native autograd binding
(`engine_training.py`, validation pending as of 2026-10-09); the older module composition
(`integration.py`, `backend="sm100"`) remains a distinct forward program.

**Engine training contract.** The trainer evaluates all eight leaves on one rank and uses
`shard_layer(module, trainable=True)` on each forward. Serving forward calls are unchanged:
chain projections and the fixed FP32 eight-leaf tree round once to BF16. Backward calls the
native convolution, KDA recurrence, gate and norm bindings. Projection gradients are dense
BF16 `dY @ W` and `dY.T @ X` through `mm_sm100/general`; any backward reduction width not
multiple of 64 receives paired zero padding for that native ABI. Its segmentation may depend
on token count, as permitted for backward. Distributed backward and mutable-cache backward
are explicitly refused. No new CUDA arithmetic or forward recipe is introduced.

GLM native `programs="engine"` also binds chain KV decompression and the checkpoint FP8
leaf-tree output projection for training. The FP8 master gradient follows the existing
`glm_clamped_fp8_master_dense_surrogate_v1` convention: BF16 master weights supply dX and
unquantized BF16 inputs supply dW; quantization is straight-through. The forward still
encodes the master and executes the frozen leaf-tree recipe. Training rebuilds live weight
views/copies for gradient and no-grad scoring. After serving weight sync, call
`NativeModel.refresh_engine_shards()` (or `engine.refresh_shard` for an adapter-owned shard)
before graph replay; refresh preserves captured tensor addresses. Adapter-specific attention
shards must be refreshed at that same boundary.

Its validation gates (projection float64 oracles, frozen-forward bitwise equality, repeated gradients,
post-update weights, and a complete layer float64 gradient oracle using the existing tolerances) belong to
the GLM model, which is not in this tree. GPU results are pending; the presence of these bindings is not a
validated engine/trainer identity claim. The existing KDA, convolution and general GEMM SASS
gates apply to the exact binaries used by this backward. Full-model post-optimizer identity
and engine program/tag transport remain separate required gates.

**Head sharding (TP = 1, 2, 4, 8).** `shard_layer(module, rank, world)`: rank r owns heads
`[r H/t, (r+1) H/t)`. Column-parallel (weight rows): q/k/v projections and their depthwise convolutions
(fused into one `[3 H_l 128]`-channel conv), f_b, g_b, b, `dt_bias`, `A_log`. Replicated: f_a, g_a and
the output-norm weight. Row-parallel: o_proj (columns of its weight).
* Every projection runs `mm_sm100/general`'s `sm100_tcgen05_chain_v1`, the mm family's program, called
  here and not modified. It is one ascending 16-wide tcgen05 chain per output element, from a zeroed FP32
  accumulator, rounded once. An element's bits read only its input row, its weight row and K: never M, N
  or the tile. So a shard's columns are the TP = 1 columns, and a token's are the same in any batch.
  `mm_sm100`'s segtree recipe is not used here, because its segment count S(K, N) reads N, i.e. the shard
  width.
* **o_proj: TP-invariant fixed tree** (pik ROOT / PUBLISH, as in `deepseek_v41_cache_sm100/tp.py`).
  - K = H·128 is cut into `TP_LEAVES = 8` leaves of H/8 heads. Each leaf is one FP32 chain (`bmm`,
    FP32 root). The 8 leaves meet in one balanced adjacent-pair FP32 tree (`leaf_tree_kernel`, this
    package), rounded once to bf16.
  - Rank r holds the aligned leaves `[8r/t, 8(r+1)/t)` and publishes its FP32 subtree root.
  - `tree_combine` (one process) or `DistributedReducer` (`all_gather` in rank order, then the same
    kernel; NCCL never adds) completes the tree.
* FP8-checkpoint projections are declined. Their TP-invariant GEMM (a per-leaf FP32 blocksum, as DeepSeek
  uses) belongs to the mm seam.

**State pool.** `KdaStatePool(slots, H_local)`:
- Engine records: `[slots, H_l, 189,456]` uint8 (state planes, X rows packed lower-triangular, P rows 48 wide, a 16-byte POS). The slot stride is free (`rec_slot_stride`; an engine may
  place each slot in its own page). An engine can wrap its own tensors with
  `KdaStatePool(..., records=, conv=)`.
- q/k/v conv windows in vLLM's dim-first layout `[slots, 3 H_l 128, 3]`, model dtype.
- `state_shapes(world, H)` / `state_dtypes()` mirror vLLM's `MambaStateShapeCalculator.kda_state_shape`.
  The recurrent state is the record, not a `[H, 128, 128]` fp32 tensor: 2.89x the bytes (3.37x before the state planes). It carries the
  open chunk, which is what makes rollout == trainer bitwise.

**Batches.** `EngineBatch(cu_seqlens, state_indices, has_initial_state, decode, null_block_id)`;
`batch_from_vllm(GDNAttentionMetadata)` builds one from vLLM's metadata (non-speculative).
- A batch with any prefill runs the chunked path for every sequence, decodes included: a one-token
  prefill row and a decode row are bitwise the same.
- `has_initial_state = False` starts a sequence fresh at position 0. Slot reuse needs no clearing.
  Preemption is recompute and is bitwise the same.
- A pure one-token batch (`decode=True`) runs the decode kernels. It has no host synchronization and
  fixed launch shapes; padded lanes are named by `null_block_id` (vLLM's `NULL_BLOCK_ID`) or a negative
  slot, and touch no state. It is CUDA-graph capturable.

**Relations (bitwise).** Measured by the engine-layer tests of the GLM model, which are not in this tree:

| Relation |
|---|
| layer output at TP = 2 / 4 / 8 (single-process, rank-order tree) == TP = 1 |
| each rank's records and conv windows == the TP = 1 pool's, restricted to its heads |
| continuous batching at TP = 1 and 2 == one whole-sequence layer call, every token. The schedule mixes prefill chunks with one-token lanes, pads decode calls with the null block, reuses slots after a request finishes, and preempts one request (recompute into another slot) |
| CUDA-graph captured decode steps (whole layer: GEMMs, conv update, gate, beta, decode kernels, norm, o_proj tree) == eager |
| padded decode lanes (null block, negative slot) touch no record or conv byte |
| records inside larger per-slot pages (engine strides) == dense pool |
| **real two-process NCCL TP = 2** (two GPUs), functional, prefill and decode == TP = 1 |

Composition check: the engine layer at TP = 1 is within 1e-2 rel-L2 of the module's own `backend="sm100"`
forward (cuBLAS projections, a different GEMM program).

## Owed

1. Engine decode margin at B=256 is ~3% (0.97x); the B=256 step sits at the per-item latency of both
   kernels (see Performance).
2. vLLM / SGLang model-file integration (the `KimiGatedDeltaNetAttention`-style wrapper that owns the
   layer's kv_cache tensors and calls `engine.forward`) is not written. `engine.py` is the seam it calls.
   Speculative multi-token decode is not served.
3. FP8-checkpoint projections under TP: the mm seam's per-leaf FP32 GEMM (declined here).
4. Registry: declared in `spec.py` (`kda.glm` / `sm100_kda_v1`); there is no central `kda` family table.

## Cleanup-preserved development conclusion

README records fixed recurrence recipe, engine state records, bitwise relations, FP64 accuracy and FLA 0.4.2 temporary-baseline measurements; trainer and engine backward are separate source paths. This records the historical experiment conclusion while standalone runs, logs and tuning scripts are removed. Existing numerical contract and measured tables above remain the source of detailed conditions; production selection is governed by the current resolver/model route.
