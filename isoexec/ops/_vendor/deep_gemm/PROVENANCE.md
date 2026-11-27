# DeepGEMM source provenance

This private package contains copied upstream source, not an independent IsoExec
implementation of the DeepGEMM kernels. No separately installed `deep_gemm` Python
package is required.

| Source | Revision | Retained license |
| --- | --- | --- |
| [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `5081f8513af57e61696de4f4f04b0e707ffb503e` | [MIT](licenses/deep_gemm.txt) |
| [DeepSeek DeepJIT](https://github.com/deepseek-ai/DeepJIT) | `2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43` | [MIT](licenses/deep_jit.txt) |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` | [BSD-3-Clause](licenses/cutlass.txt) |

`upstream_manifest.json` maps every copied file to its source path, original SHA-256
and local SHA-256. Modified entries explain the local changes. Original copyright
and license notices remain in the source and in `licenses/`.

## Scope of the copy

The copy is pruned to what the IsoExec BF16 model calls (see "Pruned to BF16"
below). `include/` holds only the headers in the include closure of what this
package compiles: the host extension (`csrc/python_api.cpp`) and the BF16 JIT
kernel families on SM100 (`sm100_bf16_gemm`, `sm100_bf16_mega_moe`,
`sm100_bf16_mega_gate` and the locality-domain probe), plus the SM100 FP8 MQA logits headers that
`attention/dsa_sm100` compiles ahead of time (section "Pruned to BF16"). The closure was computed with the compiler's own
dependency output (`-M`) for the host build and for the device pass of each
family's kernel header (`nvcc --cubin -M` at `sm_100a`). `csrc/` holds the host
code of those families and the shared runtime; the standalone
`csrc/indexing/main.cu` was never copied.

## Local changes

IsoExec authored `__init__.py` and `_extension.py` to load and build this private
namespace lazily. Build products go to the extension and JIT caches, outside the
source package. CUDA, PyTorch, a C++20 compiler and Ninja remain build and runtime
dependencies. Warm up before CUDA graph capture.

Upstream files changed locally (each listed with its edit in the manifest):

- The MLOpart callback import is qualified, unused testing and legacy Python
  imports are omitted, and optional elfutils diagnostics degrade to
  backtrace/dladdr when development headers are absent.
- The SM100 BF16 GEMM takes an optional leaf count (`bf16_gemm_nt_leaves`, IsoExec's
  `mm_sm100` leaf-tree program): K is cut into L equal leaves (L a power of two, at
  most 8), each one ascending chain from +0 into its own FP32 tensor-memory
  accumulator, and the leaves are folded by the balanced adjacent-pair tree
  (`__fadd_rn`) before the one rounding of the store. The fold is progressive: leaf l
  accumulates in tensor-memory slot l % 2; when an odd leaf completes, the epilogue
  (`epilogue/sm100_fold_leaf_pair`) forms the pair node and carries it through a binary
  counter of log2(L / 2) partials (equal-level nodes only, lower leaves on the left), and
  the last pair is folded in the store; per accumulator stage the tmem full / empty
  barriers complete once per leaf pair. Every value formed is a node of the tree, so the
  root is bit for bit the all-at-once fold, with 2 + log2(L / 2) accumulator slots
  instead of L (wider tiles fit tensor memory). The edit spans the binding in
  `apis/gemm.hpp` and `_api.py`, `GemmDesc.num_leaves` in `heuristics/config.hpp`,
  the layout filter in `heuristics/sm100.hpp` (non-swapped layouts whose leaf
  accumulators fit tensor memory), the template argument in
  `jit_kernels/impls/sm100_bf16_gemm.hpp`, and the kernel and epilogue headers
  `impls/sm100_bf16_gemm.cuh` and `epilogue/sm100_store_cd.cuh`. With one leaf, the
  default, the kernel runs upstream's program.
- Padded 64-row swap-AB tile of the SM100 BF16 GEMM (normal GEMMs, 1-CTA, K-major B, only when a
  caller requests a block-N multiple of 64): each CTA loads 64 weight rows into a B stage allocated
  for 128 and the MMA keeps UMMA M = 128; rows 64-127 land in tensor-memory lanes the store never
  writes out (`sm100_store_cd_swap_ab` stores the valid columns only). Twice the working CTAs at
  decode sizes, half the weight bytes each; every output element's chain is unchanged. Reason: the
  short-K projections stream twice cuBLAS's bytes per working CTA (dense lane r26). The edit spans
  `impls/sm100_bf16_gemm.cuh`, `epilogue/sm100_store_cd_swap_ab.cuh` and `heuristics/sm100.hpp`.
- Early launch of the SM100 BF16 GEMM (`set_early_launch`, Normal and Batched GEMMs, PDL
  launches only; off by default): mode 1 triggers the dependent launch
  (`cudaTriggerProgrammaticLaunchCompletion`) in every thread right after its own
  `cudaGridDependencySynchronize`; mode k >= 2 also loads up to k - 1 B stages of the
  CTA's first tile before that wait (the caller promises that B is not written by the
  preceding kernel: weights). The activation and every store stay after the wait.
  Tensor memory is addressed relative to the allocated base (a co-resident dependent CTA
  may hold other columns; the upstream `tmem_ptr == 0` assumption is replaced).
  Operands, MMAs and their order are unchanged, so every bit is. Reason: the short-K
  projections start after their predecessor ends while cuBLAS starts 1-2 us before it
  (dense lane r20). The edit spans `impls/sm100_bf16_gemm.cuh`, the template argument in
  `jit_kernels/impls/sm100_bf16_gemm.hpp`, `runtime/runtime.hpp`, `apis/config.hpp` and
  `_api.py`.
  Mode bit 64 (on top of a mode >= 1; opt-in) is the lean post-wait path: the TMEM base, the
  block scheduler and the epilogue operator (pure functions of shared memory written before
  the cluster sync, blockIdx and the shape) are formed before the wait, and the TMA producer
  (lane 0 of warp 0) triggers only after its first stage is issued; every other thread
  triggers right after the wait. Reason: a per-CTA trace measured 0.35 us between the wait's
  release and the first TMA issue on every call (dense lane r33).
  Mode bit 128 (opt-in) is the lean tail: the final barrier becomes one split-phase cluster barrier
  (`barrier.cluster.arrive` / `.wait`, an implicit one-CTA cluster when not clustered); the epilogue
  threads arrive from inside the store right after the CTA's last tensor-memory read (an optional
  `tail_arrive` argument of `sm100_store_cd` / `sm100_store_cd_swap_ab`, default off), every other
  thread at the end, and warp 0 frees tensor memory once all have arrived, overlapping the last
  shared-memory writes and TMA store issue. Reason: r33 traced cluster sync + free (0.26 us) after
  the last store on every call.
- Register partials for leaf trees (`set_leaf_register_partials(True)`, opt-in; dense lane WOB-DESIGN
  round 33 step (a)): the launch takes two epilogue warpgroups (384 threads); the binary counter's
  partials of the progressive fold live in the epilogue threads' registers (`sm100_fold_leaf_pair_regs`,
  `sm100_store_cd_leaf_regs` in `epilogue/sm100_store_cd.cuh`; warpgroup h owns half of every store
  block's columns, warp w reads lane quadrant w % 4), so tensor memory holds the leaf ring only. Each
  tree node is formed from the same two FP32 values in the same order. Host: the leaf-tree tensor-memory
  filter counts two leaf slots, layouts limited to UMMA N <= 96 (register budget), launch config
  (`heuristics/sm100.hpp`), the knob in `heuristics/runtime.hpp`, `apis/config.hpp`, `_api.py`.
- The shared MegaMoE layout header tags combine readiness with a per-launch
  workspace epoch (`layout/mega_moe.cuh` `isoexec_launch_epoch`) for the canonical
  specialization below, instead of `%gridid + 1`, because grid ids repeat across
  CUDA-graph replays of one kernel node. The upstream default is unchanged.
- The optional `isoexec_canonical=True` BF16 MegaMoE specialization
  (`bf16_mega_moe`, IsoExec recipe `deepseek_v41_bf16_actor_v1`;
  `impls/sm100_bf16_mega_moe.cuh`, its JIT host code
  `jit_kernels/impls/sm100_bf16_mega_moe.hpp`, the binding in `apis/mega_moe.hpp`
  and `mega/__init__.py`) fixes the rounding program of IsoExec's clamped SwiGLU
  expert MLP (canonical FP32 weighted SwiGLU with one BF16 rounding, combination in
  ascending global expert id with the shared expert last), uses the per-launch
  readiness tag above, and reads
  the L1 weights in IsoExec's master layout `[E, 2I, H]` (gate rows, then up rows)
  through a 5D TMA map whose box is upstream's granularity-8 gate/up interleaved
  tile, so the trainable masters are used in place. It requests precise CUDA
  arithmetic and has its own JIT key. Upstream's BF16 default is unchanged.
- The upstream SM100 BF16 grouped GEMM bindings `m_grouped_bf16_gemm_nt_contiguous`,
  `m_grouped_bf16_gemm_nn_contiguous` and `k_grouped_bf16_gemm_tn_contiguous` (with their argument check) are
  re-exposed (SM90 branches removed; the kernels and host implementations were already part of the copy): the MoE
  trainer's forward, dgrad and wgrad on device-resident group layouts; with the psum layout the k-grouped GEMM reads
  the groups' ends from the device, so no host count and no per-layer host sync.
- The BF16 MegaGate is vendored as IsoExec's router program
  (`bf16_mega_gate_isoexec`, called by `isoexec/ops/moe/router_sm100` `route_mega`), so
  its ids and weights equal `router_sm100.route()` bit for bit:
  - Logits: the hidden axis is cut into 4 fixed segments (`layout/mega_gate.cuh`
    `kNumSegments`, H/4 each, whole 64-wide K blocks; upstream's `H % 256 == 0`). In
    `impls/sm100_bf16_mega_gate.cuh` each segment is ONE ascending tcgen05 chain from zero
    (accumulate flag off on its first UMMA) in its own TMEM accumulator, never split or
    re-associated. Upstream's split-K across CTAs is kept as ownership: a slice owns
    `4 / num_split_k` whole segments (`num_split_k` in {1, 2, 4}; upstream's 8 would cut a
    segment and is not offered) and folds them in the epilogue by the segment tree
    (a subtree), the gate folds the slices (the upper levels), so every split evaluates
    ((s0 + s1) + (s2 + s3)) with `__fadd_rn`. Because the split is bit-neutral,
    deterministic mode no longer forbids it (`heuristics/mega_gate.hpp`).
  - Schedule-policy port (isoexec/ops/schedule_policy.py): `kNumSegments` is now 1 -- the policy gives a
    tensor-core sum one segment per shard -- so the logit is ONE chain over the whole hidden axis, K is never
    split (`num_split_k` = 1; `heuristics/mega_gate.hpp` sets `deterministic` when there is one segment) and the
    hidden alignment is one K block (64). The paragraph above describes the 4-segment program this replaced.
    The gate's renormaliser is `select.cuh`'s `selected_sum` (a chain in ascending expert id from +0). The
    canonical MegaMoE combine is unchanged: its chain starts from the first contribution, taken as is
    (`impls/sm100_bf16_mega_moe.cuh`).
  - TMEM budget: both epilogue stages of all segment accumulators share the 512 columns
    (`kAccumCols = 256 / segments-per-CTA`), so each heuristic candidate's largest token
    block is `get_max_block_tokens` (a 128-expert group on a CTA pair holds N/2 columns
    per CTA); every other launch-plan rule is upstream's.
  - Gate: the epilogue stores FP32 partial logits (no scoring); after upstream's score
    barrier each gate warp runs `router_sm100`'s `select_tree` program for one token. The
    selection functions are verbatim copies of `router_sm100/csrc/select.cuh`
    (`namespace rt100`) and `router.cu`'s `segment_tree` in `epilogue/sm100_mega_gate.cuh`
    (byte-compared by `router_sm100/tests/test_route_mega_cpu.py`). Upstream's scoring,
    top-k network, bias cache, image bias, fixed/random routing, physical map and
    finite-score trap are removed (`layout/mega_gate.cuh` drops `RoutingArgs` and the
    metadata cache). The kernel compiles without fast math, with `--fmad=false --ftz=false
    --prec-div=true --prec-sqrt=true` (the router extension's arithmetic profile).
  - The binding takes the router's selection arguments and writes its table (int32 ids,
    FP32 weights `[T, K+S]`, ascending expert id, shared slots last); optional
    `forced_num_split_k` / `forced_num_mma_ctas` restrict the ownership for tests, and
    `get_mega_gate_isoexec_config` reports the plan. Upstream's `bf16_mega_gate` program is
    not part of this copy.
- `mega/__init__.py` `SymmBuffer` rendezvouses its symmetric buffer through IsoExec's
  `isoexec/ops/moe/symm_rendezvous.py` `rendezvous` instead of torch's
  `_symmetric_memory.rendezvous`. torch numbers rendezvous handle exchanges per process,
  not per group, so a rank that set up a buffer its peers did not (a subgroup alone, a
  branch only some ranks took) would wait forever in the next rendezvous they share; the
  entry point first gathers the group's setup counts and buffer sizes and, on a skew,
  raises `RendezvousSkewError` on every rank (the caller keeps the NCCL realization).
  The buffer, its handle and the single-rank path are otherwise upstream's.
- The pruning edits listed in "Pruned to BF16".

## Pruned to BF16

IsoExec carries no FP8/FP4 material. This copy keeps only what the BF16 model
calls and deletes the rest instead of carrying it for later. The callers are
`isoexec/ops/mm/deepgemm_sm100` (`bf16_gemm_nt`, `bf16_gemm_nt_leaves`, `einsum`
with `bhr,hdr->bhd`, `set_block_size_multiple_of`, `use_deterministic_algorithms`,
`get_pdl`/`set_pdl`) and `isoexec/ops/moe/deepseek_v41_sm100/mega.py`
(`SymmBuffer` with `mma_type="bf16xbf16"`, `get_block_m_for_mega_moe`,
`bf16_mega_moe(..., isoexec_canonical=True)`) and `isoexec/ops/moe/router_sm100`
(`bf16_mega_gate_isoexec`, `get_mega_gate_isoexec_config`). The manifest's `pruned_to_bf16`
block lists every removed upstream file with its pinned upstream hash and reason.

Kept:

- The SM100 BF16 GEMM family (`jit_kernels/impls/sm100_bf16_gemm.hpp`,
  `impls/sm100_bf16_gemm.cuh` and its epilogue, MMA, scheduler, TMA and PTX
  headers), bound as `bf16_gemm_nt`, `bf16_gemm_nt_leaves` and `einsum`
  (`bhr,hdr->bhd` only).
- The SM100 BF16 MegaMoE (`jit_kernels/impls/sm100_bf16_mega_moe.hpp`,
  `heuristics/mega_moe.hpp`, `impls/sm100_bf16_mega_moe.cuh`, the MegaMoE layout,
  scheduler, symmetric-buffer and barrier headers), bound as `bf16_mega_moe` and
  `bf16_mega_moe_isoexec`, with `get_symm_buffer_size_for_mega_moe`,
  `get_token_alignment_for_mega_moe` and `get_block_m_for_mega_moe`.
- The SM100 BF16 MegaGate as IsoExec's router program
  (`jit_kernels/impls/sm100_bf16_mega_gate.hpp`, `heuristics/mega_gate.hpp`,
  `apis/mega_gate.hpp`, `impls/sm100_bf16_mega_gate.cuh`, `epilogue/sm100_mega_gate.cuh`,
  `layout/mega_gate.cuh`, `scheduler/mega_gate.cuh`), bound as `bf16_mega_gate_isoexec`
  and `get_mega_gate_isoexec_config` (see "Local changes").
- The locality-domain probe that MegaMoE launches use
  (`jit_kernels/impls/sm100_locality_domain.hpp`, `impls/sm100_locality_domain.cuh`,
  `apis/locality_domain.hpp`, `runtime/locality_domain.hpp`, `runtime/mlopart.hpp`,
  `locality_domain.py`, `utils/mlopart.py`).
- The config bindings (`apis/config.hpp`), the epilogue classes the BF16 GEMM
  signature takes (`apis/epilogue_class.hpp` without `FP8CastWithClamp`), the JIT
  runtime (DeepJIT), and the CUTLASS/CuTe headers in the closure.

Dropped because they are FP8/FP4:

- The FP8xFP4 GEMMs on SM100 (`sm100_fp8_fp4_gemm_1d1d`) and FP8 GEMMs on SM90
  (`sm90_fp8_gemm_1d1d`, `sm90_fp8_gemm_1d2d`) with every `fp8_*`/`fp4_*` GEMM
  binding and alias, and `fp8_einsum`.
- The FP8xFP4 MegaMoE (`sm100_fp8_fp4_mega_moe`, `fp8_fp4_mega_moe` and its
  IsoExec canonical `fp8_fp4_mega_moe_isoexec` specialization),
  `transform_weights_for_mega_moe` and its SF helpers.
- The FP8/FP4 MQA logits bindings and JIT host code (`apis/attention.hpp`,
  `jit_kernels/impls/sm100_mqa_logits.hpp`), the sparse and SM90 variants
  (`sm100_sparse_mqa_logits`, `sm90_fp8_mqa_logits`, `layout/sparse_mqa_logits.cuh`).
  The SM100 FP8 kernel headers themselves are kept (next section).
- The scaling-factor layout kernels (`smxx_layout`, `apis/layout.hpp`,
  `utils/layout.py`), which only transpose FP32 scaling factors or pack them into
  UE8M0 for the FP8/FP4 kernels.
- The cuBLASLt GEMMs (`smxx_cublaslt`, including `cublaslt_nvfp4_gemm_nt`). The BF16
  GEMM and einsum reached cuBLASLt only outside deterministic mode, which IsoExec
  always sets, so those dispatches are removed and the SM100 kernel serves both modes.
- The FP8 output epilogue binding (`epilogue.FP8CastWithClamp`) and the FP8/FP4
  cast helpers of `utils/math.py`.

Kept for the V4.1 cache program's one FP8 exception (FP8 index keys and queries,
`port/CACHE-SPEC.md` sections 3-4; executor rules section 2):

- The SM100 MQA logits kernels (`impls/sm100_mqa_logits.cuh`,
  `scheduler/sm100_mqa_logits.cuh`, `scheduler/sm100_paged_mqa_logits.cuh`,
  `layout/mqa_logits.cuh`, `common/ring_pipeline.cuh`; added 2026-11-18 from the
  pinned revision, byte-identical to vLLM 0.33's bundled copy of the kernel). They
  are not part of this package's JIT: `isoexec/ops/attention/dsa_sm100` compiles
  them ahead of time in its `mqa` sub-extension (`csrc/mqa/mqa.cu` holds the
  template instantiations -- H 32 / 64, D 128, FP8, BLOCK_Q 128 / H, SPLIT_KV 384,
  3 Q / KV / TMEM stages, 128 + 384 threads, 148 CTAs, PAGE_KV 64 / 128, metadata
  threads 256 / 512 / 1024 -- the tensor maps and the launches that upstream's JIT
  host code generated), with this `include/` as its only include root, hidden host
  visibility and `-fno-gnu-unique`, and the same load-time symbol check as this
  package. The kernels' arithmetic is unchanged. Edits (manifest entries carry
  them; each is launch structure, addressing or schedule, never a value):
  `cutlass::arch::launch_dependent_grids()` (its header is outside the vendored
  CUTLASS closure) becomes `cudaTriggerProgrammaticLaunchCompletion()`; a null
  `cu_seq_len_k_start` means key 0 for every row; the paged schedule takes a
  request run's context as the maximum over its rows (a run may end with padded
  rows of context 0) and reads a run's block-table row through `indices` (one row
  per sequence instead of per token). Re-vendoring: copy the five headers from the
  new revision, re-apply the three edits, update the manifest digests, rebuild the
  `mqa` sub-extension and rerun `dsa_sm100/tests/test_index_logits_gpu.py` and the
  native bit-equality script (`reviews/v41-release-dev/attn/bench/index_bench.py`).
  The engine's gathered prefill route (`dsa_sm100/csrc/mqa/gather.cu`, IsoExec code)
  relies on the unmodified contiguous kernel's per-row key start: row r's logit of key
  kv is stored at column `kv - cu_seq_len_k_start[r]` and only for
  `cu_seq_len_k_start[r] <= kv < cu_seq_len_k_end[r]` (`impls/sm100_mqa_logits.cuh`,
  `output_bases` and `store_token`), so rows of different sequences may share a Q
  block. No edit is needed for it; a re-vendor must keep that store contract
  (`deepseek_v41_cache_sm100/tests/test_index_gather_route_gpu.py`).

Dropped because the BF16 model does not call them:

- Every SM90 family (`sm90_bf16_gemm`, `sm90_bmk_bnk_mn`,
  `sm90_tf32_hc_prenorm_gemm`, `heuristics/sm90.hpp`) and the SM90 branches of the
  kept bindings.
- The TF32 hyperconnection pre-norm GEMM (`sm100_tf32_hc_prenorm_gemm`,
  `apis/hyperconnection.hpp`), MegaMHC (`sm100_mega_mhc`, `apis/mega_mhc.hpp`,
  `layout/mega_mhc.cuh`) and einsum `bmk,bnk->mn` (`sm100_bmk_bnk_mn`), and upstream's
  MegaGate scoring program (`bf16_mega_gate`; the files are kept as the router program).
- The unused bindings of the kept BF16 GEMM family (`bf16_gemm_nn`/`tn`/`tt`,
  M-grouped contiguous/masked and K-grouped BF16 GEMMs, einsum `bhd,hdr->bhr` and
  `bhd,bhr->hdr`), `get_symm_buffer_for_mega_moe`, and `utils/dist.py`.

Edited upstream files (manifest entries carry the details): `csrc/python_api.cpp`,
`csrc/apis/gemm.hpp`, `csrc/apis/einsum.hpp`, `csrc/apis/mega_moe.hpp`, the MegaGate
files above (all but `scheduler/mega_gate.cuh`, which keeps upstream bytes),
`csrc/apis/epilogue_class.hpp`, `csrc/jit_kernels/impls/runtime_utils.hpp`
(includes the shared `heuristics/utils.hpp` instead of the removed
`heuristics/sm90.hpp`), `_api.py`, `mega/__init__.py` (`SymmBuffer` defaults to
`bf16xbf16` and rendezvouses through `symm_rendezvous`), `utils/__init__.py` and
`utils/math.py`.

Shared code on the BF16 path keeps upstream bytes, so FP8/FP4 type support
remains inside files the BF16 kernels and host code need. These are the CUTLASS
numeric types (`cutlass/float8.h`, `float_subbyte.h`, `exmy_base.h`), the
dtype-generic epilogue operators and store code (`epilogue/operators.cuh`
`CastToFP8WithClamp`, `sm100_store_cd*.cuh`), the UE8M0 helpers of
`common/math.cuh`, the FP8/FP4 cases of the shared host helpers
(`runtime_utils.hpp`, `utils/layout.hpp`, `heuristics/config.hpp`,
`heuristics/mega_moe.hpp`, `jit_kernels/impls/epilogue_class.hpp`) and the
scaling-factor sizing branch of `get_symm_buffer_size_for_mega_moe`, which BF16
calls with `bf16xbf16` and so never takes. No binding reaches them.

## Isolation from other DeepGEMM copies in the process

vLLM bundles an upstream DeepGEMM built from the same `deep_gemm`/`deep_jit` C++
namespaces. GCC emits inline globals such as `deep_gemm::jit` (the JIT runtime that
holds the include root and cache) as `STB_GNU_UNIQUE`, which the dynamic loader
unifies process-wide even across `RTLD_LOCAL` extensions, so whichever `_C.init`
ran last would choose the include root for both copies. This copy is private by
construction:

- `_extension.py` builds with `-fvisibility=hidden -fvisibility-inlines-hidden
  -fno-gnu-unique` and refuses the built object if any dynamic symbol is
  GNU_UNIQUE or an exported `deep_gemm`/`deep_jit` definition.
- The JIT environment prefix is `ISOEXEC_DG` (`DG_*` configures upstream copies
  only); the disk cache is `ISOEXEC_DG_JIT_CACHE_DIR`, default
  `${XDG_CACHE_HOME:-~/.cache}/isoexec/deep_gemm_jit`.
- The cache-key signature is `isoexec-deep_gemm-<revision[:12]>-include-<sha256 of
  include/>[:16]+cutlass-<version>`, so cubins are never exchanged with an upstream
  cache and any vendored header change gets a new key.
- `_api.check_jit_identity()` fails closed at load unless the JIT include root is
  this package's `include/`; `init` refuses re-initialization with another root or
  signature.

## Build and run-time requirements

The host extension (`csrc/python_api.cpp`) is built by IsoExec's own extension build (`_extension.py`, torch's
`cpp_extension` with the shared SM100 profile) on first import; the BF16 kernels are JIT-compiled by the vendored
DeepJIT runtime at first call with the CUDA toolkit's `nvcc` (`CUDA_HOME`), as every IsoExec SM100 extension already
requires. Nothing is imported from an installed `deep_gemm` or from vLLM's bundled copy
(`tests/test_deepgemm_vendor_isolation_cpu.py`, both load orders).

## How to re-vendor

1. Check out upstream DeepGEMM at the new commit (with its `third-party/cutlass` and DeepJIT submodules) and record
   the three revisions in the table above.
2. Copy the include closure of the kept entry points (compute it with `nvcc --cubin -M` at `sm_100a` for each kept
   kernel family's header and `-M` for the host build), plus `csrc/` of those families, preserving paths.
3. Re-apply the local changes listed under "Local changes" (the leaf-count GEMM, the canonical BF16 MegaMoE
   specialization and its launch epoch, the `SymmBuffer` rendezvous through `symm_rendezvous`, the MegaGate router program with the verbatim `select.cuh` / `segment_tree`
   copies taken from `router_sm100/csrc`, the import and diagnostic edits, the BF16 prune) and regenerate
   `upstream_manifest.json` (original and local SHA-256 per file, the pruned list with reasons).
4. Run `isoexec/ops/tests/test_deepgemm_vendor_{provenance,isolation}_cpu.py` and
   `moe/router_sm100/tests/test_route_mega_cpu.py`, then the GPU batteries of `mm/deepgemm_sm100`,
   `moe/deepseek_v41_sm100` and `moe/router_sm100` (`test_route_mega_gpu.py`; bits equal to the previous copy and to
   `route()` on every gated shape).
