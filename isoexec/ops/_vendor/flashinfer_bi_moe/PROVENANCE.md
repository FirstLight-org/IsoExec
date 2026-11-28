# FlashInfer batch-invariant BF16 MoE: source provenance

Copied upstream source, not an independent IsoExec implementation.

| Source | Revision | Licence |
| --- | --- | --- |
| FlashInfer `flashinfer/fused_moe/batch_invariant` (#9283) and the modules it imports | release v0.7.5rc3 (tag commit `ed086f9d9e927bb189edb7d8de8269831a40e697`; PyPI wheel `flashinfer_python-0.7.5rc3-py3-none-any.whl`, sha256 `c1b588a773cb8c6fea9ac396b9ac94453962bdf1d82228e6666aad28193cc655`) | Apache-2.0 (`licenses/flashinfer-LICENSE.txt`, `licenses/flashinfer-NOTICE.txt`); parts derived from Thinking Machines' batch-invariant ops, MIT (`licenses/LICENSE.batch-invariant-ops.txt`) |

The directory mirrors FlashInfer's package layout so that the upstream relative imports are unchanged. Every file keeps
its upstream copyright and SPDX lines.

## What is copied
`fused_moe/batch_invariant/{api,combine,mterm,pdl,permute,route,route_plan,schedule,spec}.py` and `blocks/*`;
`gemm/batch_invariant/{generator,heuristic,spec}.py`, `blocks/{operands,mainloop}.py`;
`batch_invariant/{canon,jit,support}.py`, `fp_interface/*`, `sampling/sterm.py`, `quantization/cute_ext.py` (the FP32
`.rn` arithmetic primitives the kernels use; no quantizer); `tllm_enums.py`; `cute_dsl/availability.py`;
`cute_dsl/attention/fmha/helpers/{__init__,static_persistent_tile_scheduler}.py`.
Not copied: FlashInfer's package `__init__` modules (replaced by empty ones), the dense / FP8 / FP4 GEMM entry points of
`gemm/batch_invariant/api.py`, `fused_moe/batch_invariant/{runner,lean_cases,reference,kernel}.py` (the unified
`MoELayer` binding, the Lean case generator, the NVFP4-importing CPU reference), FlashInfer's JIT disk cache.

## Local changes (each with the restriction or need it serves)
Plumbing (no kernel change):
- `__init__.py` of every package: empty (FlashInfer's import the whole library).
- `gemm/batch_invariant/api.py`: only `sm_count_of`, `part_of`, `persistent_clusters` and their device helpers.
- `jit/cute_dsl_core.py`, `utils.py`: IsoExec-authored minimal stand-ins (in-process compile cache;
  `get_compute_capability`).
- `tllm_enums.py`: the API-logging decorator is an identity function.
- `gemm/batch_invariant/blocks/operands.py`: the tile-scheduler import is relative.
- Isolation from other CuTe-DSL users (restriction: own cache, no state shared with a host FlashInfer or FA4): every
  `cute.compile` call of the copy goes through `batch_invariant/jit.compile_isolated`, i.e. `cute.compile(...,
  no_cache=True)` (the DSL's process-wide IR-hash JIT cache and its file cache are bypassed); compiled kernels live only
  in `jit/cute_dsl_core.py`'s in-process dict, which refuses a compile inside CUDA graph capture (restriction: compile
  at placement, graphs replay compiled kernels only). Call sites: `fused_moe/batch_invariant/{route,route_plan,combine}.py`
  (1 each), `permute.py` (5), `blocks/compile.py` (6: the upstream 3 plus the weighted chain's 3 entries below).
- `batch_invariant/jit.py` `build_and_load`: calls the in-process cache with `(module, kernel, compile_fn)` only (the
  cache keys on (module, kernel); upstream's source-closure walk named `gemm/batch_invariant/schedule.py`, which is not
  vendored, and raised `FileNotFoundError` at every build).

Kernel changes (restriction served: IsoExec's V4.1 recipe `deepseek_v41_bf16_actor_v1` -- BF16 projection before the
activation, IEEE `expf` SiLU as ATen computes it, the route weight applied before FC2, one rounding; the combine an add
chain from the first term). Each is gated on a compile-time constant; every other chain and the default combine compile
the same kernels under the same names:
- `fused_moe/batch_invariant/spec.py`: the gated chain `v41_actor` (params `(limit,)`) in `ACTS` / `ACT_PARAMS`, new
  `WEIGHTED_ACTS`, `v41_actor_term` (`h = cvt_bf16(mul(mul(div(gc, add(expf(neg gc), 1)), uc), w))`, `gc = fmin(g', L)`,
  `uc = fmax(fmin(u', L), -L)`, `g' / u'` the accumulators rounded to bf16 and widened); `glu_pre` / `glu_elem` take an
  optional `w` (the row weight leaf, required iff the chain is weighted); `combine_reduction_from_first` and
  `combine_elem(..., from_first=False)`; `MoESpec.gemm1` uses the leaf `w` and `MoESpec.combine` the from-first unit-weight
  fold for the weighted chain (its fingerprint is distinct; every existing fingerprint is unchanged).
- `fused_moe/batch_invariant/mterm.py`: `TermAlgebra.rnd_bf16` (the `cvt bf16` term) and the opaque `expf` (no exact CPU
  model: `evaluate` takes `syms["expf"]`).
- `batch_invariant/quantization/cute_ext.py`: `CuteF32.rnd_bf16` (`cvt.rn.bf16.f32` then the exact widening) and
  `CuteF32.expf` (`cute.math.exp(x, fastmath=False)`, the device libdevice `expf`; not the declared `exp_term`, not `ex2`).
- `fused_moe/batch_invariant/blocks/epilogue.py`: `GroupedChain.weighted`; `_chain_elem(..., w=None)`;
  `grouped_epilogue_loop(..., row_w=None)` reads one fp32 weight per fragment row at `min(row0 + r, Tk - 1)`.
- `fused_moe/batch_invariant/blocks/kernel.py`: `GroupedBlockGemm.weighted(a, b, out, counts, row_w, grid, stream)`; the
  kernel's trailing `mRowW` is passed for the weighted chain only.
- `fused_moe/batch_invariant/blocks/swap_bf16.py`: `SwappedGroupedGemm.weighted` / `gathered_weighted`; the trailing
  `mRowW` likewise; `swapped_epilogue_loop(..., row_w=None)` reads `row_w[row0 + j]` for a live token column only.
- `fused_moe/batch_invariant/blocks/compile.py`: the weighted chain compiles the new entries with a fake `row_w (sym,)`
  fp32 operand after `counts` (kernel names carry the chain: `..._glu_v41_actor_bf16_p<limit bits>_...`).
- `fused_moe/batch_invariant/combine.py`: `CombineKernel(..., from_first=False)`, the fold of
  `combine_reduction_from_first` with unit weights: the kernel reads NO weight and multiplies nothing (the route weight
  entered GEMM1; `combine_unfinalized(y, None, dest, from_first=True, tokens=T)`), dropped slots skipped by `selp`, no live
  slot `+0`), kernel name suffix `_ff`.
- `fused_moe/batch_invariant/schedule.py`: `chain_nodes` passes the `w` leaf for the weighted chain.
- `fused_moe/batch_invariant/api.py`: `row_weights(scales, src)`; `grouped_gemm(..., row_weights=None)` and
  `grouped_gemm_gather(..., row_weights=None)` (required iff the chain is weighted, refused otherwise);
  `combine_unfinalized(..., from_first=False)`; `fused_moe` with a weighted chain passes the row weights to GEMM1 and
  combines unit weights from the first term (`w_round` refused there).
  `_const`: the cached constant vector is allocated on its device (`torch.full(..., device=dev)` instead of a host
  `torch.full(...).to(dev)`; same values), IsoExec's tensor-ownership rule.
- `fused_moe/batch_invariant/permute.py`: `PlanSmallKernel` / `get_compiled_plan_small` (decode latency): the plan of
  `n <= 256` slots in one CTA (the same counts / dest / src as `PlanKernel`, `align = 1`); `api.permute_plan` uses it
  for `n <= SMALL_MAX_N`; its weighted form also writes `row_weights`'s values (`api.permute_plan_weighted`: one launch
  instead of the plan + three) (`tests/test_flashinfer_bi_moe_vendor_gpu.py::test_one_cta_plan_is_the_plan`).
Planned: the TP column split and h exchange.

FP8 / FP4 prune (restriction: BF16 only, no FP8 / FP4 instantiation carried): the copied GEMM-class tables
(`gemm/batch_invariant/spec.py` `IN_FMTS`, `MMA_K_OF`, `MMA_OP_OF`, `ELEM_BITS_OF`, `MMA_RATE_OF`, `SF_VEC`, `SF_FMT`,
`CLASS_TAG`; `generator.py` `CTA_CHUNKS_MAX_TN`, `SWAP_WIDE_FMTS`, `BREUSE_FMTS`; `blocks/operands.py` dtype tables) keep
only `bf16`; the FP8 / FP4 format sets are empty; the block-scaled and block-scale record constraints of
`Knobs.well_formed` and the block-scaled instruction names of `Knobs.instruction` are deleted. The block-scale helpers (`spec.reduction_blockscale`, `block_scale_term`, `scale_term`, `sf_bytes`) and their call
sites in `GemmSpec.acc` / `out` and `Knobs.plan` / `verify` are deleted; what remains are format-membership tests
against the now empty sets (always false).

## Run-time requirements
torch, `nvidia-cutlass-dsl` 4.8.0 (CuTe DSL; compiles in process at first call), `apache-tvm-ffi` (the compiled kernels'
calling convention), `cuda.bindings` (cuda-python, a cutlass-dsl dependency): all already required by IsoExec for the
vendored FlashAttention-4 backward. No `flashinfer` import (CPU test with a meta-path blocker:
`isoexec/ops/tests/test_flashinfer_bi_moe_vendor_cpu.py`). Compile at placement, outside CUDA graph capture.

## How to re-vendor
Unpack the new FlashInfer wheel (or check out the tag), copy the files listed in `upstream_manifest.json` from its
`flashinfer/` package to the same relative paths, re-apply the local changes above, then regenerate the manifest
(per-file upstream path, upstream and local sha256) and run the CPU test and the MoE GPU battery.
