"""Declare environment overrides and their runtime forwarding channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Disposition vocabulary.
FUNCTION = "function"
DEPLOYMENT = "deployment"
DIAGNOSTIC = "diagnostic"

# Forwarding channels a declaration names: the trainer actors' and the engine actors' runtime env
# (both built by ``runtimes/environment.resolved_environment``), plus ``ISOEXEC_VLLM_ENV``, a dict
# applied in-process before vLLM init.
TRAIN = "train"
ENGINE = "engine"
VLLM_ENV = "vllm_env"  # runtimes/vllm/vllm_patches.py ISOEXEC_VLLM_ENV

# Validate runtime-side names at import so misspelled declarations cannot bypass checks.
SIDE_VOCABULARY: frozenset[str] = frozenset({"trainer", "engine", "both"})


@dataclass(frozen=True)
class Flag:
    """One env override."""

    name: str
    default: str
    sides: Tuple[str, ...]  # subset of {"trainer", "engine", "both"}
    selects: str
    disposition: str
    forwarded_by: Tuple[str, ...] = ()  # subset of {TRAIN, ENGINE, VLLM_ENV}
    notes: str = ""
    default_dynamic: bool = False  # code default is COMPUTED, not a literal (see below)

    @property
    def has_malformed_sides(self) -> bool:
        """True when ``sides`` uses a token outside the documented vocabulary."""
        return any(s not in SIDE_VOCABULARY for s in self.sides)


# Ordered by concern rather than alphabetically.

FLAGS: List[Flag] = [
    Flag(
        "ISOEXEC_MOE_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_MOE_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_MLA_SM90_PROFILE", "fast", ("both",),
        "MLA forward compiler arithmetic profile", FUNCTION, (TRAIN, ENGINE),
        notes="Selects FTZ and division policy; strict and fast are distinct numerical identities.",
    ),
    Flag(
        "ISOEXEC_MLA_SM90_CUDA_HOME", "", ("both",),
        "CUDA toolkit used to build MLA forward kernels (default: torch's CUDA_HOME)", FUNCTION, (TRAIN, ENGINE),
        notes="Compared conservatively: this path selects compiler binaries and headers, not just a cache.",
    ),
    Flag(
        "ISOEXEC_MLA_SM90_CUTLASS", "", ("both",),
        "CUTLASS checkout used to build MLA forward kernels (default: the one make setup installs)", FUNCTION, (TRAIN, ENGINE),
        notes="Compared conservatively: selecting another header tree can change generated numerical code.",
    ),
    Flag(
        "ISOEXEC_MLA_SM90_BUILD_DIR", "", ("both",),
        "external MLA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_MLA_SM90_BUILD_VERBOSE", "0", ("both",),
        "MLA compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GQA_SM90_CUTLASS", "", ("both",),
        "CUTLASS checkout used to build the vendored FA3 kernels (default: the one make setup installs)",
        FUNCTION, (TRAIN, ENGINE),
        notes="Compared conservatively: selecting another header tree can change generated numerical code.",
    ),
    Flag(
        "ISOEXEC_GQA_SM90_BUILD_DIR", "", ("both",),
        "external gqa_sm90 (vendored FA3) extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GQA_SM90_BUILD_VERBOSE", "0", ("both",),
        "gqa_sm90 compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_MLA_SM90_PARTIAL_BYTES", "1073741824", ("both",),
        "maximum MLA partial-tree scratch bytes", DEPLOYMENT, (TRAIN, ENGINE),
        notes="Bounds published canonical-tree nodes; changes scheduling, not the tree or its merge order.",
    ),
    Flag(
        "ISOEXEC_MLA_SM100_PARTIAL_BYTES", "2147483648", ("both",),
        "maximum MLA partial-tree scratch bytes (SM100)", DEPLOYMENT, (TRAIN, ENGINE),
        notes="Bounds published canonical-tree nodes (2080 B per row-segment slot); changes which items the "
              "plan cuts, not the tree or its merge order (the leaf's own fold of a tree equals the published "
              "combine, gated by mla_sm100's rel_combine_tree_widths). A saturated-decode cut that does not fit "
              "refuses rather than cutting less.",
    ),
    # Master switch.
    Flag(
        "ISOEXEC",
        "0",
        ("both",),
        "master IsoExec enable; gates every forwarding block",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="special-cased: presence gates the whole allowlist loop",
    ),
    # VLLM_ENV channel: delivered by the ISOEXEC_VLLM_ENV dict, not by either Ray-actor allowlist.
    Flag(
        "VLLM_USE_AOT_COMPILE",
        "0",
        ("engine",),
        "disable vLLM AOT compile (batch-invariant setup)",
        DEPLOYMENT,
        (VLLM_ENV,),
        notes="ISOEXEC_VLLM_ENV dict only; in-process pre-init",
    ),
    Flag(
        "VLLM_USE_V2_MODEL_RUNNER",
        "1",
        ("engine",),
        "pin vLLM Model Runner V2, the runner every IsoExec engine hook and the breakable PIECEWISE graphs target",
        DEPLOYMENT,
        (VLLM_ENV,),
        notes="ISOEXEC_VLLM_ENV dict and runtimes/environment.resolved_environment, so an engine actor builds "
        "its VllmConfig on V2 regardless of vLLM's per-architecture default.",
    ),
    Flag(
        "NCCL_ALGO",
        "",
        ("both",),
        "deterministic NCCL all-reduce (allreduce:tree) matching the trainer pin",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE receives it through ISOEXEC_VLLM_ENV; the TRAINER through a direct env_vars[] assignment gated "
        "on the NCCL pin, not the allowlist loop.",
    ),
    Flag(
        "NCCL_MIN_NCHANNELS",
        "",
        ("both",),
        "NCCL min channels=1 (determinism pin)",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE via ISOEXEC_VLLM_ENV; TRAINER via a direct env_vars[] assignment gated on the NCCL pin.",
    ),
    Flag(
        "NCCL_MAX_NCHANNELS",
        "",
        ("both",),
        "NCCL max channels=1 (determinism pin)",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE via ISOEXEC_VLLM_ENV; TRAINER via a direct env_vars[] assignment gated on the NCCL pin.",
    ),
    # Spec / attention backend: function-half, must match on both sides.
    Flag(
        "VLLM_BATCH_INVARIANT",
        "0",
        ("both",),
        "vLLM batch-invariant kernel path",
        FUNCTION,
        (TRAIN, ENGINE, VLLM_ENV),
        notes="also hard-set to 1 in ISOEXEC_VLLM_ENV (in-process, pre-init)",
    ),
    Flag(
        "ISOEXEC_MM_SM100",
        "0",
        ("both",),
        "the Blackwell tcgen05 segment-tree dense GEMM (signature-moving; sm100 only)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Not bit-equal to Triton or to cuBLASLt by contract (tcgen05 accumulation and the fixed per-weight "
        "K split move the bits by design), so a flip requires a gate re-freeze and both sides must agree on it. "
        "The binding record is the impl id and ops/mm/mm_sm100/segments.json's digest "
        "(ops/mm/mm_sm100/integration.py::record).",
    ),
    Flag(
        "ISOEXEC_MM_SM90_SEGTREE",
        "1",
        ("both",),
        "the Hopper wgmma segment-tree dense GEMM as the sm90 mm provider (recipe selector; default on)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="On by default: ops/mm/mm_sm90 is installed outermost on matmul_persistent on trainer and engine. "
        "0 removes it for a diagnostic A/B, leaving the vendored Triton kernel; its S = 1 program is measured "
        "bit-equal to that kernel on census shapes, but the recorded impl id (sm90_wgmma_segtree vs "
        "triton_batch_invariant) and segment-table digest change, so both sides must agree "
        "(ops/mm/mm_sm90/integration.py::record).",
    ),

    Flag(
        "ISOEXEC_MODEL_PATH",
        "",
        ("both",),
        "the model path that labels the dense-GEMM coverage report",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Names the model the process is already running in the shape-table coverage report. The census "
        "itself is registered from the constructed model (mm_shapes.set_live_context); this selects nothing "
        "and cannot move a bit.",
    ),
    Flag(
        "ISOEXEC_MM_FWD_ONLY",
        "0",
        ("trainer",),
        "scope the batch-invariant mm/addmm override to the FORWARD (backward -> cuBLAS)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Backward-only and therefore bitwise-neutral by construction: the predicate returns Triton unless a "
        "backward graph task is executing on this thread, which no forward is. Checkpoint recompute forwards "
        "are explicitly re-pinned to Triton, since they must match the original forward bitwise and the live "
        "gate cannot see them. Fails closed -- no hook means no scoping.",
    ),
    Flag(
        "ISOEXEC_MM_FWD_ONLY_BMM",
        "0",
        ("trainer",),
        "bmm half: scope the batch-invariant aten::bmm override to the FORWARD",
        DEPLOYMENT,
        (TRAIN,),
        notes="The bmm half of ISOEXEC_MM_FWD_ONLY, kept independent because the two install on different "
        "Libraries at different moments. Same backward-only discriminator and recompute re-entry hook, so "
        "bitwise-neutral by construction; a failed post-install verification de-registers and leaves vLLM "
        "unscoped bmm in place.",
    ),
    # GDN.
    Flag("ISOEXEC_GDN", "0", ("both",), "GDN engine patch (IsoExec installs no trainer-side fla shim)", FUNCTION, (TRAIN,)),
    Flag(
        "ISOEXEC_GDN_KERNEL",
        "recurrent",
        ("both",),
        "delta-rule kernel: chunk|cpr|recurrent (MUST agree)",
        FUNCTION,
        (TRAIN,),
        notes="Default and vocabulary are core/gdn_kernel_env's, which both the declaration site (models/qwen3_5) "
        "and the executing sites (ops/gdn/shared/gdn_ops) now parse through. 'chunk' was the executing default while no "
        "model declares a chunk composition, so unset named a function the process did not run. An explicitly set "
        "value also selects the GDN package (ops/_dispatch alternatives): on sm90 'cpr' selects ops/gdn/gdn_cpr_sm90 "
        "(private state pool) instead of ops/gdn/gdn_recurrent_sm90, and on sm100 'recurrent' selects "
        "ops/gdn/gdn_recurrent_sm100 (state in vLLM's pages, with ISOEXEC_GDN_NATIVE_STATE=1) instead of "
        "ops/gdn/gdn_cpr_sm100. See docs/gdn-modes.md.",
    ),
    Flag(
        "ISOEXEC_GDN_TRAINER_KERNEL",
        "",
        ("trainer",),
        "override trainer delta-rule kernel only (ablation)",
        FUNCTION,
        (TRAIN,),
        notes="Not an IsoExec configuration: the contract pins ONE gdn.core kernel for all four sites, so a build "
        "refuses when this asks for a trainer/engine split. Same vocabulary, same parser.",
    ),
    Flag(
        "ISOEXEC_GDN_NATIVE_KERNELS",
        "0",
        ("both",),
        "vLLM native fused GDN kernels on both runtimes",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GDN_NATIVE_STATE",
        "0",
        ("engine",),
        "GDN state in vLLM mamba kv_cache (forces fp32 cache)",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="Required by the recurrent GDN packages (gdn_recurrent_sm90, gdn_recurrent_sm100) and refused by the "
        "CPR packages, whose state pool is private.",
    ),
    Flag(
        "ISOEXEC_GDN_CHUNKED_PREFILL",
        "0",
        ("engine",),
        "GDN continuation-chunk resume from carried state",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GDN_NATIVE_CONV",
        "0",
        ("both",),
        "cpr: native conv fn/update pair (MUST agree)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE",
        "1",
        ("both",),
        "byte-exact native_matched_prep elementwise fusion with size-gated exact L2 chain",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Default ON. Strict CUDA shape/dtype/layout admission; OFF restores the canonical eager expression. The "
        "L2 candidate keeps torch.sum itself so the pinned fp32 reduction association cannot change.",
    ),
    Flag(
        "ISOEXEC_GDN_ANALYTIC_CONV_BWD",
        "0",
        ("trainer",),
        "packed analytic native-conv VJP (backward-only, default OFF pending live proof)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_GDN_CONV_BWD_ROWS",
        "8192",
        ("trainer",),
        "analytic native-conv VJP row chunk size (backward-only)",
        DIAGNOSTIC,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_GDN_CPR_FUSED_SCATTER",
        "1",
        ("engine",),
        "one-launch open-chunk buffer scatter (data movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GDN_CPR_DECODE_TUNE",
        "1",
        ("engine",),
        "measure the sm90 CPR decode split (CTAs per head) per batch size when a pool binds",
        DEPLOYMENT,
        (ENGINE,),
        notes="Execution only: every split of gdn_cpr_sm90's decode gives the same bits (csrc/decode.cu; the "
        "battery runs every split), so the measured table may differ between processes, ranks and devices. Run "
        "once per (device, value heads, key heads) outside graph capture; 0 keeps the shape heuristic.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_FUSED_ROWS",
        "1",
        ("engine",),
        "one-launch slot->row resolution, both widths (integer index movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Integer index arithmetic only -- clamp, gather, width cast -- so there is nothing to round. OFF "
        "restores the ATen chain, bit-identical.",
    ),
    Flag(
        "ISOEXEC_SHARD_SAMPLER",
        "1",
        ("engine",),
        "Model Runner V2 vocab-parallel sampling head: shard-local Gumbel-max + leaf8 logprob, O(rows) exchange",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Bit-identical to the gathered path (vLLM gumbel_sample on the all-gathered fp32 logits, the leaf8 "
        "full-row logprob at the token, vLLM's rank count) for every row without NaN logits: the Gumbel value is "
        "vLLM's per-column expression keyed by the global column, the argmax a total order (largest value, lowest "
        "column), the leaf states and their merge the recipe's own code. A row NaN in every column gets the gathered "
        "path's token (probed at init), a NaN logprob and rank 0; a row NaN in only some columns has no order-free "
        "argmax and may get a different (in-range) token. Engages only for rows with no logits processing, no "
        "top-k/top-p, logprobs<=0 and no speculative (drafts always decline)/structured/batch-sharded sampling, "
        "on SM90, with an unpadded vocabulary split into the recipe's 8 leaves: TP must divide 8 (TP <= 8). "
        "Installs only with the V2 leaf8 logprob hook active and the vLLM gumbel source / Triton version it was "
        "verified against (vllm_patches.SHARD_SAMPLER_*); its extension is built and probed at model init and "
        "agreed over TP. Everything else takes the gathered path (declines are counted in the "
        "[ISOEXEC-SHARD-SAMPLER] census). OFF restores the gathered path for every step.",
    ),
    Flag(
        "ISOEXEC_FUSED_SAMPLER",
        "1",
        ("engine",),
        "serve plain V1 and Model Runner V2 decode sampling with the SM100 decode sampler "
        "(ops/logprobs/logprobs_sm100/sampler.py)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Logprobs stay the pinned rowinv recipe's (leaf_states + merge_rows, the trainer's kernels), so ON "
        "and OFF return the same logprob bits for a given token. Tokens are drawn by Gumbel-max with Philox "
        "noise instead of vLLM's exponential race on the torch RNG: deterministic and batch invariant, but a "
        "different stream, so ON and OFF sample different tokens. Steps with top-k/top-p, min_p, per-request "
        "seeds, speculative decoding or logprobs outside {0, 1} keep vLLM's sampler either way. Under Model "
        "Runner V2 the served steps draw V1's stream (not V2's per-request seed/position stream), and any "
        "active logits processor (logit bias, allowed tokens, min_tokens, penalties, bad words) keeps V2's "
        "sampler.",
    ),
    Flag(
        "ISOEXEC_GDN_FUSED_SPLIT",
        "1",
        ("engine",),
        "one-launch q/k/v + alpha/beta materialisation for the GDN native core (byte movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="One Triton kernel writes the five contiguous outputs the ATen chain produced in five launches. No "
        "float arithmetic at all -- masked load, masked store, source dtype -- so the proof is byte equality, "
        "not an allclose. OFF restores the ATen chain, bit-identical.",
    ),
    Flag(
        "ISOEXEC_FUSED_RESIDUAL_NORM",
        "1",
        ("both",),
        "run each residual join + following RMSNorm as one fused kernel (compiler/realization/fusion.py)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The fused kernel computes rn_bf16(x + r) (what aten.add on two bf16 tensors computes), stores it, "
        "and runs the unchanged RMS recipe of the arch (sm90 v1; sm100 the width's v1/v2) on it; backward is the unfused backward op for op. The pass "
        "runs on both sides over the same captured graph, so the binding digests agree. OFF keeps the pairs.",
    ),
    Flag(
        "ISOEXEC_MLP_FUSED_ACT",
        "1",
        ("both",),
        "bind sm90 mlp.act to the one-launch fused silu*mul kernel (ops/mlp/mlp_sm90)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Recipe v1 is the eager F.silu(g) * (u + 0.0), rounded where it rounds (bf16 silu, then the bf16 "
        "product), so OFF and ON give the same bits; the impl id differs, so both sides must agree for the "
        "binding digest. OFF binds the eager three-launch statement.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_MIN_PAGES",
        "0",
        ("engine",),
        "minimal vLLM mamba pages under cpr (private pools carry the state)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Engine memory only, moves no bits: cpr never reads vLLM GDN state pages (it uses them as a "
        "slot-id source), so they shrink and the freed KV pool returns to attention. Scoped to cpr "
        "without native state; recurrent and native-state keep full pages.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_APC",
        "0",
        ("engine",),
        "prefix caching under cpr via the boundary-state store",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="Engine-only but FUNCTION-half: it changes which tokens the GDN layers scan for a request. Requires "
        "ISOEXEC_GDN_CHUNKED_PREFILL=1 and cpr on the gdn_cpr_sm90 pool. Every worker publishes the boundary "
        "checkpoints it holds in a /dev/shm index and the scheduler admits a prefix-cache hit only where all of "
        "them hold the checkpoint of exactly that prefix (runtimes/vllm/cpr_apc.py), at any topology; 1 and "
        "2/shm/shared are equivalent. Set by register_native / the verl integration when prefix caching is "
        "enabled under cpr.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_APC_MB",
        "1024",
        ("engine",),
        "device-memory ceiling for the CPR-APC boundary-state store",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Capacity only: an eviction costs a cache hit, never a wrong resume. An evicted checkpoint leaves "
        "the index at once and its bytes stay readable for APC_RETIRE_STEPS more engine steps, so a hit the "
        "scheduler admitted just before still resumes it; the budget bounds the live checkpoints, retiring ones "
        "are a transient on top.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_SLEEP",
        "0",
        ("engine",),
        "CPR state arena is DISCARDED at engine sleep and re-created at wake",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Engine memory and time only: the arena contents are dead across a generate/train/generate boundary, "
        "and the wake path zeroes the arena and resets every row map so the pool equals a fresh build. It "
        "removes the D2H backup and H2D restore of that dead state and holds the arena released through the "
        "weight-broadcast window.",
    ),
    Flag(
        "ISOEXEC_VLLM_EXACT_SLEEP",
        "1",
        ("engine",),
        "level-2 engine sleep for IsoExec models: Parameters discarded (every coverage-checked update rewrites "
        "them), every other weights-pool byte kept in a pinned stash and restored at the weights wake; 0 = level 1",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Engine memory and time only: the bytes a forward reads after the update are the same as with level 1 "
        "(Parameters from the stream, all else restored), so no numerics change. Removes the D2H/H2D copy of the "
        "whole weight pool at every sleep/wake. A forward after a level-2 sleep and before a finished update is "
        "refused, not served from discarded weights. verl's rollout server picks the sleep level from it; a worker "
        "whose own value is 0 sleeps an IsoExec model at level 1 even when level 2 is requested (stock level 2 "
        "never runs on an IsoExec model).",
    ),
    Flag(
        "ISOEXEC_GDN_PIN_CONFIGS",
        "1",
        ("both",),
        "pin every autotuned FLA kernel to configs[0], identically in every process -- ON, and "
        "CORRECTNESS MACHINERY, not a tuning knob",
        FUNCTION,
        (TRAIN,),
        notes="Correctness machinery, not a tuning knob. The OFF path hands the kernel back to the Triton autotuner, "
        "which can select a racy config that returns different results run to run on identical inputs; "
        "autotuning is also per-process, so trainer and engine can land on different configs and therefore "
        "different reduction orders. Set 0 only to reproduce an unpinned baseline, never on a gate run.",
    ),
    Flag(
        "ISOEXEC_GDN_CONFIG_INDEX",
        "0",
        ("both",),
        "which entry of each kernel's own config list PIN_CONFIGS pins",
        FUNCTION,
        (TRAIN,),
        notes="Only for bisecting a bad config on a new stack; it MUST match on trainer and engine or the two pin "
        "different reduction orders.",
    ),
    Flag(
        "ISOEXEC_GDN_FLA_BACKWARD",
        "0",
        ("trainer",),
        "FLA fused Triton VJP (backward is free)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Trainer-only; the backward carries no batch-invariance contract.",
    ),
    Flag(
        "ISOEXEC_OPTIMIZER_DIGEST_BEFORE",
        "0",
        ("trainer",),
        "re-digest the parameters before every optimizer update (default: reuse the previous update's digest)",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Trainer-only receipt cost (integrations/verl/engine.py); the weights change only in the update, so "
        "the default digests each weight version once. Selects no kernel and moves no bit.",
    ),
    Flag(
        "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR",
        "",
        ("trainer",),
        "directory for on-disk full-distribution comparison evidence (rows, keys, refusals)",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Trainer-only; read by integrations/verl/full_distribution.py. Empty disables recording. It selects nothing "
        "and cannot move a bit: the comparison outcome is identical with or without it.",
    ),
    Flag(
        "FLA_TILELANG",
        "1",
        ("trainer",),
        "FLA TileLang backend (0 on B200; arch-specific)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Forwarded by a direct env_vars[] assignment in the megatron branch rather than the allowlist loop. "
        "Arch-scoped.",
    ),
    # MoE flags that affect numerical agreement between runtimes.
    Flag(
        "ISOEXEC_MOE_PERMUTE_SORT",
        "0",
        ("engine",),
        "counting-sort permute index build, decoupled from the fused-router flag",
        DEPLOYMENT,
        (TRAIN,),
        notes="Selects the same counting-sort index build ISOEXEC_MOE_ROUTER_O2 gates, without also asking for "
        "the fused router, so it serves routers the O2 guard declines. Score-function-agnostic: a counting sort "
        "over a boolean map plus a gather, no arithmetic. EP=1 only, since at EP>1 the counting sort would "
        "leave the tail uninitialised.",
    ),
    Flag(
        "ISOEXEC_MOE_ROUTER_O2",
        "0",
        ("engine",),
        "fused router + permute-sort (engine-only, bitwise)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_MOE_COMBINE_SORT",
        "0",
        ("engine",),
        "stable counting-sort combine (integer permutation)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_MOE_COMBINE_SORT_BLOCK_P",
        "1024",
        ("engine",),
        "combine-sort tile knob (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_MOE_COMBINE_SORT_WARPS",
        "4",
        ("engine",),
        "combine-sort warp knob (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_MOE_COMBINE_SORT_MAX_WORK",
        "33554432",
        ("engine",),
        "combine-sort work cap (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_MOE_COMBINE_BLOCK_H",
        "512",
        ("engine",),
        "combine tile H (reduction-free, neutral)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag("ISOEXEC_MOE_COMBINE_WARPS", "4", ("engine",), "combine warps (neutral)", DEPLOYMENT, (TRAIN,)),
    Flag(
        "ISOEXEC_MOE_PACKED_EXPERTS",
        "1",
        ("both",),
        "the unified MoE layer's expert parameters are slices of one packed [E,2I,H] / [E,H,I] buffer, "
        "so the per-forward gate_up/down stack is a free alias instead of a full copy",
        DEPLOYMENT,
        (TRAIN,),
        notes="Not a cache: there is exactly ONE copy of the bytes, because each parameter's .data IS a slice of "
        "the buffer, so the 'stack' cannot go stale relative to the parameters. Bitwise-neutral by construction "
        "-- same layout, same values, same kernels, and under grad the backward hands each parameter the slice "
        "cat/stack would have. Off restores the exact torch.stack expression. Packing is REFUSED unless every "
        "expert parameter privately owns its whole storage, so a DDP / distributed-optimizer / FSDP bucket is "
        "never severed (A41); a break is observed by the param_storage mutation epoch and falls back to "
        "torch.stack, and a re-pack is licensed once per weight sync. See ops/moe/shared/packed_experts.py.",
    ),
    # pik / collectives.
    Flag("ISOEXEC_PIK", "0", ("both",), "pik TP/EP-invariant row-parallel reduction", FUNCTION, (TRAIN, ENGINE)),
    Flag(
        "ISOEXEC_WGRAD_FUSION",
        "1",
        ("trainer",),
        "gradient-accumulation fusion: linear weight gradients accumulate into Megatron's fp32 main_grad "
        "in the wgrad GEMM epilogue (ops/mm/wgrad.py; sm90 CUTLASS, else cuBLAS beta=1)",
        FUNCTION,
        (TRAIN,),
        notes="Backward only: no forward, logprob or engine bit depends on it. Gradients are rounded once "
        "(main_grad + chain) instead of twice (bf16 chain, then the DDP hook's fp32 add) and are deterministic "
        "run to run. On sm90, a call the CUTLASS kernel serves (16-byte aligned operands) computes the same "
        "element chain under every tile config, so a TP shard's gradient is bitwise the matching block of the "
        "unsharded one; a call it does not serve (unaligned operands, and every call on sm100) uses cuBLAS "
        "addmm(beta=1, fp32 out), a different chain, so the gradient bits of a layer can differ between those "
        "two paths (across layers, ranks or architectures). 0 restores the bf16 wgrad + DDP-hook add.",
    ),
    Flag(
        "ISOEXEC_TRAINER_BUCKET_PACK",
        "1",
        ("trainer",),
        "Megatron DDP layout places each packed projection (GDN in_proj, attention q|k|v|gate) as adjacent "
        "rows in row order, so the packed GEMM operand is a view of the bucket instead of a per-forward cat",
        FUNCTION,
        (TRAIN,),
        notes="Storage placement only: the GEMM reads the same bytes in the same order as torch.cat produced, "
        "so forward bits are unchanged (mm segments depend on K only). 0 keeps Megatron's own layout.",
    ),
    Flag(
        "ISOEXEC_PIK_LEAVES",
        "8",
        ("both",),
        "leaf-tree fan-out G (contract constant, MUST match)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_PIK_LEAF_DTYPE",
        "bf16",
        ("both",),
        "leaf dtype (contract constant, MUST match)",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="A contract constant that must match on both runtimes. It is not a tuning knob: leaf dtype changes the "
        "policy function, so logprobs from a bf16-leaf run and an fp32-leaf run are different numbers and the "
        "two are not comparable.",
    ),
    Flag(
        "ISOEXEC_PIK_P2P",
        "1",
        ("both",),
        "symmetric-memory P2P vs NCCL transport (bitwise-equal)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Selects the engine transport (sm90 and sm100 admit P2P); the trainer is always NCCL. An "
        "architecture listed in pik_tp_invariant.NCCL_ONLY_ARCHS (none today) is NCCL-only: unset or 0 is NCCL "
        "there, and an explicit 1 is refused.",
    ),
    Flag(
        "ISOEXEC_PIK_TRAINER_P2P",
        "1",
        ("trainer",),
        "trainer pik transport: symmetric-memory P2P per group, admitted by physical device identity",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bitwise-neutral (transport only). A group-agreed probe admits a TP group whose ranks share a host "
        "(hostname + boot id), are not MIG instances and sit on pairwise distinct GPUs by UUID, so Ray actors "
        "that each see their GPU as cuda:0 get P2P; groups spanning hosts or sharing a GPU, and NCCL-only "
        "architectures, keep NCCL. 0 on any rank keeps that rank's whole group on NCCL (the rank still joins "
        "the probe and votes no).",
    ),
    Flag(
        "ISOEXEC_PIK_TRAINER_P2P_MAX_MB",
        "32",
        ("trainer",),
        "trainer: per-rank owner-reduce wire MiB above which an admitted P2P group uses the NCCL owner path",
        DEPLOYMENT,
        (TRAIN,),
        notes="Transport only (both classes are bit-admitted against the legacy path). 4xH100: P2P wins at 1k "
        "tokens (21 MB/rank), the NCCL owner path from ~4k tokens (84 MB). 0 = no limit. Group-agreed: the "
        "physical probe takes the minimum over the group's ranks (0 counts as no limit), so ranks may set "
        "different values and still pick the same transport.",
    ),
    Flag(
        "ISOEXEC_PIK_OVERLAP",
        "",
        ("trainer",),
        "pik comm/compute overlap: pipelined row-parallel sites and async dgrad tree sums on a comm lane",
        DEPLOYMENT,
        (TRAIN,),
        notes="Schedule only: every element keeps its leaves, tree and root round. Empty = off (both sides). "
        "1 turns it on only with more than one hardware queue (CUDA_DEVICE_MAX_CONNECTIONS > 1; Megatron/verl "
        "set 1, where streams cannot overlap). Agreed across the TP group at first use (a split setting disables "
        "it for the group instead of hanging). The lane runs only symmetric-memory kernels: an NCCL-transport "
        "chunk or gradient sum, or an unadmitted class, runs on the caller's stream.",
        default_dynamic=True,
    ),
    Flag(
        "ISOEXEC_PIK_OVERLAP_CHUNKS",
        "0",
        ("trainer",),
        "pik overlap: token chunks per row-parallel site (0 = off, -1 = derived per site, n = fixed)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Speed only; chunk bounds are 128-row aligned; any depth is bitwise-equal. -1 derives the depth "
        "from the site's GEMM/reduce time ratio (shape, TP size, wire planes, transport, SM count, hardware "
        "queues; pik.lane.plan_chunks). Off by default: slower on 4xH100 at the measured trainer shapes. "
        "Agreed across the TP group.",
    ),
    Flag(
        "ISOEXEC_PIK_OVERLAP_MIN_M",
        "0",
        ("trainer",),
        "pik overlap: rows from which a row-parallel site is pipelined (0 = derived: a chunk fills a wave)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Speed only. Agreed across the TP group.",
    ),
    Flag(
        "ISOEXEC_PIK_OVERLAP_COMM_SMS",
        "0",
        ("trainer",),
        "pik overlap: SMs withheld from a producer GEMM chunk while it overlaps a reduce (0 = none)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Speed only: caps the sm90 segment-tree GEMM's persistent CTAs; tile arithmetic is unchanged.",
    ),
    Flag(
        "ISOEXEC_PIK_ONESHOT_MB",
        "2",
        ("both",),
        "one-shot/two-shot crossover MiB (both paths equal)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_NATIVE_NORM_MEMO",
        "0",
        ("both",),
        "memoize the DEVICE-derived half of torch _native's RMSNorm admission predicate",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Host-time only. It memoizes the two device-property lookups inside torch _fused_rms_norm admission "
        "predicate, which depend on nothing but the device index, by rebinding the module globals the predicate "
        "resolves through -- no re-registration and no change to the kernel that runs.",
    ),
    Flag(
        "ISOEXEC_ROPE_SM90_MAX_POSITIONS",
        "262144",
        ("both",),
        "rope_sm90 cos/sin table capacity (positions beyond it are a device assert)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Capacity only: a row's bits are the same for every capacity that contains its position.",
    ),
    Flag(
        "ISOEXEC_ROPE_SM100_MAX_POSITIONS",
        "262144",
        ("both",),
        "rope_sm100 cos/sin table capacity (positions beyond it are a device assert)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Capacity only: the table kernel computes each row from its own position, so a row's bits are the "
        "same for every capacity that contains its position.",
    ),
    Flag(
        "ISOEXEC_ROPE_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_ROPE_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_NORMS_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_NORMS_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_CONV_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_CONV_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_LOGPROBS_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_LOGPROBS_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_LOGPROBS_SHARD_SM90_BUILD_DIR", "", ("engine",),
        "external CUDA extension build cache (vocab-parallel sampling head)", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_LOGPROBS_SHARD_SM90_BUILD_VERBOSE", "0", ("engine",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_GDN_CPR_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_GDN_CPR_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_MM_BWD_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_MM_BWD_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_DIGEST_SM90_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_DIGEST_SM90_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_DIGEST_SM100_BUILD_DIR", "", ("both",),
        "external CUDA extension build cache", DEPLOYMENT, (TRAIN, ENGINE),
        default_dynamic=True,
        notes="Path only; kernel sources, compiler flags and arithmetic are unchanged.",
    ),
    Flag(
        "ISOEXEC_DIGEST_SM100_BUILD_VERBOSE", "0", ("both",),
        "CUDA extension compiler log verbosity", DIAGNOSTIC, (TRAIN, ENGINE),
    ),
    Flag(
        "ISOEXEC_PIK_FASTLAUNCH",
        "0",
        ("both",),
        "memoize Triton's per-call launch derivation for pik's PINNED kernels (host-time only)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Host-time only: it memoizes what Triton re-derives on every launch (specialization binder, cache key, "
        "grid callable), all of which are constant for a pinned kernel at a fixed shape class, which pik "
        "kernels are by construction.",
    ),
    Flag(
        "ISOEXEC_PIK_FUSED_BARRIER",
        "0",
        ("both",),
        "fold pik's symmetric-memory barriers INTO the tree all-reduce kernel: 1 launch per site",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Cannot move a bit: the reduction core is generated once and pasted into both templates, and every "
        "(world, numel, wire, root) re-proves it on live operands with a bit-pattern compare before use, "
        "falling back per shape on any mismatch. Rank uniformity is a hard requirement and is enforced rather "
        "than assumed, because the two launch structures rendezvous differently -- disagreement hangs rather "
        "than slows down.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_AR",
        "1",
        ("both",),
        "owner tree all-reduce for row-parallel sites: leaf/node wire planes, ONE reduce launch per site",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Same leaves, same balanced fp32 tree, same single root round as the legacy pik site; only who "
        "evaluates which element and how bytes move changes (symmetric-memory one-/two-shot with in-kernel "
        "per-block barriers over a fixed small grid, or NCCL all_to_all + all_gather across nodes). Agreed by a "
        "group collective on first use (a split flag disables it everywhere instead of hanging) and each kernel "
        "class is bit-compared against the legacy path on live operands before use. On by default (the production "
        "configuration; with ISOEXEC_PIK_OWNER_LAMPORT=1 and ARNORM=1 it is the fused Lamport add+RMSNorm site); "
        "=0 opts out to the legacy pik site, which computes the same bits.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_ONESHOT_KB",
        "",
        ("both",),
        "owner all-reduce one-shot/two-shot crossover, KiB of one rank's wire planes (empty = per-world table)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only: both branches evaluate the same per-element expression. Must be identical across a "
        "TP group (checked at first use).",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_LEAF_MIN_M",
        "1",
        ("both",),
        "owner all-reduce: rows from which the bf16 leaf wire replaces the legacy fp32 node wire",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only: leaf wire and node wire are the same tree expression (a node is the fp32 sum of the "
        "same leaves). Below it the legacy GEMM schedule and node wire run, with the barriers and root cast "
        "folded into the one reduce launch. Agreed across the TP group at first use. Default 1 (leaf wire at every M, "
        "the input the Lamport owner kernel takes); 1024 was the barrier-protocol crossover.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_NODE",
        "0",
        ("both",),
        "owner all-reduce below LEAF_MIN_M: one-launch fp32-node reduce instead of the legacy site",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only (same tree, same root round). Off by default: at world 4 the legacy Triton push "
        "kernel measured faster on fp32 nodes at M >= 256. Agreed across the TP group at first use.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_ROUTE",
        "0",
        ("both",),
        "route generic pik tree_all_reduce calls (fp32 nodes) through the owner kernel",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only. Must be identical across ranks: it changes the launch structure of those calls.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_PUSH",
        "1",
        ("both",),
        "owner all-reduce two-shot: owners push roots to every rank (1, the legacy schedule) or ranks pull (0)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only; agreed across the TP group at first use.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_MAX_BLOCKS",
        "",
        ("both",),
        "owner all-reduce grid cap (blocks of 512 threads, grid-stride); each block does one handshake",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only: the grid moves which thread evaluates an element, never the expression. Must be "
        "identical across a TP group (checked at first use).",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_LAMPORT",
        "1",
        ("both",),
        "owner all-reduce on p2p: Lamport (flag-in-data) protocol, one launch, no barrier round trips",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only: same tree, same root round; receive buffers pre-filled with an all-ones NaN sentinel "
        "(a wire element equal to it is sent as the canonical NaN, which every fp32 tree add returns anyway), "
        "three generations advanced in-kernel (graph-replay safe). Effective only with ISOEXEC_PIK_OWNER_AR=1. "
        "Agreed across the TP group at first use; each class is bit-compared against the legacy path.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_LAMPORT_KB",
        "16384",
        ("both",),
        "owner Lamport all-reduce: one generation of the receive areas, KiB (6 x this per rank, symmetric)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed/memory only: a site whose wire does not fit one generation runs the barrier owner kernel "
        "(same bits). Must be identical across a TP group (checked at first use).",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_LAMPORT_ONESHOT_KB",
        "",
        ("both",),
        "owner Lamport all-reduce: one-shot/two-shot crossover, KiB of one rank's wire (empty = per-world table)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Speed only: both branches evaluate the same per-element expression. Agreed at first use.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_LAMPORT_BLOCKS",
        "",
        ("both",),
        "owner Lamport all-reduce grid cap (blocks of 512 threads; empty = half the SMs; min over the group)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Bits unaffected (rows are packed per block when the grid is smaller). Blocks spin on rows written "
        "by peers and by other blocks of the same grid, so the whole grid must be resident at once: the cap is "
        "at most one block per SM, and the default leaves half the SMs to kernels running concurrently on other "
        "streams. Nothing enforces residency against those kernels: a larger value is safe only when concurrent "
        "kernels that wait on peers (NCCL: one CTA per channel) occupy at most SM count - cap SMs.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_ARNORM",
        "1",
        ("both",),
        "fuse the residual join + RMSNorm consuming a row-parallel output into the owner Lamport kernel",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Launch structure only: the epilogue is norms recipe v1 (add_rms_fwd's per-row program, same "
        "lane chain, fold, rstd and roundings) on the bf16 root. Wired by the residual+norm fusion pass on "
        "proven single-consumer pairs; no-grad forwards only; admitted per class by a group-agreed bit compare "
        "against the legacy reduction followed by the bound unfused add+norm. Requires ISOEXEC_PIK_OWNER_AR=1.",
    ),
    Flag(
        "ISOEXEC_MM_SM90_PDL",
        "1",
        ("both",),
        "launch the sm90 segment-tree GEMM with programmatic dependent launch (griddepcontrol)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Schedule only: built with CUTLASS_ENABLE_GDC_FOR_SM90, every kernel schedule waits on its "
        "predecessor before its first global access and triggers its dependents after its last mainloop tile; "
        "no operand or rounding changes. Read once per process by the extension.",
    ),
    Flag(
        "ISOEXEC_PIK_OWNER_PDL",
        "1",
        ("both",),
        "launch the owner Lamport kernel with programmatic dependent launch (prologue overlaps the GEMM tail)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Schedule only: the kernel touches no memory before griddepcontrol.wait. Agreed at first use.",
    ),
    Flag(
        "ISOEXEC_PIK_AR_VEC",
        "4",
        ("both",),
        "element-to-thread layout of the p2p reduce kernel: vec CONTIGUOUS elements per thread",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The layout moves which thread computes an element and nothing else: the element set, each element load "
        "sources, the tree association and the single rounding are unchanged, and every (world, numel, wire, "
        "root, branch) re-proves the layout against the vec=1 reference. Contiguous elements per thread is what "
        "lets Triton emit 128-bit accesses.",
    ),
    Flag(
        "ISOEXEC_PIK_FUSED_ROOT_CAST",
        "1",
        ("both",),
        "round the tree ROOT to bf16 in the reduce kernel's store instead of a separate cast kernel",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Deliberately separate from ISOEXEC_PIK_FUSED_BARRIER: fusion is bit-neutral, this MOVES a "
        "rounding point, so it is a different claim with a different proof. Legal because the root, unlike an "
        "internal node, is the same node at every TP size, so rounding it cannot make the expression TP- "
        "dependent. Applied only where the round is provably the next operation -- no bias, fp32 leaves, bf16 "
        "output -- and the guard is asserted identical in both row-parallel twins so they cannot drift.",
    ),
    Flag(
        "ISOEXEC_ENGINE_NCCL_UNPIN",
        "0",
        ("engine",),
        "unpin engine NCCL (proven neutral, -16% generate)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS",
        "8",
        ("engine",),
        "channel CEILING for the engine unpin (empty = fully unpinned ~24ch)",
        DEPLOYMENT,
        (TRAIN,),
        notes="A fully unpinned engine NCCL holds a large block of non-sleepable buffers through the whole training "
        "window; a channel ceiling keeps most of the latency win at a fraction of that memory. Bitwise-neutral "
        "for the same reason the unpin is: pik owns the invariant reduction. Set empty to restore the full "
        "unpin.",
    ),
    Flag(
        "ISOEXEC_MOE_AFTER_FC2_OWNER",
        "0",
        ("engine",),
        "owner combine with routing weights applied after the fixed FC2 tree",
        DEPLOYMENT,
        (TRAIN, ENGINE, VLLM_ENV),
        notes="Opt-in SM90 owner composition preserving the BF16 FC2 root, BF16 routing "
        "multiply, ascending-expert FP32 fold and final BF16 round. Unsupported shapes "
        "retain the existing combine path; training backward is unchanged.",
    ),
    Flag(
        "ISOEXEC_MOE_AFTER_FC2_GATHER",
        "0",
        ("both",),
        "gather FC1 inputs directly for no-grad experts weighted after FC2",
        DEPLOYMENT,
        (TRAIN, ENGINE, VLLM_ENV),
        notes="Opt-in SM90 gathered FC1 omits the probability multiply when weights belong "
        "after FC2, preserving rounded SiLU and up-product operations. Existing geometry "
        "admission and fallback remain; grad-enabled expert execution is unchanged.",
    ),
    # Parallelism topology / resharding.
    # Engine-arg policy: proven bitwise-safe, hence deployment-half.
    Flag(
        "ISOEXEC_MOE_PIK_OWNER_COMBINE",
        "0",
        ("engine",),
        "owner-computes MoE combine (token-slice owner does tree+topk-sum, all-gather bf16 finals)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bit-identical to the all-reduce combine -- only the executing rank changes: same peer partials, same "
        "pik tree order, same ascending-expert fp32 k-sum, same single bf16 round. Read by the unified "
        "owner host (moe_pik_combine_owner._engine_eligible), which the engine reaches through ops/moe/resolve.py "
        "moe.combine -> moe_sm90.integration.engine_combine. TRAINER_ENV sets it to 1.",
    ),
    Flag(
        "ISOEXEC_MOE_SHARED_OWNER_FUSION",
        "0",
        ("engine",),
        "exact shared+routed owner composition: one exchange and one output barrier",
        DEPLOYMENT,
        (TRAIN,),
        notes="Requires ISOEXEC_MOE_PIK_OWNER_COMBINE=1. One owner composition replaces the separate shared "
        "reduction, shared gate launch, routed owner exchange and final add, while preserving the shared tree, "
        "its bf16 root round, the ascending-k sum and the single final round. On the unified host the owner "
        "combine has no routed-only form (schema 3's fp32 fold and the residual cast are one rounding schedule), "
        "so with this OFF and a shared expert present the engine runs the in-place trees. Each live geometry is "
        "admitted by a bit-pattern comparison with a group-wide verdict before use; the trainer is unchanged.",
    ),
    Flag(
        "ISOEXEC_PIK_FUSED_OWNER_COMBINE",
        "0",
        ("engine",),
        "fold the owner-combine's two symm-mem barriers INTO its push kernel: 1 launch",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The owner combine stages, barriers, pushes and barriers itself rather than entering pik generic path, "
        "so ISOEXEC_PIK_FUSED_BARRIER cannot reach it; this folds those two barriers into the push kernel. "
        "Bitwise-equal by admission, and enablement must be uniform across ranks or the differing launch "
        "structures hang instead of slowing down.",
    ),
    Flag(
        "ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS",
        "8",
        ("engine",),
        "grid gate for the fused owner-combine: above this many blocks, fusion is refused",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Not a tuning knob but the measured crossover between the per-block in-kernel barrier and the device- "
        "wide barrier pair. Both paths are bitwise-equal by admission, so moving it cannot move a bit; raising "
        "it without a measurement re-arms a known regression.",
    ),
    Flag(
        "ISOEXEC_MOE_OC_ROWS32",
        "1",
        ("engine",),
        "owner-combine my_rows as an int32 VIEW of the counting sort's [T,k] output",
        DEPLOYMENT,
        (TRAIN,),
        notes="An integer-only view of the counting sort output, so it is bit-free by construction; it removes the "
        "separate rows-build launches from the owner-combine preamble.",
    ),
    Flag(
        "ISOEXEC_MOE_OC_WIRE_STAGE",
        "1",
        ("engine",),
        "fc2 leaf-tree stores the owner-combine wire bytes straight into symm staging",
        DEPLOYMENT,
        (TRAIN,),
        notes="The fc2 leaf tree stores the owner-combine wire bytes straight into symmetric staging instead of "
        "producing them and copying them in. Refusals are named rather than silent, and a corrupted-wire "
        "control is part of the battery.",
    ),
    Flag(
        "ISOEXEC_MOE_PIK_BF16_WIRE",
        "1",
        ("engine",),
        "MoE pik combine ships bf16 leaves on the wire, FP32 root (m_leaves==1 only)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bit-identical by construction plus a composition battery (tree, then top-k sum, then round). The root "
        "stays fp32 because the downstream fixed-order sum dtype follows its input, so a bf16 root would change "
        "it. A first-call losslessness self-check falls back permanently; fires only when world == G.",
    ),
    # Trainer-only MLA backward tuning and optional implementation switches.
    Flag(
        "ISOEXEC_GDN_PADDED_DECODE",
        "0",
        ("engine",),
        "run the GDN chunk kernel over the FULL C rows of every open-chunk buffer on a static cu "
        "grid, instead of ragged per-slot fills",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bitwise-neutral by the proven prefix-invariance property -- rows past the fill are stale buffer "
        "content and rows below it cannot see them -- so DEPLOYMENT, not FUNCTION. It costs roughly twice the "
        "GDN decode FLOPs and buys fully static decode shapes, the prerequisite for CUDA-graph capture, which "
        "is why it stays off rather than being deleted.",
    ),
    Flag(
        "ISOEXEC_GDN_SLOT_MAP_SIZE",
        "65536",
        ("engine",),
        "minimum GDN slot->buffer map reservation (512 KiB/layer at the default)",
        DEPLOYMENT,
        (TRAIN,),
        notes="vLLM finalizes this reservation against resolved KV block count plus one null-padding entry, "
        "after memory profiling and before capture. The registered map then keeps its address for every "
        "replay; a larger block space requires rebuilding the engine.",
    ),
    Flag(
        "ISOEXEC_GDN_GROUP_ALIAS",
        "1",
        ("engine",),
        "alias every GDN layer's metadata to the canonical mamba KV-cache group (one slot-id space)",
        DEPLOYMENT,
        (TRAIN,),
        notes="A correctness mechanism, not a knob. vLLM's KV-cache grouping heuristic can split the GDN layers "
        "across several mamba KV-cache groups with separate block tables; without the alias the layers disagree "
        "on state slot ids and decode folds the unmapped ones onto null row 0. 0 refuses a multi-group geometry "
        "loudly.",
    ),
    # Declare actor-read flags here; launcher exports alone do not reach every worker.
    Flag(
        "ISOEXEC_PIK_AR_CROSSOVER",
        "legacy",
        ("both",),
        "which one-shot/two-shot crossover pik's all-reduce uses: legacy | arch | calibrate",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Read on both sides and raises on an unknown value, so an unforwarded export would leave the two "
        "runtimes on different crossovers -- a different branch of the same tree, but the arm would be "
        "measuring a split.",
    ),
    # Engine-side and both-side flags read in the vLLM actor or its worker subprocess.
    Flag(
        "ISOEXEC_GDN_CPR_HOST_META",
        "1",
        ("engine",),
        "host-built GDN chunk metadata instead of the vendored prepare_chunk_indices/offsets",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Bitwise-neutral by assertion rather than assumption -- the two paths produce equal tensors -- so it "
        "can only move time.",
    ),
    Flag(
        "ISOEXEC_GDN_SEQ_META_CACHE",
        "1",
        ("both",),
        "memoize the per-forward GDN sequence metadata (lens / cu clone / chunked cu / state "
        "indices) instead of rebuilding it in every GDN layer",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="A pure host-side memo of pure functions of cu_seqlens, keyed on the object identity plus its version "
        "counter with a strong reference held, so an identity match implies the same unmutated object and there "
        "is no value comparison to get wrong. It issues no device op a rebuild would not, so it can only move "
        "time.",
    ),
    Flag(
        "ISOEXEC_PACKED_META_CACHE",
        "1",
        ("both",),
        "memoize the packed-sequence HOST READS (cu_seqlens .tolist()/.cpu()) per forward instead " "of once per layer",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Memoizes the host reads of a packed thd forward -- the cu_seqlens .tolist() / .cpu() calls that each "
        "land in pageable host memory and therefore block -- once per forward instead of once per layer. Host- "
        "side only: no device op changes.",
    ),
    Flag(
        "ISOEXEC_AUTOFUSE_FAST_CFG",
        "0",
        ("both",),
        "build a compiled region's inductor-config pin ONCE instead of on every call",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="A compiled region rebuilds its inductor-config pin on every call; this builds it once. Host-time only "
        "-- the pin content is unchanged, so the compiled code and its bits are unchanged.",
    ),
    Flag(
        "ISOEXEC_BWD_COMPILE",
        "0",
        ("trainer",),
        "compile admitted BACKWARD-only regions (ops/compile/bwd_compile.py); decision comes from a "
        "persisted ledger, never from a runtime benchmark",
        FUNCTION,
        (TRAIN,),
        notes="TRAIN-only by construction: every registered site is reachable exclusively from an "
        "autograd.Function.backward, so the engine never executes one and the forward gate is untouched. "
        "FUNCTION because a compiled region reassociates backward reductions -- legal, since the backward "
        "carries no bitwise contract, but it moves gradients and the gate compares two no_grad forwards, so it "
        "cannot see that. Admission is an fp64-oracle battery, not a bit check.",
    ),
    Flag(
        "ISOEXEC_BWD_COMPILE_LEDGER",
        "",
        ("trainer",),
        "path of the backward-region compile ledger (default $XDG_CACHE_HOME/skyrl-isoexec/bwd_compile.json, else ~/.cache/skyrl-isoexec/bwd_compile.json)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "ISOEXEC_BWD_COMPILE_ROLE",
        "reader",
        ("trainer",),
        "reader (default) consumes verdicts and can never write one; writer is the offline battery",
        DEPLOYMENT,
        (TRAIN,),
        notes="a production step must NEVER be a writer: a verdict written from a live step would "
        "make the pinned artifact a function of that step's shapes, which is exactly the "
        "run-to-run gradient drift the ledger exists to remove. record() raises in a reader.",
    ),
]


def actor_forwarding_list(channel: str | None = None) -> List[str]:
    """The flag names this table says a channel forwards, sorted and de-duplicated."""
    if channel is not None:
        return sorted({f.name for f in FLAGS if channel in f.forwarded_by})
    return sorted({f.name for f in FLAGS if f.name != "ISOEXEC"})


# Classify numerical environment reads and declare their required forwarding channels.

# Upstream switches read by the vendored kernels use the same audited default table.
FLAGS.extend(
    Flag(name, default, ("both",), selection, FUNCTION, (TRAIN, ENGINE))
    for name, default, selection in (
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8", "deterministic cuBLAS workspace configuration"),
        ("CUBLASLT_WORKSPACE_SIZE", "1", "deterministic cuBLASLt workspace size"),
        ("FLA_CI_ENV", "0", "FLA CI kernel configuration"),
        ("FLA_COMPILER_MODE", "0", "FLA compiler specialization mode"),
        ("FLA_DISABLE_BACKEND_DISPATCH", "0", "FLA backend dispatch"),
        ("FLA_TRIL_PRECISION", "ieee", "FLA triangular solve dot precision"),
        ("FLA_USE_CUDA_GRAPH", "0", "FLA autotuner CUDA graph timing"),
        ("FLA_USE_FAST_OPS", "0", "FLA fast op dispatch"),
        ("FLA_USE_TMA", "0", "FLA tensor memory access path"),
        ("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0", "GDN backward recomputation"),
        ("PIK_AUTOTUNE_M_BUCKET_FLOOR", "1", "pik row-count autotune bucket floor"),
        ("ISOEXEC_GDN_FUSED_CONV_BWD", "1", "GDN fused convolution backward"),
        ("USE_DEFAULT_FLA_NORM", "0", "FLA norm implementation"),
    )
)

NUMERICS_SWITCHES: Tuple[str, ...] = (
    "CUBLASLT_WORKSPACE_SIZE",
    "CUBLAS_WORKSPACE_CONFIG",
    "FLA_CI_ENV",
    "FLA_COMPILER_MODE",
    "FLA_DISABLE_BACKEND_DISPATCH",
    "FLA_TRIL_PRECISION",
    "FLA_USE_CUDA_GRAPH",
    "FLA_USE_FAST_OPS",
    "FLA_USE_TMA",
    "GDN_RECOMPUTE_SUPPRESS_LEVEL",
    "ISOEXEC",
    "ISOEXEC_BWD_COMPILE",
    "ISOEXEC_BWD_COMPILE_LEDGER",
    "ISOEXEC_BWD_COMPILE_ROLE",
    "ISOEXEC_FUSED_RESIDUAL_NORM",
    "ISOEXEC_FUSED_SAMPLER",
    "ISOEXEC_GDN",
    "ISOEXEC_GDN_ANALYTIC_CONV_BWD",
    "ISOEXEC_GDN_CHUNKED_PREFILL",
    "ISOEXEC_GDN_CONFIG_INDEX",
    "ISOEXEC_GDN_CONV_BWD_ROWS",
    "ISOEXEC_GDN_CPR_APC",
    "ISOEXEC_GDN_CPR_APC_MB",
    "ISOEXEC_GDN_CPR_FUSED_ROWS",
    "ISOEXEC_GDN_CPR_FUSED_SCATTER",
    "ISOEXEC_GDN_CPR_HOST_META",
    "ISOEXEC_GDN_CPR_MIN_PAGES",
    "ISOEXEC_GDN_CPR_SLEEP",
    "ISOEXEC_GDN_FLA_BACKWARD",
    "ISOEXEC_GDN_FUSED_CONV_BWD",
    "ISOEXEC_GDN_FUSED_SPLIT",
    "ISOEXEC_GDN_GROUP_ALIAS",
    "ISOEXEC_GDN_KERNEL",
    "ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE",
    "ISOEXEC_GDN_NATIVE_CONV",
    "ISOEXEC_GDN_NATIVE_KERNELS",
    "ISOEXEC_GDN_NATIVE_STATE",
    "ISOEXEC_GDN_PADDED_DECODE",
    "ISOEXEC_GDN_PIN_CONFIGS",
    "ISOEXEC_GDN_SEQ_META_CACHE",
    "ISOEXEC_GDN_SLOT_MAP_SIZE",
    "ISOEXEC_GQA_SM90_CUTLASS",
    "ISOEXEC_MLA_SM90_CUDA_HOME",
    "ISOEXEC_MLA_SM90_CUTLASS",
    "ISOEXEC_MLA_SM90_PROFILE",
    "ISOEXEC_MLP_FUSED_ACT",
    "ISOEXEC_MM_FWD_ONLY",
    "ISOEXEC_MM_FWD_ONLY_BMM",
    "ISOEXEC_MM_SM100",
    "ISOEXEC_MM_SM90_PDL",
    "ISOEXEC_MM_SM90_SEGTREE",
    "ISOEXEC_MODEL_PATH",
    "ISOEXEC_MOE_AFTER_FC2_GATHER",
    "ISOEXEC_MOE_AFTER_FC2_OWNER",
    "ISOEXEC_MOE_COMBINE_BLOCK_H",
    "ISOEXEC_MOE_COMBINE_SORT",
    "ISOEXEC_MOE_COMBINE_SORT_BLOCK_P",
    "ISOEXEC_MOE_COMBINE_SORT_MAX_WORK",
    "ISOEXEC_MOE_COMBINE_SORT_WARPS",
    "ISOEXEC_MOE_COMBINE_WARPS",
    "ISOEXEC_MOE_OC_ROWS32",
    "ISOEXEC_MOE_OC_WIRE_STAGE",
    "ISOEXEC_MOE_PACKED_EXPERTS",
    "ISOEXEC_MOE_PERMUTE_SORT",
    "ISOEXEC_MOE_PIK_BF16_WIRE",
    "ISOEXEC_MOE_PIK_OWNER_COMBINE",
    "ISOEXEC_MOE_ROUTER_O2",
    "ISOEXEC_NATIVE_NORM_MEMO",
    "ISOEXEC_PACKED_META_CACHE",
    "ISOEXEC_PIK",
    "ISOEXEC_PIK_AR_CROSSOVER",
    "ISOEXEC_PIK_AR_VEC",
    "ISOEXEC_PIK_FASTLAUNCH",
    "ISOEXEC_PIK_FUSED_BARRIER",
    "ISOEXEC_PIK_FUSED_OWNER_COMBINE",
    "ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS",
    "ISOEXEC_PIK_FUSED_ROOT_CAST",
    "ISOEXEC_PIK_LEAF_DTYPE",
    "ISOEXEC_PIK_LEAVES",
    "ISOEXEC_PIK_ONESHOT_MB",
    "ISOEXEC_PIK_OWNER_AR",
    "ISOEXEC_PIK_OWNER_ARNORM",
    "ISOEXEC_PIK_OWNER_LAMPORT",
    "ISOEXEC_PIK_OWNER_LAMPORT_BLOCKS",
    "ISOEXEC_PIK_OWNER_LAMPORT_KB",
    "ISOEXEC_PIK_OWNER_LAMPORT_ONESHOT_KB",
    "ISOEXEC_PIK_OWNER_LEAF_MIN_M",
    "ISOEXEC_PIK_OWNER_MAX_BLOCKS",
    "ISOEXEC_PIK_OWNER_NODE",
    "ISOEXEC_PIK_OWNER_ONESHOT_KB",
    "ISOEXEC_PIK_OWNER_PDL",
    "ISOEXEC_PIK_OWNER_PUSH",
    "ISOEXEC_PIK_OWNER_ROUTE",
    "ISOEXEC_PIK_P2P",
    "ISOEXEC_SHARD_SAMPLER",
    "ISOEXEC_TRAINER_BUCKET_PACK",
    "ISOEXEC_WGRAD_FUSION",
    "PIK_AUTOTUNE_M_BUCKET_FLOOR",
    "USE_DEFAULT_FLA_NORM",
    "VLLM_BATCH_INVARIANT",
)

SIDE_LOCAL_SWITCHES: Dict[str, str] = {
    "ISOEXEC_GDN_CPR_DECODE_TUNE": "engine-only execution choice (decode CTAs per head, measured per process); "
                                   "every split gives the same bits, so processes and ranks may differ",
    "ISOEXEC_VLLM_EXACT_SLEEP": "engine sleep level / host-backup policy; the bytes a forward reads after the "
                                "weight update are the same at level 1 and exact level 2, so processes may differ",
    "TORCH_EXTENSIONS_DIR": "a filesystem cache root; compiler inputs are classified separately",
    "FA_CLC": "read at import by vendored FlashAttention-4 utils.py for its forward's scheduler; no vendored "
              "backward path reads it",
    "FA_DISABLE_2CTA": "read at import by vendored FlashAttention-4 utils.py; the SM100 GQA backward takes its "
                       "2-CTA choice as an argument (disable_2cta=False) and never reads it",
    "QUACK_ARCH": "vendored QuACK's CPU-compilation arch override for the trainer-only SM100 GQA backward, which "
                  "is outside the cross-host contract; unset in deployments",
    "XDG_CACHE_HOME": "a user filesystem cache root; selects no numerical recipe",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "engine-only graph capture mechanism for PIECEWISE steps: captured segments "
                                    "replay the same kernels on the same operands, and every metadata-dependent "
                                    "op runs as an eager break, so it selects no numerical recipe",
    "ISOEXEC_PIK_TRAINER_P2P": "transport only (symmetric memory vs NCCL bytes); the tree and its bits do not "
                               "depend on it, and every rank of a TP group votes in one collective probe",
    "ISOEXEC_PIK_TRAINER_P2P_MAX_MB": "transport only (size crossover P2P vs NCCL owner reduce); same tree, "
                                      "same bits; the group uses the minimum over its ranks (probe)",
    "ISOEXEC_PIK_OVERLAP": "schedule only (which stream runs a reduce, how many token chunks a site has); "
                           "per-element leaves/tree/root are unchanged; agreed across the TP group",
    "ISOEXEC_PIK_OVERLAP_CHUNKS": "schedule only (token chunks of a pipelined row-parallel site); agreed across "
                                  "the TP group",
    "ISOEXEC_PIK_OVERLAP_MIN_M": "schedule only (rows from which a site is pipelined); agreed across the TP group",
    "ISOEXEC_PIK_OVERLAP_COMM_SMS": "schedule only (persistent GEMM CTA cap during overlap); tile arithmetic is "
                                    "independent of the CTA count",
    "CUDA_DEVICE_MAX_CONNECTIONS": "hardware work-queue count; pik/lane.py reads it only to decide whether "
                                   "comm/compute overlap can run (schedule only, agreed within the TP group); "
                                   "trainer and engine may differ",
    "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES": "torch's ordinal check, relaxed by pik only after a group "
                                                "probe proved distinct physical GPUs; transport only",
    "ISOEXEC_GQA_SM90_BUILD_DIR": "a filesystem cache location; compiler inputs are classified separately",
    "ISOEXEC_GQA_SM90_BUILD_VERBOSE": "compiler logging only; selects no kernel",
    "ISOEXEC_MLA_SM90_BUILD_DIR": "a filesystem cache location; compiler inputs are classified separately",
    "ISOEXEC_MLA_SM90_BUILD_VERBOSE": "compiler logging only; selects no kernel",
    "ISOEXEC_MLA_SM90_PARTIAL_BYTES": "scratch capacity for publishing canonical-tree nodes; preserves merge order",
    "ISOEXEC_MLA_SM100_PARTIAL_BYTES": "scratch capacity for publishing canonical-tree nodes; preserves merge order",
    "PATH": "build executable discovery tail after the explicitly selected CUDA toolkit bin directory",
    "VLLM_ISOEXEC_REGISTERED": "engine worker bootstrap marker set by explicit API registration; selects no numerical kernel",
    "VLLM_USE_V2_MODEL_RUNNER": "engine-only runner selection; the IsoExec engine runs only on Model Runner V2, while trainers use their own execution owner",
    "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR": (
        "an evidence directory: turns on the [ISOEXEC-MOE-EP] routing census printout; selects no kernel"
    ),
    "CUDA_VISIBLE_DEVICES": (
        "read only to LOG which devices a process sees when it joins a communicator (runtimes/vllm/nccl_channels); "
        "device visibility is per process by construction and selects no kernel"
    ),
    # Read only inside ops/gdn/shared/vendored_fla_bwd, FLA 0.5.2's own switches, unchanged by vendoring. That tree
    # is the trainer-only GDN backward; an engine never loads it, so trainer and engine cannot disagree through them.
    "FLA_CACHE_MODE": (
        "trainer-only GDN backward (vendored_fla_bwd): FLA's on-disk kernel-config cache; unset means 'disabled', "
        "which consults no config file, and the backward kernels whose candidates differ in bits are pinned"
    ),
    "FLA_CONFIG_DIR": "trainer-only GDN backward (vendored_fla_bwd): FLA config-cache directory, read only when FLA_CACHE_MODE is set",
    "FLA_GPU_NAME": "trainer-only GDN backward (vendored_fla_bwd): FLA config-cache GPU name, read only when FLA_CACHE_MODE is set",
    "FLA_CACHE_RESULTS": (
        "trainer-only GDN backward (vendored_fla_bwd): Triton's on-disk autotune-result cache for FLA's autotuners; "
        "the pinned backward kernels have one candidate, so it selects nothing for them"
    ),
    "FLA_DISABLE_TENSOR_CACHE": "trainer-only GDN backward (vendored_fla_bwd): memoizes chunk-index tensors by identity; selects no kernel",
    "FLA_TENSOR_CACHE_SIZE": "trainer-only GDN backward (vendored_fla_bwd): size of that index-tensor memo; selects no kernel",
    "CUDA_HOME": "vendored_fla_bwd's has_usable_nvcc TileLang probe, which nothing calls (backend dispatch is the identity)",
    "CUDA_PATH": "vendored_fla_bwd's has_usable_nvcc TileLang probe, which nothing calls (backend dispatch is the identity)",
    "TRITON_F32_DEFAULT": (
        "WRITTEN, not read, by vendored_fla_bwd/utils/_device.py, and only on pre-Ampere NVIDIA GPUs (no TF32), "
        "which IsoExec does not run on"
    ),
    "ISOEXEC_ROPE_SM90_MAX_POSITIONS": "rope_sm90 table capacity; a row's bits do not depend on it",
    "ISOEXEC_ROPE_SM100_MAX_POSITIONS": "rope_sm100 table capacity; a row's bits do not depend on it",
    "ISOEXEC_ROPE_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_ROPE_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_NORMS_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_NORMS_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_CONV_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_CONV_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_LOGPROBS_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_LOGPROBS_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_LOGPROBS_SHARD_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_LOGPROBS_SHARD_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_GDN_CPR_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_GDN_CPR_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_MOE_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_MOE_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_MM_BWD_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_MM_BWD_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_DIGEST_SM90_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_DIGEST_SM90_BUILD_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_DIGEST_SM100_BUILD_DIR": "a filesystem cache location",
    "ISOEXEC_DIGEST_SM100_BUILD_VERBOSE": "a logging level; selects no kernel",
    "NCCL_ALGO": "reported inside the nccl sub-map (not compared across sides)",
    "NCCL_MAX_NCHANNELS": "reported inside the nccl sub-map (not compared across sides)",
    "NCCL_NVLS_ENABLE": (
        "engine NCCL transport choice (runtimes/vllm/vllm_plugin.py defaults it off for memory); NCCL moves bytes "
        "only, every numerics-bearing reduction is pik's fixed tree, so it selects no reduction order"
    ),
    "PYTORCH_CUDA_ALLOC_CONF": "caching-allocator segment policy, checked only to respect a user setting; selects no kernel",
    "PYTORCH_ALLOC_CONF": "caching-allocator segment policy, checked only to respect a user setting; selects no kernel",
    "NCCL_MIN_NCHANNELS": "reported inside the nccl sub-map (not compared across sides)",
    "PIK_CACHE": "a filesystem cache location",
    "PIK_VERBOSE": "a logging level; selects no kernel",
    "ISOEXEC_AUTOFUSE_FAST_CFG": "how the pinned Inductor config is entered per compiled-region call (ops/compile/cfgpin.py); "
                                 "the fast pin is proven equal to the slow one before use, so it selects no kernel",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "vLLM's process model; selects no kernel",
}


# Names the environment builders assign directly rather than through the generic forwarding
# loop; ``actor_forwarding_tuple``, the loop's iterable, excludes them.
_DIRECT_FORWARDS = frozenset(
    {
        "ISOEXEC",
        "FLA_TILELANG",
        "NCCL_ALGO",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
    }
)


def actor_forwarding_tuple(channel: str) -> List[str]:
    """Flags forwarded by a channel's generic loop; direct assignments are excluded."""
    return sorted(set(actor_forwarding_list(channel)) - _DIRECT_FORWARDS)


def malformed_sides() -> List[Flag]:
    """Flags whose `sides` uses a token outside `SIDE_VOCABULARY`."""
    return [f for f in FLAGS if f.has_malformed_sides]


def _validate_sides() -> None:
    """Refuse at import if any declaration uses a side token no reader matches."""
    bad = malformed_sides()
    if bad:
        detail = ", ".join(f"{f.name}={f.sides}" for f in bad)
        raise ValueError(
            f"Flag.sides must be a subset of {sorted(SIDE_VOCABULARY)}; got {detail}. "
            f"A token outside the vocabulary matches no reader, so every check written against "
            f"`sides` silently passes for that flag."
        )


_validate_sides()


def get(name: str) -> Flag:
    for f in FLAGS:
        if f.name == name:
            return f
    raise KeyError(name)
