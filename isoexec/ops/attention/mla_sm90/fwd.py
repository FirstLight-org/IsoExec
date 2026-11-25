"""Forward of the deterministic absorbed-MLA op on SM90 (derived from DeepSeek FlashMLA, MIT)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import torch

from isoexec.ops.build_cache import cutlass_revision, require_pinned_cutlass, sm90_build_dir

from isoexec.ops.attention.mla_sm90.spec import PAGE

__all__ = ["MlaSm90Plan", "mla_sm90_fwd", "mla_sm90_plan", "PROFILES", "SEG_TABLE",
           "EXEC_OVERRIDE_VARIANTS", "profile_digest", "seg_table_digest", "seg_tokens_for"]

_SRC = Path(__file__).resolve().parent / "csrc" / "fwd"
_SOURCES = ("bindings.cpp", "leaf_bf16.cu", "plan.cu", "combine.cu")
_BUILD_LOCK = Lock()

HEAD_DIM_K = 576
HEAD_DIM_V = 512
M_BLOCK = 64
PARTIAL_WIDTH = HEAD_DIM_V + 4  # O[512], L, M, 2 pad (16-byte aligned slots)
ITEM_OVERHEAD_BLOCKS = 5  # FlashMLA's fixed_overhead_num_blocks: an item's Q copy, setup and publish, in KV blocks
FOLD_OVERHEAD_BLOCKS = 1  # an in-place segment boundary (L reduction, fold, K reload), in KV blocks
GRANULARITY = {"auto": 0, "single": 2, "whole": 3, "pairs": 4, "mixed": 5}  # csrc/fwd/params.h Granularity
_PLAN_SMEM_INTS = 160 * 1024 // 4  # dynamic shared memory the plan kernel may use (of SM90's 232 KB)
_SNAKE_ITEMS_PER_WORKER = 2  # csrc/fwd/plan.cu: above this the schedule deals whole trees out snake-wise and splits none
_MAX_WORKER_WAVES = 4  # leaf workers are capped at this many waves of SMs (each carries a fold stack)

# Execution tuning may change work partitioning, never the reduction tree or output bits.
EXEC_OVERRIDE_VARIANTS = [
    {"granularity": "single"},
    {"granularity": "whole"},
    {"granularity": "pairs"},
    {"granularity": "mixed"},
    {"granularity": "single", "num_parts": 1000},
    {"granularity": "pairs", "reverse": True},
    {"granularity": "mixed", "num_parts": 3},
    {"num_parts": 1},
    {"num_parts": 2},
    {"num_parts": 7},
    {"num_parts": 64},
    {"num_parts": 1000},
    {"reverse": True},
    {"reverse": True, "num_parts": 3},
    {"item_overhead": 0, "fold_overhead": 0},
    {"item_overhead": 100000},
    {"publish_rows": True},
    {"combine_single_group": True},
    {"combine_config": 1},
    {"combine_config": 3},
]
_GRAN_TARGET = 1
_NO_TARGET = 2**31 - 1
MAX_TREE_LEAVES = 1024
EMPTY_LSE = float(torch.tensor(-1e30, dtype=torch.float32) / torch.tensor(1.4426950408889634, dtype=torch.float32))

# Choose a fixed segment size from the joint prefill/decode benchmark grid.
SEG_TABLE = {
    (5, 8192): 2048, (5, 10240): 2560, (5, 32768): 2048, (5, 131072): 2048,
    (10, 8192): 2048, (10, 10240): 2560, (10, 32768): 2048, (10, 131072): 2048,
    (20, 8192): 2048, (20, 10240): 2560, (20, 32768): 2048, (20, 131072): 2048,
}


def seg_tokens_for(h_q: int, max_seqlen_k: int) -> int:
    """SEG for a deployment: the smallest max-context class that covers max_seqlen_k (beyond the table: largest)."""
    classes = sorted({c for (h, c) in SEG_TABLE if h == int(h_q)})
    if not classes:
        raise KeyError(f"no SEG table entry for h_q={h_q}")
    cls = next((c for c in classes if c >= int(max_seqlen_k)), classes[-1])
    return SEG_TABLE[(int(h_q), cls)]


def seg_table_digest() -> str:
    """Digest of the SEG table, compared at bind time."""
    return hashlib.sha256(repr(sorted(SEG_TABLE.items())).encode()).hexdigest()[:16]


_COMMON_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-D_USE_MATH_DEFINES",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
)

# Keep the selected numerical profile consistent across every forward kernel.
PROFILES = {
    "fast": ("--fmad=false", "--ftz=true", "--prec-div=false"),
    "strict": ("--fmad=false", "--ftz=false"),
}
DEFAULT_PROFILE = "fast"


def _profile_name(profile: str | None) -> str:
    name = profile or os.environ.get("ISOEXEC_MLA_SM90_PROFILE", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(f"unknown mla_sm90 profile {name!r}; expected one of {sorted(PROFILES)}")
    return name


def profile_digest(profile: str | None = None) -> str:
    """Digest of the forward identity inputs: flag profile and kernel sources."""
    name = _profile_name(profile)
    h = hashlib.sha256()
    h.update(" ".join(_COMMON_CUDA_FLAGS + PROFILES[name]).encode())
    for path in sorted(_SRC.iterdir()):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(None)
def extension(profile: str | None = None):
    """Build (once) and load the forward extension for a flag profile."""
    name = _profile_name(profile)
    import torch.utils.cpp_extension as cpp

    # The same toolkit as every other IsoExec CUDA source (CUDA_HOME, which setup pins) unless overridden.
    cuda_home = os.environ.get("ISOEXEC_MLA_SM90_CUDA_HOME") or cpp.CUDA_HOME
    if not cuda_home:
        raise RuntimeError("mla_sm90 found no CUDA toolkit; set CUDA_HOME or ISOEXEC_MLA_SM90_CUDA_HOME")
    cutlass_inc = require_pinned_cutlass("mla", "SM90")
    # Headers select generated code, so a different CUTLASS must never reuse a cached build.
    identity = f"{cutlass_revision(cutlass_inc) or 'unrecorded'}@{cutlass_inc.resolve()}"
    cutlass_tag = hashlib.sha256(identity.encode()).hexdigest()[:8]
    tag = (f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}-{name}-"
           f"{profile_digest(name)}-c{cutlass_tag}")
    build = sm90_build_dir("ISOEXEC_MLA_SM90_BUILD_DIR", "mla-sm90", tag, _SRC, append_tag_to_override=True)
    build.mkdir(parents=True, exist_ok=True)
    include = [str(_SRC), str(cutlass_inc), str(cutlass_inc.parent / "tools" / "util" / "include"),
               str(Path(cuda_home) / "targets" / "x86_64-linux" / "include" / "cccl"),
               str(Path(cuda_home) / "include" / "cccl")]
    env = {"TORCH_CUDA_ARCH_LIST": "9.0a", "CUDA_HOME": cuda_home,
           "PATH": f"{cuda_home}/bin:{os.environ.get('PATH', '')}"}
    with _BUILD_LOCK, patch.dict(os.environ, env), patch.object(cpp, "CUDA_HOME", cuda_home):
        return cpp.load(
            name=f"isoexec_mla_sm90_fwd_{name}",
            sources=[str(_SRC / f) for f in _SOURCES],
            build_directory=str(build),
            extra_include_paths=[p for p in include if Path(p).is_dir()],
            extra_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES"],
            extra_ldflags=["-lcuda"],
            extra_cuda_cflags=list(_COMMON_CUDA_FLAGS + PROFILES[name]) + ["--threads", "16"],
            verbose=os.environ.get("ISOEXEC_MLA_SM90_BUILD_VERBOSE") == "1",
        )


def _num_sms() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


@dataclasses.dataclass
class MlaSm90Plan:
    """Per-step plan: one work-item list per leaf worker, plus the combine rows of the trees that were split."""

    items: torch.Tensor  # int32 [item_cap, 8]
    heads: torch.Tensor  # int32 [workers]: first item of each worker's list
    combine_rows: torch.Tensor  # int32 [combine_cap, 4]
    counts: torch.Tensor  # int32 [8]: items, slots, combine rows, overflow, max n, total cost, cost target
    scratch: torch.Tensor  # int32 [9*mb_cap + 3*workers]: the plan kernel's records, cost order and worker heap
    h_q: int
    seg_tokens: int
    num_seqs: int
    max_seqlen_q: int
    slot_cap: int
    max_n: int
    workers: int  # leaf workers launched
    exact: bool
    combine_config: int | None = None  # None: chosen from capacities
    debug_flags: int = 0  # execution-only debug switches (publish_rows, combine_single_group)
    may_empty: bool = False  # rows without keys may occur: outputs are pre-filled with the sentinel

    @property
    def item_cap(self) -> int:
        return self.items.shape[0]

    @property
    def combine_cap(self) -> int:
        return self.combine_rows.shape[0]

    def check(self) -> None:
        """Synchronize and assert that the capacities held (debug; not for captured regions)."""
        c = self.counts.tolist()
        if c[3]:
            raise RuntimeError(f"mla_sm90 plan overflow: counts={c}, item_cap={self.item_cap}, "
                               f"slot_cap={self.slot_cap}, combine_cap={self.combine_cap}")


def _exact_sizes(cu_seqlens_q, seqused_k, seg, h_q):
    """Worst-case (one item per segment) item, slot and combine-row counts from host copies of the lengths."""
    import numpy as np

    # One device-to-host copy for both length arrays (numpy from here: the arrays are small and torch's per-op
    # overhead would dominate the plan)
    both = torch.cat((cu_seqlens_q.reshape(-1).to(torch.int64), seqused_k.reshape(-1).to(torch.int64))).cpu().numpy()
    b = int(seqused_k.shape[0])
    cu, s = both[:b + 1], both[b + 1:]
    q_len = cu[1:] - cu[:-1]
    rows = q_len * h_q
    may_empty = bool((np.minimum(np.maximum(q_len - s, 0), q_len) > 0).any())
    nmb = (rows + M_BLOCK - 1) // M_BLOCK
    total_mb = int(nmb.sum())
    if total_mb == 0:
        return 0, 0, 0, 0, 0, 0, 1, may_empty
    req = np.repeat(np.arange(b), nmb)
    mb = np.arange(total_mb) - (np.cumsum(nmb) - nmb)[req]
    fr = mb * M_BLOCK
    nrows = np.minimum(rows[req] - fr, M_BLOCK)
    base = s[req] - q_len[req]
    p0 = base + fr // h_q
    p1 = base + (fr + nrows - 1) // h_q
    live = p1 >= 0
    nmax = np.where(live, p1 // seg + 1, 0)
    split = live & (nmax > 1)
    items = int(nmax[live].sum())
    slots = int((nrows * nmax)[split].sum())
    lo = np.maximum(np.maximum((seg - base) * h_q, 0), fr)
    rows_split = np.maximum(fr + nrows - lo, 0)
    combine = int(rows_split[split].sum())
    mandatory = split & ~((p0 >= 0) & (p0 // seg == p1 // seg))
    max_n = int(nmax[live].max()) if bool(live.any()) else 1
    return (total_mb, items, slots, combine, int((nrows * nmax)[mandatory].sum()),
            int(rows_split[mandatory].sum()), max_n, may_empty)


def _slot_budget() -> int:
    """Largest partial buffer (bytes) a plan may use for published tree nodes; the schedule splits within it."""
    return int(os.environ.get("ISOEXEC_MLA_SM90_PARTIAL_BYTES", 1 << 30)) // (PARTIAL_WIDTH * 4)


def mla_sm90_plan(cu_seqlens_q, seqused_k, seg_tokens, h_q, max_seqlen_q, capacity=None,
                  exec_override=None) -> MlaSm90Plan:
    """Build deterministic MLA work and combine plans from packed sequence lengths."""
    seg = int(seg_tokens)
    if seg <= 0 or seg % 128:
        raise ValueError("seg_tokens must be a positive multiple of 128")
    h_q = int(h_q)
    max_seqlen_q = int(max_seqlen_q)
    ov = dict(exec_override or {})
    dev = cu_seqlens_q.device
    b = int(seqused_k.shape[0])
    gran_name = ov.pop("granularity", "auto")
    if gran_name not in GRANULARITY:
        raise ValueError(f"exec_override granularity must be one of {sorted(GRANULARITY)}, got {gran_name!r}")
    granularity = GRANULARITY[gran_name]
    reverse = bool(ov.pop("reverse", False))
    item_overhead = int(ov.pop("item_overhead", ITEM_OVERHEAD_BLOCKS))
    fold_overhead = int(ov.pop("fold_overhead", FOLD_OVERHEAD_BLOCKS))
    workers_override = ov.pop("num_parts", None)
    combine_config = ov.pop("combine_config", None)
    debug_flags = (1 if ov.pop("publish_rows", False) else 0) | ((4 << 16) if ov.pop("combine_single_group", False) else 0)
    if ov:
        raise ValueError(f"unknown exec_override keys {sorted(ov)}")
    workers = int(workers_override) if workers_override is not None else _num_sms()
    if workers <= 0:
        raise ValueError("num_parts must be positive")
    # Workers cost a fold stack (2^depth records of 64 rows each), and nothing is gained past a few waves
    workers = min(workers, _MAX_WORKER_WAVES * _num_sms())
    budget = _slot_budget()
    if capacity is None:
        mblocks, items, slots, combine, mand_slots, mand_comb, max_n, may_empty = _exact_sizes(
            cu_seqlens_q, seqused_k, seg, h_q)
        exact = True
        workers = workers if workers_override is not None else max(1, min(workers, items))
        if granularity == GRANULARITY["auto"] and mblocks >= _SNAKE_ITEMS_PER_WORKER * workers:
            # The plan kernel splits no tree when whole trees already saturate the workers (csrc/fwd/plan.cu)
            slots, combine = mand_slots, mand_comb
        else:
            slots = min(slots, max(budget, mand_slots))
            combine = min(combine, max(slots // 2, mand_comb))
    else:
        cap = {"max_seqlen_k": int(capacity)} if isinstance(capacity, int) else dict(capacity)
        max_k = int(cap["max_seqlen_k"])
        tokens = int(cap.get("num_tokens", b * max_seqlen_q))
        rows = tokens * h_q
        mblocks = min(b * math.ceil(max_seqlen_q * h_q / M_BLOCK), math.ceil(rows / M_BLOCK) + b)
        n_bound = max(1, math.ceil(max_k / seg))
        items = int(cap.get("items", mblocks * n_bound))
        # A decode step's row block is one token (no m_block padding); longer queries pad up to 64 rows per block
        slots = int(cap.get("slots", (rows + (M_BLOCK * b if max_seqlen_q > 1 else 0)) * n_bound if n_bound > 1 else 0))
        # Row blocks that straddle a segment boundary must be split, and only queries longer than one token can:
        # a decode step's slots are the schedule's choice and the budget may cut them.
        if max_seqlen_q == 1:
            slots = min(slots, budget)
        may_empty = bool(cap.get("null_lanes", True))  # rows with no keys (padded lanes) get the sentinel
        combine = int(cap.get("combine_rows", rows if n_bound > 1 else 0))
        max_n = n_bound
        exact = False
        workers = int(cap.get("workers", workers))

    if max_n > MAX_TREE_LEAVES and combine > 0:
        raise ValueError(f"{max_n} segments per row exceed the combine kernel's {MAX_TREE_LEAVES}; raise seg_tokens")

    plan = MlaSm90Plan(
        items=torch.zeros(max(items, 1), 8, dtype=torch.int32, device=dev),
        heads=torch.zeros(workers, dtype=torch.int32, device=dev),
        combine_rows=torch.zeros(combine, 4, dtype=torch.int32, device=dev),
        counts=torch.zeros(8, dtype=torch.int32, device=dev),
        scratch=torch.zeros(9 * max(mblocks, 1) + 3 * workers, dtype=torch.int32, device=dev),
        h_q=h_q, seg_tokens=seg, num_seqs=b, max_seqlen_q=max_seqlen_q, slot_cap=slots, max_n=max_n,
        workers=workers, exact=exact,
        combine_config=combine_config, debug_flags=debug_flags, may_empty=may_empty,
    )
    if b > 0:
        # Shared memory for the plan kernel: the staged lengths and (if they fit) the row-block records
        if 2 * b + 1 > _PLAN_SMEM_INTS:
            raise ValueError(f"{b} requests exceed the plan kernel's shared-memory staging ({_PLAN_SMEM_INTS} ints)")
        smem_ints = min(2 * b + 1 + 9 * max(mblocks, 1) + 3 * workers, _PLAN_SMEM_INTS)
        cost_cap = (max_n * seg + PAGE - 1) // PAGE + item_overhead + max_n * fold_overhead
        extension().plan(_i32(cu_seqlens_q), _i32(seqused_k), plan.items, plan.heads, plan.combine_rows, plan.counts,
                         plan.scratch, h_q, seg, workers, granularity, 0, item_overhead, fold_overhead, slots,
                         max(mblocks, 1), cost_cap, reverse, smem_ints)
    return plan


def _combine_config(entries: int, max_n: int) -> int:
    """Combine worker shape from host capacities (execution only; every shape evaluates the same tree)."""
    if entries <= 16 and max_n >= 16:
        return 1
    if entries >= 1024 and max_n <= 64:
        return 3
    return 2


def _i32(t: torch.Tensor) -> torch.Tensor:
    return t if (t.dtype == torch.int32 and t.is_contiguous()) else t.to(torch.int32).contiguous()


def _aligned(t: torch.Tensor) -> torch.Tensor:
    if t.data_ptr() % 16 == 0 and t.is_contiguous():
        return t
    out = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    off = (-out.data_ptr()) % 16 // t.element_size()
    view = out[off: off + t.numel()].view(t.shape)
    view.copy_(t)
    return view


def mla_sm90_fwd(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg_tokens, max_seqlen_q,
                 plan=None, out=None, exec_override=None):
    """Deterministic absorbed-MLA forward."""
    if q.dim() != 3 or q.shape[-1] != HEAD_DIM_K or q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bf16 [T, h_q, {HEAD_DIM_K}], got {tuple(q.shape)} {q.dtype}")
    if kv_cache.dim() != 4 or tuple(kv_cache.shape[1:]) != (PAGE, 1, HEAD_DIM_K) or kv_cache.dtype != torch.bfloat16:
        raise ValueError(f"kv_cache must be bf16 [num_blocks, {PAGE}, 1, {HEAD_DIM_K}], got {tuple(kv_cache.shape)}")
    t, h_q = int(q.shape[0]), int(q.shape[1])
    seg = int(seg_tokens)
    if out is None:
        o = torch.empty(t, h_q, HEAD_DIM_V, dtype=torch.bfloat16, device=q.device)
        lse = torch.empty(t, h_q, dtype=torch.float32, device=q.device)
    else:
        o, lse = out
        if tuple(o.shape) != (t, h_q, HEAD_DIM_V) or o.dtype != torch.bfloat16 or not o.is_contiguous():
            raise ValueError("out[0] must be contiguous bf16 [T, h_q, 512]")
        if tuple(lse.shape) != (t, h_q) or lse.dtype != torch.float32 or not lse.is_contiguous():
            raise ValueError("out[1] must be contiguous float32 [T, h_q]")
    if t == 0 or int(seqused_k.shape[0]) == 0:
        return o, lse
    if exec_override is not None or plan is None:
        plan = mla_sm90_plan(cu_seqlens_q, seqused_k, seg, h_q, max_seqlen_q, exec_override=exec_override)
    if plan.h_q != h_q or plan.seg_tokens != seg or plan.num_seqs != int(seqused_k.shape[0]):
        raise ValueError(f"plan (h_q={plan.h_q}, seg={plan.seg_tokens}, B={plan.num_seqs}) does not match the call "
                         f"(h_q={h_q}, seg={seg}, B={int(seqused_k.shape[0])})")

    return _run(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg, plan, o, lse)


def _run(q, kv_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale, seg, plan, o, lse):
    h_q = int(q.shape[1])
    q = _aligned(q)
    if o.data_ptr() % 16:
        raise ValueError("out tensor must be 16-byte aligned")
    partials = torch.empty(max(plan.slot_cap, M_BLOCK), PARTIAL_WIDTH, dtype=torch.float32, device=q.device)
    # Reserve scratch for pending tree nodes and their per-thread statistics.
    stack = torch.empty(plan.workers, max(1, (plan.max_n - 1).bit_length()), M_BLOCK + 2,
                        PARTIAL_WIDTH, dtype=torch.float32, device=q.device)
    if plan.may_empty:
        # Rows without keys are published by no kernel: out = 0 and lse = the empty-state sentinel. Real rows are
        # overwritten by the kernels.
        o.zero_()
        lse.fill_(EMPTY_LSE)
    combine_entries = plan.combine_cap
    combine_config = plan.combine_config if plan.combine_config is not None else _combine_config(combine_entries, plan.max_n)
    extension().fwd(q, kv_cache, _i32(block_table), _i32(cu_seqlens_q), _i32(seqused_k),
                    plan.items, plan.heads, plan.counts, plan.combine_rows, o, lse, partials, _aligned(stack),
                    float(softmax_scale), h_q, seg, plan.workers, combine_entries, plan.max_n,
                    combine_config, plan.debug_flags, int(q.shape[0]) == plan.num_seqs * plan.max_seqlen_q)
    return o, lse


