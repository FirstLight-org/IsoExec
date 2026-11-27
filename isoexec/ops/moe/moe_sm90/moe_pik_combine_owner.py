"""Owner-computes MoE combination: each token owner folds peer partials with the fixed tree and ascending-expert fp32 sum.
The unified path seeds the routed sum from its first term, preserves the shared branch's bf16 root/gate rounds,
and narrows routed + fp32(shared) once; symmetric-memory exchange is stream-ordered and graph-capturable.
Fused barriers and direct wire staging require their admission checks; unsupported shapes use the eager composition."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys

import torch
import torch.distributed as dist

from isoexec.core.paths import pik_cache_dir

# The tree over `world` peers is pik's balanced tree (left = lower rank), emitted straight-line so the
# evaluation order is pinned and matches pik.codegen / the landed tree_all_reduce.

_CACHE_DIR = pik_cache_dir() / "owner_combine"


def _ensure_canonical_pik():
    """Load PIK under its single supported runtime identity before importing a child module."""
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    return ensure_pik()


def _load(name: str, src: str):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)  # atomic: concurrent ranks must not read a half-written file
    spec = importlib.util.spec_from_file_location(f"owner_combine.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.kernel


def _tree_src(world: int, in_bf16: bool) -> str:
    """Straight-line balanced tree over in0..in{world-1} at scalar offset `base`, into t0 (fp32).
    Post-order, left = lower rank -- identical order to pik.codegen._emit / plan.combine_order."""
    cast = ".to(tl.float32)" if in_bf16 else ""
    leaf = "        t{d} = tl.load(in{lo} + base, mask=hmask, other=0.0)" + cast + "\n"

    def emit(lo, hi, depth, out):
        if hi - lo == 1:
            out.append(leaf.format(d=depth, lo=lo))
            return
        mid = (lo + hi) // 2
        emit(lo, mid, depth, out)
        emit(mid, hi, depth + 1, out)
        out.append(f"        t{depth} = t{depth} + t{depth + 1}\n")

    body: list[str] = []
    emit(0, world, 0, body)
    return "".join(body)


_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, out_ptr, T, H, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        base = row * H + hoff
{tree}        acc = acc + t0
    tl.store(out_ptr + t * H + hoff, acc.to(tl.bfloat16), mask=hmask)
"""


# Push each owner's bf16 rows directly to peer output buffers, then barrier before copy-out.
# The tree, expert sum and final round match the local-store/all-gather form.


def _oc_core(world: int, k: int, in_bf16: bool) -> str:
    """Generate the float arithmetic shared verbatim by separate-barrier and fused-barrier push templates.
    Offsets, peer tree, expert sum, rounding and stores are identical; k keys the generated constexpr variant."""
    stores = "".join(f"    tl.store(out{i} + goff, res, mask=hmask)\n" for i in range(world))
    return f"""    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        base = row * H + hoff
{_tree_src(world, in_bf16)}        acc = acc + t0
    res = acc.to(tl.bfloat16)
    goff = (out_base + t) * H + hoff
{stores}"""


_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, {outs}, out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr):
{core}"""


@functools.cache
def _owner_kernel(world: int, k: int, in_bf16: bool):
    ins = ", ".join(f"in{i}" for i in range(world))
    src = _TMPL.format(ins=ins, tree=_tree_src(world, in_bf16))
    tag = f"_owner_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, src)


def _push_src(world: int, k: int, in_bf16: bool) -> str:
    ins = ", ".join(f"in{i}" for i in range(world))
    outs = ", ".join(f"out{i}" for i in range(world))
    return _PUSH_TMPL.format(ins=ins, outs=outs, core=_oc_core(world, k, in_bf16))


@functools.cache
def _owner_push_kernel(world: int, k: int, in_bf16: bool):
    tag = f"_owner_push_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, _push_src(world, k, in_bf16))


def _named_tree_src(world: int, in_bf16: bool, *, inp: str, tmp: str, base: str, indent: str) -> str:
    """The same balanced peer tree as :func:`_tree_src`, with disjoint SSA names, so one program can
    carry both the routed and the shared tree. Both use the same recursive emitter, left = lower rank.
    """
    cast = ".to(tl.float32)" if in_bf16 else ""
    leaf = f"{indent}{tmp}{{d}} = tl.load({inp}{{lo}} + {base}, mask=hmask, other=0.0){cast}\n"

    def emit(lo, hi, depth, out):
        if hi - lo == 1:
            out.append(leaf.format(d=depth, lo=lo))
            return
        mid = (lo + hi) // 2
        emit(lo, mid, depth, out)
        emit(mid, hi, depth + 1, out)
        out.append(f"{indent}{tmp}{depth} = {tmp}{depth} + {tmp}{depth + 1}\n")

    body: list[str] = []
    emit(0, world, 0, body)
    return "".join(body)


# Reuse pik's release/acquire barrier emitter with flat flag index pid0*NHB+pid1.
# Each 2-D block needs its own epoch slot and NB >= s*NHB.
# Blocks publish to every peer before waiting; monotone epochs survive block retirement.
# All ranks must launch the same grid.
_FUSED_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, {outs}, {fls}, self_fl, ep_ptr,
           my_slot0, my_slot1, wbase1, NB,
           out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr,
           NHB: tl.constexpr, WORLD: tl.constexpr):
    pid = tl.program_id(0) * NHB + tl.program_id(1)
    one = tl.arange(0, 1)
    lane = tl.arange(0, WORLD)

{lead}    tl.debug_barrier()   # every peer has arrived before any thread reads peer memory

{core}{trail}"""


def _fused_push_src(world: int, k: int, in_bf16: bool) -> str:
    _ensure_canonical_pik()
    import pik.codegen as CG  # type: ignore

    ins = ", ".join(f"in{i}" for i in range(world))
    outs = ", ".join(f"out{i}" for i in range(world))
    fls = ", ".join(f"fl{i}" for i in range(world))
    lead = CG._fused_barrier(0, world, "0", "pid")
    trail = "\n    tl.debug_barrier()   # this block's pushes precede its release\n" + CG._fused_barrier(
        1, world, "wbase1", "NB + pid"
    )
    return _FUSED_PUSH_TMPL.format(
        ins=ins, outs=outs, fls=fls, lead=lead, core=_oc_core(world, k, in_bf16), trail=trail
    )


@functools.cache
def _owner_push_fused_kernel(world: int, k: int, in_bf16: bool):
    tag = f"_owner_push_fused_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, _fused_push_src(world, k, in_bf16))


# Staging: reuse pik's symmetric-memory double-buffered pool -- stage this rank's partial, barrier,
# read peers over NVLink. Falls back to NCCL all_gather when symm-mem is unavailable.


def _fence(device) -> None:
    """Order this stream after pik's comm lanes before touching a shared symmetric pool."""
    from pik.lane import fence  # type: ignore

    fence(torch.device(device))


def _sym_pool(n, device, group, dtype):
    _ensure_canonical_pik()
    from pik.allreduce import (
        _sym_pool as pik_sym_pool,  # type: ignore
    )

    _fence(device)
    return pik_sym_pool(n, device, group, dtype)


# Keep bf16 output separate from input staging to prevent push/read aliasing.
# Double-buffer phases are reusable at N+2 after peers complete their reads of phase N.
_OUT_SYM: dict = {}

# Shared and routed fc2 leaves need separate double buffers, even with identical dtypes.
# Key by group membership so equal-sized groups cannot alias.
_SHARED_IN_SYM: dict = {}


def _out_sym_pool(n, device, group):
    _ensure_canonical_pik()
    from pik.allreduce import _SymPool, _get_sym_pool  # type: ignore

    _fence(device)
    key = (device, _group_key(group))  # by membership: equal-sized groups must not share a pool
    return _get_sym_pool(_OUT_SYM, key, n, device, group, torch.bfloat16, _SymPool)


def _shared_in_sym_pool(n, device, group, dtype):
    _ensure_canonical_pik()
    from pik.allreduce import _SymPool, _get_sym_pool  # type: ignore

    _fence(device)
    key = (device, _group_key(group), dtype)
    return _get_sym_pool(_SHARED_IN_SYM, key, n, device, group, dtype, _SymPool)


def _p2p_ok(group) -> bool:
    """pik's own answer (``allreduce.p2p_available``): under the trainer's physical policy that is the group's
    probe, so a group is never treated as P2P-capable without one."""
    _ensure_canonical_pik()
    from pik.allreduce import p2p_available  # type: ignore

    return p2p_available(group)


# FC2 may write directly into current-phase staging; pointer equality then skips the exchange copy.
# Until owner combine records the group, refuse the sink and count the reason.
_WIRE_ENV = "ISOEXEC_MOE_OC_WIRE_STAGE"

# ``None`` is a VALID group -- torch.distributed's default process group -- so it cannot serve as the
# "not recorded yet" sentinel or the sink would refuse forever on those configurations. The sentinel is
# a private object no caller can supply, keeping "recorded" distinct from "recorded as the default".
_WIRE_UNRECORDED = object()
_WIRE_GROUP: dict = {"group": _WIRE_UNRECORDED, "device": None}
_WIRE_COUNTS = {"grants": 0, "staged_hits": 0, "copy_falls": 0, "refusals": 0, "promotes": 0}

# Named refusal accounting. The sink's rank-local refusals are not equivalent -- "no combine has run
# yet" is expected once per process, "world != leaf count" is a configuration statement, and "the pool
# would grow under capture" is a warmup bug -- so each names itself in the counters.
_WIRE_REFUSALS: dict = {}
_WIRE_REASONS = {
    "group_not_recorded": "no owner combine has run in this process yet, so the sink does not know "
    "which process group (hence which symmetric pool) the fc2 partial will cross. Expected exactly "
    "once, at the first MoE layer of the first forward; if it repeats, the owner combine is not "
    "running at all and the wire-stage lever is INERT",
    "device_mismatch": "the recorded owner-combine device is not the device this fc2 ran on",
    "no_symm_p2p": "symmetric memory / P2P is unavailable on the recorded group (world<2, or "
    "torch.distributed._symmetric_memory did not import)",
    "bf16_wire_off": "ISOEXEC_MOE_PIK_BF16_WIRE=0, so the combine's wire is fp32 and there is "
    "no bf16 buffer for the leaftree to store into",
    "world_ne_leaves": "world != the pik plan's num_leaves, so this combine ships an fp32 wire "
    "(the bf16 wire is admissible only at world == G)",
    "pool_growth_under_capture": "the symmetric staging pool would have to GROW, and this call is "
    "inside a CUDA-graph capture where a symmetric allocation is a hang rather than an error. Run "
    "every shape once eagerly before capture",
    "exception": "the sink raised; see the traceback printed once above",
}


def wire_stage_enabled() -> bool:
    return os.environ.get(_WIRE_ENV, "1") == "1"


def wire_stage_counts() -> dict:
    """Per-process tally: ``staged_hits`` counts exchanges that consumed producer-staged bytes (a
    pointer match), ``copy_falls`` those that paid the copy anyway, and ``refused_by_reason`` names
    which rank-local refusal path declined (see ``_WIRE_REASONS``).
    """
    return {**_WIRE_COUNTS, "refused_by_reason": dict(_WIRE_REFUSALS)}


def _wire_refuse(reason: str, detail: str = ""):
    """Count and name a rank-local refusal, then fall open. Always returns None (the bit-identical
    fp32 path).
    """
    _WIRE_COUNTS["refusals"] += 1
    _WIRE_REFUSALS[reason] = _WIRE_REFUSALS.get(reason, 0) + 1
    if _WIRE_REFUSALS[reason] == 1:
        print(
            f"[ISOEXEC-MOE] owner_wire_sink REFUSED [{reason}]"
            f"{': ' + detail if detail else ''}. {_WIRE_REASONS.get(reason, '')}. The owner combine "
            "falls back to its stage copy: bit-identical, ~15.7us/site slower. Read "
            "moe_pik_combine_owner.wire_stage_counts().",
            flush=True,
        )
    return None


def owner_wire_sink(P: int, H: int, device):
    """Return a bf16 [P,H] view of current-phase symmetric staging, or None for the copy path.
    Refuse capture-time pool growth; eager grants and refusals rendezvous at the sink or subsequent _stage.
    Shared-expert staging uses a separate allocation so nothing overwrites routed bytes before combine;
    new sym_partial callers must preserve this lifetime and phase contract."""
    if not (wire_stage_enabled() and owner_combine_enabled()):
        return None
    group, dev = _WIRE_GROUP["group"], _WIRE_GROUP["device"]
    if group is _WIRE_UNRECORDED:
        return _wire_refuse("group_not_recorded")
    if dev != device:
        return _wire_refuse("device_mismatch", f"recorded {dev}, asked for {device}")
    if not _p2p_ok(group):
        return _wire_refuse("no_symm_p2p")
    try:
        from isoexec.ops.collectives.pik_tp_invariant import get_plan
        from isoexec.ops.moe.moe_sm90.moe_batch_invariant import _pik_bf16_wire_enabled

        if not _pik_bf16_wire_enabled():
            return _wire_refuse("bf16_wire_off")
        world, leaves = dist.get_world_size(group), get_plan().num_leaves
        if world != leaves:
            return _wire_refuse("world_ne_leaves", f"world={world}, num_leaves={leaves}")
        _ensure_canonical_pik()
        from pik.allreduce import _SYM_IN, _capturing  # type: ignore

        key = (device, world, torch.bfloat16)
        pool = _SYM_IN.get(key)
        if (pool is None or pool.cap < P * H) and _capturing():
            return _wire_refuse(
                "pool_growth_under_capture", f"need {P * H} elems, have {None if pool is None else pool.cap}"
            )
        inp, _ = _sym_pool(P * H, device, group, torch.bfloat16).stage()
    except Exception:  # noqa: BLE001 -- any surprise here must degrade to the copy path, loudly once
        if "exception" not in _WIRE_REFUSALS:
            import traceback

            print("[ISOEXEC-MOE] owner_wire_sink raised:\n" + traceback.format_exc(), flush=True)
        return _wire_refuse("exception")
    _WIRE_COUNTS["grants"] += 1
    if _WIRE_COUNTS["grants"] == 1:
        print(
            f"[ISOEXEC-MOE] owner_wire_sink GRANTED (first grant, [{P},{H}] bf16 on {device}): the fc2 "
            "leaf-tree may now store the owner combine's wire bytes straight into the current-phase "
            "symmetric staging buffer. ENGAGEMENT is this line PLUS wire_stage_counts()['staged_hits']>0 "
            "-- the exchange's own pointer match -- never the flag being exported.",
            flush=True,
        )
    v = inp[: P * H].view(P, H)
    return v


# Reuse pik's flag-pool implementation in a separate registry keyed by group membership.
# Sharing by world size would rendezvous with wrong peers or thrash incompatible block counts.
_OC_FLAGS: dict = {}
_OC_FLAG_KEEP: list = []


def _oc_flag_pool(nb: int, device, group):
    """Flag pool for THIS group, grown geometrically, never freed while a kernel may hold it."""
    _ensure_canonical_pik()
    from pik.allreduce import _capturing, _FlagPool  # type: ignore

    key = (device, _group_key(group))
    pool = _OC_FLAGS.get(key)
    if pool is None or pool.nb < nb:
        if _capturing():
            # Allocating and rendezvous-ing inside a capture is illegal, and a half-built symmetric
            # allocation hangs rather than erroring. Refuse visibly.
            raise RuntimeError(
                f"fused owner-combine: the flag pool must grow to {nb} blocks, but this is inside a "
                "CUDA graph capture. Every shape must run once EAGERLY before capture (vLLM's "
                "cudagraph_num_of_warmups=1), so the pools are sized before the graph is recorded."
            )
        want = max(nb, 2 * pool.nb if pool is not None else 0, 1024)
        if pool is not None:
            _OC_FLAG_KEEP.append(pool)  # a launched kernel may still hold the old pointers
        pool = _FlagPool(want, device, group)
        _OC_FLAGS[key] = pool
    return pool


# Fused and unfused paths use different barriers, so rank disagreement deadlocks.
# Geometry is group-wide; _oc_group_on and _agree make enablement and admission group-wide too.
_FUSED_ENV = "ISOEXEC_PIK_FUSED_OWNER_COMBINE"


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


_FUSED_ON = _env_on(_FUSED_ENV)

# Per (world, s, nhb, k, wire) admission verdict. True = proven bitwise-equal to the unfused
# barrier+push+barrier exchange on LIVE operands at this exact launch geometry.
_FUSED_ADMIT: dict = {}
_FUSED_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_FUSED_FIRST_REJECT: dict = {}
_FUSED_GROUP: dict = {}


def fused_owner_enabled() -> bool:
    """Whether this process is allowed to try the fused single-launch owner exchange."""
    return _FUSED_ON


def set_fused_owner_enabled(on: bool) -> None:
    """Test/bench hook. Flipping this does NOT bypass admission: every shape still proves."""
    global _FUSED_ON
    _FUSED_ON = bool(on)


def fused_owner_counts() -> dict:
    """Per-process tally of the fused owner combine.

    ``calls`` counts dispatches onto the fused path, i.e. Python-side entries. Under CUDA graphs that
    is the capture-time (and warmup) entry, not the replay -- a replay re-runs the kernel without
    re-entering this module.
    """
    return {
        **_FUSED_COUNTS,
        "enabled": _FUSED_ON,
        "shapes": {str(k): v for k, v in _FUSED_ADMIT.items()},
        "first_reject": dict(_FUSED_FIRST_REJECT) or None,
    }


def _group_key(group):
    try:
        return tuple(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001 -- default group, or an older torch
        return id(group)


def _oc_group_on(group, device) -> bool:
    """Has this GROUP agreed the fused exchange may be attempted? One collective, once, ever."""
    _ensure_canonical_pik()
    from pik.allreduce import _agree, _capturing  # type: ignore

    key = _group_key(group)
    got = _FUSED_GROUP.get(key)
    if got is not None:
        return got
    if _capturing():
        _FUSED_COUNTS["capture_skips"] += 1
        return False
    got = _agree(_FUSED_ON, group, device)
    _FUSED_GROUP[key] = got
    if got != _FUSED_ON:
        print(
            f"[ISOEXEC-MOE] the FUSED owner-combine is DISABLED for process group {key}: this rank "
            f"has {_FUSED_ENV}={'on' if _FUSED_ON else 'off'} but at least one peer disagrees, and "
            "running the two launch structures side by side would deadlock. Set the flag "
            "identically on every rank of a TP group.",
            flush=True,
        )
    elif got:
        print(
            f"[ISOEXEC-MOE] FUSED owner-combine ENABLED and agreed by all "
            f"{dist.get_world_size(group) if dist.is_initialized() else 1} ranks of group {key}. "
            "Each launch geometry still has to prove itself bitwise against the "
            "barrier+push+barrier reference before it is used.",
            flush=True,
        )
    return got


def _oc_reject(key, reason: str, detail: str = "") -> None:
    """Refuse the fused exchange for this geometry, LOUDLY and once. Never silently."""
    global _FUSED_FIRST_REJECT
    _FUSED_ADMIT[key] = False
    _FUSED_COUNTS["rejected"] += 1
    if _FUSED_FIRST_REJECT:
        return
    _FUSED_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    print(
        f"[ISOEXEC-MOE] WARNING: the FUSED owner-combine was REFUSED at {key} -- {reason}. {detail} "
        "The result is unaffected (this geometry falls back to the barrier+push+barrier reference, "
        "which is what every frozen bit was proven against), but the launch-count saving is not "
        "being taken here. Read moe_pik_combine_owner.fused_owner_counts().",
        flush=True,
    )


def _oc_admit(key, ex, group, device) -> bool:
    """Require live bitwise equality between fused and reference exchange with group-wide agreement.
    Two back-to-back exchanges restore both double-buffer phases; every exit, including exceptions,
    participates in _agree so ranks cannot choose different barrier paths."""
    _ensure_canonical_pik()
    from pik.allreduce import _agree  # type: ignore

    reason = detail = ""
    ref = ex.unfused().clone()
    local_ok = True
    try:
        got = ex.fused().clone()
    except Exception as e:  # noqa: BLE001
        _FUSED_COUNTS["errors"] += 1
        local_ok, reason, detail = False, "the fused kernel raised", repr(e)
    else:
        if ref.shape != got.shape or ref.dtype != got.dtype:
            local_ok, reason = False, "shape/dtype mismatch"
            detail = f"ref={ref.shape}/{ref.dtype} got={got.shape}/{got.dtype}"
        elif not torch.equal(_bits(ref), _bits(got)):
            local_ok, reason = False, "BIT PATTERNS DIFFER"
            detail = f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements"

    agreed = _agree(local_ok, group, device)
    if not agreed:
        if local_ok:
            _FUSED_COUNTS["peer_vetoes"] += 1
            reason = "a PEER rank refused it"
            detail = "this rank's own compare passed; the group runs the reference path together"
        _oc_reject(key, reason, detail)
        return False
    _FUSED_ADMIT[key] = True
    _FUSED_COUNTS["admitted"] += 1
    if _FUSED_COUNTS["admitted"] == 1:
        # Positive banner, once: silence on the success path is indistinguishable from the flag never
        # reaching this process, or from every shape being first seen under capture and skipped.
        print(
            f"[ISOEXEC-MOE] FUSED owner-combine ADMITTED (first geometry {key}): this geometry now "
            "runs as ONE launch instead of barrier+push+barrier. Bit patterns were compared against "
            "the reference exchange on live operands and agreed across the group. A decode trace "
            "should show c10d::symmetric_memory::barrier_kernel drop by 2 calls per MoE layer per "
            "step. Read moe_pik_combine_owner.fused_owner_counts() for the served-call count.",
            flush=True,
        )
    return True


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's BIT PATTERN as an integer tensor. ``torch.equal`` on floats is not a bit compare:
    it says -0.0 == 0.0 and NaN != NaN, so a reduction that flipped a sign bit passes."""
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    return t


def _served_tick(key) -> None:
    """Served-call count on the fused path. Loud early, then rare; never an install banner."""
    _FUSED_COUNTS["calls"] += 1
    n = _FUSED_COUNTS["calls"]
    if n <= 2 or n % 4096 == 0:
        print(
            f"[ISOEXEC-MOE-FUSED-OWNER] pid={os.getpid()} served={n} dispatches on the FUSED "
            f"owner-combine (one launch, no separate barriers); last geometry {key}",
            flush=True,
        )


class _Exchange:
    """The push exchange in its two launch structures, sharing every operand.

    Both members stage the same partial into the same double-buffered symmetric pool, read the same
    peer buffers, launch a kernel generated from the same ``_oc_core`` string, and copy out the same
    slice. The only difference is whether the two rendezvous are separate symm-mem barrier kernels
    around the push, or the push kernel's own prologue and epilogue.
    """

    def __init__(self, partial, my_rows, s, Tp, T, H, k, group, wire_bf16, block):
        self.partial, self.my_rows = partial, my_rows
        self.s, self.Tp, self.T, self.H, self.k = s, Tp, T, H, k
        self.group, self.wire_bf16, self.block = group, wire_bf16, block
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.dev = partial.device
        self.dt = torch.bfloat16 if wire_bf16 else torch.float32
        self.P, _ = partial.shape

    def _stage(self):
        """Stage this rank's partial and hand back the peer views plus the output pool.

        A dtype-converting ``copy_`` rounds RNE exactly like ``.to(dt)``, so the fp32 partial lands in
        the bf16 wire buffer in one pass. If the fc2 leaf-tree already stored the wire bytes into this
        very buffer (``owner_wire_sink``), the pointer compare detects it and the copy is skipped; any
        mismatch falls open to the bit-identical copy.
        """
        pool = _sym_pool(self.P * self.H, self.dev, self.group, self.dt)
        inp, h_in = pool.stage()
        if self.partial.dtype == self.dt and self.partial.data_ptr() == inp.data_ptr():
            _WIRE_COUNTS["staged_hits"] += 1  # producer already staged these exact bytes
            if _WIRE_COUNTS["staged_hits"] == 1:
                # A grant only means the sink handed out a buffer; this means the exchange found the
                # producer's bytes already in it and skipped the copy.
                print(
                    f"[ISOEXEC-MOE] owner combine CONSUMED producer-staged wire bytes (first hit, "
                    f"[{self.P},{self.H}] bf16, rank {self.rank}/{self.world}): the fc2 leaf-tree's "
                    "store IS the staging, so the [P,H] cast-copy before the lead barrier is gone.",
                    flush=True,
                )
        else:
            if self.partial.dtype == torch.bfloat16 and wire_stage_enabled():
                _WIRE_COUNTS["copy_falls"] += 1
            inp[: self.P * self.H].view(self.P, self.H).copy_(self.partial)
        peers = [h_in.get_buffer(r, (self.P * self.H,), self.dt) for r in range(self.world)]
        opool = _out_sym_pool(self.Tp * self.H, self.dev, self.group)
        out_sym, h_out = opool.stage()
        outs = [h_out.get_buffer(r, (self.Tp * self.H,), torch.bfloat16) for r in range(self.world)]
        return pool, h_in, peers, opool, out_sym, h_out, outs

    def _finish(self, pool, opool, out_sym):
        """Copy out [:T] to a fresh tensor, dropping the world-padding and detaching from the
        symmetric buffer so the double-buffered pool may reuse it next layer. CUDA-graph safe: shapes
        are static and the copy is stream-ordered.
        """
        pool.flip()
        result = out_sym[: self.Tp * self.H].view(self.Tp, self.H)[: self.T].contiguous()
        opool.flip()
        return result

    @property
    def grid(self):
        import triton

        return (self.s, triton.cdiv(self.H, self.block))

    def unfused(self):
        """The reference exchange: barrier, push, barrier -- three launches, and the default."""
        pool, h_in, peers, opool, out_sym, h_out, outs = self._stage()
        h_in.barrier()  # every rank's partial visible to peers
        pk = _owner_push_kernel(self.world, self.k, self.wire_bf16)
        pk[self.grid](
            *peers,
            self.my_rows,
            *outs,
            self.rank * self.s,
            self.Tp,
            self.H,
            K=self.k,
            BLOCK=self.block,
            num_warps=4,
        )
        h_out.barrier()  # every rank's owned slice has landed in every peer's out buffer
        return self._finish(pool, opool, out_sym)

    def fused(self):
        """Run the shared owner arithmetic with leading and trailing barriers inside one kernel.
        The leading wait makes input/output phase N reusable at N+2 after peer reads/copy-outs complete.
        The trailing release/acquire wait follows every peer block's stores and debug barrier,
        so copy-out observes all pushes; each output has one writer."""
        import triton

        nhb = triton.cdiv(self.H, self.block)
        fp = _oc_flag_pool(self.s * nhb, self.dev, self.group)
        pool, _h_in, peers, opool, out_sym, _h_out, outs = self._stage()
        # No leading barrier here: the kernel's own prologue is the barrier. The staging copy above is
        # still ordered before it, since a kernel block runs only after the copy kernel retired.
        pk = _owner_push_fused_kernel(self.world, self.k, self.wire_bf16)
        pk[self.grid](
            *peers,
            self.my_rows,
            *outs,
            *fp.peers,
            fp.mine,
            fp.epoch,
            fp.my_slot(0),
            fp.my_slot(1),
            fp.wait_base(1),
            fp.nb,
            self.rank * self.s,
            self.Tp,
            self.H,
            K=self.k,
            BLOCK=self.block,
            NHB=nhb,
            WORLD=self.world,
            num_warps=4,
        )
        return self._finish(pool, opool, out_sym)


# Fused barriers issue grid*world remote atomics, so admit only small grids (default at most 8 blocks).
_MAX_BLOCKS_ENV = "ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS"


def _max_fused_blocks() -> int:
    try:
        return max(1, int(os.environ.get(_MAX_BLOCKS_ENV, "8")))
    except ValueError:
        return 8


def _owner_push_exchange(ex):
    """Dispatch only. ``_Exchange.unfused`` is the default and the reference; ``_Exchange.fused`` is
    used only for a geometry proven bitwise-identical to it on live operands in this process and small
    enough that fusing is not a regression.

    Every branch here is taken identically on every rank of the group. The one input that is not
    invariant by construction, the env flag, is settled by ``_oc_group_on`` before it is read, by a
    collective every rank enters; the grid gate is a pure function of rank-invariant shape facts.
    """
    _ensure_canonical_pik()
    from pik.allreduce import _capturing  # type: ignore

    if not _oc_group_on(ex.group, ex.dev):
        return ex.unfused()

    import triton

    key = (ex.world, ex.s, triton.cdiv(ex.H, ex.block), ex.k, "bf16" if ex.wire_bf16 else "fp32")
    verdict = _FUSED_ADMIT.get(key)
    if verdict is None:
        blocks = ex.s * triton.cdiv(ex.H, ex.block)
        cap = _max_fused_blocks()
        if blocks > cap:
            _oc_reject(
                key,
                "the grid exceeds the MEASURED fusion crossover",
                f"{blocks} blocks > {cap}: the in-kernel handshake costs ~2.0 us PER BLOCK (flat in "
                f"payload, measured 8xH100), so fusing this geometry would cost ~{blocks * 2.0:.0f} us "
                f"against a device-wide barrier pair's 10-35 us. Raise {_MAX_BLOCKS_ENV} only with a "
                "measurement that says otherwise.",
            )
            return ex.unfused()
        if ex.world < 2 or (ex.world & (ex.world - 1)):
            _oc_reject(key, "world size is not a power of two", f"world={ex.world}")
            return ex.unfused()
        if _capturing():
            # A compare needs a host sync, which capture forbids. Do NOT record a verdict: the
            # geometry gets another chance the next time it is seen eagerly.
            _FUSED_COUNTS["capture_skips"] += 1
            return ex.unfused()
        try:
            verdict = _oc_admit(key, ex, ex.group, ex.dev)
        except RuntimeError as e:  # the flag pool refused to grow (capture); never a wrong answer
            _oc_reject(key, "the flag pool could not be sized", repr(e))
            return ex.unfused()
    if not verdict:
        return ex.unfused()
    _served_tick(key)
    return ex.fused()


def _owner_combine(partial, rows, T, k, group, wire_bf16, block: int = 1024):
    """Owner-computes combine.

    ``partial`` is this rank's [P,H] leaf-subtree fc2 partial (fp32); ``rows`` is the [T,k] global
    permuted-row index per token, ascending expert. Returns bf16 [T,H], bit-identical on every rank.
    """
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    P, H = partial.shape
    dev = partial.device
    dt = torch.bfloat16 if wire_bf16 else torch.float32
    _WIRE_GROUP["group"], _WIRE_GROUP["device"] = group, dev  # the sink needs the group; record it

    # shape-only ownership: contiguous slice, pad T up to a multiple of world
    s = -(-T // world)  # ceil, host-known
    Tp = s * world
    if rows.dtype == torch.int32 and T % world == 0:
        # The counting-sort kernel already emitted [T,k] int32 contiguous, and at a divisible T the
        # pad is vacuous, so this rank's slice is a contiguous VIEW -- no fill, copy or cast.
        my_rows = rows[rank * s : (rank + 1) * s]
    else:
        rows_p = rows.new_zeros(Tp, k)
        rows_p[:T] = rows
        my_rows = rows_p[rank * s : (rank + 1) * s].contiguous().to(torch.int32)  # [s,k]

    grid = (s, triton.cdiv(H, block))

    if _p2p_ok(group):
        # symm-mem PUSH exchange, no NCCL in the hot path: the kernel reads peers, computes each owned
        # token's tree + k-sum, and writes the finished bf16 row into every peer's symmetric output
        # buffer at its global offset. Two barriers complete it, or zero on the admitted fused path.
        return _owner_push_exchange(_Exchange(partial, my_rows, s, Tp, T, H, k, group, wire_bf16, block))

    # NCCL correctness fallback (no NVLink / no symm-mem): all-gather partials, run the kernel over
    # local copies, all-gather the finals. Same tree/k-sum/round, bit-identical to the P2P push.
    out_local = torch.empty(s, H, dtype=torch.bfloat16, device=dev)
    kern = _owner_kernel(world, k, wire_bf16)
    staged = partial.to(dt) if partial.dtype != dt else partial
    buf = torch.empty(world, P * H, dtype=dt, device=dev)
    dist.all_gather_into_tensor(buf, staged.contiguous().view(-1), group=group)
    peers = [buf[r] for r in range(world)]
    kern[grid](*peers, my_rows, out_local, s, H, K=k, BLOCK=block, num_warps=4)
    full = torch.empty(Tp, H, dtype=torch.bfloat16, device=dev)
    dist.all_gather_into_tensor(full, out_local, group=group)
    return full[:T]


# Admission for the unified host's shared+routed composition (on in TRAINER_ENV).
_SHARED_OWNER_ENV = "ISOEXEC_MOE_SHARED_OWNER_FUSION"


def shared_owner_fusion_enabled() -> bool:
    """Exact shared+routed composition on the unified host. Declared default OFF; TRAINER_ENV sets it."""
    return _env_on(_SHARED_OWNER_ENV, "0")


def _rows32_enabled() -> bool:
    return os.environ.get("ISOEXEC_MOE_OC_ROWS32", "1") == "1"


def _build_rows(sorted_indices, T, k, rows=None):
    """``rows[t, :]`` = the k permuted-row positions of token t, ascending expert.

    Reproduces ``moe_batch_invariant._fixed_order_combine``'s ``rows`` (the same stable argsort of
    ``sorted_indices``). When enabled, the counting-sort kernel is asked for the [T,k] int32 form
    directly -- the identical permutation, minus the int64 materialization the general path converts
    away afterwards -- and falls back to the int64 path whenever the kernel declines.
    """
    from isoexec.ops.moe.moe_sm90.moe_fused_permute import validate_combine_rows

    rows = validate_combine_rows(rows, sorted_indices, T)
    if rows is not None:
        return rows
    if _rows32_enabled():
        from isoexec.ops.moe.moe_sm90.moe_combine_rows_kernel import stable_combine_rows

        rows = stable_combine_rows(sorted_indices, T, dtype=torch.int32)
        if rows is not None:
            return rows
    from isoexec.ops.moe.moe_sm90.moe_combine_rows_kernel import stable_argsort_order

    order = stable_argsort_order(sorted_indices, T)
    return order.view(T, k)


def owner_combine_enabled() -> bool:
    return os.environ.get("ISOEXEC_MOE_PIK_OWNER_COMBINE", "0") == "1"


def _wire_bf16(partial, group) -> bool:
    """Match ``_PikTreeAllReduce``'s wire decision exactly: bf16 only at world==G with a lossless
    leaf, else fp32. Reuses the same self-check state so the rounding decision is shared.

    A bf16 partial IS the wire: it only ever arrives from ``owner_wire_sink``, where the fc2 leaf-tree
    stored the already-rounded leaf value, so the losslessness check is vacuous by construction.
    """
    if partial.dtype == torch.bfloat16:
        return True
    from isoexec.ops.collectives.pik_tp_invariant import get_plan
    from isoexec.ops.moe.moe_sm90.moe_batch_invariant import _bf16_wire, _pik_bf16_wire_enabled

    if not (
        _pik_bf16_wire_enabled()
        and partial.dtype == torch.float32
        and dist.get_world_size(group) == get_plan().num_leaves
    ):
        return False
    if not _bf16_wire["checked"]:
        _bf16_wire["checked"] = True
        _bf16_wire["ok"] = bool((partial.to(torch.bfloat16).to(torch.float32) == partial).all())
    return _bf16_wire["ok"]


# Unified owner composition, per token:
#   routed = ascending-expert fp32 sum of ETP roots, seeded by the first term
#   shared = bf16(bf16(ETP tree(shared leaves)) * gate_bf16)
#   result = bf16(routed + fp32(shared))
# The owner folds the residual-boundary cast into its store; live group-wide admission checks these bits.


def _unified_owner_core(world: int, k: int, routed_bf16: bool, shared: bool, shared_bf16: bool, gated: bool,
                        weight_after_fc2: bool = False) -> str:
    """Generate unified owner arithmetic: seed routed fp32 sum from its first term, preserving signed zero.
    Keep the shared bf16 root and gate-product rounds, then narrow their sum once at the store.
    Padded owner slots read safe rows and are discarded by [:T] copy-out."""
    if weight_after_fc2 and routed_bf16:
        raise ValueError("After-FC2 owner requires FP32 routed wire")
    stores = "".join(f"    tl.store(out{i} + goff, result, mask=hmask)\n" for i in range(world))
    routed_tree = _named_tree_src(world, routed_bf16, inp="rin", tmp="rt", base="rbase", indent="        ")
    if weight_after_fc2:
        routed_tree += """        prob_t = tl.minimum(global_t, T - 1)
        route_prob_bf = tl.load(route_probs_ptr + prob_t * K + j).to(tl.bfloat16)
        route_root_bf = rt0.to(tl.bfloat16)
        rt0 = (route_root_bf.to(tl.float32) * route_prob_bf.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
"""
    src = f"""    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    global_t = out_base + t
    routed_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        rbase = row * H + hoff
{routed_tree}        if j == 0:
            routed_acc = rt0
        else:
            routed_acc = routed_acc + rt0
"""
    if shared:
        shared_tree = _named_tree_src(world, shared_bf16, inp="sin", tmp="st", base="sbase", indent="    ")
        src += f"""    src_t = tl.minimum(global_t, T - 1)
    sbase = src_t * H + hoff
{shared_tree}    shared_root_bf = st0.to(tl.bfloat16)
"""
        if gated:
            src += """    gate_bf = tl.load(gate_ptr + src_t).to(tl.bfloat16)
    shared_bf = (shared_root_bf.to(tl.float32) * gate_bf.to(tl.float32)).to(tl.bfloat16)
"""
        else:
            src += "    shared_bf = shared_root_bf\n"
        src += "    result = (routed_acc + shared_bf.to(tl.float32)).to(tl.bfloat16)\n"
    else:
        src += "    result = routed_acc.to(tl.bfloat16)\n"
    src += f"    goff = global_t * H + hoff\n{stores}"
    return src


_UNIFIED_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({args}, out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr):
{core}"""


def _unified_push_src(world: int, k: int, routed_bf16: bool, shared: bool, shared_bf16: bool, gated: bool,
                      weight_after_fc2: bool = False) -> str:
    args = [f"rin{i}" for i in range(world)] + ["rows_ptr"]
    if weight_after_fc2:
        args.append("route_probs_ptr")
    if shared:
        args += [f"sin{i}" for i in range(world)]
        if gated:
            args.append("gate_ptr")
    args += [f"out{i}" for i in range(world)]
    return _UNIFIED_PUSH_TMPL.format(
        args=", ".join(args),
        core=_unified_owner_core(world, k, routed_bf16, shared, shared_bf16, gated, weight_after_fc2)
    )


@functools.cache
def _unified_owner_push_kernel(world: int, k: int, routed_bf16: bool, shared: bool, shared_bf16: bool, gated: bool,
                               weight_after_fc2: bool = False):
    tag = f"_unified_owner_push_c{world}_k{k}_r{'bf16' if routed_bf16 else 'fp32'}" + (
        f"_s{'bf16' if shared_bf16 else 'fp32'}{'_gated' if gated else ''}" if shared else "_noshared"
    )
    if weight_after_fc2:
        tag += "_after_fc2"
    return _load(tag, _unified_push_src(world, k, routed_bf16, shared, shared_bf16, gated, weight_after_fc2))


def _engine_p2p(group) -> bool:
    """The transport decision the ENGINE made (``select_engine_transport`` monkeypatches
    ``pik.allreduce.p2p_available`` to False under ISOEXEC_PIK_P2P=0 and on an NCCL-only arch, of which
    none ships today), not this module's own ``_p2p_ok``: with pik on NCCL the owner exchange stays off and the
    composition is the reference."""
    _ensure_canonical_pik()
    from pik.allreduce import p2p_available  # type: ignore

    try:
        return bool(p2p_available(group))
    except Exception:  # noqa: BLE001
        return False


def _unified_pools_ready(P: int, T: int, Tp: int, H: int, device, group, rdt, sdt) -> bool:
    """Would serving this geometry GROW a symmetric pool? Under CUDA-graph capture a symmetric
    allocation hangs rather than errors, so a growing geometry runs the reference instead. Shape
    facts only, so every rank answers the same."""
    _ensure_canonical_pik()
    from pik.allreduce import _SYM_IN  # type: ignore

    world = dist.get_world_size(group)
    rp = _SYM_IN.get((device, world, rdt))
    if rp is None or rp.cap < P * H:
        return False
    if sdt is not None:
        sp = _SHARED_IN_SYM.get((device, _group_key(group), sdt))
        if sp is None or sp.cap < T * H:
            return False
    op = _OUT_SYM.get((device, world))
    return op is not None and op.cap >= Tp * H


def stage_shared_partial(T: int, H: int, dtype: torch.dtype, group, device):
    """A ``[T,H]`` view of the current-phase shared-input symmetric buffer for the shared expert's fc2
    leaf GEMM to store into, or ``None`` (the GEMM allocates, the exchange copies -- bit-identical).

    Rank-local and fail-open by design, like ``owner_wire_sink``: the exchange detects a producer-
    staged buffer by pointer compare, so ranks may disagree freely. Refuses when the owner composition
    is off (flags), pik is not on P2P, or the pool would have to grow under capture."""
    if not (owner_combine_enabled() and shared_owner_fusion_enabled()):
        return None
    if group is None or not dist.is_initialized() or dist.get_world_size(group) < 2 or not _engine_p2p(group):
        return None
    try:
        from pik.allreduce import _capturing  # type: ignore

        key = (device, _group_key(group), dtype)
        pool = _SHARED_IN_SYM.get(key)
        if (pool is None or pool.cap < T * H) and _capturing():
            return None
        buf, _ = _shared_in_sym_pool(T * H, device, group, dtype).stage()
    except Exception:  # noqa: BLE001 -- staging is an optimisation; the copy path is the reference
        return None
    return buf[: T * H].view(T, H)


class _UnifiedExchange:
    """One symmetric-memory push exchange of the unified composition: stage both partials (or find
    them already staged by their producers), one group barrier, the owner kernel, one group barrier,
    copy out ``[:T]``."""

    def __init__(self, routed_partial, shared, my_rows, s, Tp, T, H, k, group, routed_bf16, block: int = 1024,
                 *, route_probabilities=None):
        self.routed_partial = routed_partial
        self.route_probabilities = route_probabilities
        self.shared = shared  # SharedExpertPartial | None
        self.my_rows = my_rows
        self.s, self.Tp, self.T, self.H, self.k = s, Tp, T, H, k
        self.group, self.block = group, block
        self.routed_bf16 = routed_bf16
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.dev = routed_partial.device
        self.P = routed_partial.shape[0]

    def run(self) -> torch.Tensor:
        import triton

        rdt = torch.bfloat16 if self.routed_bf16 else torch.float32
        rpool = _sym_pool(self.P * self.H, self.dev, self.group, rdt)
        rinp, rh = rpool.stage()
        if self.routed_partial.dtype == rdt and self.routed_partial.data_ptr() == rinp.data_ptr():
            _ENGINE_COUNTS["routed_staged_hits"] += 1  # the fc2 leaf-tree's wire store WAS the staging
            _WIRE_COUNTS["staged_hits"] += 1  # the census owner_wire_sink's grant banner points at
        else:
            _ENGINE_COUNTS["routed_copies"] += 1
            if self.routed_partial.dtype == torch.bfloat16 and wire_stage_enabled():
                _WIRE_COUNTS["copy_falls"] += 1
            rinp[: self.P * self.H].view(self.P, self.H).copy_(self.routed_partial)
        rpeers = [rh.get_buffer(r, (self.P * self.H,), rdt) for r in range(self.world)]

        spool = speers = None
        gate = None
        if self.shared is not None:
            sp = self.shared.partial
            sdt = sp.dtype
            spool = _shared_in_sym_pool(self.T * self.H, self.dev, self.group, sdt)
            sinp, sh = spool.stage()
            if sp.data_ptr() == sinp.data_ptr():
                _ENGINE_COUNTS["shared_staged_hits"] += 1  # stage_shared_partial handed the GEMM this buffer
            else:
                _ENGINE_COUNTS["shared_copies"] += 1
                sinp[: self.T * self.H].view(self.T, self.H).copy_(sp)
            speers = [sh.get_buffer(r, (self.T * self.H,), sdt) for r in range(self.world)]
            if self.shared.gate is not None:
                gate = self.shared.gate.reshape(-1)
                if not gate.is_contiguous():
                    gate = gate.contiguous()

        opool = _out_sym_pool(self.Tp * self.H, self.dev, self.group)
        out_sym, oh = opool.stage()
        outs = [oh.get_buffer(r, (self.Tp * self.H,), torch.bfloat16) for r in range(self.world)]

        # Both staging copies precede this on the stream; one group rendezvous publishes both pools.
        rh.barrier()
        pk = _unified_owner_push_kernel(
            self.world,
            self.k,
            self.routed_bf16,
            self.shared is not None,
            self.shared is not None and self.shared.partial.dtype == torch.bfloat16,
            gate is not None,
            self.route_probabilities is not None,
        )
        args = [*rpeers, self.my_rows]
        if self.route_probabilities is not None:
            args.append(self.route_probabilities)
        if speers is not None:
            args += speers
            if gate is not None:
                args.append(gate)
        args += outs
        grid = (self.s, triton.cdiv(self.H, self.block))
        pk[grid](
            *args,
            self.rank * self.s,
            self.T,
            self.H,
            K=self.k,
            BLOCK=self.block,
            num_warps=4,
            enable_fp_fusion=False,  # load-bearing: no FFMA through the bf16 casts
        )
        oh.barrier()  # every rank's owned slice has landed in every peer's out buffer

        rpool.flip()
        if spool is not None:
            spool.flip()
        result = out_sym[: self.Tp * self.H].view(self.Tp, self.H)[: self.T].contiguous()
        opool.flip()
        return result


# Per-geometry admission of the unified composition, group-agreed, refusals loud and permanent.
_ENGINE_ADMIT: dict = {}
_ENGINE_FIRST_REJECT = None
_ENGINE_COUNTS = {
    "calls": 0,
    "served": 0,
    "admitted": 0,
    "rejected": 0,
    "fallbacks": 0,
    "capture_skips": 0,
    "routed_staged_hits": 0,
    "routed_copies": 0,
    "shared_staged_hits": 0,
    "shared_copies": 0,
}


def engine_owner_counts() -> dict:
    """The unified host's owner-combine census: ``served`` is the number of MoE layer-forwards whose
    combine ran as the owner exchange (engagement evidence for ``moe.combine:pik_leaf_tree`` --
    ``core/enforce.SERVED_COUNTERS``); ``fallbacks`` ran the in-place trees; ``*_staged_hits`` count
    exchanges that found a producer's bytes already in the symmetric buffer and skipped the copy."""
    return {
        **_ENGINE_COUNTS,
        "shapes": {str(k): v for k, v in _ENGINE_ADMIT.items()},
        "first_reject": _ENGINE_FIRST_REJECT,
    }


def _engine_reject(key, reason: str) -> None:
    global _ENGINE_FIRST_REJECT
    _ENGINE_ADMIT[key] = False
    _ENGINE_COUNTS["rejected"] += 1
    if _ENGINE_FIRST_REJECT is None:
        _ENGINE_FIRST_REJECT = {"key": str(key), "reason": reason}
    print(
        f"[ISOEXEC-MOE] unified owner combine REFUSED at {key}: {reason}. This geometry runs the in-place "
        "composition (pik tree all-reduce of the [P,H] partial + the shared expert's own tree all-reduce + the "
        "k-order fold) from now on -- the result is unaffected, the data movement is not saved here. Read "
        "moe_pik_combine_owner.engine_owner_counts().",
        flush=True,
    )


def _engine_reference(routed, shared, eager_fn):
    """The current composition, from the deferred pieces: the expert region's own tree on the rows,
    the shared expert's own tree + gate, then the eager combine. fp32 ``[T,H]``."""
    from isoexec.ops.moe.carriers import ExpertOutputs

    red = ExpertOutputs(
        rows=routed.reduce(routed.rows),
        layout="permuted",
        routing=routed.routing,
        extra=routed.extra,
        combine_rows=routed.combine_rows,
    )
    # Large PIK reductions lend a reusable output pool. The shared fc2 tree
    # uses the same pool/dtype and can overwrite these routed rows. Consume
    # them into the owned fp32 k-sum before starting that second reduction.
    # The canonical arithmetic remains: fp32 k-sum + rounded shared output;
    # no bf16 round is introduced and no full [T*k,H] clone is needed.
    out = eager_fn(red, None)
    return out + shared.eager() if shared is not None else out


def _engine_eligible(routed, shared, group, T: int, k: int) -> tuple[bool, str]:
    """Rank-local eligibility, settled group-wide by ``_agree`` before anything changes its launch
    structure. Every clause names itself."""
    if not owner_combine_enabled():
        return False, "ISOEXEC_MOE_PIK_OWNER_COMBINE is off on this rank"
    if shared is not None and not shared_owner_fusion_enabled():
        return False, (
            "ISOEXEC_MOE_SHARED_OWNER_FUSION is off: the unified host has no routed-only owner "
            "composition (its fp32 k-sum and the residual cast are one rounding schedule; adding a "
            "reduced shared output to a bf16-rounded routed sum would be a second round)"
        )
    if not _engine_p2p(group):
        return False, ("pik is not on the symmetric-memory P2P transport "
                       "(ISOEXEC_PIK_P2P=0, an NCCL-only arch, or no symm-mem)")
    world = dist.get_world_size(group)
    if world < 2 or (world & (world - 1)):
        return False, f"world={world} is not a power of two >= 2"
    rows = routed.rows
    if routed.weight_after_fc2:
        probabilities, indices = routed.routing.probs, routed.routing.indices
        if (rows.dtype != torch.float32 or probabilities.dtype != torch.float32
                or tuple(probabilities.shape) != (T, k) or tuple(indices.shape) != (T, k)
                or not rows.is_contiguous() or routed.combine_rows is None
                or routed.combine_rows.dtype != torch.int32 or tuple(routed.combine_rows.shape) != (T, k)
                or routed.combine_rows.device != rows.device or not routed.combine_rows.is_contiguous()
                or not isinstance(routed.extra, torch.Tensor) or routed.extra.numel() != T * k
                or routed.extra.device != rows.device
                or probabilities.device != rows.device or indices.device != rows.device
                or indices.dtype not in (torch.int32, torch.int64)):
            return False, "After-FC2 owner requires FP32 routed rows/probabilities and colocated integer routes"
    if rows.ndim != 2 or rows.shape[0] != T * k or T <= 0:
        return False, f"routed rows {tuple(rows.shape)} are not [T*k={T * k}, H]"
    if rows.dtype not in (torch.float32, torch.bfloat16):
        return False, f"routed rows dtype {rows.dtype} is neither fp32 nor bf16"
    if routed.reduce is None or routed.extra is None:
        return False, "permuted_partial rows without a deferred reduce / the permutation"
    if shared is not None:
        sp = shared.partial
        if sp.ndim != 2 or tuple(sp.shape) != (T, rows.shape[1]):
            return False, f"shared partial {tuple(sp.shape)} is not [T={T}, H={rows.shape[1]}]"
        if sp.dtype not in (torch.float32, torch.bfloat16) or not sp.is_contiguous():
            return False, f"shared partial dtype {sp.dtype} / contiguous={sp.is_contiguous()}"
        if shared.gate is not None and (shared.gate.numel() != T or shared.gate.dtype != torch.bfloat16):
            return False, f"shared gate {tuple(shared.gate.shape)} {shared.gate.dtype} is not bf16[T]"
    return True, ""


def _engine_my_rows(rows: torch.Tensor, T: int, k: int, world: int, rank: int):
    """This rank's contiguous owner slice of the [T,k] combine rows, world-padded (pad rows -> row 0)."""
    s = -(-T // world)
    Tp = s * world
    if rows.dtype == torch.int32 and T % world == 0:
        return rows[rank * s : (rank + 1) * s], s, Tp
    rows_p = rows.new_zeros(Tp, k)
    rows_p[:T] = rows
    return rows_p[rank * s : (rank + 1) * s].contiguous().to(torch.int32), s, Tp


def _engine_owner_dispatch(routed, shared, group, eager_fn) -> torch.Tensor:
    """Admit once per geometry, then serve. The reference is ``_engine_reference`` -- the very
    composition this binding replaces, built from the same deferred pieces -- and admission compares
    the int16 bit patterns of ``bf16(reference)`` and the candidate, on PRIVATE CLONES of both partials
    so that neither path's staging can alias the other's symmetric buffer mid-compare. The verdict is
    ``_agree``d: one dissenting rank puts the whole group on the reference path for that geometry."""
    from isoexec.ops.moe.carriers import ExpertOutputs, SharedExpertPartial

    _ENGINE_COUNTS["calls"] += 1
    T, k = routed.routing.indices.shape
    T, k = int(T), int(k)
    rows_p = routed.rows
    dev = rows_p.device
    # The sink needs the group before the NEXT layer's fc2 runs; record it on the first dispatch, as
    # `_owner_combine` did.
    _WIRE_GROUP["group"], _WIRE_GROUP["device"] = group, dev
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    H = int(rows_p.shape[-1])
    # Preserve the FP32 wire until the complete FC2 tree has been reduced.
    after_fc2 = routed.weight_after_fc2
    routed_bf16 = not after_fc2 and (
        rows_p.dtype == torch.bfloat16 or (rows_p.dtype == torch.float32 and _wire_bf16(rows_p, group)))
    sdt = None if shared is None else shared.partial.dtype
    gated = shared is not None and shared.gate is not None
    key = (world, T, H, k, "bf16" if routed_bf16 else "fp32", None if sdt is None else str(sdt).split(".")[-1], gated, after_fc2)

    verdict = _ENGINE_ADMIT.get(key)
    if verdict is False:
        _ENGINE_COUNTS["fallbacks"] += 1
        return _engine_reference(routed, shared, eager_fn)

    _ensure_canonical_pik()
    from pik.allreduce import _agree, _capturing  # type: ignore

    if verdict is None:
        if _capturing():
            # The compare needs a host sync. No verdict is recorded: the geometry is admitted the
            # next time it is seen eagerly (vLLM warms every capture size up eagerly first).
            _ENGINE_COUNTS["capture_skips"] += 1
            _ENGINE_COUNTS["fallbacks"] += 1
            return _engine_reference(routed, shared, eager_fn)
        local_ok, why = _engine_eligible(routed, shared, group, T, k)
        # Private clones: the live partials may BE views of the symmetric staging pools (the fc2 wire
        # sink, stage_shared_partial), and the reference's own tree all-reduce stages through the same
        # pool. Two paths on private copies cannot alias; once per geometry, the copies are cheap.
        routed_c = ExpertOutputs(
            rows=rows_p.clone(),
            layout=routed.layout,
            routing=routed.routing,
            extra=routed.extra,
            combine_rows=routed.combine_rows,
            reduce=routed.reduce,
            weight_after_fc2=after_fc2,
        )
        shared_c = None if shared is None else SharedExpertPartial(shared.partial.clone(), shared.gate, shared.finish)
        ref = _engine_reference(routed_c, shared_c, eager_fn)
        ref_bf = ref.to(torch.bfloat16)
        got = None
        # All ranks must enter the candidate exchange together, including its barriers.
        local_ok = _agree(local_ok, group, dev)
        if local_ok:
            try:
                rows = _build_rows(routed.extra, T, k, routed.combine_rows)
                my_rows, s, Tp = _engine_my_rows(rows, T, k, world, rank)
                got = _UnifiedExchange(routed_c.rows, shared_c, my_rows, s, Tp, T, H, k, group, routed_bf16,
                                       route_probabilities=_owner_route_probabilities(routed)).run()
            except Exception as exc:  # noqa: BLE001 -- the agreement below keeps every rank together
                got = None
                local_ok, why = False, f"candidate raised {exc!r}"
            else:
                if got.shape != ref_bf.shape or got.dtype != ref_bf.dtype:
                    local_ok, why = (
                        False,
                        f"shape/dtype mismatch ref={ref_bf.shape}/{ref_bf.dtype} got={got.shape}/{got.dtype}",
                    )
                elif not torch.equal(_bits(ref_bf), _bits(got)):
                    local_ok = False
                    why = f"BIT PATTERNS DIFFER: {int((_bits(ref_bf) != _bits(got)).sum().item())}/{ref_bf.numel()} elements"
        agreed = _agree(local_ok, group, dev)
        if not agreed:
            _engine_reject(key, why or "a PEER rank refused it (this rank's own compare passed)")
            _ENGINE_COUNTS["fallbacks"] += 1
            return ref
        _ENGINE_ADMIT[key] = True
        _ENGINE_COUNTS["admitted"] += 1
        if _ENGINE_COUNTS["admitted"] == 1:
            print(
                f"[ISOEXEC-MOE] unified owner combine ADMITTED (first geometry {key}, rank {rank}/{world}): "
                "int16 bit patterns of bf16(pik tree all-reduce of the [P,H] partial -> k-order fp32 fold + "
                "shared expert's own tree * gate) matched the owner exchange on live operands, agreed across the "
                "group. From here each rank evaluates the SAME trees for its own tokens only, reading peers over "
                "NVLink: no [P,H] all-reduce and no separate shared-expert all-reduce per MoE layer. Read "
                "moe_pik_combine_owner.engine_owner_counts().",
                flush=True,
            )
        _ENGINE_COUNTS["served"] += 1
        return got

    # Admitted. Under capture a pool that would have to grow is a hang, not an error -- shape facts,
    # so every rank takes the same branch.
    if _capturing() and not _unified_pools_ready(
        int(rows_p.shape[0]),
        T,
        -(-T // world) * world,
        H,
        dev,
        group,
        torch.bfloat16 if routed_bf16 else torch.float32,
        sdt,
    ):
        _ENGINE_COUNTS["capture_skips"] += 1
        _ENGINE_COUNTS["fallbacks"] += 1
        return _engine_reference(routed, shared, eager_fn)
    rows = _build_rows(routed.extra, T, k, routed.combine_rows)
    my_rows, s, Tp = _engine_my_rows(rows, T, k, world, rank)
    got = _UnifiedExchange(rows_p, shared, my_rows, s, Tp, T, H, k, group, routed_bf16,
                           route_probabilities=_owner_route_probabilities(routed)).run()
    _ENGINE_COUNTS["served"] += 1
    n = _ENGINE_COUNTS["served"]
    if n <= 2 or n % 4096 == 0:
        print(
            f"[ISOEXEC-MOE-OWNER] pid={os.getpid()} served={n} MoE combines on the unified owner exchange "
            f"(routed staged hits {_ENGINE_COUNTS['routed_staged_hits']} / copies {_ENGINE_COUNTS['routed_copies']}, "
            f"shared staged hits {_ENGINE_COUNTS['shared_staged_hits']} / copies {_ENGINE_COUNTS['shared_copies']}); "
            f"last geometry {key}",
            flush=True,
        )
    return got


def _owner_route_probabilities(routed):
    """Owner rows are ascending-expert order, while router slots are score order."""
    if not routed.weight_after_fc2:
        return None
    routing = routed.routing
    return routing.probs.gather(1, routing.indices.argsort(dim=1))


def _negotiate_owner_mode(mode, group, device):
    """Agree immutable opt-in and arithmetic mode before any rank can take a different path.

    This runs once per bound combine/group, during eager warmup. Both legacy and
    after-FC2 modes participate, including ranks with the feature disabled.
    """
    _ensure_canonical_pik()
    from pik.allreduce import _capturing

    if _capturing():
        raise RuntimeError("Owner mode negotiation requires an eager warmup before CUDA graph capture")
    local = torch.tensor(mode, dtype=torch.int32, device=device)
    peers = torch.empty(dist.get_world_size(group) * 2, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(peers, local, group=group)
    if not bool((peers.view(-1, 2) == local).all().item()):
        raise RuntimeError("Owner combine ranks disagree on opt-in or weight_after_fc2 mode")


def engine_combine_owner(eager_fn, *, owner_exchange: bool = True):
    """Wrap eager combination with the engine owner path when owner_group and deferred pieces are supplied.
    Without owner wiring, run eager_fn; refuse a group without deferred pieces or deferred pieces without a group.

    ``owner_exchange=False`` keeps the ``owner_group`` seam but always composes the deferred pieces with
    ``_engine_reference`` (the fc2 trees, then ``eager_fn``): for an ``eager_fn`` whose fold is not the owner
    kernel's. The owner kernel starts the routed fold from its first term (``_unified_owner_core``), which is
    ``fused_combine``'s fold; a combine whose chain is seeded with +0 (moe_sm100) gives +0 where that gives -0
    on a token whose terms are all -0, so the exchange would be another program than the eager path."""
    from isoexec.ops.moe.carriers import SharedExpertPartial

    modes = {}
    refused = []

    def combine(routed, shared, *, owner_group=None):
        deferred = routed.layout == "permuted_partial"
        shared_deferred = isinstance(shared, SharedExpertPartial)
        if owner_group is None and not deferred and not shared_deferred:
            return eager_fn(routed, shared)
        if owner_group is None or not deferred or (shared is not None and not shared_deferred):
            raise RuntimeError(
                "[ISOEXEC-MOE] owner combine was handed half a seam: "
                f"owner_group={'set' if owner_group is not None else 'None'}, routed.layout={routed.layout!r}, "
                f"shared={type(shared).__name__}. nn.MoE placement wires the group, the deferred "
                "expert rows and the deferred shared expert together; refusing rather than reducing twice or never"
            )
        opted = os.environ.get("ISOEXEC_MOE_AFTER_FC2_OWNER", "0") == "1"
        if opted and not owner_exchange and not refused:
            refused.append(True)
            print("[ISOEXEC-MOE] ISOEXEC_MOE_AFTER_FC2_OWNER=1 is not served by this combine: the owner exchange "
                  "folds the routed terms from the first term and this arch's combine from +0, two programs on a "
                  "token whose terms are all -0. Every rank runs the reference composition (the fc2 trees, then "
                  "the eager combine).", flush=True)
        mode = (opted and owner_exchange, routed.weight_after_fc2)
        if owner_group not in modes:
            _negotiate_owner_mode(mode, owner_group, routed.rows.device)
            modes[owner_group] = mode
        elif modes[owner_group] != mode:
            raise RuntimeError("Owner combine mode changed after negotiation; rebuild the bound model")
        if not owner_exchange or (routed.weight_after_fc2 and not mode[0]):
            # Keep the after-FC2 path opt-in; a binder without the exchange has the reference only.
            return _engine_reference(routed, shared, eager_fn)
        return _engine_owner_dispatch(routed, shared, owner_group, eager_fn)

    combine.__name__ = combine.__qualname__ = "engine_combine_owner"
    return combine
