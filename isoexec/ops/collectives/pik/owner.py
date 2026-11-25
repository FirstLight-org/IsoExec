"""Owner tree all-reduce: pik's fixed leaf tree in ONE launch per row-parallel site.

Arithmetic: every output element is ``round_root(tree(operands))`` where the operands are the W*C wire planes of
the W ranks in rank-major order (rank r's plane j is operand ``r*C + j``) and ``tree`` is the balanced fp32
tree of ``pik.plan.combine_order``. A plane is either a bf16 LEAF (leaf wire: C = leaves per rank) or an fp32
internal NODE (node wire: C = 1, the producer's own subtree of the same tree). Because every rank boundary is
a node of the G-leaf tree, both wires and every world size W | G evaluate the same expression, and the root
is rounded once. The kernel (``csrc/owner_allreduce.cu``) has exactly one float-touching function.

Transports (bits never depend on which one runs):
  * ``p2p``  -- symmetric memory over NVLink, ONE launch: barrier; one-shot (every rank reduces everything)
    or two-shot (each rank reduces the token slice it owns, then pulls the other owners' finished roots);
    barrier. Per-block flags over a small fixed grid, never one handshake per 1024 elements.
  * ``nccl`` -- any topology, multi-node included: all_to_all of each owner's operand slice, the SAME kernel
    in local mode over the received planes, all_gather of the finished roots. CUDA-graph capturable.

Rank uniformity: the flag, the branch thresholds and the block cap are agreed by a collective the first time a
group is seen (eagerly), and every launch-structure decision afterwards is a pure function of (n, world, C,
dtypes) and that agreed config. Each (transport, world, C, dtypes, branch, vec) class is admitted once by a
group-agreed bit-pattern compare against the legacy pik path on live operands before use.
"""

from __future__ import annotations

import functools
import logging
import os
import socket
import zlib

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

ENV_ON = "ISOEXEC_PIK_OWNER_AR"
ENV_ONESHOT_KB = "ISOEXEC_PIK_OWNER_ONESHOT_KB"
ENV_MAX_BLOCKS = "ISOEXEC_PIK_OWNER_MAX_BLOCKS"
ENV_LEAF_MIN_M = "ISOEXEC_PIK_OWNER_LEAF_MIN_M"

THREADS = 512
FLAG_CAP = 1024  # flag/epoch capacity in blocks (the grid is capped far below this)
_MAX_WORLD = 16

# One rank's wire payload (bytes) at or below which the one-shot branch runs. Both branches give the same
# bits; this is speed only. Measured on 2xH100 (world 2); 4 and 8 are placeholders until measured there.
_DEFAULT_ONESHOT_BYTES = {2: 1 << 20, 4: 256 << 10, 8: 128 << 10, 16: 64 << 10}
_DEFAULT_MAX_BLOCKS = {2: 132, 4: 32, 8: 32, 16: 32}  # measured: 132 best at world 2, 32 best at world 4
# Leaf wire (no local fold, bf16 leaves on the wire) from this many rows up; below it the legacy fp32 node wire
# and the legacy GEMM schedule are kept and only the barriers/cast move into the reduce launch. Default 1: the
# leaf wire at every M is what the Lamport owner kernel (with the fused add+RMSNorm epilogue) consumes, and the
# 27B generation benchmark measured it as the fastest configuration. The barrier-protocol bench that once
# favoured the node wire below M=1024 predates the Lamport kernel.
_DEFAULT_LEAF_MIN_M = 1


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


# On by default: the owner path (Lamport protocol, fused add+RMSNorm) is the production row-parallel site.
# ``ISOEXEC_PIK_OWNER_AR=0`` opts out (legacy pik site, same bits); the value is still agreed per TP group.
# Resolved from the environment at first use, not at import, so a value a process sets after importing
# this module (a launcher, a plugin, a test) applies.
_ON: bool | None = None


def enabled() -> bool:
    global _ON
    if _ON is None:
        _ON = _env_on(ENV_ON, "1")
    return _ON


def set_enabled(on: bool) -> None:
    """Test/bench hook. Takes effect for groups not yet agreed (see ``reset``); admission still applies."""
    global _ON
    _ON = bool(on)


# --------------------------------------------------------------------------------------------- extension


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    from isoexec.core.paths import pik_cache_dir

    here = os.path.dirname(os.path.abspath(__file__))
    # Built for the live device: a Hopper-only image refuses every launch on Blackwell, and the refusal
    # sends each row-parallel site to the legacy NCCL path. One build directory and module per arch.
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    build = pik_cache_dir() / f"ext_owner_sm{arch}"
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"pik_owner_allreduce_sm{arch}",
        sources=[os.path.join(here, "csrc", "owner_allreduce.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--fmad=false"],
        build_directory=str(build),
        verbose=bool(os.environ.get("PIK_VERBOSE")),
    )


def _launch(in_ptrs, tmp_ptrs, flag_ptrs, epoch, out, *, n, plane_stride, chunk, base, rank, max_blocks, mode,
            planes, in_bf16, vec, blocks):
    _ext().owner_allreduce(
        list(in_ptrs), list(tmp_ptrs), list(flag_ptrs), epoch, out, int(n), int(plane_stride), int(chunk),
        int(base), int(rank), int(max_blocks), int(mode), int(planes), bool(in_bf16), int(vec), int(blocks),
        THREADS,
    )


# --------------------------------------------------------------------------------- pure (rank-uniform) policy


def wire(plan, tp: int) -> tuple[int, torch.dtype]:
    """(planes per rank, wire dtype). Leaf wire whenever it costs no more bytes than the fp32 node (m <= 2):
    it removes the local fold launch. At m >= 4 the fp32 node is half the bytes of the leaves."""
    m = plan.leaves_per_rank(tp)
    if plan.bf16_leaves and m <= 2:
        return m, torch.bfloat16
    return 1, (torch.bfloat16 if (plan.bf16_leaves and m == 1) else torch.float32)


def root_dtype_for(plan, bias, out_dtype) -> torch.dtype:
    """The node the kernel stores. A bf16-leaf plan rounds the root before anything else touches it, so the
    root IS bf16; an fp32-leaf plan rounds it only when the cast is the very next operation."""
    if plan.bf16_leaves or (bias is None and out_dtype == torch.bfloat16):
        return torch.bfloat16
    return torch.float32


_ELT = {torch.bfloat16: 2, torch.float32: 4}


def vec_for(n: int) -> int:
    return 8 if n % 8 == 0 else 1


def _twoshot_mode() -> int:
    """3 = push (owner stores roots into every rank; default), 2 = pull. Read once per group in state()."""
    return 2 if os.environ.get("ISOEXEC_PIK_OWNER_PUSH", "1") == "0" else 3


def branch(n: int, world: int, planes: int, elt: int, oneshot_bytes: int) -> int:
    """1 = one-shot, 2 = two-shot. Speed only: both evaluate the same expression per element."""
    return 1 if n * planes * elt <= oneshot_bytes else 2


def chunk_for(n: int, world: int, vec: int) -> int:
    c = -(-n // world)
    return -(-c // vec) * vec


def grid_for(work_vecs: int, max_blocks: int) -> int:
    return max(1, min(max_blocks, -(-work_vecs // THREADS)))


# ------------------------------------------------------------------------------------- group agreement


def _group_key(group):
    try:
        return tuple(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001 -- default group, or an older torch
        return ("world", dist.get_world_size(group))


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


class _P2P:
    """Symmetric flags (fixed size) plus growable symmetric input planes and two-shot root staging."""

    def __init__(self, group, device):
        from torch.distributed import _symmetric_memory as symm_mem

        self.group = group if group is not None else dist.group.WORLD
        self.device = device
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        numel = 2 * FLAG_CAP * _MAX_WORLD
        self.flags = symm_mem.empty(numel, dtype=torch.int32, device=device)
        self.flags.zero_()
        from . import allreduce as _ar

        self.h_flags = _ar.rendezvous(self.flags, self.group)
        self.flag_ptrs = [self.h_flags.get_buffer(r, (numel,), torch.int32).data_ptr() for r in range(self.world)]
        self.epoch = torch.zeros(FLAG_CAP, dtype=torch.int32, device=device)
        # Input planes per staging SLOT: slot 0 serves every unpipelined call; a pipelined site alternates
        # slots so the GEMM of chunk i+1 can write while the reduce of chunk i still reads (see row_parallel).
        self.slots: dict = {}  # slot -> [(tensor, handle, ptrs), cap, captured]
        self.tmp = None
        self.tmp_ptrs: list = []
        self.tmp_cap = 0
        self._tmp_captured = False
        self.retired: list = []  # generations a captured graph still points at
        torch.cuda.synchronize(device)
        self.h_flags.barrier()  # every rank's zeroing landed before any first publish

    def slot_ptrs(self, slot: int) -> list:
        e = self.slots.get(slot)
        return [] if e is None else e[0][2]

    def _alloc(self, nbytes):
        from torch.distributed import _symmetric_memory as symm_mem

        from . import allreduce as _ar

        t = symm_mem.empty(nbytes, dtype=torch.uint8, device=self.device)
        h = _ar.rendezvous(t, self.group)
        ptrs = [h.get_buffer(r, (nbytes,), torch.uint8).data_ptr() for r in range(self.world)]
        assert all(p % 16 == 0 for p in ptrs), "symmetric buffers must be 16-byte aligned"
        return t, h, ptrs

    def ensure_in(self, nbytes: int, slot: int = 0):
        e = self.slots.get(slot)
        cap = 0 if e is None else e[1]
        if cap < nbytes:
            if _capturing():
                raise RuntimeError("pik owner all-reduce: input planes must grow during CUDA capture; warm eagerly")
            if e is not None and e[2]:
                self.retired.append(e[0])
            cap = max(nbytes, 2 * cap, 4 << 20)
            e = [self._alloc(cap), cap, False]
            self.slots[slot] = e
        if _capturing():
            e[2] = True
        return e[0][0]

    def ensure_tmp(self, nbytes: int):
        if self.tmp_cap < nbytes:
            if _capturing():
                raise RuntimeError("pik owner all-reduce: root staging must grow during CUDA capture; warm eagerly")
            if self.tmp is not None and self._tmp_captured:
                self.retired.append(self.tmp)
            cap = max(nbytes, 2 * self.tmp_cap, 4 << 20)
            self.tmp = self._alloc(cap)
            self.tmp_ptrs, self.tmp_cap, self._tmp_captured = self.tmp[2], cap, False
        if _capturing():
            self._tmp_captured = True
        return self.tmp[0]


class _State:
    def __init__(self, on, transport, oneshot_bytes, max_blocks, p2p=None):
        self.on = on
        self.transport = transport
        self.oneshot_bytes = oneshot_bytes
        self.max_blocks = max_blocks
        self.p2p = p2p


_STATES: dict = {}
_ADMIT: dict = {}
COUNTS = {"p2p_calls": 0, "nccl_calls": 0, "legacy_fallbacks": 0, "admitted": 0, "rejected": 0,
          "capture_skips": 0, "oneshot": 0, "twoshot": 0}


def reset() -> None:
    """Test hook: forget agreements and verdicts (pools of live groups are dropped too)."""
    _STATES.clear()
    _ADMIT.clear()
    for k in COUNTS:
        COUNTS[k] = 0


def counts() -> dict:
    return {**COUNTS, "enabled": enabled(), "groups": {str(k): (v.on, v.transport) for k, v in _STATES.items()},
            "verdicts": {str(k): v for k, v in _ADMIT.items()}}


def _say(msg: str, warn: bool = False) -> None:
    print(msg, flush=True)
    (logger.warning if warn else logger.info)(msg)


def state(group, device):
    """The agreed per-group state, or None when the owner path is off for this group.

    The first call per group is a collective every rank enters (whether or not its own flag is set), so a
    split flag disables the path everywhere instead of hanging. Inside a capture nothing is decided.
    """
    if not dist.is_initialized():
        return None
    if torch.device(device).type != "cuda":
        # The owner kernels are CUDA kernels; a CPU (gloo) group runs the legacy site. Rank-uniform: every
        # rank of a group holds its operands on the same device type.
        return None
    key = (_group_key(group), str(device))
    st = _STATES.get(key)
    if st is not None:
        return st if st.on else None
    if _capturing():
        COUNTS["capture_skips"] += 1
        return None
    from . import allreduce as _ar

    world = dist.get_world_size(group)
    oneshot = _env_int(ENV_ONESHOT_KB, -1)
    oneshot_bytes = oneshot << 10 if oneshot >= 0 else _DEFAULT_ONESHOT_BYTES.get(world, 64 << 10)
    max_blocks = max(1, min(FLAG_CAP, _env_int(ENV_MAX_BLOCKS, _DEFAULT_MAX_BLOCKS.get(world, 32))))
    host = zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF
    want_p2p = bool(_ar.p2p_available(group))
    push = _twoshot_mode()
    leaf_min_m = max(0, _env_int(ENV_LEAF_MIN_M, _DEFAULT_LEAF_MIN_M))
    node_below = _env_on("ISOEXEC_PIK_OWNER_NODE")  # below leaf_min_m: one-launch node wire instead of legacy
    mine = torch.tensor([int(enabled()) + 2 * push + 8 * int(node_below), oneshot_bytes, max_blocks, leaf_min_m, host, int(want_p2p)],
                        dtype=torch.int64, device=device)
    allv = torch.empty((world, mine.numel()), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(allv, mine, group=group)
    rows = allv.cpu().tolist()
    same_cfg = all(r[:4] == rows[0][:4] for r in rows)
    on = same_cfg and bool(rows[0][0] & 1) and world <= _MAX_WORLD and (world & (world - 1)) == 0
    if not same_cfg:
        _say(f"[ISOEXEC-PIK] owner all-reduce DISABLED for group {key[0]}: ranks disagree on "
             f"({ENV_ON}+push, {ENV_ONESHOT_KB}, {ENV_MAX_BLOCKS}, {ENV_LEAF_MIN_M}) = {[r[:4] for r in rows]}; set them identically "
             "on every rank of a TP group.", warn=True)
    transport = "nccl"
    p2p = None
    if on and all(r[5] for r in rows) and len({r[4] for r in rows}) == 1:
        ok = True
        try:
            p2p = _P2P(group, device)
        except Exception as e:  # noqa: BLE001 -- no symmetric memory here: NCCL transport, same bits
            ok = False
            _say(f"[ISOEXEC-PIK] owner all-reduce: symmetric memory unavailable ({e!r}); NCCL transport", True)
        if _ar._agree(ok, group, device):
            transport = "p2p"
        else:
            p2p = None
    st = _State(on, transport, oneshot_bytes, max_blocks, p2p)
    from .lane import p2p_max_bytes

    st.p2p_max_bytes = p2p_max_bytes(group) if transport == "p2p" else 0  # group-agreed (lane probe)
    st.twoshot = push
    st.single_host = len({r[4] for r in rows}) == 1  # pipeline depth model (pipeline.site_chunks)
    st.leaf_min_m = leaf_min_m
    st.node_below = node_below
    _STATES[key] = st
    if on:
        _say(f"[ISOEXEC-PIK] owner tree all-reduce ENABLED for group {key[0]} (world={world}, transport="
             f"{transport}, oneshot<={oneshot_bytes} B/rank, max_blocks={max_blocks}); each class is still "
             "bit-compared against the legacy path before use. Read pik.owner.counts().")
    return st if on else None


# ------------------------------------------------------------------------------------------ the transports


def reduce_local(operands, root_dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
    """The kernel's arithmetic over LOCAL operands, no communication: ``operands[r]`` is rank r's
    ``[C, n]`` wire planes. Used by the NCCL transport and by single-GPU world-size simulation."""
    world = len(operands)
    planes, n = operands[0].shape
    dt = operands[0].dtype
    for t in operands:
        assert t.shape == (planes, n) and t.dtype == dt and t.is_contiguous()
    out = torch.empty(n, dtype=root_dtype, device=operands[0].device) if out is None else out
    if n == 0:
        return out
    vec = vec_for(n)
    _launch([t.data_ptr() for t in operands], [], [], None, out, n=n, plane_stride=n, chunk=0, base=0, rank=0,
            max_blocks=0, mode=0, planes=planes, in_bf16=dt == torch.bfloat16, vec=vec,
            blocks=max(1, min(1024, -(-(n // vec) // THREADS))))
    return out


def _run_nccl(staged: torch.Tensor, group, root_dtype: torch.dtype, out=None) -> torch.Tensor:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    planes, n = staged.shape
    vec = vec_for(n)
    chunk = chunk_for(n, world, vec)
    npad = chunk * world
    src = staged
    if npad != n:
        src = torch.zeros((planes, npad), dtype=staged.dtype, device=staged.device)
        src[:, :n] = staged
    # One all_to_all per plane. Batching both planes into one call needs a destination-major copy of all
    # planes (2 S bytes of HBM traffic), which measured ~5% slower on a TP4 trainer forward (M=11.5k) than
    # the per-plane calls. A copy-free batch needs grouped send/recv (or a destination-major leaf GEMM
    # epilogue).
    recv = torch.empty((planes, world, chunk), dtype=staged.dtype, device=staged.device)
    for j in range(planes):
        dist.all_to_all_single(recv[j].view(-1), src[j], group=group)  # recv[j][r] = rank r's plane j, my slice
    mine = torch.empty(chunk, dtype=root_dtype, device=staged.device)
    elt = staged.element_size()
    _launch([recv.data_ptr() + r * chunk * elt for r in range(world)], [], [], None, mine, n=chunk,
            plane_stride=world * chunk, chunk=0, base=0, rank=0, max_blocks=0, mode=0, planes=planes,
            in_bf16=staged.dtype == torch.bfloat16, vec=vec, blocks=max(1, min(1024, -(-(chunk // vec) // THREADS))))
    direct = out is not None and npad == n
    full = out.view(-1) if direct else torch.empty(npad, dtype=root_dtype, device=staged.device)
    dist.all_gather_into_tensor(full, mine, group=group)
    COUNTS["nccl_calls"] += 1
    if out is not None and not direct:
        out.view(-1).copy_(full[:n])
        return out.view(-1)
    return full[:n]


def _run_p2p(st: _State, staged: torch.Tensor, root_dtype: torch.dtype, mode: int, out=None,
             slot: int = 0) -> torch.Tensor:
    from .lane import fence

    fence(staged.device)  # no-op on a lane: its own tail is not waited on
    p = st.p2p
    planes, n = staged.shape
    elt = staged.element_size()
    inp = p.ensure_in(planes * n * elt, slot)
    if staged.data_ptr() != inp.data_ptr():
        inp[: planes * n * elt].view(staged.dtype).view(planes, n).copy_(staged)  # caller did not stage
    vec = vec_for(n)
    out = torch.empty(n, dtype=root_dtype, device=staged.device) if out is None else out
    if mode == 1:
        chunk, tmp_ptrs = 0, []
        blocks = grid_for(n // vec, st.max_blocks)
        COUNTS["oneshot"] += 1
    else:
        chunk = chunk_for(n, p.world, vec)
        p.ensure_tmp(chunk * p.world * _ELT[root_dtype])
        tmp_ptrs = p.tmp_ptrs
        blocks = grid_for(chunk // vec, st.max_blocks)
        COUNTS["twoshot"] += 1
    _launch(p.slot_ptrs(slot), tmp_ptrs, p.flag_ptrs, p.epoch, out, n=n, plane_stride=n, chunk=chunk, base=0,
            rank=p.rank, max_blocks=FLAG_CAP, mode=mode, planes=planes, in_bf16=staged.dtype == torch.bfloat16,
            vec=vec, blocks=blocks)
    COUNTS["p2p_calls"] += 1
    return out


# ------------------------------------------------------------------------------ legacy reference + admission


def legacy(staged: torch.Tensor, group, root_dtype: torch.dtype) -> torch.Tensor:
    """What pik computed before this module: local fold of the planes to the rank's fp32 node, the legacy tree
    all-reduce over nodes, then the root cast. The admission reference and the refusal fallback."""
    import triton

    from . import allreduce as _ar
    from .codegen import tree_reduce_kernel

    planes, n = staged.shape
    if planes > 1:
        node = torch.empty(n, dtype=torch.float32, device=staged.device)
        tree_reduce_kernel(planes, in_bf16=staged.dtype == torch.bfloat16)[(triton.cdiv(n, 1024),)](
            staged, node, n, staged.stride(0), BLOCK=1024, num_warps=4
        )
    else:
        node = staged[0].clone()
    if node.dtype == torch.bfloat16:
        res = _ar._tree_all_reduce_legacy(node, group, None, None, root_dtype)
    else:
        res = _ar._tree_all_reduce_legacy(node, group, None, None, None)
    return res.to(root_dtype)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _class_key(st, world, planes, dt, root_dtype, mode, vec):
    return (st.transport, world, planes, str(dt), str(root_dtype), mode, vec)


def _admit(key, st, staged, group, root_dtype, mode) -> bool:
    from . import allreduce as _ar

    ref = legacy(staged, group, root_dtype).clone()
    why = ""
    try:
        got = (_run_p2p(st, staged, root_dtype, mode) if st.transport == "p2p"
               else _run_nccl(staged, group, root_dtype)).clone()
        ok = torch.equal(_bits(ref), _bits(got))
        if not ok:
            why = f"{int((_bits(ref) != _bits(got)).sum())}/{ref.numel()} elements differ"
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"raised {e!r}"
    agreed = _ar._agree(ok, group, staged.device)
    _ADMIT[key] = agreed
    COUNTS["admitted" if agreed else "rejected"] += 1
    if not agreed:
        _say(f"[ISOEXEC-PIK] WARNING: owner all-reduce REFUSED for class {key}: {why or 'a peer refused'}. "
             "The result is unaffected (the legacy path serves this class).", warn=True)
    elif COUNTS["admitted"] == 1:
        _say(f"[ISOEXEC-PIK] owner all-reduce ADMITTED (first class {key}): bit patterns equal the legacy "
             "path on live operands, agreed across the group.")
    return agreed


def _resolve(flat: torch.Tensor, group, root_dtype: torch.dtype, st):
    """(mode, admitted) for this call's class; admits the class on first sight (eagerly: a host sync)."""
    if st.transport != "p2p" and _capturing():
        # NCCL all_to_all inside a CUDA graph hung on replay in the 2xH100 bench (M=32); never capture it.
        COUNTS["capture_skips"] += 1
        return 0, False
    world = dist.get_world_size(group)
    planes, n = flat.shape
    vec = vec_for(n)
    mode = branch(n, world, planes, flat.element_size(), st.oneshot_bytes) if st.transport == "p2p" else 0
    if mode == 2:
        mode = getattr(st, "twoshot", 2)
    key = _class_key(st, world, planes, flat.dtype, root_dtype, mode, vec)
    verdict = _ADMIT.get(key)
    if verdict is None:
        if _capturing():
            COUNTS["capture_skips"] += 1
            verdict = False
        else:
            verdict = _admit(key, st, flat, group, root_dtype, mode)
    return mode, verdict


def admitted_mode(flat: torch.Tensor, group, root_dtype: torch.dtype, st):
    """The P2P branch for this call's class if that class is already admitted, else None. No side effects and
    no collective (what may run on a side stream needs no admission)."""
    if st is None or st.transport != "p2p":
        return None
    world = dist.get_world_size(group)
    planes, n = flat.shape
    vec = vec_for(n)
    mode = branch(n, world, planes, flat.element_size(), st.oneshot_bytes)
    if mode == 2:
        mode = getattr(st, "twoshot", 2)
    return mode if _ADMIT.get(_class_key(st, world, planes, flat.dtype, root_dtype, mode, vec)) else None


def _execute(st, flat: torch.Tensor, group, root_dtype: torch.dtype, mode: int, out=None, slot: int = 0):
    """The admitted owner reduce of ``flat`` ([C, n] planes) on the CURRENT stream; ``out``: [n] or None."""
    if st.transport == "p2p":
        from . import allreduce as _ar

        try:
            return _run_p2p(st, flat, root_dtype, mode, out=out, slot=slot)
        except _ar.SymmetricSetupSkew:
            # Raised on every rank of the group (the check is collective): the group leaves P2P together and
            # takes the NCCL transport (same tree, same bits) from here on.
            st.transport = "nccl"
            COUNTS["legacy_fallbacks"] += 1
    return _run_nccl(flat, group, root_dtype, out=out)


def all_reduce(staged: torch.Tensor, group, root_dtype: torch.dtype, st=None, out=None) -> torch.Tensor:
    """Reduce ``staged`` ([C, ...] wire planes, this rank's) with the fixed tree; returns the root, shaped like
    one plane. Falls back to the legacy path (same bits) when the class is refused or unadmitted in capture."""
    shape = staged.shape[1:]
    planes = staged.shape[0]
    flat = staged.reshape(planes, -1)
    st = st if st is not None else state(group, staged.device)
    if st is None:
        COUNTS["legacy_fallbacks"] += 1
        return legacy(flat, group, root_dtype).view(shape)
    mode, verdict = _resolve(flat, group, root_dtype, st)
    if not verdict:
        COUNTS["legacy_fallbacks"] += 1
        return legacy(flat, group, root_dtype).view(shape)
    res = _execute(st, flat, group, root_dtype, mode, out=None if out is None else out.view(-1))
    return res.view(shape)


def stage(st, planes: int, rows: int, cols: int, dtype: torch.dtype, device, slot: int = 0,
          fenced: bool = True) -> torch.Tensor:
    """Where the producer GEMM should write this rank's wire planes: the symmetric input itself on p2p (no
    copy-in), a fresh tensor otherwise. ``slot`` selects one of the pipelined staging buffers; ``fenced``
    (default) first orders the caller after the comm lanes -- a pipelined site orders its slots itself."""
    if st is not None and st.transport == "p2p" and not (rows * cols == 0):
        if fenced:
            from .lane import fence

            fence(torch.device(device))  # the producer writes this buffer on the caller's stream
        nbytes = planes * rows * cols * _ELT[dtype]
        try:
            inp = st.p2p.ensure_in(nbytes, slot)
        except RuntimeError:
            return torch.empty((planes, rows, cols), dtype=dtype, device=device)
        return inp[:nbytes].view(dtype).view(planes, rows, cols)
    return torch.empty((planes, rows, cols), dtype=dtype, device=device)


def _produce(x2, w, staged, *, plan, planes, m_leaves, tp_size, tp_rank, k_full):
    from .gemm import ti_gemm, ti_gemm_leaves

    if plan.bf16_leaves and planes == m_leaves:
        ti_gemm_leaves(x2, w, plan=plan, tp_size=tp_size, k_full=k_full, out=staged)
    else:
        ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, out=staged[0])


def row_parallel(x2, w, bias, *, plan, tp_size, tp_rank, k_full, group, out_dtype):
    """The whole row-parallel site: wire planes straight into the transport's input, then one all-reduce.
    Returns None when the owner path is off for this group (the caller runs the legacy site)."""
    if tp_size < 2 or x2.shape[0] == 0:
        return None
    st = state(group, x2.device)
    if st is None:
        return None
    from . import lane as _lane
    from . import pipeline as _pipe

    planes, wdt = wire(plan, tp_size)
    m_leaves = plan.leaves_per_rank(tp_size)
    M, N = x2.shape[0], w.shape[0]
    if M < getattr(st, "leaf_min_m", 0):
        if not st.node_below:
            return None  # the legacy site measured faster here (world-4 site bench); same bits either way
        planes, wdt = 1, (wdt if planes == 1 else torch.float32)  # legacy node wire + GEMM schedule, one launch
    root = root_dtype_for(plan, bias, out_dtype)
    bounds = [(0, M)]
    if not _capturing() and _lane.overlap_agreed(group, x2.device):
        bounds = _pipe.pipeline_bounds(M, _pipe.site_chunks(st, M, N, x2.shape[1], tp_size, planes, wdt, root),
                                       _lane.overlap_min_m())
    if len(bounds) > 1:
        full = _pipe.row_parallel_pipelined(x2, w, bounds, st, group, root, plan=plan, planes=planes, wdt=wdt,
                                       m_leaves=m_leaves, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full)
    else:
        _pipe.PIPE_COUNTS["serial_sites"] += 1
        staged = stage(st, planes, M, N, wdt, x2.device)
        _produce(x2, w, staged, plan=plan, planes=planes, m_leaves=m_leaves, tp_size=tp_size, tp_rank=tp_rank,
                 k_full=k_full)
        st_sized = _pipe.pick_state(st, staged.numel() * staged.element_size())  # P2P small, NCCL owner large
        full = lamport_all_reduce(staged, group, root, st=st_sized)  # p2p only; None outside its domain
        if full is None:
            full = all_reduce(staged, group, root, st=st_sized)
    if bias is not None:
        full = full + bias.to(full.dtype)  # after the reduction, exactly once (same as the legacy site)
    return full.to(out_dtype)


def tree_all_reduce_routed(partial: torch.Tensor, group, out, root_dtype):
    """``pik.allreduce.tree_all_reduce`` on the p2p transport, routed here when the owner path is on.
    Returns None to leave the call to the legacy body."""
    if not partial.is_cuda or partial.numel() == 0:
        return None
    st = state(group, partial.device)
    if st is None or st.transport != "p2p" or not _env_on("ISOEXEC_PIK_OWNER_ROUTE"):
        return None  # generic callers carry fp32 nodes, where the legacy kernel measured faster
    rd = root_dtype if root_dtype is not None else partial.dtype
    if out is not None and (out.dtype != rd or not out.is_contiguous() or out.data_ptr() % 32):
        out = None
    return all_reduce(partial.contiguous().view(1, *partial.shape), group, rd, st=st, out=out)


# =============================================================================================================
# Lamport protocol + fused residual-add/RMSNorm epilogue (csrc/owner_lamport.cu)
#
# The same tree, the same root round, the same transport topology (p2p only); what changes is how a consumer
# learns its operands have landed: the data is the flag (receive buffers pre-filled with an all-ones NaN
# sentinel, three rotating generations advanced in-kernel), so a site is ONE launch with no barrier round trips,
# launched with programmatic dependent launch. Optionally the kernel also runs the residual join + RMSNorm of
# the norm that consumes the row-parallel output (norms recipe v1, exactly ``norms_sm90.add_rms_fwd``'s per-row
# program), so the decode site "o_proj/fc2 reduction -> h += root -> norm(h)" is one kernel instead of two.
# Every (protocol, world, planes, dtypes, branch, epilogue) class is admitted by a group-agreed bit compare
# against the legacy path (+ the bound unfused add+norm) on live operands before use, like the classes above.
# =============================================================================================================

ENV_LAMPORT = "ISOEXEC_PIK_OWNER_LAMPORT"
ENV_LAMPORT_KB = "ISOEXEC_PIK_OWNER_LAMPORT_KB"
ENV_LAMPORT_BLOCKS = "ISOEXEC_PIK_OWNER_LAMPORT_BLOCKS"
ENV_LAMPORT_ONESHOT_KB = "ISOEXEC_PIK_OWNER_LAMPORT_ONESHOT_KB"
ENV_ARNORM = "ISOEXEC_PIK_OWNER_ARNORM"
ENV_PDL = "ISOEXEC_PIK_OWNER_PDL"

_LAMPORT_THREADS = 512
_DEFAULT_LAMPORT_KB = 16 << 10  # one generation of the wire receive area: covers M <= 409 at N=5120, 2 bf16 planes
_DEFAULT_LAMPORT_ONESHOT_BYTES = {2: 512 << 10, 4: 128 << 10, 8: 64 << 10, 16: 32 << 10}
_NORM_IMPLS = frozenset({"sm90_add_rms_v1"})  # unfused programs the fused epilogue reproduces (recipe v1, L=32)


def fuses_norm_impl(impl) -> bool:
    """Whether the fused add+norm epilogue reproduces the bound unfused norm program ``impl``."""
    return impl in _NORM_IMPLS

LAMPORT_COUNTS = {"calls": 0, "fused_norm": 0, "deferred": 0, "fallbacks": 0, "admitted": 0, "rejected": 0,
                  "capture_skips": 0, "oneshot": 0, "twoshot": 0}


@functools.lru_cache(maxsize=1)
def _lext():
    from torch.utils.cpp_extension import load

    from isoexec.core.paths import pik_cache_dir

    here = os.path.dirname(os.path.abspath(__file__))
    build = pik_cache_dir() / "ext_owner_lamport"
    build.mkdir(parents=True, exist_ok=True)
    # sm_90 SASS plus compute_90 PTX: the protocol and arithmetic are arch-neutral, and sm_100 JITs the PTX.
    return load(
        name="pik_owner_lamport",
        sources=[os.path.join(here, "csrc", "owner_lamport.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=[sm_90,compute_90]", "--fmad=false",
                           "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
        build_directory=str(build),
        verbose=bool(os.environ.get("PIK_VERBOSE")),
    )


def lamport_enabled() -> bool:
    return enabled() and _env_on(ENV_LAMPORT, "1")


def arnorm_enabled() -> bool:
    return lamport_enabled() and _env_on(ENV_ARNORM, "1")


class _LamportBuffers:
    """Three generations of the wire receive area (recv1) and of the root/h receive area (recv2), symmetric,
    pre-filled with the sentinel; plus the LOCAL control words (generation counter, retired-block counter, the
    extent each generation last used)."""

    def __init__(self, group, device, gen_bytes: int):
        """Local half: allocate and sentinel-fill (no collective; may fail on one rank alone)."""
        from torch.distributed import _symmetric_memory as symm_mem

        self.group = group if group is not None else dist.group.WORLD
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.gen1 = self.gen2 = -(-int(gen_bytes) // 16) * 16
        self._local = []
        for nbytes in (3 * self.gen1, 3 * self.gen2):
            t = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
            t.fill_(0xFF)
            self._local.append(t)
        self.ctl = torch.zeros(8, dtype=torch.int64, device=device)
        torch.cuda.synchronize(device)

    def connect(self) -> None:
        """Collective half: rendezvous both areas and fence the sentinel fill. Every rank of the group must call
        it (lamport_state agrees on the local half first)."""
        from . import allreduce as _ar

        self._keep = []
        ptrs = []
        for t in self._local:
            h = _ar.rendezvous(t, self.group)
            p = [h.get_buffer(r, (t.numel(),), torch.uint8).data_ptr() for r in range(self.world)]
            self._keep.append((t, h))
            ptrs.append(p)
        self.recv1, self.recv2 = ptrs
        self._keep[0][1].barrier()  # every rank's sentinel fill landed before any first push
        if not all(x % 16 == 0 for p in ptrs for x in p):
            raise RuntimeError("symmetric buffers must be 16-byte aligned")


class _LamportCfg:
    def __init__(self, on, arnorm, gen_bytes, blocks, oneshot_bytes, pdl, bufs):
        self.on, self.arnorm, self.gen_bytes, self.blocks = on, arnorm, gen_bytes, blocks
        self.oneshot_bytes, self.pdl, self.bufs = oneshot_bytes, pdl, bufs


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_lamport_blocks(device) -> int:
    """Half the SMs (the other half stays free for concurrent communication kernels; see lamport_state)."""
    return max(1, _sm_count(device) // 2)


def lamport_state(st, group, device):
    """The group-agreed Lamport configuration hung off an owner state ``st`` (p2p transport only), or None.
    First call per group is a collective (eager); inside a capture nothing is decided."""
    if st is None or st.transport != "p2p":
        return None
    cfg = getattr(st, "lamport", None)
    if cfg is not None:
        return cfg if cfg.on else None
    if _capturing():
        LAMPORT_COUNTS["capture_skips"] += 1
        return None
    world = dist.get_world_size(group)
    gen_bytes = max(0, _env_int(ENV_LAMPORT_KB, _DEFAULT_LAMPORT_KB)) << 10
    ob = _env_int(ENV_LAMPORT_ONESHOT_KB, -1)
    oneshot_bytes = ob << 10 if ob >= 0 else _DEFAULT_LAMPORT_ONESHOT_BYTES.get(world, 32 << 10)
    # Blocks spin on rows written by peers and by other blocks of the same grid, so the whole grid must be
    # resident at once: at most one 512-thread block per SM, and by default at most half the SMs, so kernels
    # running concurrently on other streams that themselves wait on peers (NCCL's, at most one CTA per
    # channel) cannot keep part of the grid queued. Rank-uniform: the minimum over the group is used.
    blocks = max(1, min(_sm_count(device), _env_int(ENV_LAMPORT_BLOCKS, _default_lamport_blocks(device))))
    pdl = _env_on(ENV_PDL, "1")
    mine = torch.tensor([int(lamport_enabled()), int(arnorm_enabled()), gen_bytes, oneshot_bytes, int(pdl), blocks],
                        dtype=torch.int64, device=device)
    allv = torch.empty((world, mine.numel()), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(allv, mine, group=group)
    rows = allv.cpu().tolist()
    same = all(r[:5] == rows[0][:5] for r in rows)
    on = same and bool(rows[0][0]) and gen_bytes > 0 and world <= _MAX_WORLD
    if not same:
        _say(f"[ISOEXEC-PIK] owner Lamport all-reduce DISABLED for group {_group_key(group)}: ranks disagree on "
             f"({ENV_LAMPORT}, {ENV_ARNORM}, {ENV_LAMPORT_KB}, {ENV_LAMPORT_ONESHOT_KB}, {ENV_PDL}) = "
             f"{[r[:5] for r in rows]}; set them identically on every rank of a TP group.", warn=True)
    bufs = None
    if on:
        # Three steps, each agreed before the next, so no rank enters a collective (rendezvous, barrier) its
        # peers skip: build the kernel, allocate the buffers, rendezvous them. Any failure anywhere keeps the
        # barrier protocol on every rank (it serves every class, same bits).
        from . import allreduce as _ar

        def _step(what, fn):
            try:
                return fn(), True
            except Exception as e:  # noqa: BLE001
                _say(f"[ISOEXEC-PIK] owner Lamport all-reduce unavailable ({what}: {e!r}); barrier protocol kept",
                     True)
                return None, False

        _, ok = _step("extension build", _lext)
        on = _ar._agree(ok, group, device)
        if on:
            bufs, ok = _step("buffer allocation", lambda: _LamportBuffers(group, device, gen_bytes))
            on = _ar._agree(ok, group, device)
        if on:
            _, ok = _step("rendezvous", bufs.connect)
            on = _ar._agree(ok, group, device)
        if not on:
            bufs = None
    cfg = _LamportCfg(on, on and bool(rows[0][1]), gen_bytes, min(r[5] for r in rows), oneshot_bytes,
                      bool(rows[0][4]), bufs)
    st.lamport = cfg
    if on:
        _say(f"[ISOEXEC-PIK] owner Lamport all-reduce ENABLED for group {_group_key(group)} (world={world}, "
             f"generation={gen_bytes} B, oneshot<={oneshot_bytes} B/rank, blocks<={cfg.blocks}, pdl={cfg.pdl}, "
             f"fused add+norm={cfg.arnorm}); each class is bit-compared against the legacy path before use.")
    return cfg if on else None


def _lamport_mode(cfg, world: int, M: int, N: int, planes: int, elt: int, root_elt: int) -> int:
    """1 = one-shot, 2 = two-shot, 0 = does not fit one generation (the barrier owner path serves it)."""
    wire = M * planes * N * elt
    slot = lambda m: -(-(m * planes * N * elt) // 16) * 16  # noqa: E731
    if wire <= cfg.oneshot_bytes and world * slot(M) <= cfg.bufs.gen1:
        return 1
    if world * slot(-(-M // world)) <= cfg.bufs.gen1 and M * N * root_elt <= cfg.bufs.gen2:
        return 2
    return 0


def _lamport_grid(cfg, mode: int, M: int, N: int, planes: int, world: int) -> int:
    """Rank-uniform grid: one block per row of the larger row pass (the kernel packs rows per block when the
    grid is smaller), and one wire vector per thread in the push phase; capped at one block per SM."""
    own = M if mode == 1 else -(-M // world)
    rows = max(own, M - own if mode == 2 else 0)
    push = -(-(M * planes * (N // 8)) // _LAMPORT_THREADS)
    return max(1, min(cfg.blocks, max(rows, push, 8)))


def _run_lamport(cfg, staged, root_dtype, mode, *, resid=None, weight=None, eps=0.0, norm_mode=0, out=None,
                 y=None):
    from .lane import fence

    fence(staged.device)  # collectives on a comm lane retire first: one collective at a time per group
    b = cfg.bufs
    planes, M, N = staged.shape
    out = torch.empty((M, N), dtype=root_dtype, device=staged.device) if out is None else out
    if norm_mode and y is None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=staged.device)
    _lext().lamport_allreduce(
        [staged.data_ptr()], b.recv1, b.recv2, b.ctl, resid, weight, out, y, M, N, M * N, b.gen1, b.gen2, b.rank,
        b.world, mode, planes, staged.dtype == torch.bfloat16, int(norm_mode), float(eps),
        _lamport_grid(cfg, mode, M, N, planes, b.world), _LAMPORT_THREADS, cfg.pdl,
    )
    LAMPORT_COUNTS["calls"] += 1
    LAMPORT_COUNTS["oneshot" if mode == 1 else "twoshot"] += 1
    return out, y


def lamport_local(operands, root_dtype, *, resid=None, weight=None, eps=0.0, norm_mode=0):
    """The Lamport kernel's arithmetic over LOCAL operands (no protocol): ``operands[r]`` is rank r's
    ``[C, M, N]`` wire planes. Single-GPU world-size simulation of the fused epilogue. Returns (root|h, y)."""
    W = len(operands)
    planes, M, N = operands[0].shape
    for t in operands:
        assert t.shape == (planes, M, N) and t.dtype == operands[0].dtype and t.is_contiguous()
    out = torch.empty((M, N), dtype=root_dtype, device=operands[0].device)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=operands[0].device) if norm_mode else None
    if M == 0:
        return out, y
    _lext().lamport_allreduce(
        [t.data_ptr() for t in operands], [], [], None, resid, weight, out, y, M, N, M * N, 0, 0, 0, W, 0, planes,
        operands[0].dtype == torch.bfloat16, int(norm_mode), float(eps),
        max(1, min(1024, M)), _LAMPORT_THREADS, False,
    )
    return out, y


def _lamport_admit(key, cfg, st, staged, group, root_dtype, mode, ref_norm, norm_args) -> bool:
    """Group-agreed bit compare of the Lamport class against the legacy tree (+ the bound unfused add+norm)."""
    from . import allreduce as _ar

    planes, M, N = staged.shape
    ref = legacy(staged.reshape(planes, -1), group, root_dtype).view(M, N).clone()
    refs = [ref]
    if ref_norm is not None:
        resid, weight, eps, _ = norm_args
        ry, rh = ref_norm(ref, resid, weight, eps)
        refs = [rh.clone(), ry.clone()]
    why = ""
    try:
        if ref_norm is not None:
            resid, weight, eps, nm = norm_args
            h, yy = _run_lamport(cfg, staged, root_dtype, mode, resid=resid, weight=weight, eps=eps, norm_mode=nm)
            got = [h.clone(), yy.clone()]
        else:
            got = [_run_lamport(cfg, staged, root_dtype, mode)[0].clone()]
        bad = [int((_bits(a) != _bits(b)).sum()) for a, b in zip(refs, got)]
        ok = not any(bad)
        if not ok:
            why = f"{bad} elements differ of {ref.numel()}"
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"raised {e!r}"
    agreed = _ar._agree(ok, group, staged.device)
    _ADMIT[key] = agreed
    LAMPORT_COUNTS["admitted" if agreed else "rejected"] += 1
    if not agreed:
        _say(f"[ISOEXEC-PIK] WARNING: owner Lamport class REFUSED {key}: {why or 'a peer refused'}. The result is "
             "unaffected (the barrier owner path / unfused norm serve this class).", warn=True)
    else:
        _say(f"[ISOEXEC-PIK] owner Lamport all-reduce ADMITTED class {key} (live M={staged.shape[1]}): bit patterns "
             "equal the legacy path on live operands, agreed across the group.")
    return agreed


def _lamport_key(cfg, staged, root_dtype, mode, norm_args, impl):
    nm = 0 if norm_args is None else norm_args[3]
    wdt = None if norm_args is None else str(norm_args[1].dtype)
    return ("lamport", cfg.bufs.world, staged.shape[0], str(staged.dtype), str(root_dtype), mode, nm, wdt, impl,
            cfg.pdl)


def _lamport_verdict(cfg, st, staged, group, root_dtype, mode, ref_norm=None, norm_args=None, impl=None):
    key = _lamport_key(cfg, staged, root_dtype, mode, norm_args, impl)
    verdict = _ADMIT.get(key)
    if verdict is None:
        if _capturing():
            LAMPORT_COUNTS["capture_skips"] += 1
            return False
        verdict = _lamport_admit(key, cfg, st, staged, group, root_dtype, mode, ref_norm, norm_args)
    return verdict


def _lamport_preadmit(cfg, st, staged, group, root_dtype, ref_norm=None, norm_args=None, impl=None) -> None:
    """Admit every Lamport branch of this class on the first EAGER call, whatever its size, so a runtime that
    captures its graphs without an eager call at each size still gets the protocol inside them. The operands
    are this call's live rows (cycled when the call has fewer rows than a branch needs); rank-uniform: every
    rank sees the same (M, N, planes) and runs the same admissions, each a group-agreed compare."""
    if _capturing():
        return
    planes, M, N = staged.shape
    world, elt = cfg.bufs.world, staged.element_size()
    per_row = planes * N * elt
    one = max(1, min(cfg.oneshot_bytes // per_row, 64))
    for m in (one, max(one + 1, 2 * world + 1)):
        mode = _lamport_mode(cfg, world, m, N, planes, elt, _ELT[root_dtype])
        if mode == 0 or _ADMIT.get(_lamport_key(cfg, staged, root_dtype, mode, norm_args, impl)) is not None:
            continue
        idx = torch.arange(m, device=staged.device) % M
        sub = staged.index_select(1, idx).contiguous()
        sub_args = None
        if norm_args is not None:
            resid, weight, eps, nm = norm_args
            sub_args = (resid.index_select(0, idx).contiguous(), weight, eps, nm)
        _lamport_verdict(cfg, st, sub, group, root_dtype, mode, ref_norm=ref_norm, norm_args=sub_args, impl=impl)


def lamport_all_reduce(staged: torch.Tensor, group, root_dtype: torch.dtype, st=None, out=None):
    """``all_reduce`` of ``staged`` ([C, M, N] wire planes) by the Lamport kernel, or None when it does not
    apply (not p2p, off, too large for one generation, refused, or unadmitted inside a capture)."""
    if staged.dim() != 3 or staged.shape[2] % 8 or not staged.is_contiguous():
        return None
    st = st if st is not None else state(group, staged.device)
    cfg = lamport_state(st, group, staged.device)
    if cfg is None:
        return None
    planes, M, N = staged.shape
    mode = _lamport_mode(cfg, cfg.bufs.world, M, N, planes, staged.element_size(), _ELT[root_dtype])
    if mode == 0 or M == 0:
        return None
    if not _lamport_verdict(cfg, st, staged, group, root_dtype, mode):
        return None
    return _run_lamport(cfg, staged, root_dtype, mode, out=out)[0]


# ------------------------------------------------------------------------- deferred root -> fused add + norm


class RootHandoff:
    """One row-parallel site's reduction, deferred to the add+norm that consumes its output.

    The fusion pass (``compiler.realization.fusion``) creates one per (sublayer whose output IS its row-parallel
    output, pre-norm that absorbs the residual join of that output) pair, proven on the captured graph: the
    sublayer's output has no other consumer. The row-parallel binding then stages its wire planes and returns a
    placeholder of the root's shape; the add+norm binding, called with exactly that placeholder, runs the
    reduction with the norm fused (or, outside the fused domain, the plain reduction into the placeholder and
    then the bound unfused add+norm -- the same program). Anything else is a wiring error and raises."""

    __slots__ = ("pending", "site")

    def __init__(self, site: str = ""):
        self.pending = None
        self.site = site

    def take(self, x):
        p = self.pending
        if p is None:
            return None
        self.pending = None
        if p["out"] is not x:
            raise RuntimeError(f"{self.site}: a deferred row-parallel root was not consumed by its own add+norm "
                               "(the fusion pass proved a single consumer; the graph changed under it)")
        return p


def row_parallel_deferred(x2, w, bias, *, plan, tp_size, tp_rank, k_full, group, out_dtype, handoff):
    """``row_parallel`` that leaves the reduction to ``handoff``'s add+norm. Returns the placeholder root, or
    None when the deferral does not apply (the caller runs the ordinary site, same bits)."""
    # A pending root still held here belongs to a forward that was abandoned between this site and its norm
    # (an exception); it is dropped whether or not this call defers (its placeholder never reached anything
    # but that norm). The caller (tp_tree.row_parallel_handoff) also clears it when this returns None.
    handoff.pending = None
    if tp_size < 2 or x2.shape[0] == 0 or bias is not None or out_dtype != torch.bfloat16 or not arnorm_enabled():
        return None
    st = state(group, x2.device)
    cfg = lamport_state(st, group, x2.device)
    if cfg is None or not cfg.arnorm:
        return None
    planes, wdt = wire(plan, tp_size)
    root = root_dtype_for(plan, None, out_dtype)
    M, N = x2.shape[0], w.shape[0]
    if root != torch.bfloat16 or N % 256 or N * 2 > 48 * 1024:  # recipe-v1 lanes; one row in 48 KiB of smem
        return None
    # mode 0 (larger than one generation) still defers: the add+norm half then runs the barrier reduction and
    # the unfused add+norm (the same program), and an eager call of any size admits the Lamport branches.
    mode = _lamport_mode(cfg, cfg.bufs.world, M, N, planes, _ELT[wdt], 2)
    from .gemm import ti_gemm, ti_gemm_leaves

    m_leaves = plan.leaves_per_rank(tp_size)
    staged = stage(st, planes, M, N, wdt, x2.device)
    if plan.bf16_leaves and planes == m_leaves:
        ti_gemm_leaves(x2, w, plan=plan, tp_size=tp_size, k_full=k_full, out=staged)
    else:
        ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, out=staged[0])
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x2.device)
    handoff.pending = {"out": out, "staged": staged, "st": st, "cfg": cfg, "group": group, "mode": mode}
    LAMPORT_COUNTS["deferred"] += 1
    return out


def add_norm_deferred(x, residual, weight, eps, *, norm_mode: int, impl: str, unfused, handoff):
    """The add+norm half of a ``RootHandoff`` pair. ``unfused(x, residual, weight, eps) -> (y, h)`` is the bound
    unfused program (the admission reference and the fallback). Returns (y, h)."""
    p = handoff.take(x)
    if p is None:
        return unfused(x, residual, weight, eps)
    staged, st, cfg, group, mode = p["staged"], p["st"], p["cfg"], p["group"], p["mode"]
    N = x.shape[-1]
    # Only shapes, dtypes, the bound program and grad mode decide (the same on every rank): fusing is a
    # collective decision (admission, and the fused kernel's protocol differs from the plain one), so a
    # rank-local fact such as an operand's alignment must not. Misaligned operands are copied instead.
    fusable = (
        fuses_norm_impl(impl) and residual.dtype == torch.bfloat16 and residual.shape == x.shape
        and residual.dim() == 2 and weight is not None
        and weight.dtype in (torch.bfloat16, torch.float32) and weight.numel() == N
        and not (torch.is_grad_enabled() and (residual.requires_grad or weight.requires_grad))
    )
    if fusable:
        if residual.stride(-1) != 1 or residual.stride(0) % 8 or residual.data_ptr() % 16:
            residual = residual.clone(memory_format=torch.contiguous_format)
        if not weight.is_contiguous() or weight.data_ptr() % 16:
            weight = weight.clone(memory_format=torch.contiguous_format)
    norm_args = (residual, weight, float(eps), norm_mode)
    if fusable:
        _lamport_preadmit(cfg, st, staged, group, torch.bfloat16, ref_norm=unfused, norm_args=norm_args, impl=impl)
    if fusable and mode and _lamport_verdict(cfg, st, staged, group, torch.bfloat16, mode, ref_norm=unfused,
                                    norm_args=norm_args, impl=impl):
        h = torch.empty_like(x)
        _, y = _run_lamport(cfg, staged, torch.bfloat16, mode, resid=residual, weight=weight, eps=float(eps),
                            norm_mode=norm_mode, out=h)
        LAMPORT_COUNTS["fused_norm"] += 1
        return y, h
    # Outside the fused domain: the plain reduction into the placeholder, then the unfused add+norm.
    LAMPORT_COUNTS["fallbacks"] += 1
    if lamport_all_reduce(staged, group, torch.bfloat16, st=st, out=x) is None:
        all_reduce(staged, group, torch.bfloat16, st=st, out=x)
    return unfused(x, residual, weight, eps)


def lamport_counts() -> dict:
    return dict(LAMPORT_COUNTS)
