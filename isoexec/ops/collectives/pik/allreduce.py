"""Deterministic all-reduce: the upper levels of the same combine tree."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from .codegen import (
    fused_p2p_allreduce_kernel,
    p2p_allreduce_kernel,
    tree_reduce_kernel,
)
from . import lane as _lane  # noqa: F401  -- loaded with the package so pik_bootstrap aliases one identity
from .fastlaunch import launch as fastlaunch

logger = logging.getLogger(__name__)

# Use one-shot gathering for small payloads to avoid an extra collective.
ONESHOT_MAX_BYTES = int(os.environ.get("ISOEXEC_PIK_ONESHOT_MB", "2")) << 20

# Switch large P2P payloads to owned-slice reduction and peer writes.
P2P_ONESHOT_MAX_BYTES = int(os.environ.get("ISOEXEC_PIK_ONESHOT_MB", "2")) << 20


_SYM: dict = {}
#: When True, no collective hands back a view of a symmetric buffer: the two-shot paths copy the root out.
#: Autograd and its hosts own the tensors they receive (verl frees checkpoint-input gradients with
#: ``untyped_storage().resize_(0)``, legal on a caching-allocator tensor, fatal on a symmetric pool), and a
#: pool view would be overwritten by the next collective while a gradient still holds it. The trainer sets
#: it (``pik_tp_invariant.select_trainer_transport``); the engine consumes roots before the next collective.
_PRIVATE_OUTPUTS = False


def set_private_outputs(on: bool) -> None:
    global _PRIVATE_OUTPUTS
    _PRIVATE_OUTPUTS = bool(on)
_SYM_IN: dict = {}
_SYM_OUT: dict = {}
# Geometric growth bounds retained buffers while keeping captured pointers valid.
_CAPTURED_POOLS: list = []


def _get_sym_pool(registry, key, n, device, group, dtype, pool_type):
    """Keep captured symmetric pointers alive across later eager pool growth."""
    pool = registry.get(key)
    if pool is None or pool.cap < n:
        if _capturing():
            raise RuntimeError(
                f"symmetric_pool.cap: need {n} elements during CUDA capture; warm this shape eagerly first"
            )
        capacity = max(n, 2 * pool.cap if pool is not None else 0)
        previous = pool
        pool = pool_type(capacity, device, group, dtype)
        pool._captured = False
        registry[key] = pool
        if previous is not None and previous._captured:
            _CAPTURED_POOLS.append(previous)
    if not pool._captured and _capturing():
        pool._captured = True
    return pool


class _SymInPool(torch.nn.Module):
    """Double-buffer symmetric inputs; the leading barrier protects outstanding peer reads."""

    def __init__(self, n: int, device, group, dtype=torch.float32):
        super().__init__()
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.dtype = dtype
        self.inp = [symm_mem.empty(cap, dtype=dtype, device=device) for _ in range(2)]
        for index, tensor in enumerate(self.inp):
            self.register_buffer(f"input_{index}", tensor, persistent=False)
        self.h_in = [rendezvous(b, g) for b in self.inp]
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], self.h_in[self.phase]

    def flip(self):
        self.phase ^= 1


class _SymOutPool(torch.nn.Module):
    """Store symmetric outputs separately from input buffers, keyed by root dtype."""

    def __init__(self, n: int, device, group, out_dtype=torch.float32):
        super().__init__()
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.out_dtype = out_dtype
        self.register_buffer("out", symm_mem.empty(cap, dtype=out_dtype, device=device), persistent=False)
        self.h_out = rendezvous(self.out, g)


class _SymPool(torch.nn.Module):
    """Self-contained pool: double-buffered staging plus an output buffer."""

    def __init__(self, n: int, device, group, dtype=torch.float32, out_dtype=None):
        super().__init__()
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.dtype = dtype
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.inp = [symm_mem.empty(cap, dtype=dtype, device=device) for _ in range(2)]
        for index, tensor in enumerate(self.inp):
            self.register_buffer(f"input_{index}", tensor, persistent=False)
        self.h_in = [rendezvous(b, g) for b in self.inp]
        self.register_buffer("out", symm_mem.empty(cap, dtype=self.out_dtype, device=device), persistent=False)
        self.h_out = rendezvous(self.out, g)
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], self.h_in[self.phase]

    def flip(self):
        self.phase ^= 1


class _PoolView:
    """A view pairing one staging pool with one output pool."""

    def __init__(self, inp: _SymInPool, outp: _SymOutPool):
        self._in = inp
        self._out = outp

    @property
    def cap(self):
        return min(self._in.cap, self._out.cap)

    @property
    def dtype(self):
        return self._in.dtype

    @property
    def out_dtype(self):
        return self._out.out_dtype

    @property
    def out(self):
        return self._out.out

    @property
    def h_out(self):
        return self._out.h_out

    @property
    def inp(self):
        return self._in.inp

    @property
    def phase(self):
        return self._in.phase

    def stage(self):
        return self._in.stage()

    def flip(self):
        self._in.flip()


def _sym_in_pool(n: int, device, group, dtype) -> _SymInPool:
    # Keyed by the group's membership, not its size: equal-sized groups (TP and EP) must never share a pool
    # rendezvoused with another group's peers.
    key = (device, _group_key(group), dtype)
    return _get_sym_pool(_SYM_IN, key, n, device, group, dtype, _SymInPool)


def _sym_out_pool(n: int, device, group, out_dtype) -> _SymOutPool:
    key = (device, _group_key(group), out_dtype)
    return _get_sym_pool(_SYM_OUT, key, n, device, group, out_dtype, _SymOutPool)


def _sym_pool(n: int, device, group, dtype=torch.float32, out_dtype=None) -> _PoolView:
    # Default the root dtype to the input dtype.
    if out_dtype is None:
        out_dtype = dtype
    inp = _sym_in_pool(n, device, group, dtype)
    outp = _sym_out_pool(n, device, group, out_dtype)
    key = (device, _group_key(group), dtype, out_dtype)
    view = _SYM.get(key)
    if view is None or view._in is not inp or view._out is not outp:
        view = _PoolView(inp, outp)
        _SYM[key] = view
    return view


class SymmetricSetupSkew(RuntimeError):
    """The ranks of a group have taken part in different numbers of pik symmetric-memory setups."""


# torch keys a rendezvous's handle exchange by a sequence number that is per PROCESS, not per group: a rank
# that set up a buffer with a group its peers are not in (a subgroup alone, or one group growing its pool
# while another does not) is one step ahead of them, and the next rendezvous they share waits forever on
# keys the others never write. Every pik rendezvous goes through ``rendezvous``, which first checks that
# the group's ranks have performed equally many; a group that has not takes the NCCL transport instead
# (same bits). The check covers pik's own setups; other symmetric-memory users must be symmetric.
_RENDEZVOUS = [0]
_SKEWED: set = set()


def rendezvous(tensor: torch.Tensor, group):
    """``symm_mem.rendezvous(tensor, group)``, refused with ``SymmetricSetupSkew`` (on every rank of the group)
    when its ranks are not at the same point of the process-wide setup sequence."""
    from torch.distributed import _symmetric_memory as symm_mem

    g = group if group is not None else dist.group.WORLD
    key = _group_key(g)
    if key in _SKEWED:
        raise SymmetricSetupSkew(f"group {key} already took the NCCL transport after a setup skew")
    world = dist.get_world_size(g)
    counts = torch.empty(world, dtype=torch.int64, device=tensor.device)
    dist.all_gather_into_tensor(counts, torch.tensor([_RENDEZVOUS[0]], dtype=torch.int64, device=tensor.device),
                                group=g)
    counts = counts.tolist()
    if len(set(counts)) > 1:
        _SKEWED.add(key)
        raise SymmetricSetupSkew(
            f"group {key}: its ranks have performed {counts} symmetric-memory setups; a rendezvous now "
            "would wait on keys some never write. This group takes the NCCL transport (same bits).")
    handle = symm_mem.rendezvous(tensor, g)
    _RENDEZVOUS[0] += 1
    return handle


def p2p_available(group=None) -> bool:
    """May this group move pik's bytes through symmetric memory? Bits never depend on the answer.

    Under the ``physical`` peer policy (the trainer, see ``lane.py``) the answer is a group-agreed probe of
    host and device UUID, so Ray actors that each see their GPU as ``cuda:0`` still get P2P; otherwise it
    is the historical rule (any group of more than one rank; the engine's ranks see every GPU).
    """
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        if not (dist.is_initialized() and dist.get_world_size(group) > 1):
            return False
        if _group_key(group if group is not None else dist.group.WORLD) in _SKEWED:
            return False
    except Exception:  # noqa: BLE001
        return False
    from . import lane as _lane

    if _lane.peer_policy() == "physical":
        return _lane.physical_p2p(group)
    return True


def _p2p_size_ok(t: torch.Tensor, group) -> bool:
    """Under the trainer's physical policy, large payloads take NCCL (``lane.p2p_max_bytes``, group-agreed;
    transport only)."""
    cap = _lane.p2p_max_bytes(group)
    return not cap or t.numel() * t.element_size() <= cap


def gather_last_dim(local: torch.Tensor, group, out: torch.Tensor) -> bool:
    """All-gather ``local`` (same shape on every rank) over symmetric memory, rank ``r``'s tensor landing in
    ``out[..., r * w:(r + 1) * w]`` (``w = local.shape[-1]``), widened to ``out.dtype`` on the copy.

    One staging copy into the double-buffered input pool, one barrier, then one copy per peer straight from
    its buffer: bytes only, so the result is NCCL ``all_gather``'s bit for bit. Returns False, on every rank
    of the group alike, when this group has no P2P transport (refused arch or ``ISOEXEC_PIK_P2P=0``, which
    patch ``p2p_available`` off), its pool setup is refused for a setup skew (``rendezvous``), or a CUDA-graph
    capture would need a pool setup; the caller then takes NCCL.
    The pool is the all-reduce's own bf16 input pool for this group (keyed by the group, double-buffered),
    so the gather adds no rendezvous beyond growing it once to the largest logits shard.
    """
    if not (local.is_cuda and local.is_contiguous() and local.numel() and p2p_available(group)):
        return False
    world, width, n = dist.get_world_size(group), local.shape[-1], local.numel()
    assert out.shape == (*local.shape[:-1], world * width), (out.shape, local.shape, world)
    if _capturing():
        pool = _SYM_IN.get((local.device, _group_key(group), local.dtype))
        if pool is None or pool.cap < n:  # a rendezvous cannot run inside a capture
            return False
    else:
        try:
            pool = _sym_in_pool(n, local.device, group, local.dtype)
        except SymmetricSetupSkew:  # raised on every rank of the group: all take NCCL together
            return False
    from .lane import fence

    fence(local.device)  # an overlapped collective on a lane may still read this pool
    inp, handle = pool.stage()
    inp[:n].copy_(local.view(-1))
    handle.barrier()
    for rank in range(world):
        out[..., rank * width:(rank + 1) * width].copy_(handle.get_buffer(rank, tuple(local.shape), local.dtype))
    # Phase reuse is safe: this pool's next write to this phase is two collectives on, behind a barrier
    # every peer reaches only after these reads.
    pool.flip()
    return True


def sym_partial(shape, device, group=None, dtype=torch.float32, out_dtype=None) -> torch.Tensor:
    """A view of the requested shape into this rank's current symmetric staging buffer."""
    from .lane import fence

    fence(device)  # a producer GEMM writes this staging buffer on the caller's stream
    n = 1
    for s in shape:
        n *= s
    inp, _ = _sym_pool(n, device, group, dtype, out_dtype).stage()
    return inp[:n].view(*shape)


def _p2p_shot(n: int, elt: int, world: int) -> bool:
    """Select two-shot slice reduction or one-shot full reduction."""
    from .ar_branch import two_shot

    return two_shot(n, elt, world, P2P_ONESHOT_MAX_BYTES)


# Admit each vector layout only after bitwise validation on that shape.
_AR_VEC_ENV = "ISOEXEC_PIK_AR_VEC"


def _ar_vec() -> int:
    try:
        v = int(os.environ.get(_AR_VEC_ENV, "4").strip() or "1")
    except ValueError:
        return 1
    return v if v in (2, 4, 8) else 1


_VEC_ADMIT: dict = {}
_VEC_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_VEC_FIRST_REJECT: dict = {}


def vec_counts() -> dict:
    """Per-process tally of the vec-layout admission."""
    return {
        **_VEC_COUNTS,
        "vec": _ar_vec(),
        "shapes": {str(k): v for k, v in _VEC_ADMIT.items()},
        "first_reject": dict(_VEC_FIRST_REJECT) or None,
    }


def _vec_reject(key, reason: str, detail: str = "") -> None:
    global _VEC_FIRST_REJECT
    _VEC_ADMIT[key] = 1
    _VEC_COUNTS["rejected"] += 1
    if _VEC_FIRST_REJECT:
        return
    _VEC_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    msg = (
        f"[ISOEXEC-PIK] WARNING: the vec element-to-thread layout was REFUSED at {key} -- "
        f"{reason}. {detail} The result is unaffected (this shape runs the vec=1 reference "
        "layout, which every frozen bit was proven against); only the ~2 us/site store-side "
        "saving is not taken here. Read pik.allreduce.vec_counts()."
    )
    print(msg, flush=True)
    logger.warning(msg)


def _vec_admit(key, partial: torch.Tensor, group, root_dtype, want: int) -> int:
    """Admit vectorization only after a group-wide bitwise comparison on live operands."""
    device = partial.device
    reason = detail = ""
    ref = _p2p_unfused(partial, group, None, root_dtype, _vec=1).clone()
    local_ok = True
    try:
        got = _p2p_unfused(partial, group, None, root_dtype, _vec=want).clone()
    except Exception as e:  # noqa: BLE001
        _VEC_COUNTS["errors"] += 1
        local_ok, reason, detail = False, "the vec kernel raised", repr(e)
    else:
        if not torch.equal(_bits(ref), _bits(got)):
            local_ok, reason = False, "BIT PATTERNS DIFFER"
            detail = f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements"
    agreed = _agree(local_ok, group, device)
    if not agreed:
        if local_ok:
            _VEC_COUNTS["peer_vetoes"] += 1
            reason = "a PEER rank refused it"
            detail = "this rank's own compare passed; the group runs vec=1 together"
        _vec_reject(key, reason, detail)
        return 1
    _VEC_ADMIT[key] = want
    _VEC_COUNTS["admitted"] += 1
    if _VEC_COUNTS["admitted"] == 1:
        msg = (
            f"[ISOEXEC-PIK] AR vec layout ADMITTED (first shape {key}, vec={want}): bit patterns "
            "equal the vec=1 reference on live operands, agreed across the group. "
            "Read pik.allreduce.vec_counts() for the per-shape verdicts."
        )
        print(msg, flush=True)
        logger.warning(msg)
    return want


def _vec_resolved(world, n, dt_in, dt_out, branch_push, partial, group, root_dtype) -> int:
    want = _ar_vec()
    if want == 1:
        return 1
    key = (world, n, str(dt_in), str(dt_out), "push" if branch_push else "local", want)
    verdict = _VEC_ADMIT.get(key)
    if verdict is not None:
        return verdict
    if _capturing():
        # A compare needs a host sync; the shape gets another chance when seen eagerly.
        _VEC_COUNTS["capture_skips"] += 1
        return 1
    return _vec_admit(key, partial, group, root_dtype, want)


def _p2p_unfused(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None, _vec=None):
    """One kernel: read every rank's partial over NVLink, apply the tree, and write back."""
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    n = partial.numel()
    shape = partial.shape
    dt_in = partial.dtype
    dt_out = root_dtype if root_dtype is not None else dt_in
    in_bf16 = dt_in == torch.bfloat16
    out_bf16 = dt_out == torch.bfloat16
    from .lane import fence

    fence(partial.device)
    if out is None and _PRIVATE_OUTPUTS:
        out = torch.empty(shape, device=partial.device, dtype=dt_out)
    pool = _sym_pool(n, partial.device, group, dt_in, dt_out)
    inp, h_in = pool.stage()

    push = _p2p_shot(n, partial.element_size(), world)
    if _vec is None:
        _vec = _vec_resolved(world, n, dt_in, dt_out, push, partial, group, root_dtype)
        _VEC_COUNTS["calls"] += 1
        # Admission ran both paths and flipped the pool twice (net zero); re-resolve the staging view.
        inp, h_in = pool.stage()

    if partial.data_ptr() != inp.data_ptr():
        inp[:n].copy_(partial.view(-1))  # caller didn't stage; see sym_partial
    h_in.barrier()  # every rank's partial is now visible to every peer

    peers_in = [h_in.get_buffer(r, (n,), dt_in) for r in range(world)]

    if not push:
        # One-shot: each rank reads all peers and computes the whole tensor. Double-buffering makes the
        # trailing barrier unnecessary.
        dst = out.view(-1) if out is not None else torch.empty(n, device=partial.device, dtype=dt_out)
        k = p2p_allreduce_kernel(world, push=False, in_bf16=in_bf16, out_bf16=out_bf16, vec=_vec)
        k[(triton.cdiv(n, 1024),)](*peers_in, dst, 0, n, BLOCK=1024, num_warps=4)
        pool.flip()
        return dst.view(*shape)

    # Two-shot: rank r owns slice r and pushes it into every peer's out. The trailing barrier is unavoidable
    # here, because the caller reads slices that peers wrote.
    assert n % world == 0, f"two-shot P2P needs numel {n} divisible by world {world}"
    chunk = n // world
    peers_out = [pool.h_out.get_buffer(r, (n,), dt_out) for r in range(world)]
    k = p2p_allreduce_kernel(world, push=True, in_bf16=in_bf16, out_bf16=out_bf16, vec=_vec)
    k[(triton.cdiv(chunk, 1024),)](*peers_in, *peers_out, rank * chunk, chunk, BLOCK=1024, num_warps=4)
    pool.h_out.barrier()  # every slice has landed on every rank
    pool.flip()
    if out is not None:
        out.view(-1).copy_(pool.out[:n])
        return out.view(*shape)
    return pool.out[:n].view(*shape)  # no copy-out: hand back the symmetric buffer


# Fuse barriers with the fixed reduction tree only after numerical qualification.
_FUSED_ENV = "ISOEXEC_PIK_FUSED_BARRIER"


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


_FUSED_ON = _env_on(_FUSED_ENV)

# Cache the group-agreed fused-kernel verdict for each shape and dtype combination.
_FUSED_ADMIT: dict = {}
_FUSED_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_FUSED_FIRST_REJECT: dict = {}
# Per process group: has this group AGREED that the fused path may be attempted at all?
_FUSED_GROUP: dict = {}


def fused_enabled() -> bool:
    """Whether this process is allowed to try the fused single-launch collective."""
    return _FUSED_ON


def set_fused_enabled(on: bool) -> None:
    """Test/bench hook; flipping this does not bypass admission."""
    global _FUSED_ON
    _FUSED_ON = bool(on)


def fused_counts() -> dict:
    """Per-process tally of the fused collective, plus the first rejection if any."""
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


def _agree(local_ok: bool, group, device) -> bool:
    """Require every rank to agree before selecting a collective implementation."""
    if not dist.is_initialized() or dist.get_world_size(group) < 2:
        return local_ok
    t = torch.tensor([1 if local_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _fused_group_on(group, device) -> bool:
    """Has this group agreed the fused path may be attempted?"""
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
        msg = (
            f"[ISOEXEC-PIK] the FUSED single-launch all-reduce is DISABLED for process group "
            f"{key}: this rank has {_FUSED_ENV}={'on' if _FUSED_ON else 'off'} but at least one "
            "peer disagrees, and running the two launch structures side by side would deadlock. "
            "Set the flag identically on every rank of a TP group."
        )
        print(msg, flush=True)
        logger.warning(msg)
    elif got:
        msg = (
            f"[ISOEXEC-PIK] FUSED single-launch tree all-reduce ENABLED and agreed by all "
            f"{dist.get_world_size(group) if dist.is_initialized() else 1} ranks of group {key}. "
            "Each shape still has to prove itself bitwise against the reference path before it "
            "is used. Read pik.allreduce.fused_counts()."
        )
        print(msg, flush=True)
        logger.info(msg)
    return got


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 -- older torch without the query
        return False


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bit pattern as an integer tensor."""
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float16:
        return t.view(torch.int16)
    return t


def flag_slot(region: int, world: int, rank: int, nb: int) -> int:
    """Where rank ``rank`` publishes its arrival in ANY rank's flag array, for ``region``."""
    return (region * world + rank) * nb


def flag_wait_base(region: int, world: int, nb: int) -> int:
    """Start of ``region`` in a rank's OWN flag array; the spin adds ``lane*nb + block``."""
    return region * world * nb


class _FlagPool:
    """Symmetric int32 flag matrix plus local epoch counters for the in-kernel barrier."""

    def __init__(self, nb: int, device, group):
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        world = dist.get_world_size(g)
        rank = dist.get_rank(g)
        self.nb = nb
        self.world = world
        self.rank = rank
        self.flags = symm_mem.empty(2 * world * nb, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.h = rendezvous(self.flags, g)
        self.peers = [self.h.get_buffer(r, (2 * world * nb,), torch.int32) for r in range(world)]
        self.mine = self.peers[rank]
        self.epoch = torch.zeros(2 * nb, dtype=torch.int32, device=device)
        # rendezvous is collective, so ranks agree on when; this barrier makes them agree that the zeroing
        # landed before the first publish. Once per allocation, never in the hot path.
        self.h.barrier()

    # Slot a peer writes this rank's arrival into, and where this rank looks for theirs.
    def my_slot(self, region: int) -> int:
        return flag_slot(region, self.world, self.rank, self.nb)

    def wait_base(self, region: int) -> int:
        return flag_wait_base(region, self.world, self.nb)


_FLAGS: dict = {}
_FLAG_KEEP: list = []


def _flag_pool(nb: int, device, group) -> _FlagPool:
    key = (device, dist.get_world_size(group) if dist.is_initialized() else 1)
    pool = _FLAGS.get(key)
    if pool is None or pool.nb < nb:
        if _capturing():
            # Allocating and rendezvous-ing inside a capture is illegal, and a half-built symmetric
            # allocation hangs rather than erroring, so refuse visibly instead.
            raise RuntimeError(
                f"pik fused collective: the flag pool must grow to {nb} blocks, but this is "
                "inside a CUDA graph capture. Run every shape once EAGERLY before capturing "
                "(a warmup pass), so the pools are sized before the graph is recorded."
            )
        want = max(nb, 2 * pool.nb if pool is not None else 0, 1024)
        if pool is not None:
            _FLAG_KEEP.append(pool)  # a launched kernel may still hold the old pointers
        pool = _FlagPool(want, device, group)
        _FLAGS[key] = pool
    return pool


def _p2p_fused(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None):
    """Fuse synchronization into the peer reduction kernel."""
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    n = partial.numel()
    shape = partial.shape
    dt_in = partial.dtype
    dt_out = root_dtype if root_dtype is not None else dt_in
    in_bf16 = dt_in == torch.bfloat16
    out_bf16 = dt_out == torch.bfloat16
    from .lane import fence

    fence(partial.device)
    if out is None and _PRIVATE_OUTPUTS:
        out = torch.empty(shape, device=partial.device, dtype=dt_out)
    pool = _sym_pool(n, partial.device, group, dt_in, dt_out)
    inp, h_in = pool.stage()

    if partial.data_ptr() != inp.data_ptr():
        inp[:n].copy_(partial.view(-1))  # caller didn't stage; see sym_partial
    # The kernel barrier runs after the staging copy; no separate input barrier is needed.

    peers_in = [h_in.get_buffer(r, (n,), dt_in) for r in range(world)]
    push = _p2p_shot(n, partial.element_size(), world)

    if not push:
        grid = triton.cdiv(n, 1024)
        base, count = 0, n
        dst = out.view(-1) if out is not None else torch.empty(n, device=partial.device, dtype=dt_out)
        outs = [dst]
    else:
        assert n % world == 0, f"two-shot P2P needs numel {n} divisible by world {world}"
        chunk = n // world
        grid = triton.cdiv(chunk, 1024)
        base, count = rank * chunk, chunk
        outs = [pool.h_out.get_buffer(r, (n,), dt_out) for r in range(world)]

    fp = _flag_pool(grid, partial.device, group)
    k = fused_p2p_allreduce_kernel(world, push=push, in_bf16=in_bf16, out_bf16=out_bf16)
    k[(grid,)](
        *peers_in,
        *outs,
        *fp.peers,
        fp.mine,
        fp.epoch,
        fp.my_slot(0),
        fp.my_slot(1),
        fp.wait_base(1),
        fp.nb,
        base,
        count,
        BLOCK=1024,
        WORLD=world,
        num_warps=4,
    )
    pool.flip()
    if not push:
        return dst.view(*shape)
    if out is not None:
        out.view(-1).copy_(pool.out[:n])
        return out.view(*shape)
    return pool.out[:n].view(*shape)


def _fused_reject(key, reason: str, detail: str = "") -> None:
    """Refuse the fused path for this shape, once and loudly."""
    global _FUSED_FIRST_REJECT
    _FUSED_ADMIT[key] = False
    _FUSED_COUNTS["rejected"] += 1
    if _FUSED_FIRST_REJECT:
        return
    _FUSED_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    msg = (
        f"[ISOEXEC-PIK] WARNING: the FUSED single-launch tree all-reduce was REFUSED at "
        f"{key} -- {reason}. {detail} The result is unaffected (this shape falls back to "
        "the reference barrier+kernel path, which is what every frozen bit was proven "
        "against), but the launch-count saving is not being taken here. Read "
        "pik.allreduce.fused_counts()."
    )
    print(msg, flush=True)
    logger.warning(msg)


def _fused_admit(key, partial: torch.Tensor, group, root_dtype) -> bool:
    """Admit fused reduction only after a group-wide bitwise comparison on live operands."""
    device = partial.device
    reason = detail = ""
    ref = _p2p_unfused(partial, group, None, root_dtype)
    ref = ref.clone()
    local_ok = True
    try:
        got = _p2p_fused(partial, group, None, root_dtype)
        got = got.clone()
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
        _fused_reject(key, reason, detail)
        return False
    _FUSED_ADMIT[key] = True
    _FUSED_COUNTS["admitted"] += 1
    if _FUSED_COUNTS["admitted"] == 1:
        # Log the first successful admission so active optimizations are observable.
        msg = (
            f"[ISOEXEC-PIK] FUSED barrier+reduce ADMITTED (first shape {key}): this shape now runs "
            f"as ONE launch instead of barrier+kernel[+barrier]. Bit patterns were compared against "
            f"the reference path on live operands and agreed across the group. Read "
            f"pik.allreduce.fused_counts() for the full per-shape verdict list; a decode trace "
            f"should show c10d::symmetric_memory::barrier_kernel drop by 2 calls/step per admitted "
            f"two-shot site."
        )
        print(msg, flush=True)
        logger.warning(msg)
    return True


def _tree_all_reduce_p2p(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None):
    """Symmetric-memory tree all-reduce: dispatch to the reference launch structure or the fused one."""
    world = dist.get_world_size(group)
    if not _fused_group_on(group, partial.device):
        return _p2p_unfused(partial, group, out, root_dtype)

    n = partial.numel()
    dt_out = root_dtype if root_dtype is not None else partial.dtype
    key = (world, n, str(partial.dtype), str(dt_out))
    verdict = _FUSED_ADMIT.get(key)
    if verdict is None:
        if world < 2 or (world & (world - 1)):
            _fused_reject(key, "world size is not a power of two", f"world={world}")
            return _p2p_unfused(partial, group, out, root_dtype)
        if _capturing():
            # A compare needs a host sync, which capture forbids. Record no verdict, so the shape gets
            # another chance the next time it is seen eagerly.
            _FUSED_COUNTS["capture_skips"] += 1
            return _p2p_unfused(partial, group, out, root_dtype)
        # Admission ran both paths, discarded both answers, and left the staging pool on the phase it
        # started on, so this call runs for real as if admission had not happened.
        verdict = _fused_admit(key, partial, group, root_dtype)
    if not verdict:
        return _p2p_unfused(partial, group, out, root_dtype)
    _FUSED_COUNTS["calls"] += 1
    return _p2p_fused(partial, group, out, root_dtype)


# Qualify the fused FP32-to-BF16 root cast against the separate cast.
_ROOT_CAST_ENV = "ISOEXEC_PIK_FUSED_ROOT_CAST"
# Default on: with the staging/output pools split by dtype the copy-in this used to reintroduce is gone.
# Admission is unchanged, and a refusal simply leaves the round where it was.
_ROOT_CAST_ON = _env_on(_ROOT_CAST_ENV, "1")
_ROOT_CAST_ADMIT: dict = {}
_ROOT_CAST_COUNTS = {"absorbed": 0, "admitted": 0, "rejected": 0, "capture_skips": 0}


def set_root_cast_enabled(on: bool) -> None:
    """Test/bench hook for the root-round absorption. Admission still applies per shape."""
    global _ROOT_CAST_ON
    _ROOT_CAST_ON = bool(on)


def root_cast_counts() -> dict:
    return {**_ROOT_CAST_COUNTS, "enabled": _ROOT_CAST_ON, "shapes": {str(k): v for k, v in _ROOT_CAST_ADMIT.items()}}


def tree_all_reduce_rounded(
    partial: torch.Tensor,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fuse the output cast only for shapes with proven bitwise equivalence."""
    want_narrow = partial.dtype == torch.float32 and out_dtype == torch.bfloat16
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if not (_ROOT_CAST_ON and want_narrow and world > 1):
        return tree_all_reduce(partial, group=group).to(out_dtype)

    key = (world, partial.numel(), str(partial.dtype))
    verdict = _ROOT_CAST_ADMIT.get(key)
    if verdict is None:
        if _capturing():
            _ROOT_CAST_COUNTS["capture_skips"] += 1
            return tree_all_reduce(partial, group=group).to(out_dtype)
        verdict = _root_cast_admit(key, partial, group)
    if not verdict:
        return tree_all_reduce(partial, group=group).to(out_dtype)
    _ROOT_CAST_COUNTS["absorbed"] += 1
    return tree_all_reduce(partial, group=group, root_dtype=torch.bfloat16)


def _root_cast_admit(key, partial: torch.Tensor, group) -> bool:
    """Verify the fused cast against the separate cast on live operands."""
    ref = tree_all_reduce(partial, group=group).to(torch.bfloat16).clone()
    try:
        got = tree_all_reduce(partial, group=group, root_dtype=torch.bfloat16).clone()
    except Exception as e:  # noqa: BLE001
        local_ok, why = False, f"the narrowing-root path raised: {e!r}"
    else:
        local_ok = torch.equal(_bits(ref), _bits(got))
        why = (
            ""
            if local_ok
            else (
                f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements differ -- "
                "the in-kernel round is NOT the same rounding as torch's"
            )
        )
    agreed = _agree(local_ok, group, partial.device)
    if not agreed:
        _ROOT_CAST_ADMIT[key] = False
        _ROOT_CAST_COUNTS["rejected"] += 1
        msg = (
            f"[ISOEXEC-PIK] WARNING: absorbing the ROOT's fp32->bf16 round into the reduce "
            f"kernel was REFUSED at {key} -- "
            f"{why or 'a PEER rank refused it'}. The result is unaffected (the round happens "
            "in its own kernel, as it always has) but the launch is not being saved here. "
            "Read pik.allreduce.root_cast_counts()."
        )
        print(msg, flush=True)
        logger.warning(msg)
        return False
    _ROOT_CAST_ADMIT[key] = True
    _ROOT_CAST_COUNTS["admitted"] += 1
    return True


def _tree_reduce(stacked: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Combine rank-ordered bf16 leaves or fp32 nodes into an fp32 root."""
    import triton

    c = stacked.shape[0]
    n = out.numel()
    # Through the memoized launcher; with ISOEXEC_PIK_FASTLAUNCH off this is the plain bracket call.
    fastlaunch(
        tree_reduce_kernel(c, in_bf16=stacked.dtype == torch.bfloat16),
        (triton.cdiv(n, 1024),), stacked, out, n, stacked.stride(0), BLOCK=1024, num_warps=4
    )
    return out


# Report the transport actually used, including P2P fallbacks.
_TRANSPORT_COUNTS = {"p2p_calls": 0, "nccl_calls": 0, "p2p_fallbacks": 0}
_FIRST_FALLBACK: dict = {}
_TRANSPORT_HOOKS: list = []
_LAST_STATE: str = "none"


def transport_state() -> str:
    """Which transport this process's tree all-reduce has actually been running on."""
    c = _TRANSPORT_COUNTS
    if c["p2p_fallbacks"]:
        return "p2p_fallback"
    if c["p2p_calls"] and c["nccl_calls"]:
        return "p2p_and_nccl"
    if c["p2p_calls"]:
        return "p2p"
    if c["nccl_calls"]:
        return "nccl"
    return "none"


def transport_counts() -> dict:
    """Per-process call counts plus the first fallback's exception, if any."""
    return {**_TRANSPORT_COUNTS, "state": transport_state(), "first_fallback": dict(_FIRST_FALLBACK) or None}


def register_transport_hook(fn) -> None:
    """Notify registered hooks when the active collective transport changes."""
    _TRANSPORT_HOOKS.append(fn)
    if _LAST_STATE != "none":
        impl = _tree_all_reduce_p2p if _LAST_STATE == "p2p" else _tree_reduce
        try:
            fn(_LAST_STATE, impl)  # catch-up, this hook only
        except Exception as e:  # noqa: BLE001
            logger.warning("[pik] transport hook %r failed: %r", fn, e)


def _fire_transport_hooks(state: str, impl_fn) -> None:
    for fn in list(_TRANSPORT_HOOKS):
        try:
            fn(state, impl_fn)
        except Exception as e:  # noqa: BLE001 -- a hook must never break the reduce
            logger.warning("[pik] transport hook %r failed: %r", fn, e)


def _note_transport(kind: str, impl_fn) -> None:
    global _LAST_STATE
    _TRANSPORT_COUNTS[f"{kind}_calls"] += 1
    state = transport_state()
    if state != _LAST_STATE:
        _LAST_STATE = state
        _fire_transport_hooks(state, impl_fn)


def _note_p2p_fallback(
    exc: BaseException, partial: torch.Tensor, world: int, *, operation: str = "all-reduce"
) -> None:
    """Count the fallback and report it once per process, naming the exception."""
    global _FIRST_FALLBACK
    _TRANSPORT_COUNTS["p2p_fallbacks"] += 1
    if _FIRST_FALLBACK:
        return
    _FIRST_FALLBACK = {
        "exception": repr(exc),
        "shape": tuple(partial.shape),
        "dtype": str(partial.dtype),
        "world": world,
    }
    msg = (
        f"[ISOEXEC-PIK] WARNING: pik symmetric-memory P2P {operation} FAILED and fell back to the "
        f"NCCL transport -- {exc!r} (first seen at shape={tuple(partial.shape)} "
        f"dtype={partial.dtype} world={world}). The RESULT is unaffected (both transports "
        "preserve the contracted values), but this collective pays NCCL latency. Read "
        "pik.allreduce.transport_counts() for the running tally."
    )
    print(msg, flush=True)
    logger.warning(msg)


def tree_all_reduce(
    partial: torch.Tensor,
    group=None,
    out: torch.Tensor | None = None,
    backend: str | None = None,
    root_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Combine per-rank subtree partials into the full tree, on every rank."""
    # Keep internal FP32 tree nodes on an FP32 wire.
    assert partial.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"partial must be fp32 (internal node) or bf16 (leaf, m=1), got {partial.dtype}"
    # Validate input and root dtypes before selecting the reduction path.
    assert root_dtype in (None, partial.dtype, torch.float32, torch.bfloat16), (
        f"root_dtype={root_dtype} with partial dtype {partial.dtype}: the root may be the "
        "partial's dtype, fp32 (widened bf16 leaf wire), or bf16 (narrowed root round)"
    )
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        # No reduce: the root is the partial itself. The bf16 -> fp32 upcast is exact.
        return partial.to(root_dtype) if root_dtype is not None else partial

    if backend is None:
        # The owner all-reduce (pik/owner.py): one launch, same tree. Entered on every rank whatever its own
        # flag says -- the first call per group is the collective that agrees whether it is on at all.
        from . import owner as _owner

        res = _owner.tree_all_reduce_routed(partial, group, out, root_dtype)
        if res is not None:
            return res
    return _tree_all_reduce_legacy(partial, group, out, backend, root_dtype)


def _tree_all_reduce_legacy(partial, group, out, backend, root_dtype) -> torch.Tensor:
    """The pre-owner transports: symmetric-memory barrier+kernel(+barrier), or NCCL bytes + pik's tree."""
    world = dist.get_world_size(group)
    partial = partial.contiguous()

    backend = backend or ("p2p" if (p2p_available(group) and _p2p_size_ok(partial, group)) else "nccl")
    if backend == "p2p":
        try:
            # Do not pre-allocate `out`: without private outputs (the engine) the two-shot P2P path can hand
            # back its symmetric output buffer directly, and forcing a destination reintroduces a copy-out.
            res = _tree_all_reduce_p2p(partial, group, out, root_dtype)
        except Exception as e:  # noqa: BLE001 -- no NVLink / no symm-mem: fall back
            # Report it once, naming `e`: a silent fallback is what makes the transport unobservable.
            _note_p2p_fallback(e, partial, world)
        else:
            _note_transport("p2p", _tree_all_reduce_p2p)
            return res

    _note_transport("nccl", _tree_reduce)
    return _tree_all_reduce_nccl(partial, group, out, root_dtype)


def _tree_all_reduce_nccl(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype) -> torch.Tensor:
    """The NCCL transport: all_gather / all_to_all move bytes, ``_tree_reduce`` does the arithmetic."""
    world = dist.get_world_size(group)

    # Evaluate the tree in FP32 and round only at the requested root.
    want_bf16_root = (root_dtype or partial.dtype) == torch.bfloat16
    if partial.dtype == torch.bfloat16:
        partial = partial.to(torch.float32)
    if out is None or out.dtype != torch.float32:
        out = torch.empty_like(partial)

    if partial.numel() * 4 <= ONESHOT_MAX_BYTES:
        # One-shot: every rank gathers every partial and evaluates the whole tree.
        buf = torch.empty((world, *partial.shape), device=partial.device, dtype=torch.float32)
        dist.all_gather_into_tensor(buf, partial, group=group)
        res = _tree_reduce(buf, out)
        return res.to(torch.bfloat16) if want_bf16_root else res

    # Two-shot: all-to-all the slices, tree-reduce the slice this rank owns, all-gather back. Same bytes on
    # the wire as a ring all-reduce, but the arithmetic is ours.
    n = partial.numel()
    assert n % world == 0, (
        f"two-shot needs numel {n} divisible by world {world}; pad M or N, or lower ONESHOT_MAX_BYTES"
    )
    chunk = n // world
    flat = partial.view(-1)

    recv = torch.empty((world, chunk), device=partial.device, dtype=torch.float32)
    dist.all_to_all_single(recv.view(-1), flat, group=group)  # recv[r] = rank r's slice `me`

    mine = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    _tree_reduce(recv, mine)

    dist.all_gather_into_tensor(out.view(-1), mine, group=group)
    return out.to(torch.bfloat16) if want_bf16_root else out


def _tree_reduce_scatter_p2p(partial: torch.Tensor, group) -> torch.Tensor:
    """Reduce this rank's slice with the scalar tree and an FP32 root; keep the final cast separate."""
    import triton

    from .lane import fence

    fence(partial.device)
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    n = partial.numel()
    chunk = n // world
    pool = _sym_in_pool(n, partial.device, group, partial.dtype)
    inp, handle = pool.stage()
    if partial.data_ptr() != inp.data_ptr():
        inp[:n].copy_(partial.view(-1))
    handle.barrier()
    peers = [
        handle.get_buffer(r, (n,), partial.dtype).narrow(0, rank * chunk, chunk)
        for r in range(world)
    ]
    out = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    kernel = p2p_allreduce_kernel(
        world, push=False, in_bf16=partial.dtype == torch.bfloat16, out_bf16=False, vec=1
    )
    kernel[(triton.cdiv(chunk, 1024),)](*peers, out, 0, chunk, BLOCK=1024, num_warps=4)
    pool.flip()
    return out


def tree_reduce_scatter(
    partial: torch.Tensor,
    group=None,
    root_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return this rank's equal row slice of the fixed-tree reduction, using P2P when available."""
    assert partial.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"partial must be fp32 (internal node) or bf16 (leaf, m=1), got {partial.dtype}"
    assert root_dtype in (None, partial.dtype, torch.float32), (
        f"root_dtype={root_dtype} with partial dtype {partial.dtype}: the only decoupled "
        "combination is bf16 partial (leaf wire) + fp32 root"
    )
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        # No reduce and no scatter: the whole tensor is this rank's slice. The bf16 -> fp32 upcast is exact.
        return partial.to(root_dtype) if root_dtype is not None else partial

    # Require equal first-dimension slices so flat splitting preserves token ownership.
    assert partial.shape[0] % world == 0, (
        f"tree_reduce_scatter: first dim {partial.shape[0]} not divisible by world {world} "
        f"(shape {tuple(partial.shape)}). Megatron sequence parallelism requires the sequence "
        "dim to divide the TP size; pad the sequence (or run SP off for this shape)."
    )

    partial = partial.contiguous()
    out_shape = (partial.shape[0] // world, *partial.shape[1:])
    if partial.is_cuda and partial.numel() and p2p_available(group) and _p2p_size_ok(partial, group):
        try:
            mine = _tree_reduce_scatter_p2p(partial, group)
        except Exception as exc:  # noqa: BLE001 -- same visible fallback as all-reduce
            _note_p2p_fallback(exc, partial, world, operation="reduce-scatter")
        else:
            _note_transport("p2p", _tree_reduce_scatter_p2p)
            res = mine.view(out_shape)
            return res.to(torch.bfloat16) if (root_dtype or partial.dtype) == torch.bfloat16 else res

    # Send BF16 leaves unchanged and widen inside the tree; internal nodes stay FP32.

    n = partial.numel()
    chunk = n // world  # first-dim divisibility above implies numel divisibility
    flat = partial.view(-1)

    # Exchange owned slices, reduce each with the fixed tree, and omit the final gather.
    recv = torch.empty((world, chunk), device=partial.device, dtype=partial.dtype)
    dist.all_to_all_single(recv.view(-1), flat, group=group)  # recv[r] = rank r's slice `me`

    mine = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    _tree_reduce(recv, mine)

    _note_transport("nccl", _tree_reduce)
    res = mine.view(out_shape)
    return res.to(torch.bfloat16) if (root_dtype or partial.dtype) == torch.bfloat16 else res
