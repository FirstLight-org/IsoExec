"""The symmetric-memory workspace and the reducer of the deterministic tree all-reduce.

``TreeReducer(group)`` reduces each rank's FP32 subtree root to ``BF16(tree(root_0 .. root_{W-1}))``: the
fixed adjacent-pair FP32 tree in rank order, rounded once, bitwise ``leaf_tree(stack(all_gather(root)))``
(transport ``symm_tree_allreduce_v1``). The transport is the phased one (``phased.py``): one fused launch
per chunk over a private workspace, serialized by the owning runtime's ``execution_scope``. It is a drop-in
for ``attention/deepseek_v41_cache_sm100/tp.DistributedReducer`` (``gather_sum`` / ``run``) and callable on a
root directly.

The workspace is allocated and rendezvoused OUTSIDE CUDA-graph capture on the exact process group. Its
memory per rank does not depend on the world size. The transport is node-local: ``prepare`` refuses a group whose
ranks do not share one host (``collectives/node_local.NodeLocalityError``) or do not sit on pairwise distinct GPUs
(``SymmetricPeersError``) before any symmetric-memory call; such a group runs the NCCL realization
``DistributedReducer`` (all-gather + the same tree, the same bits).
"""
from __future__ import annotations

import os

import torch

from ._extension import extension

TREE_TRANSPORT = "symm_tree_allreduce_v1"

#: Capacity of the workspace's FP32 box (32-bit words per parity half, all sender slots together; the BF16
#: result box has the same element count). Larger payloads are reduced in consecutive chunks (the
#: reduction is elementwise). ``ISOEXEC_DETREDUCE_TREE_LAMPORT_CAP`` overrides; capacity only, never bits.
WORKSPACE_CAP = int(os.environ.get("ISOEXEC_DETREDUCE_TREE_LAMPORT_CAP", str(4 << 20)))

#: Must match csrc/detreduce.cu (kMaxCta, kMaxWorld, kTagGrid): peer-mapped flag block of [2 phases][kMaxCta][kMaxWorld]
#: int32 followed by the call credits [kTagGrid][kMaxWorld] int32.
MAX_CTA, MAX_WORLD = 1024, 16
TAG_GRID = 256
FLAG_WORDS = 2 * MAX_CTA * MAX_WORLD + TAG_GRID * MAX_WORLD
SYNC_BYTES = FLAG_WORDS * 4 + (MAX_CTA + 2 + TAG_GRID) * 4


def tree_bytes(cap: int = WORKSPACE_CAP) -> int:
    """Device bytes per rank of a TreeWorkspace (independent of the world size)."""
    cap = max(int(cap), 1)
    return 2 * cap * 4 + cap * 4 + SYNC_BYTES


class _Sync:
    """Peer-mapped flag block [2 phases][max_cta][max_world] + call credits [tag_grid][max_world], and local
    (epoch[max_cta], call, finished, tepoch[tag_grid])."""

    def __init__(self, group, device):
        import torch.distributed as dist
        from torch.distributed import _symmetric_memory as symm_mem

        ext = extension()
        self.max_cta, self.max_world = int(ext.max_cta()), int(ext.max_world())
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world > self.max_world:
            raise ValueError(f"world {self.world} exceeds {self.max_world}")
        self.flags = symm_mem.empty((2 * self.max_cta * self.max_world + int(ext.tag_grid()) * self.max_world,),
                                    dtype=torch.int32, device=device)
        self.flags.zero_()
        self.state = torch.zeros(self.max_cta + 2 + int(ext.tag_grid()), dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        from isoexec.ops.collectives.symm_setup import rendezvous
        self._h = rendezvous(self.flags, group, site="detreduce_sm100.flags")
        self.flag_ptrs = [int(p) for p in self._h.buffer_ptrs]

    def bytes(self) -> int:
        return self.flags.numel() * 4 + self.state.numel() * 4

    def barrier(self):
        self._h.barrier()


def _symm(shape, dtype, device, group, site):
    from torch.distributed import _symmetric_memory as symm_mem

    from isoexec.ops.collectives.symm_setup import rendezvous
    t = symm_mem.empty(shape, dtype=dtype, device=device)
    h = rendezvous(t, group, site=site)
    return t, h, [int(p) for p in h.buffer_ptrs]


class TreeWorkspace:
    """FP32 box (2 parity halves of ``cap`` words: W single-writer sender slots) + BF16 result box (``cap``
    elements, 2 halves) + flags, per rank."""

    def __init__(self, group, device, cap: int = WORKSPACE_CAP):
        import torch.distributed as dist

        self.group, self.device = group, torch.device(device)
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.sync = _Sync(group, self.device)
        self.lcap = int(cap)
        self.lbox, self._hl, self.lbox_ptrs = _symm((2 * max(self.lcap, 1),), torch.int32, self.device, group,
                                                         "detreduce_sm100.lbox")
        self.lbox.fill_(-1)        # sentinel 0xFFFFFFFF in every word
        self.lycap = self.lcap      # BF16 elements per parity half of the result box
        self.lybox, self._hly, self.lybox_ptrs = _symm((self.lycap,), torch.int32, self.device, group,
                                                            "detreduce_sm100.lybox")   # 2 halves x lycap/2 words
        self.lybox.fill_(-1)
        torch.cuda.synchronize(self.device)
        self.sync.barrier()

    def bytes(self) -> int:
        return self.lbox.numel() * 4 + self.lybox.numel() * 4 + self.sync.bytes()


class TreeReducer:
    """Bitwise drop-in for ``DistributedReducer`` (all_gather + leaf_tree) over the phased transport."""

    transport = TREE_TRANSPORT

    def __init__(self, group=None, *, cap: int = WORKSPACE_CAP):
        import torch.distributed as dist

        self.dist, self.group = dist, group
        self.world = dist.get_world_size(group)
        self.cap = cap
        self._ws = None
        self._phased = None

    def prepare(self, device):
        """Allocate + rendezvous the workspace (collective; outside capture, same order on every rank)."""
        if self.world > 1 and self._ws is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{TREE_TRANSPORT}: prepare the workspace before CUDA graph capture")
            from isoexec.ops.collectives.node_local import require_symmetric_peers

            # symmetric memory between distinct GPUs of one NVLink domain only: a group across hosts
            # (NodeLocalityError) or with two ranks on one GPU (SymmetricPeersError, a NodeLocalityError) is refused
            # by name before any allocation and takes DistributedReducer (same bits)
            require_symmetric_peers(self.group, f"detreduce_sm100 TreeReducer ({TREE_TRANSPORT})")
            from .phased import PhasedTransport

            # a setup skew (symm_setup.SymmetricSetupSkew) is raised on every rank of the group alike: no workspace,
            # the caller keeps the all-gather form (same bits)
            self._ws = TreeWorkspace(self.group, device, cap=self.cap)
            self._phased = PhasedTransport(self._ws, extension())
            sigs = [None] * self.world
            self.dist.all_gather_object(sigs, self._phased.signature(), group=self.group)
            if len(set(sigs)) != 1:
                raise RuntimeError(f"phased transport kernel/geometry differs across ranks: {sigs}")
        return self

    def execution_scope(self, *, capture=False):
        """The owning runtime's execution scope (every reduction, eager or graph replay, runs inside one)."""
        from contextlib import nullcontext

        if self.world == 1:
            return nullcontext()
        if self._phased is None:
            raise RuntimeError("prepare the phased transport before execution")
        return self._phased.execution_scope(capture=capture)

    def execution_scope_owner(self):
        """The thread ident holding this reducer's execution scope, or None."""
        return None if self._phased is None else self._phased.ownership.owner

    def holds_execution_scope(self) -> bool:
        """True when the calling thread is inside this reducer's execution scope (or none is needed: W = 1)."""
        import threading

        return self.world == 1 or (self._phased is not None
                                   and self._phased.ownership.owner == threading.get_ident())

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return x
        self.prepare(x.device)
        root = x if x.dtype == torch.float32 else x.float()
        return self._phased(root)

    def __call__(self, root: torch.Tensor) -> torch.Tensor:
        return self._reduce(root)

    # attention/deepseek_v41_cache_sm100/tp.DistributedReducer
    def gather_sum(self, x: torch.Tensor) -> torch.Tensor:
        return self._reduce(x)

    def run(self, gen):
        try:
            msg = next(gen)
            while True:
                msg = gen.send([self.gather_sum(s) for s in msg[1]])
        except StopIteration as stop:
            return self.gather_sum(stop.value)


__all__ = ["SYNC_BYTES", "TREE_TRANSPORT", "TreeReducer", "TreeWorkspace", "WORKSPACE_CAP", "tree_bytes"]
