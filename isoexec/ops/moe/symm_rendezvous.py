"""The MoE ops' (and the vendored DeepGEMM MegaMoE buffer's) route to THE process-wide symmetric-memory setup entry
point, ``isoexec.ops.collectives.symm_setup.rendezvous``, which counts every setup with pik's counter
(``pik.allreduce.rendezvous``): one counter per process for pik's pools, the tree reducer and the MoE exchanges.
This module is a thin delegate with no state of its own: ``rendezvous(buf, group, site=)`` refuses inside CUDA graph
capture, then calls ``symm_setup.rendezvous``; ``RendezvousSkewError`` IS ``SymmetricSetupSkew`` (raised on every rank
of a group whose ranks are at different setup counts or offer buffers of different sizes; callers then take their
NCCL realization together, the same bits).

A rank that never enters the rendezvous at all (it skipped the setup) still hangs its peers: whether to set up is the
caller's group-wide decision and must not depend on rank-local state.

Every call site in ``isoexec/ops/moe`` and ``isoexec/ops/_vendor/deep_gemm`` goes through this function
(``tests/test_symm_rendezvous_cpu.py`` scans the sources).
"""
from __future__ import annotations

import torch

from isoexec.ops.collectives import symm_setup as _shared       # THE entry point (one process-wide counter)

RendezvousSkewError = _shared.SymmetricSetupSkew


def group_ranks(group) -> tuple[int, ...]:
    """The global ranks of ``group`` (``None``: the default group): the identity every pool or cache of
    symmetric-memory state is keyed by (two groups of one size are different groups; two groups over the same ranks
    share one torch rendezvous sequence)."""
    import torch.distributed as dist
    g = group if group is not None else dist.group.WORLD
    return tuple(int(r) for r in dist.get_process_group_ranks(g))


def skewed(group) -> bool:
    """True once a rendezvous over these ranks was refused (identical on every rank of the group)."""
    return _shared.skewed(group)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _gather(mine: list[int], group, device: torch.device) -> list[list[int]]:
    """Every rank's ``mine`` in group-rank order (collective over ``group``)."""
    import torch.distributed as dist
    world = dist.get_world_size(group)
    backend = str(dist.get_backend(group)).lower()
    dev = device if (device.type == "cuda" and "nccl" in backend) else torch.device("cpu")
    t = torch.tensor(mine, dtype=torch.int64, device=dev)
    out = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(out, t, group=group)
    return [[int(v) for v in o.tolist()] for o in out]


def rendezvous(buf: torch.Tensor, group, *, site: str = "moe"):
    """``torch.distributed._symmetric_memory.rendezvous(buf, group)``, refused with ``RendezvousSkewError`` on every
    rank of ``group`` when its ranks are not at the same point of the process-wide setup sequence or pass buffers of
    different sizes. Collective over ``group``; outside CUDA graph capture."""
    import torch.distributed as dist
    if _capturing():
        raise RuntimeError("symmetric-memory rendezvous inside CUDA graph capture: set up outside capture")
    g = group if group is not None else dist.group.WORLD
    return _shared.rendezvous(buf, g, site=site)


__all__ = ["RendezvousSkewError", "group_ranks", "rendezvous", "skewed"]
