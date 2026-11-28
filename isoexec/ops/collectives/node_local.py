"""Admission of node-local collective mechanisms: refuse, by name, a group whose ranks do not share one host.

Symmetric-memory transports (``detreduce_sm100``'s tree reducer, the MegaMoE expert-parallel buffers) map every
peer's device memory into each rank, which only works inside one NVLink domain. A group that spans hosts must
take the mechanism's NCCL realization instead (bitwise equal, by each mechanism's contract), so the check runs
before any symmetric-memory allocation or rendezvous and names the hosts and the mechanism, instead of leaving
the failure to ``symm_mem.rendezvous``.

A host is identified by ``(hostname, boot id)``: two containers on different machines may share a hostname.
The check is a collective (every rank of the group, outside CUDA graph capture, same order); its result is
cached per group.

``symmetric_admission(group)`` extends the host check to everything torch's symmetric memory needs from the peers
(one host, pairwise distinct physical GPUs by device UUID, distinct ordinals unless overlap is allowed) and returns
a NAMED verdict that every rank derives from the same gathered rows: the transport decision of every
symmetric-memory fast path is taken from it once, at prepare time, never from a local exception on one rank.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import NamedTuple

import torch


class NodeLocalityError(RuntimeError):
    """A node-local mechanism was asked to span hosts."""


_CACHE: dict = {}


def host_identity() -> str:
    """``hostname/boot-id`` of this process's machine."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "?"
    return f"{socket.gethostname()}/{boot}"


def group_hosts(group) -> list[str]:
    """The host identity of every rank of ``group`` in rank order (collective; cached per group)."""
    import torch.distributed as dist

    key = id(group)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is group:
        return hit[1]
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("node-locality admission is a collective; run it before CUDA graph capture")
    hosts = [None] * dist.get_world_size(group)
    dist.all_gather_object(hosts, host_identity(), group=group)
    _CACHE[key] = (group, hosts)
    return hosts


def one_host(group) -> bool:
    """True when every rank of ``group`` runs on the same host (collective)."""
    return len(set(group_hosts(group))) == 1


def require_one_host(group, mechanism: str) -> None:
    """Raise :class:`NodeLocalityError` naming ``mechanism`` and the hosts when ``group`` spans hosts."""
    hosts = group_hosts(group)
    if len(set(hosts)) != 1:
        by_host: dict[str, list[int]] = {}
        for rank, host in enumerate(hosts):
            by_host.setdefault(host, []).append(rank)
        raise NodeLocalityError(
            f"{mechanism} is node-local (symmetric memory inside one NVLink domain), but its group spans "
            f"{len(by_host)} hosts: {by_host}; use its NCCL realization for this group")


# ------------------------------------------------------------------------------ symmetric-memory peer admission
# mature reference: ``pik/lane.py`` ``classify`` / ``_probe`` (the trainer's physical peer policy): one gather of
# every rank's (host, device UUID, ordinal), a pure decision from the gathered rows, so every rank takes the same
# branch before any rendezvous. Deviation: the verdict names its reason (receipts) and the tried transport is not
# probed with a test rendezvous (each caller's own setup goes through ``symm_setup``'s counted agreement).

_OVERLAP_ALLOW = "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"


class SymmetricPeersError(NodeLocalityError):
    """A symmetric-memory mechanism was asked to run over a group whose ranks cannot map each other's device
    memory as distinct peers (two ranks on one GPU, ranks on several hosts, or colliding device ordinals torch
    refuses). Subclass of :class:`NodeLocalityError`, so every caller that keeps its NCCL realization for a group
    across hosts keeps it here too."""


class PeerAdmission(NamedTuple):
    """Whether a group may use symmetric memory, a short reason tag for transport receipts
    (``one_host_distinct_gpus`` | ``single_rank`` | ``multi_host`` | ``shared_device`` | ``overlapping_ordinals`` |
    ``no_cuda_device``), and a sentence naming the ranks and devices."""

    symmetric: bool
    reason: str
    detail: str = ""


_ADMISSION: dict = {}


def device_identity(device=None) -> tuple[str, int]:
    """``(UUID, ordinal)`` of this rank's current CUDA device, or ``("", -1)`` without one."""
    if not torch.cuda.is_available():
        return "", -1
    index = torch.cuda.current_device() if device is None else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    try:
        uuid = str(torch.cuda.get_device_properties(index).uuid)
    except Exception:  # noqa: BLE001 -- no uuid exposed: treated as unknown (refused below)
        uuid = ""
    return uuid, int(index)


def peer_row(device=None) -> tuple[str, str, int, bool]:
    """This rank's ``(host, device UUID, ordinal, torch overlap allowance)``."""
    uuid, ordinal = device_identity(device)
    allow = os.environ.get(_OVERLAP_ALLOW, "").strip().lower() in ("1", "true", "yes", "on")
    return host_identity(), uuid, ordinal, allow


def classify_peers(rows) -> PeerAdmission:
    """Pure, rank-uniform decision from every rank's :func:`peer_row` (group-rank order). Symmetric memory needs
    one host, pairwise distinct physical GPUs, and (torch's allocator check) distinct ordinals unless every rank
    allows overlapping ordinals."""
    rows = [tuple(r) for r in rows]
    if len(rows) < 2:
        return PeerAdmission(True, "single_rank", "a group of one rank needs no peer mapping")
    by_host: dict[str, list[int]] = {}
    for rank, row in enumerate(rows):
        by_host.setdefault(row[0], []).append(rank)
    if len(by_host) != 1:
        return PeerAdmission(False, "multi_host",
                             f"the group spans {len(by_host)} hosts {by_host}; symmetric memory is node-local")
    missing = [rank for rank, row in enumerate(rows) if not row[1]]
    if missing:
        return PeerAdmission(False, "no_cuda_device", f"ranks {missing} report no CUDA device identity")
    by_uuid: dict[str, list[int]] = {}
    for rank, row in enumerate(rows):
        by_uuid.setdefault(row[1], []).append(rank)
    shared = {u: r for u, r in by_uuid.items() if len(r) > 1}
    if shared:
        return PeerAdmission(False, "shared_device",
                             "; ".join(f"ranks {r} share one GPU ({u})" for u, r in shared.items())
                             + " (torch's symmetric memory refuses peers on one device)")
    ordinals = [row[2] for row in rows]
    if len(set(ordinals)) != len(ordinals) and not all(row[3] for row in rows):
        return PeerAdmission(False, "overlapping_ordinals",
                             f"distinct GPUs behind colliding device ordinals {ordinals} (one visible GPU per rank); "
                             f"torch refuses them unless {_OVERLAP_ALLOW}=1 on every rank")
    return PeerAdmission(True, "one_host_distinct_gpus", f"{len(rows)} ranks on one host, distinct GPUs")


def symmetric_admission(group) -> PeerAdmission:
    """The group-agreed verdict whether ``group`` may use symmetric memory (collective at the first call per group:
    every rank of the group, outside CUDA graph capture, in the same order; cached per group object afterwards).
    Every rank derives the verdict from the same gathered rows, so every rank takes the same transport; nothing is
    decided from a local exception."""
    import torch.distributed as dist

    if group is None and not dist.is_initialized():
        return PeerAdmission(True, "single_rank", "no process group")
    g = group if group is not None else dist.group.WORLD
    if dist.get_world_size(g) == 1:
        return PeerAdmission(True, "single_rank", "a group of one rank needs no peer mapping")
    key = id(g)
    hit = _ADMISSION.get(key)
    if hit is not None and hit[0] is g:
        return hit[1]
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("symmetric-memory peer admission is a collective; run it before CUDA graph capture")
    rows = [None] * dist.get_world_size(g)
    dist.all_gather_object(rows, peer_row(), group=g)
    verdict = classify_peers(rows)
    _ADMISSION[key] = (g, verdict)
    return verdict


def require_symmetric_peers(group, mechanism: str) -> None:
    """Raise, naming ``mechanism`` and the reason, when ``group`` may not use symmetric memory: across hosts
    :class:`NodeLocalityError` (as :func:`require_one_host`), otherwise :class:`SymmetricPeersError`."""
    verdict = symmetric_admission(group)
    if verdict.symmetric:
        return
    if verdict.reason == "multi_host":         # the historical error type for hosts (no second collective)
        raise NodeLocalityError(f"{mechanism} is node-local (symmetric memory inside one NVLink domain), but "
                                f"{verdict.detail}; use its NCCL realization for this group")
    raise SymmetricPeersError(f"{mechanism} needs symmetric memory between distinct GPUs, refused for this group "
                              f"({verdict.reason}: {verdict.detail}); use its NCCL realization for this group")


__all__ = ["NodeLocalityError", "PeerAdmission", "SymmetricPeersError", "classify_peers", "device_identity",
           "group_hosts", "host_identity", "one_host", "peer_row", "require_one_host", "require_symmetric_peers",
           "symmetric_admission"]
