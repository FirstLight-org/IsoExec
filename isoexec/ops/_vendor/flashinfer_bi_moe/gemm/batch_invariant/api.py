# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified for IsoExec: only the device / part plumbing the MoE classes use (sm_count_of, part_of,
# persistent_clusters) is kept from flashinfer/gemm/batch_invariant/api.py; the FP8 / FP4 / dense GEMM entry points are
# not copied.
"""Device part and persistent-grid queries of the batch-invariant GEMM classes (FlashInfer v0.7.5rc3, trimmed)."""

from __future__ import annotations

import functools
import logging
from typing import Dict, Tuple

import torch

from .generator import Knobs

logger = logging.getLogger(__name__)

def _device_key(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def sm_count_of(device: torch.device) -> int:
    return _sm_count_cached(_device_key(device))


# the devices' answers to the occupancy query, keyed (device index, cluster size). A failed query is retried on the
# next call up to _QUERY_ATTEMPTS times; then the part's nominal capacity is kept in its place (logged once), so a device
# whose query cannot succeed is neither asked again nor re-ranked on every call.
_CO_RESIDENT: Dict[Tuple[int, int], int] = {}
_QUERY_FAILURES: Dict[Tuple[int, int], int] = {}
_QUERY_ATTEMPTS = 3
_REGISTERED_DEVICES: set = set()


def _query_co_resident_clusters(device_index: int, cluster_size: int) -> int:
    """The CuTe DSL's GPC-aware occupancy query for a one-CTA-per-SM body, with the device's primary context made
    current on the calling thread for its duration (the query needs one; the previous current context is restored).
    No stream work and no synchronisation: legal during CUDA graph capture."""
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cu  # the DSL's own driver binding

    def ok(result):
        err, *values = result
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{err}")
        return values[0] if values else None

    ok(cu.cuInit(0))
    dev = ok(cu.cuDeviceGet(device_index))
    prev = ok(cu.cuCtxGetCurrent())
    ctx = ok(cu.cuDevicePrimaryCtxRetain(dev))
    try:
        ok(cu.cuCtxSetCurrent(ctx))
        n = int(
            cutlass_utils.HardwareInfo(device_id=device_index).get_max_active_clusters(
                cluster_size
            )
        )
    finally:
        ok(cu.cuCtxSetCurrent(prev))
        ok(cu.cuDevicePrimaryCtxRelease(dev))
    if n <= 0:
        raise RuntimeError(f"the occupancy query returned {n}")
    return n


def _co_resident_clusters(
    device_index: int, cluster_size: int, sm_count: int, family: str
) -> int:
    """How many clusters of ``cluster_size`` CTAs the device can host at once: the device's answer, asked once and kept.
    A failed query (logged) gives the part's nominal capacity; the next call asks again, up to ``_QUERY_ATTEMPTS`` times
    in all, after which the nominal capacity is kept for the process like an answer (a launch parameter: the bits never
    depend on it)."""
    from .heuristic import capacity_for

    key = (device_index, cluster_size)
    if key not in _CO_RESIDENT:
        try:
            _CO_RESIDENT[key] = _query_co_resident_clusters(device_index, cluster_size)
        except Exception as e:  # noqa: BLE001
            nominal = capacity_for(family, sm_count).get(cluster_size) or max(
                1, sm_count // cluster_size
            )
            _QUERY_FAILURES[key] = _QUERY_FAILURES.get(key, 0) + 1
            if _QUERY_FAILURES[key] >= _QUERY_ATTEMPTS:
                _CO_RESIDENT[key] = nominal
                logger.warning(
                    "batch_invariant: the co-resident cluster query failed %d times on device %d (%s: %s); the part's "
                    "nominal cluster capacity (%d clusters of %d CTAs) is kept for this process",
                    _QUERY_FAILURES[key],
                    device_index,
                    type(e).__name__,
                    e,
                    nominal,
                    cluster_size,
                )
            else:
                logger.warning(
                    "batch_invariant: the co-resident cluster query failed on device %d (%s: %s); the part's nominal "
                    "cluster capacity is used for this call",
                    device_index,
                    type(e).__name__,
                    e,
                )
            return nominal
    return _CO_RESIDENT[key]


def _register_device(device_index: int, sm_count: int, family: str) -> None:
    """Once per device (again after a failed query, until its retries are spent): the device's L2 size and the
    co-resident cluster counts of the cluster sizes the space uses, into the heuristic's tables; a query's answer, or
    the nominal capacity kept after the retries, is recorded."""
    if device_index in _REGISTERED_DEVICES:
        return
    from .generator import CLUSTERS
    from .heuristic import register_capacity, register_l2

    register_l2(
        family, sm_count, torch.cuda.get_device_properties(device_index).L2_cache_size
    )
    complete = True
    for size in sorted({cm * cn for cm, cn in CLUSTERS}):
        n = _co_resident_clusters(device_index, size, sm_count, family)
        if (device_index, size) in _CO_RESIDENT:
            register_capacity(family, sm_count, size, n)
        else:
            complete = False
    if complete:
        _REGISTERED_DEVICES.add(device_index)


def persistent_clusters(device: torch.device, cfg: Knobs) -> int:
    """The persistent grid in clusters for ``cfg`` on ``device``: the device's co-resident cluster count for the knobs'
    cluster shape (cached per device and cluster size), so that the compiled grid is what the device can run at once;
    a launch parameter that decides which CTA computes which tile and nothing else (the bits never depend on it). A
    grid of ``SMs // size`` would leave its surplus clusters to run as a serial tail under the static persistent
    assignment whenever a GPC cannot host a whole cluster of every SM."""
    if cfg.grid_clusters:
        return cfg.grid_clusters
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm, fam = part_of(device)
    return _co_resident_clusters(index, cfg.cluster_m * cfg.cluster_n, sm, fam)


def part_of(device: torch.device) -> Tuple[int, str]:
    """The part the dispatcher models for ``device``: its SM count and architecture family (`heuristic.family_of`); the
    first call for a device also records its L2 size and co-resident cluster capacity for the heuristic."""
    index = _device_key(device)
    sm, fam = _part_cached(index)
    _register_device(index, sm, fam)
    return sm, fam


# Both are device properties, fixed for the process, and are read on every call: cached per device index.
@functools.lru_cache(maxsize=None)
def _sm_count_cached(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _part_cached(index: int) -> Tuple[int, str]:
    from ...batch_invariant.support import compute_capability
    from .heuristic import family_of

    return _sm_count_cached(index), family_of(
        compute_capability(torch.device("cuda", index))
    )

