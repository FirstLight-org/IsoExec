"""Pipelined row-parallel site: token chunks so the producer GEMM of chunk i+1 overlaps the owner reduce of
chunk i on the group's comm lane (``lane.py``). Schedule only -- every output row is reduced by the same
admitted owner class over the same leaf planes, so the per-element expression is the unpipelined one.

Kept apart from ``owner.py`` (the kernel protocol); it uses owner's public pieces: ``stage`` (with a staging
slot), ``_produce``, ``_resolve``/``_execute`` and ``legacy``.
"""

from __future__ import annotations

import torch

from .owner import _execute, _produce, _resolve, legacy, stage

#: Row granule of a pipeline chunk: a multiple of every GEMM M tile, so chunking adds no partial tiles.
_CHUNK_ROWS_ALIGN = 128
PIPE_COUNTS = {"pipelined_sites": 0, "chunks": 0, "serial_sites": 0, "nccl_by_size": 0}


def pick_state(st, nbytes: int):
    """The owner state to reduce ``nbytes`` (this rank's wire planes) with: an admitted P2P group moves a
    reduce above its P2P cap (``st.p2p_max_bytes``, agreed by the group's probe, see ``lane.p2p_max_bytes``)
    over NCCL instead (the NCCL owner transport measured faster there). Rank-uniform: a pure function of the
    size and the group-agreed threshold. Serves the serial, pipelined and dgrad paths alike."""
    cap = getattr(st, "p2p_max_bytes", 0) if st is not None else 0
    if st is None or st.transport != "p2p" or not cap or nbytes <= cap:
        return st
    sh = getattr(st, "_nccl_shadow", None)
    if sh is None:
        from .owner import _State

        sh = _State(st.on, "nccl", st.oneshot_bytes, st.max_blocks, None)
        for a in ("twoshot", "leaf_min_m", "node_below", "single_host", "p2p_max_bytes"):
            if hasattr(st, a):
                setattr(sh, a, getattr(st, a))
        st._nccl_shadow = sh
    PIPE_COUNTS["nccl_by_size"] += 1
    return sh


def site_chunks(st, M: int, N: int, k_local: int, world: int, planes: int, wdt, root) -> int:
    """The site's chunk count: ``ISOEXEC_PIK_OVERLAP_CHUNKS`` if set, else ``lane.plan_chunks`` from the
    site's shape, TP size, wire, transport and SM count (all rank-uniform; the SM count is group-agreed)."""
    from . import lane as _lane

    explicit = _lane.overlap_chunks()
    if explicit >= 0:
        return max(1, explicit)  # 0 = off (default), n = fixed depth
    if _lane.hw_queues() <= 1:  # one hardware queue: streams cannot overlap, chunks would only add launches
        return 1
    from .owner import _ELT

    link = st.transport if st.transport == "p2p" else ("nccl" if getattr(st, "single_host", True) else
                                                     "nccl_internode")
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _lane.plan_chunks(M, N, k_local, world, planes, _ELT[wdt], _ELT[root], link, sms)


def pipeline_bounds(M: int, chunks: int, min_m: int) -> list[tuple[int, int]]:
    """Row ranges of the pipelined site (one range = no pipeline). Pure: rank-uniform for a given M."""
    if chunks < 2 or M < max(min_m, 2 * _CHUNK_ROWS_ALIGN):
        return [(0, M)]
    rows = -(-M // chunks)
    rows = -(-rows // _CHUNK_ROWS_ALIGN) * _CHUNK_ROWS_ALIGN
    return [(r, min(M, r + rows)) for r in range(0, M, rows)]


def row_parallel_pipelined(x2, w, bounds, st, group, root, *, plan, planes, wdt, m_leaves, tp_size, tp_rank,
                            k_full):
    """GEMM of chunk i+1 on the caller's stream overlaps the reduce of chunk i on the group's lane.

    Every output row is one chunk's row, reduced by the same admitted kernel class over the same planes:
    the per-element expression (leaves, tree, root round) is the unpipelined one. Staging alternates two
    symmetric slots; the GEMM writing slot s waits for the lane event of the reduce that last read s
    (that reduce retiring on this rank implies every peer finished reading it: the kernel's second barrier).
    """
    from . import lane as _lane

    M, N = x2.shape[0], w.shape[0]
    dev = x2.device
    ln = _lane.lane(group, dev)
    main = torch.cuda.current_stream(dev)
    out = torch.empty((M, N), dtype=root, device=dev)
    free = [None, None]
    sms = _lane.overlap_comm_sms()
    for i, (r0, r1) in enumerate(bounds):
        s = i & 1
        if free[s] is not None:
            main.wait_event(free[s])
        if i == 0:
            _lane.fence(dev)  # anything else on a lane (an early dgrad sum) retires before slot reuse
        staged = stage(st, planes, r1 - r0, N, wdt, dev, slot=s, fenced=False)
        with _sm_budget(sms if i > 0 else 0):  # chunk 0 overlaps nothing on this site
            _produce(x2[r0:r1], w, staged, plan=plan, planes=planes, m_leaves=m_leaves, tp_size=tp_size,
                     tp_rank=tp_rank, k_full=k_full)
        flat = staged.view(planes, -1)
        cst = pick_state(st, flat.numel() * flat.element_size())
        mode, ok = _resolve(flat, group, root, cst)  # cached after the first site of a class (rank-uniform)
        if not ok:  # this class serves the legacy path: finish the chunk synchronously, same bits
            main.wait_stream(ln.stream)
            out[r0:r1].view(-1).copy_(legacy(flat, group, root))
            continue
        if cst.transport != "p2p":
            # Never NCCL on the lane (cross-communicator order with main-stream NCCL): this chunk's reduce
            # runs on the caller's stream, after the lane's earlier chunks.
            main.wait_stream(ln.stream)
            _execute(cst, flat, group, root, mode, out=out[r0:r1].view(-1), slot=s)
            continue
        ls = ln.enter()
        with torch.cuda.stream(ls):
            _execute(cst, flat, group, root, mode, out=out[r0:r1].view(-1), slot=s)
        free[s] = ln.leave()
        PIPE_COUNTS["chunks"] += 1
    out.record_stream(ln.stream)
    main.wait_stream(ln.stream)
    PIPE_COUNTS["pipelined_sites"] += 1
    return out


class _sm_budget:
    """Cap the producer GEMM's persistent CTAs so ``reserve`` SMs stay free for the lane (0 = no cap)."""

    def __init__(self, reserve: int):
        self.reserve = reserve
        self.prev = None

    def __enter__(self):
        if self.reserve > 0:
            try:
                from isoexec.ops.mm.mm_sm90 import ops as sm90

                self.prev = sm90.set_sm_limit(max(1, sm90.device_sm_count() - self.reserve))
            except Exception:  # noqa: BLE001 -- no sm90 GEMM here: nothing to cap
                self.prev = None
        return self

    def __exit__(self, *exc):
        if self.reserve > 0 and self.prev is not None:
            from isoexec.ops.mm.mm_sm90 import ops as sm90

            sm90.set_sm_limit(self.prev)
        return False


