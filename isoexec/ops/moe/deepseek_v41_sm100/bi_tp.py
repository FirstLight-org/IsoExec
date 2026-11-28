"""TP-sharded experts (``tp_experts``'s data flow) on the vendored FlashInfer batch-invariant BF16 MoE kernels
(``isoexec/ops/_vendor/flashinfer_bi_moe``): plan, GEMM1 on the rank's I slice with the gated epilogue chain, h
exchange, GEMM2 on the rank's output slice (one K chain over the global I), slot-order combine, output exchange.

Placement: the rank's masters re-laid out once for the kernels (GEMM1 rows [up | gate] interleaved, the BI layout;
GEMM2 the output slice as is). Records (tile, swap, cluster, raster: placement; the bits do not depend on them): the
checked-in table ``bi_records.TABLE`` keyed by (tp, tokens, GEMM), else the dispatcher's first record; both are fixed
functions of the shape, identical on every rank and run. ``warm(sizes)`` compiles them (outside capture: the in-process
kernel cache refuses a compile inside CUDA graph capture). ``tune(sizes)`` is the opt-in tool that times the
dispatcher's top-ranked records and returns table entries in ``bi_records.TABLE``'s format; it changes nothing.

Program: ``deepseek_v41_bf16_actor_v1`` (the vendored ``v41_actor`` GEMM1 chain with the route weight before FC2, the
from-first unit-weight combine that reads no weights): the bits of ``ops.moe_forward`` / ``tp_experts``.
"""
from __future__ import annotations

import torch

from .tp_experts import check_tp

SLOTS = 7        # top-6 routed + the shared expert as a slot
CANDIDATES = 6   # records measured per (GEMM, size) at warm-up


def _bi():
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import api
    return api


def _sched():
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import schedule
    return schedule


class LocalExchange:
    """TP = 1: the exchanges are identities."""

    @staticmethod
    def gather_h(h):
        return h

    @staticmethod
    def gather_out(y):
        return y


class BiTPExperts:
    def __init__(self, gate_up_r, down_r, *, tp: int, limit: float, exchange):
        from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
        bi = _bi()
        G, I2s, H = gate_up_r.shape
        Is = I2s // 2
        self.G, self.H, self.Is, self.I = G, H, Is, down_r.size(2)
        self.Hs = down_r.size(1)
        self.tp = tp
        check_tp(self.I, self.Hs * tp, tp)
        # BI's canonical fc1 is [up rows | gate rows]; IsoExec's masters are [gate | up]
        fc1 = torch.cat((gate_up_r[:, Is:], gate_up_r[:, :Is]), 1)
        self.w1i = bi.interleave_fc1(fc1)
        self.chain1 = GroupedChain("glu", "v41_actor", "bf16", (float(limit),))
        self.w2 = down_r.contiguous()
        self.exchange = exchange
        self.records: dict = {}
        self.sm, self.fam = bi._part(gate_up_r.device)

    # ------------------------------------------------------------------------------------------------ records
    def _ranked(self, M, gemm):
        sch = _sched()
        rows, nact = sch.rows_per_expert(M, SLOTS, self.G), sch.active_experts(M, SLOTS, self.G)
        if gemm == 1:
            return sch.rank_grouped(rows, 2 * self.Is, self.H, self.G, self.chain1, self.sm, self.fam, CANDIDATES,
                                    False, nact)
        return sch.rank_grouped(rows, self.Hs, self.I, self.G, "plain", self.sm, self.fam, CANDIDATES, False, nact)

    def record(self, M, gemm):
        hit = self.records.get((M, gemm))         # an explicit override (tests, tuning) first
        if hit is not None:
            return hit
        from .bi_records import lookup
        hit = lookup(self.tp, M, gemm)
        return hit if hit is not None else self._ranked(M, gemm)[0]

    def warm(self, sizes):
        """Compile the records of every size (the table's or the dispatcher's first); outside capture."""
        self._sweep(sizes, measure=False)

    def tune(self, sizes):
        """Opt-in tuning tool: time the dispatcher's top-ranked records per (GEMM, size) and return
        ``{(tp, size, gemm): Knobs}`` of the fastest (``bi_records.TABLE`` entries); the op's records are unchanged."""
        return self._sweep(sizes, measure=True)

    def _sweep(self, sizes, measure):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the TP experts outside CUDA graph capture")
        found = {}
        dev = self.w1i.device
        for M in sizes:
            ids = torch.stack([torch.randperm(self.G - 1, device=dev)[:SLOTS - 1] for _ in range(M)]).sort(1).values
            ids = torch.cat((ids, torch.full((M, 1), self.G - 1, dtype=torch.int64, device=dev)), 1).int().contiguous()
            w = torch.rand(M, SLOTS, device=dev)
            x = torch.zeros((M, self.H), dtype=torch.bfloat16, device=dev)
            if not measure:
                self._local(x, ids, w, exchange=False)            # compiles the table's / dispatcher's records
                continue
            for gemm in (1, 2):
                best, best_t = None, None
                cands = list(self._ranked(M, gemm))
                if M >= 2048:                         # the 2-CTA 256-row records the cost model under-ranks at prefill
                    sch = _sched()
                    N, K, ch = (2 * self.Is, self.H, "swiglu") if gemm == 1 else (self.Hs, self.I, "plain")
                    cands += [c for c in sch.all_valid_configs(N, K, self.G, ch)
                              if c.use_2cta_instrs and c.tile_m == 256 and c.tile_n in (128, 256) and c not in cands]
                for cfg in cands:
                    self.records[(M, gemm)] = cfg
                    self._local(x, ids, w, exchange=False)       # compiles
                    torch.cuda.synchronize()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(5):
                        self._local(x, ids, w, exchange=False)
                    b.record()
                    torch.cuda.synchronize()
                    t = a.elapsed_time(b)
                    if best_t is None or t < best_t:
                        best, best_t = cfg, t
                found[(self.tp, M, gemm)] = best
                self.records.pop((M, gemm), None)
        torch.cuda.synchronize()
        return found

    # ------------------------------------------------------------------------------------------------ forward
    def front(self, x, ids, w):
        """Plan and GEMM1 on this rank's I slice: ``(h_r [Tk, I/tp] bf16, state)``."""
        bi = _bi()
        M = x.size(0)
        Tk = M * SLOTS
        c1, c2 = self.record(M, 1), self.record(M, 2)
        pad = max(c1.tile_n if c1.swap_ab else 0, c2.tile_n if c2.swap_ab else 0)
        counts, dest, src, rw = bi.permute_plan_weighted(ids, self.G, w)   # one launch at decode
        kw = dict(row_weights=rw)                               # the route weight enters GEMM1 (before FC2)
        hb = getattr(self.exchange, "h_buffer", None)             # NVLink exchange: GEMM1 writes into its region
        h = hb(Tk + pad, self.Is) if hb is not None else torch.empty((Tk + pad, self.Is), dtype=torch.bfloat16,
                                                                     device=x.device)
        if c1.swap_ab and c1.tile_n <= bi.GATHER_TOKEN_TILE_MAX:
            bi.grouped_gemm_gather(x, src, SLOTS, self.w1i, counts, c1, self.chain1, out=h[:Tk], Tk=Tk, **kw)
        else:
            from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant.permute import (
                get_compiled_gather, warps_per_row)
            xp = torch.empty((Tk + pad, self.H), dtype=torch.bfloat16, device=x.device)
            if Tk:
                get_compiled_gather(SLOTS, "bf16", warps_per_row(Tk, self.H // 8, bi.sm_count_of(x.device)))(
                    x, src, xp[:Tk], Tk, self.H)
            bi.grouped_gemm(xp, self.w1i, counts, c1, self.chain1, out=h[:Tk], Tk=Tk, **kw)
        return h[:Tk], (counts, dest, w, c2, pad, Tk, h)

    def back(self, h_full, state):
        """GEMM2 on this rank's output slice over the gathered ``h [Tk, I]`` and the slice combine -> ``[M, H/tp]``."""
        bi = _bi()
        counts, dest, w, c2, pad, Tk, _ = state
        if pad and h_full.size(0) < Tk + pad:
            # a swapped GEMM2 reads tile_n rows past Tk: they only feed output rows >= Tk, which the store clips, so
            # their values are never read into a result (no zero fill needed)
            hf = torch.cat((h_full, h_full.new_empty((pad, self.I))), 0)
        else:
            hf = h_full
        y = bi.grouped_gemm(hf, self.w2, counts, c2, "plain", torch.bfloat16, Tk=Tk)
        # unit-weight add chain from the first term, slot order (ascending id, shared last); reads no weights
        ob = getattr(self.exchange, "out_buffer", None)           # NVLink exchange: the combine writes in place
        out = ob(w.size(0), self.Hs) if ob is not None else None
        return bi.combine_unfinalized(y, None, dest, out=out, from_first=True, tokens=w.size(0))

    def _local(self, x, ids, w, exchange=None):
        h, state = self.front(x, ids, w)
        if self.tp == 1 and isinstance(exchange or self.exchange, LocalExchange):
            return self.back(state[-1], state)      # TP = 1: GEMM1's padded buffer is GEMM2's input (no copy)
        if exchange is False:                    # one rank's compute only (timing): a zero h_full stands in
            hf = torch.zeros((h.size(0), self.I), dtype=torch.bfloat16, device=x.device)
        else:
            hf = (exchange or self.exchange).gather_h(h)
        return self.back(hf, state)

    def forward(self, x, ids, w):
        """``x [M, H]`` replicated bf16, ``ids [M, 7]`` int32 (shared slot id G - 1 last), ``w [M, 7]`` fp32 ->
        ``[M, H]`` replicated bf16."""
        return self.exchange.gather_out(self._local(x, ids, w, self.exchange))


__all__ = ["BiTPExperts", "CANDIDATES", "LocalExchange", "SLOTS"]
