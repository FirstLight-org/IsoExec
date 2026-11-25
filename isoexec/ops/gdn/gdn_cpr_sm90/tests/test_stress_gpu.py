"""Stress: many stateful packs through CUDA graphs, slots reused and moved, against eager calls.

A pool of records serves a stream of prefill and decode calls with random packs: sequences continue,
move to other slots, restart in slots that held other sequences, and leave slots stale. Each call shape is
captured once in a CUDA graph and replayed with new contents written into the captured input tensors
(inputs, cu_seqlens, slots). The same stream of calls is also run eagerly on a second pool. Outputs and
every meaningful record section must agree bitwise after every call.

Inputs are made on the current stream and the replays run on it (a graph's work reads the tensors the
current stream wrote, so no cross-stream race can masquerade as a kernel fault -- see README).
"""

from __future__ import annotations

import random

import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, I32, SM90, state_equal

pytestmark = SM90

HK, HV, POOL = 2, 4, 12


class _Graphed:
    """One captured call shape: static input tensors, refreshed in place before each replay. Warming up
    runs the call once on the pool, so the pool is restored before the capture (which runs nothing)."""

    def __init__(self, fn, statics, pool):
        self.statics = statics
        snapshot = pool.data.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn(**statics)
        torch.cuda.current_stream().wait_stream(side)
        pool.data.copy_(snapshot)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(**statics)

    def __call__(self, **values):
        for name, value in values.items():
            self.statics[name].copy_(value)
        self.graph.replay()
        return self.out


def _inputs(rng, T, N):
    g = torch.Generator(device=DEV).manual_seed(rng.randrange(1 << 30))
    r = lambda *s: torch.randn(*s, device=DEV, generator=g).to(BF)
    return dict(q=r(T, HK, 128), k=r(T, HK, 128), v=r(T, HV, 128), a=r(T, HV), b=r(T, HV))


def test_graphs_with_slot_reuse_match_eager():
    rng = random.Random(7)
    g0 = torch.Generator(device=DEV).manual_seed(5)
    A_log = torch.log(torch.empty(HV, device=DEV).uniform_(1, 16, generator=g0))
    dt_bias = torch.randn(HV, device=DEV, generator=g0) * 0.5
    eager, graphed = G.CprState.allocate(POOL, HV, DEV, zero=False), G.CprState.allocate(POOL, HV, DEV, zero=False)
    for st in (eager, graphed):
        st.data.fill_(0xFF)   # NaN-poisoned pool: every record must be (re)started before it is read
        st.reset(torch.arange(POOL, device=DEV))
    # call shapes: prefill of N sequences over T tokens, decode of N sequences
    shapes = [("prefill", 300, 3), ("prefill", 130, 4), ("decode", 5, 5), ("prefill", 70, 2), ("decode", 3, 3)]
    graphs = {}

    def prefill_fn(q, k, v, a, b, cu, si, so):
        return G.prefill(q, k, v, a, b, A_log, dt_bias, cu, state=graphed, slot_in=si, slot_out=so)

    def decode_fn(q, k, v, a, b, si, so):
        return G.decode(q, k, v, a, b, A_log, dt_bias, graphed, si, slot_out=so)

    for step in range(30):
        kind, T, N = shapes[step % len(shapes)]
        # choose N distinct output slots; each continues a slot (possibly moving) or starts fresh (NULL in)
        outs = rng.sample(range(POOL), N)
        ins = []
        for so in outs:
            choice = rng.random()
            if choice < 0.5:
                ins.append(so)                                   # continue in place
            elif choice < 0.75:
                ins.append(-1)                                   # a new sequence in a used slot
            else:
                free = [s for s in range(POOL) if s not in outs and s not in ins]
                ins.append(rng.choice(free) if free else -1)     # move a sequence to another slot
        if kind == "prefill":   # random cut points; repeated ones make empty sequences
            cu = torch.tensor([0] + sorted(rng.choices(range(T + 1), k=N - 1)) + [T], device=DEV, dtype=I32)
        x = _inputs(rng, T if kind == "prefill" else N, N)
        si, so = torch.tensor(ins, device=DEV, dtype=I32), torch.tensor(outs, device=DEV, dtype=I32)
        if kind == "prefill":
            want = G.prefill(**x, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu, state=eager, slot_in=si, slot_out=so)
            key = (kind, T, N)
            if key not in graphs:
                graphs[key] = _Graphed(prefill_fn, dict(**{n: t.clone() for n, t in x.items()}, cu=cu.clone(),
                                                        si=si.clone(), so=so.clone()), graphed)
            got = graphs[key](**x, cu=cu, si=si, so=so)
        else:
            want = G.decode(**x, A_log=A_log, dt_bias=dt_bias, state=eager, slot_in=si, slot_out=so)
            key = (kind, T, N)
            if key not in graphs:
                graphs[key] = _Graphed(decode_fn, dict(**{n: t.clone() for n, t in x.items()}, si=si.clone(), so=so.clone()),
                                       graphed)
            got = graphs[key](**x, si=si, so=so)
        torch.cuda.synchronize()
        assert bitwise_equal(got, want), f"step {step}: outputs differ"
        assert state_equal(eager, graphed, outs), f"step {step}: records differ"
