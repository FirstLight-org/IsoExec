"""TP-sharded experts on the vendored BI kernels (``bi_tp``, the recipe program): every TP degree and every record
evaluates ``deepseek_v41_bf16_actor_v1`` bit for bit (== ``ops.moe_forward`` on the same rows). Ranks simulated in one
process (the h and output exchanges as concatenations)."""
from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe.deepseek_v41_sm100 import bi_tp, ops, rows, tp_experts  # noqa: E402

LIMIT = 10.0
KW = dict(top_k=6, routed_scale=1.5, limit=LIMIT)
GEOMETRY = [pytest.param(256, 1024, 1024, id="small"), pytest.param(384, 5120, 2304, id="v41")]


def _setup(E, H, I, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device="cuda") / H ** 0.5).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device="cuda") / I ** 0.5).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device="cuda") / H ** 0.5).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device="cuda") * 0.01
    return gu, dn, rw, rb


def _sim(experts, x, ids, w):
    fronts = [e.front(x, ids, w) for e in experts]
    h = torch.cat([f[0] for f in fronts], 1)
    return torch.cat([e.back(h, f[1]) for e, f in zip(experts, fronts)], 1)


@pytest.mark.parametrize("E,H,I", GEOMETRY)
def test_every_tp_degree_and_record_is_the_recipe(E, H, I):
    gu, dn, rw, rb = _setup(E, H, I)
    for T in (1, 33, 300):
        x = (torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(T)) / 2).to(
            torch.bfloat16)
        want = ops.moe_forward(x, rw, rb, gu, dn, **KW)
        rt = rows.route(x, rw, rb, top_k=6, routed_scale=1.5, num_experts=E)
        ids = torch.cat((rt.ids, torch.full((T, 1), E, dtype=torch.int32, device="cuda")), 1).contiguous()
        w = rt.probs_ext.contiguous()
        for tp in (1, 2, 4, 8):
            experts = [bi_tp.BiTPExperts(*tp_experts.shard_masters(gu, dn, tp, r), tp=tp, limit=LIMIT,
                                         exchange=None) for r in range(tp)]
            got = _sim(experts, x, ids, w)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (T, tp)
            # every candidate record pair of rank 0 gives the same slice
            e0 = experts[0]
            if tp == 8:
                h_all = [e.front(x, ids, w)[0] for e in experts]
                ref = None
                for c1 in e0._ranked(T, 1)[:3]:
                    for c2 in e0._ranked(T, 2)[:3]:
                        e0.records[(T, 1)], e0.records[(T, 2)] = c1, c2
                        h0, st = e0.front(x, ids, w)
                        out = e0.back(torch.cat([h0] + h_all[1:], 1), st)
                        ref = out if ref is None else ref
                        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (T, c1, c2)
                e0.records.clear()
