"""Real ``torch.distributed`` point-to-point hand-offs (gloo, 2 and 4 processes): every rank runs the
contiguous CP attention core and the GDN conv + state core concurrently, with ``DistTransport``.
Forward rows must equal a single-process CP=1 reference bit for bit; gradients must match it to fp32
rounding; a record/replay recompute must reproduce the plain-mode gradients exactly.
"""

from __future__ import annotations

import pytest

from isoexec.ops.collectives.tests._gloo_harness import assert_claim, run_world

CU = [0, 12, 40, 48, 64, 96]  # one sequence spans several ranks at cp=4


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch
    import torch.distributed as dist

    from isoexec.ops.collectives import tests as _t  # noqa: F401
    from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan
    from isoexec.ops.collectives.cp_transport import DistTransport
    from isoexec.ops.collectives.tests import test_cp_simulated_cpu as sim

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    transport = DistTransport(dist.group.WORLD)
    transport.warmup(torch.device("cpu"))
    total = CU[-1]
    plan = ContiguousPlan.build(CU, world, rank)
    rows = slice(plan.lo, plan.lo + plan.local)

    class OneRank(sim.World):
        """This process's rank only, the others are real processes."""

        def __init__(self):
            self.cp, self.plans = world, [None] * world
            self.plans[rank] = plan
            self.cus = [None] * world
            local = torch.tensor(plan.local_cu, dtype=torch.int32)
            register_plan(local, plan)
            self.cus[rank] = local

    def attention(mode=None):
        from isoexec.ops.attention.reference import ref_attention_core_qk
        from isoexec.ops.collectives.cp_attention import make_cp_attention_core
        from isoexec.ops.collectives.cp_transport import cp_region

        w = OneRank()
        q, k, v, dout = sim._attention_inputs(total, seed=3)
        core = make_cp_attention_core(ref_attention_core_qk, transport_factory=lambda: transport, cp=world,
                                      rank=rank, site="attn")
        leaves = [x[rows].clone().requires_grad_(True) for x in (q, k, v)]
        run = lambda: core(*leaves, w.cus[rank], plan.max_q, sim.D ** -0.5)  # noqa: E731
        if mode == "recompute":
            with torch.no_grad(), cp_region("record"):
                run()
            with cp_region("replay"):
                o = run()
        else:
            o = run()
        o.backward(dout[rows])
        return o.detach(), [x.grad for x in leaves]

    def gdn(mode=None):
        from isoexec.ops.collectives.cp_transport import cp_region
        from isoexec.ops.gdn.cp import make_cp_conv, make_cp_gdn_core
        from isoexec.ops.gdn.reference import ref_gdn_core_stateful

        w = OneRank()
        x, a, b, cw, A_log, dt_bias, dout = sim._gdn_inputs(total, seed=4)
        conv = make_cp_conv(sim.tap_conv, transport_factory=lambda: transport, cp=world, rank=rank, site="g.conv")
        core = make_cp_gdn_core(ref_gdn_core_stateful, transport_factory=lambda: transport, cp=world, rank=rank,
                                site="g.core")
        leaves = [t[rows].clone().requires_grad_(True) for t in (x, a, b)]
        params = [p.clone().requires_grad_(True) for p in (cw, A_log, dt_bias)]
        kd = sim.GK * sim.DK

        def run():
            y = conv(leaves[0], params[0], w.cus[rank])
            q = y[:, :kd].unflatten(-1, (sim.GK, sim.DK))
            k = y[:, kd:2 * kd].unflatten(-1, (sim.GK, sim.DK))
            v = y[:, 2 * kd:].unflatten(-1, (sim.GV, sim.DV))
            return core(q, k, v, leaves[1], leaves[2], params[1], params[2], w.cus[rank])

        if mode == "recompute":
            with torch.no_grad(), cp_region("record"):
                run()
            with cp_region("replay"):
                o = run()
        else:
            o = run()
        o.backward(dout[rows])
        return o.detach(), [t.grad for t in leaves]

    res = {}
    # CP=1 references, computed locally by the simulator.
    q, k, v, dout = sim._attention_inputs(total, seed=3)
    ref_o, ref_g = sim._attention(sim.World(CU, 1), q, k, v, dout)
    o, g = attention()
    res["attn_fwd_bitwise"] = torch.equal(o, ref_o[rows])
    res["attn_bwd_close"] = all(torch.allclose(x, y[rows], rtol=2e-5, atol=2e-5) for x, y in zip(g, ref_g))
    o2, g2 = attention(mode="recompute")
    res["attn_replay_exact"] = torch.equal(o2, o) and all(torch.equal(x, y) for x, y in zip(g2, g))

    inputs = sim._gdn_inputs(total, seed=4)
    ref_o, ref_g = sim._gdn(sim.World(CU, 1), *inputs)
    o, g = gdn()
    res["gdn_fwd_bitwise"] = torch.equal(o, ref_o[rows])
    res["gdn_bwd_close"] = all(torch.allclose(x, y[rows], rtol=2e-5, atol=2e-5) for x, y in zip(g, ref_g[:3]))
    o2, g2 = gdn(mode="recompute")
    res["gdn_replay_exact"] = torch.equal(o2, o) and all(torch.equal(x, y) for x, y in zip(g2, g))
    transport.flush()
    dist.barrier()
    out[rank] = res
    dist.destroy_process_group()


CLAIMS = ("attn_fwd_bitwise", "attn_bwd_close", "attn_replay_exact", "gdn_fwd_bitwise", "gdn_bwd_close",
          "gdn_replay_exact")


@pytest.mark.parametrize("world", [2, 4])
def test_dist_transport_cp_matches_cp1(world, tmp_path):
    results = run_world(_body, world=world, rendezvous=tmp_path / "rdzv")
    for claim in CLAIMS:
        assert_claim(results, claim, world)
