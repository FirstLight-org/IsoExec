"""Context-parallel forward/backward tests using two CPU Gloo ranks."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.collectives.tests._gloo_harness import assert_claim, run_world

WORLD = 2
CU = [0, 8, 20, 24]  # padded lengths 8, 12, 4: multiples of 2*cp
HQ, HK, D = 4, 2, 3


def _placement(cp: int, rank: int, group):
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=Axis(cp, rank, group), side="trainer", leaves=8)


def _eager_core(q, k, v, cu_seqlens, max_q, scale):
    """Causal GQA attention per packed sequence, in the input dtype: the CP=1 reference AND the inner core."""
    out = torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=q.dtype)
    rep = q.shape[1] // k.shape[1]
    cu = [int(x) for x in cu_seqlens.tolist()]
    for a, b in zip(cu, cu[1:]):
        n = b - a
        mask = torch.ones((n, n), dtype=torch.bool).tril()
        kk, vv = k[a:b].repeat_interleave(rep, dim=1), v[a:b].repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[a:b], kk) * scale
        probs = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        out[a:b] = torch.einsum("hqk,khd->qhd", probs, vv)
    return out


def _install_gloo_shims():
    import torch.distributed as dist

    def reduce_scatter_tensor(out, inp, op=None, group=None):
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        total = inp.clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
        out.copy_(total.view(world, -1, *inp.shape[1:])[rank])

    dist.reduce_scatter_tensor = reduce_scatter_tensor


def _body(rank: int, world: int, out: dict, init_method: str) -> None:
    import torch.distributed as dist

    from isoexec.ops.collectives.cp_gather import build_layout, layout_for, make_cp_attention_core, make_cp_gather
    from isoexec.ops.collectives.cp_layout import local_to_global

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    _install_gloo_shims()
    group = dist.group.WORLD
    pl = _placement(world, rank, group)
    res: dict = {}
    total = CU[-1]
    cu = torch.tensor(CU, dtype=torch.int32)
    mine = local_to_global(CU, world, rank)
    layout = layout_for(cu, world, rank)
    res["layout_is_cached_per_boundary_tensor"] = layout_for(cu, world, rank) is layout
    res["layout_matches_a_fresh_build"] = bool(torch.equal(layout.mine, build_layout(CU, world, rank, "cpu").mine))

    # (1) gather == the global packed stream
    stream = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    gather = make_cp_gather(pl)
    local = stream[mine].clone().requires_grad_(True)
    res["gather_restores_the_global_stream"] = bool(torch.equal(gather(local, layout).detach(), stream))
    with torch.no_grad():
        res["gather_moves_data_without_autograd"] = bool(torch.equal(gather(stream[mine], layout), stream))

    # (2) adjoint: SUM over ranks of each rank's gradient, this rank's rows (integer-valued: exact)
    coeff = torch.arange(1, total * 2 + 1, dtype=torch.float32).reshape(total, 2)
    (gather(local, layout) * (coeff * float(rank + 1))).sum().backward()
    res["gather_adjoint_is_the_group_sum_of_own_rows"] = bool(torch.equal(local.grad, (coeff * 3.0)[mine]))

    # (3) + (4) the wrapped attention core against the CP=1 call
    gen = torch.Generator().manual_seed(7)
    full = [torch.randn((total, h, D), dtype=torch.float64, generator=gen) for h in (HQ, HK, HK)]
    weight = torch.randn((total, HQ, D), dtype=torch.float64, generator=gen)
    scale = D**-0.5
    ref_in = [t.clone().requires_grad_(True) for t in full]
    ref_out = _eager_core(*ref_in, cu, 12, scale)
    (ref_out * weight).sum().backward()

    core = make_cp_attention_core(_eager_core, pl)

    def step():
        leaves = [t[mine].clone().requires_grad_(True) for t in full]
        o = core(*leaves, cu, 12, scale)
        (o * weight[mine]).sum().backward()
        return o.detach(), [t.grad.clone() for t in leaves]

    o1, g1 = step()
    o2, g2 = step()
    res["kept_rows_are_the_cp1_rows_bitwise"] = bool(torch.equal(o1, ref_out.detach()[mine]))
    res["core_reports_its_layout"] = core.cp_layout == "gather_full_keep_owned"
    res["gradients_match_cp1"] = all(
        torch.allclose(g, r.grad[mine], rtol=0, atol=1e-12) for g, r in zip(g1, ref_in)
    )
    res["gradients_are_repeatable"] = all(torch.equal(a, b) for a, b in zip(g1, g2)) and bool(torch.equal(o1, o2))
    with torch.no_grad():
        res["forward_without_autograd"] = bool(torch.equal(core(*[t[mine] for t in full], cu, 12, scale), o1))

    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_world(_body, world=WORLD, rendezvous=tmp_path_factory.mktemp("cp-gather") / "store")


CLAIMS = (
    "layout_is_cached_per_boundary_tensor",
    "layout_matches_a_fresh_build",
    "gather_restores_the_global_stream",
    "gather_moves_data_without_autograd",
    "gather_adjoint_is_the_group_sum_of_own_rows",
    "kept_rows_are_the_cp1_rows_bitwise",
    "core_reports_its_layout",
    "gradients_match_cp1",
    "gradients_are_repeatable",
    "forward_without_autograd",
)


@pytest.mark.parametrize("claim", CLAIMS)
def test_context_parallel_wire(results, claim):
    assert_claim(results, claim, WORLD)


def test_other_degrees_and_a_missing_group_refuse_before_any_collective():
    from isoexec.ops.collectives.cp_gather import CpGatherError, make_cp_attention_core, make_cp_gather
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)

    def placement(cp):
        p = object.__new__(Placement)  # bypass Placement's own admission: this is the wire's refusal
        for name, value in dict(tp=one, ep=one, dp=one, pp=one, etp=one, cp=cp, side="trainer", leaves=8,
                                leaf_dtype="bf16", host_groups=None, sp=False).items():
            object.__setattr__(p, name, value)
        return p

    with pytest.raises(CpGatherError, match="order-free only for two"):
        make_cp_gather(placement(Axis(4, 0, object())))
    with pytest.raises(CpGatherError, match="no process group"):
        make_cp_attention_core(lambda *a: None, placement(Axis(2, 0, None)))
