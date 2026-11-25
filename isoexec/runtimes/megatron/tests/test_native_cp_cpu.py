"""Megatron trainer binding under contiguous context parallelism (CPU, simulated ranks).

``bind_trainer_ops(cp_group=...)`` must wrap the attention core (``cp_gather`` record) and the GDN core
and conv (``cp_state`` ownership) while keeping each kernel's CP=1 identity, and the bound components
must run the CP forward and backward end to end. Projections here are plain CPU matmuls, whose bits
may depend on the row count, so end-to-end values are compared to fp32 tolerance; bitwise CP=k == CP=1
of the cores themselves is ``ops/collectives/tests/test_cp_simulated_cpu.py`` (CPU) and the GPU battery.
"""

from __future__ import annotations

import pytest
import torch
from torch import distributed as dist
from torch import fx, nn

import isoexec.nn as inn
from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan
from isoexec.ops.collectives.cp_transport import MailboxTransport
from isoexec.ops.table import OpTable
from isoexec.runtimes.context_parallel import ContextParallelBindingError
from isoexec.runtimes.megatron._binding import bind_trainer_ops
from isoexec.schema import make_sequence_context

CU = [0, 10, 34, 40, 64]


def _graph_module(component):
    root = nn.Module()
    root.layer = component
    graph = fx.Graph()
    inputs = (graph.placeholder("hidden"), graph.placeholder("seq"))
    graph.output(graph.call_module("layer", inputs))
    return fx.GraphModule(root, graph)


def _attention():
    return inn.Attention(hidden=32, heads=4, kv_heads=2, head_dim=8, rope_theta=10000.0, eps=1e-6, device="cpu")


def _gdn():
    return inn.GDN(hidden=32, key_heads=2, value_heads=4, key_head_dim=8, value_head_dim=8, conv_kernel=4,
                   device="cpu")


def _init(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.2).to(p.dtype))
    for p in module.parameters():
        p.data = p.data.float()


@pytest.fixture(autouse=True)
def _sm90(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")  # production collective declarations resolve; replaced below


def _bound(make, cp, rank, box):
    module = make()
    _init(module, seed=1)
    gm = _graph_module(module)
    transport = MailboxTransport(box, cp, rank)
    bind_trainer_ops(gm, tp_group=dist.ProcessGroup(0, 1), op_table=OpTable(mode="reference"),
                     cp_group=dist.ProcessGroup(rank, cp) if cp > 1 else None,
                     cp_transport_factory=lambda: transport)
    module.slots.mm = lambda x, w: x.float() @ w.float().t()
    module.slots.row_parallel = lambda y, w: y.reshape(y.shape[0], -1).float() @ w.float().t()
    module.slots.tp_copy = lambda x: x
    return module


def _run(make, cp, x, dy):
    box, outs, leaves, modules = {}, [], [], []
    plans = [ContiguousPlan.build(CU, cp, r) for r in range(cp)]
    keep = []
    for rank, plan in enumerate(plans):
        module = _bound(make, cp, rank, box)
        modules.append(module)
        local = torch.tensor(plan.local_cu, dtype=torch.int32)
        register_plan(local, plan)
        keep.append(local)
        seq = make_sequence_context(torch.tensor(plan.positions(), dtype=torch.int64), local, max_q=plan.max_q,
                                    max_kv=max(plan.max_q, plan.max_k), cu_seqlens_cpu=local.clone())
        xl = x[plan.lo:plan.lo + plan.local].clone().requires_grad_(True)
        leaves.append(xl)
        outs.append(module(xl, seq))
    for rank in reversed(range(cp)):
        p = plans[rank]
        outs[rank].backward(dy[p.lo:p.lo + p.local])
    grads = {name: sum(dict(m.named_parameters())[name].grad for m in modules)
             for name, _ in modules[0].named_parameters()}
    return torch.cat([o.detach() for o in outs]), torch.cat([x_.grad for x_ in leaves]), grads, modules


@pytest.mark.parametrize("make", [_attention, _gdn], ids=["attention", "gdn"])
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_binding_runs_and_matches_cp1(make, cp):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(CU[-1], 32, generator=g)
    dy = torch.randn(CU[-1], 32, generator=g)
    ref_y, ref_dx, ref_grads, _ = _run(make, 1, x, dy)
    y, dx, grads, modules = _run(make, cp, x, dy)
    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dx, ref_dx, rtol=1e-4, atol=1e-4)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-4)
    slots = modules[0].slots.resolved()
    if make is _attention:
        assert slots["cp_gather"].impl == "contiguous_p2p_prefix_fwd_ranked_sum_bwd"
        assert slots["cp_gather"].inputs == {"cp": cp}
        assert slots["core"].impl == "ref.attention" and slots["core"].inputs == {"scale": modules[0].scale, "head_dim": 8}
    else:
        assert slots["core"].impl == "ref.gdn_recurrent"
        assert slots["core"].owns["collectives.cp_state"] == "contiguous_p2p_state_chain"
        assert slots["conv"].owns["collectives.cp_state"] == "contiguous_p2p_state_chain"


def test_cp1_binding_keeps_the_identity_gather():
    module = _bound(_attention, 1, 0, {})
    assert module.slots.resolved()["cp_gather"].impl == "none"


def test_gdn_core_without_a_state_hand_off_is_refused():
    from types import SimpleNamespace

    from isoexec.core.bindings import BoundOp

    reference = OpTable(mode="reference")

    def bind(op, *, site, **kwargs):
        if op == "gdn.core":
            return BoundOp(op, "test.cpr_like", None, "test", fn=lambda *a, **k: None,
                           owns={"gdn.l2norm": "test.cpr_like", "gdn.gating": "test.cpr_like"})
        return reference.bind(op, site=site, **kwargs)

    gm = _graph_module(_gdn())
    with pytest.raises(Exception, match="state hand-off|no state"):
        bind_trainer_ops(gm, tp_group=dist.ProcessGroup(0, 1), op_table=SimpleNamespace(bind=bind),
                         cp_group=dist.ProcessGroup(0, 2), cp_transport_factory=lambda: None)
    assert ContextParallelBindingError


@pytest.mark.parametrize("arch,cp,expected", [("sm90", 2, "contiguous"), ("sm90", 8, "contiguous"),
                                              ("sm100", 2, "zigzag"), ("sm100", 4, None)])
def test_one_cp_layout_per_model(monkeypatch, arch, cp, expected):
    """Contiguous wherever every attention/GDN kernel has its contiguous form (sm90: trainer_core_qk and the
    recurrent state hand-off), else #190's zigzag at CP=2 (gqa_sm100 owner tiles + head-wise GDN)."""
    from isoexec.ops import _dispatch
    from isoexec.runtimes.context_parallel import select_cp_layout

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    root = nn.Module()
    root.attn, root.gdn = _attention(), _gdn()
    graph = fx.Graph()
    h, s = graph.placeholder("hidden"), graph.placeholder("seq")
    graph.output(graph.call_module("gdn", (graph.call_module("attn", (h, s)), s)))
    gm = fx.GraphModule(root, graph)
    if expected is None:
        with pytest.raises(ContextParallelBindingError, match="cp=2 only"):
            select_cp_layout(gm, cp=cp)
    else:
        assert select_cp_layout(gm, cp=cp) == expected


@pytest.mark.parametrize("arch,cp,required,expected", [
    ("sm90", 2, "zigzag", "zigzag"),        # SkyRL on sm90: the host's zigzag rows, not the model's preference
    ("sm100", 2, "zigzag", "zigzag"),
    ("sm90", 4, "zigzag", "cp=2 only"),
    ("sm90", 4, "contiguous", "contiguous"),
    ("sm100", 2, "contiguous", "no contiguous CP form"),
    ("sm90", 2, "ring", "must be 'contiguous' or 'zigzag'"),
])
def test_a_host_that_shards_its_own_rows_states_the_layout(monkeypatch, arch, cp, required, expected):
    """Both layouts hold T/cp rows per rank, so a host whose packer cuts one way while the model binds the
    other would run on the wrong tokens without a shape error: the host's layout wins or is refused."""
    from isoexec.ops import _dispatch
    from isoexec.runtimes.context_parallel import select_cp_layout

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    root = nn.Module()
    root.attn, root.gdn = _attention(), _gdn()
    graph = fx.Graph()
    h, s = graph.placeholder("hidden"), graph.placeholder("seq")
    graph.output(graph.call_module("gdn", (graph.call_module("attn", (h, s)), s)))
    gm = fx.GraphModule(root, graph)
    if expected in ("zigzag", "contiguous"):
        assert select_cp_layout(gm, cp=cp, required=required) == expected
    else:
        with pytest.raises(ContextParallelBindingError, match=expected):
            select_cp_layout(gm, cp=cp, required=required)


def test_skyrl_builds_the_zigzag_layout_its_packer_cuts():
    """SkyRL's packer and scorer use Megatron's zigzag cut, so its model build must request that layout."""
    import inspect

    from isoexec.integrations.skyrl import megatron as skyrl_megatron

    assert 'cp_layout="zigzag" if mc.context_parallel_size > 1 else None' in inspect.getsource(
        skyrl_megatron.build_module)
