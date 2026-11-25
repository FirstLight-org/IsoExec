"""The residual + norm fusion pass rewrites every join -> pre-norm pair and changes no bits (CPU, reference ops)."""

import operator

import pytest
import torch

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.realization.fusion import fuse_residual_norms
from isoexec.compiler.source import ModelSource
from isoexec.compiler.text import text_inputs, text_roots
from isoexec.core.bindings import BoundOp
from isoexec.models.qwen3_5.tests.tiny import TINY_DENSE_HF
from isoexec.ops.table import OpResolutionError, OpTable
from isoexec.runtimes.parallel import trivial_placement


class _FusedTable:
    """Answers the pass's fused bind with the reference norm applied to the bf16 sum (the claim the
    sm90 kernel is tested against on GPU); everything else refuses like an arch without the kernel."""

    def __init__(self, refuse_roles=()):
        self.calls = []
        self.refuse_roles = set(refuse_roles)

    def bind(self, op, *, site, inputs=None, convention=None, facts=None):
        role = op.rsplit(".", 1)[-1]
        if role in self.refuse_roles:
            raise OpResolutionError(f"{site}: no fused kernel")
        self.calls.append((op, site, dict(facts or {})))
        norm = OpTable(mode="reference").bind(op, site=site, inputs=inputs, convention=convention,
                                              facts={"width": facts["width"]})

        def fn(x, residual, w, eps):
            h = residual + x
            return norm.fn(h, w, eps), h

        return BoundOp(op, "test_add_rms", None, "test", fn=fn, owns={facts["residual"]: "test_add_rms"})


def _model():
    src = ModelSource("isoexec.models.qwen3_5")
    author = src.build_meta(TINY_DENSE_HF).train()
    author = author.to_empty(device="cpu")
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in author.parameters():
            p.copy_((torch.randn(p.shape, generator=gen) * 0.05).to(p.dtype))
    ex = text_inputs(author, device=torch.device("cpu"), max_tokens=16, profile="trainer")
    captured = capture(author, ex.args, ex.kwargs, dynamic_shapes=ex.dynamic_shapes,
                       root_layouts=ex.root_layouts, seq=ex.seq)
    inn.place(author, trivial_placement(side="trainer"), op_table=OpTable(mode="reference"))
    return author, captured, ex


def _pairs(gm):
    norms = [n for n in gm.graph.nodes if n.op == "call_module"
             and type(gm.get_submodule(str(n.target))) is inn.RMSNorm]
    plain = sum(1 for n in norms if not n.kwargs and n.args[0].target is torch.ops.aten.add.Tensor)
    fused = sum(1 for n in norms if "residual" in n.kwargs)
    return plain, fused


def test_pass_rewrites_every_pair_and_keeps_layer_metadata():
    _, captured, _ = _model()
    gm = captured.program
    assert _pairs(gm) == (8, 0)  # 4 layers x 2 joins, the last feeding the final norm
    table = _FusedTable()
    assert fuse_residual_norms(gm, op_table=table) == 8
    assert _pairs(gm) == (0, 8)
    assert not any(n.target is torch.ops.aten.add.Tensor for n in gm.graph.nodes)
    absorbed = sorted(f["residual"] for _, _, f in table.calls)
    assert absorbed == ["residual.add.attn"] * 4 + ["residual.add.mlp"] * 4
    for n in gm.graph.nodes:
        if n.op == "call_module" and "residual" in n.kwargs:
            items = [u for u in n.users if u.target is operator.getitem]
            assert len(items) == 2
            assert all(u.meta.get("isoexec_logical_layer") == n.meta.get("isoexec_logical_layer") for u in items)
    assert fuse_residual_norms(gm, op_table=table) == 0  # idempotent


def test_refused_stage_keeps_its_pair():
    _, captured, _ = _model()
    gm = captured.program
    assert fuse_residual_norms(gm, op_table=_FusedTable(refuse_roles={"final"})) == 7
    assert _pairs(gm) == (1, 7)


def test_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("ISOEXEC_FUSED_RESIDUAL_NORM", "0")
    _, captured, _ = _model()
    assert fuse_residual_norms(captured.program, op_table=_FusedTable()) == 0


@pytest.mark.parametrize("training", [True])
def test_forward_and_gradients_are_bit_identical(training):
    author, captured, ex = _model()
    gm = captured.program

    def run():
        author.zero_grad(set_to_none=True)
        out = gm(**text_roots(ex.seq.input_ids, ex.seq))
        out.float().square().mean().backward()
        grads = {k: p.grad.clone() for k, p in author.named_parameters() if p.grad is not None}
        return out.detach(), grads

    expected, expected_grads = run()
    assert fuse_residual_norms(gm, op_table=_FusedTable()) == 8
    got, got_grads = run()
    assert torch.equal(expected.view(torch.uint8), got.view(torch.uint8))
    assert expected_grads.keys() == got_grads.keys()
    assert sum(k.endswith("norm.weight") for k in expected_grads) == 9  # every pre-norm and the final norm
    for k in expected_grads:
        assert torch.equal(expected_grads[k].view(torch.uint8), got_grads[k].view(torch.uint8)), k


def _install_handoff_row_parallel(author):
    """Give every row-parallel sublayer a binding that offers ``with_handoff`` (what pik's tp_tree binding does)
    and records which handoff each call ran under; at tp=1 the deferred path never engages, so bits are equal."""
    from dataclasses import replace

    seen = []
    for module in author.modules():
        if not getattr(type(module), "ROW_PARALLEL_OUTPUT", False):
            continue
        rp = module.slots.resolved()["row_parallel"]
        base = rp.fn

        def fn(x, w, _b=base):
            return _b(x, w)

        def with_handoff(handoff, _b=base):
            def deferred(x, w):
                seen.append(handoff)
                return _b(x, w)

            return deferred

        fn.with_handoff = with_handoff
        module.slots.row_parallel = replace(rp, fn=fn)
    return seen


@pytest.fixture
def _test_norm_fuses(monkeypatch):
    """Let the test table's add+norm program count as one the fused epilogue reproduces."""
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    import pik.owner as owner  # type: ignore

    monkeypatch.setattr(owner, "_NORM_IMPLS", owner._NORM_IMPLS | {"test_add_rms"})


def test_row_parallel_norm_pairs_are_wired_once_and_change_no_bits(_test_norm_fuses):
    from isoexec.compiler.realization.fusion import pair_row_parallel_norms

    author, captured, ex = _model()
    gm = captured.program
    expected = gm(**text_roots(ex.seq.input_ids, ex.seq)).detach()
    seen = _install_handoff_row_parallel(author)
    assert fuse_residual_norms(gm, op_table=_FusedTable()) == 8
    # every join absorbs a sublayer output (4 mixers + 4 MLPs), so every fused norm is paired
    norms = [gm.get_submodule(str(n.target)) for n in gm.graph.nodes
             if n.op == "call_module" and "residual" in n.kwargs]
    handoffs = [getattr(m.slots.resolved()["add_norm"].fn, "root_handoff", None) for m in norms]
    assert all(h is not None for h in handoffs) and len(set(map(id, handoffs))) == 8
    assert pair_row_parallel_norms(gm) == 0  # idempotent
    for m in norms:  # identities are untouched: only the callables changed
        assert m.slots.resolved()["add_norm"].impl == "test_add_rms"
    got = gm(**text_roots(ex.seq.input_ids, ex.seq)).detach()
    assert torch.equal(expected.view(torch.uint8), got.view(torch.uint8))
    assert sorted(map(id, seen)) == sorted(map(id, handoffs))  # each sublayer ran under its own norm's handoff


def test_row_parallel_norm_pairing_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("ISOEXEC_PIK_OWNER_ARNORM", "0")
    author, captured, _ = _model()
    _install_handoff_row_parallel(author)
    gm = captured.program
    assert fuse_residual_norms(gm, op_table=_FusedTable()) == 8
    assert not any(getattr(m.slots.resolved().get("add_norm").fn, "root_handoff", None) is not None
                   for m in gm.modules() if type(m) is inn.RMSNorm and "add_norm" in m.slots.resolved())


def _paired_norms(gm):
    return [n for n in gm.graph.nodes if n.op == "call_module" and "residual" in n.kwargs
            and getattr(gm.get_submodule(str(n.target)).slots.resolved()["add_norm"].fn, "root_handoff", None)]


def test_norm_programs_the_epilogue_does_not_reproduce_are_not_paired(monkeypatch):
    """E.g. sm100's add+norm: the pair would defer every call and never fuse."""
    author, captured, _ = _model()
    _install_handoff_row_parallel(author)
    gm = captured.program
    assert fuse_residual_norms(gm, op_table=_FusedTable()) == 8
    assert _paired_norms(gm) == []


def test_a_sublayer_called_at_two_sites_is_never_paired(monkeypatch, _test_norm_fuses):
    """The pairing rewrites the sublayer MODULE: if a second site calls it, that site would defer too and hand a
    placeholder to a consumer that is not the paired norm. Such modules are refused; the others still pair."""
    from isoexec.compiler.realization.fusion import pair_row_parallel_norms

    author, captured, _ = _model()
    seen = _install_handoff_row_parallel(author)
    gm = captured.program
    monkeypatch.setenv("ISOEXEC_PIK_OWNER_ARNORM", "0")
    assert fuse_residual_norms(gm, op_table=_FusedTable()) == 8
    monkeypatch.delenv("ISOEXEC_PIK_OWNER_ARNORM")
    subs = [n for n in gm.graph.nodes if n.op == "call_module"
            and getattr(type(gm.get_submodule(str(n.target))), "ROW_PARALLEL_OUTPUT", False)]
    assert len(subs) == 8
    # Site 2 now calls site 1's module (a shared block): neither site may defer.
    shared = subs[0].target
    subs[2].target = shared
    gm.recompile()
    assert pair_row_parallel_norms(gm) == 6
    module = gm.get_submodule(str(shared))
    assert getattr(module.slots.resolved()["row_parallel"].fn, "root_handoff", None) is None
    assert len(_paired_norms(gm)) == 6
    # A second pass (or a second graph over the same modules) wires nothing twice.
    assert pair_row_parallel_norms(gm) == 0
    assert seen == []


def test_tp_tree_handoff_clears_a_leftover_root_when_it_does_not_defer(monkeypatch):
    """An aborted no-grad forward can leave a root pending; the next call that does not defer (here tp=1)
    must drop it, or the norm's take() would raise on the next ordinary call."""
    from isoexec.ops.collectives import tp_tree
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    import sys
    import types

    ensure_pik()
    import pik.owner as owner  # type: ignore

    linear = types.ModuleType("pik.linear")  # the real one autotunes on a GPU at import
    linear.row_parallel_linear = lambda x, w, b, **kw: x @ w.t()
    monkeypatch.setitem(sys.modules, "pik.linear", linear)
    rp = tp_tree._make_row_parallel(tp=1, rank=0, group=None, get_plan=lambda: None)
    h = owner.RootHandoff("site")
    h.pending = {"out": object()}
    x, w = torch.randn(3, 8), torch.randn(4, 8)
    with torch.no_grad():
        y = rp.with_handoff(h)(x, w)
    assert h.pending is None and torch.equal(y, x @ w.t())
