"""Check binding-specific GDN ownership of l2 normalization, gating and optional convolution. Core
attachment settles convolution, and chunk length remains a kernel-owned constant.
"""

from __future__ import annotations

import inspect
import itertools

import pytest
import torch

from isoexec.nn._slots import BoundOp, SlotError, Unbound
from isoexec.nn.component import PlacementError
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context

from isoexec.nn.tests import _gdn_fixtures as F

CU_SEQLENS = (0, 4, 9)

#: What the shipped trainer core computes in-kernel, beside its own op.
TRAINER_CORE_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")


def _placed(side: str = "trainer"):
    gdn = F.component(F.spec(), device="cpu")
    gdn.apply_placement(F.placement(side=side), site=F.SITE, op_table=OpTable(mode="reference"))
    return gdn


def _fake_core(owns, *, record=None):
    """An adapter-shaped ``gdn.core``: an 8-argument callable carrying an ownership statement."""

    def engine_core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens):
        if record is not None:
            record.append(tuple(q.shape))
        return torch.zeros(v.shape, dtype=v.dtype, device=v.device)

    engine_core.owns = dict(owns)
    return BoundOp("gdn.core", owns.get("gdn.core", "fake_paged_recurrence"), 1, "test.adapter", {}, engine_core, owns=owns)


def _sequence():
    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    pos = torch.cat([torch.arange(b - a) for a, b in itertools.pairwise(CU_SEQLENS)]).to(torch.int64)
    return make_sequence_context(pos, cu, max_q=max(b - a for a, b in itertools.pairwise(CU_SEQLENS)))




def test_a_trainer_core_owns_l2norm_and_gating_and_leaves_the_conv_its_own_stage():
    gdn = _placed("trainer")
    core = gdn.slots.resolved()["core"]
    assert core.ops == TRAINER_CORE_OPS, "the identity report reads this off the module that executes"
    assert "conv.short" not in core.ops
    assert gdn.slots.resolved()["conv"].op == "conv.short"
    assert gdn.slots.unfilled() == {}


def test_the_core_call_convention_carries_no_chunk_length():
    """One copy of the cpr chunk, on the callable that uses it. Nothing here to compare it against."""
    gdn = _placed("trainer")
    params = inspect.signature(gdn.slots.core).parameters
    assert [name for name, p in params.items() if p.kind != p.KEYWORD_ONLY] == [
        "q", "k", "v", "alpha", "beta", "A_log", "dt_bias", "cu_seqlens"
    ]
    assert params["cu_seqlens_cpu"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["cu_seqlens_cpu"].default is None
    assert dict(gdn.slots.resolved()["core"].inputs) == {}
    assert not hasattr(gdn, "chunk_size")




def test_an_engine_leaves_the_core_and_the_conv_for_attach_core():
    gdn = _placed("engine")
    waiting = gdn.slots.unfilled()
    assert set(waiting) == {"core", "conv"}
    assert "attach_core" in waiting["core"].fill_verb and "attach_core" in waiting["conv"].fill_verb
    with pytest.raises(RuntimeError, match=r"nothing has filled this slot"):
        gdn.slots.core(None, None, None, None, None, None, None, None)


def test_an_attached_core_that_reads_the_paged_conv_window_takes_the_conv_out_of_the_composition():
    gdn = _placed("engine")
    gdn.attach_core(_fake_core({op: "fake_paged_recurrence" for op in (*TRAINER_CORE_OPS, "conv.short")}))
    assert gdn.slots.resolved()["core"].provider == "test.adapter"
    conv = gdn.slots.unfilled()["conv"]
    assert "fake_paged_recurrence" in conv.fill_verb and "in-kernel" in conv.fill_verb


def test_an_attached_core_that_does_not_claim_the_conv_leaves_it_bound_as_its_own_op():
    gdn = _placed("engine")
    gdn.attach_core(_fake_core({op: "fake_paged_recurrence" for op in TRAINER_CORE_OPS}))
    assert gdn.slots.resolved()["conv"].op == "conv.short"
    assert gdn.slots.unfilled() == {}


@pytest.mark.parametrize(
    "owns",
    [
        pytest.param({}, id="claims_nothing"),
        pytest.param({"gdn.core": "x", "gdn.l2norm": "x"}, id="claims_the_norm_but_not_the_gating"),
        pytest.param({"gdn.core": "x", "gdn.gating": "x"}, id="claims_the_gating_but_not_the_norm"),
    ],
)
def test_a_core_that_claims_neither_in_kernel_stage_is_refused(owns):
    """Two ops with no owner anywhere is a composition with nothing to compare against the peer."""
    gdn = _placed("engine")
    with pytest.raises(PlacementError, match=r"gdn\.(l2norm|gating)"):
        gdn.attach_core(_fake_core(owns))
    assert "core" in gdn.slots.unfilled(), "a refused attach leaves the slot unfilled, not half-filled"


def test_attach_core_refuses_another_op_id():
    gdn = _placed("engine")
    wrong = _fake_core({op: "x" for op in TRAINER_CORE_OPS})
    wrong = BoundOp("attention.core.gqa", wrong.impl, wrong.version, wrong.provider, {}, wrong.fn, owns=wrong.owns)
    with pytest.raises(SlotError, match=r"takes 'gdn\.core'"):
        gdn.attach_core(wrong)


def test_attach_core_refuses_to_replace_a_resolved_trainer_core():
    """Overwriting a resolved core would be a silent kernel swap on a gradient path."""
    gdn = _placed("trainer")
    with pytest.raises(PlacementError, match="already filled"):
        gdn.attach_core(_fake_core({op: "x" for op in TRAINER_CORE_OPS}))


def test_attach_core_before_placement_is_refused_by_name():
    gdn = F.component(F.spec(), device="cpu")
    with pytest.raises(PlacementError, match="apply_placement has not run"):
        gdn.attach_core(_fake_core({op: "x" for op in TRAINER_CORE_OPS}))




def test_the_forward_skips_the_convolution_exactly_when_the_core_owns_it():
    """The bound core's ownership determines whether forward calls a separate convolution."""
    calls: list[tuple[int, ...]] = []
    for owns_conv in (True, False):
        gdn = _placed("engine")
        ops = {op: "fake_paged_recurrence" for op in TRAINER_CORE_OPS}
        if owns_conv:
            ops["conv.short"] = "fake_paged_recurrence"
        gdn.attach_core(_fake_core(ops))
        assert isinstance(gdn.slots.conv, Unbound) is owns_conv
        conv_calls: list[int] = []
        if not owns_conv:
            bound = gdn.slots.conv
            gdn.slots.conv = lambda x, w, cu, _f=bound, _c=conv_calls: (_c.append(1), _f(x, w, cu))[1]
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(CU_SEQLENS[-1], F.HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)
        with torch.no_grad():
            out = gdn(x, _sequence())
        assert tuple(out.shape) == (CU_SEQLENS[-1], F.HIDDEN)
        assert len(conv_calls) == (0 if owns_conv else 1)
        calls.append((len(conv_calls),))
    assert calls == [(0,), (1,)]
