"""Check closed operation slots, unfilled sentinels, provenance records and binding compatibility."""

from __future__ import annotations

import pytest
import torch

from isoexec.nn import RMSNorm, place, resolved_ops, verify_placed
from isoexec.ops.table import OpResolutionError, OpTable

resolve = OpTable(mode="reference").bind
from isoexec.nn._slots import BoundOp, OpSlots, SlotError, Unbound, UnfilledSlotError
from isoexec.nn.component import PlacementError
from isoexec.nn.inspect import InventoryError

from isoexec.nn.tests import _fixtures as F


def test_an_unfilled_slot_raises_naming_the_site_the_slot_and_the_verb():
    slots = OpSlots({"core": "attention.core.gqa"}, site="lm.layers.0.mixer")
    assert isinstance(slots.core, Unbound)  # a callable, never None
    with pytest.raises(UnfilledSlotError, match=r"lm\.layers\.0\.mixer\.core: .*apply_placement"):
        slots.core(1, 2, 3)


def test_the_site_is_stamped_on_the_sentinels_when_placement_records_it():
    slots = OpSlots({"core": "attention.core.gqa"}, site="<unplaced>")
    slots.site = "lm.layers.7.mixer"
    with pytest.raises(UnfilledSlotError, match="lm.layers.7.mixer.core"):
        slots.core()


def test_the_slot_set_is_closed():
    slots = OpSlots({"core": "attention.core.gqa"}, site="s")
    with pytest.raises(SlotError, match="'kore' is not one of the declared slots"):
        slots.kore = len
    with pytest.raises(SlotError):
        _ = slots.kore


def test_a_bare_callable_is_recorded_as_a_test_substitution():
    """A substituted op has to be VISIBLE; it must never pass for what the resolver would have put here."""
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert attn.slots.resolved()["core"].provider == "ops.reference"

    def fake_core(q, k, v, cu, max_q, scale):
        return torch.zeros_like(q)

    attn.slots.core = fake_core
    record = attn.slots.resolved()["core"]
    assert record.provider == "test" and record.op == "attention.core.gqa"
    assert record.impl.endswith("fake_core") and record.fn is fake_core
    assert resolved_ops(attn)[F.SITE]["core"].provider == "test"


def test_a_slot_holds_a_callable_or_a_resolution_and_nothing_else():
    slots = OpSlots({"core": "attention.core.gqa"}, site="s")
    with pytest.raises(TypeError, match="a slot holds a callable or a BoundOp"):
        slots.core = 3


def test_inputs_are_the_values_the_callable_was_built_with_and_are_immutable():
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(tp=1), site=F.SITE, op_table=OpTable(mode="reference"))
    rope = attn.slots.resolved()["qk_rope"]
    assert dict(rope.inputs) == {"style": "neox", "rotary_dim": 16, "theta": F.ROPE_THETA}
    with pytest.raises(TypeError):
        rope.inputs["style"] = "interleaved"


def test_a_composed_stage_reports_every_op_it_owns():
    """The q/k norm and the rotation are one stage; the ids come off the callable, not a declaration."""
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert attn.slots.resolved()["qk_rope"].ops == ("rope.apply", "norms.qk")
    assert attn.slots.resolved()["qk_norm"].impl == "ref.qk_zero_centered"

    bare = F.component(F.spec(qk_norm=None))
    bare.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert bare.slots.resolved()["qk_rope"].ops == ("rope.apply",)
    assert "qk_norm" in bare.slots.unfilled()


def test_ownership_is_immutable_and_independent_of_callable_attributes():
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    stage = attn.slots.resolved()["qk_rope"]
    expected = {"rope.apply": "ref.rope", "norms.qk": "ref.qk_zero_centered"}
    assert dict(stage.owns) == expected
    assert tuple(stage.owns) == stage.ops
    assert not hasattr(stage.fn, "owns")
    with pytest.raises(TypeError):
        stage.owns["norms.qk"] = "other"
    stage.fn.owns = {"norms.qk": "misleading"}
    assert dict(stage.owns) == expected
    assert dict(attn.slots.resolved()["mm"].owns) == {}


def test_a_slot_refuses_a_resolution_for_an_op_it_does_not_declare():
    """Reject a binding from another op while accepting declared local fusion ownership."""
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    rope = attn.slots.resolved()["qk_rope"]

    # (1) computes another op entirely.
    with pytest.raises(SlotError, match="takes 'attention.core.gqa'"):
        attn.slots.core = BoundOp("gdn.core", "cpr", 1, "runtimes.vllm", {}, len)

    # (2) computes this slot's op, but answers under an op id this kind does not run. Rule (1)
    # alone admits it, which is why there are two.
    def stage(*a, **k):
        return None

    stage.owns = {"gdn.core": "x", "attention.core.gqa": "x"}
    with pytest.raises(SlotError, match="is no slot of"):
        attn.slots.core = BoundOp("gdn.core", "x", 1, "runtimes.vllm", {}, stage, owns=stage.owns)

    # The fused stage the rule is shaped around still goes in: its op is ``norms.qk``, the slot
    # declares ``rope.apply``, and the component has a slot for both.
    attn.slots.qk_rope = rope


def test_version_is_none_where_the_provider_declares_none():
    """A fabricated version would compare equal to a peer's real v1 kernel. None cannot."""
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert attn.slots.resolved()["mm"].version is None  # the torch references carry no version
    assert attn.slots.resolved()["tp_copy"].version is None  # no movement at tp=1


def test_an_engine_leaves_the_core_unbound_for_attach_core():
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(side="engine"), site=F.SITE, op_table=OpTable(mode="reference"))
    waiting = verify_placed(attn)[F.SITE]
    assert "attach_core" in waiting["core"]
    with pytest.raises(UnfilledSlotError, match="attach_core"):
        attn.slots.core()

    # The slot refuses a resolution for another op id -- one contract, at the one place every
    # injection point writes through, rather than a copy inside each ``attach_core``.
    with pytest.raises(SlotError, match="takes 'attention.core.gqa'"):
        attn.attach_core(BoundOp("gdn.core", "cpr", 1, "runtimes.vllm", {}, len))
    paged = BoundOp("attention.core.gqa", "sm90_fa3_segsplit_v1_paged", 2, "runtimes.vllm", {"num_pages": 4}, len)
    attn.attach_core(paged)
    assert attn.slots.resolved()["core"] is paged
    with pytest.raises(PlacementError, match="already filled"):
        attn.attach_core(paged)


def test_a_trainer_resolves_its_own_core_and_attach_core_refuses_to_replace_it():
    attn = F.component(F.spec())
    attn.apply_placement(F.placement(side="trainer"), site=F.SITE, op_table=OpTable(mode="reference"))
    assert verify_placed(attn)[F.SITE] == {}
    with pytest.raises(PlacementError, match="already filled"):
        attn.attach_core(BoundOp("attention.core.gqa", "x", 1, "runtimes.vllm", {}, len))


def test_an_ungated_row_says_that_nothing_fills_the_gate():
    attn = F.component(F.spec(out_gate=None))
    attn.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert verify_placed(attn)[F.SITE] == {"gate": "nothing does: this row has no output gate"}


@pytest.mark.parametrize("tp", [1, 2])
def test_the_row_parallel_seam_resolves_the_leaf_tree_at_every_tp(tp):
    """Even TP=1 must resolve the rounded leaf tree rather than one full-contraction GEMM."""
    attn = F.component(F.spec())
    placement = F.placement(tp=tp)
    attn.apply_placement(placement, site=F.SITE, op_table=OpTable(mode="reference"))
    assert "row_parallel" not in verify_placed(attn)[F.SITE]  # filled, at both degrees
    seam = attn.slots.resolved()["row_parallel"]
    assert seam.op == "collectives.row_parallel"
    assert attn.slots.resolved()["mm"].impl == "ref.mm" and seam.impl != "ref.mm"  # never mm under this id
    assert seam.fn is not attn.slots.mm
    assert dict(seam.inputs) == {"tp": tp, "leaves": placement.leaves, "leaf_dtype": placement.leaf_dtype}
    # The CPU reference retains ref.leaf_tree with no version; claiming pik_tree
    # would falsely match the declared SM90 twin class.
    assert seam.provider == "ops.reference" and seam.version is None
    assert seam.impl == "ref.leaf_tree"
    assert seam.impl != "pik_tree"


def test_an_unplaced_component_is_refused_by_the_inspection_walk():
    attn = F.component(F.spec())
    with pytest.raises(InventoryError, match="was never placed"):
        resolved_ops(attn)
    with pytest.raises(InventoryError, match="was never placed"):
        verify_placed(attn)


def test_a_forward_before_placement_refuses_as_an_installation_mistake():
    norm = RMSNorm(hidden=F.HIDDEN, device="cpu")
    with pytest.raises(UnfilledSlotError, match=r"RMSNorm.*\.norm"):
        norm(torch.zeros(2, F.HIDDEN, dtype=torch.bfloat16))


def test_a_norms_convention_selects_the_implementation_that_states_it():
    """Use the held gamma convention to select a matching norm binding."""
    resolved = {}
    for convention in ("rms_zero_centered", "rms"):
        norm = RMSNorm(hidden=F.HIDDEN, convention=convention, role="pre_attn", device="cpu")
        norm.apply_placement(F.placement(), site=f"{F.NS}.pre_attn_norm", op_table=OpTable(mode="reference"))
        resolved[convention] = norm.slots.resolved()["norm"]
    assert {c: r.impl for c, r in resolved.items()} == {"rms_zero_centered": "ref.rms_zero_centered", "rms": "ref.rms"}
    assert {r.inputs["convention"] for r in resolved.values()} == {"rms_zero_centered", "rms"}
    assert resolved["rms"].fn is not resolved["rms_zero_centered"].fn
    assert {r.provider for r in resolved.values()} == {"ops.reference"}


def test_a_norm_refuses_a_convention_no_implementation_states():
    """Reject a norm convention absent from the available implementations."""
    with pytest.raises(ValueError, match=r"convention must be one of"):
        RMSNorm(hidden=F.HIDDEN, convention="rms_ones_centered", role="pre_attn", device="cpu")
    with pytest.raises(OpResolutionError, match=r"no reference binding .* in norm convention"):
        resolve("norms.rms.pre_attn", site=f"{F.NS}.pre_attn_norm.norm", convention="rms_ones_centered")
    with pytest.raises(OpResolutionError, match=r"on offer: \('rms_zero_centered', 'rms'\)"):
        resolve("norms.rms.final", site=f"{F.NS}.final_norm.norm", convention="gated_rms")


def test_same_operation_binds_independently_at_two_model_sites():
    model = torch.nn.ModuleDict({
        "plain": RMSNorm(hidden=8, eps=1e-6, convention="rms", role="final", device="cpu"),
        "centered": RMSNorm(hidden=8, eps=1e-6, convention="rms_zero_centered", role="final", device="cpu"),
    })
    table = OpTable(mode="reference")
    place(model, F.placement(), op_table=table)
    plain, centered = (model[k].slots.resolved()["norm"] for k in ("plain", "centered"))
    assert plain.op == centered.op
    assert plain.impl != centered.impl
    assert all(m._op_table is table for m in model.values())
    with torch.no_grad():
        model["plain"].weight.fill_(1)
        model["centered"].weight.zero_()
    x = torch.randn(3, 8, requires_grad=True)
    a, b = model["plain"](x), model["centered"](x)
    assert torch.equal(a, b)
    da = torch.autograd.grad(a.sum(), x, retain_graph=True)[0]
    db = torch.autograd.grad(b.sum(), x)[0]
    assert torch.equal(da, db)



def test_finalized_slots_refuse_rebinding_and_retargeting():
    slots = OpSlots({"core": "attention.core.gqa"}, site="lm.mixer")
    slots.core = BoundOp("attention.core.gqa", "ref.attention", None, "test", fn=lambda x: x)
    slots.freeze()
    assert slots.frozen
    for change in (lambda: setattr(slots, "core", len), lambda: setattr(slots, "site", "other"),
                   lambda: slots.declare_unbound("core", "attach later")):
        with pytest.raises(SlotError, match="bindings are finalized"):
            change()
    assert slots.core(3) == 3


def test_absent_slots_are_explicit_and_pending_slots_cannot_hide_in_prose():
    norm = RMSNorm(hidden=8, eps=1e-6, convention="rms", role="final", device="cpu")
    norm.apply_placement(F.placement(), site="lm.norm", op_table=OpTable(mode="reference"))
    norm.slots.declare_unbound("norm", "nothing does: misleading diagnostic")
    assert "norm" in verify_placed(norm, required_only=True)["lm.norm"]
    norm.slots.declare_unbound("norm", "configuration omits this operation", required=False)
    assert verify_placed(norm, required_only=True)["lm.norm"] == {}
    assert "norm" in verify_placed(norm)["lm.norm"]
