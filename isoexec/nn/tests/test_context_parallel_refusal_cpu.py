"""Check CP admission, attention gather installation and unsupported component refusals. Per-site
records expose the gather as a trainer-only placement operation.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from isoexec.nn import Attention, Component, LatentAttention, PlacementError, place, resolved_ops
from isoexec.ops.table import OpTable
from isoexec.schema import Axis, Placement

from isoexec.nn.tests import _fixtures as F
from isoexec.nn.tests import _gdn_fixtures as GF
from isoexec.nn.tests import _moe_fixtures as MF

GROUP = object()

# Read the qualified two-rank CP implementation identity from its declaration.
CP_OP = "collectives.cp_gather"
CP_IMPL = "all_gather_unzigzag_fwd_sum_rs_bwd"
NO_MOVEMENT = "none"


def placement(cp: int, *, tp: int = 1, side: str = "trainer") -> Placement:
    """A ``Placement`` at the CP degree under test. ``Placement`` itself requires a trainer side and
    a live group for cp>1, so those are not what this file is testing."""
    one = Axis(1, 0, None)
    tp_axis = Axis(tp, 0, GROUP if tp > 1 else None)
    return Placement(
        tp=tp_axis,
        ep=one,
        dp=one,
        pp=one,
        etp=tp_axis if tp > 1 else one,
        cp=Axis(cp, 0, GROUP if cp > 1 else None),
        side=side,
        leaves=8,
    )


def _placed_attention(cp: int, *, side: str = "trainer") -> Attention:
    module = F.component(F.spec())
    place(module, placement(cp, side=side), prefix=F.SITE, op_table=OpTable(mode="reference"))
    return module


def _record(module) -> dict[str, tuple]:
    """``{slot: (op, impl, sorted inputs)}`` for every resolved slot of the site."""
    return {
        slot: (r.op, r.impl, tuple(sorted((k, repr(v)) for k, v in r.inputs.items())))
        for slot, r in resolved_ops(module)[F.SITE].items()
    }




def test_attention_is_placed_at_cp_two_and_installs_the_gather():
    """Require the CP gather record and wrapped attention core at degree two."""
    assert Attention.CONTEXT_PARALLEL is not None
    attn = _placed_attention(cp=2)
    gather = attn.slots.resolved()["cp_gather"]
    assert (gather.op, gather.impl) == (CP_OP, CP_IMPL)
    assert dict(gather.inputs) == {"cp": 2}, "only the degree is the caller's; the layout is the impl's"
    assert gather.version == 1 and gather.provider == "ops.collectives"
    core = attn.slots.resolved()["core"]
    assert core.fn.cp_layout == "gather_full_keep_owned"
    # The callable that PERFORMS the gather says so, and it agrees with the slot that owns the
    # record rather than emitting a second, input-less record for one movement.
    assert core.owns[CP_OP] == CP_IMPL and core.owns[core.op] == core.impl


def test_the_attention_core_s_own_resolution_does_not_move_across_the_cut():
    """What CP changes is which rank held the bytes, never which kernel reads them.

    If this ever fails, the seam has started selecting arithmetic, and the cp=2 trainer is no longer
    comparable to the cp=1 engine it is qualified against.
    """
    one, two = _placed_attention(cp=1), _placed_attention(cp=2)
    a, b = one.slots.resolved()["core"], two.slots.resolved()["core"]
    assert (a.op, a.impl, a.version, a.provider, dict(a.inputs)) == (b.op, b.impl, b.version, b.provider, dict(b.inputs))
    assert a.fn is not b.fn, "the cp=2 core is the wrapped one, or nothing gathers"


def test_an_engine_side_placement_records_the_identity_and_waits_for_its_core():
    """The side that never runs the seam still fills the slot, because an absent entry and a
    recorded 'no movement here' read the same to a human and differently to a peer."""
    engine = _placed_attention(cp=1, side="engine")
    assert engine.slots.resolved()["cp_gather"].impl == NO_MOVEMENT
    assert "core" in engine.slots.unfilled()




def test_gdn_is_placed_at_cp_two_with_the_head_wise_region():
    """GDN states a head-wise form: cp=2 records the all-to-all region, cp=1 records no movement."""
    from isoexec.nn import GDN

    assert GDN.CONTEXT_PARALLEL is not None
    records = {}
    for cp in (1, 2):
        module = GF.component(GF.spec())
        place(module, placement(cp=cp), prefix=F.SITE, op_table=OpTable(mode="reference"))
        records[cp] = module.slots.resolved()["cp_heads"]
    assert (records[2].op, records[2].impl, dict(records[2].inputs)) == (
        "collectives.cp_heads", "head_all_to_all_both_directions", {"cp": 2})
    assert records[1].impl == NO_MOVEMENT


@pytest.mark.parametrize("tp", [1, 2])
def test_gdn_s_head_wise_region_cuts_this_tp_rank_s_heads(monkeypatch, tp):
    """Under TP the in-projection columns a rank holds are its TP shard's heads, so the CP region must
    re-cut the local heads; the global counts would read columns this rank does not have."""
    from isoexec.ops.collectives import cp_heads

    seen = {}

    def region(placement, **heads):
        seen.update(heads)
        return lambda *args: None

    monkeypatch.setattr(cp_heads, "make_gdn_region", region)
    module = GF.component(GF.spec())
    place(module, placement(cp=2, tp=tp), prefix=F.SITE, op_table=OpTable(mode="reference"))
    assert (seen["key_heads"], seen["value_heads"]) == (GF.KEY_HEADS // tp, GF.VALUE_HEADS // tp)
    assert (seen["key_heads"], seen["value_heads"]) == (module.key_heads_local, module.value_heads_local)


def test_moe_states_a_per_token_form():
    """Routing, expert rows and the fixed-order combine read one token, so MoE admits CP."""
    from isoexec.nn import MoE

    assert MoE.CONTEXT_PARALLEL is not None and "per token" in MoE.CONTEXT_PARALLEL
    assert MF.component(MF.spec()).CONTEXT_PARALLEL == MoE.CONTEXT_PARALLEL


@pytest.mark.parametrize("ep", [1, 2])
def test_moe_is_placed_at_cp_two_with_the_cp_one_record(ep):
    """A mixture is per token: a CP cut changes which rank holds a token, never a binding. At EP>1 the
    ranks of the CP group are expert peers, so the dispatch record is the CP=1 one too."""
    from isoexec.nn import MoE

    assert MoE.CONTEXT_PARALLEL is not None

    def record(cp: int) -> dict[str, tuple]:
        module = MF.component(MF.spec())
        one = Axis(1, 0, None)
        at = Placement(tp=one, ep=Axis(ep, 0, GROUP if ep > 1 else None), dp=one, pp=one, etp=one,
                       cp=Axis(cp, 0, GROUP if cp > 1 else None), side="trainer", leaves=8)
        place(module, at, prefix=MF.SITE, op_table=OpTable(mode="reference"))
        return {slot: (r.op, r.impl, tuple(sorted((k, repr(v)) for k, v in r.inputs.items())))
                for slot, r in resolved_ops(module)[MF.SITE].items()}

    assert record(2) == record(1)


def test_a_kind_that_states_no_form_still_refuses_a_context_parallel_placement_by_name():
    """Reject unsupported CP kinds with site-specific diagnostics."""
    module = LatentAttention(hidden=16, heads=2, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                             qk_rope_head_dim=4, v_head_dim=4, rope_theta=10000)
    with pytest.raises(PlacementError, match="context parallelism") as exc:
        place(module, placement(cp=2), prefix=F.SITE, op_table=OpTable(mode="reference"))
    message = str(exc.value)
    assert F.SITE in message, "a refusal at placement names the site the composition would run at"
    assert "cp=1" in message, "and says what to do instead"
    assert type(module).NO_CONTEXT_PARALLEL in message, "and says why THIS kind has none"


def test_the_absorbed_latent_row_states_no_form_either():
    """MLA has no qualified context-parallel gather and must refuse CP."""
    assert LatentAttention.CONTEXT_PARALLEL is None
    assert "attention.core.gqa" in LatentAttention.NO_CONTEXT_PARALLEL


def test_the_refusal_is_the_contract_s_and_not_each_kind_s_to_remember():
    """The base placement contract rejects undeclared CP forms for new component kinds."""

    class Novel(Component):
        SLOT_OPS: ClassVar[dict] = {"mm": "mm"}
        SLOTS = ("mm",)

        def facts(self):
            return {"kind": "novel"}

        def _place(self, placement):  # pragma: no cover - the refusal fires before this runs
            raise AssertionError("_place must not be reached at cp>1")

    assert Novel.CONTEXT_PARALLEL is None
    with pytest.raises(PlacementError, match="states no context-parallel form"):
        place(Novel(), placement(cp=2), prefix=F.SITE, op_table=OpTable(mode="reference"))


def test_a_kind_that_states_a_form_is_placed_rather_than_refused():
    """The declaration is a real door and not a comment: the refusal reads it, which is how the port
    that built the ``cp_gather`` seam turned it on by stating the form it built."""

    class WithForm(Component):
        SLOT_OPS: ClassVar[dict] = {}
        SLOTS: tuple = ()
        CONTEXT_PARALLEL = "gathers the full sequence and keeps this rank's rows"

        def facts(self):
            return {"kind": "with_form"}

        def _place(self, placement):
            self.saw_cp = placement.cp.size

    module = WithForm()
    place(module, placement(cp=2), prefix=F.SITE, op_table=OpTable(mode="reference"))
    assert module.saw_cp == 2




def test_the_record_comparison_is_sensitive_to_the_cut_now():
    """A CP=1 control pair matches; CP=1 versus CP=2 differs only in the gather binding."""
    control_a, control_b = _record(_placed_attention(cp=1)), _record(_placed_attention(cp=1))
    assert control_a == control_b

    cut = _record(_placed_attention(cp=2))
    differing = {slot for slot in control_a if control_a[slot] != cut[slot]}
    assert differing == {"cp_gather"}, f"the cut must reach the record, and only at the seam: {differing}"
    assert cut["cp_gather"][:2] == (CP_OP, CP_IMPL)
    assert control_a["cp_gather"][:2] == (CP_OP, NO_MOVEMENT)
    # ...and the CP degree is in the record now, where it was in nothing at all before.
    assert "cp" in {key for record in cut.values() for key, _ in record[2]}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
