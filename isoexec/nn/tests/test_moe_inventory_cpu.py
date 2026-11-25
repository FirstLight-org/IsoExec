"""Ordered mixture, embedding and head inventory parity, including parameter-identity ties.
Selection-only router bias is compared as a synced buffer on the component side.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn

from isoexec.nn import RMSNorm, inventory, verify_placed
from isoexec.nn.component import PlacementError
from isoexec.nn.embed import Embedding
from isoexec.nn.head import LMHead
from isoexec.nn.inspect import InventoryError
from isoexec.nn.moe import BIAS_ROUTING, MoE
from isoexec.ops.table import OpTable
from isoexec.schema import col, expert, moe_vocabulary, replicated, row, stack_vocabulary, ties_for, torch_dtype

from isoexec.nn.tests import _moe_fixtures as F

CASES = [
    pytest.param(kwargs, id=name)
    for kwargs, name in [
        ({}, "shared_gated"),
        ({"shared_expert_gate": False}, "shared_ungated"),
        ({"shared_intermediate": 0}, "no_shared"),
        ({"routing": BIAS_ROUTING}, "sigmoid_bias_shared"),
        ({"routing": BIAS_ROUTING, "shared_expert_gate": False}, "sigmoid_bias_ungated"),
        ({"routing": BIAS_ROUTING, "shared_intermediate": 0}, "sigmoid_bias_no_shared"),
        ({"num_experts": 2, "top_k": 1}, "two_experts"),
    ]
]


@pytest.mark.parametrize("kwargs", CASES)
def test_inventory_equals_the_schema_vocabulary(kwargs):
    mlp = F.spec(**kwargs)
    inv = inventory(F.component(mlp, device="meta"), prefix=F.SITE)
    want = list(moe_vocabulary(F.NS, mlp, F.HIDDEN))
    held_as_buffer = set(inv.buffers)
    assert [t for t in want if t.name not in held_as_buffer] == list(inv.values())
    assert [t for t in want if t.name in held_as_buffer] == list(inv.buffers.values())
    # And nothing was lost in the split: every vocabulary row is in exactly one half.
    assert len(want) == len(inv) + len(inv.buffers)


@pytest.mark.parametrize("kwargs", CASES)
def test_the_correction_bias_is_a_synced_buffer_exactly_when_the_routing_reads_one(kwargs):
    mlp = F.spec(**kwargs)
    moe = F.component(mlp)
    inv = inventory(moe, prefix=F.SITE)
    expected = {f"{F.SITE}.router.expert_bias"} if mlp.routing == BIAS_ROUTING else set()
    assert set(inv.buffers) == expected
    assert (moe.router.expert_bias is not None) == bool(expected)


def test_the_comparison_is_ordered_and_field_for_field():
    """Guard the guard: the rows carry the axis, and an axis field difference has to show."""
    rows = list(inventory(F.component(F.spec()), prefix=F.SITE).values())
    head = [t.name[len(F.SITE) + 1 :] for t in rows[:4]]
    assert head == ["router.weight", "experts.0.gate", "experts.0.up", "experts.0.down"]
    assert [t.name[len(F.SITE) + 1 :] for t in rows[-4:]] == [
        "shared.gate",
        "shared.up",
        "shared.down",
        "shared.expert_gate",
    ]
    by_name = {t.name[len(F.SITE) + 1 :]: t for t in rows}
    # The router is fp32 while the checkpoint is bf16: an exact upcast, and the one held-dtype
    # difference in this kind. Everything else is bf16.
    assert (by_name["router.weight"].dtype, by_name["router.weight"].shard) == ("fp32", replicated())
    assert {t.dtype for t in rows if t.name != f"{F.SITE}.router.weight"} == {"bf16"}
    assert by_name["experts.3.gate"].shard == expert() and by_name["experts.3.down"].shard == expert()
    assert by_name["shared.gate"].shard == col() and by_name["shared.down"].shard == row()
    assert by_name["shared.expert_gate"].shard == replicated()
    assert by_name["experts.0.down"].shape == (F.HIDDEN, F.INTERMEDIATE)


def test_the_expert_axis_is_the_expert_partition_and_not_the_role_split():
    """``expert()`` is the first partition. The role split inside an expert only exists at ETP>1 and
    is a function of the parameter's name, which is why one declaration covers both degrees."""
    axes = F.component(F.spec()).experts[0].tensor_axes()
    assert set(axes) == {"gate", "up", "down"} and set(axes.values()) == {expert()}


def test_state_specs_are_empty_for_every_piece_of_the_stack():
    """A mixture, an embedding and a head declare no engine-side cache; only a mixer does."""
    assert F.component(F.spec()).state_specs() == ()
    assert F.embedding().state_specs() == ()
    assert F.head().state_specs() == ()


def test_facts_record_the_effective_configuration():
    facts = F.component(F.spec(routing=BIAS_ROUTING, routed_scale=2.5, shared_expert_gate=False)).facts()
    assert (facts["kind"], facts["routing"], facts["routed_scale"]) == ("moe", BIAS_ROUTING, 2.5)
    assert facts["shared_expert_gate"] is False and facts["shared_intermediate"] == F.SHARED_INTERMEDIATE
    assert (facts["router_dtype"], facts["held_dtype"]) == ("fp32", "bf16")  # family policy, not arguments
    # A row with no shared expert reports no gate, whatever the argument said: facts() is the
    # EFFECTIVE configuration, and moe_vocabulary registers no expert_gate without a shared expert.
    assert F.component(F.spec(shared_intermediate=0, shared_expert_gate=True)).facts()["shared_expert_gate"] is False


def test_construction_is_global_shaped_on_meta_and_allocates_nothing():
    moe = F.component(F.spec(routing=BIAS_ROUTING))
    assert {p.device.type for p in moe.parameters()} == {"meta"}
    assert len(moe.experts) == F.NUM_EXPERTS and moe.ep_range is None
    assert tuple(moe.router.weight.shape) == (F.NUM_EXPERTS, F.HIDDEN)
    assert moe.router.expert_bias.device.type == "meta"
    assert moe.site is None and moe.placement is None


# Use a fresh subprocess to verify construction imports neither kernels nor the arch probe.
_CONSTRUCT_PROBE = """
import sys
import isoexec.nn.moe as M, isoexec.nn.embed as E, isoexec.nn.head as H
from isoexec.nn import inventory
moe = M.MoE(hidden=64, num_experts=8, top_k=2, intermediate=32, shared_intermediate=16,
            routing="sigmoid_noaux_bias_select")
emb = E.Embedding(vocab=128, hidden=64)
head = H.LMHead(vocab=100, hidden=64, padded_vocab=128, tied_to=emb.weight)
inv = inventory(moe, prefix="m")
assert (len(inv), len(inv.buffers)) == (29, 1), (len(inv), len(inv.buffers))
assert all(p.device.type == "meta" for p in moe.parameters())
assert head.weight is emb.weight
print("OPS:" + ",".join(sorted(m for m in sys.modules if m.split(".")[:2] == ["isoexec", "ops"])))
print("ARCH:" + ",".join(sorted(m for m in sys.modules if m == "isoexec.core.arch")))
"""


def test_constructing_the_mixture_and_the_two_ends_never_triggers_arch_selection():
    out = subprocess.run([sys.executable, "-c", _CONSTRUCT_PROBE], capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.splitlines()
    assert "OPS:" in lines, lines  # no isoexec.ops module imported at all
    assert "ARCH:" in lines, lines  # and therefore no ARCH probe either


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": 9}, "selects more experts"),
        ({"routing": "greedy"}, "routing must be one of"),
        ({"num_experts": 0}, "must be >= 1"),
    ],
)
def test_an_inexpressible_mixture_is_refused_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MoE(**{"hidden": F.HIDDEN, "num_experts": 8, "top_k": 2, "intermediate": 32, **kwargs})




def _stack_tree(*, tied: bool) -> nn.Module:
    """``lm.embed`` / ``lm.final_norm`` / ``lm.head`` -- ``stack_vocabulary``'s order, as a tree."""
    tree = nn.Module()
    embed = F.embedding()
    tree.add_module("embed", embed)
    tree.add_module("final_norm", RMSNorm(hidden=F.HIDDEN, eps=F.EPS, role="final"))
    tree.add_module("head", F.head(tied_to=embed.weight if tied else None))
    return tree


@pytest.mark.parametrize("tied", [False, True])
def test_the_embedding_and_head_equal_the_stack_vocabulary(tied):
    inv = inventory(_stack_tree(tied=tied), prefix="lm")
    assert list(inv.values()) == list(stack_vocabulary(F.config(F.spec(), tied_head=tied)))
    assert inv.buffers == {}


def test_a_tied_head_is_one_parameter_under_two_names(capsys):
    """PROOF 4: the tie is resolved from parameter identity, has exactly one owner, and is the
    declaration ``ties_for(cfg)`` makes from the config side."""
    tree = _stack_tree(tied=True)
    inv = inventory(tree, prefix="lm")
    assert tree.head.weight is tree.embed.weight  # one object; that is what a tie is
    assert inv.ties == ties_for(F.config(F.spec(), tied_head=True))
    assert [(t.logical, t.aliases, t.owner) for t in inv.ties] == [("lm.embed.weight", ("lm.head.weight",), "first")]
    assert inv.aliases == ("lm.head.weight",)
    assert "lm.head.weight" not in inv and "lm.embed.weight" in inv  # exactly one owner emits
    with capsys.disabled():
        print(f"\n  tie: {inv.ties[0]} == ties_for(cfg)[0]; aliases {inv.aliases}; owners {len(inv)} rows")


def test_an_untied_head_owns_its_own_rows():
    inv = inventory(_stack_tree(tied=False), prefix="lm")
    assert inv.ties == () and inv.aliases == ()
    assert "lm.head.weight" in inv and inv["lm.head.weight"].shape == (F.PADDED_VOCAB, F.HIDDEN)


def test_a_tie_of_the_wrong_shape_is_refused_rather_than_reshaped():
    other = nn.Parameter(torch.empty(F.PADDED_VOCAB, F.HIDDEN + 1, dtype=torch.bfloat16, device="meta"))
    with pytest.raises(ValueError, match="a tie is ONE tensor under two names"):
        F.head(tied_to=other)


def test_an_engine_head_leaves_its_two_host_shaped_stages_unbound():
    """Engine row selection and log-softmax await host adapter bindings."""
    head = F.head()
    head.apply_placement(F.placement(side="engine"), site="lm.head", op_table=OpTable(mode="reference"))
    waiting = verify_placed(head)["lm.head"]
    assert set(waiting) == {"lm_head_slice", "log_softmax"}
    assert "engine adapter assigns it" in waiting["lm_head_slice"]
    assert "vocab-parallel group" in waiting["log_softmax"]
    assert set(head.slots.resolved()) == {"head", "tp_copy"}


def test_a_trainer_head_resolves_both_stages_and_records_no_test_provider():
    """Trainer head stages resolve with provenance rather than bare fixture callables."""
    head = F.head()
    head.apply_placement(F.placement(), site="lm.head", op_table=OpTable(mode="reference"))
    assert verify_placed(head)["lm.head"] == {}
    resolved = head.slots.resolved()
    assert set(resolved) == {"head", "lm_head_slice", "log_softmax", "tp_copy"}
    assert {r.provider for r in resolved.values()} == {"ops.reference", "ops.collectives"}
    assert resolved["lm_head_slice"].impl == "ref.identity"
    assert resolved["log_softmax"].impl == "ref.log_softmax"
    # At TP=1 there is no vocabulary shard, so no group is handed over and none is recorded.
    assert "seams" not in resolved["log_softmax"].inputs


def test_a_trainer_head_refuses_a_log_softmax_that_cannot_take_the_vocab_parallel_group():
    """At TP>1 the normalizer sums over columns this rank does not hold. ``ref_log_softmax`` states
    the plain-torch row and exposes no group seam, so it refuses rather than normalizing over a
    column shard and returning a plausible distribution."""
    head = LMHead(vocab=F.VOCAB, hidden=F.HIDDEN, padded_vocab=F.PADDED_VOCAB)
    with pytest.raises(PlacementError, match=r"no keyword-only seam 'group'"):
        head.apply_placement(F.placement(tp=4), site="lm.head", op_table=OpTable(mode="reference"))


def test_an_undivisible_vocabulary_is_refused_by_both_ends():
    """The vocab-parallel select maps a token id to its owner by integer division; unequal shards
    would send some ids to a rank that holds no row for them."""
    pieces = [
        Embedding(vocab=130, hidden=F.HIDDEN),
        LMHead(vocab=100, hidden=F.HIDDEN, padded_vocab=130),
    ]
    for piece in pieces:
        with pytest.raises(PlacementError, match="does not divide tp="):
            piece.apply_placement(F.placement(tp=4), site="lm.x", op_table=OpTable(mode="reference"))


def test_a_module_with_parameters_and_no_axes_is_still_refused_under_a_mixture():
    """The walk's own refusal, reached through the mixture's submodules rather than a flat tree."""

    class Stray(nn.Module):
        def __init__(self):
            super().__init__()
            self.odd = nn.Parameter(torch.empty(4, dtype=torch.bfloat16, device="meta"))

    moe = F.component(F.spec())
    moe.add_module("stray", Stray())
    with pytest.raises(InventoryError, match=r"stray: Stray holds parameters \['odd'\]"):
        inventory(moe, prefix=F.SITE)


def test_the_router_dtype_this_tier_holds_is_the_one_the_family_declares():
    """Compare the component's fp32 router dtype with the family declaration. The local copy
    preserves kernel-free construction.
    """
    from isoexec.nn.moe import ROUTER_DTYPE
    from isoexec.ops.moe.spec import ROUTER_DTYPE as DECLARED

    assert ROUTER_DTYPE == DECLARED, (
        f"nn/moe.py holds the router {ROUTER_DTYPE!r} while ops/moe/spec.py declares {DECLARED!r}; "
        f"the declaration is the owner of this value and the copy must follow it"
    )
    # And the copy is what the component actually holds, not just what the module says.
    moe = F.component(F.spec())
    assert moe.router.weight.dtype is torch_dtype(DECLARED)
    assert moe.facts()["router_dtype"] == DECLARED
