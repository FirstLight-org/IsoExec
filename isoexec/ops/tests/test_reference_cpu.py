"""Check unique family reference ownership, import-light assembly and distinct reference/kernel identities.
Reference bindings must never produce a production kernel's execution digest."""

from __future__ import annotations

import subprocess
import sys

import pytest

from isoexec.core.bindings import BoundOp
from isoexec.core.registry_build import build_registry
from isoexec.ops import reference as ops_reference
from isoexec.ops.collectives.reference import REF_LEAF_DTYPE, REF_LEAVES
from isoexec.ops.reference import (
    REFERENCE_FAMILIES,
    ReferencePlan,
    family_bindings,
    reference_bindings,
)

# Explicit expected operation surface detects missing reference bindings.
OP_IDS: frozenset[str] = frozenset(
    {
        "attention.core.gqa",
        "attention.core.mla",
        "attention.gate",
        "collectives.row_parallel",
        "conv.short",
        "embed.lookup",
        "gdn.core",
        "logprobs.head",
        "logprobs.lm_head_slice",
        "logprobs.log_softmax",
        "mlp.act",
        "mm",
        "moe.combine",
        "moe.expert_mlp",
        "moe.route_mechanics",
        "moe.route_score",
        "moe.shared_expert",
        "norms.gated_out",
        "norms.mla",
        "norms.qk",
        "norms.rms.final",
        "norms.rms.pre_attn",
        "norms.rms.pre_mlp",
        "residual.add.attn",
        "residual.add.mlp",
        "rope.apply",
    }
)

def reference_table(*, conventions=None):
    """The assembled reference bindings, ``{op id: BoundOp}``."""
    return reference_bindings(ReferencePlan(conventions=dict(conventions or {})))


#: The ops ``gdn.core`` COMPUTES but is not asked for separately, because the reference runs the
#: l2 norm and the gating in-kernel exactly as the production fused impl does.
GDN_OWNED: frozenset[str] = frozenset({"gdn.l2norm", "gdn.gating"})


def test_the_table_states_every_op_id_and_no_others():
    assert frozenset(reference_table()) == OP_IDS


def test_every_family_answers_the_one_call_shape():
    """Nine families, one ``bindings(plan)``, and no family's op ids written down in the assembler."""
    plan = ReferencePlan()
    seen: dict[str, str] = {}
    for family in REFERENCE_FAMILIES:
        entries = family_bindings(family, plan)
        assert entries, f"ops/{family}/reference.py states no op id"
        for op, binding in entries.items():
            assert isinstance(binding, BoundOp), f"{family} bound {op!r} to {type(binding).__name__}"
            assert op not in seen, f"{op!r} claimed by both {seen[op]} and {family}"
            seen[op] = family
    assert frozenset(seen) == OP_IDS


def test_no_reference_impl_id_is_declared_by_any_kernel():
    """Keep every reference implementation outside production registry IDs and equivalence classes."""
    registry = build_registry()
    table = reference_table()
    for op, binding in table.items():
        ids = {op: binding.impl, **dict(binding.owns)}
        for owned_op, impl in ids.items():
            assert impl.startswith("ref."), f"{owned_op}:{impl} is not in the ref.* namespace"
            declared = registry.get_op(owned_op).impls if registry.has_op(owned_op) else {}
            assert impl not in declared, (
                f"{owned_op}:{impl} is BOTH a reference binding and a declared implementation; a "
                f"reference that shares an impl id with a kernel digests equal to it while computing "
                f"a different function"
            )


def test_gdn_core_owns_the_steps_it_computes_in_kernel():
    table = reference_table()
    assert frozenset(table["gdn.core"].owns) - {"gdn.core"} == GDN_OWNED
    for owned in GDN_OWNED:
        assert owned not in table, f"{owned!r} is owned by gdn.core and must not also be bound"


def test_the_leaf_plan_is_the_callers_and_is_declared_on_the_binding():
    """Require reference bindings to record the caller's leaf plan."""
    default = reference_bindings(ReferencePlan())["collectives.row_parallel"]
    assert dict(default.inputs) == {"leaves": REF_LEAVES, "leaf_dtype": REF_LEAF_DTYPE}
    other = reference_bindings(ReferencePlan(leaves=4, leaf_dtype="fp32"))["collectives.row_parallel"]
    assert dict(other.inputs) == {"leaves": 4, "leaf_dtype": "fp32"}
    assert other.fn is not default.fn


@pytest.mark.parametrize(
    "op, convention, impl",
    [
        ("norms.rms.pre_attn", "rms", "ref.rms"),
        ("norms.rms.pre_attn", "rms_zero_centered", "ref.rms_zero_centered"),
        ("norms.mla", "rms", "ref.rms"),
        ("norms.qk", "rms", "ref.qk_rms"),
        ("norms.qk", "rms_zero_centered", "ref.qk_zero_centered"),
        ("norms.gated_out", "gated_rms", "ref.gated_rms"),
        ("norms.gated_out", "gated_rms_zero_centered", "ref.gated_rms_zero_centered"),
    ],
)
def test_a_stated_norm_convention_selects_its_own_implementation(op, convention, impl):
    """A convention SELECTS. ``rms`` and ``rms_zero_centered`` read the same gamma bytes two ways,
    so which one a site runs is a lookup and never a flag on one function."""
    binding = reference_bindings(ReferencePlan(conventions={op: convention}))[op]
    assert binding.impl == impl and binding.inputs["convention"] == convention


@pytest.mark.parametrize(
    "conventions, names",
    [
        ({"norms.qk": "gated_rms"}, ("norms.qk", "gated_rms")),
        ({"norms.mla": "rms_zero_centered"}, ("norms.mla", "rms_zero_centered")),
        ({"norms.gated_out": "rms"}, ("norms.gated_out", "rms")),
    ],
)
def test_a_convention_no_family_states_refuses_by_name(conventions, names):
    """Not a fallback to the binding that is there: reading ``rms`` weights with an
    ``rms_zero_centered`` implementation is the +-1 checkpoint-mapping bug class, which produces
    plausible numbers rather than an error."""
    with pytest.raises(ValueError) as exc:
        reference_table(conventions=conventions)
    for name in names:
        assert name in str(exc.value)


def test_an_op_id_that_states_no_convention_refuses_by_name():
    with pytest.raises(ValueError, match="mlp.act"):
        reference_table(conventions={"mlp.act": "rms"})


def test_two_families_claiming_one_op_id_refuse(monkeypatch):
    """The merge is not first- or last-wins. Two references for one op id is a composition that has
    not been stated, and which of them a table held would otherwise depend on the order of a tuple."""

    def collide(family, plan):
        if family == "mlp":
            return {"mm": BoundOp("mm", "ref.not_mm", None, "ops.reference", fn=lambda x, w: x)}
        return family_bindings(family, plan)

    monkeypatch.setattr(ops_reference, "family_bindings", collide)
    with pytest.raises(ValueError) as exc:
        reference_table()
    assert "'mm'" in str(exc.value) and "ops/mlp/reference.py" in str(exc.value)


_IMPORT_PROBE = """
import sys
import isoexec.ops.reference
leaked = sorted(m for m in sys.modules if m.split(".")[:2] in (["isoexec", "ops"], ["isoexec", "core"])
                and m not in ("isoexec.ops", "isoexec.ops.reference", "isoexec.core", "isoexec.core.bindings", "isoexec.ops._families"))
print("PULLED:" + ",".join(leaked))
"""

_BUILD_PROBE = """
import sys
import torch
from isoexec.ops.reference import ReferencePlan, reference_bindings
reference_bindings(ReferencePlan())
print("CUDA_INIT:" + str(torch.cuda.is_initialized()))
print("RUNTIME:" + ",".join(sorted(m for m in sys.modules if m.split(".")[0] in ("vllm", "megatron"))))
"""


def _run(probe: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=600, check=False
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.splitlines()


def test_importing_the_assembler_imports_no_family_and_no_arch_table():
    """Keep family defaults lazy so importing the reference assembler cannot probe architecture."""
    assert "PULLED:" in _run(_IMPORT_PROBE), _run(_IMPORT_PROBE)


def test_building_a_reference_table_initializes_no_cuda_and_imports_no_runtime():
    """A reference is what a host runs when no kernel was selected; building one must reach neither
    a device nor a serving framework."""
    lines = _run(_BUILD_PROBE)
    assert "CUDA_INIT:False" in lines, lines
    assert "RUNTIME:" in lines, lines


def test_building_a_reference_table_selects_no_kernel_package():
    """The arm is not arch-dispatched. ``ops/_dispatch.selected`` answers which KERNEL package serves
    a seam, and the reference arm is the answer for when none does -- so it must never ask."""
    from isoexec.ops import _dispatch

    asked: list[str] = []

    def refuse(seam):
        asked.append(seam)
        raise AssertionError(f"the reference arm asked _dispatch.selected({seam!r})")

    real, _dispatch.selected = _dispatch.selected, refuse
    try:
        reference_table()
    finally:
        _dispatch.selected = real
    assert asked == []
