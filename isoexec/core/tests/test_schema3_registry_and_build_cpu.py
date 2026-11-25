"""Check registry input domains, equivalence declarations, builder validation, and graph replay."""

import dataclasses
import types
from typing import Any, cast

import pytest

from isoexec.core.registry import (
    ANY,
    TWIN_PROOFS,
    ImplSpec,
    OneOf,
    OpSpec,
    Registry,
    RegistryError,
)
from isoexec.core.registry_build import build_registry, declare_into

SM90 = frozenset({"sm90"})
# ``op`` is the one required keyword M6.4 left standing, and most cases here are about some OTHER
# rule, so `_impl` supplies a name they can ignore. Grouping and refusal cases pass their own.
DECL: dict = {"op": "mm"}
GATE_REF = "ops/collectives/tests/test_bf16_leaf_scheme_cpu.py"
# Use a unique sentinel because proof lookup matches source substrings.
NO_SUCH_GATE = "zzz_no_such_gate_sentinel_do_not_write_this_string_in_a_gate"


def _impl(impl_id, **kw):
    d = dict(DECL)
    d.update(kw)
    return ImplSpec(impl_id, 1, SM90, **d)


# ImplSpec declarations


def test_the_identity_fields_are_required():
    with pytest.raises(TypeError):
        cast(Any, ImplSpec)("x")
    with pytest.raises(TypeError):
        cast(Any, ImplSpec)("x", 1, SM90)
    with pytest.raises(RegistryError, match="at least one supported arch"):
        ImplSpec("x", 1, frozenset(), op="mm")


def test_a_twin_class_with_no_battery_is_refused():
    """A twin claim is a test that passes; TWIN_PROOFS is where the test is named."""
    with pytest.raises(RegistryError, match="TWIN_PROOFS does not name a battery"):
        _impl("x", twin_class="gnd.cpr")
    _impl("x", twin_class=next(iter(TWIN_PROOFS)))


def test_an_input_declaration_must_be_a_key_and_a_domain():
    """Inputs require ANY or OneOf domains; kernel constants are not caller inputs."""
    with pytest.raises(RegistryError, match="neither ANY nor OneOf"):
        _impl("x", inputs={"provider": object()})
    for literal in ("bf16", 1, True, None, ("a", "b")):
        with pytest.raises(RegistryError, match="neither ANY nor OneOf"):
            _impl("x", inputs={"convention": literal})
    _impl("x", inputs={"convention": OneOf("bf16"), "tp": ANY})


def test_add_groups_flat_declarations_by_the_op_each_one_names():
    """Several packages contributing to one op id is the normal case; only a repeated impl id is a
    contradiction. This is what makes the per-package get-or-create helpers unnecessary."""
    reg = Registry.from_impls([_impl("a", op="mm"), _impl("b", op="rope.apply")])
    reg.add(_impl("c", op="mm"))
    assert set(reg.get_op("mm").impls) == {"a", "c"}
    assert reg.get_op("rope.apply").impls["b"].op == "rope.apply"
    with pytest.raises(RegistryError, match="already has impl"):
        reg.add(_impl("a", op="mm"))
    with pytest.raises(RegistryError, match="declares no op"):
        reg.add(_impl("nameless", op=""))


def test_the_imperative_authoring_path_is_gone():
    """Registry declarations use flat ImplSpec records, without mutable OpSpec authoring."""
    assert not hasattr(Registry, "register_op")
    assert not hasattr(OpSpec, "add_impl")


def test_a_forwarders_inner_failure_is_not_blamed_on_the_forwarder():
    """Preserve AttributeError from the architecture module behind a lazy forwarder."""
    inner = "isoexec.ops.ghost.ghost_sm90.spec"
    forwarder = types.ModuleType("isoexec.ops.ghost.spec")
    forwarder.__getattr__ = lambda name: (_ for _ in ()).throw(
        AttributeError(f"module {inner!r} has no attribute {name!r}")
    )
    with pytest.raises(AttributeError, match="ghost_sm90.spec"):
        declare_into(Registry(), forwarder)


def test_declare_into_adds_a_modules_flat_declarations():
    """One shape, and modules accumulate into one registry -- which is what makes family order in
    ``registry_build._FAMILIES`` a presentation choice rather than a correctness one."""
    first = types.ModuleType("first.spec")
    setattr(first, "IMPLS", (_impl("a", op="mm"),))
    second = types.ModuleType("second.spec")
    setattr(second, "IMPLS", (_impl("b", op="mm"),))
    reg = Registry()
    declare_into(reg, first)
    declare_into(reg, second)
    assert set(reg.get_op("mm").impls) == {"a", "b"}


def test_production_registry_is_lint_clean_and_fully_declared():
    reg = build_registry()
    assert reg.lint() == []
    for op in reg.ops.values():
        for impl in op.impls.values():
            assert impl.op == op.name
            assert not impl.twin_class or impl.twin_class in TWIN_PROOFS
    # The digest compares the log-softmax implementation actually bound by each host.
    assert set(reg.get_op("logprobs.log_softmax").impls) == {"sm90_online_leaf8_v1"}


def test_regions_are_registered_with_their_production_implementations():
    reg = build_registry()
    for rid, impl in (
        ("logprobs.head", "eager"),
        ("embed.lookup", "eager"),
        ("norms.qk", "sm90_qk_norm_rope_v2"),
        ("attention.gate", "eager_swish"),
        ("residual.add.attn", "eager"),
        ("residual.add.mlp", "eager"),
        ("moe.shared_expert", "eager"),
    ):
        # Production implementations must remain registered and bindable.
        assert impl in reg.get_op(rid).impls
    # The head's fp32 boundary and the core's bf16 cache are the KERNELS' -- implied by
    # (op, impl_id, version) and written once beside them -- so neither op declares an input.
    assert not reg.get_op("logprobs.head").impls["eager"].inputs
    for impl in reg.get_op("attention.core.gqa").impls.values():
        assert not impl.inputs


# binding-only ops and fused stages


def _reg():
    return Registry.from_impls(
        [
            _impl("ref", op="mm"),
            _impl("fused", op="moe.expert_mlp"),
            _impl("eager", op="moe.combine"),
        ]
    )


def test_an_op_is_a_name_and_its_impls():
    """Operation records expose their name and implementations only."""
    reg = _reg()
    assert not hasattr(reg, "installed_keys") and not hasattr(reg, "region_ops")
    assert {f.name for f in dataclasses.fields(OpSpec)} == {"name", "impls"}
    assert {f.name for f in dataclasses.fields(ImplSpec)} == {
        "impl_id",
        "version",
        "supported_archs",
        "inputs",
        "op",
        "twin_class",
    }
    assert "gdn.state" not in build_registry().ops


def test_a_twin_class_may_not_span_two_op_ids():
    """Equivalent implementations must serve one operation; fused ownership belongs to bindings."""
    cls = next(iter(TWIN_PROOFS))
    reg = Registry.from_impls([_impl("a", op="mm", twin_class=cls), _impl("b", op="rope.apply", twin_class=cls)])
    with pytest.raises(RegistryError, match="spans 2 op ids"):
        reg.assert_twin_classes_single_op()
    assert reg.lint() and "spans 2 op ids" in reg.lint()[0]
    one_op = Registry.from_impls([_impl("a", op="mm", twin_class=cls), _impl("b", op="mm", twin_class=cls)])
    assert not one_op.lint()
    assert not build_registry().lint()
