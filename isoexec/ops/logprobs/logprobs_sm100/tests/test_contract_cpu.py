"""Declarations, the arch seam and geometry refusals -- all CPU, no extension build."""

from __future__ import annotations

import torch

from isoexec.core.registry import Registry
from isoexec.ops.logprobs.logprobs_sm100 import integration, ops
from isoexec.ops.logprobs.logprobs_sm100.spec import (
    DEFAULT_RECIPE_BY_CLASS,
    IMPL,
    IMPL_BY_RECIPE,
    IMPL_V2,
    IMPLS,
    LANES,
    LEAVES,
    RECIPE_VERSION,
    RECIPE_VERSION_V2,
    STRIP,
)


def test_standalone_declaration():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {"logprobs.head", "logprobs.lm_head_slice", "logprobs.log_softmax"}
    impl = reg.ops["logprobs.log_softmax"].impls[IMPL]
    assert impl.version == ops.RECIPE_VERSION == 1
    assert impl.supported_archs == frozenset({"sm100"})
    assert not impl.twin_class


def test_both_recipes_are_declared_and_neither_claims_a_twin_class():
    """Recipe v2 is a NEW impl id declared ALONGSIDE v1, at its own version."""
    reg = Registry.from_impls(IMPLS)
    impls = reg.ops["logprobs.log_softmax"].impls
    assert set(impls) == {IMPL, IMPL_V2} == {"sm100_online_leaf8_v1", "sm100_twopass_leaf8_v2"}
    assert impls[IMPL].version == RECIPE_VERSION == 1
    assert impls[IMPL_V2].version == RECIPE_VERSION_V2 == 2
    for spec_ in impls.values():
        assert spec_.supported_archs == frozenset({"sm100"})
        assert not spec_.twin_class
    assert IMPL_BY_RECIPE == {1: IMPL, 2: IMPL_V2}
    assert not Registry.from_impls(IMPLS).lint()


def test_the_default_recipe_rule_is_keyed_on_dtype_and_vocab_and_never_on_rows():
    """Do-item 2's rule, as a declaration."""
    assert all(isinstance(k, tuple) and len(k) == 2 for k in DEFAULT_RECIPE_BY_CLASS)
    assert {k[0] for k in DEFAULT_RECIPE_BY_CLASS} == {"bf16", "fp32"}
    assert {k[1] for k in DEFAULT_RECIPE_BY_CLASS} == {248320, 154880, 151680}
    assert set(DEFAULT_RECIPE_BY_CLASS.values()) <= {RECIPE_VERSION, RECIPE_VERSION_V2}
    for V in (248320, 154880, 151680):
        assert ops.default_recipe("bf16", V) == RECIPE_VERSION_V2
        assert ops.default_recipe(torch.bfloat16, V) == RECIPE_VERSION_V2
        assert ops.default_recipe("fp32", V) == RECIPE_VERSION
        assert ops.default_recipe(torch.float32, V) == RECIPE_VERSION
    assert ops.default_recipe("bf16", 4096) == RECIPE_VERSION
    assert ops.default_recipe("bf16", 64) == RECIPE_VERSION


def test_the_pin_is_a_process_setting_and_the_declared_impl_id_follows_it():
    from isoexec.ops.logprobs.logprobs_sm100 import integration

    was = ops.RECIPE
    try:
        assert ops.select_recipe(RECIPE_VERSION) == RECIPE_VERSION
        assert integration.IMPL == IMPL
        assert ops.select_recipe(RECIPE_VERSION_V2) == RECIPE_VERSION_V2
        assert integration.IMPL == IMPL_V2
        assert ops.select_recipe(dtype="bf16", vocab=151680) == RECIPE_VERSION_V2
        assert ops.select_recipe(dtype="fp32", vocab=151680) == RECIPE_VERSION
        for bad in (0, 3, "v2"):
            try:
                ops.select_recipe(bad)
                raise AssertionError(f"recipe {bad!r} must be refused")
            except ValueError:
                pass
        try:
            ops.select_recipe()
            raise AssertionError("select_recipe() with no recipe and no class must be refused")
        except ValueError:
            pass
    finally:
        ops.select_recipe(was)


def test_impl_id_is_never_the_sm90_spelling():
    from isoexec.ops.logprobs.logprobs_sm90.spec import IMPL as SM90_IMPL

    assert IMPL == "sm100_online_leaf8_v1"
    assert IMPL != SM90_IMPL


def test_geometry_constants_match_sm90():
    from isoexec.ops.logprobs.logprobs_sm90.spec import LANES as SM90_LANES
    from isoexec.ops.logprobs.logprobs_sm90.spec import LEAVES as SM90_LEAVES
    from isoexec.ops.logprobs.logprobs_sm90.spec import STRIP as SM90_STRIP

    assert (LEAVES, STRIP, LANES) == (SM90_LEAVES, SM90_STRIP, SM90_LANES) == (8, 8, 256)


def test_the_recipe_version_has_one_definition():
    from isoexec.ops.logprobs.logprobs_sm100 import spec

    assert ops.RECIPE_VERSION is spec.RECIPE_VERSION


def test_lm_head_slice_twins_claim_the_existing_row_select_class():
    reg = Registry.from_impls(IMPLS)
    impls = reg.ops["logprobs.lm_head_slice"].impls
    assert set(impls) == {"identity", "sampled_rows"}
    for spec_ in impls.values():
        assert spec_.twin_class == "logprobs.row_select"
        assert spec_.supported_archs == frozenset({"sm100"})


def test_head_and_slice_are_declared_but_never_build_this_packages_kernel():
    from isoexec.ops.logprobs import resolve

    assert resolve.HEAD_IMPL == "eager" and resolve.SLICE_IMPL == "identity"


def test_geometry_and_refusals():
    for V in (248320, 154880, 151680):
        assert all(ops.supports(V, w) for w in (1, 2, 4, 8))
    assert not ops.supports(248077)
    try:
        ops.full_logprobs(torch.zeros(2, 64))
        raise AssertionError("CPU input must be refused")
    except ValueError:
        pass
    assert integration.full_logprobs(torch.zeros(2, 64)) is None


def test_the_arch_selects_this_package(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    assert _dispatch.selected("logprobs") == "logprobs.logprobs_sm100"
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    assert _dispatch.selected("logprobs") is None  # no accelerator: the reference impls


def test_integration_exports_the_same_impl_id():
    assert ops.RECIPE == RECIPE_VERSION, "the package default pin is recipe v1"
    assert integration.IMPL == IMPL


if __name__ == "__main__":
    test_standalone_declaration()
    test_geometry_and_refusals()
    print("OK")
