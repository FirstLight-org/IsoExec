"""Declarations, the arch seam and geometry refusals -- all CPU, no extension build."""

from __future__ import annotations

import torch

from isoexec.core.registry import Registry
from isoexec.ops.logprobs.logprobs_sm100 import integration, ops
from isoexec.ops.logprobs.logprobs_sm100.spec import (
    IMPL,
    IMPL_BY_RECIPE,
    IMPLS,
    LANES,
    LEAVES,
    RECIPE_VERSION,
    STRIP,
)


def test_standalone_declaration():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {"logprobs.head", "logprobs.lm_head_slice", "logprobs.log_softmax"}
    impl = reg.ops["logprobs.log_softmax"].impls[IMPL]
    assert impl.version == ops.RECIPE_VERSION == 3
    assert impl.supported_archs == frozenset({"sm100"})
    assert not impl.twin_class


def test_one_schedule_is_declared_and_claims_no_twin_class():
    """Versions 1 and 2 (per-dtype, per-vocabulary recipes) are retired: one identity, one version."""
    reg = Registry.from_impls(IMPLS)
    impls = reg.ops["logprobs.log_softmax"].impls
    assert set(impls) == {IMPL} == {"sm100_lse_segments_v3"}
    assert impls[IMPL].version == RECIPE_VERSION == 3
    assert impls[IMPL].supported_archs == frozenset({"sm100"}) and not impls[IMPL].twin_class
    assert IMPL_BY_RECIPE == {3: IMPL}
    assert not Registry.from_impls(IMPLS).lint()


def test_the_recipe_does_not_depend_on_dtype_or_vocab():
    for V in (248320, 154880, 151680, 129280, 4096, 64):
        for dtype in ("bf16", "fp32", torch.bfloat16, torch.float32):
            assert ops.default_recipe(dtype, V) == RECIPE_VERSION


def test_the_pin_is_a_process_setting_and_the_declared_impl_id_follows_it():
    from isoexec.ops.logprobs.logprobs_sm100 import integration

    was = ops.RECIPE
    try:
        assert ops.select_recipe(RECIPE_VERSION) == RECIPE_VERSION
        assert integration.IMPL == IMPL
        assert ops.select_recipe(dtype="bf16", vocab=151680) == RECIPE_VERSION
        assert ops.select_recipe(dtype="fp32", vocab=151680) == RECIPE_VERSION
        for bad in (0, 1, 2, 4, "v3"):  # 1 and 2 named the retired recipes
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

    assert IMPL == "sm100_lse_segments_v3"
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
    assert ops.RECIPE == RECIPE_VERSION, "the package default pin is the one recipe"
    assert integration.IMPL == IMPL


if __name__ == "__main__":
    test_standalone_declaration()
    test_geometry_and_refusals()
    print("OK")
