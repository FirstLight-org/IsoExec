"""Standalone declaration and CPU refusals (no extension build). Mirrors conv_sm90/tests/test_contract_cpu.py."""

import torch

from isoexec.core.registry import Registry
from isoexec.ops.gdn.conv_sm100 import ops
from isoexec.ops.gdn.conv_sm100.spec import IMPLS


def test_the_package_declares_the_op_its_kernel_serves():
    """This package declares only conv.short, with two arithmetic identities: the default and the
    forward-only fma.rn option."""
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {"conv.short"}
    assert set(reg.ops["conv.short"].impls) == {"sm100_cuda_v1", "sm100_conv_fma_v1"}
    impl = reg.ops["conv.short"].impls["sm100_cuda_v1"]
    assert impl.version == ops.RECIPE_VERSION == 1
    assert impl.supported_archs == frozenset({"sm100"})
    assert not impl.twin_class
    assert not impl.inputs  # the activation is the GDN row's, not a caller's choice at this op


def test_recipe_v2_is_declared_separately_and_is_not_a_twin_of_v1():
    """Recipe v2 (explicit fma.rn taps) is a different arithmetic identity."""
    reg = Registry.from_impls(IMPLS)
    v1 = reg.ops["conv.short"].impls["sm100_cuda_v1"]
    v2 = reg.ops["conv.short"].impls["sm100_conv_fma_v1"]
    assert v2.version == ops.RECIPE_VERSION_FMA == 2 != v1.version
    assert v2.supported_archs == frozenset({"sm100"})
    assert not v2.twin_class and not v1.twin_class
    assert not v2.inputs


def test_the_default_recipe_is_v1_and_an_unknown_recipe_is_refused():
    """The option is opt-in by name: the default argument is v1, and a version this build does not
    carry is a refusal rather than a fallback to a different rounding schedule."""
    import inspect

    for fn in (ops.varlen, ops.update):
        assert inspect.signature(fn).parameters["recipe"].default == ops.RECIPE_VERSION == 1
    for bad in (0, 3, "2", None):
        try:
            ops._recipe(bad)
            raise AssertionError(f"recipe={bad!r} must be refused")
        except ValueError:
            pass


def test_the_recipe_version_has_one_definition():
    """``spec.py`` declares it, ``ops.py`` imports it. The import runs that way because ``spec.py``
    must stay import-light: reading op metadata may not load the extension."""
    from isoexec.ops.gdn.conv_sm100 import spec

    assert ops.RECIPE_VERSION is spec.RECIPE_VERSION


def test_cpu_inputs_refused_and_the_arch_seam_selects_the_family():
    try:
        ops.varlen(torch.zeros(4, 8), torch.zeros(8, 4), cu_seqlens=torch.tensor([0, 4]))
        raise AssertionError("CPU input must be refused")
    except ValueError:
        pass
    from isoexec.ops import _dispatch

    # A SEAM row, not a family row: `conv` has no `ops/conv/spec.py` and no entry in
    # `registry_build._FAMILIES` -- the conv's facts are the GDN row's and `ops/gdn/spec.py`
    # declares it. The row stays because a seam names one kernel package per arch and gdn ships two.
    assert _dispatch.FAMILIES["conv"].archs == {
        "sm90": "gdn.conv_sm90",
        "sm100": "gdn.conv_sm100",
        "cpu": None,
    }
    assert _dispatch.FAMILIES["conv"].verbs == ("install",)  # installs the convolution adapter
    from isoexec.core.registry_build import _FAMILIES

    assert "conv" not in _FAMILIES


def test_width_matches_conv_sm90():
    """The tap count is the op's identity across architectures: both packages must agree."""
    from isoexec.ops.gdn.conv_sm90 import spec as spec_sm90
    from isoexec.ops.gdn.conv_sm100 import spec as spec_sm100

    assert spec_sm100.WIDTH == spec_sm90.WIDTH == 4


if __name__ == "__main__":
    test_the_package_declares_the_op_its_kernel_serves()
    test_recipe_v2_is_declared_separately_and_is_not_a_twin_of_v1()
    test_the_default_recipe_is_v1_and_an_unknown_recipe_is_refused()
    test_the_recipe_version_has_one_definition()
    test_cpu_inputs_refused_and_the_arch_seam_selects_the_family()
    test_width_matches_conv_sm90()
    print("OK")
