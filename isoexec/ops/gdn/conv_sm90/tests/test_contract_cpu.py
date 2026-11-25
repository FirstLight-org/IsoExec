"""Standalone declaration and CPU refusals (no extension build)."""

import torch

from isoexec.core.registry import Registry
from isoexec.ops.gdn.conv_sm90 import ops
from isoexec.ops.gdn.conv_sm90.spec import IMPLS


def test_the_package_declares_the_op_its_kernel_serves():
    """This package declares only conv.short with its SM90 arithmetic identity."""
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == {"conv.short"}
    impl = reg.ops["conv.short"].impls["sm90_cuda_v1"]
    assert impl.version == ops.RECIPE_VERSION == 1
    assert impl.supported_archs == frozenset({"sm90"})
    assert not impl.twin_class
    assert not impl.inputs  # the activation is the GDN row's, not a caller's choice at this op


def test_the_recipe_version_has_one_definition():
    """``spec.py`` declares it, ``ops.py`` imports it. The import runs that way because ``spec.py``
    must stay import-light: reading op metadata may not load the extension."""
    from isoexec.ops.gdn.conv_sm90 import spec

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
    # Per-site implementation identity is returned by the family resolver; production
    # dispatch no longer mutates a host table through a bind verb.
    from isoexec.core.registry_build import _FAMILIES

    assert "conv" not in _FAMILIES


if __name__ == "__main__":
    test_the_package_declares_the_op_its_kernel_serves()
    test_cpu_inputs_refused_and_the_arch_seam_selects_the_family()
    print("OK")
