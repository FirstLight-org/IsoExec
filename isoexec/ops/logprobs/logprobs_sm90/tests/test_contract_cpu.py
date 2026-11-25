"""Standalone declaration and CPU refusals (no extension build)."""

import torch

from isoexec.core.registry import Registry
from isoexec.ops.logprobs.logprobs_sm90 import integration, ops
from isoexec.ops.logprobs.logprobs_sm90.spec import IMPLS


def test_standalone_declaration():
    """The recipe is declared once, on ``logprobs.log_softmax``. The arch-bearing
    ``logprobs_sm90.log_softmax`` said the same thing under an op id no host bound, so it reached
    no digest and is gone."""
    reg = Registry.from_impls(IMPLS)
    assert "logprobs_sm90.log_softmax" not in reg.ops
    impl = reg.ops["logprobs.log_softmax"].impls["sm90_online_leaf8_v1"]
    assert impl.version == ops.RECIPE_VERSION == 1
    assert impl.supported_archs == frozenset({"sm90"})
    assert not impl.twin_class


def test_the_recipe_version_has_one_definition():
    """``spec.py`` declares it, ``ops.py`` imports it -- that direction, because ``spec.py`` must
    stay import-light and cannot reach a module that loads the extension."""
    from isoexec.ops.logprobs.logprobs_sm90 import spec

    assert ops.RECIPE_VERSION is spec.RECIPE_VERSION


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


if __name__ == "__main__":
    test_standalone_declaration()
    test_geometry_and_refusals()
    print("OK")
