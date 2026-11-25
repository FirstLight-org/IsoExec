"""Check SM100 scoring bindings without constructing CUDA extensions."""

import pytest
import torch

from isoexec.ops import _dispatch
from isoexec.ops.table import OpTable


def test_bound_scoring_recipe_refuses_a_later_process_pin_change(monkeypatch):
    """The binding still refuses to run under a pin other than the one it was placed for. Only one recipe is
    declared now, so the changed pin is simulated."""
    from isoexec.ops.logprobs.logprobs_sm100 import ops

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    bound = OpTable().bind("logprobs.log_softmax", site="model.score")
    assert bound.impl == ops.impl_id(ops.RECIPE_VERSION)
    monkeypatch.setitem(ops.IMPL_BY_RECIPE, ops.RECIPE_VERSION + 1, "sm100_some_later_recipe")
    monkeypatch.setattr(ops, "RECIPE", ops.RECIPE_VERSION + 1)
    with pytest.raises(RuntimeError, match="recipe changed after placement"):
        bound.fn(torch.empty(1), None, 1.0)


def test_sm100_head_preserves_native_bf16_gemm_rounding(monkeypatch):
    import isoexec.ops.mm.resolve as mm_resolve

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    native = OpTable().bind("mm", site="model.head.mm")
    rounded = torch.tensor([[1.25]], dtype=torch.bfloat16)
    from isoexec.core.bindings import BoundOp
    fake = BoundOp("mm", native.impl, native.version, native.provider, native.inputs,
                   lambda x, w: rounded)
    monkeypatch.setattr(mm_resolve, "resolve", lambda op, *, site: fake)
    head = OpTable().bind("logprobs.head", site="model.head")
    out = head.fn(torch.empty(1), torch.empty(1))
    assert out is rounded
    assert out.dtype is torch.bfloat16
    assert head.owns["mm"] == native.impl
