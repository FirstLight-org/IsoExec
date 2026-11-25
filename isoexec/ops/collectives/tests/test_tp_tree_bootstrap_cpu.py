"""Each lazy collective binding must work before another operation initializes PIK."""

import builtins
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.collectives import tp_tree


@pytest.mark.parametrize("entry", ["row", "rs", "partial", "finish", "allreduce"])
def test_first_call_bootstraps_before_importing_pik(monkeypatch, entry):
    initialized = False
    calls = []
    plan = SimpleNamespace(bf16_leaves=False)

    def initialize(placement):
        nonlocal initialized
        initialized = True
        return plan

    def kernel(value, *args, **kwargs):
        calls.append(kwargs)
        return value

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("pik."):
            assert initialized, "cold binding imported PIK before bootstrap"
            return SimpleNamespace(
                row_parallel_linear=kernel, row_parallel_linear_rs=kernel, ti_gemm=kernel,
                tree_all_reduce=kernel, tree_all_reduce_rounded=kernel,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tp_tree, "_reduction_plan", initialize)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    placement = SimpleNamespace(leaves=8, tp=SimpleNamespace(size=2, rank=0, group=object()))
    x = torch.ones(2, 4)
    w = torch.ones(4, 4)
    if entry == "row":
        bound = tp_tree.make_row_parallel(placement)
    elif entry == "rs":
        bound = tp_tree.make_row_parallel_rs(placement)
    elif entry in ("partial", "finish"):
        bound = tp_tree.make_row_parallel_partial(placement)
    else:
        bound = tp_tree.make_tree_all_reduce(placement)
    assert not initialized, "binding construction must remain lazy"
    if entry == "finish":
        result = bound.finish(x)
    elif entry == "allreduce":
        result = bound(x)
    else:
        result = bound(x, w)
    assert torch.equal(result, x)
    assert len(calls) == 1
    if entry in ("row", "rs", "partial"):
        assert calls[0]["plan"] is plan
