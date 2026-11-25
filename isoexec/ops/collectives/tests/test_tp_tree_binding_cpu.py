import pytest
import torch
from torch import distributed as dist

from isoexec.ops import _dispatch
from isoexec.ops.collectives import tp_tree
from isoexec.schema import Axis, Placement, ShardError


@pytest.fixture(autouse=True)
def kernel_defaults(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.delenv("ISOEXEC_PIK_LEAVES", raising=False)
    monkeypatch.delenv("ISOEXEC_PIK_LEAF_DTYPE", raising=False)


def _must_not_execute(*args, **kwargs):
    raise AssertionError("Binding executed the projection")


def test_tp1_binding_preserves_the_existing_bf16_leaf_defaults():
    group = dist.ProcessGroup(0, 1)
    cuda_initialized = torch.cuda.is_initialized()
    bound = tp_tree.bind_row_parallel(tp_group=group, site="model.mlp.row_parallel")

    assert bound.op == "collectives.row_parallel"
    assert bound.impl == "pik_tree" and bound.provider == "ops.collectives" and bound.version == 1
    assert dict(bound.inputs) == {"tp": 1, "leaves": 8, "leaf_dtype": "bf16"}
    assert callable(bound.fn)
    assert torch.cuda.is_initialized() is cuda_initialized


def test_binding_snapshots_kernel_settings_and_passes_the_native_group(monkeypatch):
    monkeypatch.setenv("ISOEXEC_PIK_LEAVES", "4")
    monkeypatch.setenv("ISOEXEC_PIK_LEAF_DTYPE", "fp32")
    received = {}

    def build(**kwargs):
        received.update(kwargs)
        return _must_not_execute

    monkeypatch.setattr(tp_tree, "_make_row_parallel", build)
    group = dist.ProcessGroup(0, 1)
    bound = tp_tree.bind_row_parallel(tp_group=group, site="model.mlp.row_parallel")
    monkeypatch.setenv("ISOEXEC_PIK_LEAVES", "8")
    monkeypatch.setenv("ISOEXEC_PIK_LEAF_DTYPE", "bf16")

    plan = received["get_plan"]()
    assert received["group"] is group
    assert received["tp"] == 1 and received["rank"] == 0
    assert plan.num_leaves == 4 and plan.leaf_dtype == torch.float32
    assert dict(bound.inputs) == {"tp": 1, "leaves": 4, "leaf_dtype": "fp32"}
    assert bound.fn is _must_not_execute


def test_legacy_placement_keeps_its_explicit_reduction_settings(monkeypatch):
    received = {}

    def build(**kwargs):
        received.update(kwargs)
        return _must_not_execute

    monkeypatch.setattr(tp_tree, "_make_row_parallel", build)
    group = dist.ProcessGroup(0, 1)
    one = Axis(1, 0, group)
    placement = Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one,
                          side="trainer", leaves=2, leaf_dtype="fp32")
    result = tp_tree.make_row_parallel(placement)

    plan = received["get_plan"]()
    assert received["group"] is group
    assert plan.num_leaves == 2 and plan.leaf_dtype == torch.float32
    assert result is _must_not_execute


def test_tp2_binding_uses_the_same_fixed_leaf_tree():
    bound = tp_tree.bind_row_parallel(tp_group=dist.ProcessGroup(1, 2), site="model.mlp.row_parallel")
    assert bound.impl == "pik_tree" and bound.inputs == {"tp": 2, "leaves": 8, "leaf_dtype": "bf16"}


def test_tp_must_own_whole_leaves():
    with pytest.raises(ShardError, match="must divide leaves"):
        tp_tree.bind_row_parallel(tp_group=dist.ProcessGroup(0, 3), site="model.mlp.row_parallel")


@pytest.mark.parametrize("setting,value", [
    ("ISOEXEC_PIK_LEAVES", "3"),
    ("ISOEXEC_PIK_LEAF_DTYPE", "fp16"),
])
def test_invalid_kernel_settings_refuse_binding(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match=setting):
        tp_tree.bind_row_parallel(tp_group=dist.ProcessGroup(0, 1), site="model.mlp.row_parallel")
