"""TP1 row-parallel GEMM with a bf16 root: the sm90 segment-tree producer writes its node in its own dtype.

At TP1 ``pik/linear.py`` hands ``ti_gemm`` a bf16 ROOT buffer while the plan still has m = 8 leaves on the
rank. ``mm_sm90.ops.leaf_node`` only writes an fp32 node for m > 1, so both the admission reference (scratch
``empty_like(out)``, bf16) and any dispatch of ``sm90seg`` into the root raised
``leaf_node writes a contiguous [M, N] torch.float32 buffer for m=8`` and every 1-GPU engine failed to build
(fix_cpr_async.md). The producer now stages its node in ``sm90_node_dtype`` and rounds the root once into
the caller's buffer. A fake producer enforcing ``leaf_node``'s dtype contract stands in for the kernel.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")


@pytest.fixture
def gemm(monkeypatch):
    """``pik.gemm`` loaded as in test_pik_workspace_cpu.py (the package's __init__ initialises CUDA)."""
    if "pik" not in sys.modules:
        package = types.ModuleType("pik")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "pik")]
        monkeypatch.setitem(sys.modules, "pik", package)
    arch = importlib.import_module("pik.arch")
    monkeypatch.setattr(arch, "current", lambda device=None: arch.HOPPER)
    module = importlib.import_module("pik.gemm")
    monkeypatch.setattr(module, "_WORKSPACE", {})
    monkeypatch.setattr(module, "_WORKSPACE_RETIRED", [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return module


def _tree(gemm, x, w, leaf_k, m, leaf_dtype):
    return gemm._eager_leaf_reference(x, w, leaf_k, m, leaf_dtype)


@pytest.fixture
def fake_sm90(monkeypatch, gemm):
    calls = []

    def leaf_node(x, w, leaf_k, out, *, leaf_dtype=torch.bfloat16, config=None):
        m = x.shape[1] // leaf_k
        want = gemm.sm90_node_dtype(m, leaf_dtype)
        if out.dtype is not want:  # the real kernel's contract (mm_sm90/ops.py)
            raise ValueError(f"leaf_node writes a contiguous {want} buffer for m={m}")
        calls.append(out.dtype)
        out.copy_(_tree(gemm, x, w, leaf_k, m, leaf_dtype))
        return out

    ops = types.ModuleType("isoexec.ops.mm.mm_sm90.ops")
    ops.leaf_node = leaf_node
    ops.available = lambda: True
    pkg = types.ModuleType("isoexec.ops.mm.mm_sm90")
    pkg.ops = ops
    monkeypatch.setitem(sys.modules, "isoexec.ops.mm.mm_sm90", pkg)
    monkeypatch.setitem(sys.modules, "isoexec.ops.mm.mm_sm90.ops", ops)
    return calls


@pytest.mark.parametrize("m,leaf_dtype,node", [(1, torch.bfloat16, torch.bfloat16), (1, torch.float32, torch.float32),
                                                (2, torch.bfloat16, torch.float32), (8, torch.bfloat16, torch.float32),
                                                (8, torch.float32, torch.float32)])
def test_node_dtype_table(gemm, m, leaf_dtype, node):
    assert gemm.sm90_node_dtype(m, leaf_dtype) is node


@pytest.mark.parametrize("leaf_dtype", [torch.bfloat16, torch.float32])
def test_sm90seg_into_a_bf16_root_rounds_its_fp32_node_once(gemm, fake_sm90, leaf_dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 8 * 32, generator=g).bfloat16()
    w = torch.randn(7, 8 * 32, generator=g).bfloat16()
    root = torch.empty(5, 7, dtype=torch.bfloat16)
    gemm._dispatch(x, w, root, 32, 8, "sm90seg", leaf_dtype)
    assert fake_sm90 == [torch.float32]  # the producer wrote its fp32 node, not the bf16 root
    assert torch.equal(root, _tree(gemm, x, w, 32, 8, leaf_dtype).to(torch.bfloat16))  # one RNE of the fp32 root


def test_admission_scratch_takes_the_callers_dtype(gemm, fake_sm90, monkeypatch):
    """The admission reference no longer raises at TP1 and judges candidates on the bits they would store."""
    monkeypatch.setattr(gemm, "_sm90seg_ok", lambda m, k: True)
    seen = []
    real = gemm._dispatch

    def spy(x, w, out, leaf_k, m, sched, leaf_dtype=torch.float32):
        seen.append((sched, out.dtype))
        if sched != "sm90seg":  # a stand-in candidate storing the same rounded root
            out.copy_(_tree(gemm, x, w, leaf_k, m, leaf_dtype).to(out.dtype))
            return out
        return real(x, w, out, leaf_k, m, sched, leaf_dtype)

    monkeypatch.setattr(gemm, "_dispatch", spy)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 8 * 32, generator=g).bfloat16()
    w = torch.randn(6, 8 * 32, generator=g).bfloat16()
    root = torch.empty(4, 6, dtype=torch.bfloat16)
    admitted = gemm._admit_schedules(x, w, root, 32, 8, ["fused", "sm90seg"], torch.bfloat16)
    assert admitted == ["fused", "sm90seg"]
    assert all(dtype is torch.bfloat16 for _, dtype in seen)
