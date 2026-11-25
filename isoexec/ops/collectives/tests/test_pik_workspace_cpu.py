"""Scratch ownership: eager generations die; captured generations remain address-stable."""

import gc
import importlib
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")


@pytest.fixture
def gemm(monkeypatch):
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


def test_ascending_eager_requests_release_old_generations(gemm):
    previous = []
    device = torch.device("cpu")
    for size in range(1, 34):
        view = gemm._workspace(1, 1, size, device)
        assert view.shape == (1, 1, size)
        current = gemm._WORKSPACE[device, torch.float32]
        if not previous or previous[-1]() is not current:
            previous.append(weakref.ref(current))
        assert size <= current.numel() < 2 * size
        del view, current
    gc.collect()
    assert all(reference() is None for reference in previous[:-1])
    assert previous[-1]() is not None
    assert gemm._WORKSPACE_RETIRED == []


def test_only_captured_generations_survive_growth(gemm, monkeypatch):
    device = torch.device("cpu")
    gemm._workspace(1, 1, 8, device)
    captured = weakref.ref(gemm._WORKSPACE[device, torch.float32])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gemm._workspace(1, 1, 8, device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    gemm._workspace(1, 1, 9, device)
    eager = weakref.ref(gemm._WORKSPACE[device, torch.float32])
    gemm._workspace(1, 1, 17, device)
    gc.collect()
    assert eager() is None
    assert captured() is gemm._WORKSPACE_RETIRED[0]
    assert len(gemm._WORKSPACE_RETIRED) == 1
    assert sum(t.numel() for t in gemm._WORKSPACE_RETIRED) < gemm._WORKSPACE[device, torch.float32].numel()


def test_capture_requires_eager_capacity_and_keeps_existing_storage(gemm, monkeypatch):
    device = torch.device("cpu")
    gemm._workspace(1, 1, 8, device)
    old = gemm._WORKSPACE[device, torch.float32]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm this shape eagerly"):
        gemm._workspace(1, 1, 9, device)
    assert gemm._WORKSPACE[device, torch.float32] is old
    assert gemm._WORKSPACE_RETIRED == []


def test_dtypes_have_independent_capacity(gemm):
    device = torch.device("cpu")
    fp32 = gemm._workspace(1, 2, 8, device)
    bf16 = gemm._workspace(1, 2, 8, device, torch.bfloat16)
    assert fp32.dtype == torch.float32
    assert bf16.dtype == torch.bfloat16
    assert fp32.data_ptr() != bf16.data_ptr()
    assert len(gemm._WORKSPACE) == 2
