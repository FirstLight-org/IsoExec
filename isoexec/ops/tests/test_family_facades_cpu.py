"""Family APIs select providers lazily; runtime callers never choose an architecture."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops import _dispatch

_ROOT = Path(__file__).resolve().parents[2]

# Dispatch entry -> lazy facade; rope and gqa live under attention and conv under GDN.
FACADES = {
    "conv": "isoexec.ops.gdn.conv",
    "gdn": "isoexec.ops.gdn",
    "gqa": "isoexec.ops.attention.gqa",
    "logprobs": "isoexec.ops.logprobs",
    "moe": "isoexec.ops.moe",
    "norms": "isoexec.ops.norms",
    "rope": "isoexec.ops.attention.rope",
}


def test_importing_facades_does_not_import_backend_kernels(tmp_path):
    code = f"""
import importlib, sys
for name in {sorted(FACADES.values())!r}:
    importlib.import_module(name)
assert not any("_sm90" in name for name in sys.modules if name.startswith("isoexec.ops."))
import torch
assert not torch.cuda.is_initialized()
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(_ROOT.parent)},
    )


@pytest.mark.parametrize("family", _dispatch.FAMILIES)
def test_facade_exports_resolve_only_the_requested_module(monkeypatch, family):
    assert set(FACADES) == set(_dispatch.FAMILIES)
    facade = importlib.import_module(FACADES[family])
    calls = []
    sentinel = object()

    def resolve(selected_family, module, *, reference=False):
        calls.append((selected_family, module, reference))
        return SimpleNamespace(**{symbol: sentinel for _, symbol in facade._EXPORTS.values() if symbol})

    monkeypatch.setattr(_dispatch, "backend_module", resolve)
    for name, (module, symbol) in facade._EXPORTS.items():
        calls.clear()
        value = facade.__getattr__(name)
        assert calls == [(family, module, True)]
        assert value is sentinel if symbol else isinstance(value, SimpleNamespace)
    with pytest.raises(AttributeError, match="no public export"):
        facade.__getattr__("unknown_backend_attribute")


def test_cpu_reference_import_does_not_select_an_accelerator(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    from isoexec.ops import norms

    assert norms.RECIPE_VERSION
    assert _dispatch.selected("norms") is None
    with pytest.raises(_dispatch.ArchSelectionError, match="no accelerator implementation"):
        _dispatch.backend_module("norms", "ops")


def test_unsupported_accelerator_cannot_use_facade_reference_escape(monkeypatch):
    # sm120: a real accelerator arch no family row names (sm100 has its own row now).
    monkeypatch.setattr(_dispatch, "ARCH", "sm120")
    from isoexec.ops import norms

    with pytest.raises(_dispatch.ArchSelectionError, match="sm120"):
        norms.__getattr__("ops")


def test_synthetic_future_provider_supplies_exports_and_declarations(monkeypatch):
    """A test-only provider proves dispatch follows the row, without shipping unproved kernels."""
    from isoexec.core.registry import ImplSpec
    from isoexec.ops import gdn

    metadata = SimpleNamespace(
        IMPLS=(ImplSpec(op="gdn.core", impl_id="test_future_core", version=1, supported_archs=frozenset({"sm100"})),)
    )
    sentinel = object()
    calls = []
    provider = SimpleNamespace(build_state_pool=lambda **kwargs: (calls.append(kwargs), sentinel)[1])
    imported = []
    original = importlib.import_module

    def load(name, package=None):
        imported.append(name)
        if name == "isoexec.ops.gdn.test_future.spec":
            return metadata
        if name == "isoexec.ops.gdn.test_future":
            return provider
        return original(name, package)

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setitem(_dispatch.FAMILIES["gdn"].archs, "sm100", "gdn.test_future")
    monkeypatch.setattr(importlib, "import_module", load)
    _dispatch._assert_proven.cache_clear()
    try:
        assert gdn.build_state_pool(capacity=7) is sentinel
        assert calls == [{"capacity": 7}]
        assert _dispatch.declarations("gdn") is metadata
        assert not any("gdn_cpr_sm90" in name for name in imported)
    finally:
        _dispatch._assert_proven.cache_clear()


@pytest.mark.parametrize("arch,expected", [("sm90", "selected"), ("cpu", "reference")])
def test_gdn_state_pool_matches_selected_arithmetic(monkeypatch, arch, expected):
    from isoexec.ops import gdn

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    calls = []
    selected = SimpleNamespace(build_state_pool=lambda **kwargs: calls.append(("selected", kwargs)))
    from isoexec.ops.gdn.shared import gdn_cpr_state

    monkeypatch.setattr(gdn_cpr_state, "build_cpr_gdn", lambda **kwargs: calls.append(("reference", kwargs)))
    original = importlib.import_module

    def load(name, package=None):
        if name == f"isoexec.ops.{_dispatch.FAMILIES['gdn'].archs['sm90']}":
            return selected
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    allocator = object()
    gdn.build_state_pool(max_num_seqs=4, on_alloc=allocator)
    assert calls == [(expected, {"max_num_seqs": 4, "on_alloc": allocator})]


def test_reference_gdn_core_uses_kernel_chunk_and_expands_heads(monkeypatch):
    from isoexec.ops import gdn

    monkeypatch.setattr(_dispatch, "ARCH", "cpu")
    calls = []

    def core(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2]

    from isoexec.ops.gdn.shared import gdn_ops

    monkeypatch.setattr(gdn_ops, "fla_chunk_size", lambda: 64)
    monkeypatch.setattr(gdn_ops, "gdn_native_cpr", core)
    q = torch.ones(3, 1, 2)
    v = torch.ones(3, 2, 2)
    gate = torch.ones(3, 2)
    cu = torch.tensor([0, 3], dtype=torch.int64)
    assert torch.equal(gdn.gdn_core(q, q, v, gate, gate, None, None, cu), v)
    args, kwargs = calls[0]
    assert args[0].shape == (1, 3, 2, 2)
    assert kwargs["cu_seqlens"].dtype == torch.int32
    assert len(calls) == 1
    assert gdn.refuse_incompatible_flags() is None


def test_model_runtime_and_integration_imports_use_neutral_families():
    violations = []
    inspected = 0
    for directory in ("core", "models", "runtimes", "integrations", "checks", "sync"):
        for path in (_ROOT / directory).rglob("*.py"):
            if "tests" in path.parts:
                continue
            inspected += 1
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [item.name for item in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                if any(
                    name and "ops" in name.split(".") and any(part.endswith("_sm90") for part in name.split("."))
                    for name in names
                ):
                    violations.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert inspected > 50
    assert not violations, violations
