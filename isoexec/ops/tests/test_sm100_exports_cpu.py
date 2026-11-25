"""Every family facade export, and every facade name production code imports, resolves on SM100.

A facade resolves its exports through the arch table at call time, so a name the selected SM100
package does not provide only fails when the first caller asks for it -- on a B200, at engine or
trainer start. These tests ask for every one of them with the arch pinned, on a CPU.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

from isoexec.ops import _dispatch

_ROOT = Path(__file__).resolve().parents[2]

FACADES = {
    "conv": "isoexec.ops.gdn.conv",
    "gdn": "isoexec.ops.gdn",
    "gqa": "isoexec.ops.attention.gqa",
    "logprobs": "isoexec.ops.logprobs",
    "moe": "isoexec.ops.moe",
    "norms": "isoexec.ops.norms",
    "rope": "isoexec.ops.attention.rope",
}

#: SM90 recipes with an SM100 counterpart bound by the resolver instead. Only ``ops/`` may name them:
#: the importer is an SM90-only stage (``ops/moe/eager.combine_fixed_order`` is ``pik_leaf_tree``).
SM90_ONLY = {
    "moe": {
        "build_combine_rows",  # SM100 folds with the native combine (sm100_after_fc2_combine_v2)
        "engine_route_mechanics_o2",  # SM90 fused O2 router; SM100 binds sm100_fused_topk_v2
        "engine_router_o2_counts",
        "fused_combine",
        "route_to_permuted_rows",  # SM90 fused permute; SM100 dispatches with moe_sm100.ops.route
    },
    "rope": {"rope_apply_eager"},  # Megatron's bf16 rope; rope_sm100 is a different recipe
    # The vocab-parallel sampling head engages only on sm90 (vllm_patches checks capability (9, 0));
    # SM100 keeps the gathered-vocabulary sampler path.
    "logprobs": {"shard_sample", "shard_sample_build"},
}


def _names(module) -> dict[str, tuple]:
    return {**module._EXPORTS, **getattr(module, "_SHARED_EXPORTS", {})}


def test_facades_cover_the_arch_table():
    assert set(FACADES) == set(_dispatch.FAMILIES)


@pytest.mark.parametrize("family", sorted(FACADES))
def test_every_export_resolves_on_sm100(monkeypatch, family):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    facade = importlib.import_module(FACADES[family])
    refused = SM90_ONLY.get(family, set())
    assert refused <= set(facade._EXPORTS), "an SM90-only name must be a kernel-package export"
    for name in sorted(_names(facade)):
        if name in refused:
            with pytest.raises(_dispatch.ArchSelectionError, match=name):
                facade.__getattr__(name)
        else:
            assert facade.__getattr__(name) is not None, name


@pytest.mark.parametrize("family", sorted(SM90_ONLY))
def test_sm90_only_exports_still_resolve_on_sm90(monkeypatch, family):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    facade = importlib.import_module(FACADES[family])
    for name in sorted(SM90_ONLY[family]):
        assert facade.__getattr__(name) is not None, name


def test_shared_exports_have_one_owner_on_every_arch(monkeypatch):
    """Process-wide state (the storage epoch, the injected allocator) must be one module object."""
    from isoexec.ops import moe

    seen = {}
    for arch in ("sm90", "sm100", "cpu"):
        monkeypatch.setattr(_dispatch, "ARCH", arch)
        seen[arch] = {name: moe.__getattr__(name) for name in moe._SHARED_EXPORTS}
    assert seen["sm90"] == seen["sm100"] == seen["cpu"]
    assert seen["sm100"]["param_storage"] is importlib.import_module("isoexec.ops.moe.shared.param_storage")
    assert seen["sm100"]["set_buffer_allocator"] is seen["sm100"]["param_storage"].set_buffer_allocator


def test_shared_exports_keep_the_unsupported_arch_gate(monkeypatch):
    from isoexec.ops import moe

    monkeypatch.setattr(_dispatch, "ARCH", "sm120")
    with pytest.raises(_dispatch.ArchSelectionError, match="sm120"):
        moe.__getattr__("set_buffer_allocator")


def _facade_imports():
    """``(importer, facade module, name)`` for every production ``from <facade> import name``."""
    facades = set(FACADES.values())
    for path in sorted(_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        dotted = ".".join(("isoexec",) + path.relative_to(_ROOT).with_suffix("").parts)
        package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
        package = package.removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module in facades:
                for alias in node.names:
                    yield path.relative_to(_ROOT).as_posix(), module, alias.name


def test_production_facade_imports_resolve_on_sm100(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    family_of = {module: family for family, module in FACADES.items()}
    checked, problems = 0, []
    for importer, module, name in _facade_imports():
        family = family_of[module]
        facade = importlib.import_module(module)
        submodule = hasattr(facade, "__path__") and importlib.util.find_spec(f"{module}.{name}") is not None
        if name in vars(facade) or submodule:
            continue  # defined on the facade itself, or an arch-neutral submodule
        checked += 1
        if name in SM90_ONLY.get(family, ()):
            if not importer.startswith("ops/"):
                problems.append(f"{importer}: {module}.{name} is an SM90-only export")
            continue
        try:
            facade.__getattr__(name)
        except Exception as e:  # noqa: BLE001 -- report every failure, not the first
            problems.append(f"{importer}: {module}.{name}: {type(e).__name__}: {e}")
    assert checked > 10
    assert not problems, "\n".join(problems)
