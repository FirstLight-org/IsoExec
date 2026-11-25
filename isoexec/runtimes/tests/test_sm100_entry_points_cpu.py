"""Runtime entry points import, register and place on SM100 without a GPU.

The vLLM plugin is registered in a fresh interpreter, as vLLM's plugin loader does in every worker;
placement binds every op of the released model geometries for the trainer and the engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from isoexec.ops import _dispatch

_ROOT = Path(__file__).resolve().parents[2]

_MODELS = {
    "isoexec.models.qwen3_5": "qwen3_5/tests/fixtures/qwen3_5_9b.json",
    "isoexec.models.qwen3_5_moe": "qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json",
    "isoexec.models.qwen2_moe": "qwen2_moe/tests/fixtures/qwen1_5_moe_a2_7b.json",
}

_REGISTER = """
import os
from isoexec.ops import _dispatch
_dispatch.ARCH = "sm100"
os.environ["VLLM_ISOEXEC_REGISTERED"] = "1"
from isoexec.runtimes.vllm import factory, model, vllm_plugin  # noqa: F401
vllm_plugin.register()
from isoexec.ops.moe import param_storage
assert vllm_plugin._installed
assert param_storage._ALLOCATOR is vllm_plugin.alloc_sleepable
from vllm.model_executor.warmup import kernel_warmup
assert kernel_warmup._warmup_bf16x3_router_gemm._isoexec_skips_native
import torch
assert not torch.cuda.is_initialized()
print("REGISTERED")
"""


def _subprocess_env() -> dict[str, str]:
    """A CPU-only child environment with the repo root first on PYTHONPATH.

    The caller's PYTHONPATH is kept after the root: it is how host packages (e.g. a SkyRL checkout)
    are selected, and dropping it would silently swap in whatever the venv installed instead.
    """
    path = os.pathsep.join(p for p in (str(_ROOT.parent), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": path}


def test_vllm_plugin_registers_on_sm100(tmp_path):
    pytest.importorskip("vllm")
    env = _subprocess_env()
    env.pop("VLLM_PLUGINS", None)
    done = subprocess.run([sys.executable, "-B", "-c", _REGISTER], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and "REGISTERED" in done.stdout, done.stdout[-2000:] + done.stderr[-4000:]


@pytest.mark.parametrize("package", sorted(_MODELS))
@pytest.mark.parametrize("side", ["trainer", "engine"])
def test_released_models_place_on_sm100(monkeypatch, package, side):
    from isoexec import nn
    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.parallel import trivial_placement

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    raw = json.loads((_ROOT / "models" / _MODELS[package]).read_text())
    model = ModelSource(package).build_meta(raw)
    nn.place(model, trivial_placement(side=side))
    bound = [b for slots in nn.resolved_ops(model).values() for b in slots.values()]
    assert bound
    accelerated = {b.provider for b in bound if b.provider.rsplit(".", 1)[-1].endswith("_sm100")}
    assert {"ops.mm.mm_sm100", "ops.norms.norms_sm100"} <= accelerated
    assert not [b for b in bound if "_sm90" in b.provider], "an SM90 package served an SM100 binding"
    if package.endswith("moe"):
        assert {b.provider for b in bound if b.op.startswith("moe.")} == {"ops.moe.moe_sm100"}


def test_storage_entry_points_on_sm100(monkeypatch):
    """The runtimes' packing calls reach the MoE storage bookkeeping through the facade."""
    import torch

    from isoexec import storage
    from isoexec.compiler.source import ModelSource
    from isoexec.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
    from isoexec.models.qwen3_5_moe.tests.tiny import TINY_MOE_HF

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    config = ModelSource("isoexec.models.qwen3_5_moe").build_config(TINY_MOE_HF)
    model = Qwen3_5MoeForConditionalGeneration(config, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert storage.attach_packed_experts(model) > 0
    storage.prepare_packed_storage(model)
    storage.refresh_packed_storage(model)
    storage.release_packing_to_storage_owner(model)


_IMPORT_ALL = """
import importlib, pkgutil
from isoexec.ops import _dispatch
_dispatch.ARCH = "sm100"
imported, missing = 0, set()
for package in ("isoexec.runtimes", "isoexec.integrations", "isoexec.models", "isoexec.storage",
                "isoexec.sync", "isoexec.compiler"):
    root = importlib.import_module(package)
    names = [package] + [i.name for i in pkgutil.walk_packages(root.__path__, prefix=package + ".")
                         if ".tests" not in i.name]
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as e:
            # Only a third-party host package (or a host version a module refuses) may be absent.
            if (e.name or "").split(".")[0] == "isoexec":
                raise
            missing.add(e.name or name)
            continue
        imported += 1
assert imported > 50, (imported, missing)
print("IMPORTED", imported, sorted(missing))
"""


def test_runtime_and_integration_modules_import_on_sm100(tmp_path):
    """In a fresh interpreter: importing real host modules here would replace other tests' fakes."""
    env = _subprocess_env()
    done = subprocess.run([sys.executable, "-B", "-c", _IMPORT_ALL], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0 and "IMPORTED" in done.stdout, done.stdout[-2000:] + done.stderr[-4000:]


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_attention_and_gdn_projections_pack_on_both_arches(monkeypatch, arch):
    """Qwen3.5-0.8B's q|k|v|gate and GDN in_proj at K=1024. SM100's K segmentation reads K alone, so
    the packed weight is the program the separate member GEMMs (the trainer's) run, as on SM90."""
    from isoexec import nn
    from isoexec.ops.mm.packed_proj import attach
    from isoexec.runtimes.parallel import trivial_placement

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    attention = nn.Attention(hidden=1024, heads=8, kv_heads=2, head_dim=256, rope_theta=1e7,
                             rotary_fraction=0.25, qk_norm="rms_zero_centered", eps=1e-6,
                             out_gate="sigmoid", device="meta")
    gdn = nn.GDN(hidden=1024, key_heads=16, value_heads=16, key_head_dim=128, value_head_dim=128,
                 conv_kernel=4, output_activation="silu", eps=1e-6, device="meta")
    for prefix, module in (("layers.3.mixer", attention), ("layers.0.mixer", gdn)):
        nn.place(module, trivial_placement(side="engine"), prefix=prefix)
    assert attach(attention) == 1
    assert attach(gdn) == 1
