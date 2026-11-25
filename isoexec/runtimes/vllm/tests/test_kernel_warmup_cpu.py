"""vLLM's SM100 CuTe-DSL router warmup is skipped for IsoExec models and kept for every other model."""

import sys

import pytest

kernel_warmup = pytest.importorskip("vllm.model_executor.warmup.kernel_warmup")

_CUTEDSL_ROUTER = "vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl"


def test_router_warmup_skips_isoexec_models_only(monkeypatch):
    import torch

    from isoexec.runtimes.vllm.factory import DenseVllmModel, VllmModel
    from isoexec.runtimes.vllm.vllm_patches import patch_vllm_kernel_warmup

    calls = []
    monkeypatch.setattr(kernel_warmup, "_warmup_bf16x3_router_gemm", lambda model, n: calls.append((model, n)))
    monkeypatch.delitem(sys.modules, _CUTEDSL_ROUTER, raising=False)
    assert patch_vllm_kernel_warmup()
    assert patch_vllm_kernel_warmup()  # idempotent: one wrapper, not two
    warmup = kernel_warmup._warmup_bf16x3_router_gemm
    for cls in (DenseVllmModel, VllmModel):
        assert warmup(cls.__new__(cls), 2048) is None
    assert calls == [] and _CUTEDSL_ROUTER not in sys.modules
    other = torch.nn.Linear(2, 2)
    warmup(other, 2048)
    assert calls == [(other, 2048)]

