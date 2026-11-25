"""Engine context parallelism (vLLM PCP/DCP) is refused for IsoExec models before any worker starts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isoexec.checks import Refusal
from isoexec.runtimes.vllm.validation import (
    ENGINE_CONTEXT_PARALLEL_REFUSAL,
    context_parallel_refusal,
    engine_kwargs_context_parallel_refusal,
    parallel_config_context_parallel_refusal,
)


@pytest.mark.parametrize(("pcp", "dcp"), [(1, 1), (None, None), (None, 1)])
def test_no_engine_context_parallelism_is_admitted(pcp, dcp):
    assert context_parallel_refusal(pcp, dcp) is None
    assert engine_kwargs_context_parallel_refusal({}) is None
    assert engine_kwargs_context_parallel_refusal(None) is None
    assert engine_kwargs_context_parallel_refusal({"tensor_parallel_size": 4}) is None


@pytest.mark.parametrize(("pcp", "dcp"), [(2, 1), (4, 1), (1, 2), (1, 4), (2, 2)])
def test_prefill_and_decode_context_parallelism_are_refused(pcp, dcp):
    for refusal in (context_parallel_refusal(pcp, dcp),
                    engine_kwargs_context_parallel_refusal({"prefill_context_parallel_size": pcp,
                                                            "decode_context_parallel_size": dcp}),
                    parallel_config_context_parallel_refusal(
                        SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp))):
        assert refusal is not None and refusal.startswith(ENGINE_CONTEXT_PARALLEL_REFUSAL)
        assert f"prefill_context_parallel_size={pcp}" in refusal and f"decode_context_parallel_size={dcp}" in refusal


@pytest.mark.parametrize("key", ["prefill_context_parallel_size", "decode_context_parallel_size"])
def test_register_native_refuses_engine_context_parallelism(key):
    from isoexec.runtimes.vllm.registration import register_native

    # The refusal precedes every use of the source, config or checkpoint.
    with pytest.raises(Refusal, match="engine context parallelism is not supported"):
        register_native(None, {}, checkpoint="/nonexistent", engine_kwargs={key: 2})


def test_vllm_config_verification_refuses_native_architectures_only(monkeypatch):
    from vllm.config.model import ModelConfig

    from isoexec.runtimes.vllm import validation
    from isoexec.runtimes.vllm.registration import NATIVE_ARCH, NATIVE_DENSE_ARCH

    seen = []
    native_verify = lambda self, parallel_config: seen.append(self.architectures[0])  # noqa: E731
    monkeypatch.setattr(ModelConfig, "verify_with_parallel_config", native_verify)
    validation.install_context_parallel_refusal((NATIVE_ARCH, NATIVE_DENSE_ARCH))
    validation.install_context_parallel_refusal((NATIVE_ARCH, NATIVE_DENSE_ARCH))  # idempotent
    verify = ModelConfig.verify_with_parallel_config
    plain = SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
    for arch in (NATIVE_ARCH, NATIVE_DENSE_ARCH):
        verify(SimpleNamespace(architectures=[arch]), plain)
        for parallel in (SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=1),
                         SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=2)):
            with pytest.raises(Refusal, match="use trainer CP with an engine without CP"):
                verify(SimpleNamespace(architectures=[arch]), parallel)
    # Other models keep vLLM's own verification (which decides about their context parallelism).
    verify(SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
           SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=2))
    assert seen == [NATIVE_ARCH, NATIVE_DENSE_ARCH, "DeepseekV3ForCausalLM"]
