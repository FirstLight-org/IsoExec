"""Which arch provides the decode sampler, and the flag that keeps vLLM's own sampler."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from isoexec.runtimes.vllm import vllm_patches as patches  # noqa: E402


@pytest.mark.parametrize("arch", ["sm90", "cpu"])
def test_no_decode_sampler_off_sm100(monkeypatch, arch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", arch)
    assert patches._resolve_decode_sampler() is None


def test_sm100_resolves_the_logprobs_sm100_sampler(monkeypatch):
    from isoexec.ops import _dispatch
    from isoexec.ops.logprobs.logprobs_sm100 import sampler

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    assert patches._resolve_decode_sampler() is sampler


def test_disabled_by_flag_installs_nothing(monkeypatch):
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    before = Sampler.__call__
    monkeypatch.setenv("ISOEXEC_FUSED_SAMPLER", "0")
    assert patches.patch_vllm_v2_fused_sampler() is False
    assert Sampler.__call__ is before
