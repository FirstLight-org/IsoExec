"""Sparse-attention backward provider rules (ops.backward_provider): cuDNN deterministic by default, IsoExec's
atomics-free backward where cuDNN lacks the interface, the non-deterministic mode refused by name (diagnostic
override only, never at TP > 1)."""
import pytest

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("ISOEXEC_DSA_BWD", raising=False)
    monkeypatch.delenv("ISOEXEC_DSA_BWD_ALLOW_NONDET", raising=False)
    monkeypatch.setattr(ops, "BWD_PROVIDER", None)
    monkeypatch.setattr(ops, "BWD_TP_DEGREE", 1)

    def cudnn(ns, det):
        monkeypatch.setattr(ops, "_CUDNN_DSA", {"ns": ns, "det": det})
    return monkeypatch, cudnn


def test_default_and_fallback(env):
    mp, cudnn = env
    cudnn(object(), True)
    assert ops.backward_provider() == "cudnn_det"
    cudnn(object(), False)                      # an older cuDNN without `deterministic`
    assert ops.backward_provider() == "isoexec"
    cudnn(None, False)                          # no cuDNN DSA namespace
    assert ops.backward_provider() == "isoexec"
    mp.setenv("ISOEXEC_DSA_BWD", "isoexec")
    cudnn(object(), True)
    assert ops.backward_provider() == "isoexec"


def test_nondeterministic_refused(env):
    mp, cudnn = env
    cudnn(object(), True)
    mp.setenv("ISOEXEC_DSA_BWD", "cudnn")
    with pytest.raises(ops.NondeterministicBackwardError):
        ops.backward_provider()
    mp.setenv("ISOEXEC_DSA_BWD_ALLOW_NONDET", "1")
    assert ops.backward_provider() == "cudnn"
    for tp in (2, None):                         # TP > 1, or a multi-process run of unknown TP degree
        mp.setattr(ops, "BWD_TP_DEGREE", tp)
        if tp is None:
            mp.setattr(ops, "_tp_degree", lambda: None)
        with pytest.raises(ops.NondeterministicBackwardError):
            ops.backward_provider()


def test_unknown_name_and_tags(env):
    mp, _ = env
    mp.setenv("ISOEXEC_DSA_BWD", "fast")
    with pytest.raises(ValueError):
        ops.backward_provider()
    assert set(ops.BWD_PROGRAM_TAGS) == {"cudnn_det", "isoexec", "cudnn"}
    assert len(set(ops.BWD_PROGRAM_TAGS.values())) == 3


def test_head_counts_cudnn_does_not_serve_take_the_isoexec_kernel():
    """cuDNN's DSA backward serves 16/32/64/96/128 query heads per rank; other counts (e.g. 8 on a small checkpoint)
    take the IsoExec fallback by shape instead of raising inside cuDNN."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops
    assert ops.provider_for_heads("cudnn_det", 64) == "cudnn_det"
    assert ops.provider_for_heads("cudnn_det", 8) == "isoexec"
    assert ops.provider_for_heads("cudnn", 8) == "isoexec"
    assert ops.provider_for_heads("isoexec", 64) == "isoexec"
