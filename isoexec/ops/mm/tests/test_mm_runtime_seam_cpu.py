"""The runtime seam must be INVISIBLE to everything that inspects ``bi.matmul_persistent``.

The seam points vLLM's batch-invariant module at our vendored kernels (kernel-ownership.md §5.3).
It is the one thing under `ops/mm/` that can break a caller without breaking a number, because two
separate mechanisms in this tree read the *object* bound there rather than calling it:

  1. ``ops/mm/trainer.mm_impl_id()`` reads ``bi.matmul_persistent._isoexec_sm90_segtree`` (and
     ``_isoexec_sm100``) to decide whether this process owes the registry's ``sm90_wgmma_segtree``
     or its ``triton_batch_invariant``. That predicate exists so a process cannot ATTEST a provider it
     does not run -- it is what retired the `fingerprint:mm:*` exemption. A seam that hides the marker
     makes the engine attest Triton while running the segment-tree GEMM, and
     `fingerprint:mm:engine_{prefill,decode}` violates on every engine run. (This is not hypothetical:
     the first version of the seam did exactly that, and `gate_vllm_engine.py` caught it as
     `refused=2`.)
  2. Every installed provider stamps its own marker, and the identity tests read it back.

So the contract of the seam is not "calls the right kernel", it is "calls the right kernel AND
answers questions about itself as if it were that kernel". This test pins the second half, on the
CPU, with no vLLM present -- the proxy is pure Python and the delegation is the whole mechanism.

Also pins the reason the seam cannot simply bind by value: `install_mm_sm90` (and sm100's
`install_mm_sm100`) rebind the vendored module's global AFTER the seam may have run.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

V = importlib.import_module("isoexec.ops.mm.mm_batch_invariant")
seam = importlib.import_module("isoexec.ops.mm.mm_runtime_seam")


@pytest.fixture
def restore_v():
    saved = V.matmul_persistent
    yield
    V.matmul_persistent = saved


def test_proxy_delegates_provider_markers(restore_v):
    """A marker stamped by a LATER install must be visible through the seam's proxy."""
    proxy = seam._mm_provider

    def fake_segtree(a, b, bias=None):
        return ("segtree", a, b, bias)

    fake_segtree._isoexec_sm90_segtree = True
    V.matmul_persistent = fake_segtree

    assert getattr(proxy, "_isoexec_sm90_segtree", False) is True, (
        "the seam hides the installed provider's marker -- ops/mm/trainer.mm_impl_id() would "
        "report `triton_batch_invariant` for a process running the segment-tree GEMM, and "
        "fingerprint:mm:engine_* would violate on every engine run"
    )
    assert proxy(1, 2, bias=3) == ("segtree", 1, 2, 3)
    from isoexec.ops.mm.trainer import mm_impl_id

    assert mm_impl_id() == "sm90_wgmma_segtree"


def test_proxy_delegates_a_second_marker(restore_v):
    proxy = seam._mm_provider

    def fake_sm100(a, b, bias=None):
        return "sm100"

    fake_sm100._isoexec_sm100 = True
    V.matmul_persistent = fake_sm100
    assert getattr(proxy, "_isoexec_sm100", False) is True
    assert getattr(proxy, "_isoexec_sm90_segtree", False) is False, "a marker nobody stamped must stay False"


def test_proxy_resolves_late_not_by_value(restore_v):
    """Rebinding the vendored global after the fact must change what the seam calls.

    This is the split-brain guard: `install_mm_sm90` / `install_mm_sm100` rebind that global at
    install time, and the engine's entry point has to follow them or the two runtimes run different
    kernels while each side's own bitwise gate stays green.
    """
    proxy = seam._mm_provider
    V.matmul_persistent = lambda a, b, bias=None: "first"
    assert proxy(0, 0) == "first"
    V.matmul_persistent = lambda a, b, bias=None: "second"
    assert proxy(0, 0) == "second"


def test_proxy_declares_itself_vendored(restore_v):
    """`_isoexec_vendored` is the proxy's OWN marker -- the plumbing gate asserts it on the engine
    side to prove the seam installed at all, so it must not be delegated away."""
    V.matmul_persistent = lambda a, b, bias=None: None
    assert seam._mm_provider._isoexec_vendored is True


def test_seam_is_a_noop_without_a_runtime():
    """A trainer-only process has no engine to patch and must not raise or claim it patched one."""
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vLLM present; this pins the no-runtime path")
    assert seam._runtime_module() is None
    assert seam.runtime_batch_invariant_mode_on() is False
    assert seam.install_vendored_mm() is False
    assert seam.vendored_mm_installed() is False


def test_vendored_kernels_are_reachable_without_a_runtime():
    """The whole point of §5.3: the numerics import torch and our tree, and nothing else."""
    for name in ("matmul_persistent", "mm_batch_invariant", "addmm_batch_invariant", "bmm_batch_invariant"):
        assert callable(getattr(V, name))
    assert V.provenance()["module"] == "vllm.model_executor.layers.batch_invariant"
