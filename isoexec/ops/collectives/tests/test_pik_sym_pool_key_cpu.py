"""Require staging and reduction to share wire buffers regardless of root output dtype.
CPU pool stubs check None-dtype normalization and distinct output buffers without CUDA symmetric memory."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.dirname(_HERE)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BOOT = _load_by_path("_ix_pik_bootstrap_poolkey", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
import pik.allreduce as AR  # type: ignore  # noqa: E402


class _FakeInPool:
    """Stands in for _SymInPool: fake double-buffered staging, keyed upstream by wire dtype."""

    def __init__(self, n, device, group, dtype=torch.float32):
        self.cap = max(n, 1 << 20)
        self.dtype = dtype
        self.inp = [torch.zeros(self.cap, dtype=dtype) for _ in range(2)]
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], None

    def flip(self):
        self.phase ^= 1


class _FakeOutPool:
    """Stands in for _SymOutPool: fake output buffer, keyed upstream by root dtype."""

    def __init__(self, n, device, group, out_dtype=torch.float32):
        self.cap = max(n, 1 << 20)
        self.out_dtype = out_dtype
        self.out = torch.zeros(self.cap, dtype=out_dtype)
        self.h_out = None


@pytest.fixture()
def stubbed(monkeypatch):
    monkeypatch.setattr(AR, "_SymInPool", _FakeInPool)
    monkeypatch.setattr(AR, "_SymOutPool", _FakeOutPool)
    monkeypatch.setattr(AR.dist, "get_world_size", lambda group=None: 4)
    AR._SYM.clear()
    AR._SYM_IN.clear()
    AR._SYM_OUT.clear()
    yield
    AR._SYM.clear()
    AR._SYM_IN.clear()
    AR._SYM_OUT.clear()


def test_none_out_dtype_shares_the_pool_with_its_resolved_dtype(stubbed):
    """The staging lookup and the reduce lookup must be the SAME pool.

    (fp32, None) is what ``sym_partial`` asks for; (fp32, fp32) is what ``_tree_all_reduce_p2p``
    resolves to when the caller passes no ``root_dtype``. If these differ, the GEMM stages into a
    buffer no peer ever reads and the reduce copies it across.
    """
    staged = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    reduced = AR._sym_pool(1024, "cpu", None, torch.float32, torch.float32)
    assert staged is reduced, "staging pool and reduce pool diverged -> a full copy-in per site"
    assert len(AR._SYM_IN) == 1, f"one staging pool, {len(AR._SYM_IN)} allocated"


def test_root_cast_cannot_unalias_the_staging(stubbed):
    """Require root-cast admission to retain the same wire staging buffer, regardless of output dtype."""
    staged = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    reduced = AR._sym_pool(1024, "cpu", None, torch.float32, torch.bfloat16)
    assert staged.stage()[0] is reduced.stage()[0], (
        "root-cast reduce reads a different staging buffer than sym_partial staged into -> "
        "a full copy-in per site whenever ISOEXEC_PIK_FUSED_ROOT_CAST is admitted"
    )
    assert reduced.out_dtype == torch.bfloat16 and staged.out_dtype == torch.float32
    assert staged.out is not reduced.out, "fp32 and bf16 roots must not share an output buffer"


def test_sym_partial_returns_the_buffer_the_reduce_will_stage_into(stubbed):
    """End to end at the pointer level: the whole point of ``sym_partial`` is data_ptr equality."""
    view = AR.sym_partial((16, 64), "cpu", None, dtype=torch.float32)
    inp, _ = AR._sym_pool(16 * 64, "cpu", None, torch.float32, torch.float32).stage()
    assert view.data_ptr() == inp.data_ptr(), (
        "sym_partial handed out a buffer the reduce does not read -> _tree_all_reduce_p2p takes "
        "its `partial.data_ptr() != inp.data_ptr()` copy-in branch on every single call"
    )
    # ...and the SAME pointer must serve a bf16-root reduce (the root-cast fast path)
    inp_bf16_root, _ = AR._sym_pool(16 * 64, "cpu", None, torch.float32, torch.bfloat16).stage()
    assert view.data_ptr() == inp_bf16_root.data_ptr()


def test_bf16_wire_with_fp32_root_still_gets_its_own_view(stubbed):
    """The decoupled combination keeps its own OUTPUT while SHARING the bf16 staging.

    bf16 partial + fp32 root is the MoE-combine wire saving: the wire buffer is bf16 (shared with
    every other bf16-wire site -- that is what makes ``moe_batch_invariant``'s staging zero-copy)
    and the root is fp32, which must not collapse into the bf16 output buffer.
    """
    a = AR._sym_pool(1024, "cpu", None, torch.bfloat16, torch.float32)
    b = AR._sym_pool(1024, "cpu", None, torch.bfloat16, None)
    c = AR._sym_pool(1024, "cpu", None, torch.float32, torch.float32)
    assert a is not b, "(bf16 wire, fp32 root) collapsed into (bf16, bf16)"
    assert a is not c and b is not c
    assert b.out_dtype == torch.bfloat16 and a.out_dtype == torch.float32
    assert a.stage()[0] is b.stage()[0], "same wire dtype must share the staging buffers"
    assert a.stage()[0].dtype == torch.bfloat16
    assert a.out is c.out, "fp32 roots share one output pool regardless of wire dtype"


def test_growth_still_reallocates(stubbed):
    """Splitting the pools must not disturb the capacity check."""
    small = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    small_in = small.stage()[0]
    same = AR._sym_pool(2048, "cpu", None, torch.float32, None)
    assert same is small, "still inside cap -> must reuse"
    big = AR._sym_pool((1 << 20) + 1, "cpu", None, torch.float32, None)
    assert big.stage()[0] is not small_in, "beyond cap -> must reallocate the staging"
    assert AR._sym_pool(16, "cpu", None, torch.float32, torch.float32) is big, "grown pool must be the shared one"


def test_two_groups_of_one_size_never_share_a_symmetric_pool(stubbed, monkeypatch):
    """A pool rendezvoused with one group is unusable by another group of the same size: its barrier
    waits on the first group's ranks. TP4 beside an EP4 or DP4 group hung that way, so pools key by
    the group's ranks, not its size."""
    tp, ep = object(), object()
    ranks = {id(tp): [0, 1, 2, 3], id(ep): [0, 4, 8, 12]}
    monkeypatch.setattr(AR.dist, "get_process_group_ranks", lambda g: ranks[id(g)])
    a = AR._sym_pool(1024, "cpu", tp, torch.float32, None)
    b = AR._sym_pool(1024, "cpu", ep, torch.float32, None)
    assert a is not b and len(AR._SYM_IN) == 2
    assert AR._sym_pool(1024, "cpu", tp, torch.float32, None) is a


def test_a_rendezvous_whose_ranks_are_at_different_setups_is_refused(monkeypatch):
    """torch numbers rendezvous per process: ranks at different counts would wait forever on each other's
    keys. ``rendezvous`` refuses that on every rank (one gather of the counts) and the group leaves P2P."""
    group = object()
    monkeypatch.setattr(AR, "_group_key", lambda g: ("g",))
    monkeypatch.setattr(AR.dist, "get_world_size", lambda g=None: 2)
    monkeypatch.setattr(AR.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(AR, "_SKEWED", set())
    monkeypatch.setattr(AR, "_RENDEZVOUS", [3])
    peer = [5]
    monkeypatch.setattr(AR.dist, "all_gather_into_tensor",
                        lambda out, mine, group=None: out.copy_(torch.tensor([int(mine), peer[0]])))
    with pytest.raises(AR.SymmetricSetupSkew, match=r"\[3, 5\]"):
        AR.rendezvous(torch.zeros(4), group)
    assert AR._RENDEZVOUS == [3] and ("g",) in AR._SKEWED
    assert not AR.p2p_available(group)
    peer[0] = 3  # even once the counts agree again, the group stays on NCCL: its peers already left P2P
    with pytest.raises(AR.SymmetricSetupSkew):
        AR.rendezvous(torch.zeros(4), group)
