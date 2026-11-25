"""The trainer core with its host boundaries (``cu_seqlens_cpu``, built once per micro-batch by the runtime)
runs forward AND backward without a single device-to-host transfer or stream synchronization: CUDA's sync
debug mode turns any into an error. Gradients equal those of the call without host metadata, bitwise.

The core reads its operands as column views (q/k/v of the conv output, alpha/beta of the in_proj output),
as the trainer hands them over, with the same bits as packed operands.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import integration
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, SM90, make

pytestmark = SM90
NAMES = ("q", "k", "v", "a", "b", "A_log", "dt_bias")


@contextlib.contextmanager
def _no_host_sync():
    torch.cuda.synchronize()
    before = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(before)


def _column_views(d, hk, hv):
    """q/k/v as adjacent column windows of one [T, (2hk + hv) 128] buffer, alpha/beta as columns of another."""
    T = d["q"].shape[0]
    qkv = torch.cat([d["q"].reshape(T, -1), d["k"].reshape(T, -1), d["v"].reshape(T, -1)], -1)
    kd = hk * 128
    proj = torch.cat([torch.zeros(T, 24, device=DEV, dtype=BF), d["b"], d["a"]], -1)
    return (qkv[:, :kd].unflatten(-1, (hk, 128)), qkv[:, kd:2 * kd].unflatten(-1, (hk, 128)),
            qkv[:, 2 * kd:].unflatten(-1, (hv, 128)), proj[:, 24 + hv:], proj[:, 24:24 + hv])


@pytest.mark.parametrize("save_states", [False, True])
@pytest.mark.parametrize("lanes,hk,hv", [([2900, 2911], 4, 12), ([600, 64, 1, 1300, 77], 2, 6)])
def test_forward_backward_with_host_boundaries_never_syncs(lanes, hk, hv, save_states):
    d = make(lanes, hk, hv, seed=17, scale=0.3)
    host = torch.tensor(d["cu"], dtype=torch.int32)
    do = (torch.randn(sum(lanes), hv, 128, device=DEV) * 0.01).to(BF)

    def run(cu_cpu, views):
        leaves = [d[n].detach().clone().requires_grad_(True) for n in NAMES]
        if views:
            q, k, v, a, b = _column_views({n: t for n, t in zip(NAMES, leaves)}, hk, hv)
            inputs = [q, k, v, a, b, leaves[5], leaves[6]]
        else:
            inputs = leaves
        o = integration.gdn_cpr_sm90_core(*inputs, d["cut"], save_states=save_states, cu_seqlens_cpu=cu_cpu)
        return o, torch.autograd.grad(o, leaves, do)

    # Both sides take the column views: autograd's slice accumulation turns a -0 gradient into +0 exactly
    # as the trainer's split-columns backward does, so packed leaves are not the comparison for views.
    want, want_g = run(None, True)
    packed, packed_g = run(host, False)
    assert bitwise_equal(packed, want)
    run(host, True)   # warm every lazy load / compile outside the checked region
    with _no_host_sync():
        got, got_g = run(host, True)
    torch.cuda.synchronize()
    assert bitwise_equal(got, want)
    for name, g, w in zip(NAMES, got_g, want_g):
        assert bitwise_equal(g, w), f"d{name} differs"
