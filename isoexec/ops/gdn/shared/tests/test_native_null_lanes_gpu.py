"""Null-block decode lanes of the native GDN core (vendored_fla ``fused_sigmoid_gating`` rewrite rule).

A decode lane whose state index is the null block (row 0: padded CUDA-graph lanes, warmup dummies) owns no
state. The kernel now writes the zero output row for it (the engine used to zero those rows with a separate
``masked_fill_`` per GDN layer). Pinned here:

* null lanes: output exactly +0.0, and no state row is read or written for them;
* valid lanes: output and state bits identical to a call that contains only the valid lanes (the change is
  confined to the early-return branch, so a valid lane's program is unchanged);
* the BV64 path and the vendored wrapper agree on both (``gdn_native_core_kernel`` dispatches either).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DEV = "cuda"
H, HV, K, V = 4, 12, 128, 128  # the Qwen3.8-27B TP4 local shard


def _inputs(n: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(n, 1, H, K, generator=g, device=DEV).to(torch.bfloat16)
    k = torch.randn(n, 1, H, K, generator=g, device=DEV).to(torch.bfloat16)
    v = torch.randn(n, 1, HV, V, generator=g, device=DEV).to(torch.bfloat16)
    a = (torch.randn(n, HV, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    b = (torch.randn(n, HV, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    A_log = torch.randn(HV, generator=g, device=DEV) * 0.3
    dt_bias = torch.randn(HV, generator=g, device=DEV) * 0.3
    return q, k, v, a, b, A_log, dt_bias


def _state(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(rows, HV, V, K, generator=g, device=DEV) * 0.1


@pytest.mark.parametrize("bv64", [False, True])
def test_null_lanes_zero_valid_lanes_unchanged(bv64, monkeypatch):
    from isoexec.ops.gdn.shared import gdn_ops

    if not bv64:
        import isoexec.ops.gdn.shared.gdn_native_core_bv64 as m

        monkeypatch.setattr(m, "maybe_native_core_bv64", lambda *a, **k: None)

    n, rows = 16, 32
    idx = torch.tensor([3, 0, 7, 0, 0, 12, 1, 5, 0, 9, 30, 0, 2, 4, 0, 6], dtype=torch.int32, device=DEV)
    q, k, v, a, b, A_log, dt_bias = _inputs(n, seed=1)
    base = _state(rows, seed=2)

    # all lanes, null ones included
    state_all = base.clone()
    o_all = gdn_ops.gdn_native_core_kernel(q, k, v, a, b, A_log, dt_bias, ssm_state=state_all,
                                           state_indices=idx, cu_seqlens=None)[:, 0]
    # only the valid lanes
    keep = (idx > 0).nonzero().flatten()
    state_ref = base.clone()
    o_ref = gdn_ops.gdn_native_core_kernel(q[keep], k[keep], v[keep], a[keep], b[keep], A_log, dt_bias,
                                           ssm_state=state_ref, state_indices=idx[keep], cu_seqlens=None)[:, 0]

    null = (idx <= 0)
    assert torch.equal(o_all[null], torch.zeros_like(o_all[null]))
    assert not torch.signbit(o_all[null].float()).any(), "null rows must be +0.0"
    assert torch.equal(o_all[keep].view(torch.int16), o_ref.view(torch.int16)), "valid lanes changed bits"
    assert torch.equal(state_all.view(torch.int32), state_ref.view(torch.int32)), "state rows differ"
    assert torch.equal(state_all[0].view(torch.int32), base[0].view(torch.int32)), "null row 0 was written"
