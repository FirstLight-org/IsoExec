"""Shared inputs, runners and state comparisons for the gdn_cpr_sm90 gate battery (GPU only)."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G

DEV = "cuda"
BF = torch.bfloat16
I32 = torch.int32

SM90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="SM90 GPU required"
)

# every decode execution plan (CTAs per head); results must not depend on it
DECODE_PLANS = list(G._DECODE_SPLITS)


def make(lanes, hk, hv, seed=0, scale=1.0):
    """Random raw inputs for a pack of sequences with the given lengths."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(lanes)
    r = lambda *s: (torch.randn(*s, device=DEV, generator=g) * scale).to(BF)
    d = dict(q=r(T, hk, 128), k=r(T, hk, 128), v=r(T, hv, 128), a=r(T, hv), b=r(T, hv),
             A_log=torch.log(torch.empty(hv, device=DEV).uniform_(1, 16, generator=g)),
             dt_bias=torch.randn(hv, device=DEV, generator=g) * 0.5)
    cu = [0]
    for L in lanes:
        cu.append(cu[-1] + L)
    d["cu"] = cu
    d["cut"] = cut(cu)
    return d


def cut(cu):
    return torch.tensor(cu, device=DEV, dtype=I32)


def ids(xs):
    return torch.tensor(list(xs), device=DEV, dtype=I32)


def rows(d, sl):
    """The raw token inputs of a row selection (a slice or an index tensor)."""
    return [d[n][sl] for n in ("q", "k", "v", "a", "b")]


def prefill(d, sl=slice(None), cu=None, **kw):
    q, k, v, a, b = rows(d, sl)
    return G.prefill(q, k, v, a, b, d["A_log"], d["dt_bias"], d["cut"] if cu is None else cut(cu), **kw)


def decode_stream(d, lanes, state, *, slot_of=None, start=None, plan=None, order=None, steps=None):
    """Feed sequence n's tokens [start[n], lanes[n]) one per decode step, into slot ``slot_of[n]``.

    ``plan`` is a split or a callable step -> split. Returns the outputs of the
    decoded rows scattered into a [T, hv, 128] tensor (other rows zero).
    """
    n = len(lanes)
    cu = d["cu"]
    slot_of = slot_of or [i + 1 for i in range(n)]
    start = start or [0] * n
    out = torch.zeros(sum(lanes), d["v"].size(1), 128, device=DEV, dtype=BF)
    for t in range(steps if steps is not None else max(lanes)):
        idx = [i for i in range(n) if start[i] + t < lanes[i]]
        if not idx:
            break
        if order is not None:
            idx = [i for i in order if i in idx]
        r = torch.tensor([cu[i] + start[i] + t for i in idx], device=DEV)
        split = plan(t) if callable(plan) else plan
        out[r] = G.decode(*rows(d, r), d["A_log"], d["dt_bias"], state, ids(slot_of[i] for i in idx),
                          split=split)
    return out


def fresh(nslots, hv, *, poison=False):
    """A pool of ``nslots`` records. ``poison`` fills every byte with 0xFF (NaN floats, pos = -1)."""
    st = G.CprState.allocate(nslots, hv, DEV, zero=not poison)
    if poison:
        st.data.fill_(0xFF)
    return st


def state_equal(a, b, slots_a, slots_b=None) -> bool:
    """Bitwise equality of what a record means: S, S^, every pos copy, and the open chunk's rows."""
    slots_b = slots_a if slots_b is None else slots_b
    for sa, sb in zip(list(slots_a), list(slots_b)):
        if not (bitwise_equal(a.S[sa], b.S[sb]) and bitwise_equal(a.sh[sa], b.sh[sb])
                and bitwise_equal(a.pos[sa], b.pos[sb])):
            return False
        pos = a.pos[sa]
        if not bool((pos == pos[:, :1]).all()):
            return False   # the NPOS copies must agree
        j = int(pos[0, 0]) % 64
        for name in ("kb", "xb", "gb", "bb"):
            if not bitwise_equal(getattr(a, name)[sa, :, :j], getattr(b, name)[sb, :, :j]):
                return False
        for name in ("vb", "db"):
            if not bitwise_equal(getattr(a, name)[sa, :, :, :j], getattr(b, name)[sb, :, :, :j]):
                return False
        if j >= 16 and not bitwise_equal(a.pb[sa, :, : j % 16, : 16 * (j // 16)], b.pb[sb, :, : j % 16, : 16 * (j // 16)]):
            return False
    return True


def rel_err(got, want):
    got, want = got.double(), want.double()
    scale = max(want.abs().max().item(), 1e-30)
    return (got - want).abs().max().item() / scale, ((got - want).norm() / max(want.norm().item(), 1e-30)).item()
