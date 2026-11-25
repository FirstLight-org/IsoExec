"""Shared inputs, runners and record comparisons for the gdn_cpr_sm100 gate battery (GPU only)."""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.gdn.gdn_cpr_sm100 import ops as G

DEV = "cuda"
BF = torch.bfloat16
I32 = torch.int32

SM100 = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                           reason="SM100 (B200) GPU required")


def biteq(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Integer-view equality: a tolerance is never used for a claim between kernels."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {BF: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
            torch.int32: torch.int32, torch.uint8: torch.uint8}.get(a.dtype, a.dtype)
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def make(lanes, hk, hv, seed=0, scale=1.0, slow=True):
    """Random raw inputs for a pack of sequences with the given lengths."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(lanes)
    r = lambda *s: (torch.randn(*s, device=DEV, generator=g) * scale).to(BF)
    alo, ahi = (0.002, 0.03) if slow else (1.0, 16.0)
    d = dict(q=r(T, hk, 128), k=r(T, hk, 128), v=r(T, hv, 128), a=r(T, hv), b=r(T, hv),
             A_log=torch.log(torch.empty(hv, device=DEV).uniform_(alo, ahi, generator=g)),
             dt_bias=torch.randn(hv, device=DEV, generator=g) * 0.5)
    cu = [0]
    for L in lanes:
        cu.append(cu[-1] + L)
    d["cu"] = cu
    d["cut"] = cut(cu)
    d["lanes"] = list(lanes)
    return d


def cut(cu):
    return torch.tensor(list(cu), device=DEV, dtype=I32)


def ids(xs):
    return torch.tensor(list(xs), device=DEV, dtype=I32)


def rows(d, sl):
    return [d[n][sl] for n in ("q", "k", "v", "a", "b")]


def prefill(d, sl=slice(None), cu=None, **kw):
    q, k, v, a, b = rows(d, sl)
    return G.prefill(q, k, v, a, b, d["A_log"], d["dt_bias"], d["cut"] if cu is None else cut(cu), **kw)


def decode_step(d, row_ids, state, slots, **kw):
    r = torch.as_tensor(list(row_ids), device=DEV)
    return G.decode(*rows(d, r), d["A_log"], d["dt_bias"], state, slots, **kw)


def fresh(nslots, hv, *, poison=False):
    """A pool of ``nslots`` records. ``poison`` fills every byte with 0xFF (NaN floats, pos = -1)."""
    st = G.CprState.allocate(nslots, hv, DEV, zero=not poison)
    if poison:
        st.data.fill_(0xFF)
    return st


def state_equal(a, b, slots_a, slots_b=None) -> bool:
    """Bitwise equality of what a record MEANS: S, S^, every pos copy, and the open chunk's rows.

    Bytes past the open chunk are explicitly not compared: the contract says they may hold anything.
    """
    slots_b = slots_a if slots_b is None else slots_b
    for sa, sb in zip(list(slots_a), list(slots_b)):
        sa, sb = int(sa), int(sb)
        if not (biteq(a.S[sa], b.S[sb]) and biteq(a.sh[sa], b.sh[sb]) and biteq(a.pos[sa], b.pos[sb])):
            return False
        pos = a.pos[sa]
        if not bool((pos == pos[:, :1]).all()):
            return False   # the NPOS copies must agree
        # AND EVERY HEAD OF THE SLOT MUST AGREE, which nothing asserted before round 5. The prepared
        # mode's plan kernel derives `j0` for EVERY head from head 0's `pos` (`csrc/prepared.cu`,
        # `ChunkRec`), and the round-4 plan kernel already derived each sequence's CHUNK COUNT the same
        # way, so cross-head equality is load-bearing and was argued by induction and checked nowhere.
        if not bool((pos == pos[:1, :1]).all()):
            return False
        j = int(pos[0, 0]) % 64
        for name in ("kb", "vb", "db"):
            if not biteq(getattr(a, name)[sa, :, :j], getattr(b, name)[sb, :, :j]):
                return False
        for name in ("xb", "gb", "bb"):
            if not biteq(getattr(a, name)[sa, :, :j], getattr(b, name)[sb, :, :j]):
                return False
        if j >= 16 and not biteq(a.pb[sa, :, : j % 16, : 16 * (j // 16)],
                                 b.pb[sb, :, : j % 16, : 16 * (j // 16)]):
            return False
    return True


def rel_err(got, want):
    got, want = got.double(), want.double()
    scale = max(want.abs().max().item(), 1e-30)
    return ((got - want).abs().max().item() / scale,
            ((got - want).norm() / max(want.norm().item(), 1e-30)).item())
