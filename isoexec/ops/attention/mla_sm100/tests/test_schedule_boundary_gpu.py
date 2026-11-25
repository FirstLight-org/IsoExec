"""The trainer adapter at the DECLARED span against the paged kernel, raw bits, at the span's own boundaries.

``mla_sm100_attention`` (what ``attention.core.mla`` resolves to on SM100) packs each sequence's latents and
runs the whole sequence in one call. The engine side of the same recipe is the paged kernel over a cache
with right-aligned queries. With ``S = fwd.segment_span()`` this module puts sequences on both sides of
every boundary the declaration has (``S - 1``, ``S``, ``S + 1`` keys, a second segment with a short tail,
the longest declared context, a third segment) and requires, for each: adapter rows == paged prefill rows
== chunked prefill rows == decode rows, a decode row alone == inside a mixed batch at every position, and
repeated calls equal.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="requires an SM100 (B200) device")

NOPE, VALUE = 64, 64


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("heads", (5, 20))
def test_adapter_rows_equal_paged_rows_at_the_declared_span_boundaries(heads):
    from isoexec.ops.attention.mla_sm100 import fwd, spec
    from isoexec.ops.attention.mla_sm100.ops import mla_sm100_attention
    from isoexec.ops.attention.mla_sm90.ops import _absorb_fwd, _unabsorb_fwd
    from isoexec.ops.attention.mla_sm90.tests._harness import SCALE, Request, Slice, assemble, decode, full

    S = fwd.segment_span()
    assert S == fwd.schedule_for(spec.LONGEST_CONTEXT).S
    lens = [S - 1, S, S + 1, S + 65, spec.LONGEST_CONTEXT, 2 * S + 5]
    T = sum(lens)
    g = torch.Generator(device="cuda").manual_seed(heads)
    mk = lambda *shape, scale=1.0: (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)
    q_nope, q_pe = mk(T, heads, NOPE, scale=0.5), mk(T, heads, 64)
    c_kv, k_pe = mk(T, 512), mk(T, 64)
    w_uk, w_uv = mk(heads, NOPE, 512, scale=0.2), mk(heads, VALUE, 512, scale=0.2)

    def adapter():
        return mla_sm100_attention(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, seq_lens=lens, scale=SCALE, seg_tokens=S)

    trainer = adapter()
    assert torch.equal(_bits(trainer), _bits(adapter())), "the adapter is not repeatable"
    assert torch.isfinite(trainer.float()).all()

    # the engine's operands: the absorbed query rows and the latent cache rows of each sequence
    q = _absorb_fwd(q_nope, w_uk, q_pe)
    kv = torch.cat((c_kv, k_pe), dim=-1)
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    reqs = [Request(kv[s:s + n].contiguous(), q[s:s + n].contiguous(), tag=f"L{n}") for s, n in zip(starts, lens)]

    def paged(slices, seed):
        b = assemble(slices, page_perm_seed=seed, fill="nan")
        out, _ = fwd.mla_sm100_fwd(b.q, b.kv_cache, b.block_table, b.cu_seqlens_q, b.seqused_k, SCALE, S,
                                   b.max_seqlen_q)
        return b, _unabsorb_fwd(out, w_uv)

    start_of = {id(r): s for r, s in zip(reqs, starts)}

    def want(slices):
        return torch.cat([trainer[start_of[id(sl.req)] + sl.q_start:start_of[id(sl.req)] + sl.q_end] for sl in slices])

    # whole prefill of every sequence in one paged call, permuted pages, NaN in unwritten slots
    sl = [full(r) for r in reqs]
    _, got = paged(sl, 1)
    assert torch.equal(_bits(got), _bits(want(sl))), "paged prefill != adapter"
    # chunked prefill: the 300 rows that straddle each boundary, over the cached prefix
    sl = [Slice(r, max(0, min(r.length, S + 150) - 300), min(r.length, S + 150)) for r in reqs]
    _, got = paged(sl, 2)
    assert torch.equal(_bits(got), _bits(want(sl))), "chunked prefill != adapter"
    # decode steps around each boundary and at each sequence's end, one mixed batch; then reversed; then alone
    for offset in (-2, -1, 0, 1, 2, 65):
        sl = [decode(r, min(r.length - 1, S - 1 + offset)) for r in reqs] + [decode(r, r.length - 1) for r in reqs]
        _, got = paged(sl, 3 + offset)
        ref = want(sl)
        assert torch.equal(_bits(got), _bits(ref)), f"decode != adapter at offset {offset}"
        _, got_r = paged(sl[::-1], 11 + offset)
        assert torch.equal(_bits(got_r), _bits(ref.flip(0))), f"reversed decode batch differs at offset {offset}"
        for j in (0, 2, len(sl) - 1):
            _, one = paged([sl[j]], 17 + j)
            assert torch.equal(_bits(one), _bits(ref[j:j + 1])), f"decode row {j} alone differs at offset {offset}"
