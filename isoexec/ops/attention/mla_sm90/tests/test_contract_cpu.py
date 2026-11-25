"""CPU gates: kernel constants, lazy exports, reference self-consistency, op argument refusals."""


import pytest
import torch

from isoexec.ops.attention.mla_sm90 import spec
from isoexec.ops.attention.mla_sm90.reference import mla_reference_fwd, reference_attention


def test_this_package_declares_no_impl_and_the_family_declares_it():
    """The family owns registry declarations; the kernel package keeps its constants."""
    from isoexec.core.registry_build import build_registry

    assert not hasattr(spec, "IMPLS")
    assert spec.PAGE == 64
    assert set(build_registry().get_op("attention.core.mla").impls) == {
        "fa3_absorbed_ns1", "sm90_absorbed_segtree", "sm100_absorbed_segtree",
    }


def test_the_segtree_declaration_carries_the_contract_this_kernel_was_built_to():
    """SEG changes the rounding tree and is the only declared input.
    Page size and KV dtype are constants; the SEG-tree recipe has no bitwise twin claim with FA3."""
    from isoexec.core.registry import ANY
    from isoexec.core.registry_build import build_registry

    impl = build_registry().get_op("attention.core.mla").impls["sm90_absorbed_segtree"]
    assert impl.version == 1
    assert impl.supported_archs == frozenset({"sm90"})
    assert impl.inputs == {"seg_tokens": ANY}
    assert impl.twin_class == ""
    assert "num_splits" not in impl.inputs
    # The two recipes therefore sit in different equivalence classes, which is the whole reason the
    # selection has to be one decision both hosts read rather than two that happen to agree.
    fa3 = build_registry().get_op("attention.core.mla").impls["fa3_absorbed_ns1"]
    assert fa3.impl_id != impl.impl_id and not (impl.twin_class and impl.twin_class == fa3.twin_class)


def test_lazy_exports_tolerate_missing_kernels():
    import isoexec.ops.attention.mla_sm90 as pkg

    assert pkg.available("mla_reference_fwd")
    for name in ("mla_sm90_fwd", "mla_sm90_attention_grads"):
        if not pkg.available(name):
            with pytest.raises(ImportError):
                getattr(pkg, name)
    with pytest.raises(AttributeError):
        pkg.not_an_export


def _toy(T_lens, used, heads=2, blocks=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(sum(T_lens), heads, 576, generator=g, dtype=torch.float64) * .2
    kv = torch.randn(blocks, 64, 1, 576, generator=g, dtype=torch.float64) * .2
    table = torch.tensor([[2, 0, 5], [1, 4, 3]][: len(used)], dtype=torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(T_lens).cumsum(0)), dtype=torch.int32)
    return q, kv, table, cu, torch.tensor(used, dtype=torch.int32)


def test_reference_forward_matches_dense_softmax():
    q, kv, table, cu, used = _toy([3, 70], [100, 130])
    out, lse = mla_reference_fwd(q, kv, table, cu, used, 0.25)
    for b, (s, e) in enumerate([(0, 3), (3, 73)]):
        keys = kv[table[b].long()].reshape(-1, 576)[: used[b]]
        for i in range(s, e):
            p = int(used[b]) - (e - s) + (i - s)
            score = torch.einsum("hd,kd->hk", q[i], keys[: p + 1]) * 0.25
            torch.testing.assert_close(lse[i], torch.logsumexp(score, -1), rtol=0, atol=1e-12)
            torch.testing.assert_close(out[i], score.softmax(-1) @ keys[: p + 1, :512], rtol=0, atol=1e-12)


def test_reference_is_invariant_to_page_ids_and_future_tokens():
    q, kv, table, cu, used = _toy([3, 70], [100, 130])
    out, lse = mla_reference_fwd(q, kv, table, cu, used, 0.25)
    kv2 = kv.clone()
    kv2[0, 36:] = float("nan")  # beyond seqused of request 0 (page 0 holds tokens 64..127, used 100)
    kv2[3, 2:] = float("inf")  # beyond seqused of request 1 (page 3 holds tokens 128.., used 130)
    out2, lse2 = mla_reference_fwd(q, kv2, table, cu, used, 0.25)
    assert torch.equal(out, out2) and torch.equal(lse, lse2)


def test_absorbed_reference_equals_unabsorbed_reference():
    g = torch.Generator().manual_seed(9)
    lens, H = [5, 70], 2
    T = sum(lens)
    qn, qp = torch.randn(T, H, 16, generator=g, dtype=torch.float64), torch.randn(T, H, 64, generator=g, dtype=torch.float64)
    c, kp = torch.randn(T, 512, generator=g, dtype=torch.float64) * .1, torch.randn(T, 64, generator=g, dtype=torch.float64) * .1
    wk, wv = torch.randn(H, 16, 512, generator=g, dtype=torch.float64) * .05, torch.randn(H, 512, 512, generator=g, dtype=torch.float64) * .05
    y = reference_attention(qn, qp, c, kp, wk, wv, seq_lens=lens, scale=.125)
    ql = torch.einsum("thn,hnl->thl", qn, wk)
    q = torch.cat((ql, qp), -1)
    kv = torch.zeros(3, 64, 1, 576, dtype=torch.float64)
    kv.view(-1, 576)[:5] = torch.cat((c, kp), -1)[:5]
    kv.view(-1, 576)[64:134] = torch.cat((c, kp), -1)[5:]
    table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32)
    out, _ = mla_reference_fwd(q, kv, table, torch.tensor([0, 5, 75], dtype=torch.int32),
                               torch.tensor(lens, dtype=torch.int32), .125)
    torch.testing.assert_close(y, torch.einsum("thl,hvl->thv", out, wv), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kw,exc", [
    (dict(seg_tokens=500), ValueError), (dict(seg_tokens=0), ValueError), (dict(prompt_lens=[2]), TypeError),
    (dict(scale=float("nan")), ValueError), (dict(seq_lens=[5]), ValueError),
])
def test_op_refuses_bad_arguments(kw, exc):
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    T, H = 4, 2
    args = [torch.zeros(T, H, 8, dtype=torch.bfloat16), torch.zeros(T, H, 64, dtype=torch.bfloat16),
            torch.zeros(T, 512, dtype=torch.bfloat16), torch.zeros(T, 64, dtype=torch.bfloat16),
            torch.zeros(H, 8, 512, dtype=torch.bfloat16), torch.zeros(H, 16, 512, dtype=torch.bfloat16)]
    base = dict(seq_lens=[4], scale=.25, seg_tokens=512)
    base.update(kw)
    with pytest.raises(exc):
        mla_sm90_attention(*args, **base)


def test_op_refuses_non_bf16():
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    T, H = 4, 2
    args = [torch.zeros(T, H, 8), torch.zeros(T, H, 64), torch.zeros(T, 512), torch.zeros(T, 64),
            torch.zeros(H, 8, 512), torch.zeros(H, 16, 512)]
    with pytest.raises(ValueError, match="bf16"):
        mla_sm90_attention(*args, seq_lens=[4], scale=.25, seg_tokens=512)
