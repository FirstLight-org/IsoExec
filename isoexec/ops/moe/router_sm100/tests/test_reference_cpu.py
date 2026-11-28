"""Gate 1 (CPU): the declared programs select exactly what the semantic references select.

``reference.route_reference`` is the declared program; ``semantic_ordered`` is HF's
``DeepseekV3MoE.route_tokens_to_experts`` with each ``torch.topk`` given the declared tie rule, and
``deepgemm_mega_gate`` is DeepGEMM's ``test_mega_gate.py`` oracle for routing B. 10^5 rows per
routing, a quarter of them with planted ties (keys quantised onto a coarse grid so equal keys and
equal group scores are common). On tie-free rows the selection must also equal HF's literal
``torch.topk`` code as a set.
"""

from __future__ import annotations

import torch

from isoexec.ops.moe.router_sm100 import reference as R

ROWS = 100_000


def _logits(rows, E, seed, tie_frac=0.25):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, E, generator=g) * 2.0
    nt = int(rows * tie_frac)
    # planted ties: a coarse grid makes equal sigmoid scores (and equal top-2 group sums) common
    x[:nt] = torch.round(x[:nt] * 2.0) / 2.0
    # a slice of fully duplicated experts inside a row
    x[nt : nt + nt // 4, 1::2] = x[nt : nt + nt // 4, 0::2]
    return x


def _bias(E, seed, zero=False):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.zeros(E) if zero else torch.round(torch.randn(E, generator=g) * 8.0) / 64.0


def _ranked(logits, **kw):
    return R.route_reference(logits, return_rank_order=True, **kw)[0].long()


def _check_grouped(E, k, ng, tg, seed):
    x = _logits(ROWS, E, seed)
    for zero_bias in (False, True):
        b = _bias(E, seed, zero_bias)
        ours = _ranked(x, routing=R.GROUPED, top_k=k, n_group=ng, topk_group=tg, expert_bias=b,
                       routed_scale=2.5)
        sem = R.semantic_ordered(x, b, ng, tg, k, 2.5)
        diff = (ours != sem).any(1)
        assert int(diff.sum()) == 0, f"E={E} ng={ng} tg={tg}: {int(diff.sum())} rows differ"
        # tie-free rows: the literal HF code (torch.topk, sorted=False) selects the same SET
        nt = ROWS // 4 + ROWS // 16
        hf_idx, hf_w = R.hf_deepseek_v3(x[nt:], b, ng, tg, k, 2.5)
        assert torch.equal(torch.sort(hf_idx, 1).values, torch.sort(ours[nt:], 1).values)
        # weights: the declared half-fold vs HF's torch.sum, fp32 tolerance
        _, w = R.route_reference(x[nt:], routing=R.GROUPED, top_k=k, n_group=ng, topk_group=tg,
                                 expert_bias=b, routed_scale=2.5)
        hw = torch.gather(hf_w, 1, torch.argsort(hf_idx, 1))
        torch.testing.assert_close(w, hw, rtol=2e-6, atol=1e-7)


def test_gate1_grouped_deepseek_v3():
    _check_grouped(256, 8, 8, 4, seed=11)


def test_gate1_ungrouped_256():
    _check_grouped(256, 8, 1, 1, seed=12)


def test_gate1_glm47_flash():
    _check_grouped(64, 4, 1, 1, seed=13)


def test_gate1_sqrtsoftplus():
    for E, seed in ((256, 21), (384, 22), (512, 23)):
        x = _logits(ROWS // 2, E, seed)
        for bias in (_bias(E, seed), None):
            ours = _ranked(x, routing=R.SQRTSOFTPLUS, top_k=6, expert_bias=bias, routed_scale=2.5)
            dg_idx, dg_w = R.deepgemm_mega_gate(x, bias, 6, 2.5)
            assert int((ours != dg_idx).any(1).sum()) == 0
            _, w = R.route_reference(x, routing=R.SQRTSOFTPLUS, top_k=6, expert_bias=bias,
                                     routed_scale=2.5, return_rank_order=True)
            torch.testing.assert_close(w, dg_w, rtol=2e-6, atol=1e-7)


def test_bypass_vs_minus_inf_fill_is_the_only_divergence():
    """HF fills masked groups with -inf; the declared program bypasses them. They can differ only on
    rows whose surviving groups hold fewer than k finite keys -- planted here with NaN logits."""
    E, k, ng, tg = 256, 8, 8, 4
    x = torch.randn(64, E)
    x[:, :] = float("nan")
    x[:, :3] = 0.5                      # group 0 holds 3 finite keys; every other key is NaN -> -inf
    b = torch.zeros(E)
    ours = _ranked(x, routing=R.GROUPED, top_k=k, n_group=ng, topk_group=tg, expert_bias=b)
    # the declared program never selects outside the surviving groups ...
    G = R.group_scores(R._canon(R.score(x, R.GROUPED) + b), ng)
    keep = R.selected_groups(G, tg).repeat_interleave(E // ng, 1)
    assert bool(torch.gather(keep, 1, ours).all())
    # ... and takes the three finite keys first, then the lowest surviving ids
    assert torch.equal(ours[:, :3], torch.arange(3).expand(64, 3))


def test_reference_refusals_and_shapes():
    x = torch.randn(4, 256)
    idx, w = R.route_reference(x, routing=R.SQRTSOFTPLUS, top_k=6, num_shared=1,
                               mask=torch.tensor([True, False, True, True]))
    assert idx.shape == (4, 7) and w.shape == (4, 7)
    assert torch.equal(idx[1], torch.full((7,), -1, dtype=torch.int32))
    assert torch.equal(w[1], torch.zeros(7))
    assert int(idx[0, 6]) == 256 and float(w[0, 6]) == 1.0
    assert bool((idx[0, 1:6] > idx[0, :5]).all())          # ascending global expert id


def test_halffold_spelling():
    v = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    # ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)) for K = 8
    a = v[0]
    want = ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]))
    assert float(R.halffold(v)[0]) == float(want)
    v6 = torch.tensor([[1e8, 1.0, -1e8, 3.0, 0.5, 7.0]])
    a = v6[0]
    want6 = ((a[0] + a[3]) + (a[2] + a[5])) + (a[1] + a[4])
    assert float(R.halffold(v6)[0]) == float(want6)


def test_float64_twin_is_close():
    x = _logits(2000, 256, 5, tie_frac=0.0)
    b = _bias(256, 5)
    kw = dict(routing=R.GROUPED, top_k=8, n_group=8, topk_group=4, expert_bias=b, routed_scale=2.5)
    i32, w32 = R.route_reference(x, **kw)
    i64, w64 = R.route_reference(x.double(), **{**kw, "expert_bias": b.double()}, dtype=torch.float64)
    same = (i32 == i64).all(1)
    assert float(same.float().mean()) > 0.999
    torch.testing.assert_close(w32[same].double(), w64[same], rtol=1e-6, atol=1e-8)
