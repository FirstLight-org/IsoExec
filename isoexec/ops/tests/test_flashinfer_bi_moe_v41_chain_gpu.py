"""The vendored batch-invariant BF16 MoE's ``v41_actor`` GEMM1 chain (IsoExec's recipe ``deepseek_v41_bf16_actor_v1``) on an
SM100 GPU, bit for bit against IsoExec's own statements of the recipe:

* GEMM1 (the weighted chain, both records -- swapped and dense -- and both forms of the swapped record -- gather and box)
  against ``moe_sm100.ops.fc1`` (the BF16-rounded projection, one K chain) followed by
  ``deepseek_v41_sm100.rows.swiglu_rows`` (the recipe's torch statement with the route weight);
* GEMM2 (plain) against ``moe_sm100.ops.fc2(..., leaves=1)`` rounded to BF16;
* the combine (``combine_unfinalized(..., from_first=True)``, unit weights) against ``moe_sm100.ops.combine``;
* ``fused_moe`` with the weighted chain against the stage-by-stage pipeline;
* the unweighted chains: unchanged API behaviour (``row_weights`` refused), the same bits across records and forms, and
  (when the host's FlashInfer ships the batch-invariant MoE) the same bits as the host's copy.

E = 17 (16 routed experts + the shared one, always the last slot with weight 1), k = 7 slots.
"""
from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

E, K_SLOTS, LIMIT = 17, 7, 7.0
SHAPES = [(H, I) for H in (512, 5120) for I in (256, 288)]
TOKENS = (1, 7, 64, 300)


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _assert_bitwise(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    eq = _bits(a) == _bits(b)
    if not bool(eq.all()):
        idx = (~eq).nonzero()[:5].tolist()
        raise AssertionError(f"{what}: {int((~eq).sum())} of {eq.numel()} elements differ, first {idx}: "
                             f"{[a[tuple(i)].item() for i in idx]} vs {[b[tuple(i)].item() for i in idx]}")


def _bi():
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import api
    return api


def _problem(T, H, I, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1000 * T + H + I + seed)
    x = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    # projection magnitudes of a few units: the clamp at LIMIT binds on a fraction of the elements
    gate_up = (torch.randn(E, 2 * I, H, device=dev, generator=g) * (4.0 / H ** 0.5)).to(torch.bfloat16)
    down = (torch.randn(E, H, I, device=dev, generator=g) / I ** 0.5).to(torch.bfloat16)
    routed = torch.stack([torch.randperm(E - 1, device=dev, generator=g)[:K_SLOTS - 1] for _ in range(T)])
    ids = torch.cat([routed.sort(1).values, torch.full((T, 1), E - 1, device=dev)], 1).int().contiguous()
    scales = torch.rand(T, K_SLOTS, device=dev, generator=g) * 2.5
    scales[:, -1] = 1.0  # the shared expert's slot
    return x, gate_up, down, ids, scales.contiguous()


def _fc1_flashinfer(gate_up):
    """moe_sm100's ``gate_up [E, 2I, H]`` (gate rows, then up rows) -> FlashInfer's ``fc1`` (up rows, then gate rows), the
    layout ``interleave_fc1`` reads."""
    I = gate_up.shape[1] // 2
    return torch.cat([gate_up[:, I:], gate_up[:, :I]], 1).contiguous()


def _configs(T, H, I, swap):
    api = _bi()
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant.schedule import default_configs
    return default_configs(T, K_SLOTS, H, I, E, *api._part(torch.device("cuda")), swap=swap)


def _grouped_plan(counts, rows):
    from isoexec.ops.moe.moe_sm100 import ops as base
    offsets = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0, dtype=torch.int32)]).int().contiguous()
    return base.make_plan(offsets, rows)


def _gemm1(api, W, chain, x, counts, src, rw, cfg, gather):
    T, H = x.shape
    Tk = src.shape[0]
    if gather:
        return api.grouped_gemm_gather(x, src, K_SLOTS, W.w1i, counts, cfg, chain, Tk=Tk, row_weights=rw)
    pad = cfg.tile_n if cfg.swap_ab else 0
    x_buf = torch.zeros(Tk + pad, H, dtype=torch.bfloat16, device=x.device)
    x_buf[:Tk] = x[src.long() // K_SLOTS]
    return api.grouped_gemm(x_buf, W.w1i, counts, cfg, chain, Tk=Tk, row_weights=rw)


def _v41_chain():
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import spec as S
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant.blocks.epilogue import GroupedChain
    return GroupedChain("glu", "v41_actor", "bf16", S.params_key((LIMIT,)))


@pytest.mark.parametrize("H,I", SHAPES)
@pytest.mark.parametrize("T", TOKENS)
@torch.no_grad()
def test_v41_actor_pipeline_bitwise(T, H, I):
    from isoexec.ops.moe.deepseek_v41_sm100.rows import swiglu_rows
    from isoexec.ops.moe.moe_sm100 import ops as base

    api = _bi()
    x, gate_up, down, ids, scales = _problem(T, H, I)
    W = api.MoEWeights.prepare(_fc1_flashinfer(gate_up), down, act="v41_actor", params=(LIMIT,))
    chain = W.chain1
    assert chain == _v41_chain() and chain.weighted
    counts, dest, src = api.permute_plan(ids, E)
    Tk = T * K_SLOTS
    rw = api.row_weights(scales, src)
    _assert_bitwise(rw, scales.reshape(-1)[src.long()], "row_weights")
    gp = _grouped_plan(counts, Tk)
    x_rows = x[src.long() // K_SLOTS].contiguous()

    # GEMM1: the BF16 projection (one K chain) then the recipe's torch statement with the route weight
    proj = base.fc1(x_rows, gate_up, gp)  # [Tk, 2I] bf16, gate | up
    h_ref = swiglu_rows(proj, rw, LIMIT)
    (c1_sw, c2_sw), (c1_d, c2_d) = _configs(T, H, I, True), _configs(T, H, I, False)
    assert c1_sw.swap_ab and not c1_d.swap_ab
    runs = {}
    for name, cfg, gather in (("swapped-gather", c1_sw, True), ("swapped-box", c1_sw, False), ("dense", c1_d, False)):
        h = _gemm1(api, W, chain, x, counts, src, rw, cfg, gather)
        _assert_bitwise(h, h_ref, f"GEMM1 {name} T={T} H={H} I={I}")
        runs[name] = h
    with pytest.raises(ValueError):  # the weighted chain requires its operand
        _gemm1(api, W, chain, x, counts, src, None, c1_sw, False)

    # GEMM2 plain against fc2 with one leaf, rounded once
    h = runs["dense"]
    y_ref = base.fc2(h, down, gp, leaves=1).to(torch.bfloat16)
    for name, cfg in (("swapped", c2_sw), ("dense", c2_d)):
        pad = cfg.tile_n if cfg.swap_ab else 0
        h_buf = torch.zeros(Tk + pad, I, dtype=torch.bfloat16, device=x.device)
        h_buf[:Tk] = h
        y = api.grouped_gemm(h_buf, W.w2, counts, cfg, "plain", torch.bfloat16, Tk=Tk)
        _assert_bitwise(y, y_ref, f"GEMM2 {name} T={T} H={H} I={I}")

    # combine: unit weights, the add chain from +0 (moe_sm100's combine is the schedule policy's rule-2 chain seeded
    # with +0, which is the vendored class's own fold; the from-first variant is what the recipe ran before)
    plan = base.RoutePlan(gp, ids, src, dest, dest.contiguous())
    out_ref = base.combine(y_ref.float(), plan)
    out = api.combine_unfinalized(y_ref, torch.ones_like(scales), dest)
    _assert_bitwise(out, out_ref, f"combine T={T} H={H}")

    # fused_moe with the weighted chain: the same stages (its default records and form)
    fused = api.fused_moe(x, ids, scales, W)
    _assert_bitwise(fused, out_ref, f"fused_moe T={T} H={H} I={I}")


@torch.no_grad()
def test_from_first_combine_keeps_a_negative_zero_row():
    """The only bits the from-first fold changes: a token whose terms are all ``-0`` (the from-``+0`` fold gives ``+0``)."""
    api = _bi()
    T, H = 3, 512
    y = torch.full((T * K_SLOTS, H), -0.0, dtype=torch.bfloat16, device="cuda")
    y[K_SLOTS:2 * K_SLOTS, :8] = 1.5  # token 1: live non-zero terms
    dest = torch.arange(T * K_SLOTS, dtype=torch.int32, device="cuda").view(T, K_SLOTS)
    ones = torch.ones(T, K_SLOTS, device="cuda")
    ff = api.combine_unfinalized(y, ones, dest, from_first=True)
    z = api.combine_unfinalized(y, ones, dest)
    assert bool((_bits(ff[0]) == -32768).all()) and bool((_bits(z[0]) == 0).all())
    _assert_bitwise(ff[1:2, :8], z[1:2, :8], "live terms")
    # dropped slots are skipped: a token with slot 0 dropped starts from its first live term
    dest2 = dest.clone()
    dest2[2, 0] = -1
    ff2 = api.combine_unfinalized(y, ones, dest2, from_first=True)
    assert bool((_bits(ff2[2]) == -32768).all())
    dest2[2, :] = -1  # no live slot: +0
    ff3 = api.combine_unfinalized(y, ones, dest2, from_first=True)
    assert bool((_bits(ff3[2]) == 0).all())
    # the from-first combine reads no weights: None (no operand) and arbitrary weights give the same bits
    _assert_bitwise(api.combine_unfinalized(y, None, dest, from_first=True, tokens=T), ff, "no weights")
    _assert_bitwise(api.combine_unfinalized(y, torch.rand(T, K_SLOTS, device="cuda"), dest, from_first=True), ff,
                    "weights unread")


@pytest.mark.parametrize("act,params", [("silu", ()), ("swiglu_oai", (1.0, 0.0, 10.0))])
@pytest.mark.parametrize("T", (7, 300))
@torch.no_grad()
def test_unweighted_chains_unchanged(act, params, T):
    """The other chains take no row weights and keep their bits: the same rows across records and forms, the plain
    combine unchanged by the new option's default, and the host FlashInfer's copy (when it ships the class) bit for bit."""
    api = _bi()
    H, I = 512, 288
    x, gate_up, down, ids, scales = _problem(T, H, I, seed=1)
    fc1 = _fc1_flashinfer(gate_up)
    W = api.MoEWeights.prepare(fc1, down, act=act, params=params)
    assert not W.chain1.weighted
    counts, dest, src = api.permute_plan(ids, E)
    (c1_sw, _), (c1_d, _) = _configs(T, H, I, True), _configs(T, H, I, False)
    rows = [_gemm1(api, W, W.chain1, x, counts, src, None, c, g) for c, g in ((c1_sw, True), (c1_sw, False), (c1_d, False))]
    for r in rows[1:]:
        _assert_bitwise(r, rows[0], f"{act} GEMM1 records")
    with pytest.raises(ValueError):  # row weights belong to the weighted chain only
        _gemm1(api, W, W.chain1, x, counts, src, api.row_weights(scales, src), c1_sw, False)
    out = api.fused_moe(x, ids, scales, W)
    _, inter = api.fused_moe(x, ids, scales, W, return_intermediates=True)
    again = api.combine_unfinalized(inter["y"], scales, dest)
    _assert_bitwise(again, out, f"{act} combine default")
    try:
        from flashinfer.fused_moe.batch_invariant import api as host
    except Exception:  # noqa: BLE001 - the host copy is optional
        return
    hw = host.MoEWeights.prepare(fc1, down, act=act, params=params)
    _assert_bitwise(host.fused_moe(x, ids, scales, hw), out, f"{act} vs host FlashInfer")
