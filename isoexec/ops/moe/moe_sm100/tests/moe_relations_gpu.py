#!/usr/bin/env python
"""SM100 MoE determinism battery."""

from __future__ import annotations

import struct
import sys

import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.moe.moe_sm100.tests._program import (
    selected_sum_reference,
    biteq,
    canonical_plan,
    combine_reference,
    glu_epilogue_torch,
    segment_tree,
    tile_plan,
)

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail and not ok else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def adversarial_logits(T: int, E: int, device, seed: int, *, positive_infinity=False) -> torch.Tensor:
    """fp32 router rows with exact ties, NaN, -inf, subnormals and the fp32 maximum."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(T, E, generator=g, dtype=torch.float32)
    specials = [
        0.0,
        -0.0,
        float("nan"),
        float("inf") if positive_infinity else 3.0e38,
        float("-inf"),
        struct.unpack("<f", struct.pack("<I", 0x00000001))[0],
        -struct.unpack("<f", struct.pack("<I", 0x007FFFFF))[0],
        3.4028234663852886e38,
    ]
    for t in range(min(T, 8)):
        for i, s in enumerate(specials):
            if i < E:
                v[t, i] = s
    # Exact ties across expert ids, which the (score desc, id asc) rule must break.
    tie = v[:, 0].clone()
    for e in range(0, E, max(1, E // 4)):
        v[:, e] = tie
    return v.to(device)


def make_weights(E, H, I, device, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gu = (torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) / 16).to(device, dtype)
    dn = (torch.randn(E, H, I, generator=g, dtype=torch.float32) / 16).to(device, dtype)
    rw = (torch.randn(E, H, generator=g, dtype=torch.float32) / 8).to(device, dtype)
    return gu, dn, rw


def _sum_shape_diagnosis(selected: torch.Tensor, got: torch.Tensor, indices: torch.Tensor) -> str:
    """Which fp32 summation tree over the selected scores reproduces ``got``."""
    m = torch.amax(selected, dim=-1, keepdim=True)
    e = torch.exp(selected - m)
    k = e.shape[-1]
    variants = {"torch.sum": torch.sum(e, dim=-1, keepdim=True),
                "ascending-id chain from +0 (schedule policy rule 2)": selected_sum_reference(e, indices)}
    chain = e[..., :1].clone()
    for j in range(1, k):
        chain = chain + e[..., j : j + 1]
    variants["selection-order chain"] = chain
    fold = e.clone()
    width = k
    while width > 1:
        width //= 2
        fold = fold[..., :width] + fold[..., width : 2 * width]
    variants["power-of-two half fold"] = fold[..., :1]
    hits = [name for name, total in variants.items() if biteq(got, (e / total).contiguous())]
    return f"kernel matches: {hits or 'none of torch.sum / id chain / selection chain / half-fold'}"


# --------------------------------------------------------------------------- router ---


def router_relations(ops, device):
    from isoexec.ops.moe.routing import softmax_rows, stable_topk

    print("router: selection and probabilities against the dense fp32 reference")
    for E, k in ((8, 2), (32, 8), (64, 4), (256, 8), (256, 1)):
        logits = adversarial_logits(64, E, device, seed=E + k)
        idx, probs = ops.route_mechanics(logits, k)
        clean = torch.where(torch.isnan(logits), torch.full_like(logits, float("-inf")), logits)
        rv, ri = stable_topk(clean, k)
        # The declared program: amax, exp, the selected-set sum as the schedule policy's rule-2 chain
        # (ascending expert id, from +0), one IEEE divide. NOT ``softmax_rows`` (torch.sum's half fold in
        # selection order), which was this kernel's v1 program and is still sm90's.
        ev = torch.exp(rv - torch.amax(rv, dim=-1, keepdim=True))
        rp = (ev / selected_sum_reference(ev, ri)).contiguous()
        check(f"selection E={E} k={k}", torch.equal(idx, ri))
        ok = biteq(probs, rp)
        # The selected-set sum's shape is the one thing a transcription can get wrong without
        # any visible error, so when it disagrees say WHICH tree the kernel implemented. Naming
        # the disagreement is what turns a failure into a one-line fix.
        check(f"probabilities E={E} k={k} bitwise", ok, _sum_shape_diagnosis(rv, probs, ri))
        # accuracy against the fp64 softmax over the selected, old (half fold) and new (id chain) programs
        ref64 = torch.softmax(rv.double(), -1)
        fin = torch.isfinite(ref64).all(-1) & torch.isfinite(probs).all(-1)
        if bool(fin.any()):
            new_err = (probs.double() - ref64)[fin].abs().max().item()
            old_err = (softmax_rows(rv).double() - ref64)[fin].abs().max().item()
            print(f"         max abs error vs fp64, E={E} k={k}: id chain {new_err:.3e}, half fold {old_err:.3e}")

    print("router: selection under +inf and NaN (integer ids, not NaN payloads)")
    for E, k in ((64, 4), (256, 8)):
        logits = adversarial_logits(64, E, device, seed=E, positive_infinity=True)
        idx, _ = ops.route_mechanics(logits, k)
        clean = torch.where(torch.isnan(logits), torch.full_like(logits, float("-inf")), logits)
        _, ri = stable_topk(clean, k)
        check(f"selection with +inf/NaN E={E} k={k}", torch.equal(idx, ri))

    print("router: sigmoid / no-aux-bias selection against its reference")
    E, k, scale = 64, 4, 1.8
    logits = adversarial_logits(64, E, device, seed=7)
    bias = (torch.randn(E, generator=torch.Generator().manual_seed(3)) / 8).to(device)
    idx, probs = ops.route_mechanics(logits, k, expert_bias=bias, routed_scale=scale)
    scores = torch.sigmoid(logits)
    keyed = scores + bias
    keyed = torch.where(torch.isnan(keyed), torch.full_like(keyed, float("-inf")), keyed)
    _, ri = stable_topk(keyed, k)
    selected = torch.gather(scores, -1, ri.long())
    rp = (selected / (selected_sum_reference(selected, ri) + 1e-20)) * scale
    check("sigmoid/no-aux selection", torch.equal(idx, ri))
    check("sigmoid/no-aux probabilities bitwise", biteq(probs, rp.contiguous()))

    print("route_score: batch-composition invariance and fp64 accuracy")
    H, E = 2048, 64
    _, _, rw = make_weights(E, H, 64, device, torch.bfloat16, 11)
    big = (torch.randn(1024, H, generator=torch.Generator().manual_seed(5)) / 4).to(
        device, torch.bfloat16
    )
    full = ops.route_score(big, rw)
    solo = ops.route_score(big[17:18].contiguous(), rw)
    check("logits row independent of batch composition", biteq(full[17:18], solo))
    shuffled = ops.route_score(big.flip(0).contiguous(), rw)
    check("logits row independent of batch position", biteq(full[17:18], shuffled[1024 - 18 : 1024 - 17]))
    ref = big.double() @ rw.double().t()
    # Against the ROW SCALE, not element by element: a logit that lands near zero by cancellation
    # has a huge *elementwise* relative error at an ordinary absolute error, so an elementwise
    # ratio measures the cancellation and not the chain. The bitwise claims above are the
    # determinism ones; this is the accuracy sanity check.
    scale = ref.abs().max().item()
    absolute = (full.double() - ref).abs().max().item()
    check(
        "logits within 1e-5 of fp64 relative to the logit scale",
        absolute / max(scale, 1e-12) < 1e-5,
        f"abs={absolute:.3e} scale={scale:.3f} ratio={absolute / max(scale, 1e-12):.3e}",
    )

# ------------------------------------------------------------------------- dispatch ---


def dispatch_relations(ops, device):
    print("dispatch: exact integer ranks, canonical by token id")
    # Seven shapes span both fused single-launch bands and the multi-block path. The fused plan
    # takes every route count up to ``kSmallRoutes`` = 2048: a 256-thread CTA below
    # ``kSmallRoutesOneWarpSet`` = 1024 routes, and a 1024-thread CTA (32 sub-chunk warps) from
    # there to 2048. (1, 256, 8) = 8, (32, 256, 8) = 256 and (129, 64, 4) = 516 routes gate the
    # 256-thread band; (400, 64, 4) = 1600 routes is the only shape that lands in the 1024-thread
    # band -- no earlier shape did. Above 2048 the multi-block path runs: (2048, 64, 4) = 8192,
    # (3000, 8, 2) = 6000 and (4096, 256, 8) = 32768 routes, over 8, 6 and 32 blocks.
    for T, E, k in ((1, 256, 8), (32, 256, 8), (129, 64, 4), (400, 64, 4), (2048, 64, 4),
                    (3000, 8, 2), (4096, 256, 8)):
        logits = adversarial_logits(T, E, device, seed=T * 31 + E)
        idx, probs = ops.route_mechanics(logits, k)
        plan = ops.route(idx, E)
        counts, offsets, ref_inverse, restore, combine_rows = canonical_plan(idx, E)
        check(f"offsets T={T} E={E} k={k}", torch.equal(plan.grouped.offsets, offsets))
        check(f"restore (exact rank) T={T} E={E} k={k}", torch.equal(plan.restore, restore))
        check(f"inverse T={T} E={E} k={k}", torch.equal(plan.inverse, ref_inverse))
        check(f"canonical combine rows T={T} E={E} k={k}", torch.equal(plan.combine_rows, combine_rows))
        expect = tile_plan(offsets.cpu(), ops.BLOCK_M)
        total = int(plan.grouped.total)
        got = [tuple(r) for r in plan.grouped.tiles[:total].tolist()]
        check(f"tile plan T={T} E={E} k={k}", got == expect, f"{len(got)} vs {len(expect)}")
        check(
            f"empty experts bypassed T={T} E={E}",
            all(e != expert for expert, _, _ in got for e in (counts == 0).nonzero().flatten().tolist()),
        )

    print("dispatch: gathered rows are the token's bytes, unchanged")
    T, E, k, H = 256, 64, 4, 2048
    x = (torch.randn(T, H, generator=torch.Generator().manual_seed(9)) / 4).to(device, torch.bfloat16)
    logits = adversarial_logits(T, E, device, seed=99)
    idx, probs = ops.route_mechanics(logits, k)
    plan = ops.route(idx, E)
    gx, gp = ops.dispatch(x, probs, plan)
    src = x.index_select(0, (plan.inverse.long() // k))
    check("gathered activations", biteq(gx, src))
    check("gathered probabilities", biteq(gp, probs.reshape(-1).index_select(0, plan.inverse.long())))


# -------------------------------------------------------------------------- experts ---


def expert_relations(ops, device, H, E, I, k, dtype, leaves):
    tag = f"H{H}/E{E}/I{I}/k{k}/L{leaves}"
    gu, dn, rw = make_weights(E, H, I, device, dtype, seed=H + E + I)
    T = 512
    x = (torch.randn(T, H, generator=torch.Generator().manual_seed(21)) / 4).to(device, dtype)
    logits = ops.route_score(x, rw)
    idx, probs = ops.route_mechanics(logits, k)
    plan = ops.route(idx, E)
    gx, gp = ops.dispatch(x, probs, plan)

    # This is the relation that makes moe.expert_mlp.segment_tree_glu a class with two members:
    # sm100_tcgen05_v1 (the GLU epilogue fused into the grouped GEMM) and
    # sm100_tcgen05_unfused_act_v1 (a raw projection plus the standalone activation kernel).
    print(f"FC1 epilogue structure ({tag})")
    raw = ops.fc1(gx, gu, plan.grouped)                 # one K chain, rounded once
    h = ops.fc1_glu(gx, gu, plan.grouped, gp)
    check(f"fc1_glu == activation(fc1) {tag}", biteq(h, ops.activation(raw, gp)))
    h_noprob = ops.fc1_glu(gx, gu, plan.grouped, None)
    check(f"probability product is the only difference {tag}", biteq(h_noprob, ops.activation(raw, None)))
    # The declared SiLU divide is __fdividef, so the plain-torch transcription of the schedule
    # bounds rather than defines the epilogue: state how far apart they are.
    torch_form = glu_epilogue_torch(raw[:, :I].contiguous(), raw[:, I:].contiguous(), gp)
    disagree = int((h.view(torch.int16) != torch_form.view(torch.int16)).sum())
    check(
        f"the torch transcription agrees on at least 99.9% of elements {tag}",
        disagree <= h.numel() // 1000,
        f"{disagree}/{h.numel()} differ",
    )

    print(f"FC2 segment tree structure ({tag})")
    rows = ops.fc2(h, dn, plan.grouped, leaves=leaves)
    seg = I // leaves
    leaf_roots = []
    for j in range(leaves):
        hj = h[:, j * seg : (j + 1) * seg]
        dj = dn[:, :, j * seg : (j + 1) * seg]
        leaf_roots.append(ops.fc1(hj, dj, plan.grouped).float())
    check(f"fc2 == balanced tree over rounded segments {tag}", biteq(rows, segment_tree(leaf_roots)))

    print(f"ETP: cuts only at segment nodes ({tag})")
    if leaves >= 2:
        half = I // 2
        left = ops.fc2(h[:, :half].contiguous(), dn[:, :, :half].contiguous(), plan.grouped, leaves=leaves // 2)
        right = ops.fc2(h[:, half:].contiguous(), dn[:, :, half:].contiguous(), plan.grouped, leaves=leaves // 2)
        check(f"per-rank subtree fold equals single rank {tag}", biteq(rows, left + right))

    print(f"batch composition and row placement ({tag})")
    keep = 7
    token = x[keep : keep + 1].contiguous()
    solo_logits = ops.route_score(token, rw)
    solo_idx, solo_probs = ops.route_mechanics(solo_logits, k)
    check(f"solo routing matches {tag}", torch.equal(solo_idx, idx[keep : keep + 1]))
    solo_plan = ops.route(solo_idx, E)
    sx, sp = ops.dispatch(token, solo_probs, solo_plan)
    srows = ops.fc2(ops.fc1_glu(sx, gu, solo_plan.grouped, sp), dn, solo_plan.grouped, leaves=leaves)
    for j in range(k):
        mine = rows[int(plan.combine_rows[keep, j])]
        theirs = srows[int(solo_plan.combine_rows[0, j])]
        check(f"expert output slot {j} independent of the rest of the batch {tag}", biteq(mine, theirs))

    perm = torch.randperm(T, generator=torch.Generator().manual_seed(4)).to(device)
    pidx = idx.index_select(0, perm)
    pprobs = probs.index_select(0, perm)
    px = x.index_select(0, perm)
    pplan = ops.route(pidx, E)
    pgx, pgp = ops.dispatch(px, pprobs, pplan)
    prows = ops.fc2(ops.fc1_glu(pgx, gu, pplan.grouped, pgp), dn, pplan.grouped, leaves=leaves)
    where = int((perm == keep).nonzero()[0])
    ok = all(
        biteq(rows[int(plan.combine_rows[keep, j])], prows[int(pplan.combine_rows[where, j])])
        for j in range(k)
    )
    check(f"expert output independent of within-expert row placement {tag}", ok)

    print(f"combine ({tag})")
    y = ops.combine(rows, plan, dtype=dtype)
    check(f"combine == fp32 chain in canonical slot order {tag}", biteq(y, combine_reference(rows, plan.combine_rows, dtype)))
    pooled = ops.combine(prows, pplan, dtype=dtype)
    check(f"combined token independent of batch order {tag}", biteq(y[keep : keep + 1], pooled[where : where + 1]))

    print(f"EP: ship to the home rank and fold once ({tag})")
    for ranks in (2, 4):
        partial = torch.zeros_like(rows)
        share = (E + ranks - 1) // ranks
        for r in range(ranks):
            lo, hi = r * share, min(E, (r + 1) * share)
            if lo >= hi:
                continue
            mask = (idx >= lo) & (idx < hi)
            if not bool(mask.any()):
                continue
            local = torch.where(mask, idx, torch.full_like(idx, lo))
            # The rank owns only its experts' routes; every other route is parked on the
            # rank's first expert and discarded, so the rows it computes are exactly the
            # rows the single-rank run computed for those experts.
            sub = ops.route(local.contiguous(), E)
            sgx, sgp = ops.dispatch(x, probs, sub)
            srows = ops.fc2(ops.fc1_glu(sgx, gu, sub.grouped, sgp), dn, sub.grouped, leaves=leaves)
            flat_mask = mask.reshape(-1)
            dest = plan.restore.reshape(-1)[flat_mask].long()
            src = sub.restore.reshape(-1)[flat_mask].long()
            partial[dest] = srows[src]
        check(f"EP{ranks} per-rank rows equal single-rank rows {tag}", biteq(partial, rows))
        check(f"EP{ranks} home-rank fold equals single-rank combine {tag}", biteq(ops.combine(partial, plan, dtype=dtype), y))

    print(f"execution plan, repeats and graph replay ({tag})")
    again = ops.fc2(ops.fc1_glu(gx, gu, plan.grouped, gp), dn, plan.grouped, leaves=leaves)
    check(f"repeat launch {tag}", biteq(rows, again))

    graph_rows = torch.empty_like(rows)
    graph_h = torch.empty_like(h)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fc1_glu(gx, gu, plan.grouped, gp, out=graph_h)
        ops.fc2(graph_h, dn, plan.grouped, leaves=leaves, out=graph_rows)
    graph.replay()
    torch.cuda.synchronize()
    check(f"graph capture equals eager {tag}", biteq(rows, graph_rows))
    graph.replay()
    torch.cuda.synchronize()
    check(f"graph replay is stable {tag}", biteq(rows, graph_rows))

    return x, gu, dn, rw, idx, probs, plan, h, rows, y


def shared_expert_relations(ops, device):
    """The always-on expert is a dense MLP, so its whole claim is row/batch invariance -- the same claim ``ops/moe/tests/shared_expert_biteq_gpu.py`` makes for sm90's ``eager``."""
    print("shared expert: row and batch invariance of the dense projections")
    H, I = 2048, 1536
    g = torch.Generator(device="cpu").manual_seed(77)
    gate = (torch.randn(I, H, generator=g, dtype=torch.float32) / 16).to(device, torch.bfloat16)
    up = (torch.randn(I, H, generator=g, dtype=torch.float32) / 16).to(device, torch.bfloat16)
    down = (torch.randn(H, I, generator=g, dtype=torch.float32) / 16).to(device, torch.bfloat16)
    egate = (torch.randn(1, H, generator=g, dtype=torch.float32) / 8).to(device, torch.bfloat16)
    x = (torch.randn(8192, H, generator=g, dtype=torch.float32) / 4).to(device, torch.bfloat16)
    full = ops.shared_expert(x, gate, up, down, egate)
    for row, label in ((0, "first"), (5000, "position 5000")):
        solo = ops.shared_expert(x[row : row + 1].contiguous(), gate, up, down, egate)
        check(f"shared expert row {label} alone equals the same row in 8192", biteq(full[row : row + 1], solo))
    small = ops.shared_expert(x[:8].contiguous(), gate, up, down, egate)
    check("shared expert rows 0..7 in a batch of 8 equal the batch of 8192", biteq(full[:8], small))
    ref = torch.nn.functional.silu(x.double() @ gate.double().t()) * (x.double() @ up.double().t())
    ref = ref @ down.double().t()
    ref = ref * torch.sigmoid(x.double() @ egate.double().t())
    # Against the OUTPUT SCALE, not element by element: 16.7 M outputs guarantee some land near
    # zero by cancellation, where an ordinary absolute error is an enormous elementwise ratio. The
    # bitwise row-invariance checks above are the claim; this bounds the arithmetic.
    scale = ref.abs().max().item()
    absolute = (full.double() - ref).abs().max().item()
    check(
        "shared expert within 1e-2 of fp64 relative to the output scale (bf16 operands)",
        absolute / max(scale, 1e-12) < 1e-2,
        f"abs={absolute:.3e} scale={scale:.3f} ratio={absolute / max(scale, 1e-12):.3e}",
    )


def ftz_relations(ops, device):
    """Subnormals survive. And the checks are NOT vacuous.

    ``moe_sm90/tests/_ftz_check.py`` states the rule this phase obeys: a check that would also
    pass with the protection deleted proves nothing, so every case here reports how many of the
    REFERENCE values are subnormal and fails when that count is zero. The three sites where
    ``--ftz=false`` and an IEEE divide are load-bearing in this package are the router's
    ``e / sum``, the FC1 epilogue's SiLU, and the fp32 adds in the combine and in the FC2 tree.
    """
    from isoexec.ops.moe.moe_sm90.tests._ftz_check import SMALLEST_NORMAL_F32, assert_no_ftz
    from isoexec.ops.moe.routing import softmax_rows, stable_topk

    print("FTZ: the router's divide on a massive-activation row")
    fails: list = []
    for E, k in ((64, 4), (256, 8)):
        g = torch.Generator(device="cpu").manual_seed(E)
        logits = torch.randn(256, E, generator=g, dtype=torch.float32)
        # One logit ~100 above the rest, which is what a real residual stream produces: every
        # other exp(x - max) then underflows into the subnormal range. This is the shape that
        # caught O2's libdevice.div_rn bug on Hopper.
        logits[:, 0] += 100.0
        logits = logits.to(device)
        idx, probs = ops.route_mechanics(logits, k)
        vals, ref_idx = stable_topk(logits, k)
        ref = softmax_rows(vals)
        check(f"massive-activation selection E={E} k={k}", torch.equal(idx, ref_idx))
        info = assert_no_ftz(f"router probs E={E} k={k}", ref, probs, fail_list=fails)
        check(
            f"router probs keep subnormals E={E} k={k} ({info['subnormal']} subnormal references)",
            not info["vacuous"] and info["flushed"] == 0,
            f"{info}",
        )
        check(f"router probs bitwise on a massive-activation row E={E} k={k}", biteq(probs, ref))

    print("FTZ: the combine's fp32 chain over subnormal rows")
    T, k, H = 64, 4, 512
    g = torch.Generator(device="cpu").manual_seed(5)
    # Scaled so the sum of k of them is still subnormal: otherwise the reference is normal
    # and the case would report a large subnormal count while testing nothing.
    rows = (torch.rand(T * k, H, generator=g, dtype=torch.float32) * (SMALLEST_NORMAL_F32 / 8)).to(device)
    idx = (torch.arange(T * k, dtype=torch.int32) % (T * k)).reshape(T, k).to(device)
    plan = ops.route(idx.contiguous(), T * k)
    got = ops.combine(rows, plan, dtype=torch.float32)
    want = combine_reference(rows, plan.combine_rows, torch.float32)
    info = assert_no_ftz("combine over subnormals", want, got, fail_list=fails)
    check(
        f"combine keeps subnormals ({info['subnormal']} subnormal references)",
        not info["vacuous"] and info["flushed"] == 0,
        f"{info}",
    )
    check("combine bitwise over subnormal rows", biteq(got, want))

    # The FC1 epilogue is NOT checked for FTZ. Its declared SiLU divide is ``__fdividef``
    # (PTX ``div.approx.f32``), an approximate divide whose own subnormal handling is part of the
    # operator this recipe names -- the SM90 contract names it too. A flush there would be the
    # declared operator's behaviour, not the compilation profile's, so a check would be testing
    # the wrong thing and would fail for the right reason. What IS checked, above, is every site
    # where the profile is what keeps the subnormal: the router's IEEE divide and the fp32 adds.
    if fails:
        print(f"  ({len(fails)} FTZ detector notes: {fails})")










def padding_relations(ops, device):
    """Padding is explicit: nothing outside the routed rows is ever written."""
    print("padding: rows outside the plan are untouched")
    H, E, I, k, leaves = 2048, 64, 512, 4, 8
    gu, dn, rw = make_weights(E, H, I, device, torch.bfloat16, 44)
    x = (torch.randn(5, H, generator=torch.Generator().manual_seed(6)) / 4).to(device, torch.bfloat16)
    idx, probs = ops.route_mechanics(ops.route_score(x, rw), k)
    plan = ops.route(idx, E)
    gx, gp = ops.dispatch(x, probs, plan)
    R = plan.grouped.rows
    sentinel = float("-1.5e30")
    rows = torch.full((R + 64, H), sentinel, dtype=torch.float32, device=device)
    h = torch.full((R + 64, I), -7.0, dtype=torch.bfloat16, device=device)
    ops.fc1_glu(gx, gu, plan.grouped, gp, out=h[:R])
    ops.fc2(h[:R].contiguous(), dn, plan.grouped, leaves=leaves, out=rows[:R])
    check("no write past the routed rows", bool((rows[R:] == sentinel).all()))


def fp64_accuracy(ops, device, H, E, I, k, leaves):
    print("accuracy: the whole routed forward against an fp64 reference")
    gu, down, rw = make_weights(E, H, I, device, torch.bfloat16, 55)
    T = 24
    x = (torch.randn(T, H, generator=torch.Generator().manual_seed(8)) / 4).to(device, torch.bfloat16)
    idx, probs = ops.route_mechanics(ops.route_score(x, rw), k)
    y = ops.moe(x, gu, down, idx, probs, leaves=leaves)
    ref = torch.zeros(T, H, dtype=torch.float64, device=device)
    xs = x.double()
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            g = xs[t] @ gu[e].double().t()
            gate, up = g[:I], g[I:]
            hh = (gate / (1 + torch.exp(-gate))) * up * probs[t, j].double()
            ref[t] += hh @ down[e].double().t()
    scale = ref.abs().max().item()
    err = (y.double() - ref).abs().max().item() / max(scale, 1e-12)
    check(f"H{H}/E{E}/I{I}/k{k} within 4e-2 of fp64 (bf16 operands)", err < 4e-2, f"rel={err:.3e}")


#: Phase name -> the checks it runs. The whole battery is the claim; the split exists because
#: the shared machine hands out short clean windows and a phase that passed is evidence that
#: does not have to be re-earned.
PHASES = ("router", "dispatch", "experts", "shared", "ftz", "padding", "accuracy")


def main(argv=None) -> int:
    wanted = [a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("-")]
    phases = wanted or list(PHASES)
    bad = [p for p in phases if p not in PHASES]
    if bad:
        print(f"unknown phase(s) {bad}; known: {list(PHASES)}")
        return 1
    if not torch.cuda.is_available() or live_device_arch() != "sm100":
        print("INCOMPLETE: no SM100 device visible; this claim cannot be made here")
        return 2
    from isoexec.core.registry import TWIN_PROOFS, runnable_batteries
    from isoexec.ops.moe.moe_sm100 import ops

    # This battery is the named proof for these classes on this arch. Running it where the
    # registry does not cite it would produce evidence nothing reads, so fail loudly.
    mine = "isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py"
    for cls in ("moe.combine.korder_fp32", "moe.expert_mlp.segment_tree_glu"):
        if cls not in TWIN_PROOFS:
            print(f"FAIL: twin class {cls!r} is not declared in core/registry.TWIN_PROOFS")
            return 1
        if mine not in runnable_batteries(cls):
            print(f"FAIL: TWIN_PROOFS[{cls!r}] does not name this battery")
            return 1

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    print(f"phases: {phases}")
    with torch.no_grad():
        if "router" in phases:
            router_relations(ops, device)
        if "dispatch" in phases:
            dispatch_relations(ops, device)
        if "experts" in phases:
            # Qwen3.5-35B-A3B and GLM-4.7-Flash MoE geometry, plus a small adversarial one.
            for H, E, I, k, leaves in ((2048, 256, 512, 8, 8), (2048, 64, 1536, 4, 8), (512, 8, 128, 2, 2)):
                expert_relations(ops, device, H, E, I, k, torch.bfloat16, leaves)
            expert_relations(ops, device, 512, 8, 128, 2, torch.float16, 2)
        if "shared" in phases:
            shared_expert_relations(ops, device)
        if "ftz" in phases:
            ftz_relations(ops, device)
        if "padding" in phases:
            padding_relations(ops, device)
        if "accuracy" in phases:
            for H, E, I, k, leaves in ((2048, 256, 512, 8, 8), (2048, 64, 1536, 4, 8)):
                fp64_accuracy(ops, device, H, E, I, k, leaves)

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
