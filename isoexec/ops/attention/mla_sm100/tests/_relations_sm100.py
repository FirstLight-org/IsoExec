"""The relations the sm90 file states but cannot drive from here, re-stated against mla_sm100."""

from __future__ import annotations

import dataclasses

import torch

from isoexec.ops.attention.mla_sm90.tests._harness import (
    LATENT, PAGE, SCALE, WIDTH, Diff, Slice, assemble, decode, make_request,
)
from isoexec.ops.attention.mla_sm90.tests._harness import Unsupported
from isoexec.ops.attention.mla_sm90.tests._relations import Config, Rec, rows_of

__all__ = ["rel_cuda_graph_sm100", "rel_combine_tree_widths", "rel_saturated_decode_cut",
           "RELATIONS_SM100"]


def rel_cuda_graph_sm100(rec: Rec, cfg: Config):
    """Full decode CUDA graph (vLLM/SGLang): a capacity plan and out buffers captured once, replayed
    with new lengths, pages and a null lane, bit-identical to eager calls."""
    from isoexec.ops.attention.mla_sm100.ops import kernel_fwd, kernel_plan

    h, B, k = cfg.heads, 8, 2
    max_k = 4 * cfg.seg + 100
    mb = (max_k + PAGE - 1) // PAGE
    nb = B * mb + 1
    dev = "cuda"
    q = torch.zeros(B * k, h, WIDTH, device=dev, dtype=torch.bfloat16)
    kv = torch.zeros(nb, PAGE, 1, WIDTH, device=dev, dtype=torch.bfloat16)
    table = torch.zeros(B, mb, device=dev, dtype=torch.int32)
    cu = torch.arange(0, (B + 1) * k, k, device=dev, dtype=torch.int32)
    used = torch.zeros(B, device=dev, dtype=torch.int32)
    o = torch.empty(B * k, h, LATENT, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(B * k, h, device=dev, dtype=torch.float32)
    cap = {"max_seqlen_k": max_k, "num_tokens": B * k}

    def call():
        plan = kernel_plan(cu, used, cfg.seg, h, k, capacity=cap)
        # the backend's recipe: a graph of one recipe must replay that recipe's bits (round 5 found
        # this relation capturing v1 while its eager side ran v2)
        return kernel_fwd(q, kv, table, cu, used, SCALE, cfg.seg, k, plan=plan, out=(o, lse),
                          recipe=getattr(rec.backend, "recipe", 1))

    def load(step):
        g = torch.Generator().manual_seed(cfg.seed + 1100 + step)
        lens = torch.randint(k, max_k + 1, (B,), generator=g)
        lens[step % B] = k if step else max_k  # boundary: shortest and longest
        null = (step + 3) % B
        q.copy_(torch.randn(q.shape, generator=g).to(dev, torch.bfloat16))
        kv.copy_(torch.randn(kv.shape, generator=g).to(dev, torch.bfloat16))
        perm = torch.randperm(nb, generator=g)[: B * mb].view(B, mb).to(torch.int32)
        table.copy_(perm)
        used.copy_(lens.to(torch.int32))
        used[null] = 0 if step % 2 else k  # a padded lane with no KV on odd steps

    load(0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for step in range(4):
        load(step)
        graph.replay()
        torch.cuda.synchronize()
        got = (o.clone(), lse.clone())
        eager = rec.backend.fwd(dataclasses.replace(
            assemble([]), q=q, kv_cache=kv, block_table=table, cu_seqlens_q=cu, seqused_k=used,
            max_seqlen_q=k, row_ranges=[], labels=[], slices=[]))
        null = (step + 3) % B
        keep = torch.ones(B * k, dtype=torch.bool, device=dev)
        if step % 2:
            keep[null * k:(null + 1) * k] = False  # null-lane rows: compare finiteness, not bits
            fin = bool(torch.isfinite(got[0][~keep]).all())
            rec.o.diffs.append(Diff(f"replay{step}:null_lane_finite", fin, (), 0 if fin else -1, ""))
        rec.cmp_rows(f"replay{step}:graph vs eager", (got[0][keep], got[1][keep]),
                     (eager[0][keep], eager[1][keep]))


def rel_combine_tree_widths(rec: Rec, cfg: Config):
    """Every tree width n in 1..20, each against the leaf's OWN fold of the same tree and against every combine worker shape."""
    seg = cfg.seg
    widths = list(range(1, 21))
    rec.witness("tree_widths", widths)
    rec.require("widths_cover_3_to_16", set(range(3, 17)) <= set(widths), "n in 3..16 stated")
    # length (n-1)*seg + 65 gives exactly n segments, with the last one partial.
    reqs = [make_request((n - 1) * seg + 65, cfg.heads, cfg.seed + 300 + n, tag=f"n{n}")
            for n in widths]
    variants = [("whole", {"granularity": "whole"}), ("single", {"granularity": "single"}),
                ("pairs", {"granularity": "pairs"}), ("mixed", {"granularity": "mixed"}),
                ("cc0", {"combine_config": 0}), ("cc1", {"combine_config": 1}),
                ("cc2", {"combine_config": 2}), ("cc3", {"combine_config": 3}),
                ("one_group", {"combine_single_group": True}),
                ("w1", {"num_parts": 1}), ("w1000", {"num_parts": 1000})]
    for n, r in zip(widths, reqs):
        b = assemble([decode(r, r.length - 1)], page_perm_seed=n)
        try:
            base = rec.fwd(b)
        except Unsupported as e:
            rec.skip(f"n={n}/base", str(e))
            continue
        base = (base[0].clone(), None if base[1] is None else base[1].clone())
        for tag, ov in variants:
            try:
                got = rec.fwd(b, exec_override=ov)
            except Unsupported as e:
                rec.skip(f"n={n}/{tag}", str(e))
                continue
            rec.cmp_rows(f"n={n}/{tag}", got, base, b.labels)
    # and the same widths inside ONE batch, so a row's tree cannot depend on its companions
    bb = assemble([decode(r, r.length - 1) for r in reqs], page_perm_seed=77)
    try:
        batched = rec.fwd(bb)
    except Unsupported as e:
        rec.skip("batched", str(e))
        return
    for i, (n, r) in enumerate(zip(widths, reqs)):
        alone = assemble([decode(r, r.length - 1)], page_perm_seed=n)
        rec.cmp_rows(f"n={n}/batched", rows_of(bb, i, batched), rec.fwd(alone), bb.labels_of(i))


def _decode_request(length: int, heads: int, seed: int, tag: str):
    """A request that is decoded at one position: its whole KV and ONE query row, expanded over the positions so
    the harness's position index still reads it. ``make_request`` draws a query row for every position, which for
    512 requests of ``8 S + 65`` positions is 512 x (8 S + 65) x heads x 1152 bytes: 121 GB at S = 5120 and
    242 GB at S = 10240 (the span with no target batch declared) with 5 heads, more than a B200 holds beside the
    cache. The decode step reads one of those rows per request."""
    from isoexec.ops.attention.mla_sm90.tests._harness import DTYPE, Request

    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(length, WIDTH, generator=g).to("cuda", DTYPE)
    q = torch.randn(1, heads, WIDTH, generator=g).to("cuda", DTYPE).expand(length, heads, WIDTH)
    return Request(kv, q, tag)


def rel_saturated_decode_cut(rec: Rec, cfg: Config):
    """Check the decode split threshold."""
    from isoexec.ops.attention.mla_sm100 import fwd as F

    seg = cfg.seg
    B = 512
    for tag, length in (("n4", 4 * seg), ("n9_partial", 8 * seg + 65)):
        reqs = [_decode_request(length, cfg.heads, cfg.seed + 700 + i, tag=f"{tag}_{i}") for i in range(B)]
        b = assemble([decode(r, r.length - 1) for r in reqs], page_perm_seed=91)
        used = torch.full((B,), length, dtype=torch.int32)
        W = F._num_sms()
        fired = F._saturated_decode_run_target(used, seg, B, W, F.ITEM_OVERHEAD_BLOCKS,
                                               F.FOLD_OVERHEAD_BLOCKS)
        rec.witness(f"{tag}_run_target", int(fired))
        if tag == "n4":
            # the plan's cost model cuts when a segment outweighs an item's fixed overhead: stated for segments of
            # at least 8 KV blocks (512 keys); a shorter span (the policy's at a target batch of 1) need not cut
            rec.require("rule_fires_at_B512_n4", fired > 0 or W != 148 or seg < 512,
                        "the cut rule must fire at B = 512, n = 4 on a 148-SM part")
        try:
            base = rec.fwd(b)
        except Unsupported as e:
            rec.skip(f"{tag}/auto", str(e))
            continue
        base = (base[0].clone(), None if base[1] is None else base[1].clone())
        for vt, ov in (("whole", {"granularity": "whole"}), ("single", {"granularity": "single"}),
                       ("w2", {"granularity": "whole", "num_parts": 2})):
            try:
                got = rec.fwd(b, exec_override=ov)
            except Unsupported as e:
                rec.skip(f"{tag}/{vt}", str(e))
                continue
            rec.cmp_rows(f"{tag}/{vt}", got, base, b.labels)
        del b, reqs
        torch.cuda.empty_cache()


def rel_lazy_rescale_branches(rec: Rec, cfg: Config):
    """Round 5: the recipe-v2 lazy rescale branch is EXERCISED, and its boundary is where it says."""
    import math

    from isoexec.ops.attention.mla_sm100 import fwd as F
    from isoexec.ops.attention.mla_sm90.tests._harness import Request, assemble, full

    seg = cfg.seg
    h = cfg.heads
    scale_log2 = SCALE * math.log2(math.e)
    recipe = getattr(rec.backend, "recipe", 1)
    rec.witness("backend_recipe", recipe)
    length = 4 * seg                                   # four segments of seg/64 tiles each
    g = torch.Generator().manual_seed(cfg.seed + 909)

    def req(kind, D):
        tiles = torch.arange(length) // PAGE
        if kind == "rising":
            base = tiles.double() * (D / scale_log2)
        else:                                          # falling: the first tile of a segment is the max
            base = (length // PAGE - tiles).double() * (12.0 / scale_log2)
        kv = (torch.randn(length, WIDTH, generator=g) * 0.5)
        kv[:, 0] = (base + torch.rand(length, generator=g).double() * (0.5 / scale_log2)).float()
        q = torch.zeros(length, h, WIDTH)
        q[:, :, 0] = 1.0
        return Request(kv.to("cuda", torch.bfloat16), q.to("cuda", torch.bfloat16), tag=f"{kind}{D}")

    cases = [("rising", 12, True), ("falling", 0, True), ("rising", 4, False)]
    for kind, D, must_equal in cases:
        r = req(kind, D)
        for shape, sl in (("decode", [Slice(r, length - 1, length)]), ("prefill", [full(r)])):
            b = assemble(sl, page_perm_seed=D + 3)
            got = rec.fwd(b)
            v1 = F.mla_sm100_fwd(b.q, b.kv_cache, b.block_table, b.cu_seqlens_q, b.seqused_k, SCALE,
                                 seg, b.max_seqlen_q, recipe=1)
            name = f"{kind}{D}/{shape}"
            if must_equal:
                rec.cmp_rows(f"{name}/v{recipe}==v1", (got[0], got[1]), (v1[0], v1[1]), b.labels)
            else:
                differ = not (torch.equal(got[0], v1[0]) and torch.equal(got[1], v1[1]))
                rec.witness(f"{name}/differs_from_v1", differ)
                if recipe == 2:
                    rec.require(f"{name}/lazy_branch_ran", differ,
                                "a per-tile max advance of 4 log2 units must be KEPT by v2")


#: Relations this package owns, keyed the way ``run_relation`` keys the sm90 ones.
RELATIONS_SM100 = {"cuda_graph_sm100": rel_cuda_graph_sm100,
                   "combine_tree_widths": rel_combine_tree_widths,
                   "saturated_decode_cut": rel_saturated_decode_cut,
                   "lazy_rescale_branches": rel_lazy_rescale_branches}


def run_relation_sm100(name: str, backend, cfg: Config):
    """``run_relation`` for this package's own relations (same Outcome contract)."""
    import time
    import traceback

    from isoexec.ops.attention.mla_sm90.tests._harness import Unsupported
    from isoexec.ops.attention.mla_sm90.tests._relations import Outcome

    out = Outcome(name, backend.name)
    rec = Rec(out, backend)
    ident = getattr(backend, "identity", None)
    if callable(ident):
        out.witness["kernel_identity"] = ident()
    t0 = time.time()
    try:
        RELATIONS_SM100[name](rec, cfg)
        torch.cuda.synchronize()
    except Unsupported as e:
        out.skipped.append(f"remaining checks: {e}")
    except Exception as e:
        out.status = "error"
        out.notes.append(f"{type(e).__name__}: {str(e)[:400]}")
        out.notes.append(traceback.format_exc()[-1500:])
    out.seconds = round(time.time() - t0, 2)
    if out.status == "pass":
        vacuous = [n for n in out.notes if n.startswith("vacuous")]
        if out.failures or vacuous:
            out.status = "fail"
        elif not out.diffs:
            out.status = "n/a" if out.skipped else "fail"
            if not out.skipped:
                out.notes.append("no checks ran")
    return out
