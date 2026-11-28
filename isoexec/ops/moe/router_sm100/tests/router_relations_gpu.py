#!/usr/bin/env python
"""The router_sm100 relation battery (Gates 2 and 3), on one B200, against one digest.

    python isoexec/ops/moe/router_sm100/tests/router_relations_gpu.py --json battery.json

Every check is BIT equality (``torch.equal`` on the int32 view of fp32) unless named ``accuracy``.

Gate 2 -- ``route_select_only`` IS ``reference.route_reference`` (run on CUDA, op for op):
  select_vs_reference      every geometry, random / planted-tie / NaN / +-inf / all-equal rows
  select_repeat            the same call twice
  select_batch_invariance  a row alone == the same row inside T in {7, 256, 2048, 16384} with
                           duplicated rows, at shuffled positions
  select_graph_replay      CUDA-graph capture + replay == eager
  select_refusals          E % n_group != 0, a group of < 2, bias missing for A, hash layers
Gate 3 -- ``route_fused``:
  fused_vs_select_only     the fused table == select_only(the fused kernel's own logits), at the
                           default tile and at EVERY served ownership point
  fused_logits_ownership   logits identical across every (N, MR, CS, KA, persistent) point
  fused_batch_invariance   a row alone == inside T in {7, 256, 2048, 16384} (different tiles!)
  fused_repeat / fused_graph_replay
  fused_segment_decomposition   logits(S=4) == tree over four S=1 runs on the H/4 slices
  fused_logits_accuracy    vs the float64 GEMM (report)
Backward (DESIGN.md "Backward"):
  bwd_vs_reference         (dX, dW, dl, idx) == reference.route_backward_reference, op for op
  bwd_table                the backward's recomputed table == the forward fused table
  bwd_row_invariance       dl and dX rows alone == the same rows inside other batches, shuffled
  bwd_repeat / bwd_graph_replay
  bwd_accuracy             vs float64 autograd of the declared selection-fixed graph (report)
Integration:
  moe_sm100_dispatch_glm   moe_sm100.moe(route_select_only(moe_sm100 logits)) == moe_sm100.moe(
                           route_mechanics(same logits)) at GLM-4.7-Flash (n_group = topk_group = 1)
  table_format             int32 / fp32 [T, k+S], slots ascending in global expert id
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from isoexec.ops.moe.router_sm100 import ops, reference as R  # noqa: E402
from isoexec.ops.moe.router_sm100._extension import profile_digest  # noqa: E402

GEOMS = {
    "ds_v3": dict(routing=ops.GROUPED, E=256, k=8, n_group=8, topk_group=4, H=7168, S=0),
    "glm47": dict(routing=ops.GROUPED, E=64, k=4, n_group=1, topk_group=1, H=2048, S=0),
    "glm53": dict(routing=ops.GROUPED, E=288, k=8, n_group=1, topk_group=1, H=4096, S=0),
    "ds_ungrouped": dict(routing=ops.GROUPED, E=256, k=8, n_group=1, topk_group=1, H=7168, S=0),
    "v4_256": dict(routing=ops.SQRTSOFTPLUS, E=256, k=6, n_group=1, topk_group=1, H=4096, S=1),
    "v4_384": dict(routing=ops.SQRTSOFTPLUS, E=384, k=6, n_group=1, topk_group=1, H=7168, S=1),
    "v41_flash": dict(routing=ops.SQRTSOFTPLUS, E=384, k=6, n_group=1, topk_group=1, H=5120, S=1),
    "v4_512": dict(routing=ops.SQRTSOFTPLUS, E=512, k=6, n_group=1, topk_group=1, H=7168, S=1),
}


class Battery:
    def __init__(self):
        self.checks, self.fails, self.notes = 0, [], {}

    def eq(self, name, a, b):
        self.checks += 1
        ok = all(_bits_equal(x, y) for x, y in zip(a, b)) if isinstance(a, (tuple, list)) else _bits_equal(a, b)
        if not ok:
            self.fails.append(name)
            print(f"  FAIL {name}", flush=True)
        return ok

    def true(self, name, cond):
        self.checks += 1
        if not cond:
            self.fails.append(name)
            print(f"  FAIL {name}", flush=True)


def _bits_equal(a, b):
    if a is None or b is None:
        return a is b
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def kwargs(g, bias, mask=None, scale=2.5):
    return dict(routing=g["routing"], top_k=g["k"], n_group=g["n_group"], topk_group=g["topk_group"],
                expert_bias=bias, routed_scale=scale, num_shared=g["S"], mask=mask)


def logits_family(T, E, gen):
    """Random rows with every hazard the program names, in one batch."""
    x = torch.randn(T, E, generator=gen) * 2.0
    q = T // 6
    x[:q] = torch.round(x[:q] * 2.0) / 2.0                        # planted ties
    x[q:2 * q, 1::2] = x[q:2 * q, 0::2]                           # duplicated experts
    x[2 * q:2 * q + 3] = 0.25                                     # an all-equal row
    if T >= 12:
        x[2 * q + 3, ::3] = float("nan")                          # NaN keys -> -inf
        x[2 * q + 4, ::5] = float("inf")
        x[2 * q + 5, ::7] = float("-inf")
        x[2 * q + 6, :] = float("nan")                            # a whole NaN row
    return x.cuda()


def bias_for(g, gen, kind):
    if kind == "none":
        return None
    b = torch.randn(g["E"], generator=gen) / 8
    if kind == "tied":
        b = torch.round(b * 16) / 16
    return b.cuda()


def gate2(B, gen):
    print("Gate 2: select_only", flush=True)
    for gname, g in GEOMS.items():
        kinds = ("rand", "tied") + (("none",) if g["routing"] == ops.SQRTSOFTPLUS else ())
        for kind in kinds:
            b = bias_for(g, gen, kind)
            for T in (1, 7, 256, 2048, 16384):
                x = logits_family(T, g["E"], gen)
                mask = None
                if g["routing"] == ops.SQRTSOFTPLUS and T >= 7:
                    mask = (torch.rand(T, generator=gen) > 0.2).cuda()
                kw = kwargs(g, b, mask)
                out = ops.route_select_only(x, **kw)
                B.eq(f"select_vs_reference/{gname}/{kind}/T{T}", out, R.route_reference(x, **kw))
                B.eq(f"select_repeat/{gname}/{kind}/T{T}", out, ops.route_select_only(x, **kw))
                if T == 2048:
                    # batch invariance: rows alone vs inside other batches, duplicated, shuffled
                    for Tb in (7, 256, 2048, 16384):
                        m = 3 if Tb < 10 else 5
                        pick = torch.randint(0, T, (m,), generator=gen)
                        perm = torch.randperm(Tb, generator=gen)
                        pos, dup = perm[:m], perm[m:2 * m]          # disjoint slots
                        big = logits_family(Tb, g["E"], gen)
                        big[pos.cuda()] = x[pick.cuda()]
                        big[dup.cuda()] = x[pick.cuda()]
                        mb = None
                        if mask is not None:
                            mb = torch.ones(Tb, dtype=torch.bool, device="cuda")
                            mb[pos.cuda()] = mask[pick.cuda()]
                            mb[dup.cuda()] = mask[pick.cuda()]
                        ob = ops.route_select_only(big, **kwargs(g, b, mb))
                        for name, rows in (("pos", pos), ("dup", dup)):
                            B.eq(f"select_batch_invariance/{gname}/{kind}/T{Tb}/{name}",
                                 (ob[0][rows.cuda()], ob[1][rows.cuda()]),
                                 (out[0][pick.cuda()], out[1][pick.cuda()]))
        # CUDA graph replay
        b = bias_for(g, gen, "rand")
        x = logits_family(512, g["E"], gen)
        kw = kwargs(g, b)
        eager = ops.route_select_only(x, **kw)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ops.route_select_only(x, **kw)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = ops.route_select_only(x, **kw)
        graph.replay()
        graph.replay()
        B.eq(f"select_graph_replay/{gname}", gout, eager)
    # refusals
    x = torch.randn(4, 96, device="cuda")
    bb = torch.zeros(96, device="cuda")
    for name, call in (
        ("E%n_group", lambda: ops.route_select_only(x, routing=ops.GROUPED, top_k=4, n_group=7,
                                                    topk_group=2, expert_bias=bb)),
        ("group<2", lambda: ops.route_select_only(torch.randn(4, 8, device="cuda"), routing=ops.GROUPED,
                                                  top_k=2, n_group=8, topk_group=2,
                                                  expert_bias=torch.zeros(8, device="cuda"))),
        ("bias_missing", lambda: ops.route_select_only(x, routing=ops.GROUPED, top_k=4)),
        ("B_grouped", lambda: ops.route_select_only(x, routing=ops.SQRTSOFTPLUS, top_k=4, n_group=2,
                                                    topk_group=1)),
        ("hash_layers", lambda: ops._args(ops.SQRTSOFTPLUS, 96, 4, 1, 1, None, 0, None, 4, x.device,
                                          hash_routed=True)),
    ):
        try:
            call()
            torch.cuda.synchronize()
            B.true(f"select_refusals/{name}", False)
        except (RuntimeError, ValueError):
            B.true(f"select_refusals/{name}", True)


def ladder(E):
    out = []
    for mr in (128, 64):
        ce = -(-E // mr)
        for cs in (1, 2, 4):
            if ce * cs > 8 or (mr == 64 and cs == 4 and E != 64) or (E > 256 and cs == 4):
                continue
            for n in (16, 32, 64, 128) + ((256,) if mr == 128 and cs >= 2 else ()):
                for ka in (4, 2, 1):
                    out.append((n, mr, cs, ka))
    for ka in (1, 2, 4):
        out.append((128, 128, 1, ka, 1, 1))     # the persistent prefill schedule
    out.append((128, 128, 1, 2, 1, 64))         # persistent, capped at 64 clusters
    return out


def gate3(B, gen, quick):
    print("Gate 3: fused", flush=True)
    acc = {}
    for gname, g in GEOMS.items():
        b = bias_for(g, gen, "rand")
        kw = kwargs(g, b)
        wgt = (torch.randn(g["E"], g["H"], generator=gen) / 8).to("cuda", torch.bfloat16)
        T0 = 1000
        x = (torch.randn(T0, g["H"], generator=gen) / 4).to("cuda", torch.bfloat16)
        x[7] = x[3]
        idx, w, lg = ops.route_fused(x, wgt, return_logits=True, **kw)
        B.eq(f"fused_vs_select_only/{gname}/default", (idx, w), ops.route_select_only(lg, **kw))
        B.eq(f"fused_repeat/{gname}", (idx, w, lg), ops.route_fused(x, wgt, return_logits=True, **kw))
        # batch invariance: the default tile CHANGES with T
        pick = torch.tensor([0, 3, 7, 999])
        for Tb in ((7, 256, 2048) if quick else (7, 256, 2048, 16384)):
            big = (torch.randn(Tb, g["H"], generator=gen) / 4).to("cuda", torch.bfloat16)
            pos = torch.randperm(Tb, generator=gen)[:4] if Tb >= 4 else torch.arange(4) % Tb
            big[pos.cuda()] = x[pick.cuda()]
            ob = ops.route_fused(big, wgt, return_logits=True, **kw)
            B.eq(f"fused_batch_invariance/{gname}/T{Tb}",
                 (ob[0][pos.cuda()], ob[1][pos.cuda()], ob[2][pos.cuda()]),
                 (idx[pick.cuda()], w[pick.cuda()], lg[pick.cuda()]))
        single = ops.route_fused(x[3:4].contiguous(), wgt, return_logits=True, **kw)
        B.eq(f"fused_batch_invariance/{gname}/T1", single, (idx[3:4], w[3:4], lg[3:4]))
        B.eq(f"fused_duplicate_rows/{gname}", (idx[7], w[7], lg[7]), (idx[3], w[3], lg[3]))
        # graph replay
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ops.route_fused(x, wgt, **kw)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = ops.route_fused(x, wgt, **kw)
        graph.replay()
        B.eq(f"fused_graph_replay/{gname}", gout, (idx, w))
        # accuracy vs float64
        ref = x.double() @ wgt.double().t()
        err = (lg.double() - ref).abs()
        acc[gname] = {"max_abs": float(err.max()), "max_rel_to_rowmax": float((err.amax(1) / ref.abs().amax(1)).max())}
    B.notes["fused_logits_accuracy"] = acc

def backward(B, gen):
    print("Backward", flush=True)
    acc = {}
    for gname in ("ds_v3", "glm47", "v4_512"):
        g = GEOMS[gname]
        b = bias_for(g, gen, "rand")
        wgt = (torch.randn(g["E"], g["H"], generator=gen) / 8).to("cuda", torch.bfloat16)
        for T in (7, 2048, 5000):
            x = (torch.randn(T, g["H"], generator=gen) / 4).to("cuda", torch.bfloat16)
            mask = (torch.rand(T, generator=gen) > 0.1).cuda() if g["routing"] == ops.SQRTSOFTPLUS else None
            kw = kwargs(g, b, mask)
            fi, fw, lg = ops.route_fused(x, wgt, return_logits=True, **kw)
            dp = torch.randn(T, g["k"], generator=gen).cuda()
            o = ops.route_backward(dp, x, wgt, lg, **kw)
            B.eq(f"bwd_vs_reference/{gname}/T{T}", tuple(o), tuple(R.route_backward_reference(dp, x, wgt, lg, **kw)))
            B.eq(f"bwd_table/{gname}/T{T}", o[3], fi)
            B.eq(f"bwd_repeat/{gname}/T{T}", tuple(o), tuple(ops.route_backward(dp, x, wgt, lg, **kw)))
            if T == 2048:
                pick = torch.tensor([0, 5, 77, 2047])
                for Tb in (7, 4096):
                    perm = torch.randperm(Tb, generator=gen)[:4]
                    xb = (torch.randn(Tb, g["H"], generator=gen) / 4).to("cuda", torch.bfloat16)
                    lb = torch.randn(Tb, g["E"], generator=gen).cuda()
                    db = torch.randn(Tb, g["k"], generator=gen).cuda()
                    xb[perm.cuda()] = x[pick.cuda()]
                    lb[perm.cuda()] = lg[pick.cuda()]
                    db[perm.cuda()] = dp[pick.cuda()]
                    mb = None
                    if mask is not None:
                        mb = torch.ones(Tb, dtype=torch.bool, device="cuda")
                        mb[perm.cuda()] = mask[pick.cuda()]
                    ob = ops.route_backward(db, xb, wgt, lb, **kwargs(g, b, mb))
                    B.eq(f"bwd_row_invariance/{gname}/T{Tb}",
                         (ob[0][perm.cuda()], ob[2][perm.cuda()]), (o[0][pick.cuda()], o[2][pick.cuda()]))
                # graph replay
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    ops.route_backward(dp, x, wgt, lg, **kw)
                torch.cuda.current_stream().wait_stream(s_)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    go = ops.route_backward(dp, x, wgt, lg, **kw)
                graph.replay()
                B.eq(f"bwd_graph_replay/{gname}", tuple(go), tuple(o))
                # accuracy vs float64 autograd, selection held fixed
                ids = o[3][:, :g["k"]].long()
                valid = (ids >= 0).all(1)
                idc = ids.clamp(min=0)
                l64 = lg.double().requires_grad_(True)
                lsel = torch.gather(l64, 1, idc)
                sc = torch.sigmoid(lsel) if g["routing"] == ops.GROUPED else torch.nn.functional.softplus(lsel).sqrt()
                wts = sc / (sc.sum(1, keepdim=True) + 1e-20) * 2.5
                ((wts * dp.double()).sum(1) * valid).sum().backward()
                G = torch.zeros(T, g["E"], dtype=torch.float64, device="cuda").scatter(1, idc, torch.gather(l64.grad, 1, idc))
                G = G * valid[:, None]
                dw64 = G.t() @ x.double()
                dx64 = G @ wgt.double()
                rel = lambda a, r: float((a.double() - r).abs().max() / r.abs().max().clamp(min=1e-30))
                acc[gname] = {"dl": rel(o[2] * valid[:, None], torch.gather(G, 1, idc)),
                              "dW": rel(o[1], dw64), "dX_bf16": rel(o[0], dx64)}
    B.notes["bwd_accuracy_max_rel_to_max"] = acc


def integration(B, gen):
    print("Integration: moe_sm100 dispatch", flush=True)
    from isoexec.ops.moe.moe_sm100 import ops as moe
    E, k, H, I = 64, 4, 2048, 1536
    for T in (1, 128, 2048):
        x = (torch.randn(T, H, generator=gen) / 4).to("cuda", torch.bfloat16)
        rw = (torch.randn(E, H, generator=gen) / 8).to("cuda", torch.bfloat16)
        gu = (torch.randn(E, 2 * I, H, generator=gen) / 16).to("cuda", torch.bfloat16)
        dn = (torch.randn(E, H, I, generator=gen) / 16).to("cuda", torch.bfloat16)
        b = (torch.randn(E, generator=gen) / 8).cuda()
        logits = moe.route_score(x, rw)
        ri, rp = moe.route_mechanics(logits, k, expert_bias=b, routed_scale=1.8)
        oi, ow = ops.route_select_only(logits, routing=ops.GROUPED, top_k=k, expert_bias=b, routed_scale=1.8)
        # same selection as a set and the same weight per expert, bit for bit
        order = torch.argsort(ri.long(), 1)
        B.eq(f"glm_table_vs_moe_sm100_mechanics/T{T}", (torch.gather(ri, 1, order), torch.gather(rp, 1, order)),
             (oi, ow))
        y_ref = moe.moe(x, gu, dn, ri, rp)
        y_ours = moe.moe(x, gu, dn, oi, ow)
        B.eq(f"moe_sm100_dispatch_glm/T{T}", y_ours, y_ref)
    # the table format both consumers read
    g = GEOMS["v4_256"]
    xx = (torch.randn(64, 4096, generator=gen) / 4).to("cuda", torch.bfloat16)
    ww = (torch.randn(256, 4096, generator=gen) / 8).to("cuda", torch.bfloat16)
    i2, w2 = ops.route_fused(xx, ww, **kwargs(g, None))
    B.true("table_format/dtypes", i2.dtype == torch.int32 and w2.dtype == torch.float32 and i2.shape == (64, 7))
    B.true("table_format/ascending", bool((i2[:, 1:] > i2[:, :-1]).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    gen = torch.Generator().manual_seed(20260930)
    B = Battery()
    t0 = time.time()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    print(f"{uuid} digest {profile_digest()}", flush=True)
    gate2(B, gen)
    gate3(B, gen, a.quick)
    backward(B, gen)
    integration(B, gen)
    rep = {"gpu_uuid": uuid, "digest": profile_digest(), "checks": B.checks, "failures": B.fails,
           "notes": B.notes, "seconds": round(time.time() - t0, 1)}
    Path(a.json).write_text(json.dumps(rep, indent=1))
    print(f"{B.checks} checks, {len(B.fails)} failures; accuracy {B.notes.get('fused_logits_accuracy')}",
          flush=True)
    return 1 if B.fails else 0


if __name__ == "__main__":
    sys.exit(main())
