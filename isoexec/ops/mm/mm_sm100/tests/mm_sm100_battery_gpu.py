#!/usr/bin/env python3
"""Correctness and determinism battery for the SM100 deterministic dense GEMM."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

REPO_ROOT = __file__.rsplit("/isoexec/", 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _bits(t):
    """Integer view of a float tensor, so +0.0 and -0.0 are different values."""
    import torch

    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def bitwise_equal(x, y) -> bool:
    import torch

    return bool(torch.equal(_bits(x), _bits(y)))


# (M, N, K). The 12-row foundation matrix plus census shapes of the example models:
# Qwen 3.5 35B qkv at TP8/TP1, the 9B MLP fc1, the 27B qkv, and a row-parallel width.
ACCURACY_SHAPES = [
    (m, n, k)
    for k, n in ((2048, 1152), (4096, 4096), (5120, 14336))
    for m in (1, 32, 512, 4096)
] + [
    (1, 9216, 2048), (32, 9216, 2048), (512, 9216, 2048),
    (1, 12352, 2048), (32, 12352, 2048),
    (1, 24576, 4096), (512, 24576, 4096),
    (1, 5120, 14336), (32, 5120, 14336),
    (7, 1152, 2048), (130, 4096, 4096), (1023, 1152, 2048), (1024, 1152, 2048),
]

#: Populations: the ordinary one, one that lands in subnormals after the products, one with a
#: signed zero, and one wide enough that the tree and a chain would visibly disagree.
POPULATIONS = ("normal", "underflow", "signed_zero", "wide")

#: Shapes the relations run on. Kept small enough that a full M-composition sweep and a plan
#: sweep fit in one lease; the widest weights are covered by the accuracy sweep.
RELATION_SHAPES = [(1152, 2048), (4096, 4096), (5120, 14336), (9216, 2048)]
RELATION_MS = (1, 32, 512, 4096)

#: At and above this M, cuBLAS runs an ordinary unsplit GEMM on every shape here, so our unsplit
#: chain must match it bit for bit. Below it cuBLAS may pick a split-K kernel from the occupancy
#: the shape leaves -- exactly the choice DTIR forbids -- and then the bits legitimately differ.
#: Section 8 measures which shapes that happens on instead of assuming.
CUBLAS_UNSPLIT_M = 64


def _plans_at_split(seg, M, K, N, split):
    """Every compiled plan for this shape at an explicit split (an execution split, not the policy's)."""
    return seg.plans_at_split(M, K, N, split)


def make_operands(M, N, K, population, device, seed):
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    if population == "normal":
        a = torch.randn(M, K, generator=g, device=device) * 0.05
        w = torch.randn(N, K, generator=g, device=device) * 0.05
    elif population == "underflow":
        a = torch.randn(M, K, generator=g, device=device) * 2.0**-9
        w = torch.randn(N, K, generator=g, device=device) * 2.0**-9
    elif population == "signed_zero":
        a = torch.randn(M, K, generator=g, device=device) * 0.05
        w = torch.randn(N, K, generator=g, device=device) * 0.05
        a[torch.rand(M, K, generator=g, device=device) < 0.5] = -0.0
        w[torch.rand(N, K, generator=g, device=device) < 0.25] = 0.0
    elif population == "wide":
        # Magnitudes spread over ~2^20 so the grouping of the K reduction is visible in the
        # result: a tree and a chain over these do not agree, which is what makes the plan
        # relations non-vacuous.
        a = torch.randn(M, K, generator=g, device=device)
        w = torch.randn(N, K, generator=g, device=device)
        a *= 2.0 ** torch.randint(-10, 11, (M, K), generator=g, device=device).float()
        w *= 2.0 ** torch.randint(-10, 11, (N, K), generator=g, device=device).float()
    else:
        raise ValueError(population)
    return a.bfloat16().contiguous(), w.bfloat16().contiguous().t()


#: Execution-plan segment sizes below the dispatch's 256-element alignment (README section 4.17):
#: pik's `mimo7b_mlp_down` leaf (1376 = 21.5 x 64, shards 5504 / 2752), a 1040 = 16 x 65 probe that
#: is a tail on every K block (64, 128 and 256), pik's `35b_shared_fc2` 64-wide leaf, and a 16-wide
#: segment (one UMMA). (N, K, segment_k).
SUB_ALIGN_SHAPES = [(4096, 5504, 1376), (4096, 2752, 1376), (2048, 4160, 1040), (2048, 128, 64),
                    (1152, 256, 16)]


def collective_entry_points(torch, ops, seg, ext, dev, quick, failures):
    """Relations 9-12: fp32 root, publish-only, bf16 leaf, segment sizes below 256."""
    out = {}
    ms = (1, 32, 512) if quick else RELATION_MS

    # ---------------------------------------------------------------- 9. the fp32 root
    # bf16_rne(root) == the published bf16 output, bitwise, and the root itself is plan-invariant
    # bitwise across every family / A height that publishes it, and across the grid width.
    roots = []
    for (N, K) in RELATION_SHAPES:
        for pop in ("wide",) if quick else ("normal", "wide"):
            for M in ms:
                a, b = make_operands(M, N, K, pop, dev, seed=0xF00D + M + N + K)
                split = seg.segment_count(K, torch.bfloat16)
                pub = ops.run_plan(a, b, seg.plan_for(M, K, N, torch.bfloat16))
                cands = seg.root_plans(M, K, N, split)
                base = None
                for pl in cands:
                    for sl in (0,) if pl is not cands[0] else (0, 1, 37):
                        r = ops.run_plan(a, b, pl, root_dtype=torch.float32, slots=sl).clone()
                        torch.cuda.synchronize()
                        ok_round = bitwise_equal(r.bfloat16(), pub)
                        if base is None:
                            base = r
                        ok_inv = bitwise_equal(r, base)
                        roots.append({"N": N, "K": K, "M": M, "pop": pop, "plan": pl.config,
                                      "slots": sl, "rounds_to_published": ok_round,
                                      "plan_invariant": ok_inv})
                        if not ok_round:
                            failures.append(f"fp32 root N={N} K={K} M={M} {pop} {pl.config}: "
                                            f"bf16_rne(root) != the published output")
                        if not ok_inv:
                            failures.append(f"fp32 root N={N} K={K} M={M} {pop} {pl.config} "
                                            f"slots={sl}: differs from {cands[0].config}")
                del a, b, pub, base
    # And through the public surface: root_dtype=float32 on the PRODUCTION plan picks its twin.
    for (N, K) in RELATION_SHAPES:
        M = 512
        a, b = make_operands(M, N, K, "wide", dev, seed=0xF00E + N)
        prod = seg.plan_for(M, K, N, torch.bfloat16)
        try:
            r = ops.run_plan(a, b, prod, root_dtype=torch.float32)
            ok = bitwise_equal(r.bfloat16(), ops.run_plan(a, b, prod))
        except ValueError as exc:
            ok = False
            failures.append(f"fp32 root of the production plan {prod.config}: {exc}")
        roots.append({"N": N, "K": K, "M": M, "pop": "wide", "plan": f"twin({prod.config})",
                      "slots": 0, "rounds_to_published": ok, "plan_invariant": True})
        if not ok:
            failures.append(f"fp32 root of production plan {prod.config} N={N} K={K}")
        del a, b
    out["fp32_root"] = roots
    print(f"[9] fp32 root: {len(roots)} comparisons, "
          f"{sum(1 for r in roots if not (r['rounds_to_published'] and r['plan_invariant']))} failed")

    # --------------------------------------------------------------- 10. publish only
    # combine=False: the workspace bits equal combine=True's, and C is untouched (sentinel).
    pubs = []
    for (N, K) in RELATION_SHAPES:
        for M in (1, 512) if quick else (1, 32, 512, 4096):
            a, b = make_operands(M, N, K, "wide", dev, seed=0xAB1E + M + N)
            for split in (2, 4):
                if K % (split * 256) or K // split < 512:
                    continue
                for pl in seg.plans_at_split(M, K, N, split):
                    if pl.mode != seg.COMBINE or pl.workspace_floats(M, N) * 4 > (2 << 30):
                        continue
                    ws1 = torch.full((pl.workspace_floats(M, N),), float("nan"),
                                     dtype=torch.float32, device=dev)
                    c1 = ops.run_plan(a, b, pl, workspace=ws1)
                    sentinel = torch.full((M, N), -7.0, dtype=torch.bfloat16, device=dev)
                    ws2 = torch.full_like(ws1, float("nan"))
                    v = ops.run_plan(a, b, pl, workspace=ws2, out=sentinel, combine=False)
                    torch.cuda.synchronize()
                    ok_ws = bitwise_equal(ws1, ws2) and v.shape == (pl.groups, M, N)
                    ok_sink = bool((sentinel == -7.0).all().item())
                    # The host tree over the published roots is the combine kernel's output
                    # (relation 7's claim, re-checked on the publish-only workspace); capped at
                    # M <= 512 because the host fold is a serial loop.
                    ok_ref = M > 512 or bitwise_equal(ext.merge_reference(v).cuda(), c1)
                    pubs.append({"N": N, "K": K, "M": M, "split": split, "plan": pl.config,
                                 "workspace_equal": ok_ws, "sink_untouched": ok_sink,
                                 "host_tree_equals_combined": ok_ref})
                    if not (ok_ws and ok_sink and ok_ref):
                        failures.append(
                            f"publish-only N={N} K={K} M={M} S={split} {pl.config}: ws={ok_ws} "
                            f"sink={ok_sink} host-tree={ok_ref}")
                    del ws1, ws2, c1, sentinel, v
            del a, b
    out["publish_only"] = pubs
    print(f"[10] publish-only: {len(pubs)} comparisons, "
          f"{sum(1 for r in pubs if not (r['workspace_equal'] and r['sink_untouched'] and r['host_tree_equals_combined']))} failed")

    # --------------------------------------------------------------- 11. bf16 leaves
    # publish_bf16[g] == bf16_rne(fp32 partial[g]) for every g, at spc == 1, plan-invariant.
    leaves = []
    for (N, K) in RELATION_SHAPES:
        for M in (1, 512) if quick else (1, 32, 512, 4096):
            a, b = make_operands(M, N, K, "wide", dev, seed=0x1EAF + M + N)
            for split in (1, 2, 4):
                if K % (split * 256) or K // split < 512:
                    continue
                cands = [pl for pl in seg.plans_at_split(M, K, N, split)
                         if pl.mode == seg.COMBINE and pl.spc == 1] + \
                        [pl for pl in seg.root_plans(M, K, N, split) if split == 1]
                base = None
                for pl in cands:
                    if split * M * N * 4 > (2 << 30):
                        continue
                    f32 = ops.run_plan(a, b, pl, combine=False).clone()
                    b16 = ops.run_plan(a, b, pl, combine=False, leaf_dtype=torch.bfloat16).clone()
                    torch.cuda.synchronize()
                    ok = bitwise_equal(b16, f32.bfloat16())
                    if base is None:
                        base = b16
                    inv = bitwise_equal(b16, base)
                    leaves.append({"N": N, "K": K, "M": M, "split": split, "plan": pl.config,
                                   "rne_of_fp32": ok, "plan_invariant": inv})
                    if not (ok and inv):
                        failures.append(f"bf16 leaf N={N} K={K} M={M} S={split} {pl.config}: "
                                        f"rne={ok} invariant={inv}")
                    del f32, b16
                del base
            del a, b
    out["bf16_leaf"] = leaves
    print(f"[11] bf16 leaves: {len(leaves)} comparisons, "
          f"{sum(1 for r in leaves if not (r['rne_of_fp32'] and r['plan_invariant']))} failed")

    # ------------------------------------------------ 12. segment sizes below 256 (execution)
    # (a) plan invariance of the fp32 roots / leaves / bf16 output over every compiled plan at the
    #     caller's split, including three different K-block depths (64, 128, 256) whose tail blocks
    #     differ in shape; (b) the LEAF witness: leaf g of the split equals the unsplit chain of the
    #     contiguous K slice [g*seg_k, (g+1)*seg_k) -- the tail block's extra columns are the next
    #     segment's real data in (a) and TMA zero-fill past K in (b), and neither is multiplied.
    subs = []
    observed = {}
    for (N, K, sk) in SUB_ALIGN_SHAPES:
        split = K // sk
        for M in (1, 512) if quick else (1, 32, 512):
            a, b = make_operands(M, N, K, "wide", dev, seed=0x5AB + M + N + K)
            w = b.t()  # [N, K]
            plans = seg.plans_at_split(M, K, N, split)
            if not plans:
                failures.append(f"sub-256 N={N} K={K} segment_k={sk}: no plan")
                continue
            base_out = None
            leaf_ref = []
            for g in range(split):
                # A fresh dense copy: at M = 1 `.contiguous()` of the slice is a no-op that keeps
                # the row stride K, which the TMA map (rightly) refuses.
                a_g = torch.empty((M, sk), dtype=a.dtype, device=dev)
                a_g.copy_(a[:, g * sk:(g + 1) * sk])
                w_g = torch.empty((N, sk), dtype=w.dtype, device=dev)
                w_g.copy_(w[:, g * sk:(g + 1) * sk])
                one = seg.root_plans(M, sk, N, 1)
                leaf_ref.append(ops.run_plan(a_g, w_g.t(), one[0], root_dtype=torch.float32)
                                .clone() if one else None)
            for pl in plans:
                if pl.workspace_floats(M, N) * 4 > (2 << 30):
                    continue
                o = ops.run_plan(a, b, pl).clone()
                torch.cuda.synchronize()
                if base_out is None:
                    base_out = o
                inv = bitwise_equal(o, base_out)
                leaf_ok = True
                if pl.mode == seg.COMBINE and pl.spc == 1:
                    v = ops.run_plan(a, b, pl, combine=False).clone()
                    torch.cuda.synchronize()
                    leaf_ok = all(r is None or bitwise_equal(v[g], r)
                                  for g, r in enumerate(leaf_ref))
                subs.append({"N": N, "K": K, "segment_k": sk, "M": M, "plan": pl.config,
                             "spc": pl.spc, "plan_invariant": inv, "leaf_is_slice_chain": leaf_ok})
                if not (inv and leaf_ok):
                    failures.append(f"sub-256 N={N} K={K} segment_k={sk} M={M} {pl.config}: "
                                    f"invariant={inv} leaf==slice-chain={leaf_ok}")
                del o
            # Non-vacuity: the split must be observable against the unsplit chain of the same K,
            # or relation (a) would prove nothing.
            one = seg.root_plans(M, K, N, 1)
            if one:
                unsplit = ops.run_plan(a, b, one[0], root_dtype=torch.float32).bfloat16()
                torch.cuda.synchronize()
                observed.setdefault((N, K, sk), False)
                if not bitwise_equal(unsplit, base_out):
                    observed[(N, K, sk)] = True
            # and the fp32 root of the whole tree, plan-invariant, where a twin publishes it
            rbase = None
            for pl in seg.root_plans(M, K, N, split):
                r = ops.run_plan(a, b, pl, root_dtype=torch.float32).clone()
                torch.cuda.synchronize()
                if rbase is None:
                    rbase = r
                ok = bitwise_equal(r, rbase) and bitwise_equal(r.bfloat16(), base_out)
                subs.append({"N": N, "K": K, "segment_k": sk, "M": M, "plan": f"root:{pl.config}",
                             "spc": pl.spc, "plan_invariant": ok, "leaf_is_slice_chain": True})
                if not ok:
                    failures.append(f"sub-256 root N={N} K={K} segment_k={sk} M={M} {pl.config}")
            del a, b, w, base_out, leaf_ref
    # Non-vacuity per shape: at M = 1 a 64- or 16-element segment over a 2^20-spread population can
    # happen to round like the chain; the split must be observable at SOME M of the sweep.
    for key, seen in observed.items():
        if not seen:
            failures.append(f"sub-256 N, K, segment_k = {key}: the split is not observable at any M")
    out["sub_align_observable"] = {str(k): v for k, v in observed.items()}
    out["sub_align_segments"] = subs
    print(f"[12] segment_k below 256: {len(subs)} comparisons over "
          f"{sorted({(r['K'], r['segment_k']) for r in subs})}, "
          f"{sum(1 for r in subs if not (r['plan_invariant'] and r['leaf_is_slice_chain']))} failed")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one population, fewer shapes")
    ap.add_argument("--json", default="", help="write the record here")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"INCOMPLETE: torch unavailable: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("INCOMPLETE: no CUDA device")
        return 2

    from isoexec.core.arch import live_device_arch

    if live_device_arch() != "sm100":
        print(f"INCOMPLETE: this battery proves an sm100 claim and the device is "
              f"{live_device_arch()}")
        return 2

    torch.backends.cuda.matmul.allow_tf32 = False
    from isoexec.ops.mm.mm_sm100 import ops
    from isoexec.ops.mm.mm_sm100 import segments as seg
    from isoexec.ops.mm.mm_sm100._extension import extension

    try:
        ops.require_sm100()
    except RuntimeError as exc:
        print(f"INCOMPLETE: {exc}")
        return 2

    dev = "cuda"
    ext = extension()
    record = {
        "recipe": ops.RECIPE_VERSION,
        "schedule_digest": seg.schedule_digest(),
        "schedule_policy_version": seg.schedule_policy.VERSION,
        "device": torch.cuda.get_device_name(0),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "sm_count": ext.sm_count(),
        "accuracy": [],
        "relations": {},
    }
    failures: list[str] = []
    pops = ("normal",) if args.quick else POPULATIONS
    shapes = ACCURACY_SHAPES[:6] if args.quick else ACCURACY_SHAPES

    # ---------------------------------------------------------------- 1. accuracy and repeats
    # Gate before anything else: an FP64 reference, with the tolerance pinned to cuBLAS's own
    # error on the same inputs so the gate scales with the shape rather than with a guess.
    biteq_expected_true = 0
    for (M, N, K) in shapes:
        for pop in pops:
            a, b = make_operands(M, N, K, pop, dev, seed=0x5EED + M + N + K + hash(pop) % 997)
            plan = seg.plan_for(M, K, N, torch.bfloat16)
            ref = (a.double() @ b.double()).float()
            cublas = torch.matmul(a, b)
            out = ops.run_plan(a, b, plan)
            out2 = ops.run_plan(a, b, plan)
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().max().item()
            cerr = (cublas.float() - ref).abs().max().item()
            gate = max(4.0 * cerr, 1e-6)
            repeat_ok = bitwise_equal(out, out2)
            # An unsplit chain is bit-identical to cuBLAS wherever cuBLAS is itself running an
            # unsplit chain: that is the evidence that the K order this kernel consumes IS the
            # tensor core's canonical order, which is what the whole recipe rests on. A split
            # weight must not match, by design -- and neither does an unsplit one at an M where
            # cuBLAS has chosen a split-K kernel of its own, which is precisely the defect this
            # recipe exists to remove. Section 8 measures that choice directly, so the gate here
            # is only applied where cuBLAS is not making it.
            cublas_biteq = bitwise_equal(out, cublas)
            expect_biteq = plan.split == 1 and M >= CUBLAS_UNSPLIT_M
            ok = err <= gate and repeat_ok and (cublas_biteq or not expect_biteq)
            if expect_biteq:
                biteq_expected_true += 1
                if not cublas_biteq:
                    failures.append(
                        f"unsplit M={M} N={N} K={K} {pop}: not bit-identical to cuBLAS"
                    )
            if err > gate:
                failures.append(f"accuracy M={M} N={N} K={K} {pop}: {err:.3e} > {gate:.3e}")
            if not repeat_ok:
                failures.append(f"repeat M={M} N={N} K={K} {pop}: two launches differ")
            record["accuracy"].append(
                {"M": M, "N": N, "K": K, "pop": pop, "split": plan.split,
                 "plan": plan.config, "groups": plan.groups, "err": err, "cublas_err": cerr,
                 "repeat_ok": repeat_ok, "cublas_biteq": cublas_biteq,
                 "cublas_biteq_expected": expect_biteq, "ok": ok}
            )
            del a, b, ref, cublas, out, out2
    print(f"[1] accuracy + repeat: {len(record['accuracy'])} cases, "
          f"{biteq_expected_true} of them unsplit and gated bit-identical to cuBLAS")

    # -------------------------------------------- 2. non-vacuity: the split really changes bits
    # If a split output happened to equal the unsplit one on these inputs, every relation below
    # would pass for the wrong reason. Prove the split is observable first.
    nonvacuous = {}
    for (N, K) in RELATION_SHAPES:
        a, b = make_operands(32, N, K, "wide", dev, seed=99)
        unsplit = seg.Plan(config="narrow_s1", segment_k=K, split=1, spc=1)
        base = ops.run_plan(a, b, unsplit).clone()
        # Every split the kernel compiles, not only the published one: the relations below are
        # swept at all of them, so all of them have to be observable.
        for split in (2, 4):
            if K % (split * 256) or K // split < 512:
                continue
            for pl in _plans_at_split(seg, 32, K, N, split):
                differs = not bitwise_equal(ops.run_plan(a, b, pl), base)
                nonvacuous[f"{N}x{K}_S{split}_{pl.config}"] = differs
                if not differs:
                    failures.append(
                        f"vacuous: S={split} at N={N} K={K} via {pl.config} gives the same bits as "
                        f"the unsplit chain on the wide population, so the plan relations below "
                        f"prove nothing"
                    )
        del a, b, base
    record["relations"]["split_is_observable"] = nonvacuous
    print(f"[2] split observability: {nonvacuous}")

    # -------------------------------------------------------------- 3. batch composition
    # Row r of an M=1 call must equal row r of the same rows inside M=32/512/4096. The plans
    # differ across those M (different tile family, different segments per CTA, different grid),
    # so this relation also covers the plan relation at the same time.
    comp = []
    for (N, K) in RELATION_SHAPES:
        for pop in ("wide",) if args.quick else ("normal", "wide"):
            big = max(RELATION_MS)
            a_big, b = make_operands(big, N, K, pop, dev, seed=0xC0FFEE + N + K)
            out_big = ops.run_plan(a_big, b, seg.plan_for(big, K, N, torch.bfloat16))
            torch.cuda.synchronize()
            for M in RELATION_MS:
                if M == big:
                    continue
                a = a_big[:M].contiguous()
                out = ops.run_plan(a, b, seg.plan_for(M, K, N, torch.bfloat16))
                torch.cuda.synchronize()
                ok = bitwise_equal(out, out_big[:M])
                comp.append({"N": N, "K": K, "pop": pop, "M": M, "vs_M": big, "ok": ok})
                if not ok:
                    bad = (_bits(out) != _bits(out_big[:M])).sum().item()
                    failures.append(
                        f"batch composition N={N} K={K} {pop}: M={M} differs from rows of "
                        f"M={big} in {bad} elements"
                    )
                del a, out
            del a_big, b, out_big
    record["relations"]["batch_composition"] = comp
    print(f"[3] batch composition: {len(comp)} comparisons, "
          f"{sum(1 for c in comp if not c['ok'])} failed")

    # ------------------------------------------------------------------- 4. execution plans
    # Every legal plan under one schedule: segments-per-CTA 1 (one CTA per segment, tree
    # merged from global by a combine kernel) through all of them (one CTA per tile, tree merged
    # in tensor memory), across three tile families with different raster groups and grids.
    # Swept at every split the KERNEL compiles, not only the policy's one segment: explicit callers
    # (execution splits, row-parallel leaf trees) run them, and the tree's decomposition at depth 3
    # stays gated.
    plans = []
    for (N, K) in RELATION_SHAPES:
        for M in (1, 32, 512) if args.quick else RELATION_MS:
            a, b = make_operands(M, N, K, "wide", dev, seed=0xBEEF + M + N + K)
            for split in (1, 2, 4):
                if K % (split * 256) or K // split < 512:
                    continue
                legal = [p for p in _plans_at_split(seg, M, K, N, split)
                         if p.workspace_floats(M, N) * 4 <= (4 << 30)]
                if len(legal) < 2:
                    continue
                base = ops.run_plan(a, b, legal[0]).clone()
                torch.cuda.synchronize()
                for pl in legal[1:]:
                    out = ops.run_plan(a, b, pl)
                    torch.cuda.synchronize()
                    ok = bitwise_equal(out, base)
                    plans.append({"N": N, "K": K, "M": M, "split": split, "plan": pl.config,
                                  "spc": pl.spc, "groups": pl.groups, "vs": legal[0].config,
                                  "ok": ok})
                    if not ok:
                        failures.append(
                            f"plan N={N} K={K} M={M} S={split}: {pl.config} (spc={pl.spc}, "
                            f"groups={pl.groups}) differs from {legal[0].config}")
                    del out
                del base
            del a, b
    record["relations"]["execution_plans"] = plans
    print(f"[4] execution plans: {len(plans)} comparisons, "
          f"{sum(1 for p in plans if not p['ok'])} failed")

    # --------------------------------------------------------------- 5. persistent grid width
    grids = []
    for (N, K) in RELATION_SHAPES[:2] if args.quick else RELATION_SHAPES:
        for M in (1, 512):
            a, b = make_operands(M, N, K, "wide", dev, seed=0x6D1D + M + N)
            plan = seg.plan_for(M, K, N, torch.bfloat16)
            base = ops.run_plan(a, b, plan).clone()
            torch.cuda.synchronize()
            for slots in (1, 3, 8, 37, 74):
                out = ops.run_plan(a, b, plan, slots=slots)
                torch.cuda.synchronize()
                ok = bitwise_equal(out, base)
                grids.append({"N": N, "K": K, "M": M, "slots": slots, "ok": ok})
                if not ok:
                    failures.append(f"grid N={N} K={K} M={M}: {slots} cluster slots differ")
                del out
            del a, b, base
    record["relations"]["grid_width"] = grids
    print(f"[5] persistent grid width: {len(grids)} comparisons, "
          f"{sum(1 for g in grids if not g['ok'])} failed")

    # ------------------------------------------------------------- 6. CUDA graph vs eager
    graphs = []
    for (N, K) in RELATION_SHAPES[:2] if args.quick else RELATION_SHAPES:
        for M in (1, 512):
            a, b = make_operands(M, N, K, "wide", dev, seed=0x9A9A + M + N)
            plan = seg.plan_for(M, K, N, torch.bfloat16)
            eager = ops.run_plan(a, b, plan).clone()
            out = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
            need = plan.workspace_floats(M, N)
            ws = torch.zeros(max(need, 1), dtype=torch.float32, device=dev)
            # Warm up on a side stream, which is what torch asks for before capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    ops.run_plan(a, b, plan, out=out, workspace=ws)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            out.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.run_plan(a, b, plan, out=out, workspace=ws)
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            first = out.clone()
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            ok = bitwise_equal(first, eager) and bitwise_equal(out, eager)
            graphs.append({"N": N, "K": K, "M": M, "ok": ok})
            if not ok:
                failures.append(f"cuda graph N={N} K={K} M={M}: replay differs from eager")
            del a, b, eager, out, ws, graph, first
    record["relations"]["cuda_graph"] = graphs
    print(f"[6] CUDA graph vs eager: {len(graphs)} comparisons, "
          f"{sum(1 for g in graphs if not g['ok'])} failed")

    # ------------------------------------------ 7. the device merge IS the declared tree
    # Take the FP32 subtree roots the leaf published and fold them on the host with the tree as
    # written in segments.py's docstring; the combine kernel's output must equal it bit for bit.
    merges = []
    for (N, K) in RELATION_SHAPES:
        M = 32
        a, b = make_operands(M, N, K, "wide", dev, seed=0x7EEE + N)
        for split in (2, 4):
            if K % (split * 256) or K // split < 512:
                continue
            # Explicitly the fully cut tree: one CTA per segment, so every level above the leaf
            # is evaluated by the device and the host reference covers all of it. Both device
            # evaluators are checked -- the separate combine kernel and the in-kernel
            # last-arriver fold -- because each is a distinct implementation of the SAME
            # declared tree over the SAME published FP32 roots.
            cut = [p for p in _plans_at_split(seg, M, K, N, split) if p.spc == 1]
            by_mode = {}
            for p_ in cut:
                by_mode.setdefault(p_.mode, p_)
            if not by_mode:
                continue
            for mode, plan in sorted(by_mode.items()):
                ws = torch.zeros(plan.workspace_floats(M, N), dtype=torch.float32, device=dev)
                out = ops.run_plan(a, b, plan, workspace=ws)
                torch.cuda.synchronize()
                host = ext.merge_reference(ws.view(plan.groups, M, N))
                ok = bitwise_equal(out.cpu(), host)
                merges.append({"N": N, "K": K, "split": split, "groups": plan.groups,
                               "mode": mode, "config": plan.config, "ok": ok})
                if not ok:
                    failures.append(
                        f"merge N={N} K={K} S={split} {plan.config}: the device fold is not the "
                        f"declared tree")
                del ws, out, host
        del a, b
    record["relations"]["merge_is_declared_tree"] = merges
    print(f"[7] merge == declared tree: {len(merges)} comparisons, "
          f"{sum(1 for m in merges if not m['ok'])} failed")

    # ------------------------------------------- 8. the baseline is not batch-invariant
    # Not a claim about this kernel: a claim about what it replaces. cuBLAS picks its kernel, and
    # its split, from the occupancy the shape leaves, so the same weight and the same row can come
    # back with different bits at a different batch size. Where that happens, our unsplit chain
    # cannot match cuBLAS at both M -- and it is the reason the op needs a recipe of its own.
    baseline = []
    for (N, K) in RELATION_SHAPES + [(12352, 2048), (24576, 4096)]:
        a_big, b = make_operands(512, N, K, "normal", dev, seed=0xBA5E + N)
        big = torch.matmul(a_big, b)
        torch.cuda.synchronize()
        for M in (1, 32):
            small = torch.matmul(a_big[:M].contiguous(), b)
            torch.cuda.synchronize()
            same = bitwise_equal(small, big[:M])
            ours_small = ops.run_plan(a_big[:M].contiguous(), b,
                                      seg.plan_for(M, K, N, torch.bfloat16))
            ours_big = ops.run_plan(a_big, b, seg.plan_for(512, K, N, torch.bfloat16))
            torch.cuda.synchronize()
            ours_same = bitwise_equal(ours_small, ours_big[:M])
            baseline.append({"N": N, "K": K, "M": M, "cublas_batch_invariant": same,
                             "ours_batch_invariant": ours_same})
            if not ours_same:
                failures.append(
                    f"batch composition (baseline section) N={N} K={K}: ours differs between "
                    f"M={M} and M=512")
            del small, ours_small, ours_big
        del a_big, b, big
    record["relations"]["baseline_batch_invariance"] = baseline
    broken = [r for r in baseline if not r["cublas_batch_invariant"]]
    print(f"[8] cuBLAS batch invariance: {len(broken)} of {len(baseline)} "
          f"(shape, M) pairs where cuBLAS row bits change with the batch; ours: "
          f"{sum(1 for r in baseline if not r['ours_batch_invariant'])}")
    for r in broken:
        print(f"      cuBLAS differs at N={r['N']} K={r['K']} M={r['M']} vs M=512")

    # ======================================================================================
    # The collectives entry points (README section 4.17). All four are EXECUTION modes: the
    # relations below prove that none of them moves a bit of the published program, and that the
    # values they expose are exactly the tree's own nodes.
    entry = collective_entry_points(torch, ops, seg, ext, dev, args.quick, failures)
    record["relations"].update(entry)

    record["failures"] = failures
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2)
    if failures:
        print(f"\nFAIL: {len(failures)} claims broken")
        for f in failures[:40]:
            print("  -", f)
        return 1
    print("\nPASS: every relation holds bitwise")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
