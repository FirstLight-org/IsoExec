#!/usr/bin/env python
"""GPU battery: the Blackwell tcgen05 K-reduction order does not move with the tile config.

``pik/arch.py`` declares ``BLACKWELL = ArchProfile(..., verified=True)``: inside a non-split-K GEMM the
fp32 accumulation order along K is a fixed sequential chain of MMA k-steps, independent of
BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages, so the autotuner over ``pik.arch._BLACKWELL_CONFIGS`` can
change speed but never bits. ``pik.arch._verify()`` checks that premise with a small kernel over a
generic config grid; this battery checks it on the real leaf-partial kernel
(``pik.gemm._leaf_partial_kernel``, the building block every row-parallel schedule folds). Every entry
of the Blackwell config list is forced in turn (the autotuner is bypassed by calling the underlying
``@triton.jit`` function with pinned meta-parameters) and compared bitwise against an independent
backend's leaf -- the sm100 segment-tree GEMM run as one segment with an fp32 root (``ops/mm/mm_sm100``,
whose single chain is the tensor core's canonical one: ``test_unsplit_weights_are_bit_identical_to_cublas``)
-- and against every other config, at the leaf widths of the shipped models' row-parallel projections.

    CUDA_VISIBLE_DEVICES=<one B200> python isoexec/ops/collectives/tests/pik_blackwell_config_invariance_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from isoexec.ops.collectives.pik_bootstrap import ensure_pik  # noqa: E402

# Leaf widths (K / 8 leaves) of the shipped models' row-parallel projections: Qwen3.5-35B-A3B shared
# fc2 (64) and o_proj (512), Qwen1.5-MoE-A2.7B o_proj (256) and shared fc2 (704), Qwen3.5-0.8B o_proj
# (256) and down_proj (448), Qwen3.5-9B o_proj (512) and down_proj (1536); 128 and 1024 make every
# BLOCK_K in the list legal at least once.
LEAF_K_POINTS = (1536, 1024, 704, 512, 448, 256, 128, 64)
M_POINTS = (1, 8, 230, 4096)
N = 512


def _adversarial(shape, seed, dev):
    """Wide exponent spread (as pik.arch._verify uses): any reordering shows up loudly."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(*shape, generator=g) * torch.exp(torch.randn(*shape, generator=g) * 4)
    return a.to(dev).bfloat16()


def _leaf_partial_forced(x, w, leaf_k, cfg, dev):
    """One leaf partial (NUM_LOCAL_LEAVES=1) at a pinned config, bypassing the autotuner and its cache."""
    import triton

    import pik.gemm as pg

    M, n = x.shape[0], w.shape[0]
    bm, bn, bk, warps, stages = cfg
    p = torch.empty(1, M, n, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, bm) * triton.cdiv(n, bn), 1)
    pg._leaf_partial_kernel.fn[grid](
        x, w, p, M, n, leaf_k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), p.stride(0), p.stride(1), p.stride(2),
        NUM_LOCAL_LEAVES=1, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return p[0]


def _reference_leaf(x, w, dev):
    """The fp32 leaf from the sm100 segment-tree GEMM: one segment, ascending tcgen05 chain, fp32 root.
    ``None`` where no publishing plan expresses (M, leaf_k, N)."""
    from isoexec.ops.mm.mm_sm100 import ops as sm100
    from isoexec.ops.mm.mm_sm100 import segments as seg100

    M, K = x.shape
    n = w.shape[0]
    for plan in seg100.plans_at_split(M, K, n, 1):
        try:
            return sm100.run_plan(x, w.t(), plan, root_dtype=torch.float32).clone()
        except Exception:  # noqa: BLE001 -- a plan that cannot publish this M is not a reference
            continue
    return None


def main() -> int:
    ensure_pik()
    from pik.arch import BLACKWELL, _verify, current

    dev = torch.device("cuda", 0)
    arch = current(dev)
    results: dict[str, bool] = {"arch_self_check": _verify(dev)}
    if arch.name != "blackwell":
        print(f"[config-invariance] SKIP: arch={arch.name}; this battery covers the Blackwell config list", flush=True)
        return 0 if results["arch_self_check"] else 1
    from isoexec.ops.mm.mm_sm100 import ops as sm100

    sm100.require_sm100()  # the reference backend; refusing here is a FAIL, not a skip

    configs = list(BLACKWELL.configs)
    covered = set()
    for leaf_k in LEAF_K_POINTS:
        legal = [c for c in configs if leaf_k % c[2] == 0]
        covered.update(legal)
        for M in M_POINTS:
            x = _adversarial((M, leaf_k), 1000 + M, dev)
            w = _adversarial((N, leaf_k), 2000 + leaf_k, dev)
            ref = _reference_leaf(x, w, dev)
            results[f"M{M}_leafk{leaf_k}_reference_exists"] = ref is not None
            if ref is None:
                print(f"[config-invariance] FAIL M={M} leaf_k={leaf_k}: no sm100 reference plan", flush=True)
                continue
            base = None
            for cfg in legal:
                got = _leaf_partial_forced(x, w, leaf_k, cfg, dev)
                base = got if base is None else base
                ok = bool(torch.equal(got.view(torch.int32), ref.view(torch.int32))) and bool(
                    torch.equal(got.view(torch.int32), base.view(torch.int32))
                )
                results[f"M{M}_leafk{leaf_k}_cfg{cfg}"] = ok
                if not ok:
                    print(f"[config-invariance] FAIL M={M} leaf_k={leaf_k} cfg={cfg}", flush=True)
    results["every_config_exercised"] = covered == set(configs)

    bad = sorted(k for k, v in results.items() if not v)
    print(f"[config-invariance] arch={arch} RESULT: {'PASS' if not bad else 'FAIL'} "
          f"({len(results)} checks, {len(configs)} configs) {bad}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
