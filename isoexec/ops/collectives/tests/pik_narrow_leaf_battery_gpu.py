#!/usr/bin/env python
"""L1 GPU battery for the shared expert's fc2 K-reduction at every ``fc2_leaves_local``.

The region: ``moe.shared_expert``'s ``collectives.row_parallel`` seam -- ``pik.linear.row_parallel_linear``
(``ti_gemm`` over this rank's K-shard, then the cross-rank tree). ``fc2_leaves_local = leaves // etp``
selects the schedule: 8/4/2 are Triton leaf-tree kernels or the sm90 segment-tree producer, 1 short-circuits
to the sm90 producer's single leaf.
The contract says all four evaluate the SAME expression tree, so all four must be bitwise equal.

Emulated in ONE process: rank r's ``ti_gemm`` on K-slice r, folded with pik's own ``combine_order`` tree
in fp32 -- which is what ``row_parallel_linear`` + ``tree_all_reduce`` do, minus the transport. No
collective is under test here; the leaf GEMM is.

Two geometries, both real:
  h8   H=256,  shared I=128  -> leaf_k = 16   (the tiny MoE oracle checkpoints; the narrow-leaf case)
  b35  H=2048, shared I=512  -> leaf_k = 64   (the 35B production shape)

Run from the repo checkout with the managed runtime's Python (``make setup`` creates it under
``$ISOEXEC_ROOT``, default ``~/.isoexec``; ``make setup-dev`` uses ``~/.isoexec-dev``)::

    $ISOEXEC_ROOT/venv/bin/python isoexec/ops/collectives/tests/pik_narrow_leaf_battery_gpu.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 7))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from isoexec.ops.collectives.pik_bootstrap import ensure_pik  # noqa: E402

# (name, T, H, I): the shared-expert fc2 is [T, I] x [H, I]^T, so I is the reduced axis.
GEOMETRIES = {
    "h8": (230, 256, 128),
    "b35": (230, 2048, 512),
}
LEAVES = 8
TP_POINTS = (1, 2, 4, 8)


def _fold(parts, combine_order):
    slots = [p.float() for p in parts]
    for dst, lhs, rhs in combine_order(len(slots)):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--ulps",
        type=float,
        default=4.0,
        help="fp32-reference bound in bf16 ulps of the largest output (default 4: the plan rounds "
        "each of its leaves once, and 8 leaves of alternating sign cannot exceed a few ulps of the root)",
    )
    a = ap.parse_args()

    ensure_pik()
    import pik.gemm as pg
    from pik.plan import ReductionPlan, combine_order

    dev = torch.device("cuda", 0)
    plan = ReductionPlan(num_leaves=LEAVES, leaf_dtype=torch.bfloat16)
    results: dict[str, bool] = {}

    for name, (T, H, I) in GEOMETRIES.items():
        leaf_k = I // LEAVES
        torch.manual_seed(a.seed)
        h = (torch.randn(T, I, device=dev) * 0.5).bfloat16()  # the glu output
        down = (torch.randn(H, I, device=dev) * 0.05).bfloat16()
        ref = h.float() @ down.float().t()
        # bf16 has 8 mantissa bits, so one ulp at the root is 2**-8 of its magnitude.
        bound = a.ulps * (2.0**-8) * ref.abs().max().item()

        # (1) every fc2 leaf count is BITWISE equal to the others, and close to the fp32 reference.
        base = None
        for tp in TP_POINTS:
            pg._SCHED_CACHE.clear()
            kl = I // tp
            parts = [
                pg.ti_gemm(
                    h[:, r * kl : (r + 1) * kl].contiguous(),
                    down[:, r * kl : (r + 1) * kl].contiguous(),
                    plan=plan,
                    tp_size=tp,
                    tp_rank=r,
                    k_full=I,
                )
                for r in range(tp)
            ]
            y = _fold(parts, combine_order).to(torch.bfloat16)
            m = LEAVES // tp
            d_ref = (y.float() - ref).abs().max().item()
            if base is None:
                base = y
                bit = True
            else:
                bit = bool(torch.equal(y, base))
            nz = int((y != 0).sum())
            ok = bit and d_ref <= bound and nz > 0
            results[f"{name}_fc2_leaves_local_{m}"] = ok
            print(
                f"[pik-narrow-leaf] {name} leaf_k={leaf_k} tp={tp} fc2_leaves_local={m}: "
                f"bitwise_vs_fc2_leaves_local_{LEAVES}={bit} max|d| vs fp32={d_ref:.6g} (bound {bound:.6g}) "
                f"nonzero={nz}/{y.numel()} -> {'PASS' if ok else 'FAIL'}",
                flush=True,
            )

        # (2) every SCHEDULE at the widest leaf count is bitwise equal to the sm90 segment-tree producer.
        #     This is the assertion the zero-trip k-loop broke: `fused`/`split*` returned exact zeros.
        outs = {}
        for sched in ("sm90seg", "fused", "split1", "split2", "split4", "splitk"):
            out = torch.empty((T, H), device=dev, dtype=torch.float32)
            try:
                pg._dispatch(h, down, out, leaf_k, LEAVES, sched, torch.bfloat16)
            except Exception as e:  # noqa: BLE001 -- a refused schedule is reported, not swallowed
                print(f"[pik-narrow-leaf] {name} sched={sched}: REFUSED {type(e).__name__}: {e}", flush=True)
                continue
            outs[sched] = out.clone()
        cb = outs.get("sm90seg")
        assert cb is not None, f"{name}: the sm90 segment-tree producer is unavailable; the battery needs it as the reference"
        for sched, v in outs.items():
            ok = bool(torch.equal(v, cb))
            results[f"{name}_sched_{sched}_biteq_sm90seg"] = ok
            print(
                f"[pik-narrow-leaf] {name} sched={sched:8s}: bitwise==sm90seg {ok} "
                f"max|d|={(v - cb).abs().max().item():.6g} -> {'PASS' if ok else 'FAIL'}",
                flush=True,
            )

    bad = sorted(k for k, v in results.items() if not v)
    print(f"[pik-narrow-leaf] RESULT: {'PASS' if not bad else 'FAIL'} ({len(results)} checks) {bad}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
