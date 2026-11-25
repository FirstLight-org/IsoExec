#!/usr/bin/env python
"""Check shared-expert row invariance on both GEMM providers and require non-vacuous provider service.
Reference differences are reported separately; controls must detect one-ULP, signed-zero, adjacent-row and batch-dependent changes.
Exit codes distinguish pass (0), mismatch (1), vacuous evidence (2) and unavailable environment (3)."""

from __future__ import annotations

import argparse
import os
import sys

# Set workspace defaults before torch import: negative-control GEMMs precede provider installation,
# and ATen caches these sizes on the first native GEMM.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal  # noqa: E402

BF16 = torch.bfloat16
TAG = "[SHARED-EXPERT-BITEQ]"

#: Qwen3.5-35B-A3B text_config: hidden_size / shared_expert_intermediate_size.
HIDDEN = 2048
SHARED_INTERMEDIATE = 512

#: (label, tp, per-rank shared intermediate). fc1 is [HIDDEN, 2*I]; fc2 reduces K=I.
GEOMETRIES = [
    ("35B tp1 (oracle/trainer)", 1, SHARED_INTERMEDIATE // 1),
    ("35B tp4 (trainer)", 4, SHARED_INTERMEDIATE // 4),
    ("35B tp8 (engine)", 8, SHARED_INTERMEDIATE // 8),
]

#: The full batch every row is measured against, and the prefixes it is re-measured at. 512/1024
#: straddle the pinned provider's decode/trainer M-bucket boundary.
FULL_T = 8192
PREFIX_T = (1, 2, 8, 129, 512, 1024, 4096)
#: Where in the batch a probed row sits: first, early, mid, last.
PROBE_ROWS = (0, 1, 7, 5000, FULL_T - 1)
#: How far the batch is rotated for the position-invariance arm.
ROLL = 5000

POPULATIONS = ("normal", "subnormal", "large", "signed_zero")

_INT_OF = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_INT_OF[t.dtype])


def _bitdiff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing elements on an integer bit view, or the larger numel if the tensors are not even
    comparable (a dtype or shape mismatch is a total divergence, not a skip)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    return int((_bits(a) != _bits(b)).sum().item())


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _population(shape, pop: str, dev, gen) -> torch.Tensor:
    base = torch.randn(shape, device=dev, dtype=torch.float32, generator=gen)
    if pop == "normal":
        return base.to(BF16)
    if pop == "subnormal":
        # Drives bf16 subnormals through the fc1 output; the count is asserted non-zero below.
        return (base * 2.0**-120).to(BF16)
    if pop == "large":
        # The shared-expert gate saturates to exactly 1.0 / 0.0 here.
        return (base * 1.0e4).to(BF16)
    if pop == "signed_zero":
        z = torch.zeros(shape, device=dev, dtype=torch.float32)
        z.view(-1)[::2] = -0.0
        z.view(-1)[1::4] = base.view(-1)[1::4]
        return z.to(BF16)
    raise ValueError(pop)


def _weights(inter: int, dev, seed: int):
    """``nn/moe.py::_Shared``'s four parameters at their registered shapes and held dtype (bf16)."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def w(*shape):
        return (torch.randn(*shape, device=dev, dtype=torch.float32, generator=g) * 0.02).to(BF16)

    return w(inter, HIDDEN), w(inter, HIDDEN), w(HIDDEN, inter), w(1, HIDDEN)


# claim 1 + 3: row / batch invariance


def _invariance(shared_fn, label: str, inter: int, pop: str, x_full, weights, gated: bool, failures: list):
    """Every way of computing row ``r`` must give the same bits as the full-batch forward."""
    gate, up, down, eg = weights
    eg = eg if gated else None
    y_full = shared_fn(x_full, gate, up, down, eg)
    cmps = 0
    bad = 0

    def cmp(what: str, got: torch.Tensor, want: torch.Tensor):
        nonlocal cmps, bad
        cmps += 1
        d = _bitdiff(got, want)
        if d:
            bad += d
            failures.append(
                f"{label} I={inter} {pop} gated={gated} {what}: {d} differing bits, "
                f"max|d|={_maxabs(got, want)}"
            )

    # prefixes, including the decode-bucket / trainer-bucket straddle
    for T in PREFIX_T:
        y = shared_fn(x_full[:T].contiguous(), gate, up, down, eg)
        cmp(f"prefix T={T}", y, y_full[:T])
        del y

    # the row alone (engine decode), and the row inside a batch of 8 at position 3
    for r in PROBE_ROWS:
        solo = shared_fn(x_full[r : r + 1].contiguous(), gate, up, down, eg)
        cmp(f"row {r} alone", solo[0], y_full[r])
        lo = max(0, min(r - 3, FULL_T - 8))
        b8 = shared_fn(x_full[lo : lo + 8].contiguous(), gate, up, down, eg)
        cmp(f"row {r} in batch of 8", b8[r - lo], y_full[r])
        del solo, b8

    # the row at a DIFFERENT position inside a full-size batch
    y_roll = shared_fn(torch.roll(x_full, ROLL, dims=0).contiguous(), gate, up, down, eg)
    for r in PROBE_ROWS:
        cmp(f"row {r} at position {(r + ROLL) % FULL_T}", y_roll[(r + ROLL) % FULL_T], y_full[r])
    del y_roll
    return y_full, cmps, bad


def _hazards(x_full, weights, dev) -> dict:
    """Witnesses that the declared populations actually reached the arithmetic."""
    from isoexec.core.linear import packed_view
    from isoexec.ops.mm.trainer import mm_trainer
    from isoexec.ops.moe.eager import shared_expert_gate_score

    gate, up, down, eg = weights
    inter = mm_trainer(x_full, packed_view(gate, up))
    tiny = torch.finfo(BF16).tiny
    subnormal = int(((inter.abs() > 0) & (inter.abs() < tiny)).sum().item())
    g = shared_expert_gate_score(x_full, eg)
    sat = int(((g == 1.0) | (g == 0.0)).sum().item())
    del inter, g
    return {"subnormal": subnormal, "saturated_gate": sat}


# claim 2: what the shipped impl and the reference actually claim about each other


def _chain(x, weights, *, fc1: str, epi: str, fc2: str, gsc: str) -> torch.Tensor:
    """A shared expert with each of the four steps taken from the SHIPPED side or the REFERENCE
    side, lifted to fp32 at the end so the two ends of the ablation are comparable at all.

    ``fc1="ship"/"ref"``   one packed ``mm_trainer`` GEMM over ``cat(gate, up)``  |  two ``ref_mm``s
    ``epi``                ``silu(g) * (u + 0.0)`` in bf16, two rounds            |  one fp32 round
    ``fc2``                ``row_parallel_trainer`` (pik 8-leaf tree, bf16 out)   |  one fp32 ``ref_mm``
    ``gsc``                bf16 ``sigmoid(mm_trainer(x, eg))``                    |  fp32 sigmoid
    """
    from isoexec.core.linear import packed_view
    from isoexec.ops.collectives.trainer import row_parallel_trainer
    from isoexec.ops.mlp.reference import ref_silu_mul
    from isoexec.ops.mm.reference import ref_mm
    from isoexec.ops.mm.trainer import mm_trainer

    gate, up, down, eg = weights
    if fc1 == "ship":
        inter = mm_trainer(x, packed_view(gate, up))
        a, b = torch.chunk(inter, 2, dim=-1)
        a, b = a.contiguous(), b.contiguous()
    else:
        a, b = ref_mm(x, gate), ref_mm(x, up)
    h = (F.silu(a) * (b + 0.0)) if epi == "ship" else ref_silu_mul(a, b)
    # The shipped fc2 returns bf16 (``out_dtype=x.dtype``); the reference casts its own to fp32.
    # Each step keeps its NATIVE dtype and the fp32 lift happens once, at the end -- otherwise the
    # all-shipped chain multiplies in fp32 and stops being the function under ablation.
    y = row_parallel_trainer(h, down) if fc2 == "ship" else ref_mm(h, down).to(torch.float32)
    if gsc == "ship":
        from isoexec.ops.moe.eager import shared_expert_gate_score

        s = shared_expert_gate_score(x, eg)  # bf16
    else:
        s = torch.sigmoid(x.to(torch.float32) @ eg.to(torch.float32).t())
    return (y * s).to(torch.float32)


def _reference_divergence(x, weights, dev, failures: list) -> dict:
    """Measure the shipped impl against ``ref_shared_expert`` and attribute the gap per step."""
    from isoexec.ops.moe.eager import shared_expert_megatron
    from isoexec.ops.moe.reference import ref_shared_expert

    gate, up, down, eg = weights
    ship = shared_expert_megatron(x, gate, up, down, eg)
    ref = ref_shared_expert(x, gate, up, down, eg)

    # STRUCTURAL, and asserted: the two sides do not agree on the output dtype, so `bitwise_equal`
    # is False before a single bit of arithmetic is compared. If this ever flips, the ablation
    # numbers below were taken against a different function and must be re-derived.
    if ship.dtype is not BF16:
        failures.append(f"shipped moe.shared_expert returned {ship.dtype}, expected bfloat16 (fc2 out_dtype=x.dtype)")
    if ref.dtype is not torch.float32:
        failures.append(f"ref_shared_expert returned {ref.dtype}, expected float32")
    if bitwise_equal(ship, ref):
        failures.append("shipped and reference compared BITWISE EQUAL; the recorded divergence below is stale")

    n = ship.numel()
    out = {
        "n": n,
        "dtype_ship": str(ship.dtype),
        "dtype_ref": str(ref.dtype),
        "as_is_biteq": bitwise_equal(ship, ref),
        "lifted_bitdiff": _bitdiff(ship.to(torch.float32), ref),
        "lifted_maxabs": _maxabs(ship, ref),
        "ref_rounded_bitdiff": _bitdiff(ship, ref.to(BF16)),
    }

    full_ref = _chain(x, weights, fc1="ref", epi="ref", fc2="ref", gsc="ref")
    if not bitwise_equal(full_ref, ref):
        failures.append("the all-reference ablation chain is not bitwise ref_shared_expert; the ablation is invalid")
    full_ship = _chain(x, weights, fc1="ship", epi="ship", fc2="ship", gsc="ship")
    if not bitwise_equal(full_ship, ship.to(torch.float32)):
        failures.append("the all-shipped ablation chain is not bitwise shared_expert_megatron; the ablation is invalid")

    steps = {}
    for name, cfg in (
        ("fc1 (packed mm_trainer GEMM)", dict(fc1="ship", epi="ref", fc2="ref", gsc="ref")),
        ("epilogue (bf16 two-round SwiGLU)", dict(fc1="ref", epi="ship", fc2="ref", gsc="ref")),
        ("fc2 (pik 8-leaf row-parallel tree)", dict(fc1="ref", epi="ref", fc2="ship", gsc="ref")),
        ("gate (bf16 sigmoid)", dict(fc1="ref", epi="ref", fc2="ref", gsc="ship")),
    ):
        y = _chain(x, weights, **cfg)
        steps[name] = (_bitdiff(y, full_ref), _maxabs(y, full_ref))
        del y
    out["steps"] = steps
    out["all_ship_vs_all_ref"] = (_bitdiff(full_ship, full_ref), _maxabs(full_ship, full_ref))
    del ship, ref, full_ref, full_ship
    return out


# negative controls


def _split_k_shared(x, gate, up, down, eg, *, splits_when_large: int = 4, threshold: int = 64):
    """A shared expert whose fc1 K-reduction is a bf16-leaf tree with ``splits_when_large`` leaves
    once M is large and ONE leaf when it is small -- pik's own reduction shape with the single thing
    pik fixes, the leaf count, made to depend on the batch. This is the canonical batch-variance
    failure mode written out, and the harness MUST reject it; otherwise the harness proves nothing."""
    W = torch.cat([gate, up], dim=0)
    s = splits_when_large if x.shape[0] > threshold else 1
    K = x.shape[1]
    step = K // s
    acc = torch.zeros(x.shape[0], W.shape[0], device=x.device, dtype=torch.float32)
    for i in range(s):
        lo, hi = i * step, (K if i == s - 1 else (i + 1) * step)
        # bf16 leaves folded in fp32 -- pik's own shape, with the leaf COUNT made M-dependent.
        acc = acc + (x[:, lo:hi].to(torch.float32) @ W[:, lo:hi].to(torch.float32).t()).to(BF16).to(torch.float32)
    inter = acc.to(BF16)
    a, b = torch.chunk(inter, 2, dim=-1)
    h = F.silu(a.contiguous()) * (b.contiguous() + 0.0)
    y = (h.to(torch.float32) @ down.to(torch.float32).t()).to(BF16)
    if eg is not None:
        y = y * torch.sigmoid((x.to(torch.float32) @ eg.to(torch.float32).t()).to(BF16))
    return y


def _stock_fp32_shared(x, gate, up, down, eg):
    """The same algebra in fp32 through STOCK torch GEMMs -- no batch-invariant override. Measured
    M-variant on this device, which is the reason the override exists."""
    xf = x.to(torch.float32)
    gu = xf @ torch.cat([gate, up], dim=0).to(torch.float32).t()
    a, b = torch.chunk(gu, 2, dim=-1)
    h = F.silu(a.contiguous()) * b.contiguous()
    y = h @ down.to(torch.float32).t()
    if eg is not None:
        y = y * torch.sigmoid(xf @ eg.to(torch.float32).t())
    return y


def _stock_bf16_shared(x, gate, up, down, eg):
    """The same algebra in BF16 through STOCK torch GEMMs. Reported, not a control: measured
    M-invariant on these shapes on this device, so it cannot witness anything."""
    gu = x @ torch.cat([gate, up], dim=0).t()
    a, b = torch.chunk(gu, 2, dim=-1)
    h = F.silu(a.contiguous()) * (b.contiguous() + 0.0)
    y = h @ down.t()
    if eg is not None:
        y = y * torch.sigmoid(x @ eg.t())
    return y


def _row_variance(fn, x_full, weights, rows=(0, 5000)) -> int:
    """Differing bits between each probed row computed alone and inside the full batch."""
    gate, up, down, eg = weights
    big = fn(x_full, gate, up, down, eg)
    bad = 0
    for r in rows:
        solo = fn(x_full[r : r + 1].contiguous(), gate, up, down, eg)
        bad += _bitdiff(solo[0], big[r])
        del solo
    del big
    return bad


def _negative_controls(x_full, weights, dev, verbose: bool) -> tuple[list[str], list[str]]:
    """Returns (lines to print, controls that did NOT fire)."""
    from isoexec.ops.moe.eager import shared_expert_megatron

    gate, up, down, eg = weights
    lines: list[str] = []
    dead: list[str] = []

    y_full = shared_expert_megatron(x_full, gate, up, down, eg)
    r = 5000

    # NC1: one-ulp perturbation of a single input element of row r.
    xp = x_full.clone()
    col = 17
    xp[r, col] = (_bits(xp[r, col : col + 1]) + 1).view(BF16)[0]
    solo_p = shared_expert_megatron(xp[r : r + 1].contiguous(), gate, up, down, eg)
    d1 = _bitdiff(solo_p[0], y_full[r])
    lines.append(f"NC1 one-ulp input perturbation (row {r}, col {col}): {d1} output bits move (want > 0)")
    if d1 == 0:
        dead.append("NC1 one-ulp perturbation produced an IDENTICAL output row")
    del xp, solo_p

    # NC2: a sign-of-zero-only difference. torch.equal cannot see it; bitwise_equal must.
    z_pos = torch.zeros(4, 8, device=dev, dtype=BF16)
    z_neg = z_pos.clone()
    z_neg[1, 3] = -0.0
    te, be = bool(torch.equal(z_pos, z_neg)), bitwise_equal(z_pos, z_neg)
    lines.append(f"NC2 sign-of-zero only: torch.equal={te} (blind), bitwise_equal={be} (want False)")
    if not (te and not be):
        dead.append(f"NC2 comparator did not separate -0.0 from +0.0 (torch.equal={te}, bitwise_equal={be})")

    # NC3: adjacent rows must differ, or the invariance check compares a constant to itself.
    d3 = _bitdiff(y_full[r], y_full[r + 1])
    lines.append(f"NC3 adjacent rows {r} vs {r + 1}: {d3} differing bits (want > 0)")
    if d3 == 0:
        dead.append("NC3 adjacent output rows are bit-identical; the invariance arm is comparing a constant")

    # NC4: an M-dependent split-K shared expert must fail row invariance.
    d4 = _row_variance(_split_k_shared, x_full, weights)
    lines.append(f"NC4 M-dependent split-K transcription: {d4} bits move between solo and batch (want > 0)")
    if d4 == 0:
        dead.append("NC4 the split-K transcription passed row invariance; the harness cannot see batch dependence")

    # NC5: stock fp32 GEMMs, no override. Real, and the reason the override exists.
    d5 = _row_variance(_stock_fp32_shared, x_full, weights)
    lines.append(f"NC5 stock-cuBLAS fp32 transcription: {d5} bits move between solo and batch (want > 0)")
    if d5 == 0:
        dead.append("NC5 stock fp32 GEMMs were row-invariant here; NC5 proves nothing on this device")

    # Reported, not a control: the bf16 stock path IS M-invariant on these shapes. Stated so the
    # reader does not mistake NC5's fp32 arm for a claim about bf16.
    d6 = _row_variance(_stock_bf16_shared, x_full, weights)
    lines.append(
        f"    (reported) stock-cuBLAS BF16 transcription: {d6} bits move -- "
        f"M-invariant on these shapes here, so it is not a control"
    )
    del y_full
    return lines, dead




def _battery(arm: str, verbose: bool) -> tuple[dict, list[str], dict]:
    """The invariance battery on the mm provider this process currently has installed."""
    from isoexec.ops.moe.eager import shared_expert_megatron

    dev = "cuda"
    failures: list[str] = []
    digests: dict = {}
    stats = {"cmps": 0, "bad": 0, "hazards": {"subnormal": 0, "saturated_gate": 0}}
    gen = torch.Generator(device=dev).manual_seed(20260925)

    for label, tp, inter in GEOMETRIES:
        weights = _weights(inter, dev, seed=1000 + tp)
        for pop in POPULATIONS:
            x_full = _population((FULL_T, HIDDEN), pop, dev, gen)
            for gated in (True, False):
                y_full, c, b = _invariance(
                    shared_expert_megatron, label, inter, pop, x_full, weights, gated, failures
                )
                stats["cmps"] += c
                stats["bad"] += b
                digests[(label, pop, gated)] = y_full
            h = _hazards(x_full, weights, dev)
            stats["hazards"]["subnormal"] += h["subnormal"] if pop == "subnormal" else 0
            stats["hazards"]["saturated_gate"] += h["saturated_gate"] if pop == "large" else 0
            del x_full
            torch.cuda.empty_cache()
        if verbose:
            mark = "FAIL" if any(f"I={inter}" in f for f in failures) else "ok"
            print(f"{TAG}   {mark:4s} {arm} {label} I={inter} fc1=[{HIDDEN},{2 * inter}] fc2 K={inter}", flush=True)
        del weights
    return digests, failures, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--model-path",
        default=None,
        help="what the GEMM shape census derives the shared-expert GEMM shapes from "
        "(default: Qwen/Qwen3.5-35B-A3B-Base from the local Hugging Face cache)",
    )
    args = ap.parse_args()
    verbose = not args.quiet
    if args.model_path is None:
        from isoexec.tests.hf_cache import cached_snapshot

        args.model_path = cached_snapshot("Qwen/Qwen3.5-35B-A3B-Base")
        if args.model_path is None:
            print(
                f"{TAG} ENVIRONMENT: no --model-path and Qwen/Qwen3.5-35B-A3B-Base config.json is not in the "
                "local Hugging Face cache (HF_HOME / HF_HUB_CACHE). A skip is not a pass.",
                flush=True,
            )
            return 3

    if not torch.cuda.is_available():
        print(f"{TAG} ENVIRONMENT: no CUDA device. A skip is not a pass.", flush=True)
        return 3
    try:
        from isoexec.ops.collectives.pik_bootstrap import ensure_pik

        ensure_pik()
    except Exception as e:  # pragma: no cover - environment
        print(f"{TAG} ENVIRONMENT: vendored pik unavailable ({type(e).__name__}: {e}); fc2 cannot run.", flush=True)
        return 3

    from isoexec.ops.mm.trainer import mm_impl_id

    dev = "cuda"
    print(f"{TAG} device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    # ---- arm A: whatever mm this process starts with (triton_batch_invariant) -------------------
    arm_a_id = mm_impl_id()
    print(f"{TAG} ARM A: mm={arm_a_id}", flush=True)
    digests_a, failures, stats_a = _battery(arm_a_id, verbose)

    # ---- claim 2: the reference, measured -------------------------------------------------------
    ref_lines: list[str] = []
    gen = torch.Generator(device=dev).manual_seed(4242)
    for label, tp, inter in GEOMETRIES:
        weights = _weights(inter, dev, seed=1000 + tp)
        x = _population((2048, HIDDEN), "normal", dev, gen)
        rep = _reference_divergence(x, weights, dev, failures)
        ref_lines.append(
            f"{TAG}   {label}: shipped {rep['dtype_ship']} vs reference {rep['dtype_ref']} -> "
            f"bitwise_equal={rep['as_is_biteq']} (dtype mismatch alone)"
        )
        ref_lines.append(
            f"{TAG}     lifted to fp32: {rep['lifted_bitdiff']}/{rep['n']} elements differ, "
            f"max|diff|={rep['lifted_maxabs']:.6g}; reference rounded to bf16: "
            f"{rep['ref_rounded_bitdiff']}/{rep['n']} differ"
        )
        for name, (d, m) in rep["steps"].items():
            ref_lines.append(f"{TAG}     step {name:36s} {d:9d}/{rep['n']} differ, max|diff|={m:.6g}")
        del x, weights
        torch.cuda.empty_cache()

    # ---- negative controls ---------------------------------------------------------------------
    weights = _weights(SHARED_INTERMEDIATE, dev, seed=1001)
    x_nc = _population((FULL_T, HIDDEN), "normal", dev, torch.Generator(device=dev).manual_seed(77))
    nc_lines, dead = _negative_controls(x_nc, weights, dev, verbose)
    del x_nc, weights
    torch.cuda.empty_cache()

    # ---- arm B: the production mm provider -------------------------------------------------------
    install_err = None
    try:
        from isoexec.runtimes.trainer_installs import install_trainer_kernels

        import json

        with open(f"{args.model_path}/config.json", encoding="utf-8") as fh:
            raw = json.load(fh)
        rec = install_trainer_kernels(args.model_path, source="isoexec.models.qwen3_5_moe", raw_config=raw)
    except Exception as e:
        install_err = f"{type(e).__name__}: {e}"
        rec = {}
    arm_b_id = mm_impl_id()
    provider_lines: list[str] = []
    stats_b = None
    if install_err is not None:
        print(f"{TAG} ENVIRONMENT: install_trainer_kernels refused ({install_err}).", flush=True)
        print(f"{TAG} The PRODUCTION mm provider was never measured. A skip is not a pass.", flush=True)
        return 3
    print(f"{TAG} ARM B: mm={arm_b_id} (mm segment-tree installed={rec.get('mm_segtree')})", flush=True)
    if arm_b_id == arm_a_id:
        dead.append(f"ARM B did not flip the mm provider (still {arm_b_id}); the two arms are the same measurement")
    digests_b, failures_b, stats_b = _battery(arm_b_id, verbose)
    failures.extend(failures_b)

    from isoexec.ops.mm.mm_sm90.integration import stats as segtree_stats

    served = int(segtree_stats().get("served", 0))
    declined = int(segtree_stats().get("declined", 0))
    cross_bad = 0
    cross_cmps = 0
    for key, ya in digests_a.items():
        yb = digests_b.get(key)
        cross_cmps += 1
        d = _bitdiff(yb, ya) if yb is not None else ya.numel()
        cross_bad += d
        if d:
            failures.append(
                f"mm provider split {arm_a_id} vs {arm_b_id} at {key}: {d} differing bits, "
                f"max|d|={_maxabs(yb, ya) if yb is not None else float('nan')}"
            )
    provider_lines.append(
        f"{TAG} {arm_a_id} vs {arm_b_id}: {cross_cmps} full-batch outputs compared, {cross_bad} differing bits; "
        f"segment-tree provider served={served} declined={declined}"
    )
    if served == 0:
        dead.append("the sm90 segment-tree provider served ZERO calls; the provider comparison is vacuous")

    # ---- report ---------------------------------------------------------------------------------
    print(f"\n{TAG} ---- claim 2: the reference is NOT a bitwise oracle (measured, not asserted) ----", flush=True)
    for ln in ref_lines:
        print(ln, flush=True)
    print(f"\n{TAG} ---- claim 3: the two mm providers ----", flush=True)
    for ln in provider_lines:
        print(ln, flush=True)
    print(f"\n{TAG} ---- negative controls ----", flush=True)
    for ln in nc_lines:
        print(f"{TAG}   {ln}", flush=True)

    vacuous: list[str] = list(dead)
    try:
        assert_hazard_exercised("bf16 subnormals in the fc1 output", stats_a["hazards"]["subnormal"])
        assert_hazard_exercised("saturated shared-expert gate", stats_a["hazards"]["saturated_gate"])
    except Exception as e:
        vacuous.append(str(e).splitlines()[0])

    total_cmps = stats_a["cmps"] + (stats_b["cmps"] if stats_b else 0)
    total_bad = stats_a["bad"] + (stats_b["bad"] if stats_b else 0)
    print(
        f"\n{TAG} claim 1: {total_cmps} row/batch-invariance comparisons over both mm providers, "
        f"{total_bad} differing bits (bit-pattern, so -0.0 != +0.0).",
        flush=True,
    )
    print(
        f"{TAG} hazards: {stats_a['hazards']['subnormal']} bf16 subnormals in fc1 output, "
        f"{stats_a['hazards']['saturated_gate']} saturated gate elements.",
        flush=True,
    )
    for f in failures:
        print(f"{TAG}   FAIL {f}", flush=True)
    for v in vacuous:
        print(f"{TAG}   VACUOUS {v}", flush=True)

    if failures:
        print(
            "RESULT: FAIL  (moe.shared_expert:eager row/batch invariance or mm-provider bit-equality broke)",
            flush=True,
        )
        return 1
    if vacuous:
        print("RESULT: VACUOUS  (the checks passed but a control or hazard never fired; not a pass)", flush=True)
        return 2
    print(
        "RESULT: PASS  (moe.shared_expert:eager is row/batch invariant bitwise at the 35B-A3B "
        "per-rank widths on BOTH mm providers, and the two providers agree bit for bit; the "
        "plain-torch reference is NOT bitwise with it and the gap is reported above)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
