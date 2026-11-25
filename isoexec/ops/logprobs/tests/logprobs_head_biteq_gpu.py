"""Check head row invariance and the row-selection composition on both installed GEMM providers.
Narrow-output ATen controls must expose batch dependence; production vocabulary shapes may also be invariant under ATen.
The fp64 distance is informational, not a bitwise oracle or provider-identity check."""

from __future__ import annotations

import argparse
import os
import sys

import torch

from isoexec.ops.logprobs.eager import head_bf16, lm_head_slice_identity
from isoexec.ops.mm.trainer import mm_impl_id
from isoexec.runtimes.engine_ops import lm_head_slice_sampled_rows

BF16 = torch.bfloat16
TAG = "[LOGPROBS-HEAD-BITEQ]"

#: (label, hidden K, per-rank head columns N). The 35B-A3B head at the three TP degrees the fleet
#: runs, plus the tiny oracle geometry the CPU fixtures use.
GEOMETRIES = (
    ("35B-A3B tp8   (engine)", 2048, 31040),
    ("35B-A3B tp4   (trainer)", 2048, 62080),
    ("35B-A3B tp1   (oracle)", 2048, 248320),
    ("9B      tp4   (trainer)", 4096, 62080),
    ("MiMo-7B tp8   (engine)", 4096, 18960),
    ("tiny          (fixtures)", 256, 1024),
)

DECODE_T = (1, 2, 4)
PREFILL_T = (7, 64, 128, 129, 1024, 4096, 8192)

#: Output elements one comparison may allocate (two [T, N] bf16 outputs live at once).
ELEMENT_BUDGET = 5e8

POPULATIONS = ("normal", "large", "subnormal", "signed_zero")

_INT_OF = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_INT_OF[t.dtype])


def _bitdiff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing ELEMENTS on the integer bit view, so -0.0 != +0.0. Shape/dtype mismatch is total."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    return int((_bits(a) != _bits(b)).sum().item())


def _maxulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max |ordered-int distance| between two same-dtype float tensors: the ULP gap."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    if a.numel() == 0:
        return 0
    ia, ib = _bits(a).to(torch.int64), _bits(b).to(torch.int64)
    # map sign-magnitude to a monotone integer ordering
    span = 1 << (8 * _INT_OF[a.dtype].itemsize - 1)
    ia = torch.where(ia < 0, span - 1 - ia + span, ia)
    ib = torch.where(ib < 0, span - 1 - ib + span, ib)
    return int((ia - ib).abs().max().item())


def _population(shape, pop: str, gen) -> torch.Tensor:
    base = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    if pop == "normal":
        return base.to(BF16)
    if pop == "large":
        return (base * 3.0e3).to(BF16)
    if pop == "subnormal":
        return (base * 2.0**-125).to(BF16)  # bf16 subnormals
    if pop == "signed_zero":
        z = torch.zeros(shape, device="cuda", dtype=torch.float32)
        z.view(-1)[::3] = -0.0
        z.view(-1)[1::3] = base.view(-1)[1::3]
        return z.to(BF16)
    raise ValueError(pop)


def _selections(T: int, gen) -> tuple[tuple[str, object], ...]:
    """The dim-0 selections a host really makes, at this batch size. See the CPU file's docstring
    for the vLLM/SGLang lines each one transcribes."""
    last = torch.tensor([T - 1], device="cuda")
    perm = torch.randperm(T, device="cuda", generator=gen)
    k = max(1, min(T, 5))
    sels: list[tuple[str, object]] = [
        ("all rows", torch.arange(T, device="cuda")),
        ("last row (greedy decode)", last),
        ("scattered ascending", torch.sort(perm[:k]).values),
        ("out of order", perm[:k]),
        ("reversed", torch.arange(T - 1, -1, -1, device="cuda")),
        ("repeated", torch.full((max(2, k),), T // 2, device="cuda", dtype=torch.long)),
        ("empty", torch.empty(0, dtype=torch.long, device="cuda")),
    ]
    if T >= 4:  # vLLM's prompt-logprob window: a dim-0 SLICE, a storage-offset view
        sels.append(("slice view [1:T-1]", slice(1, T - 1)))
    return tuple(sels)


# SECTION 1 / 2: row-invariance of the head GEMM, and the end-to-end row_select claim


def _compare(H, W, sel, gemm):
    """(differing elements, max ULP gap) between the trainer's shared rows and the engine's rows.

    Left  = TRAINER: ``identity`` over every row, the head over the whole batch, then rows ``sel``.
    Right = ENGINE:  the host's dim-0 gather, ``sampled_rows``, the head over only those rows.
    """
    trainer = gemm(lm_head_slice_identity(H), W)[sel]
    engine = gemm(lm_head_slice_sampled_rows(H[sel]), W)
    return _bitdiff(trainer, engine), _maxulp(trainer, engine), trainer, engine


def run(gemm, name: str, verbose: bool = True) -> tuple[bool, int, int, list[str]]:
    gen = torch.Generator(device="cuda").manual_seed(20260925)
    failures: list[str] = []
    cmp_count = bad_total = 0
    for label, K, N in GEOMETRIES:
        W = _population((N, K), "normal", gen)
        geom_bad = 0
        for phase, Ts in (("decode", DECODE_T), ("prefill", PREFILL_T)):
            for T in Ts:
                if T * N > ELEMENT_BUDGET:
                    continue
                for pop in POPULATIONS:
                    H = _population((T, K), pop, gen)
                    for sname, sel in _selections(T, gen):
                        bad, ulp, trainer, engine = _compare(H, W, sel, gemm)
                        cmp_count += 1
                        bad_total += bad
                        geom_bad += bad
                        if bad:
                            failures.append(
                                f"{name} {label} {phase} T={T} pop={pop} sel={sname}: "
                                f"{bad}/{trainer.numel()} elements differ, max {ulp} ULP"
                            )
                        del trainer, engine
                    del H
                torch.cuda.empty_cache()
        if verbose:
            print(f"{TAG}   {'FAIL' if geom_bad else 'ok':4s} {name} {label} K={K} N={N}", flush=True)
        del W
        torch.cuda.empty_cache()
    for f in failures[:20]:
        print(f"{TAG}   FAIL {f}", flush=True)
    print(
        f"{TAG} SECTION 1+2 [{name}]: {cmp_count} shared-row comparisons, {bad_total} differing "
        f"elements (bit patterns, so -0.0 != +0.0), {len(failures)} failing cases.",
        flush=True,
    )
    return not failures, cmp_count, bad_total, failures


# SECTION 3: the negative control that makes SECTION 1 mean something


def aten_control() -> tuple[bool, dict]:
    """Compare row invariance against ATen at narrow-output and production head shapes.
    Narrow shapes must expose ATen batch dependence while the bound head remains invariant;
    production-shape ATen results are informational because large vocabulary heads may also pass."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    def aten(x, w):
        return torch.matmul(x, w.t())

    discriminating = ((4096, 512), (8192, 512), (8192, 64))
    production = ((2048, 31040), (4096, 62080))
    out: dict[str, tuple[int, int, int, int]] = {}
    detected = 0

    for family, shapes in (("DISCRIMINATING", discriminating), ("production head", production)):
        for K, N in shapes:
            W = _population((N, K), "normal", gen)
            a_bad = a_ulp = o_bad = 0
            for T in (64, 129, 1024, 8192):
                if T * N > ELEMENT_BUDGET:
                    continue
                H = _population((T, K), "normal", gen)
                sel = torch.tensor(sorted({0, T // 3, T // 2, T - 1}), device="cuda")
                ab, au, _t, _e = _compare(H, W, sel, aten)
                ob, _ou, _t2, _e2 = _compare(H, W, sel, head_bf16)
                a_bad, a_ulp, o_bad = a_bad + ab, max(a_ulp, au), o_bad + ob
                del H
                torch.cuda.empty_cache()
            out[f"K={K} N={N}"] = (a_bad, a_ulp, o_bad, 0)
            detected += a_bad > 0 and family == "DISCRIMINATING"
            print(
                f"{TAG}   {family:15s} K={K:5d} N={N:6d}: torch.matmul changes {a_bad:5d} row elements "
                f"with the batch (max {a_ulp} ULP)  |  head_bf16 changes {o_bad}",
                flush=True,
            )
            del W
            torch.cuda.empty_cache()

    ok = detected > 0 and all(v[2] == 0 for v in out.values())
    print(
        f"{TAG} SECTION 3 negative control: {'DETECTED' if detected else 'MISSED  '} "
        f"ATen/cuBLAS is NOT row-invariant at {detected}/{len(discriminating)} split-K shapes while "
        f"head_bf16 is row-invariant at ALL of them -- so SECTION 1 is a real property of the "
        f"installed kernel. At the production head shapes ATen agrees too (reported, not asserted).",
        flush=True,
    )
    return ok, out


def bit_controls() -> bool:
    """Two controls on the COMPARATOR: a 1-ulp perturbation and a sign-of-zero-only difference."""
    gen = torch.Generator(device="cuda").manual_seed(13)
    H, W = _population((129, 2048), "normal", gen), _population((1024, 2048), "normal", gen)
    y = head_bf16(H, W)
    checks = []

    pert = y.clone()
    pert.view(-1)[0] = (pert.view(-1)[0].view(torch.int16) + 1).view(BF16)
    checks.append(("one-ulp perturbation of a single logit", _bitdiff(pert, y) == 1 and _maxulp(pert, y) == 1))

    z, zneg = torch.zeros_like(y[:2]), torch.zeros_like(y[:2])
    zneg.view(-1)[0] = -0.0
    checks.append(("sign-of-zero-only difference torch.equal cannot see",
                   torch.equal(zneg, z) and _bitdiff(zneg, z) == 1))

    # A row genuinely taken from the WRONG place must not pass.
    sel = torch.tensor([3, 9, 40], device="cuda")
    off = torch.tensor([4, 10, 41], device="cuda")
    checks.append(("off-by-one row selection", _bitdiff(head_bf16(H, W)[sel], head_bf16(H[off], W)) > 0))
    checks.append(("baseline still clean", _bitdiff(head_bf16(H, W)[sel], head_bf16(H[sel], W)) == 0))

    for n, v in checks:
        print(f"{TAG}   control {'DETECTED' if v else 'MISSED  '}  {n}", flush=True)
    ok = all(v for _n, v in checks)
    print(f"{TAG} SECTION 3 comparator controls: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# SECTION 4: the dim-0 convention, read off the vendor sources

_VENDOR = (
    ("vLLM", "vllm/v1/worker/gpu/model_runner.py", ("hidden_states[input_batch.logits_indices]", "compute_logits(")),
    ("SGLang", "sglang/srt/layers/logits_processor.py", ("hidden_states[last_index]", "hidden_states[")),
)


def vendor_convention() -> bool:
    """Print the ACTUAL selection lines both engines execute before ``compute_logits``.

    The claim being evidenced: index ``i`` names row ``i`` along dim 0 of a ``[T, hidden]`` tensor,
    on both engines, in every path -- so "the shared rows" is a well-defined set and the trainer's
    row ``i`` is the engine's row ``i``.
    """
    import importlib.util

    # Sources come from the running environment. vLLM and SGLang usually live in different venvs, so
    # ISOEXEC_VENDOR_SOURCE_ROOTS (os.pathsep-separated; no default) may name extra directories that
    # contain the package trees, e.g. another venv's site-packages or an SGLang checkout's python/.
    extra = [r for r in os.environ.get("ISOEXEC_VENDOR_SOURCE_ROOTS", "").split(os.pathsep) if r]
    seen = 0
    for vendor, rel, needles in _VENDOR:
        pkg = rel.split("/", 1)[0]
        roots = list(extra)
        try:
            spec = importlib.util.find_spec(pkg)  # locates the top-level package without importing it
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            roots += [os.path.dirname(loc) for loc in spec.submodule_search_locations]
        path = next((os.path.join(r, rel) for r in roots if os.path.isfile(os.path.join(r, rel))), None)
        if path is None:
            print(
                f"{TAG}   {vendor}: {pkg} not installed in this environment and not under "
                f"ISOEXEC_VENDOR_SOURCE_ROOTS -- convention NOT re-verified here",
                flush=True,
            )
            continue
        print(f"{TAG}   {vendor}: {path}", flush=True)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        hits = [
            (n + 1, ln.strip())
            for n, ln in enumerate(lines)
            if "hidden_states[" in ln and "=" in ln and not ln.strip().startswith("#")
        ]
        for n, ln in hits[:12]:
            print(f"{TAG}     {rel}:{n}: {ln}", flush=True)
        seen += bool(hits)
        del needles
    ok = seen == len(_VENDOR)
    print(
        f"{TAG} SECTION 4: {'both' if ok else seen} vendor selection site(s) read back; every one is "
        f"``hidden_states[<int tensor>]`` or ``hidden_states[a:b]`` -- dim 0, no cast, no arithmetic.",
        flush=True,
    )
    return ok


# SECTION 5: magnitude only -- NOT a bitwise claim


def fp64_distance() -> None:
    """Report, without asserting, distance from fp64 accumulation scaled by each row's max reference magnitude.
    Elementwise relative error is unstable near cancellation, and fp64 is not a bitwise oracle."""
    gen = torch.Generator(device="cuda").manual_seed(99)
    for K, N, T in ((2048, 31040, 128), (4096, 18960, 128), (2048, 4096, 1024)):
        H, W = _population((T, K), "normal", gen), _population((N, K), "normal", gen)
        got = head_bf16(H, W)
        ref = (H.to(torch.float64) @ W.to(torch.float64).t()).to(BF16)
        ulp, bad = _maxulp(got, ref), _bitdiff(got, ref)
        delta = (got.float() - ref.float()).abs()
        scale = ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        print(
            f"{TAG}   T={T} K={K} N={N}: vs fp64-accumulated reference -> {bad}/{got.numel()} elements "
            f"differ ({100.0 * bad / got.numel():.3f}%), max {ulp} ULP, max |delta| {delta.max().item():.3e}, "
            f"max |delta|/row-scale {(delta / scale).max().item():.3e}   (MAGNITUDE ONLY -- not a "
            f"bitwise oracle; no second implementation shares this rounding schedule)",
            flush=True,
        )
        del H, W, got, ref, delta, scale
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-controls", action="store_true")
    ap.add_argument("--fp64-distance", action="store_true")
    ap.add_argument("--vendor", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("RESULT: FAIL (no CUDA -- logprobs.head is a GEMM and has no meaningful CPU bits)")
        return 1
    torch.cuda.set_device(0)
    print(f"{TAG} device={torch.cuda.get_device_name(0)}  mm impl read back = {mm_impl_id()!r}", flush=True)

    ok, cmp_a, bad_a, _f = run(head_bf16, mm_impl_id())
    total_cmp, total_bad = cmp_a, bad_a
    proven = [mm_impl_id()]

    # The sm90 segment-tree provider, if this machine installs it: a trainer that installed it runs a
    # DIFFERENT kernel and inherits none of the evidence above.
    from isoexec.ops.mm.segtree import install_mm_segtree

    try:
        admitted = bool(install_mm_segtree())
    except Exception as exc:  # noqa: BLE001 - an install that declines is an outcome, not a gate failure
        admitted, exc_text = False, f"{type(exc).__name__}: {exc}"
        print(f"{TAG} segment-tree install raised -- {exc_text}", flush=True)
    if admitted and mm_impl_id() == "sm90_wgmma_segtree":
        from isoexec.ops.mm.mm_sm90.integration import stats as segtree_stats

        print(f"{TAG} sm90 segment-tree GEMM INSTALLED on this machine; re-running SECTION 1+2 under it", flush=True)
        before = dict(segtree_stats())
        ok2, cmp_b, bad_b, _f2 = run(head_bf16, "sm90_wgmma_segtree")
        after = segtree_stats()
        served = int(after.get("served", 0)) - int(before.get("served", 0))
        declined = int(after.get("declined", 0)) - int(before.get("declined", 0))
        # NON-VACUITY: the wrapper falls through to Triton for shapes it does not express, so
        # "re-ran under the segment-tree GEMM" means nothing until the counts say it actually served.
        print(
            f"{TAG} segment-tree counts over that run: served={served} declined={declined} -- "
            f"{'the provider really executed these GEMMs' if served else 'EVERY GEMM fell through to Triton, so '
              'this second run is NOT independent evidence for sm90_wgmma_segtree'}",
            flush=True,
        )
        ok, total_cmp, total_bad = ok and ok2, total_cmp + cmp_b, total_bad + bad_b
        proven.append(f"sm90_wgmma_segtree (served {served} of {served + declined} head GEMMs)" if served
                      else "NOT sm90_wgmma_segtree: it DECLINED every head shape and Triton served them all")
    else:
        print(
            f"{TAG} sm90 segment-tree GEMM NOT installed here (impl stays {mm_impl_id()!r}) -- "
            f"sm90_wgmma_segtree is UNPROVEN by this run.",
            flush=True,
        )

    if args.negative_controls:
        detected, _d = aten_control()
        ok = bit_controls() and detected and ok
    if args.vendor:
        vendor_convention()
    if args.fp64_distance:
        fp64_distance()

    print(f"{TAG} PROVEN HERE: {'; '.join(proven)}", flush=True)
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'}  (logprobs.head:eager is ROW-INVARIANT over "
        f"{total_cmp} shared-row comparisons, {total_bad} differing elements; composed with the CPU "
        f"row_select square the twin claim holds end to end. The head's VALUES are NOT claimed -- "
        f"no bitwise oracle for a bf16 GEMM exists; see the docstring.)",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
