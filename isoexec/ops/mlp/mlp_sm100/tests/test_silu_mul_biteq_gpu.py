from __future__ import annotations

import sys

import torch

from isoexec.ops.mlp.eager import silu_mul_megatron
from isoexec.ops.mlp.mlp_sm100.ops import silu_mul_fused

BF16 = torch.bfloat16
GEOMETRIES = [("mimo-7B-dense", 11008), ("35B-A3B-expert", 512), ("ragged", 131)]
ROWS = (1, 2, 7, 129, 4096)
POPULATIONS = ("normal", "subnormal", "large", "signed_zero", "special")

def _bits(t):
    return t.contiguous().view(torch.int16)


def _bitdiff(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    return int((_bits(a) != _bits(b)).sum().item())


def _population(shape, pop, dev, gen):
    base = torch.randn(shape, device=dev, dtype=torch.float32, generator=gen)
    if pop == "normal":
        return base.to(BF16)
    if pop == "subnormal":
        return (base * 2.0**-120).to(BF16)
    if pop == "large":
        return (base * 30.0).to(BF16)  # SiLU saturates towards x for large positive x
    if pop == "signed_zero":
        z = torch.zeros(shape, device=dev, dtype=torch.float32)
        z.view(-1)[::2] = -0.0
        z.view(-1)[1::4] = base.view(-1)[1::4]
        return z.to(BF16)
    if pop == "special":
        pool = torch.tensor([float("inf"), float("-inf"), float("nan"), 3.3895e38, -3.3895e38, 1e-40,
                             -1e-40, 0.0, -0.0, 88.7, -88.7, 87.3, -87.3, 103.9, -103.9, 1.0, -1.0],
                            device=dev, dtype=torch.float32)
        idx = torch.randint(0, pool.numel(), shape, device=dev, generator=gen)
        return torch.where(torch.rand(shape, device=dev, generator=gen) < 0.5, pool[idx], base).to(BF16)
    raise ValueError(pop)


def _one_round_transcription(g, u):
    gf = g.to(torch.float32)
    uf = u.to(torch.float32)
    sig = gf / (1.0 + torch.exp(-gf))
    return (sig * uf).to(BF16)


def _production_shape_relation(dev, gen, failures: list[str]) -> int:
    """Check the shipped plan across decode/prefill shapes and scalar fallback."""
    pairs = 0
    for width in (131, 192, 512, 1536, 3072):
        for n in (1, 256, 1024, 4096):
            g = _population((n, width), "normal", dev, gen)
            u = _population((n, width), "large", dev, gen)
            bad = _bitdiff(silu_mul_fused(g, u), silu_mul_megatron(g, u))
            pairs += 1
            if bad:
                failures.append(f"production shape w={width} n={n}: {bad} bits differ")
    return pairs



def run() -> bool:
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(20260930)
    total_cmp = total_bad = 0
    onepass_diff_cases = 0
    failures: list[str] = []

    for label, width in GEOMETRIES:
        for n in ROWS:
            for gpop in POPULATIONS:
                g = _population((n, width), gpop, dev, gen)
                for upop in POPULATIONS:
                    u = _population((n, width), upop, dev, gen)
                    ref = silu_mul_megatron(g, u)
                    got = silu_mul_fused(g, u)
                    bad = _bitdiff(got, ref)
                    total_cmp += 1
                    total_bad += bad
                    if bad:
                        failures.append(f"{label} n={n} g={gpop} u={upop}: {bad} bits differ")
                    if _bitdiff(_one_round_transcription(g, u), ref):
                        onepass_diff_cases += 1
                del u
            del g

    shape_pairs = _production_shape_relation(dev, gen, failures)

    print(f"[MLP-SILU-MUL-BITEQ] {total_cmp} comparisons, {total_bad} differing elements, "
          f"{len(failures)} failing cases.", flush=True)
    print(f"[MLP-SILU-MUL-BITEQ] production shapes: {shape_pairs} bitwise comparisons.", flush=True)
    print(f"[MLP-SILU-MUL-BITEQ] the fp32-single-round transcription differs from the eager program "
          f"in {onepass_diff_cases}/{total_cmp} cases -- this is WHY the kernel keeps two roundings.",
          flush=True)
    for f in failures:
        print(f"[MLP-SILU-MUL-BITEQ]   FAIL {f}", flush=True)
    ok = not failures and total_bad == 0 and onepass_diff_cases > 0 and shape_pairs > 0
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  (mlp_sm100.silu_mul_fused == eager silu_mul_megatron bitwise)",
          flush=True)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("RESULT: SKIP (no CUDA)")
        return 0
    from isoexec.core.arch import live_device_arch

    arch = live_device_arch()
    if arch != "sm100":
        print(f"RESULT: SKIP (live arch is {arch!r}, not sm100)")
        return 0
    return 0 if run() else 1


if __name__ == "__main__":
    sys.exit(main())
