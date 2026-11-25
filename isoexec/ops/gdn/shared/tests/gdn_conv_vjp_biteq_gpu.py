"""The bitwise gate that licenses the fused Triton conv VJP: kernels == the eager reference.

``ops/gdn/shared/gdn_conv_vjp_triton.py`` replaces the analytic causal-conv1d VJP's per-tap ATen chain
(``index_select`` + fp32 ``mul`` + ``add``, 58 launches and ~5.5 ms per GDN layer of a 9B trainer
step) with three Triton kernels that keep every per-tap intermediate in registers. The kernels are
bit-neutral BY CONSTRUCTION -- same ascending tap order, same fp32 accumulator, same rounding
points, ``sum(dim=0)`` still owned by ATen -- and this file is the proof of that construction on
the shapes the models actually run.

WHAT IS COMPARED: ``_native_conv_vjp`` (the CUDA dispatch, i.e. the kernels) against
``_native_conv_vjp_eager`` (the reference), as raw BIT PATTERNS -- the tensors viewed as integers,
which is strictly stronger than ``torch.equal``: ``torch.equal`` calls -0.0 == +0.0, and a
sign-of-zero divergence is the first thing a re-associated accumulator produces. Zero differing
elements passes; anything else fails.

WHY THESE CASES:

  * **Real geometry.** ``conv.short`` runs on the q|k|v concat, so the 9B's conv is
    ``[3584, 8192]`` with width 4 (16x128 + 16x128 + 32x128 channels, ``linear_conv_kernel_dim 4``),
    and the 35B's per-rank conv is the same shape divided by TP. Both are here; so is the tiny
    oracle's narrow conv.
  * **Sequence packs.** 1, 3 and 8 sequences, plus length-1 (a decode-shaped row) and length-0
    sequences at the front, middle and back -- ``cu_seqlens`` with repeated bounds is what makes
    ``searchsorted`` pick a different sequence for the same token, and length-0 is the only case
    where a token's ``[start, end)`` window is empty.
  * **Row chunking.** ``ISOEXEC_GDN_CONV_BWD_ROWS`` sets the fold granularity of the ``dw``
    and ``db`` sums, so a multi-chunk run exercises a different reduction fold than a single-chunk
    one. Both are here.
  * **Hazards.** Signed zeros (the masked-tap terms are ``x * 0.0``, whose sign follows ``x``),
    subnormals (fp32 flush-to-zero would show here), and wide exponents (where an FMA contraction
    inside the accumulator would show as a 1-ULP divergence) -- run BOTH at the 23x11 toy shape and
    at every production width, because the width is what decides how many elements sample a rare
    band. The 23x11 wide_exponent case is 253 elements; the 9B width is 29.4M, and the FTZ divide
    this file failed to catch first showed up at 650 of those 29.4M.
  * **The sigmoid's subnormal window.** ``d(silu)/dz`` calls ``sigmoid(z)``, and for ``z`` in
    ``(-88.7228, -87.3345)`` ATen's ``1/(1+expf(-z))`` returns an fp32 SUBNORMAL. ``libdevice.div_rn``
    returns ``+0.0`` there instead -- Triton links libdevice with ``__CUDA_FTZ`` set -- which
    collapses the derivative to a signed zero and drops a whole tap from ``dx``. That is the defect
    ``core.triton_nonftz.sigmoid`` fixes. Two gates pin it: ``_gate_sigmoid`` sweeps the kernels'
    sigmoid itself against ``torch.sigmoid`` over the window, over every mantissa of the surrounding
    exponent band, and over the subnormals and specials; ``_gate_sigmoid_window`` drives the whole
    VJP with preactivations placed IN the window at every production width and asserts a non-zero
    hazard population, so the coverage cannot silently lapse.
  * **Batch invariance.** A token's gradient may not depend on which other sequences share its
    pack: the same tokens are run alone and inside a larger pack, and the rows must be identical.

Run: one GPU, as a file.
    python -m isoexec.ops.gdn.shared.tests.gdn_conv_vjp_biteq_gpu
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import torch
import triton
import triton.language as tl

from isoexec.core.triton_nonftz import (
    sigmoid as _nonftz_sigmoid,
)
from isoexec.ops.gdn.shared import gdn_conv_vjp_triton as FUSED
from isoexec.ops.gdn.shared import gdn_ops as G

DEV = "cuda"
_ROWS_ENV = "ISOEXEC_GDN_CONV_BWD_ROWS"

# (label, tokens, channels, taps) -- the conv widths the models run, per rank. `conv.short` runs on
# the q|k|v concat, so the width is `2 * key_heads * key_head_dim + value_heads * value_head_dim`
# divided by TP. Taken from the checkpoints' own config.json; two of these were mislabelled when the
# file landed (tiny was 1024 and 35B TP=8 was 2048, i.e. the two widths the models really run were
# the two absent from the battery), which is why the derivation is written out per row.
GEOMETRIES = [
    ("qwen35_9b TP=1", 3584, 8192, 4),  # (16*128 q) + (16*128 k) + (32*128 v) = 8192
    ("qwen35_35b TP=8", 3584, 1024, 4),  # the same 8192 concat divided by 8 ranks
    ("qwen35_35b TP=4", 3584, 2048, 4),  # ...and by 4
    ("tiny_qwen35_moe", 512, 512, 4),  # (4*32 q) + (4*32 k) + (8*32 v) = 512
]

# (label, cu_seqlens) over 23 tokens; the narrow packs run on the tiny geometry only.
PACKS = [
    ("1 seq", [0, 23]),
    ("3 seq", [0, 5, 9, 23]),
    ("8 seq", [0, 1, 3, 6, 10, 11, 15, 19, 23]),
    ("length-1 rows", [0, 1, 2, 3, 23]),
    ("length-0 middle", [0, 3, 3, 10, 10, 23]),
    ("length-0 first", [0, 0, 23]),
    ("length-0 last", [0, 23, 23]),
]

HAZARDS = ("uniform", "wide_exponent", "signed_zero", "subnormal")

# fp32's smallest NORMAL. `sigmoid(z)` below this and above zero is a subnormal, which is exactly
# what a flush-to-zero divide turns into `+0.0`; the band is `z` in about (-88.7228, -87.3345).
_FP32_MIN_NORMAL = 2.0**-126


class Ledger:
    """One row per comparison: the differing-element count on the raw bit patterns."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, int]] = []

    def add(self, case: str, tag: str, reference: torch.Tensor | None, got: torch.Tensor | None) -> None:
        if reference is None or got is None:
            self.rows.append((case, tag, 0 if (reference is None and got is None) else 1))
            return
        bits = torch.int32 if reference.dtype == torch.float32 else torch.int16
        differ = int((reference.view(bits) != got.view(bits)).sum())
        self.rows.append((case, tag, differ))

    @property
    def ok(self) -> bool:
        return all(d == 0 for _, _, d in self.rows)

    def report(self) -> str:
        lines = [f"{'case':<52} {'tensor':<4} {'differing bit patterns':>22}"]
        for case, tag, differ in self.rows:
            flag = "" if differ == 0 else "   <-- MISMATCH"
            lines.append(f"{case:<52} {tag:<4} {differ:>22d}{flag}")
        return "\n".join(lines)


def _inputs(tokens: int, channels: int, taps: int, dtype: torch.dtype, hazard: str, seed: int):
    """``(x, dy, weight)`` for one case. ``hazard`` selects the exponent/zero population."""
    g = torch.Generator(device=DEV).manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)

    if hazard == "wide_exponent":
        x = normal((tokens, channels)) * torch.exp2(
            torch.randint(-40, 40, (tokens, channels), generator=g, device=DEV).float()
        )
        dy = normal((tokens, channels)) * torch.exp2(
            torch.randint(-40, 40, (tokens, channels), generator=g, device=DEV).float()
        )
    elif hazard == "signed_zero":
        # Two thirds of the rows are +/-0.0, so most masked taps multiply a signed zero.
        sign = torch.randint(0, 2, (tokens, channels), generator=g, device=DEV).float() * 2.0 - 1.0
        x = torch.zeros((tokens, channels), device=DEV) * sign
        x[::3] = normal((x[::3].shape[0], channels))
        dy = normal((tokens, channels))
    elif hazard == "subnormal":
        x = normal((tokens, channels)) * 1e-42
        dy = normal((tokens, channels)) * 1e-40
    else:
        x = normal((tokens, channels))
        dy = normal((tokens, channels))
    weight = normal((channels, taps))
    return x.to(dtype), dy.to(dtype), weight.to(dtype)


def _compare(led: Ledger, case: str, dy, x, weight, bias, cu, activation, chunk_rows: int) -> None:
    os.environ[_ROWS_ENV] = str(chunk_rows)
    with torch.no_grad():
        reference = G._native_conv_vjp_eager(dy, x, weight, bias, cu, activation)
        got = G._native_conv_vjp(dy, x, weight, bias, cu, activation)
    for tag, ref_t, got_t in zip(("dx", "dw", "db"), reference, got):
        led.add(case, tag, ref_t, got_t)


def _cu(bounds: list[int]) -> torch.Tensor:
    return torch.tensor(bounds, dtype=torch.int32, device=DEV)


def _gate_packs(led: Ledger, quick: bool) -> None:
    """Sequence packs x dtype x activation x bias x chunking, on the narrow (tiny) conv."""
    packs = PACKS[:2] if quick else PACKS
    grid = list(itertools.product(packs, (torch.bfloat16, torch.float32), (None, "silu"), (False, True), (8192, 4)))
    for (plabel, bounds), dtype, activation, with_bias, chunk_rows in grid:
        x, dy, weight = _inputs(23, 11, 4, dtype, "uniform", seed=hash(plabel) & 0xFFFF)
        bias = torch.randn(11, device=DEV, dtype=torch.float32).to(dtype) if with_bias else None
        case = (
            f"pack {plabel} / {str(dtype).removeprefix('torch.')} / act={activation} / "
            f"bias={int(with_bias)} / rows={chunk_rows}"
        )
        _compare(led, case, dy, x, weight, bias, _cu(bounds), activation, chunk_rows)


def _gate_hazards(led: Ledger, quick: bool) -> None:
    hazards = HAZARDS[:2] if quick else HAZARDS
    for hazard, dtype in itertools.product(hazards, (torch.bfloat16, torch.float32)):
        x, dy, weight = _inputs(23, 11, 4, dtype, hazard, seed=99)
        bias = torch.randn(11, device=DEV, dtype=torch.float32).to(dtype)
        label = f"hazard {hazard} / {str(dtype).removeprefix('torch.')}"
        _compare(led, label, dy, x, weight, None, _cu([0, 5, 9, 23]), "silu", 8192)
        _compare(led, f"{label} / bias", dy, x, weight, bias, _cu([0, 5, 9, 23]), "silu", 8192)


def _gate_geometries(led: Ledger, quick: bool) -> None:
    """The production convs, on the packed batch shape the trainer benches (512/1024/2048)."""
    geometries = GEOMETRIES[:1] if quick else GEOMETRIES
    for glabel, tokens, channels, taps in geometries:
        bounds = [0, tokens // 7, tokens // 7 + 2 * (tokens // 7), tokens]
        x, dy, weight = _inputs(tokens, channels, taps, torch.bfloat16, "uniform", seed=20260903)
        _compare(led, f"{glabel} single chunk", dy, x, weight, None, _cu(bounds), "silu", 8192)
        _compare(led, f"{glabel} chunked 1024", dy, x, weight, None, _cu(bounds), "silu", 1024)


def _preactivation(x, weight, bias, cu) -> torch.Tensor:
    """The reference preactivation, single chunk -- used only to COUNT the hazard population."""
    total = x.shape[0]
    taps = weight.shape[1]
    positions, starts, _ = G._conv_vjp_sequence_bounds(total, x.device, cu)
    wf = weight.float()
    acc = torch.zeros((total, x.shape[1]), dtype=torch.float32, device=x.device)
    for tap in range(taps):
        lag = taps - 1 - tap
        source = positions - lag
        valid = source >= starts
        acc = acc + x.index_select(0, source.clamp(min=0)).float() * valid.unsqueeze(1) * wf[:, tap]
    if bias is not None:
        acc = acc + bias.float()
    return acc


def _subnormal_sigmoid_population(acc: torch.Tensor) -> int:
    """How many of these preactivations make ``torch.sigmoid`` return an fp32 subnormal."""
    sigmoid = torch.sigmoid(acc)
    return int(((sigmoid > 0.0) & (sigmoid < _FP32_MIN_NORMAL)).sum())


@triton.jit
def _sigmoid_probe_kernel(z_ptr, out_ptr, n, BC: tl.constexpr):
    """The kernels' sigmoid on its own, launched with the module's own ``_COMPILE_OPTS``."""
    off = tl.program_id(0) * BC + tl.arange(0, BC)
    mask = off < n
    z = tl.load(z_ptr + off, mask=mask, other=0.0)
    tl.store(out_ptr + off, _nonftz_sigmoid(z), mask=mask)


def _gate_sigmoid(led: Ledger) -> None:
    """The kernels' sigmoid vs ``torch.sigmoid``, bit pattern for bit pattern, over the hazard band.

    This is the sweep whose absence let a flush-to-zero divide ship: the original rounding-point
    test asserted that the SOURCE spelled ``libdevice.div_rn(...)``, which it did, while the emitted
    PTX said ``div.rn.ftz.f32``. A numerical sweep cannot be fooled that way. The last row applies
    ``ops/moe/moe_sm90/tests/_ftz_check.py``'s vacuity rule: a sweep that never produced a subnormal
    reference value would pass while testing nothing, so the subnormal population is itself gated.
    """
    populations: list[tuple[str, torch.Tensor]] = [
        # the subnormal-sigmoid window itself, dense
        ("window (-90, -85) x4M", torch.linspace(-90.0, -85.0, 4_000_000, device=DEV, dtype=torch.float64).float()),
        # the wider band where expf(-z) alone is subnormal or overflows
        ("window (-105, -80) x4M", torch.linspace(-105.0, -80.0, 4_000_000, device=DEV, dtype=torch.float64).float()),
        # EVERY fp32 mantissa of the exponent that contains the window (|z| in [64, 128), negative)
        (
            "all mantissas of z in [-128, -64)",
            (torch.arange(0, 1 << 23, device=DEV, dtype=torch.int64) | 0xC2800000).to(torch.int32).view(torch.float32),
        ),
        # every subnormal input with a small significand, both signs
        (
            "subnormal inputs",
            torch.cat(
                [
                    torch.arange(0, 1 << 20, device=DEV, dtype=torch.int32).view(torch.float32),
                    -torch.arange(0, 1 << 20, device=DEV, dtype=torch.int32).view(torch.float32),
                ]
            ),
        ),
        ("uniform (-120, 120) x4M", torch.rand(4_000_000, device=DEV) * 240.0 - 120.0),
        (
            "random bit patterns x4M",
            torch.randint(-(2**31), 2**31, (4_000_000,), device=DEV, dtype=torch.int32).view(torch.float32),
        ),
        (
            "signed zeros, band edges, extremes",
            torch.tensor(
                [
                    0.0,
                    -0.0,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                    1.4e-45,
                    -1.4e-45,
                    _FP32_MIN_NORMAL,
                    -_FP32_MIN_NORMAL,
                    3.4028235e38,
                    -3.4028235e38,
                    -87.3345,
                    -87.336544,
                    -87.34,
                    -88.0,
                    -88.7228,
                    -88.7229,
                    -103.9,
                    -104.0,
                ],
                device=DEV,
            ),
        ),
    ]
    hazard_seen = 0
    for label, z in populations:
        z = z.contiguous()
        reference = torch.sigmoid(z)
        got = torch.empty_like(z)
        _sigmoid_probe_kernel[(triton.cdiv(z.numel(), 1024),)](z, got, z.numel(), BC=1024, **FUSED._COMPILE_OPTS)
        hazard_seen += _subnormal_sigmoid_population(z)
        # NaN in == NaN out on both sides; compare its payload only where both are finite-or-equal.
        both_nan = torch.isnan(reference) & torch.isnan(got)
        differ = int(((reference.view(torch.int32) != got.view(torch.int32)) & ~both_nan).sum())
        led.rows.append((f"sigmoid: {label}", "sig", differ))
    # The gate is worthless if no input reached the subnormal band; assert it did.
    led.rows.append(("sigmoid: subnormal population reached (vacuity rule)", "pop", 0 if hazard_seen else 1))
    print(f"[gdn-conv-vjp-biteq] sigmoid swept {hazard_seen} inputs inside the subnormal-sigmoid band")


def _gate_sigmoid_window(led: Ledger, quick: bool) -> None:
    """The whole VJP with preactivations placed IN the subnormal-sigmoid window, at real widths.

    Two ways in, because they cover different things:

      * `swept`  -- fp32 x, weight zero except the lag-0 tap, so the preactivation IS x and a dense
        linspace over (-90, -85) puts >1e6 elements in and around the window.
        The tap loop still runs all four taps (three of them multiplying by 0.0).
      * `native` -- the production dtype and a full random weight, scaled so a few thousand of the
        preactivations land in the window by themselves. This is the case that says the hazard is
        reachable without the test rigging the weight.

    ``dy`` carries wide exponents in both, so the derivative's product is not itself protected.
    """
    geometries = GEOMETRIES[:1] if quick else GEOMETRIES
    for glabel, _tokens, channels, taps in geometries:
        tokens = max(512, -(-1_000_000 // channels))  # >= 1e6 elements per case
        bounds = [0, tokens // 7, tokens // 7 + 2 * (tokens // 7), tokens]
        cu = _cu(bounds)
        g = torch.Generator(device=DEV).manual_seed(88)
        dy_wide = torch.randn((tokens, channels), generator=g, device=DEV) * torch.exp2(
            torch.randint(-40, 40, (tokens, channels), generator=g, device=DEV).float()
        )
        for mode in ("swept", "native"):
            if mode == "swept":
                dtype = torch.float32
                x = (
                    torch.linspace(-90.0, -85.0, tokens * channels, device=DEV, dtype=torch.float64)
                    .float()
                    .view(tokens, channels)
                )
                weight = torch.zeros((channels, taps), device=DEV, dtype=torch.float32)
                weight[:, taps - 1] = 1.0
                dy = dy_wide
            else:
                dtype = torch.bfloat16
                # std(preactivation) ~= sqrt(taps) * 22 ~= 44, which puts ~0.2% of the elements in
                # the 1.39-wide window around -88 with no rigging at all.
                x = (torch.randn((tokens, channels), generator=g, device=DEV) * 22.0).to(dtype)
                weight = torch.randn((channels, taps), generator=g, device=DEV).to(dtype)
                dy = dy_wide.to(dtype)
            for with_bias, chunk_rows in ((False, 8192), (True, 1024)):
                bias = (
                    torch.randn(channels, generator=g, device=DEV, dtype=torch.float32).to(dtype) if with_bias else None
                )
                hazard = _subnormal_sigmoid_population(_preactivation(x, weight, bias, cu))
                case = f"sigmoid window {mode} / {glabel} / bias={int(with_bias)} / rows={chunk_rows}"
                led.rows.append((f"{case} [{hazard} in band]", "pop", 0 if hazard > 0 else 1))
                _compare(led, case, dy, x, weight, bias, cu, "silu", chunk_rows)


def _gate_hazards_at_width(led: Ledger, quick: bool) -> None:
    """Every hazard population at every production width, in the production dtype and in fp32.

    The 23x11 hazard cases in :func:`_gate_hazards` are 253 elements each. A divergence with a rate
    of 1e-4 -- which is what the flush-to-zero divide had -- is invisible there and unmissable here.
    """
    geometries = GEOMETRIES[:1] if quick else GEOMETRIES
    hazards = HAZARDS[1:2] if quick else HAZARDS[1:]
    for hazard, (glabel, tokens, channels, taps) in itertools.product(hazards, geometries):
        for dtype in (torch.bfloat16, torch.float32):
            x, dy, weight = _inputs(tokens, channels, taps, dtype, hazard, seed=20260907)
            bounds = [0, tokens // 7, tokens // 7 + 2 * (tokens // 7), tokens]
            bias = torch.randn(channels, device=DEV, dtype=torch.float32).to(dtype)
            label = f"hazard {hazard} / {glabel} / {str(dtype).removeprefix('torch.')}"
            _compare(led, label, dy, x, weight, None, _cu(bounds), "silu", 8192)
            _compare(led, f"{label} / bias / rows=1024", dy, x, weight, bias, _cu(bounds), "silu", 1024)


def _gate_tap_counts(led: Ledger) -> None:
    """Width 4 is what the models run; the kernels are generic and the reference is too."""
    for taps in (1, 2, 3, 5):
        x, dy, weight = _inputs(23, 11, taps, torch.bfloat16, "uniform", seed=taps)
        _compare(led, f"taps={taps}", dy, x, weight, None, _cu([0, 5, 9, 23]), "silu", 8192)


def _gate_batch_invariance(led: Ledger) -> None:
    """The same tokens, alone and inside a larger pack: identical gradient rows."""
    os.environ[_ROWS_ENV] = "8192"
    alone_lens, extra = [4, 5], 14
    tokens = sum(alone_lens)
    x, dy, weight = _inputs(tokens + extra, 11, 4, torch.bfloat16, "uniform", seed=31)
    cu_alone = _cu([0, 4, 9])
    cu_packed = _cu([0, 4, 9, 15, tokens + extra])
    with torch.no_grad():
        alone = G._native_conv_vjp(dy[:tokens], x[:tokens], weight, None, cu_alone, "silu")
        packed = G._native_conv_vjp(dy, x, weight, None, cu_packed, "silu")
    led.add("batch invariance", "dx", alone[0], packed[0][:tokens])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="one geometry, two packs, two hazards (smoke)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    if not G.fused_conv_backward_enabled():
        print(f"SKIP: {_ROWS_ENV.replace('CONV_BWD_ROWS', 'FUSED_CONV_BWD')} is off; nothing to gate")
        return 0

    led = Ledger()
    _gate_packs(led, args.quick)
    _gate_hazards(led, args.quick)
    _gate_hazards_at_width(led, args.quick)
    _gate_sigmoid(led)
    _gate_sigmoid_window(led, args.quick)
    _gate_geometries(led, args.quick)
    _gate_tap_counts(led)
    _gate_batch_invariance(led)

    print(led.report())
    bad = sum(1 for _, _, differ in led.rows if differ)
    print(f"\n[gdn-conv-vjp-biteq] {len(led.rows) - bad}/{len(led.rows)} tensors bit-identical to the eager VJP")
    print("RESULT:", "PASS" if led.ok else "FAIL")
    return 0 if led.ok else 1


if __name__ == "__main__":
    sys.exit(main())
