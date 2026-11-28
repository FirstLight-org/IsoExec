"""Loader and launch glue of the vendored DeepSeek DeepGEMM (MIT; see ``_vendor/deep_gemm``).

Warm up each shape before CUDA graph capture (the vendored kernels are JIT-compiled per shape).
"""
import os

import torch


def _dg():
    from isoexec.ops._vendor.deep_gemm import load
    dg = load()
    # Deterministic algorithms are a hard requirement of every IsoExec caller, even if another caller changed
    # the vendored library's process-local setting.
    dg.use_deterministic_algorithms(True)
    return dg


#: Programmatic dependent launch for the dense GEMM launches of this package (ISOEXEC_DEEPGEMM_PDL=0: off).
#: The vendored kernel touches no global memory before its ``cudaGridDependencySynchronize`` (barrier
#: init, TMEM allocation and descriptor prefetch only), so its prologue may overlap the previous kernel;
#: the launch attribute changes when a CTA starts, never what it computes.
PDL = os.environ.get("ISOEXEC_DEEPGEMM_PDL", "1") != "0"

#: Early launch of the BF16 GEMM under PDL (ISOEXEC_DEEPGEMM_EARLY; 0 off): 1 (default) triggers the dependent launch
#: right after the dependency wait, as cuBLAS's kernels do; k >= 2 also loads up to k - 1 weight (B) stages of the
#: CTA's first tile before the wait (measured no better than 1 in chains); + 64 (e.g. 65, opt-in) the lean
#: post-wait path (scheduler formed before the wait, the TMA producer triggers after its first stage); + 128 (e.g.
#: 193, opt-in) the lean tail (tensor memory freed right after the last tensor-memory read, overlapping the store). The
#: activation and every store stay after the wait and the MMAs are unchanged, so no bit changes.
EARLY = int(os.environ.get("ISOEXEC_DEEPGEMM_EARLY", "1"))


def _launch(call, *, weight_b=False, early=None):
    """Run one vendored launch with PDL on (``PDL``). PDL is the vendored library's process-wide launch
    default; it is enabled for this launch only and the previous default restored, so no other vendored
    kernel (MegaMoE, layout transforms) changes launch mode.

    ``weight_b``: the call's B operand is a weight. Early launch (``EARLY``) reads B before the dependency
    wait, which needs B not to be written by the preceding kernel; it is used only under
    ``torch.inference_mode`` (the engine), where weights are never written between projections, and
    restored to off after the launch."""
    dg = _dg()
    if not PDL:
        call(dg)
        return
    # ``early``: a caller's measured mode for this launch (shape table), honoured only while the default mode 1 is in
    # force (ISOEXEC_DEEPGEMM_EARLY other than 1 keeps its meaning, e.g. 0 = off for everyone)
    mode = early if (early is not None and EARLY == 1) else EARLY
    early = mode if weight_b and torch.is_inference_mode_enabled() else 0
    prev, prev_early = dg.get_pdl(), dg.get_early_launch()
    dg.set_pdl(True)
    dg.set_early_launch(early)
    try:
        call(dg)
    finally:
        dg.set_early_launch(prev_early)
        dg.set_pdl(prev)


def bf16_grouped_bhr_hdr_bhd(a, b, out):
    """``out[t, g] = a[t, g] @ b[g]^T`` (DeepGEMM BF16 einsum ``bhr,hdr->bhd``, deterministic mode).

    a BF16 [T, G, K] (any row/group strides, unit K stride), b BF16 [G, N, K], out BF16 [T, G, N]
    (token-major). One ascending-K tcgen05 BF16 chain per output from +0, FP32 accumulator, one BF16 RNE:
    the program of ``moe_sm100.fc1`` (attention ``grouped_gemm``); the tile / swap-AB / cluster choice is
    ownership only (bitwise equal to it on real checkpoint wo_a weights, M = 1..2048).
    """
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("BF16 operands and output required")
    if a.stride(-1) != 1 or b.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("unit K / N stride required")
    from . import jit_warm
    jit_warm.note("bhr", g=int(b.shape[0]), r=int(b.shape[1]), d=int(b.shape[2]))
    _launch(lambda dg: dg.einsum("bhr,hdr->bhd", a, b, out), weight_b=True)
    return out


__all__ = ["EARLY", "PDL", "bf16_grouped_bhr_hdr_bhd"]
