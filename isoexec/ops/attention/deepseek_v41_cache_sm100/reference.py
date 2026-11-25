"""Bit-exact CPU emulation of the row recipe ``deepseek_v41_attn_rows_v1`` (csrc/numerics.cuh).

Every operation below is one IEEE-754 binary32 round-to-nearest-even operation on CPU tensors, in the
order the kernel evaluates it, so GPU output == emulation bit for bit (tests/test_rows_gpu.py). The
float64 oracles of the composition are ``isoexec.nn.deepseek_attention`` (the model's equations).
"""
from __future__ import annotations

import math

import torch

from isoexec.ops.quant.checkpoint_sm100.reference import quantize_reference

_LANES = 32


def rope_frequencies(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """``precompute_freqs_cis`` of the pinned official inference/model.py, verbatim, on the CPU.

    (isoexec.nn.deepseek_attention.rotary_frequencies writes ``base ** -x`` for ``1 / base ** x``
    and differs from this table in the last bits.)
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _rn32(x64: torch.Tensor) -> torch.Tensor:
    """binary64 -> binary32, round to nearest even."""
    return x64.to(torch.float32)


def sum_squares_tree(v: torch.Tensor) -> torch.Tensor:
    """v fp32 [R, D] -> fp32 [R]: lane-contiguous chains of D/32, then the XOR butterfly 16..1."""
    R, D = v.shape
    E = D // _LANES
    v3 = v.reshape(R, _LANES, E)
    s = v3[..., 0] * v3[..., 0]
    for i in range(1, E):
        s = s + v3[..., i] * v3[..., i]
    lanes = torch.arange(_LANES)
    for d in (16, 8, 4, 2, 1):
        s = s + s[:, lanes ^ d]
    return s[:, 0]


def rms_emul(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 [R, D] -> bf16: y = bf16((x * inv) * w), inv = 1 / sqrt(ss / D + eps)."""
    v = x.float().cpu()
    D = v.shape[1]
    ss = sum_squares_tree(v)
    # div / add / sqrt / reciprocal each correctly rounded to binary32: evaluated exactly enough in binary64
    # and rounded once (53 >= 2*24 + 2, so the double rounding is innocuous). torch's CPU float32 sqrt is
    # NOT correctly rounded on this AVX512 build (1 ulp off for some inputs), unlike the kernel's sqrt.rn.
    t = _rn32(ss.double() / float(D))
    t = _rn32(t.double() + float(torch.tensor(eps, dtype=torch.float32)))
    inv = _rn32(1.0 / _rn32(t.double().sqrt()).double())
    return ((v * inv[:, None]) * w.float().cpu()).to(torch.bfloat16)


def rope_emul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor,
              *, inverse: bool = False, round_bf16: bool = True) -> torch.Tensor:
    """Adjacent-pair RoPE on the last 64 elements of bf16 rows [R, D]; positions int [R].

    ``round_bf16=False`` returns the FP32 rotation (the compressed-KV record quantizes it unrounded)."""
    y = x.float().cpu().clone()
    D = y.shape[1]
    a, b = y[:, D - 64::2], y[:, D - 63::2]
    c, s = cos.cpu()[positions.cpu().long()], sin.cpu()[positions.cpu().long()]
    if not inverse:
        re = a * c - b * s
        im = a * s + b * c
    else:
        re = a * c + b * s
        im = b * c - a * s
    y[:, D - 64::2], y[:, D - 63::2] = re, im
    # one bf16 rounding of the rotated tail; the prefix is a bit copy of the bf16 input
    return y.to(torch.bfloat16) if round_bf16 else y


def quant_emul(x: torch.Tensor, block: int, rule: str = "exact") -> torch.Tensor:
    return quantize_reference(x.cpu(), block_size=block, exponent_rule=rule, dequantize=True)


def index_codes(x: torch.Tensor):
    """bf16 [R, 128] -> (e4m3 codes uint8 [R, 128], exponent word int32 [R]: four equal UE8M0 bytes)."""
    codes, scales = quantize_reference(x.cpu(), block_size=128)
    sb = scales.view(torch.uint8).reshape(-1).to(torch.int64)
    packed = sb * 0x01010101
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return codes.view(torch.uint8), packed


def window_kv_emul(kvr, w, eps, cos, sin, positions):
    return quant_emul(rope_emul(rms_emul(kvr, w, eps), cos, sin, positions), 32)


def compressed_kv_emul(latent, cos, sin, positions):
    return quant_emul(rope_emul(latent, cos, sin, positions, round_bf16=False), 32, "log2f")


def index_k_emul(k, w, eps, cos, sin, positions):
    return index_codes(rope_emul(rms_emul(k, w, eps), cos, sin, positions))


def index_q_emul(q, cos, sin, positions):
    return index_codes(rope_emul(q, cos, sin, positions))
