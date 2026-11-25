"""CPU statement of the V4.1 cache-record quantizer (vLLM 0.33 ``FLASHMLA_SPARSE_DSV41`` records).

FP8 E4M3 codes with one power-of-two (UE8M0) scale per group of 32 or 128 channels. The exponent is
``e = ceil(log2(fl(max(amax, 1e-4) * fl(1/448))))``:

* ``exponent_rule="exact"``: the exact ceiling, from the FP32 exponent bits. vLLM's window-KV and index-key writers
  quantize BF16 values, for which their ``log2f`` evaluation always equals this.
* ``exponent_rule="log2f"``: CUDA libdevice ``log2f`` followed by ``ceilf``, as vLLM's compressed-KV writer (FP32
  input): it equals the exact ceiling except when the product lies a few ulps above a power of two ``2^k``,
  where ``log2f`` returns exactly ``k``. ``_LOG2F_EXACT_K`` lists, per ``k``, how many ulps above ``2^k`` that
  happens (``_log2f_exact_m``).

Codes are ``e4m3(clamp(x / 2^e, -448, 448))`` (round to nearest even); the scale byte is ``e + 127``. Finite inputs
are the domain. Dequantization ``code * 2^e`` is exact in BF16 for every reachable scale.
"""
import math

import torch

RECIPE = "deepseek_v41_cache_record_quant_v1"


def _log2f_exact_m(k: int) -> int:
    """How many FP32 ulps above 2^k libdevice ``log2f`` still returns exactly k (so ``ceilf`` gives k, not k + 1).

    log2(2^k (1 + j 2^-23)) = k + j 2^-23 / ln 2 rounds back to k while that increment is below half an ulp of the
    result, as for a correctly rounded log2. Verified on SM100 for every k in [-30, 127] against an exhaustive
    evaluation (and Triton's tl.log2 == CUDA log2f on every normal FP32 in [2^-23, 2^128)).
    """
    if k == 0:
        return 0
    mag = abs(k)                                 # the result is k + small: for k < 0 its magnitude is below |k|
    e = mag.bit_length() - 1 if k > 0 else (mag - 1).bit_length() - 1 if mag > 1 else -1
    if e < 0:
        return 0
    bound = 2.0 ** (e - 24) * 2.0 ** 23 * math.log(2.0)   # j < bound  <=>  j 2^-23 / ln 2 < half an ulp
    return math.ceil(bound) - 1 if bound > 1 else 0


_LOG2F_EXACT_K = {k: _log2f_exact_m(k) for k in range(-126, 128) if _log2f_exact_m(k)}


def validate(x, block_size, exponent_rule):
    if x.dtype not in (torch.bfloat16, torch.float32) or x.ndim < 1:
        raise TypeError("the cache-record quantizer takes a BF16 or FP32 tensor with a channel axis")
    if block_size not in (32, 128) or x.shape[-1] <= 0 or x.shape[-1] % block_size:
        raise ValueError("channel width must be a positive multiple of group32 or group128")
    if exponent_rule not in ("exact", "log2f"):
        raise ValueError("exponent_rule is 'exact' or 'log2f'")


def _exponent(r: torch.Tensor, exponent_rule: str) -> torch.Tensor:
    """FP32 r > 0 -> biased exponent (int32) of the power-of-two scale."""
    bits = r.contiguous().view(torch.int32)
    biased = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    if exponent_rule == "log2f":
        k = ((bits >> 23) & 255) - 127
        mant = bits & 0x7fffff
        limit = torch.zeros_like(k)
        for kk, m in _LOG2F_EXACT_K.items():
            limit = torch.where(k == kk, torch.full_like(k, m), limit)
        biased = torch.where((mant > 0) & (mant <= limit), biased - 1, biased)
    return biased


def quantize_reference(x, *, block_size=32, exponent_rule="exact", dequantize=False):
    """Return (E4M3 codes, UE8M0 scale bytes as float8_e8m0fnu) or the BF16 quantize/dequantize result."""
    validate(x, block_size, exponent_rule)
    z = x.detach().float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    amax = z.abs().amax(-1).clamp_min(1e-4)
    r = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    biased = _exponent(r, exponent_rule)
    scale = (biased << 23).view(torch.float32)
    codes = (z / scale.unsqueeze(-1)).clamp(-448.0, 448.0).reshape(x.shape).to(torch.float8_e4m3fn)
    if dequantize:
        values = codes.float().reshape(z.shape)
        return (values * scale.unsqueeze(-1)).reshape(x.shape).to(torch.bfloat16)
    return codes, scale.to(torch.float8_e8m0fnu)
