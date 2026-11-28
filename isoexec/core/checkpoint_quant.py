"""Checkpoint byte decoding and explicitly proposed reference weight encoding.

Shared by the component tier and native checkpoint operation candidates. These
pure tensor functions own no model state, choose no device architecture, and do
not claim parity with an upstream inference activation quantizer.
"""
import math

import torch
from torch.nn import functional as F

_STORAGES = {"deepseek_fp8": (32, 32), "deepseek_fp4": (1, 32), "glm_fp8": (128, 128)}
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _block_shape(storage):
    try:
        return _STORAGES[storage]
    except KeyError:
        raise ValueError(f"unknown checkpoint storage {storage!r}; expected {tuple(_STORAGES)}") from None



def decode_checkpoint_weight(weight, scale, storage):
    """Decode raw checkpoint multipliers to FP32, preserving row/K block axes.

    E2M1 stores consecutive K values in the low then high nibble of each byte.
    GLM's misleading ``weight_scale_inv`` name holds a multiplier, not reciprocal.
    Both inputs must share a device. No change is made to their storage.
    """
    br, bk = _block_shape(storage)
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("checkpoint weight and scale must both be matrices")
    if weight.device != scale.device:
        raise ValueError("checkpoint weight and scale must share a device")
    if storage == "deepseek_fp4":
        if weight.dtype not in (torch.int8, torch.uint8):
            raise TypeError("packed E2M1 checkpoint weights must have int8 or uint8 dtype")
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("DeepSeek FP4 scales must have UE8M0 dtype")
        codes = weight.view(torch.uint8)
        low, high = codes & 15, codes >> 4
        codes = torch.stack((low, high), dim=-1).flatten(-2).long()
        table = torch.tensor((*_FP4_MAGNITUDES, *(-v for v in _FP4_MAGNITUDES)),
                             dtype=torch.float32, device=weight.device)
        values = table[codes]
    else:
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError("FP8 checkpoint weights must have E4M3FN dtype")
        expected_dtype = torch.float32 if storage == "glm_fp8" else torch.float8_e8m0fnu
        if scale.dtype != expected_dtype:
            raise TypeError(f"{storage} scales must have {expected_dtype} dtype")
        values = weight.float()
    rows, cols = values.shape
    expected = (math.ceil(rows / br), math.ceil(cols / bk))
    if tuple(scale.shape) != expected:
        raise ValueError(f"{storage} scale shape {tuple(scale.shape)} != {expected}")
    if storage == "deepseek_fp4" and cols % 32:
        raise ValueError("packed E2M1 checkpoint K must be divisible by 32")
    expanded = scale.float().repeat_interleave(br, 0).repeat_interleave(bk, 1)
    return values * expanded[:rows, :cols]



def _scale(amax, maximum, scale_format):
    # This epsilon/rounding is deliberately named as a reference training recipe.
    # A materialized divisor: torch evaluates tensor / Python-scalar on CUDA as a multiplication by
    # the reciprocal but on the CPU as an IEEE division, so the scalar form made the encoded scales
    # device-dependent. Tensor / tensor is an IEEE division on every device (what native encoders,
    # e.g. ops/moe/clamped_swiglu_sm100 weight_quant, compute).
    raw = (amax / torch.full_like(amax, maximum)).clamp_min(1e-12)
    if scale_format == "ue8m0":
        return torch.exp2(torch.ceil(torch.log2(raw))).clamp(2.0 ** -126, 2.0 ** 127)
    if scale_format == "e4m3":
        return raw.clamp(2.0 ** -9, 448.0).to(torch.float8_e4m3fn).float()
    if scale_format == "fp32":
        return raw
    raise ValueError("scale_format must be ue8m0, e4m3 or fp32")



def _fp4_codes(values):
    """Saturating E2M1 round-to-nearest-even, including the sign of zero."""
    boundaries = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=values.device)
    magnitude = values.abs().clamp_max(6.0)
    index = torch.bucketize(magnitude.contiguous(), boundaries)
    # bucketize picks the lower code at a midpoint; odd lower codes round up.
    tie = magnitude == boundaries[index.clamp_max(6)]
    index = index + ((index < 7) & tie & (index.remainder(2) == 1)).long()
    return (index | (torch.signbit(values).long() << 3)).to(torch.uint8)



def quantize_checkpoint_weight(weight, storage):
    """Encode a master with the explicit reference training quantizer.

    Block absmax / format maximum; UE8M0 scales round upward to a power of two;
    FP32 GLM scales are unrounded. E4M3/E2M1 use saturating nearest-even codes.
    This is useful for scratch initialization and STE, not checkpoint conversion
    that promises identical outputs to an upstream GPU quantization kernel.
    """
    br, bk = _block_shape(storage)
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("master weight must be a floating point matrix")
    rows, cols = weight.shape
    if storage == "deepseek_fp4" and cols % 32:
        raise ValueError("packed E2M1 checkpoint K must be divisible by 32")
    nr, nk = math.ceil(rows / br), math.ceil(cols / bk)
    padded = F.pad(weight.detach().float(), (0, nk * bk - cols, 0, nr * br - rows))
    blocks = padded.reshape(nr, br, nk, bk).permute(0, 2, 1, 3)
    fp4 = storage == "deepseek_fp4"
    maximum = 6.0 if fp4 else 448.0
    scale_format = "fp32" if storage == "glm_fp8" else "ue8m0"
    scale = _scale(blocks.abs().amax((-1, -2)), maximum, scale_format)
    normalized = (blocks / scale[..., None, None]).clamp(-maximum, maximum)
    normalized = normalized.permute(0, 2, 1, 3).reshape(nr * br, nk * bk)[:rows, :cols]
    if fp4:
        codes = _fp4_codes(normalized)
        encoded = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    else:
        encoded = normalized.to(torch.float8_e4m3fn)
    return encoded.contiguous(), scale.to(torch.float32 if storage == "glm_fp8" else torch.float8_e8m0fnu)

