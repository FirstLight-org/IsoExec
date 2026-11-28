"""CPU contract of the checkpoint block decode: the released FP8/FP4 bytes decode exactly to the masters."""

import pytest
import torch

from isoexec.core.checkpoint_quant import decode_checkpoint_weight, quantize_checkpoint_weight


def test_fp4_nibbles_decode_low_then_high_with_block_scales():
    # one row, 64 K values: byte j holds K = 2j (low nibble) and K = 2j + 1 (high nibble)
    codes = torch.arange(64, dtype=torch.uint8) % 16
    packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 32)
    scale = torch.tensor([[2.0, 0.25]]).to(torch.float8_e8m0fnu)
    table = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0))
    want = table[codes.long()] * torch.tensor([2.0] * 32 + [0.25] * 32)
    got = decode_checkpoint_weight(packed, scale, "deepseek_fp4")
    assert got.dtype == torch.float32 and torch.equal(got.view(torch.int32), want.view(1, 64).view(torch.int32))


def test_fp8_blocks_expand_over_rows_and_k():
    w = torch.randn(40, 70).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.arange(6, dtype=torch.float32) - 3).view(2, 3).to(torch.float8_e8m0fnu)
    got = decode_checkpoint_weight(w, scale, "deepseek_fp8")
    want = w.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:40, :70]
    assert torch.equal(got, want)


@pytest.mark.parametrize("storage", ["deepseek_fp8", "deepseek_fp4"])
def test_decoded_masters_are_exact_in_bf16(storage):
    """E4M3 / E2M1 codes times UE8M0 powers of two have at most 4 significant bits: the BF16 master holds
    every decoded value exactly (the load-time decode loses nothing)."""
    g = torch.Generator().manual_seed(0)
    w, s = quantize_checkpoint_weight(torch.randn(64, 128, generator=g) * 0.05, storage)
    decoded = decode_checkpoint_weight(w, s, storage)
    assert torch.equal(decoded.to(torch.bfloat16).float(), decoded)


def test_refusals():
    with pytest.raises(ValueError, match="unknown checkpoint storage"):
        decode_checkpoint_weight(torch.zeros(1, 32, dtype=torch.uint8), torch.zeros(1, 1), "int4")
    with pytest.raises(TypeError, match="UE8M0"):
        decode_checkpoint_weight(torch.zeros(1, 16, dtype=torch.uint8), torch.zeros(1, 1), "deepseek_fp4")
    with pytest.raises(ValueError, match="scale shape"):
        decode_checkpoint_weight(torch.zeros(32, 32).to(torch.float8_e4m3fn),
                                 torch.zeros(2, 1).to(torch.float8_e8m0fnu), "deepseek_fp8")
