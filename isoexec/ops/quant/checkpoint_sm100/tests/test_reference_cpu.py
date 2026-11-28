"""Boundary examples of the cache-record quantizer, kept independent of the CUDA implementation."""
import pytest
import torch

from isoexec.ops.quant.checkpoint_sm100 import reference as ref
from isoexec.ops.quant.checkpoint_sm100.reference import quantize_reference


@pytest.mark.parametrize("block", [32, 128])
def test_zero_group_takes_the_1e4_floor(block):
    x = torch.zeros(2, 256, dtype=torch.bfloat16)
    codes, scales = quantize_reference(x, block_size=block)
    assert scales.float().eq(2.0 ** -22).all() and scales.shape == (2, 256 // block)
    assert not codes.view(torch.uint8).any()


def test_scale_boundary_rounds_up_before_quantizing():
    x = torch.zeros(2, 32, dtype=torch.bfloat16)
    x[0, 0], x[1, 0] = 448., 450.
    codes, scales = quantize_reference(x)
    assert scales.float().flatten().tolist() == [1., 2.]
    assert codes.float()[0, 0].item() == 448. and codes.float()[1, 0].item() == 224.


def test_signed_zero_and_round_to_nearest_even():
    x = torch.zeros(1, 32, dtype=torch.float32)
    x[0, 0] = 448.                                   # e = 0
    x[0, 1] = -1e-9                                  # underflows to -0
    x[0, 2] = 1.0625                                 # tie between 1.0 and 1.125 -> 1.0 (even)
    x[0, 3] = 1.1875                                 # tie between 1.125 and 1.25 -> 1.25 (even)
    codes, _ = quantize_reference(x)
    b = codes.view(torch.uint8)[0]
    assert b[1].item() == 0x80
    assert codes.float()[0, 2].item() == 1.0 and codes.float()[0, 3].item() == 1.25


def test_dequantize_is_exact_bf16():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(64, 512, generator=g) * 3).to(torch.bfloat16)
    codes, scales = quantize_reference(x)
    y = quantize_reference(x, dequantize=True)
    exact = codes.float().reshape(64, 16, 32) * scales.float().unsqueeze(-1)
    assert torch.equal(y.float(), exact.reshape(64, 512))


def test_bf16_input_rules_agree():
    """For BF16 input the log2f rule never differs from the exact ceiling (CACHE-SPEC section 1)."""
    v = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = torch.zeros(v.numel(), 32, dtype=torch.bfloat16)
    x[:, 0] = v
    _, a = quantize_reference(x, exponent_rule="exact")
    _, b = quantize_reference(x, exponent_rule="log2f")
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("k,m", [(-6, 1), (-12, 2), (-20, 5), (-4, 0), (0, 0), (5, 1)])
def test_log2f_rule_steps_down_just_above_a_power_of_two(k, m):
    """log2f(2^k (1 + j 2^-23)) == k for j <= m (measured on SM100; CACHE-SPEC section 2)."""
    assert ref._log2f_exact_m(k) == m
    j = torch.arange(0, m + 3, dtype=torch.int32)
    r = (torch.tensor([2.0 ** k], dtype=torch.float32).view(torch.int32) + j).view(torch.float32)
    exact, lg = ref._exponent(r, "exact"), ref._exponent(r, "log2f")
    expect = torch.where((j >= 1) & (j <= m), exact - 1, exact)
    assert torch.equal(lg, expect) and exact[0].item() == k + 127


@pytest.mark.parametrize("kwargs", [{"block_size": 16}, {"exponent_rule": "round"}])
def test_bad_recipe_refused(kwargs):
    with pytest.raises(ValueError):
        quantize_reference(torch.ones(2, 32, dtype=torch.bfloat16), **kwargs)


def test_bad_tensor_refused():
    with pytest.raises(TypeError):
        quantize_reference(torch.ones(2, 32, dtype=torch.float16))
    with pytest.raises(ValueError):
        quantize_reference(torch.ones(2, 33, dtype=torch.bfloat16))
