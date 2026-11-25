"""Check temperature division against the engines' fp32 tensor divisor at non-unit temperatures.
Python-scalar reciprocal multiplication can differ by one ULP."""

import pytest
import torch

from isoexec.ops.logprobs.temperature import temperature_divide


def test_temperature_one_is_the_identity_and_keeps_the_storage():
    x = torch.randn(8, 512)
    assert temperature_divide(x, 1.0) is x


def test_the_divide_matches_a_broadcast_row_tensor_bitwise():
    """The engines divide by ``[B, 1]``; we divide by a 0-dim tensor. Elementwise division does not
    depend on the divisor's shape, only on its being a tensor -- assert that, do not assume it."""
    torch.manual_seed(0)
    x = (torch.randn(37, 4096) * 8).float()
    for temperature in (0.7, 1.3, 0.05):
        engine_shaped = torch.full((37, 1), temperature, dtype=torch.float32)
        assert torch.equal(temperature_divide(x, temperature), x / engine_shaped), temperature


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the two spellings diverge on the CUDA kernel only")
def test_the_python_scalar_spelling_is_a_different_number_on_cuda():
    """Witness CUDA rounding differences between scalar reciprocal multiplication and fp32 tensor division."""
    torch.manual_seed(0)
    x = (torch.randn(37, 4096, device="cuda") * 8).float()
    scalar = x / 0.7
    assert not torch.equal(scalar, temperature_divide(x, 0.7))
    # ... and the python-float form is exactly the reciprocal multiply, which is the mechanism.
    assert torch.equal(scalar, x * (1.0 / 0.7))


def test_the_divide_matches_a_broadcast_row_tensor_bitwise_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda only")
    torch.manual_seed(0)
    x = (torch.randn(37, 4096, device="cuda") * 8).float()
    for temperature in (0.7, 1.3, 0.05):
        engine_shaped = torch.full((37, 1), temperature, dtype=torch.float32, device="cuda")
        assert torch.equal(temperature_divide(x, temperature), x / engine_shaped), temperature


def test_dtype_and_device_are_preserved():
    x = torch.randn(4, 16, dtype=torch.float32)
    out = temperature_divide(x, 0.5)
    assert out.dtype is torch.float32 and out.device == x.device and out.shape == x.shape
