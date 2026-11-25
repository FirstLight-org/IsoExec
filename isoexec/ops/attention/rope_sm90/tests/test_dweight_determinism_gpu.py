"""Synthetic fixed-input SM90 backward repeatability; not model-run evidence."""

from typing import Any, cast

import pytest
import torch

from isoexec.ops.attention.rope_sm90 import build_table, qk_norm_rope
from isoexec.ops.attention.rope_sm90._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                               reason="requires SM90")


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def _case(tokens: int, width: int, rotary: int, dtype: torch.dtype,
          weight_dtype: torch.dtype, mrope: bool) -> tuple[Any, ...]:
    generator = torch.Generator(device="cuda").manual_seed(771 + tokens + width)

    def random(shape: tuple[int, ...], output_dtype: torch.dtype) -> torch.Tensor:
        return (torch.randn(shape, device="cuda", generator=generator) * .3).to(output_dtype)

    # Packed slices retain noncontiguous token strides and a nonzero offset.
    q = random((tokens, 5, width), dtype)[:, 1:4].detach().requires_grad_()
    k = random((tokens, 4, width), dtype)[:, 1:3].detach().requires_grad_()
    wq, wk = (random((width,), weight_dtype).requires_grad_() for _ in range(2))
    dq, dk = random(tuple(q.shape), dtype), random(tuple(k.shape), dtype)
    positions = torch.arange(tokens, device="cuda")
    if mrope:
        positions = torch.stack((positions, positions + 1, positions + 2))
    sections = (rotary // 4, rotary // 8, rotary // 8) if mrope else None
    table = (build_table(1e7, rotary, max(4, tokens + 3), "cuda", mrope_section=sections,
                         mrope_interleaved=mrope) if rotary else None)
    return q, k, wq, wk, dq, dk, positions, table


@pytest.mark.parametrize("tokens,width,rotary,dtype,weight_dtype,mrope", [
    (0, 64, 0, torch.bfloat16, torch.bfloat16, False),
    (1, 128, 32, torch.float16, torch.float32, True),
    (17, 256, 64, torch.bfloat16, torch.bfloat16, True),
    (62, 256, 0, torch.float32, torch.float32, False),
    (23, 128, 128, torch.bfloat16, torch.bfloat16, False),  # Qwen3 geometry
])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_repeated_strided_backward_is_exact(tokens: int, width: int, rotary: int, dtype: torch.dtype,
                                            weight_dtype: torch.dtype, mrope: bool, zero_centered: bool) -> None:
    q, k, wq, wk, dq, dk, positions, table = _case(tokens, width, rotary, dtype, weight_dtype, mrope)
    results = []
    for _ in range(8):
        outputs = qk_norm_rope(q, k, wq, wk, positions, table, 1e-6, zero_centered=zero_centered)
        gradients = torch.autograd.grad(outputs, (q, k, wq, wk), (dq, dk))
        assert gradients[2].dtype == gradients[3].dtype == weight_dtype
        assert all(torch.isfinite(value).all() for value in (*outputs, *gradients))
        if tokens:
            assert gradients[2].abs().sum() + gradients[3].abs().sum() > 0
        results.append(tuple(_bits(value).clone() for value in (*outputs, *gradients)))
    assert all(torch.equal(left, right) for values in results[1:]
               for left, right in zip(results[0], values, strict=True))
    if tokens == 0:
        assert not results[0][-1].count_nonzero() and not results[0][-2].count_nonzero()


@pytest.mark.parametrize("zero_centered", [True, False])
def test_fixed_weight_gradient_replays_in_cuda_graph(zero_centered: bool) -> None:
    q, k, wq, wk, dq, dk, positions, table = _case(17, 256, 64, torch.bfloat16, torch.bfloat16, True)

    def step() -> tuple[torch.Tensor, ...]:
        output = qk_norm_rope(q, k, wq, wk, positions, table, 1e-6, zero_centered=zero_centered)
        return torch.autograd.grad(output, (q, k, wq, wk), (dq, dk))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    with torch.no_grad():
        q.mul_(1.125)
        dq.mul_(.75)
    graph.replay()
    expected = step()
    assert all(torch.equal(_bits(left), _bits(right)) for left, right in zip(captured, expected, strict=True))


def test_native_backward_refuses_a_different_grouping_recipe() -> None:
    q, k, wq, wk, dq, dk, _, _ = _case(17, 64, 0, torch.bfloat16, torch.bfloat16, False)
    native = cast(Any, extension())
    _, _, inverse = native.qknorm_rope_forward(q, k, wq, wk, None, None, 0, 0, 1e-6, True, True)
    with pytest.raises(RuntimeError, match="requires gshift=4"):
        native.qknorm_rope_backward(dq, dk, q, k, wq, wk, inverse, None, None, 0, 0, True, True, 0)


def test_native_entrypoints_refuse_mrope_beyond_64_mask_columns() -> None:
    q, k, wq, wk, dq, dk, positions, table = _case(17, 256, 256, torch.bfloat16, torch.bfloat16, False)
    assert table is not None
    native = cast(Any, extension())
    _, _, inverse = native.qknorm_rope_forward(q, k, wq, wk, positions, table.table, 0, 0, 1e-6, True, True)
    positions = positions.expand(3, -1).clone()
    assert not native.rope(q, k, positions, table.table, 2, 4, False, False, False, False)
    assert not native.qknorm_rope_forward(q, k, wq, wk, positions, table.table, 2, 4, 1e-6, True, True)
    assert not native.qknorm_rope_backward(dq, dk, q, k, wq, wk, inverse, positions, table.table,
                                         2, 4, True, True, 4)


@pytest.mark.parametrize("invalid", ["inverse_cpu", "inverse_rank", "dq_cpu", "dk_cpu", "dq_rank"])
def test_native_backward_refuses_invalid_saved_or_upstream_tensors(invalid: str) -> None:
    q, k, wq, wk, dq, dk, _, _ = _case(17, 64, 0, torch.bfloat16, torch.bfloat16, False)
    native = cast(Any, extension())
    _, _, inverse = native.qknorm_rope_forward(q, k, wq, wk, None, None, 0, 0, 1e-6, True, True)
    if invalid == "inverse_cpu":
        inverse = inverse.cpu()
    elif invalid == "inverse_rank":
        inverse = inverse.flatten()
    elif invalid == "dq_cpu":
        dq = dq.cpu()
    elif invalid == "dk_cpu":
        dk = dk.cpu()
    else:
        dq = dq.flatten()
    assert not native.qknorm_rope_backward(dq, dk, q, k, wq, wk, inverse, None, None, 0, 0, True, True, 4)
    # Rejection must happen before CUDA sees an invalid host pointer.
    torch.cuda.synchronize()
