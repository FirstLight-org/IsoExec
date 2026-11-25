"""Dtype dispatch preserves tensor ownership and the caller's global norm boundary."""

import pytest
import torch

from isoexec.checks import Refusal
from isoexec.runtimes.megatron.grad_kernels import (
    _make_applier,
)


def test_mixed_norm_groups_native_calls_without_copying_gradients():
    norm_op = object()
    gradients = [
        torch.full((8,), value, dtype=dtype, device="cpu")
        for value, dtype in (
            (3, torch.bfloat16),
            (4, torch.float32),
            (2, torch.bfloat16),
        )
    ]
    calls = []

    def native(op, overflow, lists, per_tensor):
        assert op is norm_op and not per_tensor
        assert len({g.dtype for g in lists[0]}) == 1
        calls.append(lists[0])
        value = (
            torch.stack([g.float().square().sum() for g in lists[0]])
            .sum()
            .sqrt()
            .reshape(1)
        )
        return value, None

    apply = _make_applier(native, norm_op, ())
    flag = torch.zeros(1, dtype=torch.int32, device="cpu")
    result, _ = apply(norm_op, flag, [gradients], False)
    torch.testing.assert_close(
        result.square(), torch.tensor([232.0], dtype=torch.float32, device="cpu")
    )
    assert len(calls) == 2
    assert {id(g) for group in calls for g in group} == {id(g) for g in gradients}


def test_scaling_preserves_input_output_pairing_and_coefficient_identity():
    scale_op = object()
    gradients = [
        torch.ones(8, dtype=dtype, device="cpu")
        for dtype in (torch.bfloat16, torch.float32)
    ]
    coefficient = torch.tensor([0.25], dtype=torch.float32, device="cpu")
    calls = []

    def native(op, flag, lists, scale):
        assert op is scale_op and scale is coefficient
        assert len({t.dtype for tensors in lists for t in tensors}) == 1
        for source, destination in zip(*lists, strict=True):
            assert source is destination
            destination.mul_(scale)
        calls.append(lists)

    apply = _make_applier(native, object(), (scale_op,))
    apply(
        scale_op,
        torch.zeros(1, dtype=torch.int32, device="cpu"),
        [gradients, gradients],
        coefficient,
    )
    assert len(calls) == 2
    assert all(bool((g == 0.25).all()) for g in gradients)


def test_uniform_input_retains_the_original_native_call():
    gradients = [torch.ones(8, dtype=torch.float32, device="cpu")]
    lists = [gradients]
    token = object()

    def native(op, flag, received, *args):
        assert received is lists
        return token

    apply = _make_applier(native, object(), ())
    assert apply(object(), None, lists) is token


def test_unknown_mixed_operation_refuses():
    gradients = [
        torch.ones(1, dtype=dtype, device="cpu")
        for dtype in (torch.bfloat16, torch.float32)
    ]
    apply = _make_applier(lambda *args: None, object(), ())
    with pytest.raises(Refusal, match="multi_tensor_applier.op"):
        apply(object(), None, [gradients])


def test_launches_hold_a_bounded_number_of_tensor_handles():
    """TE takes one handle per tensor per list from a fixed pool; a MoE holds one Parameter per expert."""
    norm_op, scale_op = object(), object()
    gradients = [torch.full((4,), 1.0 + i % 3, dtype=torch.float32) for i in range(25)]
    seen = []

    def native(op, flag, lists, *args):
        seen.append(sum(len(tensors) for tensors in lists))
        if op is norm_op:
            return torch.stack([g.square().sum() for g in lists[0]]).sum().sqrt().reshape(1), None
        for source, destination in zip(*lists, strict=True):
            destination.copy_(source * args[0])

    apply = _make_applier(native, norm_op, (scale_op,), handles=8)
    norm, _ = apply(norm_op, None, [gradients], False)
    assert seen == [8, 8, 8, 1]
    torch.testing.assert_close(norm, torch.stack([g.square().sum() for g in gradients]).sum().sqrt().reshape(1))
    seen.clear()
    apply(scale_op, None, [gradients, gradients], 0.5)
    assert max(seen) <= 8 and sum(seen) == 50
    assert all(bool((g <= 1.5).all()) for g in gradients)
