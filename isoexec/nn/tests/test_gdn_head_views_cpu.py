"""Channel-only head views preserve packed layout and original backward bits."""

from typing import Any

import pytest
import torch
from torch.overrides import TorchFunctionMode

from isoexec.nn import GDN
from isoexec.nn.tests._gdn_fixtures import placement
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context


class _LegacyReshape(TorchFunctionMode):
    """Replay the prior view expression on the actual component call path."""

    def __torch_function__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        if func is torch.Tensor.unflatten:
            value, dim, sizes = args
            assert dim == -1
            return value.reshape(*value.shape[:-1], *sizes)
        return func(*args, **({} if kwargs is None else kwargs))


@pytest.mark.parametrize("tokens", [1, 4])
def test_gdn_head_views_preserve_packed_strides_aliases_and_gradients(tokens: int) -> None:
    module = GDN(hidden=16, key_heads=2, value_heads=4, key_head_dim=4, value_head_dim=4,
                 conv_kernel=3, device="cpu")
    module.apply_placement(placement(), site="gdn", op_table=OpTable(mode="reference"))
    generator = torch.Generator().manual_seed(413)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .05)
    seed = torch.randn(tokens, 16, generator=generator).bfloat16()
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    seq = make_sequence_context(torch.arange(tokens), cu, max_q=tokens, max_kv=tokens)
    core, norm = module.slots.core, module.slots.out_norm
    views: list[torch.Tensor] = []

    def observed_core(*args: Any, **kwargs: Any) -> torch.Tensor:
        views.extend(args[:3])
        return core(*args, **kwargs)

    def observed_norm(*args: Any, **kwargs: Any) -> torch.Tensor:
        views.append(args[1])
        return norm(*args, **kwargs)

    module.slots.core, module.slots.out_norm = observed_core, observed_norm

    def run() -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[Any, ...]]]:
        module.zero_grad(set_to_none=True)
        views.clear()
        x = seed.clone().requires_grad_()
        output = module(x, seq)
        output.float().square().sum().backward()
        gradients = [x.grad, *(parameter.grad for parameter in module.parameters())]
        assert all(value is not None for value in gradients)
        # View metadata and shared-storage identity must agree, including T1.
        storage_groups = {value.untyped_storage().data_ptr(): index for index, value in enumerate(views)}
        metadata = [(tuple(value.shape), value.stride(), value.storage_offset(),
                     storage_groups[value.untyped_storage().data_ptr()]) for value in views]
        return output.detach(), [value.detach().clone() for value in gradients if value is not None], metadata

    with _LegacyReshape():
        expected, old_gradients, old_views = run()
    actual, gradients, new_views = run()
    assert old_views == new_views
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert actual.isfinite().all() and actual.abs().sum() > 0
    for left, right in zip(old_gradients, gradients, strict=True):
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
