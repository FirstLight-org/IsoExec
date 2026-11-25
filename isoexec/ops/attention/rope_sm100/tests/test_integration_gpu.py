"""SM100 RoPE registry selection and kernel integration."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100"
)

DEV = "cuda"
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def _small_tables(monkeypatch):
    """The seam's table covers 262144 positions by default; the packs here reach ~200."""
    from isoexec.ops.attention.rope_sm100 import integration as I

    monkeypatch.setenv(I.MAX_POSITIONS_FLAG, "4096")


def _seq(lengths, positions):
    from isoexec.schema import SequenceContext

    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    return SequenceContext(positions=positions, cu_seqlens=cu, max_q=max(lengths))


def test_the_sm100_rope_row_resolves_and_executes_its_declared_kernel():
    """The public table selects SM100 RoPE and preserves the direct kernel's bits."""
    from isoexec.ops import _dispatch
    from isoexec.ops.attention.rope_sm100 import build_table, rope
    from isoexec.ops.table import OpTable

    assert _dispatch.FAMILIES["rope"].archs["sm100"] == "attention.rope_sm100"
    assert _dispatch.selected("rope") == "attention.rope_sm100"
    binding = OpTable().bind("rope.apply", site="model.rope", facts={
        "style": "neox", "rotary_dim": 64, "theta": 1e6, "qk_norm": None,
    })
    assert binding.impl == "sm100_table_v1"
    q = torch.randn(17, 4, 64, device=DEV, dtype=BF).requires_grad_()
    k = torch.randn(17, 2, 64, device=DEV, dtype=BF).requires_grad_()
    positions = torch.arange(17, device=DEV)
    actual = binding.fn(q, k, style="neox", rotary_dim=64, seq=_seq([17], positions))
    expected = rope(q, k, positions, build_table(1e6, 64, 4096, DEV), style="neox")
    assert all(bitwise_equal(a, b) for a, b in zip(actual, expected, strict=True))


