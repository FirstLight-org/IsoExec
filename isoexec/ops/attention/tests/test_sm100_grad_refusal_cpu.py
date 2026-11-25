"""Forward-only SM100 adapters reject gradients before touching CUDA resources."""

import pytest
import torch


def test_packed_mla_refuses_requires_grad_before_shape_or_device_checks():
    from isoexec.ops.attention.mla_sm100.ops import mla_sm100_attention

    x = torch.empty(1, requires_grad=True)
    with pytest.raises(RuntimeError, match="forward-only; no backward"):
        mla_sm100_attention(x, x, x, x, x, x, seq_lens=(1,), scale=1.0, seg_tokens=128)

