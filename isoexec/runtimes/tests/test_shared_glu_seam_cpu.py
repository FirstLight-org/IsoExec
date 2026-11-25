"""Check shared-expert GLU epilogue seams and bitwise eager fallback for unsupported fused
operands.
"""

from __future__ import annotations

import inspect

import torch

from isoexec.ops.moe import moe_preamble_o12 as O12
from isoexec.ops.moe.eager import shared_expert_glu, shared_expert_megatron, shared_glu_epilogue
def test_host_functions_expose_the_seam():
    assert "epilogue" in inspect.signature(shared_expert_glu).parameters
    assert "glu_epilogue" in inspect.signature(shared_expert_megatron).parameters
    x = (torch.randn(5, 12) * 0.5).to(torch.bfloat16)
    ref = shared_glu_epilogue(x)
    x_glu, x_lin = torch.chunk(x, 2, dim=-1)
    assert torch.equal(ref.view(torch.int16), (torch.nn.functional.silu(x_glu) * (x_lin + 0.0)).view(torch.int16))


def test_engine_epilogue_falls_back_bitwise_off_cuda_and_counts():
    ep = O12.engine_shared_glu_epilogue(shared_glu_epilogue)
    before = dict(O12.engine_glu_counts())
    x = (torch.randn(9, 64) * 0.5).to(torch.bfloat16)  # CPU: the kernel cannot run, the chain must
    got = ep(x)
    assert torch.equal(got.view(torch.int16), shared_glu_epilogue(x).view(torch.int16))
    after = O12.engine_glu_counts()
    assert after["calls"] == before["calls"] + 1 and after["eager"] == before["eager"] + 1
    assert after["served"] == before["served"]  # nothing served off-CUDA
