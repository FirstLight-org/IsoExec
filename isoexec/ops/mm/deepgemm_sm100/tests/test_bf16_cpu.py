"""CPU contract of the DeepGEMM BF16 projection provider: which weights it serves (published segment count 1
only) and the incumbent fallback everywhere else. The bits are GPU-gated (``test_bf16_gpu.py``)."""
import torch
from torch.nn import functional as F

from isoexec.ops.mm.deepgemm_sm100 import bf16
from isoexec.ops.mm.mm_sm100 import segments as seg


def test_serves_only_one_chain_weights():
    for n, k in ((32768, 1280), (16384, 1280), (4096, 1280), (1280, 5120), (512, 5120), (24576, 1536)):
        assert bf16.serves(n, k) == (seg.segment_count(k, torch.bfloat16) == 1), (n, k)
    assert not bf16.serves(1000, 1280)            # not a DeepGEMM operand (N % 16)
    assert not bf16.serves(4096, 1000)            # K % 64


def test_host_tensors_take_the_incumbent():
    x = torch.randn(3, 1280, dtype=torch.bfloat16)
    w = torch.randn(4096, 1280, dtype=torch.bfloat16)
    assert torch.equal(bf16.linear(x, w), F.linear(x, w))


def test_incumbent_never_takes_the_composite_ops(monkeypatch):
    """A runtime overriding ``aten::linear`` / ``aten::matmul`` (vLLM's batch-invariant mode) cannot reach it."""
    def refuse(*a, **k):
        raise AssertionError("composite op on the projection path")
    monkeypatch.setattr(F, "linear", refuse)
    monkeypatch.setattr(torch, "matmul", refuse)
    x = torch.randn(2, 3, 1280, dtype=torch.bfloat16)
    for n in (4096, 1000):                        # served shape on host rows, unserved shape
        w = torch.randn(n, 1280, dtype=torch.bfloat16)
        out = bf16.linear(x, w)
        assert out.shape == (2, 3, n) and torch.equal(out, torch.mm(x.reshape(6, 1280), w.t()).view(2, 3, n))


def test_shape_table_bands():
    from isoexec.ops.mm.deepgemm_sm100 import bf16
    assert bf16._policy(1, 12288, 1280) == (65, None)
    assert bf16._policy(32, 12288, 1280) == (65, None)
    assert bf16._policy(48, 12288, 1280) == (None, None)
    assert bf16._policy(64, 12288, 1280) == (None, (512, 1))
    assert bf16._policy(128, 12288, 1280) == (None, (512, 1))
    assert bf16._policy(129, 12288, 1280) == (None, None)
    assert bf16._policy(8, 5120, 1024) == (None, None)
