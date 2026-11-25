"""CPU contract of the tail-only side-stream coefficients: on by default up to 2048 rows (a package constant),
never on host tensors or under autograd, and the two kernels it launches are bound. The bits are GPU-gated
(``test_boundary_gpu.py::test_split_parts_and_tail_overlap_bitwise``)."""
from pathlib import Path

import torch

from isoexec.ops.residual.deepseek_mhc_sm100 import ops

CSRC = Path(__file__).resolve().parents[2] / "mhc_sm100" / "csrc"


def test_tail_overlap_limit_is_a_constant():
    assert ops.OVERLAP_TAIL_MAX_ROWS == 2048 and ops.OVERLAP_TAIL_MAX_ROWS > ops.OVERLAP_MAX_ROWS


def test_host_rows_never_overlap(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_params_tail_overlapped", lambda *a: calls.append(1))
    monkeypatch.setattr(ops, "mhc_params", lambda *a, **k: (None, None, None))
    x = torch.zeros(300, 4, ops.HIDDEN, dtype=torch.bfloat16)
    w = torch.zeros(24, 4 * ops.HIDDEN)
    *_, ev = ops.mhc_params_async(x, w, None, None, rms_eps=1e-6, sinkhorn_eps=1e-6, iterations=20)
    assert ev is None and not calls


def test_split_parts_are_bound():
    bindings = (CSRC / "bindings.cpp").read_text()
    assert 'm.def("split_proj"' in bindings and 'm.def("split_tail"' in bindings
    src = (CSRC / "split_fwd.cu").read_text()
    body = src[src.index("at::Tensor split_proj_cuda"):src.index("std::vector<at::Tensor> split_tail_cuda")]
    assert "launch_proj<160, 1, true, 2>" in body                 # exactly split_forward's projection launch
    assert "pdl::launch(tail_kernel" in src[src.index("std::vector<at::Tensor> split_tail_cuda"):]


def test_attention_boundaries_overlap_wholly_up_to_2048_rows(monkeypatch):
    assert ops.OVERLAP_ATTN_MAX_ROWS == 2048
    seen = []
    monkeypatch.setattr(ops, "_params_tail_overlapped", lambda *a: seen.append("tail") or (None, None, None, None))
    monkeypatch.setattr(ops, "mhc_params", lambda *a, **k: seen.append("in-order") or (None, None, None))
    def whole(*a):                         # the whole-on-side-stream path starts by reading the caller's stream
        seen.append("whole")
        raise StopIteration
    monkeypatch.setattr(torch.cuda, "current_stream", whole)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
    w = torch.zeros(24, 4 * ops.HIDDEN)
    for rows, sub in ((2048, "attn"), (2049, "attn"), (2048, "ffn"), (32, "ffn")):
        x = torch.zeros(rows, 4, ops.HIDDEN, dtype=torch.bfloat16)
        try:
            with torch.no_grad():
                ops.mhc_params_async(x, w, None, None, rms_eps=1e-6, sinkhorn_eps=1e-6, iterations=20, sublayer=sub)
        except StopIteration:
            pass
    assert seen == ["whole", "in-order", "tail", "whole"]
