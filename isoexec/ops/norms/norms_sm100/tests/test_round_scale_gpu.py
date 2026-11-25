import re
import subprocess
from pathlib import Path

import pytest
import torch

from isoexec.ops.norms.norms_sm100 import ops

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
WIDTHS = (512, 1536, 2048, 4096, 5120, 7168)


@needs_gpu
@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("rows", [1, 4, 54, 513, 4096])
def test_matches_composition(n, rows):
    g = torch.Generator(device="cuda").manual_seed(n + rows)
    x = (torch.randn(rows, n, device="cuda", generator=g) * 3).bfloat16()
    w = torch.randn(n, device="cuda", generator=g).bfloat16()
    ref = w * ops.rms_norm(x, torch.ones(n, dtype=torch.bfloat16, device="cuda"), 1e-5, False)
    got = ops.rms_norm_round_scale(x, w, 1e-5)
    assert torch.equal(got, ref) and torch.equal(got, ops.rms_norm_round_scale(x, w, 1e-5))
    for i in (0, rows - 1):                       # batch invariance
        assert torch.equal(ops.rms_norm_round_scale(x[i:i + 1].contiguous(), w, 1e-5), got[i:i + 1])
    x3 = x.view(rows, 1, n)
    assert torch.equal(ops.rms_norm_round_scale(x3, w, 1e-5), got.view(rows, 1, n))


@needs_gpu
def test_graph_replay_and_forward_only():
    n = 4096
    x = torch.randn(4, n, device="cuda").bfloat16()
    w = torch.randn(n, device="cuda").bfloat16()
    ops.rms_norm_round_scale(x, w, 1e-5)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = ops.rms_norm_round_scale(x, w, 1e-5)
    x.copy_(torch.randn_like(x))
    g.replay()
    assert torch.equal(y, w * ops.rms_norm(x, torch.ones(n, dtype=torch.bfloat16, device="cuda"), 1e-5, False))
    with pytest.raises(RuntimeError):
        ops.rms_norm_round_scale(x.requires_grad_(), w, 1e-5)


def test_forward_kernels_have_no_atomics_cpu():
    from isoexec.ops.norms.norms_sm100._extension import extension

    out = subprocess.run(["cuobjdump", "-sass", str(Path(extension().__file__))], capture_output=True, text=True,
                         check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur:
            funcs[cur].append(line)
    fwd = {k: v for k, v in funcs.items() if "rms_fwd" in k}
    assert fwd
    for name, lines in fwd.items():
        assert not [ln for ln in lines if re.search(r"\bATOMG?\b|\bATOMS?\.|\bRED\b|\bRED\.", ln)], name
