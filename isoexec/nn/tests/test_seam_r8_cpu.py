"""Compare decoder residual seams with the direct pre-norm reference using aten traces and exact
output bits. Cases cover bf16 attention and fp32 MLP outputs before the residual round.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from isoexec.nn import DecoderBlock, place, resolved_ops
from isoexec.ops.norms.reference import ref_rms_zero_centered
from isoexec.ops.residual.reference import ref_residual_add
from isoexec.ops.table import OpTable

from isoexec.nn.tests import _fixtures as F

T = 11


class _AtenLog(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket))
        return func(*args, **(kwargs or {}))


class _Stub(nn.Module):
    """A stand-in sublayer producing the declared boundary dtype (both call shapes: the mixer is
    called ``(x, seq)``, the MLP ``(x, seq, aux)``).

    A stub and not a real mixer on purpose: the subject is the SEAM, and a real mixer would put its
    own hundred ops into the trace and hide the four this file is about.
    """

    def __init__(self, weight: torch.Tensor, out_dtype: torch.dtype) -> None:
        super().__init__()
        self.w, self.out_dtype = weight, out_dtype

    def forward(self, x, seq=None, aux=None):
        return (x.float() @ self.w.float().t()).to(self.out_dtype)


def _trace(fn):
    with _AtenLog() as log:
        out = fn()
    return log.ops, out


def _block(mlp_out_dtype: torch.dtype):
    """A placed ``nn.DecoderBlock`` whose two residual adds are real resolutions.

    Placement is what fills the two seam slots, and it is the whole point of tracing this class
    rather than a bare ``h + y``: the row runs ``residual.add.attn`` and ``residual.add.mlp`` as
    ops with owners a peer comparison can read, and R8 says that costs the graph nothing.
    """
    torch.manual_seed(0)
    H, eps = F.HIDDEN, F.EPS
    h = (torch.randn(T, H) * 0.7).to(torch.bfloat16)
    w_attn = (torch.randn(H) * 0.05).to(torch.bfloat16)
    w_mlp = (torch.randn(H) * 0.05).to(torch.bfloat16)
    mixer = _Stub((torch.randn(H, H) * 0.05).to(torch.bfloat16), torch.bfloat16)
    mlp = _Stub((torch.randn(H, H) * 0.05).to(torch.bfloat16), mlp_out_dtype)
    block = DecoderBlock(mixer=mixer, mlp=mlp, hidden=H, eps=eps, device="cpu")
    place(block, F.placement(), prefix=F.NS, op_table=OpTable(mode="reference"))
    for p in block.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        block.pre_attn_norm.weight.copy_(w_attn)
        block.pre_mlp_norm.weight.copy_(w_mlp)
    return block, h, (mixer, mlp), (w_attn, w_mlp), eps


def _run(mlp_out_dtype: torch.dtype):
    block, h, (mixer, mlp), (w_attn, w_mlp), eps = _block(mlp_out_dtype)

    def reference():
        y = ref_residual_add(h, mixer(ref_rms_zero_centered(h, w_attn, eps)))
        return ref_residual_add(y, mlp(ref_rms_zero_centered(y, w_mlp, eps)))

    ops_seam, out_seam = _trace(lambda: block(h, None))
    ops_ref, out_ref = _trace(reference)
    return ops_seam, out_seam, ops_ref, out_ref


@pytest.mark.parametrize("mlp_out_dtype", [torch.bfloat16, torch.float32])
def test_seam_matches_direct_composition_without_tensor_movement(mlp_out_dtype):
    ops_seam, out_seam, ops_ref, out_ref = _run(mlp_out_dtype)
    assert ops_seam == ops_ref, (ops_seam, ops_ref)
    assert out_seam.dtype == torch.bfloat16
    assert torch.equal(out_seam.view(torch.int16), out_ref.view(torch.int16))
    assert any("rms_norm" in op or "rsqrt" in op for op in ops_seam)
    assert "aten.add" in ops_seam
    if mlp_out_dtype == torch.float32:
        assert "aten._to_copy" in ops_seam  # the declared boundary cast
    assert not set(ops_seam) & {"aten.cat", "aten.copy_", "aten.clone", "aten.contiguous"}


def test_the_seam_holds_two_gammas_and_runs_exactly_two_ops_of_its_own():
    """The block owns its two norm children and two residual operations, without additional tensor
    movement.
    """
    block, *_ = _block(torch.bfloat16)
    own = [n for n, _ in block.named_parameters() if not n.startswith(("mixer.", "mlp."))]
    assert own == ["pre_attn_norm.weight", "pre_mlp_norm.weight"]
    assert dict(block.tensor_axes()) == {}
    seams = resolved_ops(block)[F.NS]
    assert [r.op for r in seams.values()] == ["residual.add.attn", "residual.add.mlp"]
    assert block.slots.unfilled() == {}
