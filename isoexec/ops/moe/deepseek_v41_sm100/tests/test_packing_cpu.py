"""CPU contract of the trainer's stacked expert operands: a training op saves the source parameters instead of the
stacked copy, re-stacks the same bytes in backward, and the gradients are those of saving the copy."""
import gc
import weakref

import pytest
import torch

from isoexec.ops.moe.deepseek_v41_sm100.packing import (EXPERT_PARAMS_ATTR, save_operands, saved_operands,
                                                        stacked)


def _params(seed=0, experts=3, rows=4, cols=5):
    g = torch.Generator().manual_seed(seed)
    w1 = [torch.nn.Parameter(torch.randn(rows, cols, generator=g).bfloat16()) for _ in range(experts)]
    w3 = [torch.nn.Parameter(torch.randn(rows, cols, generator=g).bfloat16()) for _ in range(experts)]
    w2 = [torch.nn.Parameter(torch.randn(cols, rows, generator=g).bfloat16()) for _ in range(experts)]
    return w1, w3, w2


def _operands(w1, w3, w2, tag):
    gu = [p for a, b in zip(w1, w3) for p in (a, b)]
    G, (I, H) = len(w1), w1[0].shape
    if tag:
        return stacked(gu, (G, 2 * I, H), 2), stacked(w2, (G, H, I), 1)
    return torch.stack(gu).view(G, 2 * I, H), torch.stack(w2)


SEEN = {}


class _Op(torch.autograd.Function):
    """A stand-in training op: y = sum_g x @ gu[g].T ..., with a backward that reads its saved operands."""

    @staticmethod
    def forward(ctx, x, gu, dn):
        save_operands(ctx, x, gu, dn)
        SEEN["fwd"] = (gu.clone(), dn.clone())
        return torch.einsum("th,gih->tgi", x.float(), gu.float()).sum(1)

    @staticmethod
    def backward(ctx, dy):
        x, gu, dn = saved_operands(ctx)
        SEEN["bwd"] = (gu, dn)
        d_gu = torch.einsum("ti,th->ih", dy, x.float()).expand(gu.shape).to(gu.dtype)
        d_gu = d_gu * torch.arange(1, gu.shape[0] + 1, dtype=gu.dtype).view(-1, 1, 1)
        d_dn = (dn.float() * dy.sum()).to(dn.dtype)
        dx = torch.einsum("ti,gih->th", dy, gu.float()).to(x.dtype)
        return dx, d_gu, d_dn


def _run(tag):
    w1, w3, w2 = _params()
    x = torch.randn(7, 5, generator=torch.Generator().manual_seed(1)).bfloat16().requires_grad_(True)
    gu, dn = _operands(w1, w3, w2, tag)
    y = _Op.apply(x, gu, dn)
    ref = weakref.ref(gu)
    del gu, dn
    gc.collect()
    kept = ref() is not None
    y.sum().backward()
    return kept, [p.grad for p in (*w1, *w3, *w2)] + [x.grad]


def test_stacked_operand_carries_its_parameters_and_equals_the_stack():
    w1, w3, w2 = _params()
    gu, dn = _operands(w1, w3, w2, True)
    ref_gu, ref_dn = _operands(w1, w3, w2, False)
    assert torch.equal(gu, ref_gu) and torch.equal(dn, ref_dn)
    assert getattr(gu, EXPERT_PARAMS_ATTR) == tuple(zip(w1, w3))
    assert getattr(dn, EXPERT_PARAMS_ATTR) == tuple((p,) for p in w2)
    assert gu.data_ptr() not in {p.data_ptr() for p in (*w1, *w3)}      # a copy: the parameters stay put


def test_backward_restacks_the_same_bytes_and_the_copy_is_not_kept():
    kept_tagged, grads_tagged = _run(True)
    fwd, bwd = SEEN["fwd"], SEEN["bwd"]
    assert torch.equal(fwd[0], bwd[0]) and torch.equal(fwd[1], bwd[1])
    kept_plain, grads_plain = _run(False)
    assert not kept_tagged             # freed after the forward: only the parameters are saved
    assert kept_plain                  # an untagged operand is saved as itself
    for a, b in zip(grads_tagged, grads_plain, strict=True):
        assert a is not None and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_in_place_update_between_forward_and_backward_is_refused():
    w1, w3, w2 = _params()
    x = torch.randn(7, 5).bfloat16().requires_grad_(True)
    gu, dn = _operands(w1, w3, w2, True)
    y = _Op.apply(x, gu, dn)
    with torch.no_grad():
        w3[1].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


def test_frozen_experts_get_no_gradient_and_the_trainable_ones_theirs():
    w1, w3, w2 = _params()
    for p in (w1[0], w3[0], w2[0], w1[1], w3[1], w2[1]):
        p.requires_grad_(False)                       # routed frozen, the last ("shared") trainable
    x = torch.randn(7, 5).bfloat16()
    gu, dn = _operands(w1, w3, w2, True)
    _Op.apply(x, gu, dn).sum().backward()
    assert all(p.grad is None for p in (w1[0], w3[0], w2[0], w1[1], w3[1], w2[1]))
    assert all(p.grad is not None for p in (w1[2], w3[2], w2[2]))
