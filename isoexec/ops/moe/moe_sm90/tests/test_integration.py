"""Unified provider: grad/no-grad parity, ETP leaf composition, and all four VJPs."""

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.moe.carriers import RoutingResult
from isoexec.ops.moe.moe_sm90 import ops
from isoexec.ops.moe.moe_sm90.integration import _Experts, expert_mlp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm90", reason="requires an SM90 (H100) device"
)


def bits(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 33, 257])
@pytest.mark.parametrize("weight_after_fc2", [False, True])
@pytest.mark.parametrize("hidden,intermediate", [(128, 128), (2048, 1408)], ids=["small", "qwen15"])
def test_training_forward_and_backward(rows, weight_after_fc2, hidden, intermediate):
    torch.manual_seed(71)
    E, H, I = 4, hidden, intermediate
    x = (torch.randn(rows, H, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    p = torch.rand(rows, device="cuda", requires_grad=True)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", dtype=x.dtype) * 0.1).requires_grad_()
    w2 = (torch.randn(E, H, I, device="cuda", dtype=x.dtype) * 0.1).requires_grad_()
    offsets = torch.tensor([0, rows // 3, rows // 3, rows // 2, rows], device="cuda", dtype=torch.int32)
    plan = ops.make_plan(offsets, rows)
    out = _Experts.apply(x, p, w1, w2, plan, 8, weight_after_fc2, False)
    with torch.no_grad():
        inter = ops.fc1(x, w1, plan)
        activation = ops.fc1_glu(x, w1, plan, None if weight_after_fc2 else p)
        root = ops.fc2(activation, w2, plan, leaves=8)
        expected = root
        if weight_after_fc2:
            expected = (expected.to(x.dtype) * p[:, None].to(x.dtype)).float()
        bits(out, expected)
    dy = torch.randn_like(out) * 0.1
    grads = torch.autograd.grad(out, (x, p, w1, w2), dy)
    # Independent per-expert PyTorch VJP at the actual rounded operands.
    # FC2 merges eight BF16 leaves; a single cuBLAS GEMM has different rounding
    # and therefore is not the operand used by the probability multiplication.
    refs = [torch.zeros_like(t) for t in (x, p, w1, w2)]
    for e, (lo, hi) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        if lo == hi:
            continue
        xx, pp = x[lo:hi].detach().requires_grad_(), p[lo:hi].detach().requires_grad_()
        a, b = w1[e].detach().requires_grad_(), w2[e].detach().requires_grad_()
        linear = xx @ a.t()
        raw = inter[lo:hi] + (linear - linear.detach())
        g, u = raw.chunk(2, -1)
        h = torch.nn.functional.silu(g) * u
        if not weight_after_fc2:
            h = (h.float() * pp[:, None]).to(x.dtype)
        h = activation[lo:hi] + (h - h.detach())
        y = h @ b.t()
        if weight_after_fc2:
            y = root[lo:hi].to(x.dtype) + (y - y.detach())
            y = y * pp[:, None].to(x.dtype)
        gx, gp, ga, gb = torch.autograd.grad(y, (xx, pp, a, b), dy[lo:hi].to(x.dtype))
        refs[0][lo:hi], refs[1][lo:hi], refs[2][e], refs[3][e] = gx, gp, ga, gb
    for got, ref in zip(grads, refs):
        assert torch.isfinite(got).all() and got.abs().sum() > 0
        torch.testing.assert_close(got.float(), ref.float(), atol=0.003, rtol=0.08)
    if weight_after_fc2:
        bits(grads[1], refs[1])
    assert torch.count_nonzero(grads[2][1]) == torch.count_nonzero(grads[3][1]) == 0


def test_route_and_etp_composition():
    torch.manual_seed(7)
    E, H, I, T = 8, 128, 512, 29
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=x.dtype) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=x.dtype) * 0.02
    idx = torch.randint(E, (T, 1), device="cuda", dtype=torch.int32)
    routing = RoutingResult(idx, torch.rand(T, 1, device="cuda"), None)
    with torch.no_grad():
        full = expert_mlp(x, routing, w1, w2)
        for etp in (2, 4, 8):
            parts = []
            for rank in range(etp):
                lo, hi = rank * I // etp, (rank + 1) * I // etp
                a = torch.cat((w1[:, lo:hi], w1[:, I + lo : I + hi]), dim=1)
                local = expert_mlp(x, routing, a, w2[:, :, lo:hi], n_leaves=8 // etp)
                parts.append(local.rows)
            while len(parts) > 1:
                parts = [parts[j] + parts[j + 1] for j in range(0, len(parts), 2)]
            bits(full.rows, parts[0])


@pytest.mark.parametrize("weight_after_fc2", [False, True])
def test_expert_wgrad_accumulates_into_main_grad(weight_after_fc2, monkeypatch):
    """Gradient-accumulation fusion for the grouped expert GEMMs: each expert's gate/up/down block lands in
    its own fp32 main_grad (Megatron's DDP contract); dx and dprobs are unchanged bit for bit."""
    from isoexec.ops.mm.wgrad import EXPERT_PARAMS_ATTR

    torch.manual_seed(5)
    E, H, I, rows = 4, 256, 192, 300
    x0 = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16) * 0.1
    p0 = torch.rand(rows, device="cuda")
    offsets = torch.tensor([0, 100, 100, 180, rows], device="cuda", dtype=torch.int32)
    plan = ops.make_plan(offsets, rows)
    experts = [tuple(torch.nn.Parameter(torch.randn(s, device="cuda", dtype=torch.bfloat16) * 0.1)
                     for s in ((I, H), (I, H), (H, I))) for _ in range(E)]
    accs = []
    for params in experts:
        for q in params:
            q.main_grad = torch.zeros(q.shape, device="cuda")
            q.grad_added_to_main_grad = False
            acc = q.expand_as(q).grad_fn.next_functions[0][0]

            def hook(*_, q=q):
                if q.grad is not None and not q.grad_added_to_main_grad:
                    q.main_grad.add_(q.grad.data)
                q.grad = None

            acc.register_hook(hook)
            accs.append(acc)
    dy = None

    def run(fused):
        nonlocal dy
        monkeypatch.setenv("ISOEXEC_WGRAD_FUSION", "1" if fused else "0")
        for params in experts:
            for q in params:
                q.main_grad.zero_()
        x, p = x0.clone().requires_grad_(), p0.clone().requires_grad_()
        w1 = torch.stack([torch.cat([g, u]) for g, u, _ in experts])
        w2 = torch.stack([d for _, _, d in experts])
        setattr(w1, EXPERT_PARAMS_ATTR, tuple((g, u) for g, u, _ in experts))
        setattr(w2, EXPERT_PARAMS_ATTR, tuple((d,) for _, _, d in experts))
        out = _Experts.apply(x, p, w1, w2, plan, 8, weight_after_fc2, False)
        if dy is None:
            dy = torch.randn_like(out) * 0.1
        out.backward(dy)
        return x.grad, p.grad, [tuple(q.main_grad.clone() for q in params) for params in experts]

    gx_f, gp_f, mg_f = run(True)
    gx_u, gp_u, mg_u = run(False)
    bits(gx_f, gx_u)
    bits(gp_f, gp_u)
    for e, (f, u) in enumerate(zip(mg_f, mg_u)):
        for a, b in zip(f, u):
            torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-2)
            if e == 1:  # no tokens
                assert torch.count_nonzero(a) == 0
    assert all(not q.grad_added_to_main_grad for params in experts for q in params)
