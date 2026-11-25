"""norms_sm90 recipe v1: bitwise invariance gates, recipe conformance and backward correctness.

Every equality claim compares INTEGER VIEWS (``ops/_testing.bitwise_equal``), never ``torch.equal``,
which reports ``+0.0 == -0.0``. Hazards are asserted to have fired.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.norms.norms_sm90 import ops as O
from isoexec.ops.norms.norms_sm90 import reference as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

EPS = 1e-6
# Production widths: decoder hidden (Qwen3.5 35B/9B, Qwen3.6 27B, GLM, MiMo), GLM MLA q_a / kv_a,
# GDN value head. 384 is the narrow-lane path's second width.
WIDTHS = (2048, 4096, 5120, 768, 512, 128, 384)


def rows(m, n, *, scale=3.0, seed=0, device="cuda"):
    """Wide-range bf16 rows with signed zeros, bf16 subnormals and a cancelling row injected."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(m, n, generator=g, device=device) * scale).bfloat16()
    flat = x.view(-1)
    if flat.numel() > 64:
        flat[::997] = torch.tensor(-0.0, dtype=torch.bfloat16, device=device)
        flat[13::991] = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
        sub = torch.tensor([1, 2, 5, 7], dtype=torch.int16, device=device).view(torch.bfloat16)
        flat[29:33] = sub
        flat[41:45] = -sub
    return x


def weight(n, *, zero_centered, seed=1, dtype=torch.bfloat16, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(n, generator=g, device=device) * 0.5
    return (w if zero_centered else 1.0 + w).to(dtype)


def hazards(x):
    """Witnesses for the value hazards a row battery must actually contain."""
    bits = x.view(torch.int16)
    return {
        "signed_zero": int((bits == torch.tensor(-32768, dtype=torch.int16)).sum()),
        "subnormal": int(((bits & 0x7F80) == 0).sum() & 0xFFFFFFFF) - int((x == 0).sum()),
        "zero": int((x == 0).sum()),
    }


# 1. the recipe: the kernels reproduce the executable statement of recipe v1, bit for bit
@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_matches_the_declared_recipe(n, zero_centered):
    x, w = rows(257, n), weight(n, zero_centered=zero_centered)
    h = hazards(x)
    assert_hazard_exercised("signed_zero", h["signed_zero"] > 0)
    assert_hazard_exercised("subnormal", h["subnormal"] > 0)
    assert bitwise_equal(O.rms_norm(x, w, EPS, zero_centered), R.rms_norm(x, w, EPS, zero_centered))


def test_matches_the_declared_recipe_gated():
    x, z = rows(129 * 8, 128).view(129, 8, 128), rows(129 * 8, 128, scale=12.0, seed=5).view(129, 8, 128)
    w = weight(128, zero_centered=True)
    assert bitwise_equal(O.gated_rms_norm(x, z, w, EPS, True), R.gated_rms_norm(x, z, w, EPS, True))


def test_matches_the_declared_recipe_add():
    x, r = rows(129, 2048), rows(129, 2048, seed=3, scale=4.0)
    w = weight(2048, zero_centered=True)
    y, h = O.add_rms_norm(x, r, w, EPS, True)
    ry, rh = R.add_rms_norm(x, r, w, EPS, True)
    assert bitwise_equal(h, rh) and bitwise_equal(y, ry)
    # the stored residual stream is torch's own bf16 add, so a host may keep either
    assert bitwise_equal(h, x + r)
    # ... and the fused form's norm is the norm of that stream
    assert bitwise_equal(y, O.rms_norm(h, w, EPS, True))


# 2. batch invariance: a row's bits never depend on its batchmates, its position, or the batch size
@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_batch_invariance(n, zero_centered):
    m = 4096
    x, w = rows(m, n), weight(n, zero_centered=zero_centered)
    full = O.rms_norm(x, w, EPS, zero_centered)
    for start, size in ((0, 1), (1, 1), (7, 3), (100, 64), (2048, 1000), (m - 1, 1), (0, m)):
        sl = x[start : start + size].clone()  # a fresh, differently aligned batch
        assert bitwise_equal(O.rms_norm(sl, w, EPS, zero_centered), full[start : start + size]), (start, size)


@pytest.mark.parametrize("n", (2048, 5120, 128))
def test_null_lanes_and_padding_are_inert(n):
    """Rows appended to a batch (graph-capture padding) change nothing about the live rows, whatever
    they contain -- including NaN, inf and whole-zero padding."""
    x, w = rows(300, n), weight(n, zero_centered=True)
    live = O.rms_norm(x, w, EPS, True)
    for fill in (float("nan"), float("inf"), 0.0, -0.0):
        pad = torch.full((212, n), fill, device="cuda", dtype=torch.bfloat16)
        padded = O.rms_norm(torch.cat([x, pad]), w, EPS, True)
        assert bitwise_equal(padded[:300], live), fill
    assert_hazard_exercised("null_lane", True)


def test_zero_row_is_defined_and_invariant():
    """An all-zero row normalizes through ``rsqrt(eps)`` and is unaffected by its neighbours."""
    n = 2048
    x, w = rows(64, n), weight(n, zero_centered=True)
    x[7] = 0.0
    x[8] = -0.0
    y = O.rms_norm(x, w, EPS, True)
    assert bitwise_equal(y, R.rms_norm(x, w, EPS, True))
    alone = O.rms_norm(x[7:9].clone(), w, EPS, True)
    assert bitwise_equal(alone, y[7:9])


# 3. phase parity: decode == prefill == trainer forward, and every execution plan agrees
@pytest.mark.parametrize("n", (2048, 5120, 128))
def test_decode_prefill_trainer_parity(n):
    x, w = rows(20480, n), weight(n, zero_centered=True)
    ref = O.rms_norm(x, w, EPS, True)
    for m in (1, 8, 64, 256, 512, 2048, 4096, 20480):  # decode ladder, prefill chunk, trainer batch
        assert bitwise_equal(O.rms_norm(x[:m].clone(), w, EPS, True), ref[:m]), m
    # trainer forward (grad mode, autograd Function) == engine forward (no_grad)
    xg = x[:2048].detach().clone().requires_grad_(True)
    wg = w.detach().clone().requires_grad_(True)
    assert bitwise_equal(O.rms_norm(xg, wg, EPS, True).detach(), ref[:2048])
    with torch.no_grad():
        assert bitwise_equal(O.rms_norm(x[:2048].clone(), w, EPS, True), ref[:2048])
    assert_hazard_exercised("phase_parity", True)


@pytest.mark.parametrize("hv", (4, 6, 8))  # per-rank GDN value heads at TP 8 / 8 / 4
def test_gated_tp_head_shapes(hv):
    t, dv = 1024, 128
    x = rows(t * 8, dv, seed=2).view(t, 8, dv)
    z = rows(t * 8, dv, seed=9, scale=8.0).view(t, 8, dv)
    w = weight(dv, zero_centered=True)
    full = O.gated_rms_norm(x, z, w, EPS, True).view(t, 8, dv)
    sl = O.gated_rms_norm(x[:, :hv].contiguous(), z[:, :hv].contiguous(), w, EPS, True).view(t, hv, dv)
    assert bitwise_equal(sl, full[:, :hv])
    assert_hazard_exercised("tp_head_slicing", True)


def test_shape_of_the_input_view_does_not_matter():
    """[T, H], [B, S, H] and [T*B, H] are the same rows."""
    n = 2048
    x, w = rows(512, n), weight(n, zero_centered=False)
    flat = O.rms_norm(x, w, EPS, False)
    three_d = O.rms_norm(x.view(8, 64, n), w, EPS, False)
    assert bitwise_equal(three_d.reshape(512, n), flat)


# 4. degenerate and awkward inputs
def test_split_view_rows_match_contiguous():
    """GLM's kv_a norm reads a [T, kv+rope] slice: a non-contiguous row stride, unit column stride."""
    t, n = 777, 512
    base = rows(t, n + 64, seed=4)
    view = base[:, :n]
    w = weight(n, zero_centered=False)
    assert bitwise_equal(O.rms_norm(view, w, EPS, False), O.rms_norm(view.contiguous(), w, EPS, False))
    assert_hazard_exercised("non_contiguous_views", not view.is_contiguous())


def test_misaligned_storage_matches():
    """A view whose base pointer is not 16-byte aligned (odd element offset) -- the host copies it."""
    n = 2048
    buf = rows(65, n + 8, seed=6)
    view = buf[:, 1 : n + 1]
    w = weight(n, zero_centered=True)
    assert bitwise_equal(O.rms_norm(view, w, EPS, True), O.rms_norm(view.contiguous(), w, EPS, True))
    assert_hazard_exercised("misaligned_storage", (view.data_ptr() % 16) != 0)


def test_strided_gate_view_matches():
    """The GDN gate is a slice of the in_proj output: free token and head strides, unit feature stride."""
    t, hv, dv = 301, 8, 128
    proj = rows(t, 3000 + hv * dv, seed=7)
    z = proj[:, 3000:].view(t, hv, dv)
    x = rows(t * hv, dv, seed=8).view(t, hv, dv)
    w = weight(dv, zero_centered=True)
    assert bitwise_equal(
        O.gated_rms_norm(x, z, w, EPS, True), O.gated_rms_norm(x, z.contiguous(), w, EPS, True)
    )


def test_gate_with_a_4_byte_aligned_token_stride():
    """Qwen3.6's per-rank gate (hv=6): the in_proj row is 4-byte but not 16-byte aligned, so the
    kernel reads it with 32-bit loads instead of the host copying it."""
    t, hv, dv = 301, 6, 128
    proj = rows(t, 3000 + hv * dv + 2 * hv, seed=23)
    z = proj[:, 3000 : 3000 + hv * dv].view(t, hv, dv)
    assert (z.stride(0) * 2) % 16 == 8 and (z.data_ptr() % 4) == 0  # the case this test exists for
    x = rows(t * hv, dv, seed=24).view(t, hv, dv)
    w = weight(dv, zero_centered=True)
    out = O.gated_rms_norm(x, z, w, EPS, True)
    assert bitwise_equal(out, O.gated_rms_norm(x, z.contiguous(), w, EPS, True))
    assert bitwise_equal(out, R.gated_rms_norm(x, z, w, EPS, True))


def test_fp32_weight_matches_bf16_value():
    """A float32 weight tensor holding exactly the bf16 values gives the same bits."""
    n = 2048
    x = rows(129, n)
    wb = weight(n, zero_centered=True)
    assert bitwise_equal(O.rms_norm(x, wb.float(), EPS, True), O.rms_norm(x, wb, EPS, True))


def test_empty_batch():
    n = 2048
    w = weight(n, zero_centered=True)
    y = O.rms_norm(torch.empty(0, n, device="cuda", dtype=torch.bfloat16), w, EPS, True)
    assert y.shape == (0, n)


def test_unsupported_width_refuses():
    assert not O.supported_width(100)
    with pytest.raises(Exception):
        O.rms_norm(rows(4, 100), weight(100, zero_centered=True), EPS, True)


# 5. CUDA graphs: capture and replay produce the eager bits, with no host sync
@pytest.mark.parametrize("n", (2048, 128))
def test_graph_replay_is_bitwise(n):
    m = 512
    x, w = rows(m, n), weight(n, zero_centered=True)
    eager = O.rms_norm(x, w, EPS, True)
    static = x.clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            O.rms_norm(static, w, EPS, True)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = O.rms_norm(static, w, EPS, True)
    for _ in range(3):
        static.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        assert bitwise_equal(out, eager)
    # a replay over different data still matches eager on that data
    x2 = rows(m, n, seed=11)
    static.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert bitwise_equal(out, O.rms_norm(x2, w, EPS, True))
    assert_hazard_exercised("graph_replay", True)


def test_graph_replay_gated_and_backward():
    t, hv, dv = 256, 8, 128
    x = rows(t * hv, dv, seed=12).view(t, hv, dv).requires_grad_(True)
    z = rows(t * hv, dv, seed=13, scale=6.0).view(t, hv, dv).requires_grad_(True)
    w = weight(dv, zero_centered=True).requires_grad_(True)
    gout = rows(t, hv * dv, seed=14)
    eager = torch.autograd.grad(O.gated_rms_norm(x, z, w, EPS, True), (x, z, w), gout)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            torch.autograd.grad(O.gated_rms_norm(x, z, w, EPS, True), (x, z, w), gout)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        grads = torch.autograd.grad(O.gated_rms_norm(x, z, w, EPS, True), (x, z, w), gout)
    g.replay()
    torch.cuda.synchronize()
    for a, b in zip(grads, eager):
        assert bitwise_equal(a, b)


# 6. backward: correct against float64 autograd, for every differentiable input
def _rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("n", (2048, 5120, 768, 128))
@pytest.mark.parametrize("zero_centered", [False, True])
def test_backward_rms_matches_float64(n, zero_centered):
    m = 1025
    x = rows(m, n).requires_grad_(True)
    w = weight(n, zero_centered=zero_centered).requires_grad_(True)
    gout = rows(m, n, seed=15)
    dx, dw = torch.autograd.grad(O.rms_norm(x, w, EPS, zero_centered), (x, w), gout)
    xd = x.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    g = (1.0 + wd) if zero_centered else wd
    yd = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * g
    dxd, dwd = torch.autograd.grad(yd, (xd, wd), gout.double())
    assert _rel(dx, dxd) < 5e-3, _rel(dx, dxd)
    assert _rel(dw, dwd) < 5e-3, _rel(dw, dwd)


def test_backward_add_rms_returns_x_residual_and_weight_grads():
    n, m = 2048, 513
    x = rows(m, n).requires_grad_(True)
    r = rows(m, n, seed=16, scale=4.0).requires_grad_(True)
    w = weight(n, zero_centered=True).requires_grad_(True)
    gy, gh = rows(m, n, seed=17), rows(m, n, seed=18, scale=0.5)
    y, h = O.add_rms_norm(x, r, w, EPS, True)
    dx, dr, dw = torch.autograd.grad((y, h), (x, r, w), (gy, gh))
    xd = x.detach().double().requires_grad_(True)
    rd = r.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    hd = xd + rd
    yd = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wd)
    dxd, drd, dwd = torch.autograd.grad((yd, hd), (xd, rd, wd), (gy.double(), gh.double()))
    assert _rel(dx, dxd) < 5e-3 and _rel(dr, drd) < 5e-3 and _rel(dw, dwd) < 5e-3
    assert bitwise_equal(dx, dr)  # both branches of the residual carry the same gradient


def test_backward_gated_matches_float64():
    t, hv, dv = 513, 8, 128
    x = rows(t * hv, dv, seed=19).view(t, hv, dv).requires_grad_(True)
    z = rows(t * hv, dv, seed=20, scale=6.0).view(t, hv, dv).requires_grad_(True)
    w = weight(dv, zero_centered=True).requires_grad_(True)
    gout = rows(t, hv * dv, seed=21)
    dx, dz, dw = torch.autograd.grad(O.gated_rms_norm(x, z, w, EPS, True), (x, z, w), gout)
    xd = x.detach().double().requires_grad_(True)
    zd = z.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    yd = (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wd)
          * torch.nn.functional.silu(zd)).reshape(t, hv * dv)
    dxd, dzd, dwd = torch.autograd.grad(yd, (xd, zd, wd), gout.double())
    assert _rel(dx, dxd) < 5e-3 and _rel(dz, dzd) < 5e-3 and _rel(dw, dwd) < 5e-3


@pytest.mark.parametrize("bands", (1, 4, 64, 4096))
def test_backward_plan_does_not_change_correctness(bands):
    """The backward's band count is an execution choice: every plan stays inside tolerance."""
    from isoexec.ops.norms.norms_sm90._extension import extension

    n, m = 2048, 3000
    x, w = rows(m, n), weight(n, zero_centered=True)
    _, _, rstd = extension().raw_rms_fwd(x, None, w, EPS, True)
    gout = rows(m, n, seed=22)
    dx, dw = extension().raw_rms_bwd(gout, x, rstd, w, None, True, bands)
    xd = x.double().requires_grad_(True)
    wd = w.double().requires_grad_(True)
    yd = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + wd)
    dxd, dwd = torch.autograd.grad(yd, (xd, wd), gout.double())
    assert _rel(dx, dxd) < 5e-3 and _rel(dw, dwd) < 5e-3


def test_no_grad_forward_does_not_build_a_graph():
    n = 2048
    x = rows(64, n).requires_grad_(True)
    w = weight(n, zero_centered=True).requires_grad_(True)
    with torch.no_grad():
        assert O.rms_norm(x, w, EPS, True).grad_fn is None
    assert O.rms_norm(x, w, EPS, True).grad_fn is not None


@pytest.mark.parametrize("family", ("rms", "gated"))
def test_weight_gradient_is_bitwise_repeatable(family):
    """Replicated weights require equal gradients; CTA arrival order must be irrelevant.

    FP32 weights expose the full accumulation result rather than hiding differences
    behind a BF16 cast. The two-rank probe also tests production BF16 weights.
    """
    t, n = 20480, 2048 if family == "rms" else 128
    x = rows(t if family == "rms" else t * 8, n, seed=29).requires_grad_(True)
    w = weight(n, zero_centered=True, dtype=torch.float32).requires_grad_(True)
    if family == "gated":
        x = x.detach().view(t, 8, n).requires_grad_(True)
        z = rows(t * 8, n, seed=30).view(t, 8, n).requires_grad_(True)
    gout = rows(t, n if family == "rms" else 8 * n, seed=31)
    first = None
    for _ in range(20):
        y = (O.rms_norm(x, w, EPS, True) if family == "rms"
             else O.gated_rms_norm(x, z, w, EPS, True))
        dw, = torch.autograd.grad(y, (w,), gout)
        assert torch.isfinite(dw).all()
        if first is None:
            first = dw.clone()
        else:
            assert bitwise_equal(dw, first)


def test_backward_gated_without_weight():
    x = rows(33 * 3, 128, seed=32).view(33, 3, 128).requires_grad_(True)
    z = rows(33 * 3, 128, seed=33).view(33, 3, 128).requires_grad_(True)
    dy = rows(33, 3 * 128, seed=34)
    # Exercise MODE0 directly; the optional-weight C++ autograd wrapper has a
    # separate existing undefined-Tensor limitation, outside this dW change.
    from isoexec.ops.norms.norms_sm90._extension import extension

    _, rstd = extension().raw_gated_fwd(x, z, None, EPS, True)
    dx, dz, dw = extension().raw_gated_bwd(dy, x, z, rstd, None, True, 0)
    assert dw is None
    xd, zd = x.detach().double().requires_grad_(True), z.detach().double().requires_grad_(True)
    yd = (xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
          * torch.nn.functional.silu(zd)).reshape(33, 3 * 128)
    dxd, dzd = torch.autograd.grad(yd, (xd, zd), dy.double())
    assert _rel(dx, dxd) < 5e-3 and _rel(dz, dzd) < 5e-3
