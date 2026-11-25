"""sm90 wgrad-accumulate GEMM: epilogue semantics, determinism, tile/TP independence, autograd contract."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

BF16, F32 = torch.bfloat16, torch.float32
# (N_out, K_in) at TP4 of Qwen3.8-27B (GDN in_proj, out_proj, attn qkv, fc1, fc2) plus ragged ones.
SHAPES = ((4120, 5120), (5120, 1536), (3584, 5120), (4352, 5120), (5120, 4352), (24, 40), (520, 136))
TOKENS = (1, 63, 64, 1000, 5801)


def _ops():
    from isoexec.ops.mm.mm_bwd_sm90 import ops

    return ops


def _operands(t, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dy = torch.randn(t, n, device="cuda", generator=gen).to(BF16)
    x = torch.randn(t, k, device="cuda", generator=gen).to(BF16)
    mg = torch.randn(n, k, device="cuda", generator=gen) * 4
    return dy, x, mg


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("t", TOKENS)
def test_epilogue_is_one_fp32_add(t, n, k):
    """``main_grad + chain`` rounded once: accumulating into zeros gives the chain, into mg0 gives mg0 + chain."""
    W = _ops()
    dy, x, mg0 = _operands(t, n, k, t * 7 + n)
    chain = W.accumulate(dy, x, torch.zeros(n, k, device="cuda"))
    got = W.accumulate(dy, x, mg0.clone())
    assert bitwise_equal(got, mg0 + chain)
    ref = dy.double().t() @ x.double()
    err = (chain.double() - ref).abs().max().item()
    assert err <= 1e-5 * max(1.0, ref.abs().max().item()) * max(1, t) ** 0.5, err


@pytest.mark.parametrize("n,k", SHAPES[:3])
def test_every_config_and_tile_order_same_bits(n, k):
    """Tile shape, cluster, swizzle and raster are speed only: every element is the same chain."""
    W = _ops()
    dy, x, mg0 = _operands(2900, n, k, 5)
    want = W.accumulate(dy, x, mg0.clone(), config="co_128x128", raster=0)
    for cfg in W.extension().config_names():
        for raster in (0, 1, 2):
            for swz in (1, 2, 8):
                got = W.accumulate(dy, x, mg0.clone(), config=cfg, raster=raster, swizzle=swz)
                assert bitwise_equal(got, want), (cfg, raster, swz)


def test_deterministic_run_to_run():
    W = _ops()
    dy, x, mg0 = _operands(5800, 4120, 5120, 9)
    a = W.accumulate(dy, x, mg0.clone())
    for _ in range(3):
        assert bitwise_equal(W.accumulate(dy, x, mg0.clone()), a)


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_tp_shard_is_block_of_unsharded(tp):
    """A column-parallel shard (rows of W) and a row-parallel shard (columns of W) get exactly the
    matching block of the TP1 gradient: TP-consistent bit for bit."""
    W = _ops()
    n, k = 8192, 4096
    dy, x, mg0 = _operands(1500, n, k, 13)
    full = W.accumulate(dy, x, mg0.clone())
    for r in range(tp):
        rows = slice(r * n // tp, (r + 1) * n // tp)
        cols = slice(r * k // tp, (r + 1) * k // tp)
        col_shard = W.accumulate(dy[:, rows].contiguous(), x, mg0[rows].clone())
        row_shard = W.accumulate(dy, x[:, cols].contiguous(), mg0[:, cols].contiguous())
        assert bitwise_equal(col_shard, full[rows])
        assert bitwise_equal(row_shard, full[:, cols])


def test_strided_main_grad_rows_of_a_bucket():
    """main_grad as adjacent rows of a larger fp32 buffer (a DDP bucket) and a strided dy column view."""
    W = _ops()
    dy_full, x, _ = _operands(777, 6000, 1536, 17)
    bucket = torch.randn(16 + 4120 * 1536 + 64, device="cuda")
    mg = bucket[16:16 + 4120 * 1536].view(4120, 1536)
    before = bucket.clone()
    dy = dy_full[:, 8:8 + 4120]  # row pitch 6000
    want = mg.clone() + W.accumulate(dy.contiguous(), x, torch.zeros(4120, 1536, device="cuda"))
    W.accumulate(dy, x, mg)
    assert bitwise_equal(mg, want)
    assert bitwise_equal(bucket[:16], before[:16]) and bitwise_equal(bucket[-64:], before[-64:])


def test_declines_unaligned():
    W = _ops()
    dy = torch.zeros(10, 13, device="cuda", dtype=BF16)
    x = torch.zeros(10, 16, device="cuda", dtype=BF16)
    assert not W.supported(dy, x, torch.zeros(13, 16, device="cuda"))
    assert not W.supported(x, x, torch.zeros(16, 16, device="cuda", dtype=BF16))


# ------------------------------------------------------------------------------- autograd contract
class _FakeDDP:
    """Megatron DDP's per-parameter contract: main_grad, grad_added_to_main_grad, the accumulator hook."""

    def __init__(self, params):
        self.accs = []
        for p in params:
            p.main_grad = torch.zeros(p.shape, dtype=F32, device=p.device)
            p.grad_added_to_main_grad = False
            acc = p.expand_as(p).grad_fn.next_functions[0][0]
            acc.register_hook(self._hook(p))
            self.accs.append(acc)

    @staticmethod
    def _hook(p):
        def hook(*_):
            if p.grad is not None and not p.grad_added_to_main_grad:
                p.main_grad.add_(p.grad.data)
            p.grad = None

        return hook


@pytest.mark.parametrize("fused", (True, False))
def test_mm_trainer_accumulates_into_main_grad(fused, monkeypatch):
    from isoexec.ops.mm.trainer import _MmFn

    monkeypatch.setenv("ISOEXEC_WGRAD_FUSION", "1" if fused else "0")
    gen = torch.Generator(device="cuda").manual_seed(3)
    w = torch.nn.Parameter(torch.randn(520, 136, device="cuda", generator=gen).to(BF16))
    ddp = _FakeDDP([w])
    xs = [torch.randn(300 + 7 * i, 136, device="cuda", generator=gen).to(BF16) for i in range(3)]
    dys = [torch.randn(x.shape[0], 520, device="cuda", generator=gen).to(BF16) for x in xs]
    for x, dy in zip(xs, dys):  # three micro-batches into one main_grad
        _MmFn.apply(x.requires_grad_(True), w).backward(dy)
        assert w.grad is None and w.grad_added_to_main_grad is False  # re-armed after the hook ran
    want = torch.zeros(520, 136, device="cuda")
    for x, dy in zip(xs, dys):
        if fused:
            from isoexec.ops.mm.mm_bwd_sm90 import ops as W

            W.accumulate(dy, x.detach(), want)
        else:
            want.add_(dy.t() @ x.detach())
    assert bitwise_equal(w.main_grad, want)
    del ddp


def test_fused_then_unfused_contribution_is_not_dropped():
    """A gradient that bypasses the fused GEMM in the same backward (here: a second, unfused use)
    still reaches main_grad once the flag is re-armed between backward passes."""
    from isoexec.ops.mm.trainer import _MmFn
    from isoexec.ops.mm.wgrad import OPT_OUT_ATTR

    w = torch.nn.Parameter(torch.randn(64, 32, device="cuda").to(BF16))
    ddp = _FakeDDP([w])  # noqa: F841 -- keeps the grad accumulator (and its hook) alive, as DDP does
    x = torch.randn(16, 32, device="cuda").to(BF16)
    dy = torch.randn(16, 64, device="cuda").to(BF16)
    _MmFn.apply(x, w).backward(dy)  # fused
    fused = w.main_grad.clone()
    setattr(w, OPT_OUT_ATTR, True)
    _MmFn.apply(x, w).backward(dy)  # unfused: the DDP hook must add it
    unfused = torch.ops.aten.mm.out(dy.t(), x, out=torch.empty(64, 32, dtype=torch.bfloat16, device="cuda"))
    assert bitwise_equal(w.main_grad, fused + unfused.float())
