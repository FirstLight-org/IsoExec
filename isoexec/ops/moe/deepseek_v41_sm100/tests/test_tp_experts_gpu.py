"""Engine TP > 1 without EP (``tp_experts``): every TP degree evaluates the TP = 1 program bit for bit.

* simulation (one process, all ranks): TP in {2, 4, 8} == ``ops.moe_forward`` (and == TP 1), V4.1 geometry and a small one;
* real ranks (2 GPUs, NCCL exchange): each rank's output == ``ops.moe_forward`` on the same rows; skipped with < 2 GPUs.
"""
from __future__ import annotations

import os

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe.deepseek_v41_sm100 import ops, tp_experts  # noqa: E402

KW = dict(top_k=6, routed_scale=1.5, limit=10.0)
GEOMETRY = [pytest.param(256, 1024, 1024, id="small"), pytest.param(384, 5120, 2304, id="v41")]


def _setup(E, H, I, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device="cuda") / H ** 0.5).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device="cuda") / I ** 0.5).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device="cuda") / H ** 0.5).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device="cuda") * 0.01
    return gu, dn, rw, rb


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.parametrize("E,H,I", GEOMETRY)
def test_every_tp_degree_is_the_tp1_program(E, H, I):
    gu, dn, rw, rb = _setup(E, H, I)
    for T in (1, 33, 300):
        x = (torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(T)) / 2).to(
            torch.bfloat16)
        want = ops.moe_forward(x, rw, rb, gu, dn, **KW)
        for tp in (1, 2, 4, 8):
            got = tp_experts.tp_moe_forward_sim(x, rw, rb, gu, dn, tp=tp, **KW)
            assert torch.equal(_bits(got), _bits(want)), (T, tp)


def test_refuses_an_inadmissible_degree():
    with pytest.raises(ValueError, match="TP in"):
        tp_experts.check_tp(2304, 5120, 3)
    with pytest.raises(ValueError, match="multiples of 8"):
        tp_experts.check_tp(2304, 5120 + 4, 8)


def _rank(rank, world, port, out):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    E, H, I = 256, 1024, 1024
    gu, dn, rw, rb = _setup(E, H, I)
    ex = tp_experts.NcclExchange()
    gu_r, dn_r = tp_experts.shard_masters(gu, dn, world, rank)
    ok = True
    for T in (1, 33, 300):
        x = (torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(T)) / 2).to(
            torch.bfloat16)
        got = tp_experts.tp_moe_forward(x, rw, rb, gu_r, dn_r, exchange=ex, **KW)
        want = ops.moe_forward(x, rw, rb, gu, dn, **KW)
        ok &= bool(torch.equal(_bits(got), _bits(want)))
    torch.save(ok, f"{out}.{rank}")
    dist.destroy_process_group()


def test_two_real_ranks_over_nccl_are_the_tp1_program(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    import torch.multiprocessing as mp
    port = 29700 + os.getpid() % 200
    mp.spawn(_rank, args=(2, port, str(tmp_path / "ok")), nprocs=2, join=True)
    assert all(torch.load(tmp_path / f"ok.{r}") for r in range(2))
