"""CP=2 trainer seams on one SM100 GPU: every local row and gradient equals the CP=1 bits.

Two processes share the GPU and exchange through Gloo (``_gloo_cuda``), so each rank runs the real kernels
and the real movement code of

* the gqa_sm100 trainer core's CP form (``gqa_sm100/context_parallel.py``): forward rows, dQ, dK and dV;
* GDN's head-wise region (``cp_heads.make_gdn_region``) around the trainer convolution and delta rule:
  output rows, the in-projection gradient rows, and the per-head parameter gradients (each rank holds its
  heads' rows of conv_w / A_log / dt_bias and exact zeros elsewhere, so the data-parallel sum is the
  CP=1 gradient).

Qwen3.6-27B's per-rank geometry (24/4 query/KV heads at head dim 256; GDN 16 key / 48 value heads of
128), packed sequences of uneven length padded to multiples of ``2 * cp``, under the production trainer flags.
The GDN core is the selected one: run once as is (chunk-parallel) and once with ``ISOEXEC_GDN_KERNEL=recurrent
ISOEXEC_GDN_NATIVE_STATE=1`` (native recurrent). Run directly or under pytest.
"""

from __future__ import annotations

import math
import tempfile

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="requires an SM100 GPU")

LENS = (6000, 3004, 2068, 8)
HQ, HK, D = 24, 4, 256
GK, GV, DK, DV, W = 16, 48, 128, 128, 4


def _bits(t):
    return t.detach().contiguous().view(-1).view(torch.uint8)


def _same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _placement(group, rank):
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=Axis(2, rank, group), side="trainer", leaves=1)


def _inputs(device):
    g = torch.Generator(device).manual_seed(0)

    def rand(*shape, s=1.0):
        return (torch.randn(*shape, generator=g, device=device) * s).bfloat16()

    cu = [0]
    for n in LENS:
        cu.append(cu[-1] + n)
    T = cu[-1]
    kd, vd = GK * DK, GV * DV
    return dict(
        cu=cu, T=T,
        q=rand(T, HQ, D), k=rand(T, HK, D), v=rand(T, HK, D), do=rand(T, HQ, D),
        proj=rand(T, 2 * kd + 2 * vd + 2 * GV, s=0.5), conv_w=rand(2 * kd + vd, 1, W, s=0.3),
        A_log=rand(GV), dt_bias=rand(GV), dy=rand(T, GV, DV),
    )


def _mix(qkv, beta, alpha, conv_w, A_log, dt_bias, seq):
    from isoexec.ops.gdn import gdn_core
    from isoexec.ops.gdn.trainer import conv_short_native

    hv = beta.shape[1]
    hk, kd = hv * GK // GV, hv * GK // GV * DK
    y = conv_short_native(qkv, conv_w, seq.cu_seqlens)
    q, k, v = y[:, :kd].unflatten(-1, (hk, DK)), y[:, kd:2 * kd].unflatten(-1, (hk, DK)), y[:, 2 * kd:].unflatten(-1, (hv, DV))
    return gdn_core(q, k, v, alpha, beta, A_log, dt_bias, seq.cu_seqlens)


class _Seq:
    def __init__(self, cu):
        self.cu_seqlens = cu


def _body(rank, world, out, init_method):
    from isoexec.ops._dispatch import install_arch_impls
    from isoexec.ops.attention.gqa_sm100.context_parallel import make_trainer_core
    from isoexec.ops.attention.gqa_sm100.integration import trainer_core
    from isoexec.ops.collectives.cp_gather import layout_for
    from isoexec.ops.collectives.cp_heads import HeadGroup, make_gdn_region
    from isoexec.ops.collectives.tests._gloo_cuda import init_world

    from isoexec.runtimes.trainer_installs import apply_trainer_env

    group = init_world(rank, world, init_method)
    apply_trainer_env(None)  # the production trainer flags; the caller's environment (GDN kernel choice) wins
    install_arch_impls()
    dev = torch.device("cuda")
    x = _inputs(dev)
    cu = torch.tensor(x["cu"], dtype=torch.int32, device=dev)
    mine = layout_for(cu, 2, rank).mine
    res = {}

    # ---- attention: CP=1 reference, then this rank's CP form
    scale = 1 / math.sqrt(D)
    leaves = [x[n].clone().requires_grad_() for n in ("q", "k", "v")]
    ref = trainer_core(*leaves, cu, max(LENS), scale)
    ref.backward(x["do"])
    local = [x[n][mine].clone().requires_grad_() for n in ("q", "k", "v")]
    o = make_trainer_core(_placement(group, rank))(*local, cu, max(LENS), scale)
    o.backward(x["do"][mine])
    res["attention_out"] = _same(o, ref[mine])
    for name, a, b in zip(("dq", "dk", "dv"), local, leaves):
        res[f"attention_{name}"] = _same(a.grad, b.grad[mine])

    # ---- GDN head-wise region around the trainer convolution and core
    kd, vd = GK * DK, GV * DV
    names = ("proj", "conv_w", "A_log", "dt_bias")
    full = {n: x[n].clone().requires_grad_() for n in names}
    seq = _Seq(cu)
    p = full["proj"]
    qkv = p[:, : 2 * kd + vd]
    beta, alpha = p[:, 2 * kd + 2 * vd : 2 * kd + 2 * vd + GV], p[:, 2 * kd + 2 * vd + GV :]
    ref = _mix(qkv, beta, alpha, full["conv_w"], full["A_log"], full["dt_bias"], seq)
    ref.backward(x["dy"])
    part = {n: (x[n][mine] if n == "proj" else x[n]).clone().requires_grad_() for n in names}
    region = make_gdn_region(_placement(group, rank), key_heads=GK, value_heads=GV, key_head_dim=DK, value_head_dim=DV)
    y = region(part["proj"], _mix, part["conv_w"], part["A_log"], part["dt_bias"], _Seq(cu))
    y.backward(x["dy"][mine])
    res["gdn_out"] = _same(y, ref[mine])
    # The in-projection gradient: z columns get none here (the gated norm is outside the region).
    res["gdn_dproj"] = _same(part["proj"].grad, full["proj"].grad[mine])
    ranges = HeadGroup.of([(0, GK, DK), (kd, GK, DK), (2 * kd, GV, DV)], 2).ranges[rank]
    rows = torch.cat([torch.arange(a, a + n, device=dev) for a, n in ranges])
    heads = torch.arange(rank * GV // 2, (rank + 1) * GV // 2, device=dev)
    for name, own in (("conv_w", rows), ("A_log", heads), ("dt_bias", heads)):
        got, want = part[name].grad, full[name].grad
        others = torch.ones(got.shape[0], dtype=torch.bool, device=dev)
        others[own] = False
        res[f"gdn_d{name}"] = _same(got[own], want[own]) and not bool(got[others].any())
    out[rank] = res


def run() -> dict:
    from isoexec.ops.collectives.tests._gloo_harness import run_world

    with tempfile.TemporaryDirectory() as tmp:
        return run_world(_body, world=2, rendezvous=f"{tmp}/rdzv", timeout_s=3600)


@pytest.fixture(scope="module")
def results():
    return run()


CLAIMS = ("attention_out", "attention_dq", "attention_dk", "attention_dv",
          "gdn_out", "gdn_dproj", "gdn_dconv_w", "gdn_dA_log", "gdn_ddt_bias")


@pytest.mark.parametrize("claim", CLAIMS)
def test_cp2_rows_equal_cp1(results, claim):
    from isoexec.ops.collectives.tests._gloo_harness import assert_claim

    assert_claim(results, claim, 2)


if __name__ == "__main__":
    for rank, res in sorted(run().items()):
        print(rank, res)
