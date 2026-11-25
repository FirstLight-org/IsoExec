"""Contiguous context parallelism on the production SM90 kernels, CP = 2, 4, 8 simulated on one GPU.

The CP wrappers run every simulated rank's own calls with the production kernels (FA3 segment split,
conv_sm90, the native recurrent GDN core); only the transport is an in-process mailbox. Bits depend on
the per-rank kernel calls alone, so this is the forward claim at any degree:

* attention: each rank's ``trainer_core_qk`` rows (queries right-aligned to ``[prefix | local]`` keys)
  == the CP=1 ``trainer_core`` rows, bit for bit, at Qwen3.8-27B TP4/TP2 head geometries, with
  sequences crossing several 2560-key segments and shard boundaries at arbitrary offsets;
* GDN: conv tail + fp32 state hand-off == the CP=1 conv + ``gdn_recurrent_sm90_core``, bit for bit;
* backward: gradients match CP=1 to bf16 rounding and are bitwise deterministic run to run.

The real NCCL transport is ``cp_contiguous_nccl_gpu.py`` (2+ GPUs).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.collectives.cp_attention import make_cp_attention_core
from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan
from isoexec.ops.collectives.cp_transport import MailboxTransport
from isoexec.ops.gdn.cp import make_cp_conv, make_cp_gdn_core

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires an SM90 GPU")

DEV = "cuda"
# Packs: short + long sequences; long ones span many 2560-key segments and several shards.
PACKS = {
    "mixed": [1104, 5200, 344, 2560, 7000, 16],
    "one_long": [16384],
    "many_short": [24, 400, 72, 1032, 8, 600, 2048, 104, 512, 1296],
}
CPS = (2, 4, 8)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "1")
    monkeypatch.setenv("ISOEXEC_GDN_FLA_BACKWARD", "1")
    from isoexec.ops.attention.gqa_sm90 import integration as gqa
    from isoexec.ops.gdn.conv_sm90 import integration as conv

    # Run-to-run gradient determinism is a claim about the deterministic attention backward: at head dim 256
    # FA3 has only an atomic backward, so (as under Megatron's deterministic_mode) the install selects
    # torch's deterministic FA2 backward, which needs torch's deterministic mode.
    # Scoped to the attention cases (``deterministic_attention``); the GDN cases run as the trainer does.
    gqa.install(deterministic_backward=True)
    conv.install()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    gqa.install(deterministic_backward=False)


class deterministic_attention:
    """torch's deterministic mode, which the FA2 deterministic backward (head dim 256) requires."""

    def __enter__(self):
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(False)


def _padded(lens, cp):
    """Pad every sequence to a multiple of cp, like verl's preprocess (tp=1)."""
    out = [0]
    for n in lens:
        out.append(out[-1] + -(-n // cp) * cp)
    return out


class World:
    def __init__(self, cu, cp):
        self.cp, self.box = cp, {}
        self.plans = [ContiguousPlan.build(cu, cp, r) for r in range(cp)]
        self.cus = []
        for plan in self.plans:
            local = torch.tensor(plan.local_cu, dtype=torch.int32, device=DEV)
            register_plan(local, plan)
            self.cus.append(local)

    def factory(self, rank):
        t = MailboxTransport(self.box, self.cp, rank)
        return lambda: t

    def rows(self, rank):
        p = self.plans[rank]
        return slice(p.lo, p.lo + p.local)


def _attention_cp(cu, cp, q, k, v, dout, scale):
    from isoexec.ops.attention.gqa_sm90.integration import trainer_core_qk

    w = World(cu, cp)
    leaves, outs = [], []
    for r in range(cp):
        core = make_cp_attention_core(trainer_core_qk, transport_factory=w.factory(r), cp=cp, rank=r, site="attn")
        ql, kl, vl = (x[w.rows(r)].detach().clone().requires_grad_(True) for x in (q, k, v))
        leaves.append((ql, kl, vl))
        outs.append(core(ql, kl, vl, w.cus[r], w.plans[r].max_q, scale))
    for r in reversed(range(cp)):
        outs[r].backward(dout[w.rows(r)])
    return torch.cat([o.detach() for o in outs]), [torch.cat([lv[i].grad for lv in leaves]) for i in range(3)]


def _attention_cp1(cu, q, k, v, dout, scale):
    from isoexec.ops.attention.gqa_sm90.integration import trainer_core

    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    cut = torch.tensor(cu, dtype=torch.int32, device=DEV)
    o = trainer_core(*leaves, cut, max(b - a for a, b in zip(cu, cu[1:])), scale)
    o.backward(dout)
    return o.detach(), [x.grad for x in leaves]


@pytest.mark.parametrize("geometry", [(6, 1, 256), (12, 2, 256), (8, 2, 128)], ids=["27b-tp4", "27b-tp2", "d128"])
@pytest.mark.parametrize("pack", sorted(PACKS))
def test_attention_cp_equals_cp1_bitwise(geometry, pack):
    with deterministic_attention():
        _attention_case(geometry, pack)


def _attention_case(geometry, pack):
    hq, hk, d = geometry
    for cp in CPS:
        cu = _padded(PACKS[pack], cp * 8)  # a shard boundary at an arbitrary in-sequence offset
        T = cu[-1]
        g = torch.Generator(device=DEV).manual_seed(cp)
        mk = lambda h, s=1.0: (s * torch.randn(T, h, d, device=DEV, generator=g)).to(torch.bfloat16)  # noqa: E731
        q, k, v, dout = mk(hq, 2.0), mk(hk), mk(hk), mk(hq)
        ref_o, ref_g = _attention_cp1(cu, q, k, v, dout, d ** -0.5)
        o, grads = _attention_cp(cu, cp, q, k, v, dout, d ** -0.5)
        assert bitwise_equal(o, ref_o), f"cp={cp}: forward rows differ from CP=1"
        for name, x, y in zip("qkv", grads, ref_g):
            err = (x.float() - y.float()).abs().max().item() / max(y.float().abs().max().item(), 1e-6)
            assert err < 2e-2, f"cp={cp}: d{name} relative error {err}"
        o2, grads2 = _attention_cp(cu, cp, q, k, v, dout, d ** -0.5)
        assert bitwise_equal(o2, o) and all(bitwise_equal(x, y) for x, y in zip(grads2, grads))


HK, HV, DK, W = 4, 12, 128, 4  # one TP4 rank of Qwen3.8-27B's GDN


def _conv(x, w, cu):
    from isoexec.ops.gdn.conv_sm90.integration import gdn_native_conv

    return gdn_native_conv(x.contiguous(), w.squeeze(1), None, cu_seqlens=cu)


def _gdn_inputs(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s, sc=1.0: (sc * torch.randn(*s, generator=g, device=DEV)).to(torch.bfloat16)  # noqa: E731
    C = 2 * HK * DK + HV * DK
    return (r(T, C), r(T, HV), r(T, HV), r(C, 1, W, sc=0.4), r(HV, sc=0.3), r(HV, sc=0.3), r(T, HV, DK))


def _gdn_forward(conv, core, x, a, b, w, A_log, dt_bias, cu_t):
    y = conv(x, w, cu_t)
    kd = HK * DK
    q = y[:, :kd].unflatten(-1, (HK, DK))
    k = y[:, kd:2 * kd].unflatten(-1, (HK, DK))
    v = y[:, 2 * kd:].unflatten(-1, (HV, DK))
    return core(q, k, v, a, b, A_log, dt_bias, cu_t)


def _gdn_cp1(cu, inputs):
    from isoexec.ops.gdn.gdn_recurrent_sm90.integration import gdn_recurrent_sm90_core

    x, a, b, w, A_log, dt_bias, dout = inputs
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, a, b, w, A_log, dt_bias)]
    cut = torch.tensor(cu, dtype=torch.int32, device=DEV)
    o = _gdn_forward(_conv, gdn_recurrent_sm90_core, *leaves, cut)
    o.backward(dout)
    return o.detach(), [t.grad for t in leaves]


def _gdn_cp(cu, cp, inputs):
    from isoexec.ops.gdn.shared.gdn_native_stateful import gdn_native_stateful

    x, a, b, w, A_log, dt_bias, dout = inputs
    world = World(cu, cp)
    params = [t.detach().clone().requires_grad_(True) for t in (w, A_log, dt_bias)]
    leaves, outs = [], []
    for r in range(cp):
        conv = make_cp_conv(_conv, transport_factory=world.factory(r), cp=cp, rank=r, site="g.conv")
        core = make_cp_gdn_core(gdn_native_stateful, transport_factory=world.factory(r), cp=cp, rank=r, site="g.core")
        lv = [t[world.rows(r)].detach().clone().requires_grad_(True) for t in (x, a, b)]
        leaves.append(lv)
        outs.append(_gdn_forward(conv, core, lv[0], lv[1], lv[2], params[0], params[1], params[2], world.cus[r]))
    for r in reversed(range(cp)):
        outs[r].backward(dout[world.rows(r)])
    grads = [torch.cat([lv[i].grad for lv in leaves]) for i in range(3)] + [p.grad for p in params]
    return torch.cat([o.detach() for o in outs]), grads


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_gdn_cp_equals_cp1_bitwise(pack):
    for cp in CPS:
        cu = _padded(PACKS[pack], cp * 8)
        inputs = _gdn_inputs(cu[-1], seed=cp)
        ref_o, ref_g = _gdn_cp1(cu, inputs)
        o, grads = _gdn_cp(cu, cp, inputs)
        assert bitwise_equal(o, ref_o), f"cp={cp}: GDN forward rows differ from CP=1"
        for name, x, y in zip(("x", "a", "b", "conv_w", "A_log", "dt_bias"), grads, ref_g):
            err = (x.float() - y.float()).abs().max().item() / max(y.float().abs().max().item(), 1e-6)
            assert err < 3e-2, f"cp={cp}: d{name} relative error {err}"
        o2, grads2 = _gdn_cp(cu, cp, inputs)
        assert bitwise_equal(o2, o) and all(bitwise_equal(x, y) for x, y in zip(grads2, grads))


def test_short_shards_pass_the_conv_tail_through():
    """cp=8 over a 16-row sequence: 2 rows per rank (< w-1), so tails pass through ranks."""
    cu = [0, 16]
    inputs = _gdn_inputs(cu[-1], seed=11)
    ref_o, _ = _gdn_cp1(cu, inputs)
    o, _ = _gdn_cp(cu, 8, inputs)
    assert bitwise_equal(o, ref_o)
