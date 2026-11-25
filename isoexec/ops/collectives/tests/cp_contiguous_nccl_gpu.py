"""Contiguous context parallelism over real NCCL point-to-point, production SM90 kernels.

Run with ``torchrun --nproc-per-node N isoexec/ops/collectives/tests/cp_contiguous_nccl_gpu.py`` (N = 2 or 4;
each rank one GPU). Every rank runs the CP attention core (FA3 segment split) and the GDN conv + state
core (conv_sm90, native recurrent) on its own rows, with :class:`DistTransport` on the world group,
and compares with the CP=1 production cores it computes itself on the whole pack:

* forward rows bit-identical to CP=1;
* gradients within bf16 rounding of CP=1, bitwise identical over two runs;
* a record/replay recompute reproduces the plain-mode gradients bit for bit.

Prints one ``CP-NCCL`` line per rank and exits non-zero on any failed claim.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    os.environ.update(ISOEXEC_GDN_KERNEL="recurrent", ISOEXEC_GDN_NATIVE_KERNELS="1", ISOEXEC_GDN_NATIVE_STATE="1",
                      ISOEXEC_GDN_FLA_BACKWARD="1")
    from isoexec.ops import _dispatch

    _dispatch.ARCH = "sm90"
    from isoexec.ops._testing import bitwise_equal
    from isoexec.ops.attention.gqa_sm90 import integration as gqa
    from isoexec.ops.collectives.cp_attention import make_cp_attention_core
    from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan
    from isoexec.ops.collectives.cp_transport import DistTransport, cp_region
    from isoexec.ops.collectives.tests import test_cp_contiguous_battery_gpu as bat
    from isoexec.ops.gdn.conv_sm90 import integration as conv_integ
    from isoexec.ops.gdn.cp import make_cp_conv, make_cp_gdn_core
    from isoexec.ops.gdn.shared.gdn_native_stateful import gdn_native_stateful

    # Deterministic attention backward (head dim 256 has only FA3's atomic backward otherwise), as under
    # Megatron's deterministic_mode; it needs torch's deterministic mode.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    gqa.install(deterministic_backward=True)
    conv_integ.install()
    transport = DistTransport(dist.group.WORLD)
    transport.warmup(torch.device("cuda"))
    claims: dict[str, bool] = {}
    for name, lens in sorted(bat.PACKS.items()):
        cu = bat._padded(lens, world * 8)
        plan = ContiguousPlan.build(cu, world, rank)
        rows = slice(plan.lo, plan.lo + plan.local)
        local_cu = torch.tensor(plan.local_cu, dtype=torch.int32, device="cuda")
        register_plan(local_cu, plan)

        # Attention (Qwen3.8-27B TP4 geometry).
        hq, hk, d = 6, 1, 256
        g = torch.Generator(device="cuda").manual_seed(1)
        mk = lambda h, s=1.0: (s * torch.randn(cu[-1], h, d, device="cuda", generator=g)).to(torch.bfloat16)  # noqa: E731
        q, k, v, dout = mk(hq, 2.0), mk(hk), mk(hk), mk(hq)
        torch.use_deterministic_algorithms(True)  # scoped to attention: the FA2 deterministic backward needs it
        try:
            ref_o, ref_g = bat._attention_cp1(cu, q, k, v, dout, d ** -0.5)
        finally:
            torch.use_deterministic_algorithms(False)
        core = make_cp_attention_core(gqa.trainer_core_qk, transport_factory=lambda: transport, cp=world, rank=rank,
                                      site=f"{name}.attn")

        def attention(mode=None):
            leaves = [x[rows].detach().clone().requires_grad_(True) for x in (q, k, v)]
            run = lambda: core(*leaves, local_cu, plan.max_q, d ** -0.5)  # noqa: E731
            if mode == "recompute":
                with torch.no_grad(), cp_region("record"):
                    run()
                with cp_region("replay"):
                    o = run()
            else:
                o = run()
            o.backward(dout[rows])
            return o.detach(), [x.grad for x in leaves]

        torch.use_deterministic_algorithms(True)  # scoped to attention: the FA2 deterministic backward needs it
        try:
            o, grads = attention()
            o2, grads2 = attention()
            o3, grads3 = attention("recompute")
        finally:
            torch.use_deterministic_algorithms(False)
        claims[f"{name}.attn.fwd_bitwise"] = bitwise_equal(o, ref_o[rows])
        claims[f"{name}.attn.bwd_close"] = all(
            (x.float() - y[rows].float()).abs().max().item() <= 2e-2 * max(y.float().abs().max().item(), 1e-6)
            for x, y in zip(grads, ref_g))
        claims[f"{name}.attn.deterministic"] = all(bitwise_equal(x, y) for x, y in zip(grads2, grads))
        claims[f"{name}.attn.replay"] = bitwise_equal(o3, o) and all(bitwise_equal(x, y) for x, y in zip(grads3, grads))

        # GDN (one TP4 rank of Qwen3.8-27B).
        inputs = bat._gdn_inputs(cu[-1], seed=2)
        ref_o, ref_g = bat._gdn_cp1(cu, inputs)
        conv = make_cp_conv(bat._conv, transport_factory=lambda: transport, cp=world, rank=rank, site=f"{name}.conv")
        gcore = make_cp_gdn_core(gdn_native_stateful, transport_factory=lambda: transport, cp=world, rank=rank,
                                 site=f"{name}.core")
        x, a, b, w, A_log, dt_bias, gdout = inputs

        def gdn(mode=None):
            leaves = [t[rows].detach().clone().requires_grad_(True) for t in (x, a, b)]
            params = [t.detach().clone().requires_grad_(True) for t in (w, A_log, dt_bias)]
            run = lambda: bat._gdn_forward(conv, gcore, leaves[0], leaves[1], leaves[2], *params, local_cu)  # noqa: E731
            if mode == "recompute":
                with torch.no_grad(), cp_region("record"):
                    run()
                with cp_region("replay"):
                    o = run()
            else:
                o = run()
            o.backward(gdout[rows])
            return o.detach(), [t.grad for t in leaves]

        o, grads = gdn()
        o2, grads2 = gdn()
        o3, grads3 = gdn("recompute")
        claims[f"{name}.gdn.fwd_bitwise"] = bitwise_equal(o, ref_o[rows])
        claims[f"{name}.gdn.bwd_close"] = all(
            (x.float() - y[rows].float()).abs().max().item() <= 3e-2 * max(y.float().abs().max().item(), 1e-6)
            for x, y in zip(grads, ref_g[:3]))
        claims[f"{name}.gdn.deterministic"] = all(bitwise_equal(x_, y_) for x_, y_ in zip(grads2, grads))
        claims[f"{name}.gdn.replay"] = bitwise_equal(o3, o) and all(bitwise_equal(x_, y_) for x_, y_ in zip(grads3, grads))
    torch.cuda.synchronize()
    dist.barrier()
    failed = sorted(k for k, ok in claims.items() if not ok)
    print(f"CP-NCCL world={world} rank={rank} claims={len(claims)} failed={failed}", flush=True)
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
