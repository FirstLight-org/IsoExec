"""The trainer core's gradients: FLA's chunked backward kernels fed the forward's own intermediates (no
determinism claimed), with and without the saved chunk states.

Tolerances, not bits (claim map: tests/README.md).
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90.integration import gdn_cpr_sm90_core
from isoexec.ops.gdn.gdn_cpr_sm90.reference import gdn_reference
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, SM90, make

pytestmark = SM90

NAMES = ("q", "k", "v", "a", "b", "A_log", "dt_bias")


@pytest.mark.parametrize("save_states", [None, True, False], ids=["default", "save", "recompute"])
@pytest.mark.parametrize("lanes,hk,hv", [([37, 70], 2, 4), ([200, 1, 129, 64], 2, 6)])
def test_backward_matches_float64_reference(lanes, hk, hv, save_states):
    d = make(lanes, hk, hv, seed=31)
    leaves = [d[n].detach().clone().requires_grad_(True) for n in NAMES]
    do = (torch.randn(sum(lanes), hv, 128, device=DEV) * 0.01).to(BF)
    kwargs = {} if save_states is None else {"save_states": save_states}
    o = gdn_cpr_sm90_core(*leaves, d["cut"], **kwargs)
    grads = torch.autograd.grad(o, leaves, do)
    assert all(g is not None for g in grads), "every differentiable input must receive a gradient"
    for g, t in zip(grads, leaves):
        assert g.shape == t.shape and g.dtype == t.dtype
    ref_leaves = [d[n].detach().double().clone().requires_grad_(True) for n in NAMES]
    ro, _ = gdn_reference(*ref_leaves, d["cu"])
    rgrads = torch.autograd.grad(ro, ref_leaves, do.double())
    for name, got, want in zip(NAMES, grads, rgrads):
        g64 = got.double()
        assert torch.isfinite(g64).all(), f"d{name} is not finite"
        rel_max = (g64 - want).abs().max().item() / max(want.abs().max().item(), 1e-30)
        rel_l2 = ((g64 - want).norm() / max(want.norm().item(), 1e-30)).item()
        # dA_log / ddt_bias are one scalar per head summed over every token with heavy cancellation
        l2_tol = 5e-2 if name in ("A_log", "dt_bias") else 1e-2
        assert rel_l2 <= l2_tol, f"d{name} relative L2 error {rel_l2:.3g}"
        assert rel_max <= 5e-2, f"d{name} relative max error {rel_max:.3g}"


@pytest.mark.parametrize("save_states", [True, False])
def test_backward_agrees_with_fla_chunk(save_states):
    """Agreement with FLA's own forward+backward gradients (the vendored FLA 0.5.2 chunk path; a different
    rounding of the same function), within the backward tolerance: max |ours - FLA| <= 5e-2 max |FLA|."""
    from isoexec.ops.gdn.shared.gdn_fla_backward import get_fla_chunk_backward

    chunk = get_fla_chunk_backward()
    lanes, hv = [64, 33, 150], 4
    d = make(lanes, 2, hv, seed=32)
    leaves = [d[n].detach().clone().requires_grad_(True) for n in NAMES]
    do = (torch.randn(sum(lanes), hv, 128, device=DEV) * 0.01).to(BF)
    ours = torch.autograd.grad(gdn_cpr_sm90_core(*leaves, d["cut"], save_states=save_states), leaves, do)
    fla = [d[n].detach().clone().requires_grad_(True) for n in NAMES]
    cu = d["cut"].long()
    of, _ = chunk(*(x.unsqueeze(0) for x in fla[:5]), use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                  use_gate_in_kernel=True, A_log=fla[5].float(), dt_bias=fla[6].float(), cu_seqlens=cu,
                  cu_seqlens_cpu=cu.cpu())
    theirs = torch.autograd.grad(of, fla, do.unsqueeze(0))
    for name, got, want in zip(NAMES, ours, theirs):
        w = want.reshape(got.shape).double()
        assert (got.double() - w).abs().max().item() <= 5e-2 * max(w.abs().max().item(), 1e-30), f"d{name}"


def test_forward_bits_do_not_depend_on_grad_mode():
    lanes, hv = [16, 9, 70], 4
    d = make(lanes, 2, hv, seed=33)
    with torch.no_grad():
        o0 = gdn_cpr_sm90_core(*(d[n] for n in NAMES), d["cut"])
    leaves = [d[n].detach().clone().requires_grad_(True) for n in NAMES]
    o1 = gdn_cpr_sm90_core(*leaves, d["cut"])
    assert bitwise_equal(o0, o1.detach())


def test_trainer_default_omits_recomputable_states():
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops

    data = make([37, 70], 2, 4, seed=35)
    args = [data[name] for name in NAMES] + [data["cut"]]
    output, saved = ops.trainer_forward(*args)
    full_output, full_saved = ops.trainer_forward(*args, save_states=True)
    assert bitwise_equal(output, full_output)
    for name in ("w", "h", "v_new"):
        assert saved[name].numel() == 0
        assert full_saved[name].numel() > 0
    for name in ("qn", "kn", "q_rstd", "k_rstd", "v", "A"):
        assert bitwise_equal(saved[name], full_saved[name]), name


@pytest.mark.parametrize("hk,hv", [(2, 4), (2, 6)])
def test_trainer_saves_are_flas_intermediates(hk, hv):
    """What the trainer forward saves for the backward is, layout and meaning, what FLA's own forward
    computes (the normalized q/k and inverse norms, A, w, the chunk states h in FLA's chunk order,
    value-major, and v_new): each within bf16 rounding of FLA's (relative L2 <= 1e-2)."""
    import inspect

    from isoexec.ops.gdn.gdn_cpr_sm90.fla_backward import load
    from isoexec.ops.gdn.shared.gdn_fla_backward import get_fla_chunk_backward

    F = load()
    ns = inspect.unwrap(get_fla_chunk_backward()).__globals__
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops

    lanes = [130, 1, 64, 257]
    d = make(lanes, hk, hv, seed=34)
    _, saved = ops.trainer_forward(d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"], d["cut"],
                                   save_states=True)
    cu = d["cut"].long()
    q4, k4, v4, a4, b4 = (d[n].unsqueeze(0).contiguous() for n in ("q", "k", "v", "a", "b"))
    qn, q_rstd = ns["l2norm_fwd"](q4)
    kn, k_rstd = ns["l2norm_fwd"](k4)
    beta = F.fused_beta_sigmoid(b4, scale=1.0)
    chunks = F.prepare_chunk_indices(cu, 64, cu_seqlens_cpu=cu.cpu())
    g = F.gdn_gate_chunk_cumsum(g=a4, A_log=d["A_log"].float(), chunk_size=64, scale=F.RCP_LN2,
                                dt_bias=d["dt_bias"].float(), cu_seqlens=cu, chunk_indices=chunks)
    w, u, A = ns["chunk_gated_delta_rule_fwd_intra"](k=kn, v=v4, g=g, beta=beta, cu_seqlens=cu, chunk_indices=chunks,
                                                    chunk_size=64)
    h, v_new, _ = F.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=g, initial_state=None, output_final_state=False,
                                                 cu_seqlens=cu, chunk_indices=chunks, state_v_first=True, chunk_size=64)
    want = dict(qn=qn, kn=kn, q_rstd=q_rstd, k_rstd=k_rstd, v=v4, A=A, w=w, h=h, v_new=v_new)
    for name, ref in want.items():
        got = saved[name][:len(chunks)] if name == "h" else saved[name]
        ref = ref.reshape(got.shape).double()
        err = ((got.double() - ref).norm() / max(ref.norm().item(), 1e-30)).item()
        assert err <= 1e-2, f"{name}: relative L2 {err:.3g}"


def test_bwd_qk_finish_validates_its_inputs():
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops

    T, hk, hv = 8, 2, 4
    bf = dict(device=DEV, dtype=BF)
    ok = dict(dq_hv=torch.zeros(T, hv, 128, **bf), dk_hv=torch.zeros(T, hv, 128, **bf), dk2_hv=torch.zeros(T, hv, 128, **bf),
              qn=torch.zeros(T, hk, 128, **bf), kn=torch.zeros(T, hk, 128, **bf),
              q_rstd=torch.zeros(T, hk, device=DEV), k_rstd=torch.zeros(T, hk, device=DEV))
    ops.extension().bwd_qk_finish(*ok.values())
    for name, bad in (("q_rstd", torch.zeros(T, hk, device=DEV, dtype=torch.float64)),
                      ("k_rstd", torch.zeros(T + 1, hk, device=DEV)),
                      ("dk2_hv", torch.zeros(T, hv + 1, 128, **bf)),
                      ("kn", torch.zeros(T, hk + 1, 128, **bf)),
                      ("dq_hv", torch.zeros(T, hv, 128, device=DEV, dtype=torch.float32))):
        args = dict(ok, **{name: bad})
        with pytest.raises(RuntimeError, match="bwd_qk_finish"):
            ops.extension().bwd_qk_finish(*args.values())
