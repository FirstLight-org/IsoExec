"""gqa_sm100 trainer core: FA4's SM100 backward on this forward -- accuracy, bitwise repeatability,
autograd wiring, refusals -- and the forward's run-to-run determinism through the trainer entry.

The backward is required to be deterministic per layout (the same inputs give the same dQ/dK/dV
bits on every run), not invariant to the head layout. Accuracy is checked against float32 autograd
through an explicit softmax attention.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(live_device_arch() != "sm100", reason="requires an SM100 GPU")

#: Per-rank shapes of the release models' attention (test_seam_gpu.MODELS) at the TP sizes that
#: give each distinct head ratio: (label, hq, hk, d).
GEOMETRIES = [
    ("35b-tp1", 16, 2, 256),
    ("35b-tp8", 2, 1, 256),
    ("9b-tp1", 16, 4, 256),
    ("0.8b-tp8", 1, 1, 256),
    ("qwen1.5-moe-tp1", 16, 16, 128),
    ("qwen1.5-moe-tp8", 2, 2, 128),
    ("d128-gqa", 8, 2, 128),
]
LENS = (700, 1, 129, 2049)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _cu(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return out


def _inputs(lens, hq, hk, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q = torch.randn(T, hq, d, device="cuda", generator=g).to(torch.bfloat16)
    k = torch.randn(T, hk, d, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(T, hk, d, device="cuda", generator=g).to(torch.bfloat16)
    do = torch.randn(T, hq, d, device="cuda", generator=g).to(torch.bfloat16)
    return q, k, v, do


def _reference(q, k, v, lens, scale):
    """float32 causal GQA, one sequence at a time."""
    outs, start = [], 0
    rep = q.shape[1] // k.shape[1]
    for n in lens:
        qi, ki, vi = q[start:start + n], k[start:start + n], v[start:start + n]
        ki, vi = ki.repeat_interleave(rep, 1), vi.repeat_interleave(rep, 1)
        s = torch.einsum("thd,shd->hts", qi, ki) * scale
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        p = s.masked_fill(~mask, float("-inf")).softmax(-1)
        outs.append(torch.einsum("hts,shd->thd", p, vi))
        start += n
    return torch.cat(outs)


def _grads(q, k, v, do, lens, scale):
    from isoexec.ops.attention.gqa_sm100.integration import trainer_core

    qs, ks, vs = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = trainer_core(qs, ks, vs, _cu(lens), max(lens), scale)
    out.backward(do)
    return out.detach(), qs.grad, ks.grad, vs.grad


@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_gradients_match_a_float32_reference(label, hq, hk, d):
    q, k, v, do = _inputs(LENS, hq, hk, d, seed=hq * 31 + d)
    scale = d ** -0.5
    out, dq, dk, dv = _grads(q, k, v, do, LENS, scale)
    ref = [t.detach().float().requires_grad_() for t in (q, k, v)]
    ref_out = _reference(*ref, LENS, scale)
    ref_out.backward(do.float())
    for name, got, want in (("out", out, ref_out.detach()), ("dq", dq, ref[0].grad), ("dk", dk, ref[1].grad),
                            ("dv", dv, ref[2].grad)):
        assert got.dtype == torch.bfloat16 and got.shape == want.shape, name
        err = float((got.float() - want).abs().max() / want.abs().max().clamp(min=1e-30))
        assert err < 2e-2, f"{label}: {name} max error {err:.4f} of max|ref|"


@pytest.mark.parametrize(("label", "hq", "hk", "d"), GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_forward_and_backward_are_bitwise_repeatable(label, hq, hk, d):
    """Ten runs, each after the allocator has been dirtied with different garbage."""
    q, k, v, do = _inputs(LENS, hq, hk, d, seed=7 * hq + d)
    scale = d ** -0.5
    first = _grads(q, k, v, do, LENS, scale)
    for run in range(10):
        junk = torch.full((1 << 24,), float(run) - 4.5, device="cuda")
        del junk
        again = _grads(q, k, v, do, LENS, scale)
        for name, a, b in zip(("out", "dq", "dk", "dv"), first, again):
            assert torch.equal(_bits(a), _bits(b)), f"{label}: {name} differs on run {run}"


def test_backward_kernel_per_head_dim():
    from isoexec.ops.attention.gqa_sm100 import ops

    assert ops.backward_kernel(128) == "FA4-sm100-deterministic"
    assert ops.backward_kernel(256) == "FA4-sm100-hd256-2cta"
    with pytest.raises(RuntimeError, match="no backward"):
        ops.backward_kernel(64)


def test_the_saved_lse_is_the_forward_lse_in_fa4s_layout():
    """FA4 recomputes P from the forward's LSE, so it must be this forward's own natural-log value."""
    from isoexec.ops.attention.gqa_sm100 import ops

    lens = (300, 129)
    q, k, v, _ = _inputs(lens, 4, 1, 256, seed=3)
    scale = 256 ** -0.5
    _, lse = ops.varlen_fwd(q, k, v, _cu(lens), max(lens), scale)
    assert lse.shape == (sum(lens), 4) and lse.dtype == torch.float32
    rep, start, want = 4, 0, []
    for n in lens:
        s = torch.einsum("thd,shd->hts", q[start:start + n].double(),
                         k[start:start + n].double().repeat_interleave(rep, 1)) * scale
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device="cuda").tril(), float("-inf"))
        want.append(torch.logsumexp(s, -1).t())
        start += n
    assert float((lse.double() - torch.cat(want)).abs().max()) < 1e-3


def test_refusals():
    from isoexec.ops.attention.gqa_sm100 import ops

    cu = _cu([64])
    q = torch.zeros(64, 4, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="head dims"):
        ops.varlen_fwd(q, q[:, :1], q[:, :1], cu, 64, 0.125)
    q = torch.zeros(64, 4, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="bf16"):
        ops.varlen_fwd(q, q[:, :1], q[:, :1], cu, 64, 0.125)
    q = torch.zeros(64, 6, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="whole number"):
        ops.varlen_fwd(q, q[:, :4], q[:, :4], cu, 64, 0.125)
    with pytest.raises(TypeError, match="int32"):
        ops.varlen_attention(q, q[:, :1], q[:, :1], cu.long(), 64, 0.125)


def test_install_receipt_names_the_recipe_and_the_backward():
    from isoexec.ops.attention.gqa_sm100 import integration, spec

    receipt = integration.install(deterministic_backward=True)
    assert receipt["impl"] == spec.TRAINER_IMPL and receipt["recipe"] == spec.RECIPE
    assert receipt["seg_tokens"] == {128: 2048, 256: 2048}
    assert receipt["backward"] == {128: "FA4-sm100-deterministic", 256: "FA4-sm100-hd256-2cta"}
    assert receipt["fa4_commit"] == spec.FA4_COMMIT
