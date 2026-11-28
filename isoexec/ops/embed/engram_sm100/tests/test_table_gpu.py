"""The master -> FP8 Engram table encode and the gate (needs a B200). Bitwise relations:

* table encode: GPU bytes == a CPU torch program (device independent), run-to-run, decode(encode(decode(T))) == decode(T),
  the sparse re-encode writes exactly the listed rows with the same bytes as the full encode.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires B200/SM100"
)

DEV = "cuda"


def iv(t):
    return t.contiguous().view(torch.uint8)


def make_table(R, D, seed, exp_lo=100, exp_hi=140):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(R, D, device=DEV, generator=g) * 3).to(torch.float8_e4m3fn)
    e = torch.randint(exp_lo, exp_hi, (R, D // 32), device=DEV, generator=g, dtype=torch.uint8)
    return w, e.view(torch.float8_e8m0fnu)


def encode_reference_cpu(master):
    """The declared program in torch ops on the CPU (no device code): per 32-group, amax floor 448*2^-126,
    scale = amax * float32(1/448) rounded UP to a power of two, code = e4m3_rn(clamp(x / scale, +-448))."""
    m = master.detach().cpu().float()
    R, D = m.shape
    x = m.view(R, D // 32, 32)
    amax = x.abs().amax(-1).clamp_min(448.0 * 2.0 ** -126)
    scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    b = scale.view(torch.int32)
    exp = ((b >> 23) & 255) + ((b & 0x7FFFFF) != 0).to(torch.int32)
    pow2 = (exp << 23).view(torch.float32)
    codes = (x / pow2.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes.view(R, D), exp.to(torch.uint8).view(torch.float8_e8m0fnu)


def decode(w, s):
    R, D = w.shape
    sc = s.float().unsqueeze(-1)
    return (w.float().view(R, D // 32, 32) * sc).view(R, D)


@pytest.mark.parametrize("D", [32, 256, 1280])
def test_table_encode_matches_cpu_program(D):
    from isoexec.ops.embed.engram_sm100.ops import quantize_table_fp8

    g = torch.Generator(device=DEV).manual_seed(D)
    m = torch.randn(777, D, device=DEV, generator=g) * torch.exp2(torch.randint(-12, 8, (777, D // 32), device=DEV, generator=g).float()).repeat_interleave(32, 1)
    m[3] = 0
    m[4, :32] = 1e30
    m[5, :32] = 1e-30
    w, s = quantize_table_fp8(m)
    cw, cs = encode_reference_cpu(m)
    assert torch.equal(iv(w).cpu(), iv(cw)) and torch.equal(iv(s).cpu(), iv(cs))
    w2, s2 = quantize_table_fp8(m)
    assert torch.equal(iv(w), iv(w2)) and torch.equal(iv(s), iv(s2))


def test_table_encode_decode_roundtrip_exact():
    """decode(encode(decode(T))) == decode(T): a decoded FP8 table survives re-encode without loss."""
    from isoexec.ops.embed.engram_sm100.ops import quantize_table_fp8

    w, s = make_table(900, 512, 9, 105, 135)
    dec = decode(w, s)
    w2, s2 = quantize_table_fp8(dec.contiguous())
    assert torch.equal(decode(w2, s2), dec)


def test_sparse_requant_touches_only_listed_rows():
    from isoexec.ops.embed.engram_sm100.ops import quantize_table_fp8, requantize_rows_, table_row_grad

    R, D = 2000, 256
    w, s = make_table(R, D, 4)
    master = decode(w, s).contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(0, R, (300,), device=DEV, generator=g)
    grad = torch.randn(300, D, device=DEV, generator=g)
    rows, gsum, nruns = table_row_grad(idx, grad, R)
    n = int(nruns.item())
    new = master.clone()
    new[rows[:n]] -= 0.3 * gsum[:n]                       # a toy optimiser step on the touched rows only
    w0, s0 = w.clone(), s.clone()
    requantize_rows_(w, s, new, positions=rows, nruns=nruns)
    fw, fs = quantize_table_fp8(new)
    touched = torch.zeros(R, dtype=torch.bool, device=DEV)
    touched[rows[:n]] = True
    assert torch.equal(iv(w)[touched], iv(fw)[touched]) and torch.equal(iv(s)[touched], iv(fs)[touched])
    assert torch.equal(iv(w)[~touched], iv(w0)[~touched]) and torch.equal(iv(s)[~touched], iv(s0)[~touched])


def test_overlay_requant_maps_master_rows_to_table_rows():
    from isoexec.ops.embed.engram_sm100.ops import quantize_table_fp8, requantize_rows_

    R, D, U = 5000, 256, 40
    w, s = make_table(R, D, 8)
    g = torch.Generator(device=DEV).manual_seed(1)
    row_ids = torch.unique(torch.randint(0, R, (U,), device=DEV, generator=g))
    master = (torch.randn(row_ids.numel(), D, device=DEV, generator=g)).contiguous()
    w0 = w.clone()
    requantize_rows_(w, s, master, row_ids=row_ids)
    fw, fs = quantize_table_fp8(master)
    assert torch.equal(iv(w)[row_ids], iv(fw)) and torch.equal(iv(s)[row_ids], iv(fs))
    rest = torch.ones(R, dtype=torch.bool, device=DEV)
    rest[row_ids] = False
    assert torch.equal(iv(w)[rest], iv(w0)[rest])


# ---------------------------------------------------------------------------------------------- SASS (no GPU needed)
def test_new_kernels_have_no_atomics_cpu():
    from isoexec.ops.embed.engram_sm100._extension import extension

    out = subprocess.run(["cuobjdump", "-sass", str(Path(extension().__file__))],
                         capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur:
            funcs[cur].append(line)
    names = [n for n in funcs if "lookup_fp8_quant_kernel" in n or "quantize_table_kernel" in n]
    assert len(names) == 2
    for n in names:
        assert not [ln for ln in funcs[n] if re.search(r"\b(ATOM|ATOMS|ATOMG|RED)\b|\bATOMG?\.|\bRED\.", ln)], n


# ---------------------------------------------------------------------------------------------- strided key / value views
@pytest.mark.parametrize("masked", [False, True])
def test_gate_accepts_strided_key_value_bitwise(masked):
    """key / value as views of one projection output (wkv layout [B,T,(S+1)*D]) == the contiguous copies, forward and backward."""
    from isoexec.ops.embed.engram_sm100.ops import engram_gate_residual

    B, T, S, D = 2, 37, 4, 256
    g = torch.Generator(device=DEV).manual_seed(5)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g).bfloat16()
    proj = r(B, T, (S + 1) * D).requires_grad_(True)
    x = r(B, T, S, D).requires_grad_(True)
    qw, kw = r(S, D).requires_grad_(True), r(S, D).requires_grad_(True)
    mask = (torch.rand(B, T, device=DEV, generator=g) > 0.3) if masked else None
    key_v = proj[..., : S * D].unflatten(-1, (S, D))
    val_v = proj[..., S * D:]
    assert not key_v.is_contiguous() and not val_v.is_contiguous()
    out = engram_gate_residual(x, key_v, val_v, qw, kw, 1e-6, mask)
    go = r(B, T, S, D)
    g1 = torch.autograd.grad(out, (proj, x, qw, kw), go)
    proj2 = proj.detach().clone().requires_grad_(True)
    k2 = proj2[..., : S * D].unflatten(-1, (S, D)).contiguous()
    v2 = proj2[..., S * D:].contiguous()
    k2.retain_grad(); v2.retain_grad()
    x2 = x.detach().clone().requires_grad_(True)
    qw2, kw2 = qw.detach().clone().requires_grad_(True), kw.detach().clone().requires_grad_(True)
    out2 = engram_gate_residual(x2, k2, v2, qw2, kw2, 1e-6, mask)
    g2 = torch.autograd.grad(out2, (k2, v2, x2, qw2, kw2), go)
    assert torch.equal(iv(out), iv(out2))
    dk, dv = g1[0][..., : S * D], g1[0][..., S * D:]
    # (autograd scatters the view gradients into a zeros tensor, which turns a kernel -0.0 into +0.0: compare values)
    assert torch.equal(dk, g2[0].flatten(-2)) and torch.equal(dv, g2[1])
    nz = dk != 0
    assert torch.equal(dk.contiguous().view(torch.int16)[nz], g2[0].flatten(-2).contiguous().view(torch.int16)[nz])
    for a, b in zip(g1[1:], g2[2:]):
        assert torch.equal(iv(a), iv(b))
