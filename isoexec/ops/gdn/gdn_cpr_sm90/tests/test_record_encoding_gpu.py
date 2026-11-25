"""Record format 3: the fp32 boundary state is stored EXACTLY as (sh = rn(S), lo = low 16 bits, tie bit).

* Host codec: every fp32 bit pattern class (ties, subnormals, infinities, +-0) round-trips exactly.
* Device round trip, bit for bit: a state loaded through the host encoder (ties of both parities, every normal
  exponent) is decoded by the kernels, carried through a chunk whose update is the identity (beta = 0 and a
  decay of exactly 1, so S_{c+1} = S_c * 1 + 0), and re-encoded by the boundary writer -- the decode step's in-kernel
  resync, and every prefill plan's commit (sweep / fused / prepared). The decoded result equals the original.
* Every record any kernel leaves is a valid code: sh == rn(decoded S) everywhere, and a tie bit is set only where
  lo == 0x8000, after real (non-identity) prefill and decode across several chunk boundaries.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, SM90, fresh, ids, make, rows

pytestmark = SM90
HK, HV = 2, 4


def _hard_states(n: int, hv: int, seed: int) -> torch.Tensor:
    """fp32 [n, hv, 128, 128] states full of the code's edge cases: ties of both parities, the whole normal
    exponent range, both signs. No NaN, no -0 (the identity chunk maps -0 to +0, which is the recipe, not the code)
    and no subnormals (whether the tensor-core accumulate keeps a subnormal seed is the recipe's matter; the host
    codec test covers subnormals, and the device decode is integer arithmetic)."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 2**31 - 1, (n * hv * 128 * 128,), generator=g, dtype=torch.int64)
    u = 0x00800000 + u % (0x7F000000 - 0x00800000)        # normal finite magnitudes, every exponent
    sign = torch.randint(0, 2, u.shape, generator=g, dtype=torch.int64) << 31
    k = u.numel()
    u[: k // 4] = (u[: k // 4] & ~0xFFFF) | 0x8000        # ties, both parities of the upper half
    u = u | sign
    u[u == (1 << 31)] = 1                                 # no -0
    x = torch.where(u >= 2**31, u - 2**32, u).to(torch.int32).view(torch.float32)
    return x.view(n, hv, 128, 128).to(DEV)


def test_host_codec_round_trips_every_value_class():
    g = torch.Generator().manual_seed(0)
    u = torch.randint(-2**31, 2**31 - 1, (1 << 18,), generator=g, dtype=torch.int64)
    u[:8192] = (u[:8192] & ~0xFFFF) | 0x8000
    u[8192:8192 + 4096] = torch.arange(4096)              # +0 and subnormals
    u[12288:12288 + 4096] = torch.arange(4096) | (1 << 31)
    u[16384:16388] = torch.tensor([0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x7F7F8000])
    x = torch.where(u >= 2**31, u - 2**32, u).to(torch.int32).view(torch.float32)
    x = x[~torch.isnan(x)]
    x = x[: x.numel() // 8 * 8]
    b, lo, tie = G.encode_state(x)
    bits = ((tie.to(torch.int64).unsqueeze(-1) >> torch.arange(8)) & 1).reshape(-1)
    assert torch.equal(G.decode_state(b, lo, bits).view(torch.int32), x.view(torch.int32))
    assert torch.equal(b, x.to(torch.bfloat16).view(torch.int16))
    assert int(bits.sum()) > 0


def _identity_inputs(T, seed):
    d = make([T], HK, HV, seed=seed)
    d["b"] = torch.full_like(d["b"], -100.0)              # beta = 1 / (1 + e^100) = 0
    d["a"] = torch.full_like(d["a"], -100.0)              # softplus ~ 4e-44: the decay rounds to exactly 1
    d["dt_bias"] = torch.zeros_like(d["dt_bias"])
    return d


@pytest.mark.parametrize("split", [1, 2, 8])
def test_decode_boundary_reencodes_the_state_exactly(split):
    n = 3
    S0 = _hard_states(n, HV, seed=split)
    st = fresh(n + 1, HV)
    st.set_S(torch.arange(1, n + 1, device=DEV), S0)
    assert bitwise_equal(st.S[1:], S0), "host codec round trip"
    d = _identity_inputs(64 * n, seed=7)
    sl = ids(range(1, n + 1))
    for t in range(64):                                   # pos 0 .. 63: the last step completes the chunk
        r = torch.tensor([64 * i + t for i in range(n)], device=DEV)
        G.decode(*rows(d, r), d["A_log"], d["dt_bias"], st, sl, split=split)
    torch.cuda.synchronize()
    assert int(st.pos[1, 0, 0]) == 64
    assert bitwise_equal(st.S[1:], S0), "decode boundary: decoded S_c -> S_c * 1 + 0 -> encoded"


def test_prefill_commit_reencodes_the_state_exactly(prefill_mode):
    n = 2
    S0 = _hard_states(n, HV, seed=11)
    st = fresh(n + 1, HV)
    st.set_S(torch.arange(1, n + 1, device=DEV), S0)
    d = _identity_inputs(128 * n, seed=8)
    q, k, v, a, b = rows(d, slice(None))
    G.prefill(q, k, v, a, b, d["A_log"], d["dt_bias"], torch.tensor([0, 128, 256], device=DEV, dtype=torch.int32),
              state=st, slot_in=ids([1, 2]))
    torch.cuda.synchronize()
    assert bitwise_equal(st.S[1:], S0), "prefill: decoded input S_0 -> two identity chunks -> encoded commit"


def _valid_code(st: G.CprState, slots) -> bool:
    for s in slots:
        S = st.S[s]
        sh = st.sh[s]
        if not bitwise_equal(sh, S.to(BF)):
            return False
        lo = st._raw("lo", torch.int16)[s].to(torch.int64) & 0xFFFF
        tie = st._tie_bits()[s]
        if bool(((tie == 1) & (lo != 0x8000)).any()):
            return False
    return True


def test_every_record_a_kernel_leaves_is_a_valid_code(prefill_mode):
    lanes = [200, 70, 129]
    d = make(lanes, HK, HV, seed=21)
    st = fresh(4, HV)
    pre = [130, 64, 1]
    sel = torch.cat([torch.arange(c, c + p) for c, p in zip(d["cu"], pre)]).to(DEV)
    cu = [0]
    for p in pre:
        cu.append(cu[-1] + p)
    q, k, v, a, b = rows(d, sel)
    G.prefill(q, k, v, a, b, d["A_log"], d["dt_bias"], torch.tensor(cu, device=DEV, dtype=torch.int32), state=st,
              slot_out=ids([1, 2, 3]), slot_in=ids([-1, -1, -1]))
    assert _valid_code(st, [1, 2, 3])
    for t in range(max(L - p for L, p in zip(lanes, pre))):   # decode on across further boundaries
        live = [i for i in range(3) if pre[i] + t < lanes[i]]
        r = torch.tensor([d["cu"][i] + pre[i] + t for i in live], device=DEV)
        G.decode(*rows(d, r), d["A_log"], d["dt_bias"], st, ids(i + 1 for i in live))
        if any((pre[i] + t + 1) % 64 == 0 for i in live):
            torch.cuda.synchronize()
            assert _valid_code(st, [i + 1 for i in live]), f"after decode step {t}"
