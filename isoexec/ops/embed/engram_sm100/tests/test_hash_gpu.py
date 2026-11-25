"""``ngram_hash`` against a CPU statement of its program (the per-(b, t, l) loop: compressed ids, -1 for dead /
absent / masked positions, blocking from the first -1 on, pad id, rolling xor of id * multiplier, modulo the head
prime, plus the head offset): bitwise at decode (history state) and prefill shapes, with masks, the error word
for out-of-vocabulary ids, and the returned history."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.embed.engram_sm100.ops import ngram_hash  # noqa: E402

L, N, H, V, VC, PAD = 2, 4, 8, 5000, 3000, 7


def _consts(seed):
    g = torch.Generator().manual_seed(seed)
    primes = torch.tensor([[[1_000_003 + 2 * (l * 100 + i * 10 + h) * 1 + 0 for h in range(H)] for i in range(N - 1)]
                           for l in range(L)], dtype=torch.int64) | 1
    offsets = (primes.flatten(1).cumsum(-1) - primes.flatten(1))
    mult = torch.randint(0, 1 << 40, (L, N), generator=g, dtype=torch.int64) * 2 + 1
    tmap = torch.randint(0, VC, (V,), generator=g, dtype=torch.int64)
    return primes, offsets, mult, tmap


def _cpu(ids, tmap, primes, offsets, mult, pad, mask=None, state=None):
    B, T = ids.shape
    cols = (N - 1) * H
    out = torch.empty(B, T, L, cols, dtype=torch.int64)
    M = (1 << 64) - 1
    for b in range(B):
        for t in range(T):
            for l in range(L):
                tok, blocked = [], False
                for s in range(N):
                    q = t - s
                    if q >= 0:
                        raw = int(ids[b, q])
                        c = -1 if (raw < 0 or raw >= V or (mask is not None and not mask[b, q])) else int(tmap[raw])
                    else:
                        c = int(state[b, N - 1 + q]) if state is not None else -1
                    blocked = blocked or c == -1
                    tok.append(pad if blocked else c)
                rolling = (tok[0] * int(mult[l, 0])) & M
                for i in range(1, N):
                    rolling ^= (tok[i] * int(mult[l, i])) & M
                    for h in range(H):
                        p = int(primes[l, i - 1, h])
                        out[b, t, l, (i - 1) * H + h] = rolling % p + int(offsets[l, (i - 1) * H + h])
    return out


@pytest.mark.parametrize("B,T,with_state,with_mask", [(1, 1, True, False), (5, 1, True, True), (1, 37, False, False),
                                                      (2, 9, True, True), (64, 1, True, False),
                                                      (3, 100, True, True), (77, 1, True, True),
                                                      (2, 2100, True, True)])
def test_hash_is_the_cpu_program(B, T, with_state, with_mask):
    primes, offsets, mult, tmap = _consts(B * 100 + T)
    g = torch.Generator().manual_seed(B + T)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = torch.rand(B, T, generator=g) > 0.2 if with_mask else None
    state = torch.randint(-1, VC, (B, N - 1), generator=g) if with_state else None
    want = _cpu(ids, tmap, primes, offsets, mult, PAD, mask, state)
    d = lambda t: None if t is None else t.cuda()
    got, new_state = ngram_hash(ids.cuda(), tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD,
                                token_mask=d(mask), state=d(state), return_state=True)
    assert torch.equal(got.cpu(), want)
    # history continuity: one call == the same tokens in two calls
    if T > 2:
        a, s1 = ngram_hash(ids[:, :2].cuda(), tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD,
                           token_mask=None if mask is None else d(mask[:, :2]), state=d(state), return_state=True)
        b_, _ = ngram_hash(ids[:, 2:].cuda(), tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD,
                           token_mask=None if mask is None else d(mask[:, 2:]), state=s1, return_state=True)
        assert torch.equal(torch.cat([a, b_], 1).cpu(), want)


def test_out_of_vocabulary_id_raises_when_validated():
    primes, offsets, mult, tmap = _consts(1)
    ids = torch.tensor([[3, V + 5, 4]]).cuda()
    with pytest.raises(ValueError):
        ngram_hash(ids, tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD)
    ngram_hash(ids, tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD, validate=False)


def test_reciprocal_path_equals_the_modulo_path():
    """The kernel with the per-prime reciprocal == the kernel's plain modulo (no reciprocal: the capture fallback),
    on real-size primes (~2^24) and large multipliers, every column."""
    from isoexec.ops.embed.engram_sm100 import ops as eo
    g = torch.Generator().manual_seed(77)
    primes = (torch.randint(1 << 23, 1 << 25, (L, N - 1, H), generator=g) | 1)
    offsets = (primes.flatten(1).cumsum(-1) - primes.flatten(1))
    mult = torch.randint(0, 1 << 40, (L, N), generator=g, dtype=torch.int64) * 2 + 1
    tmap = torch.randint(0, VC, (V,), generator=g, dtype=torch.int64)
    ids = torch.randint(0, V, (3, 4096), generator=g).cuda()
    args = (tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD)
    with_magic = eo.ngram_hash(ids, *args, validate=False)
    saved = eo._hash_magic
    eo._hash_magic = lambda p: None
    try:
        plain = eo.ngram_hash(ids, *args, validate=False)
    finally:
        eo._hash_magic = saved
    assert torch.equal(with_magic, plain)


@pytest.mark.parametrize("B,T", [(1, 1), (64, 1), (77, 1), (1, 37), (2, 2100), (3, 5000)])
def test_raw_state_is_the_mapped_state(B, T):
    """``raw_state`` (the engine's raw lookback: int32 token ids, newest first, -1 none) gives exactly the hashes of
    ``state = where(h >= 0, tmap[h.clamp(min=0)], -1)`` with ``h = raw.flip(-1)`` (the engine glue it replaces), on
    the per-column and the tiled kernel."""
    primes, offsets, mult, tmap = _consts(B * 1000 + T)
    g = torch.Generator().manual_seed(T)
    ids = torch.randint(0, V, (B, T), generator=g, dtype=torch.int64).cuda()
    raw = torch.randint(-1, V, (B, N - 1), generator=g, dtype=torch.int32).cuda()
    raw[::3, -1] = -1
    args = (tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD)
    h = raw.flip(-1).long()
    state = torch.where(h >= 0, tmap.cuda()[h.clamp(min=0)], -1)
    ref = ngram_hash(ids, *args, state=state.contiguous(), validate=False)
    got = ngram_hash(ids, *args, raw_state=raw, validate=False)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("lengths", [[1], [5, 1, 9], [37, 2, 1, 120, 7], [3000, 1500, 700]])
def test_seg_start_is_one_call_per_sequence(lengths):
    """``seg_start`` (each packed token's sequence start) hashes every packed sequence from an empty history in one
    launch, exactly the concatenation of one ``state=None`` call per sequence (the trainer's packed rows), on the
    per-column and the tiled kernel, with the error word for an out-of-vocabulary id."""
    primes, offsets, mult, tmap = _consts(sum(lengths))
    g = torch.Generator().manual_seed(len(lengths))
    T = sum(lengths)
    ids = torch.randint(0, V, (1, T), generator=g, dtype=torch.int64).cuda()
    args = (tmap.cuda(), primes.cuda(), offsets.cuda(), mult.cuda(), PAD)
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    ref = torch.cat([ngram_hash(ids[:, a:b], *args) for a, b in zip(bounds[:-1], bounds[1:])], dim=1)
    seg = torch.tensor([a for a, b in zip(bounds[:-1], bounds[1:]) for _ in range(b - a)], dtype=torch.int64).cuda()
    assert torch.equal(ngram_hash(ids, *args, seg_start=seg), ref)
    bad = ids.clone()
    bad[0, -1] = V + 3
    with pytest.raises(ValueError):
        ngram_hash(bad, *args, seg_start=seg)
