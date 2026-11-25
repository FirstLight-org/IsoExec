"""MLA relations compare raw bits and require a witness for each exercised hazard.
Numerical checks use reference.py and measured TOLERANCES; unsupported checks are skipped,
and a relation with no runnable checks is n/a."""

from __future__ import annotations

import dataclasses
import math
import time
import traceback
from typing import Callable

import torch

from isoexec.ops.attention.mla_sm90.tests._harness import (
    LATENT, PAGE, SCALE, WIDTH, Backend, Diff, Unsupported, assemble, bit_diff, decode, full, make_request, Slice,
)

# configuration and outcome


@dataclasses.dataclass
class Config:
    heads: int = 5
    seg: int = 512
    long_ctx: int = 8192 + 21  # not a multiple of 64: the last page is partial
    repeats: int = 50
    seed: int = 1000


@dataclasses.dataclass
class Outcome:
    relation: str
    backend: str
    status: str = "pass"  # pass | fail | skip | n/a | error
    diffs: list = dataclasses.field(default_factory=list)
    skipped: list = dataclasses.field(default_factory=list)
    witness: dict = dataclasses.field(default_factory=dict)
    metrics: dict = dataclasses.field(default_factory=dict)
    notes: list = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    @property
    def failures(self):
        return [d for d in self.diffs if not d.equal]

    def summary(self) -> str:
        bad = self.failures
        if self.status == "pass":
            return f"{len(self.diffs)} checks"
        if bad:
            return f"{len(bad)}/{len(self.diffs)} checks differ; first: {bad[0].name}: {bad[0].detail}"
        return "; ".join(self.notes + self.skipped)[:300]

    def as_dict(self):
        d = dataclasses.asdict(self)
        d["diffs"] = [x.as_dict() if isinstance(x, Diff) else x for x in self.diffs]
        return d


class Rec:
    """Collects diffs, skips, witnesses for one relation run."""

    def __init__(self, out: Outcome, backend: Backend):
        self.o, self.backend = out, backend

    def fwd(self, batch, **kw):
        return self.backend.fwd(batch, **kw)

    def cmp(self, name, a, b, labels=None):
        d = bit_diff(name, a, b, labels)
        self.o.diffs.append(d)
        return d.equal

    def cmp_rows(self, name, ra, rb, labels=None):
        """ra/rb = (out, lse|None) restricted to the same logical rows."""
        ok = self.cmp(f"{name}:out", ra[0], rb[0], labels)
        if ra[1] is not None and rb[1] is not None:
            ok = self.cmp(f"{name}:lse", ra[1], rb[1], labels) and ok
        return ok

    def skip(self, name, why):
        self.o.skipped.append(f"{name}: {why}")

    def witness(self, key, value):
        self.o.witness[key] = value

    def require(self, key, cond, why):
        """A hazard witness: a falsy witness turns the relation into a vacuous FAIL."""
        self.o.witness[key] = bool(cond)
        if not cond:
            self.o.notes.append(f"vacuous: witness {key!r} absent ({why})")


def rows_of(batch, i, res):
    out, lse = res
    s, e = batch.row_ranges[i]
    return out[s:e], (lse[s:e] if lse is not None else None)


# request pools


def _companions(cfg, k, n, base_seed, lengths=None):
    lengths = lengths or [37, 65, 700, 1500, 3000 + 5, cfg.long_ctx * 2 + 13]
    reqs = []
    for i in range(n):
        L = max(lengths[i % len(lengths)], k)
        reqs.append(make_request(L, cfg.heads, base_seed + i, tag=f"c{i}L{L}"))
    return reqs


def _tail(req, k, **kw):
    """The last k query rows of a request (k=1 is decode)."""
    return Slice(req, req.length - k, req.length, **kw)


def _mixed_slices(cfg, uniform_k=None):
    """A batch exercising SEG crossings, long-context decode, partial pages and ragged query lengths.

    With ``uniform_k`` every slice has that query length (FlashMLA-expressible)."""
    S = cfg.seg
    r_prompt = make_request(2 * S + 7, cfg.heads, cfg.seed + 1, tag="pre")  # full prefill across 2*SEG
    r_gen = make_request(cfg.long_ctx, cfg.heads, cfg.seed + 2, tag="gen")  # long-context decode
    r_mid = make_request(3 * S + 100, cfg.heads, cfg.seed + 3, tag="mid")  # chunk straddling SEG
    r_short = make_request(65, cfg.heads, cfg.seed + 4, tag="short")
    r_one = make_request(1, cfg.heads, cfg.seed + 5, tag="one")
    if uniform_k is not None:
        k = uniform_k
        return [_tail(r_prompt, k), _tail(r_gen, k), Slice(r_mid, S + 150 - k, S + 150),
                _tail(r_short, min(k, 65)) if k <= 65 else _tail(make_request(k + 3, cfg.heads, cfg.seed + 6), k)] + \
               ([_tail(r_one, 1)] if k == 1 else [])
    return [full(r_prompt), decode(r_gen, r_gen.length - 1), Slice(r_mid, S + 37, S + 300),
            _tail(r_gen, 70), full(r_short), full(r_one)]


# forward bitwise relations


def rel_batch_composition(rec: Rec, cfg: Config):
    """Target rows are bit-identical alone, with companions added/removed, reordered, and among many."""
    target = make_request(cfg.long_ctx, cfg.heads, cfg.seed + 10, tag="T")
    rec.witness("target_len", target.length)
    for mode, k in (("decode", 1), ("multirow", 7), ("prefill_chunk", 512)):
        comps = _companions(cfg, k, 5, cfg.seed + 20)
        many = _companions(cfg, k, 63, cfg.seed + 200, lengths=[129, 257, 1025, 2049])
        t = _tail(target, k)
        c = [_tail(r, k) for r in comps]
        layouts = {
            "alone": ([t], 0),
            "added": ([c[0], t, c[1], c[2]], 1),
            "removed": ([t, c[2]], 0),
            "reordered": ([c[2], c[1], c[4], t, c[0], c[3]], 3),
            "many": ([_tail(r, k) for r in many[:31]] + [t] + [_tail(r, k) for r in many[31:]], 31),
        }
        results = {}
        for name, (sl, idx) in layouts.items():
            b = assemble(sl, page_perm_seed=7)
            try:
                results[name] = rows_of(b, idx, rec.fwd(b))
                labels = b.labels_of(idx)
            except Unsupported as e:
                rec.skip(f"{mode}/{name}", str(e))
        if "alone" not in results:
            continue
        for name, r in results.items():
            if name != "alone":
                rec.cmp_rows(f"{mode}/{name}(B={len(layouts[name][0])})vs alone", r, results["alone"], labels)
        # Ragged companions (not FlashMLA-expressible): mixed query lengths around the target.
        sl = [_tail(comps[0], 1), t, _tail(comps[3], min(300, comps[3].length)), full(comps[1])]
        b = assemble(sl, page_perm_seed=3)
        try:
            rec.cmp_rows(f"{mode}/ragged vs alone", rows_of(b, 1, rec.fwd(b)), results["alone"], labels)
        except Unsupported as e:
            rec.skip(f"{mode}/ragged", str(e))


def rel_decode_vs_multirow(rec: Rec, cfg: Config):
    """A decode row equals the same logical row of a multi-row call, at long context and across SEG/2*SEG."""
    S = cfg.seg
    cases = [
        ("gen_long_k70", make_request(cfg.long_ctx, cfg.heads, cfg.seed + 30), 70),
        ("gen_across_seg", make_request(S + 3, cfg.heads, cfg.seed + 31), 8),
        ("gen_across_2seg", make_request(2 * S + 2, cfg.heads, cfg.seed + 32), 6),
        ("k65_across_seg", make_request(S + 90, cfg.heads, cfg.seed + 33), 65),
    ]
    for name, req, k in cases:
        multi = assemble([_tail(req, k)], page_perm_seed=1)
        try:
            m_out, m_lse = rec.fwd(multi)
        except Unsupported as e:
            rec.skip(name, str(e))
            continue
        d_out, d_lse = [], []
        for p in range(req.length - k, req.length):
            b = assemble([decode(req, p)], page_perm_seed=1)
            o, l = rec.fwd(b)
            d_out.append(o)
            d_lse.append(l)
        d_lse = None if d_lse[0] is None else torch.cat(d_lse)
        rec.cmp_rows(f"{name}(L={req.length},k={k})", (torch.cat(d_out), d_lse), (m_out, m_lse), multi.labels)
        # Batched decode of all k positions at once (vLLM decode step shape) vs multi-row.
        b = assemble([decode(req, p) for p in range(req.length - k, req.length)], page_perm_seed=2)
        rec.cmp_rows(f"{name}:batched_decode", rec.fwd(b), (m_out, m_lse), multi.labels)


def _chunk_calls(rec, req, bounds, **fwd_kw):
    outs, lses = [], []
    for a, e in zip(bounds[:-1], bounds[1:]):
        b = assemble([Slice(req, a, e)], page_perm_seed=a)
        o, l = rec.fwd(b, **fwd_kw)
        outs.append(o)
        lses.append(l)
    return torch.cat(outs), (None if lses[0] is None else torch.cat(lses))


def rel_chunked_prefill(rec: Rec, cfg: Config):
    """Chunked prefill == unchunked, with splits inside tiles, at tile edges, at SEG and SEG+-1."""
    S = cfg.seg
    L, P = 3 * S + 100, S + 200  # P: an arbitrary split point inside the second segment
    req = make_request(L, cfg.heads, cfg.seed + 40, tag="cp")
    whole = assemble([full(req)], page_perm_seed=0)
    try:
        ref = rec.fwd(whole)
    except Unsupported as e:
        rec.skip("unchunked", str(e))
        return
    g = torch.Generator().manual_seed(cfg.seed)
    rnd, x = [0], 0
    while x < L:
        x = min(L, x + int(torch.randint(1, 400, (1,), generator=g)))
        rnd.append(x)
    chunkings = {
        "uniform256": list(range(0, L, 256)) + [L],
        "irregular": [0, 1, 64, 127, 128, 300, P - 37, P + 55, 2 * S, 2 * S + 1, L],
        "random": rnd,
        "prefill_then_tokens": [0, P] + list(range(P + 1, P + 20)) + [L],
        "at_seg": [0, S - 1, S, S + 1, 2 * S, L],
    }
    rec.require("rows_beyond_seg", L > 2 * S, "rows with 3+ segments")
    for name, bounds in chunkings.items():
        try:
            got = _chunk_calls(rec, req, bounds)
        except Unsupported as e:
            rec.skip(name, str(e))
            continue
        rec.cmp_rows(f"{name}({len(bounds) - 1} chunks)", got, ref, whole.labels)
        # Batched chunks: all chunks of this chunking in ONE call (a scheduler step holding several).
    sl = [Slice(req, a, e) for a, e in zip(chunkings["irregular"][:-1], chunkings["irregular"][1:])]
    b = assemble(sl, page_perm_seed=9)
    try:
        rec.cmp_rows("irregular_all_in_one_call", rec.fwd(b), ref, whole.labels)
    except Unsupported as e:
        rec.skip("irregular_all_in_one_call", str(e))


def rel_exec_override(rec: Rec, cfg: Config):
    """Execution overrides (unit packing, grid, SM parts) are free: bits equal the default."""
    if not rec.backend.has_overrides:
        rec.o.status = "n/a"
        rec.o.notes.append("backend has no execution overrides")
        return
    variants = rec.backend.overrides()
    rec.require("override_variants_exported", len(variants) > 0,
                "fwd.py must export EXEC_OVERRIDE_VARIANTS or exec_override_variants(seg)")
    b = assemble(_mixed_slices(cfg), page_perm_seed=5)
    base = rec.fwd(b)
    for v in variants:
        rec.cmp_rows(f"override={v}", rec.fwd(b, exec_override=v), base, b.labels)


def rel_page_permutation(rec: Rec, cfg: Config):
    """Physical page ids (and table width/padding) are invisible."""
    for tag, sl in (("ragged", _mixed_slices(cfg)), ("uniform_k1", _mixed_slices(cfg, 1)),
                    ("uniform_k7", _mixed_slices(cfg, 7))):
        try:
            base = rec.fwd(assemble(sl))
        except Unsupported as e:
            rec.skip(tag, str(e))
            continue
        labels = assemble(sl).labels
        for seed in (1, 2, 3):
            rec.cmp_rows(f"{tag}/perm{seed}", rec.fwd(assemble(sl, page_perm_seed=seed)), base, labels)
        wide = assemble(sl, page_perm_seed=4, table_width=(cfg.long_ctx * 2) // PAGE + 5, table_pad="last")
        rec.cmp_rows(f"{tag}/wide_table", rec.fwd(wide), base, labels)


def rel_row_placement(rec: Rec, cfg: Config):
    """The same request at different batch slots and with different max_seqlen_q."""
    target = make_request(cfg.long_ctx, cfg.heads, cfg.seed + 60, tag="T")
    for k in (1, 5):
        comps = [_tail(r, k) for r in _companions(cfg, k, 4, cfg.seed + 61)]
        t = _tail(target, k)
        ref_b = assemble([t] + comps, page_perm_seed=0)
        try:
            ref = rows_of(ref_b, 0, rec.fwd(ref_b))
        except Unsupported as e:
            rec.skip(f"k{k}", str(e))
            continue
        for slot in (2, 4):
            sl = comps[:slot] + [t] + comps[slot:]
            b = assemble(sl, page_perm_seed=0)
            rec.cmp_rows(f"k{k}/slot{slot}", rows_of(b, slot, rec.fwd(b)), ref, ref_b.labels_of(0))
        for mq in (k + 37, 4096):
            b = assemble([t] + comps, page_perm_seed=0, max_seqlen_q=mq)
            rec.cmp_rows(f"k{k}/max_seqlen_q={mq}", rows_of(b, 0, rec.fwd(b)), ref, ref_b.labels_of(0))


def rel_repeat(rec: Rec, cfg: Config):
    """Run-to-run: the same call repeated cfg.repeats times."""
    for tag, sl in (("ragged", _mixed_slices(cfg)), ("uniform_k1", _mixed_slices(cfg, 1))):
        b = assemble(sl, page_perm_seed=11)
        try:
            first = rec.fwd(b)
        except Unsupported as e:
            rec.skip(tag, str(e))
            continue
        first = (first[0].clone(), None if first[1] is None else first[1].clone())
        n_bad = 0
        for i in range(cfg.repeats - 1):
            r = rec.fwd(b)
            ok = bit_diff("", r[0], first[0]).equal and (r[1] is None or bit_diff("", r[1], first[1]).equal)
            if not ok:
                n_bad += 1
                if n_bad == 1:
                    rec.cmp_rows(f"{tag}/repeat{i + 1}", r, first, b.labels)
        if n_bad == 0:
            rec.cmp_rows(f"{tag}/repeat x{cfg.repeats}", first, first)
        else:
            rec.o.notes.append(f"{tag}: {n_bad}/{cfg.repeats - 1} repeats differ")


def boundary_lengths(cfg):
    S = cfg.seg
    return sorted({1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, S - 1, S, S + 1, 2 * S - 1, 2 * S, 2 * S + 1,
                   4 * S + 64, 4 * S + 65})


def rel_boundary_lengths(rec: Rec, cfg: Config):
    """Borders at SEG-1/SEG/SEG+1, multiples of 64 and 128, length 1: decode alone == batched == multi-row."""
    lens = boundary_lengths(cfg)
    rec.witness("lengths", lens)
    for role in ("seed0", "seed1"):
        reqs = [make_request(L, cfg.heads, cfg.seed + 70 + L + (7919 if role == "seed1" else 0), tag=f"L{L}")
                for L in lens]
        alone = []
        for r in reqs:
            b = assemble([decode(r, r.length - 1)], page_perm_seed=r.length)
            try:
                alone.append(rec.fwd(b))
            except Unsupported as e:
                rec.skip(f"{role}/alone", str(e))
                return
        bb = assemble([decode(r, r.length - 1) for r in reqs], page_perm_seed=99)
        batched = rec.fwd(bb)
        for i, r in enumerate(reqs):
            rec.cmp_rows(f"{role}/L={r.length}/batched_decode", rows_of(bb, i, batched), alone[i], bb.labels_of(i))
            k = min(3, r.length)
            mb = assemble([_tail(r, k)], page_perm_seed=1)
            try:
                m = rec.fwd(mb)
            except Unsupported as e:
                rec.skip(f"{role}/L={r.length}/multirow", str(e))
                continue
            rec.cmp_rows(f"{role}/L={r.length}/multirow_last", (m[0][-1:], None if m[1] is None else m[1][-1:]),
                         alone[i], mb.labels[-1:])


def rel_nonfinite_unused(rec: Rec, cfg: Config):
    """NaN/inf/garbage beyond seqused_k (page tails, table padding, unreferenced pages) cannot reach bits."""
    for tag, sl in (("ragged", _mixed_slices(cfg)), ("uniform_k1", _mixed_slices(cfg, 1))):
        rec.require(f"{tag}:partial_last_page", any(s.used % PAGE for s in sl), "a slice with used % 64 != 0")
        try:
            clean = rec.fwd(assemble(sl, page_perm_seed=13, extra_pages=3, table_width=(cfg.long_ctx // PAGE) + 8))
        except Unsupported as e:
            rec.skip(tag, str(e))
            continue
        labels = assemble(sl).labels
        clean = (clean[0].clone(), None if clean[1] is None else clean[1].clone())
        for fill in ("future", "nan", "inf", "ninf", "garbage"):
            for pad in ("zero", "poison"):
                b = assemble(sl, page_perm_seed=13, extra_pages=3, fill=fill, table_pad=pad,
                             table_width=(cfg.long_ctx // PAGE) + 8)
                if fill in ("nan", "inf", "ninf"):
                    rec.require(f"{tag}:{fill}/{pad}:nonfinite_present", (~torch.isfinite(b.kv_cache)).any().item(),
                                "poison written")
                rec.cmp_rows(f"{tag}/{fill}/{pad}", rec.fwd(b), clean, labels)
        rec.witness(f"{tag}:clean_output_finite", bool(torch.isfinite(clean[0]).all()))


def rel_degenerate_shapes(rec: Rec, cfg: Config):
    """T = 0, zero-length queries, null lanes (seqused_k = 0) and vLLM profiling shapes."""
    h = cfg.heads
    dev = "cuda"

    def empty_batch(B, used):
        from isoexec.ops.attention.mla_sm90.tests._harness import Batch

        return Batch(q=torch.zeros(0, h, WIDTH, device=dev, dtype=torch.bfloat16),
                     kv_cache=torch.zeros(4, PAGE, 1, WIDTH, device=dev, dtype=torch.bfloat16),
                     block_table=torch.zeros(B, 4, device=dev, dtype=torch.int32),
                     cu_seqlens_q=torch.zeros(B + 1, device=dev, dtype=torch.int32),
                     seqused_k=torch.tensor(used, device=dev, dtype=torch.int32).view(B), max_seqlen_q=0,
                     row_ranges=[(0, 0)] * B, labels=[], slices=[None] * B, page_of=[])

    for name, B, used in (("T0_B0", 0, []), ("T0_B2", 2, [100, 5])):
        try:
            out, lse = rec.fwd(empty_batch(B, used))
            ok = tuple(out.shape) == (0, h, LATENT) and (lse is None or tuple(lse.shape) == (0, h))
            rec.o.diffs.append(Diff(f"{name}:shapes", ok, tuple(out.shape), 0 if ok else -1,
                                    "" if ok else f"out {tuple(out.shape)} lse {None if lse is None else tuple(lse.shape)}"))
        except Unsupported as e:
            rec.skip(name, str(e))
        except Exception as e:
            rec.o.diffs.append(Diff(f"{name}:no_crash", False, (), -1, f"{type(e).__name__}: {str(e)[:200]}"))

    # Null lane: a padded decode row with zero KV between real rows (full CUDA-graph decode shape).
    real = make_request(1000, h, cfg.seed + 95, tag="real")
    null = make_request(1, h, cfg.seed + 96, tag="null")
    alone_b = assemble([decode(real, 999)], page_perm_seed=0)
    try:
        alone = rec.fwd(alone_b)
        sl = [decode(real, 999), Slice(null, 0, 1, used=1)]
        b = assemble(sl, page_perm_seed=0)
        b.seqused_k[1] = 0  # the null lane: one query row, no keys
        res = rec.fwd(b)
        rec.cmp_rows("null_lane:real_row_unchanged", rows_of(b, 0, res), alone, alone_b.labels)
        fin = bool(torch.isfinite(res[0][1]).all())
        rec.o.diffs.append(Diff("null_lane:finite_out", fin, tuple(res[0][1].shape), 0 if fin else -1,
                                "" if fin else "null-lane output not finite"))
    except Unsupported as e:
        rec.skip("null_lane", str(e))
    except Exception as e:
        rec.o.diffs.append(Diff("null_lane:no_crash", False, (), -1, f"{type(e).__name__}: {str(e)[:200]}"))

    # vLLM profiling: max batch of 1-token rows over page 0 of an all-zero cache, and a big dummy prefill.
    try:
        B = 256
        from isoexec.ops.attention.mla_sm90.tests._harness import Batch

        pb = Batch(q=torch.randn(B, h, WIDTH, device=dev).to(torch.bfloat16),
                   kv_cache=torch.zeros(2, PAGE, 1, WIDTH, device=dev, dtype=torch.bfloat16),
                   block_table=torch.zeros(B, 1, device=dev, dtype=torch.int32),
                   cu_seqlens_q=torch.arange(B + 1, device=dev, dtype=torch.int32),
                   seqused_k=torch.ones(B, device=dev, dtype=torch.int32), max_seqlen_q=1,
                   row_ranges=[(i, i + 1) for i in range(B)], labels=[str(i) for i in range(B)],
                   slices=[None] * B, page_of=[])
        out, lse = rec.fwd(pb)
        ok = tuple(out.shape) == (B, h, LATENT) and bool(torch.isfinite(out).all())
        rec.o.diffs.append(Diff("profile_decode_B256:finite", ok, tuple(out.shape), 0 if ok else -1, ""))
        T = 8192
        pp = Batch(q=torch.randn(T, h, WIDTH, device=dev).to(torch.bfloat16),
                   kv_cache=torch.zeros(T // PAGE, PAGE, 1, WIDTH, device=dev, dtype=torch.bfloat16),
                   block_table=torch.arange(T // PAGE, device=dev, dtype=torch.int32).view(1, -1),
                   cu_seqlens_q=torch.tensor([0, T], device=dev, dtype=torch.int32),
                   seqused_k=torch.tensor([T], device=dev, dtype=torch.int32), max_seqlen_q=T,
                   row_ranges=[(0, T)], labels=[], slices=[None], page_of=[])
        out, lse = rec.fwd(pp)
        ok = tuple(out.shape) == (T, h, LATENT) and bool(torch.isfinite(out).all())
        rec.o.diffs.append(Diff("profile_prefill_T8192:finite", ok, tuple(out.shape), 0 if ok else -1, ""))
    except Unsupported as e:
        rec.skip("profiling", str(e))
    except Exception as e:
        rec.o.diffs.append(Diff("profiling:no_crash", False, (), -1, f"{type(e).__name__}: {str(e)[:200]}"))


def _logprob_request(cfg, seed, length):
    """One sequence in trainer terms (q_nope, q_pe, c_kv, k_pe) and in engine terms (absorbed q, paged kv)."""
    g = torch.Generator().manual_seed(seed)

    def r(*s, sc=1.0):
        return (torch.randn(*s, generator=g) * sc).to("cuda", torch.bfloat16)

    raw = [r(length, cfg.heads, 192), r(length, cfg.heads, 64), r(length, 512), r(length, 64)]
    return raw


def _logprob_weights(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(cfg.heads, 192, 512, generator=g) * 512 ** -.5).to("cuda", torch.bfloat16),
            (torch.randn(cfg.heads, 256, 512, generator=g) * 512 ** -.5).to("cuda", torch.bfloat16)]


def _engine_request(raw, wk, tag):
    from isoexec.ops.attention.mla_absorbed import absorb_query
    from isoexec.ops.attention.mla_sm90.tests._harness import Request

    q = torch.cat((absorb_query(raw[0], wk), raw[1]), dim=-1).contiguous()
    return Request(torch.cat((raw[2], raw[3]), dim=-1).contiguous(), q, tag)


def logprob_positions(cfg, L):
    S = cfg.seg
    pts = {0, 1, 63, 64, 127, 128, S - 2, S - 1, S, S + 1, 2 * S - 1, 2 * S, 2 * S + 1, 3 * S + 5, 4 * S - 1, 4 * S,
           5 * S + 64, 6 * S + 1, 7 * S - 1, 7 * S, 7 * S + 1, 8 * S - 1, 8 * S, 8 * S + 1, L - 2, L - 1}
    return sorted(p for p in pts if 0 <= p < L)


def rel_logprob_contract(rec: Rec, cfg: Config):
    """THE contract: for the same KV and logical position, a row's out and lse are bitwise identical in decode,
    MTP verify (s_q 2 and 4), full prefill (alone and batched), chunked prefill (splits at SEG, SEG-1, SEG+1,
    uniform and random), multi-turn layouts (a prefix prefilled, extended by decode, re-presented inside a
    later and longer prefill), and the trainer op (unabsorbed 256-dim rows vs engine fwd + unabsorb_output).
    Rows reach 9 segments."""
    from isoexec.ops.attention.mla_absorbed import unabsorb_output

    S = cfg.seg
    L = 8 * S + 77
    wk, wv = _logprob_weights(cfg, cfg.seed + 1200)
    raw = _logprob_request(cfg, cfg.seed + 1201, L)
    R = _engine_request(raw, wk, "R")
    other_raw = _logprob_request(cfg, cfg.seed + 1202, L)
    O = _engine_request(other_raw, wk, "O")
    rec.require("rows_with_8plus_segments", -(-L // S) >= 8, "n = ceil(L/SEG) >= 8")
    rec.witness("max_segments", -(-L // S))
    rec.witness("heads", cfg.heads)

    whole = assemble([full(R)], page_perm_seed=21)
    try:
        F_out, F_lse = rec.fwd(whole)
    except Unsupported as e:
        rec.skip("prefill", str(e))
        return

    def vs_prefill(name, res, positions):
        idx = torch.tensor(positions, device="cuda", dtype=torch.long)
        rec.cmp_rows(name, res, (F_out[idx], None if F_lse is None else F_lse[idx]),
                     [f"R@{p}" for p in positions])

    def run(name, slices, **kw):
        b = assemble(slices, **kw)
        try:
            return b, rec.fwd(b)
        except Unsupported as e:
            rec.skip(name, str(e))
            return b, None

    targets = logprob_positions(cfg, L)
    rec.witness("positions", targets)

    # 1. decode, one call per position (s_q = 1)
    outs, lses = [], []
    for p in targets:
        _, res = run("decode_alone", [decode(R, p)], page_perm_seed=p)
        outs.append(res[0])
        lses.append(res[1])
    vs_prefill("decode_alone", (torch.cat(outs), None if lses[0] is None else torch.cat(lses)), targets)

    # 2. decode, all positions in one step next to another request's decode rows
    sl = []
    for p in targets:
        sl += [decode(R, p), decode(O, min(L - 1, p + 700))]
    b, res = run("decode_batched", sl, page_perm_seed=5)
    if res is not None:
        vs_prefill("decode_batched", (res[0][0::2], None if res[1] is None else res[1][0::2]), targets)

    # 3. MTP verify: s_q = 2 and 4, the target is the LAST row of its verify window, and also not the last
    for k in (2, 4):
        for where in ("last", "first"):
            sl, pos = [], []
            for p in targets:
                end = p + 1 if where == "last" else min(L, p + k)
                start = end - k
                if start < 0:
                    continue
                sl.append(Slice(R, start, end))
                pos += list(range(start, end))
            b, res = run(f"verify_k{k}_{where}", sl, page_perm_seed=k)
            if res is not None:
                vs_prefill(f"verify_s_q={k}/target_{where}", res, pos)

    # 4. full prefill inside a batch (companion first, so R's rows sit at a nonzero packed offset)
    b, res = run("prefill_batched", [full(O), full(R)], page_perm_seed=6)
    if res is not None:
        vs_prefill("prefill_batched", rows_of(b, 1, res), list(range(L)))

    # 5. chunked prefill
    def uniform(c):
        return list(range(0, L, c)) + [L]

    g = torch.Generator().manual_seed(cfg.seed + 1203)
    rnd, x = [0], 0
    while x < L:
        x = min(L, x + int(torch.randint(1, 2 * S, (1,), generator=g)))
        rnd.append(x)
    chunkings = {
        "split_at_SEG": [0, S, L], "split_at_SEG-1": [0, S - 1, L], "split_at_SEG+1": [0, S + 1, L],
        "uniform_SEG": uniform(S), "uniform_SEG-1": uniform(S - 1), "uniform_SEG+1": uniform(S + 1),
        "uniform_2SEG+1": uniform(2 * S + 1), "uniform_97": uniform(97),
        "mixed_edges": [0, 1, 64, S - 1, 2 * S, 2 * S + 1, 4 * S - 1, 7 * S + 1, 8 * S, L], "random": rnd,
    }
    for name, bounds in chunkings.items():
        try:
            vs_prefill(f"chunked/{name}({len(bounds) - 1})", _chunk_calls(rec, R, bounds), list(range(L)))
        except Unsupported as e:
            rec.skip(f"chunked/{name}", str(e))
    bounds = chunkings["mixed_edges"]
    b, res = run("chunks_one_call", [Slice(R, a, e) for a, e in zip(bounds[:-1], bounds[1:])], page_perm_seed=8)
    if res is not None:
        vs_prefill("chunked/mixed_edges_all_in_one_call", res, list(range(L)))

    # 6. multi-turn: turn 1 prefills a prefix, generates by decode, turn 2 re-presents all of it as prompt inside a
    #    longer prefill (the unchunked prefill F is turn 3: the trainer's view of the whole conversation).
    t1 = 3 * S + 17
    gen1 = 2 * S + 3  # turn-1 generation crosses two SEG boundaries
    t2_end = t1 + gen1 + S + 40
    b, res = run("turn1_prefill", [Slice(R, 0, t1)], page_perm_seed=31)
    turn_out, turn_lse = [res[0]], [res[1]]
    step_out, step_lse = [], []
    for p in range(t1, t1 + gen1):
        _, r1 = run("turn1_decode", [decode(R, p), decode(O, p)], page_perm_seed=32)
        step_out.append(r1[0][:1])
        step_lse.append(None if r1[1] is None else r1[1][:1])
    turn_out.append(torch.cat(step_out))
    turn_lse.append(None if step_lse[0] is None else torch.cat(step_lse))
    _, r2 = run("turn2_prefill_suffix", [Slice(R, t1 + gen1, t2_end)], page_perm_seed=33)  # cached prefix + new prompt
    turn_out.append(r2[0])
    turn_lse.append(r2[1])
    lse_cat = None if turn_lse[0] is None else torch.cat(turn_lse)
    vs_prefill("multi_turn/turn1_prefill+decode+turn2_suffix vs final prefill", (torch.cat(turn_out), lse_cat),
               list(range(t2_end)))
    _, r3 = run("turn2_full_reprefill", [Slice(R, 0, t2_end), full(O)], page_perm_seed=34)  # recompute after preemption
    if r3 is not None:
        vs_prefill("multi_turn/turn2_full_reprefill(batched)", (r3[0][:t2_end], None if r3[1] is None else r3[1][:t2_end]),
                   list(range(t2_end)))
    rec.require("multi_turn_generation_beyond_seg", t1 + gen1 > 2 * S and gen1 > S,
                "turn-1 decode rows cross SEG boundaries before being re-prefilled")

    # 7. trainer: mla_sm90_attention's forward over packed sequences, compared on unabsorbed rows
    if rec.backend.name != "mla_sm90":
        rec.skip("trainer", "trainer op exists only for mla_sm90")
        return
    if rec.backend.seg_tokens != cfg.seg:
        rec.skip("trainer", "backend SEG differs from cfg.seg")
        return
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    eng_y = unabsorb_output(F_out, wv)
    dec_y = unabsorb_output(torch.cat(outs), wv)
    short_raw = _logprob_request(cfg, cfg.seed + 1204, 333)
    for name, grad_on, order in (("trainer_alone_nograd", False, ["R"]),
                                 ("trainer_packed_grad", True, ["S", "R", "O"]),
                                 ("trainer_packed_reversed_grad", True, ["O", "R", "S"])):
        parts = {"R": raw, "O": other_raw, "S": short_raw}
        lens = [parts[k][0].shape[0] for k in order]
        leaves = [torch.cat([parts[k][j] for k in order]).detach().requires_grad_(grad_on) for j in range(4)]
        w = [t.detach().requires_grad_(grad_on) for t in (wk, wv)]
        with torch.set_grad_enabled(grad_on):
            y = mla_sm90_attention(*leaves, *w, seq_lens=lens, scale=SCALE, seg_tokens=cfg.seg)
        off = sum(lens[:order.index("R")])
        yR = y.detach()[off:off + L]
        rec.cmp(f"{name}: op rows vs unabsorb(engine prefill)", yR, eng_y, [f"R@{p}" for p in range(L)])
        idx = torch.tensor(targets, device="cuda", dtype=torch.long)
        rec.cmp(f"{name}: op rows vs unabsorb(engine decode)", yR[idx], dec_y, [f"R@{p}" for p in targets])


def rel_role_split_detected(rec: Rec, cfg: Config):
    """Sensitivity: the retired role split (prompt rows unsplit, generated rows segmented) breaks multi-turn.

    Emulation with the real kernel: turn 1 generates rows by decode under the deployment SEG; turn 2 prefills
    the same tokens as prompt under a one-segment program (SEG = 2^20, so every row is root(leaf) of the
    flat chain). The relation PASSES when the suite detects the split: rows with border > SEG differ, and rows
    with border <= SEG (where both programs are root(leaf)) are identical."""
    if not getattr(rec.backend, "has_seg", False):
        rec.o.status = "n/a"
        rec.o.notes.append("needs a SEG-parametrized backend")
        return
    S = cfg.seg
    L = 3 * S + 41
    R = make_request(L, cfg.heads, cfg.seed + 1300, tag="R")
    positions = [S - 2, S - 1, S, S + 1, 2 * S, 3 * S + 40]
    dec = [rec.fwd(assemble([decode(R, p)], page_perm_seed=p)) for p in positions]  # turn 1, generated
    pre = rec.fwd(assemble([full(R)], page_perm_seed=1), seg_tokens=1 << 20)  # turn 2, "prompt role"
    short = [i for i, p in enumerate(positions) if p + 1 <= S]
    long_ = [i for i, p in enumerate(positions) if p + 1 > S]
    for i in short:
        p = positions[i]
        rec.cmp_rows(f"border {p + 1} <= SEG: decode vs role-split prefill", dec[i], (pre[0][p:p + 1], pre[1][p:p + 1]))
    differ = [positions[i] for i in long_
              if not bit_diff("", dec[i][0], pre[0][positions[i]:positions[i] + 1]).equal]
    rec.witness("rows_beyond_seg_that_differ", differ)
    rec.require("role_split_breaks_multi_turn", len(differ) == len(long_),
                f"every row with border > SEG must differ between the programs, got {differ}")


def rel_cuda_graph(rec: Rec, cfg: Config):
    """Full decode CUDA graph (vLLM/SGLang): a capacity plan and out buffers captured once, replayed with new
    lengths, pages and a null lane, bit-identical to eager calls."""
    backend = rec.backend
    if not hasattr(backend, "_fwd_mod"):
        rec.o.status = "n/a"
        rec.o.notes.append("needs a plan-capable backend (mla_sm90)")
        return
    from isoexec.ops.attention.mla_sm90.ops import kernel_fwd, kernel_plan
    h, B, k = cfg.heads, 8, 2
    max_k = 4 * cfg.seg + 100
    mb = (max_k + PAGE - 1) // PAGE
    nb = B * mb + 1
    dev = "cuda"
    q = torch.zeros(B * k, h, WIDTH, device=dev, dtype=torch.bfloat16)
    kv = torch.zeros(nb, PAGE, 1, WIDTH, device=dev, dtype=torch.bfloat16)
    table = torch.zeros(B, mb, device=dev, dtype=torch.int32)
    cu = torch.arange(0, (B + 1) * k, k, device=dev, dtype=torch.int32)
    used = torch.zeros(B, device=dev, dtype=torch.int32)
    o = torch.empty(B * k, h, LATENT, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(B * k, h, device=dev, dtype=torch.float32)
    cap = {"max_seqlen_k": max_k, "num_tokens": B * k}

    def call():
        plan = kernel_plan(cu, used, cfg.seg, h, k, capacity=cap)
        return kernel_fwd(q, kv, table, cu, used, SCALE, cfg.seg, k, plan=plan, out=(o, lse))

    def load(step):
        g = torch.Generator().manual_seed(cfg.seed + 1100 + step)
        lens = torch.randint(k, max_k + 1, (B,), generator=g)
        lens[step % B] = k if step else max_k  # boundary: shortest and longest
        null = (step + 3) % B
        q.copy_(torch.randn(q.shape, generator=g).to(dev, torch.bfloat16))
        kv.copy_(torch.randn(kv.shape, generator=g).to(dev, torch.bfloat16))
        perm = torch.randperm(nb, generator=g)[: B * mb].view(B, mb).to(torch.int32)
        table.copy_(perm)
        used.copy_(lens.to(torch.int32))
        used[null] = 0 if step % 2 else k  # a padded lane with no KV on odd steps

    load(0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for step in range(4):
        load(step)
        graph.replay()
        torch.cuda.synchronize()
        got = (o.clone(), lse.clone())
        eager = backend.fwd(dataclasses.replace(
            assemble([]), q=q, kv_cache=kv, block_table=table, cu_seqlens_q=cu, seqused_k=used,
            max_seqlen_q=k, row_ranges=[], labels=[], slices=[]))
        null = (step + 3) % B
        keep = torch.ones(B * k, dtype=torch.bool, device=dev)
        if step % 2:
            keep[null * k:(null + 1) * k] = False  # null-lane rows: compare finiteness, not bits
            fin = bool(torch.isfinite(got[0][~keep]).all())
            rec.o.diffs.append(Diff(f"replay{step}:null_lane_finite", fin, (), 0 if fin else -1, ""))
        rec.cmp_rows(f"replay{step}:graph vs eager", (got[0][keep], got[1][keep]), (eager[0][keep], eager[1][keep]))


# numerical relations

# Measured tolerance margins are documented in tests/README.md under "Tolerances".
# Output error is normalized by max|V|; LSE error is absolute. Operator and gradient
# bounds include bf16 projection rounding and match the unabsorbed-FA3 backward gate.
TOLERANCES = {
    "out_abs_over_vmax": 5.0e-3,
    "lse_abs": 1.0e-4,
    "op_y_rel": 1.0e-2,
    "op_grad_rel": 2.5e-2,
}


def _fwd_errors(batch, res, ref):
    from isoexec.ops.attention.mla_sm90.reference import mla_reference_fwd  # noqa: F401  (import check)

    out, lse = res
    r_out, r_lse = ref
    nonempty = []
    for i, s in enumerate(batch.slices):
        a, e = batch.row_ranges[i]
        nonempty += [p >= 0 for p in range(s.used - s.q_len, s.used)]
    keep = torch.tensor(nonempty, device=out.device)
    vmax = batch.kv_cache[..., :LATENT].float().abs().nan_to_num(0).amax().item()
    err = (out.double() - r_out)[keep].abs()
    m = {"out_max_abs": err.max().item() if err.numel() else 0.0,
         "out_max_abs_over_vmax": (err.max().item() / vmax) if err.numel() else 0.0,
         "out_rel_fro": ((out.double() - r_out)[keep].norm() / r_out[keep].norm().clamp_min(1e-30)).item()}
    if lse is not None:
        le = (lse.double() - r_lse)[keep].abs()
        m["lse_max_abs"] = le.max().item() if le.numel() else 0.0
    return m


def rel_fwd_vs_reference(rec: Rec, cfg: Config):
    from isoexec.ops.attention.mla_sm90.reference import mla_reference_fwd

    worst = {}
    for dist, qs in (("unit", 1.0), ("peaky", 3.0)):
        S = cfg.seg
        r1 = make_request(2 * S + 7, cfg.heads, cfg.seed + 501, q_scale=qs, tag="pre")
        r2 = make_request(3 * S + 100, cfg.heads, cfg.seed + 502, q_scale=qs, tag="mid")
        r3 = make_request(4 * S + 65, cfg.heads, cfg.seed + 503, q_scale=qs, tag="gen")
        for tag, sl in (("ragged", [full(r1), Slice(r2, S, 3 * S + 100), _tail(r3, 70), decode(r3, 4 * S + 64)]),
                        ("uniform_k1", [decode(r1, 2 * S + 6), decode(r2, 3 * S + 99), decode(r3, 4 * S + 64)])):
            b = assemble(sl, page_perm_seed=8)
            try:
                res = rec.fwd(b)
            except Unsupported as e:
                rec.skip(f"{dist}/{tag}", str(e))
                continue
            ref = mla_reference_fwd(b.q, b.kv_cache, b.block_table, b.cu_seqlens_q, b.seqused_k, SCALE)
            m = _fwd_errors(b, res, ref)
            rec.o.metrics[f"{dist}/{tag}"] = m
            for k, v in m.items():
                worst[k] = max(worst.get(k, 0.0), v)
    rec.o.metrics["worst"] = worst
    _judge(rec, "out_max_abs_over_vmax", worst, "out_abs_over_vmax")
    _judge(rec, "lse_max_abs", worst, "lse_abs")


def _judge(rec, metric, worst, tol_key):
    if metric not in worst:
        return
    ok = worst[metric] <= TOLERANCES[tol_key]
    rec.o.diffs.append(Diff(f"{metric}<={TOLERANCES[tol_key]:.1e}", ok, (), 0 if ok else 1,
                            "" if ok else f"measured {worst[metric]:.3e}"))


# op-level relations (ops.attention.mla_sm90_attention)


def _op_inputs(cfg, lens, seed, nope=192, v=256):
    g = torch.Generator().manual_seed(seed)
    T, H = sum(lens), cfg.heads

    def r(*s, sc=1.0):
        return (torch.randn(*s, generator=g) * sc).to("cuda", torch.bfloat16)

    return [r(T, H, nope), r(T, H, 64), r(T, 512), r(T, 64), r(H, nope, 512, sc=512 ** -0.5),
            r(H, v, 512, sc=512 ** -0.5)]


def _op_run(cfg, tensors, lens, grad):
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    y = mla_sm90_attention(*leaves, seq_lens=lens, scale=SCALE, seg_tokens=cfg.seg)
    y.backward(grad)
    return y.detach(), [t.grad for t in leaves]


def rel_op_packing(rec: Rec, cfg: Config):
    """The autograd op: per-sequence y is invariant to packing order (the backward is not deterministic)."""
    if rec.backend.name != "mla_sm90" or not rec.backend.has_backward:
        rec.o.status = "n/a"
        rec.o.notes.append("needs mla_sm90 fwd and the FA3 backward")
        return
    S = cfg.seg
    lens = [S + 77, 300, 1, 2 * S + 3]
    per = [_op_inputs(cfg, [n], cfg.seed + 700 + i) for i, n in enumerate(lens)]
    wk, wv = per[0][4], per[0][5]
    grads = [(torch.randn(n, cfg.heads, 256, generator=torch.Generator().manual_seed(cfg.seed + 800 + i)) * .1)
             .to("cuda", torch.bfloat16) for i, n in enumerate(lens)]

    def run(order):
        ts = [torch.cat([per[i][j] for i in order]) for j in range(4)] + [wk, wv]
        y, g = _op_run(cfg, ts, [lens[i] for i in order],
                       torch.cat([grads[i] for i in order]))
        res, off = {}, 0
        for i in order:
            n = lens[i]
            res[i] = (y[off:off + n], [x[off:off + n] for x in g[:4]])
            off += n
        return res

    alone = {i: run([i])[i] for i in range(len(lens))}
    for name, order in (("packed", [0, 1, 2, 3]), ("reversed", [3, 2, 1, 0]), ("subset", [3, 1])):
        got = run(order)
        for i in order:
            rec.cmp(f"{name}/seq{i}:y", got[i][0], alone[i][0])


def rel_op_vs_reference(rec: Rec, cfg: Config):
    if rec.backend.name != "mla_sm90" or not rec.backend.has_backward:
        rec.o.status = "n/a"
        rec.o.notes.append("needs mla_sm90 fwd and the FA3 backward")
        return
    from isoexec.ops.attention.mla_sm90.reference import reference_attention

    S = cfg.seg
    lens = [S + 77, 64, 2 * S + 5]
    ts = _op_inputs(cfg, lens, cfg.seed + 900)
    grad = (torch.randn(sum(lens), cfg.heads, 256, generator=torch.Generator().manual_seed(1)) * .1).to("cuda", torch.bfloat16)
    y, g = _op_run(cfg, ts, lens, grad)
    leaves = [t.detach().double().requires_grad_(True) for t in ts]
    ry = reference_attention(*leaves, seq_lens=lens, scale=SCALE)
    ry.backward(grad.double())
    m = {"y_rel": ((y.double() - ry).norm() / ry.norm()).item()}
    for name, a, r in zip(("q_nope", "q_pe", "c_kv", "k_pe", "w_uk", "w_uv"), g, leaves):
        m[f"grad_{name}_rel"] = ((a.double() - r.grad).norm() / r.grad.norm()).item()
    rec.o.metrics["worst"] = m
    ok = m["y_rel"] <= TOLERANCES["op_y_rel"]
    rec.o.diffs.append(Diff("op_y_rel", ok, (), 0 if ok else 1, "" if ok else f"{m['y_rel']:.3e}"))
    for k, v in m.items():
        if k.startswith("grad"):
            ok = v <= TOLERANCES["op_grad_rel"]
            rec.o.diffs.append(Diff(k, ok, (), 0 if ok else 1, "" if ok else f"{v:.3e}"))


# registry of relations and runner

RELATIONS: dict[str, Callable] = {
    "logprob_contract": rel_logprob_contract,
    "batch_composition": rel_batch_composition,
    "decode_vs_multirow": rel_decode_vs_multirow,
    "chunked_prefill": rel_chunked_prefill,
    "exec_override": rel_exec_override,
    "page_permutation": rel_page_permutation,
    "row_placement": rel_row_placement,
    "repeat": rel_repeat,
    "boundary_lengths": rel_boundary_lengths,
    "nonfinite_unused_pages": rel_nonfinite_unused,
    "degenerate_shapes": rel_degenerate_shapes,
    "cuda_graph": rel_cuda_graph,
    "role_split_detected": rel_role_split_detected,
    "fwd_vs_reference": rel_fwd_vs_reference,
    "op_packing": rel_op_packing,
    "op_vs_reference": rel_op_vs_reference,
}

SENSITIVITY_RELATIONS = ("logprob_contract", "batch_composition", "decode_vs_multirow")


def run_relation(name: str, backend: Backend, cfg: Config) -> Outcome:
    out = Outcome(name, backend.name)
    rec = Rec(out, backend)
    ident = getattr(backend, "identity", None)
    if callable(ident):
        out.witness["kernel_identity"] = ident()
    t0 = time.time()
    try:
        RELATIONS[name](rec, cfg)
        torch.cuda.synchronize()
    except Unsupported as e:  # checks recorded before the unsupported call still decide the status
        out.skipped.append(f"remaining checks: {e}")
    except Exception as e:
        out.status = "error"
        out.notes.append(f"{type(e).__name__}: {str(e)[:400]}")
        out.notes.append(traceback.format_exc()[-1500:])
    out.seconds = round(time.time() - t0, 2)
    if out.status == "pass":
        vacuous = [n for n in out.notes if n.startswith("vacuous")]
        if out.failures or vacuous:
            out.status = "fail"
        elif not out.diffs:
            out.status = "n/a" if out.skipped else "fail"
            if not out.skipped:
                out.notes.append("no checks ran")
    return out


def bits(t):
    from isoexec.ops.attention.mla_sm90.tests._harness import bits as _b

    return _b(t)
