"""Determinism relations for ``attention.core.gqa`` on SM100, compared as raw bits.

Every equality claim below is bit equality on an integer view -- never a tolerance. The one
numerical relation (``fwd_vs_reference``) compares against the float64 oracle in ``reference.py``.

    python -m isoexec.ops.attention.gqa_sm100.tests.test_relations_gpu

It is also a pytest module: ``test_relations`` runs every relation and skips off a B200, so the
default CPU lane collects it and the GPU lane witnesses it.

Exit 0 every relation passed, 1 a relation differs, 2 INCOMPLETE (a claim could not be made here).
``--only NAME`` runs one relation; ``--json PATH`` writes the outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback

import pytest
import torch

from isoexec.core.arch import live_device_arch
from isoexec.ops.attention.gqa_sm100 import ops
from isoexec.ops.attention.gqa_sm100.tests._harness import (
    DEV,
    GEOMETRIES,
    GEOMETRIES_D256,
    accuracy,
    bit_equal,
    cu,
    draw,
    first_diff,
    make_batch,
    page_kv,
    reference,
    run_packed,
    run_paged,
)


#: Read the plan's OVERFLOW_FLAG on every call this module makes. The host's capacity bound is chosen
#: so an overflow is impossible, but "impossible" was wrong twice -- once for a granularity override
#: and once for `node_size_at`'s power-of-two remainder -- and an overflow is SILENT otherwise: part of
#: the output is simply never written, and whether that shows up depends on what the caching allocator
#: left in the buffer. Setting it here is what makes the battery able to see the class of bug at all.
os.environ["ISOEXEC_GQA_SM100_CHECK"] = "1"

#: Output error normalized by max|V|; lse error absolute. bf16 output carries about 2^-8 relative, and
#: a 8192-key row's fp32 accumulation adds to that; these are the measured margins.
TOL_OUT = 0.02
TOL_LSE = 0.02


@dataclasses.dataclass
class Outcome:
    name: str
    status: str = "pass"          # pass | fail | skip | error
    checks: int = 0
    detail: str = ""
    seconds: float = 0.0


class Rec:
    """Collects the checks of one relation."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures: list[str] = []
        self.notes: list[str] = []

    def equal(self, name: str, a, b) -> None:
        self.checks += 1
        if not bit_equal(a, b):
            self.failures.append(f"{name}: {first_diff(a, b)}")

    def note(self, text: str) -> None:
        self.notes.append(text)


# --- helpers ------------------------------------------------------------------------------------


def _seg(head_dim: int, ctx: int) -> int:
    return ops.seg_for(head_dim, 8192 if ctx <= 8192 else 32768)


def _rows(batch, i: int, out, lse):
    """Request ``i``'s rows of a packed result."""
    lo = sum(batch.q_lens[:i])
    hi = lo + batch.q_lens[i]
    return out[lo:hi], lse[lo:hi]


def _paged_call(q, kv_lens, kc, vc, table, geometry, seg, max_q, **kw):
    return ops.gqa_sm100_fwd(
        q, kc, vc, cu([q.shape[0]] if isinstance(kv_lens, int) else [len(q)]),
        torch.tensor(kv_lens, device=DEV, dtype=torch.int32),
        softmax_scale=ops.default_scale(geometry.head_dim), seg_tokens=seg,
        max_seqlen_q=max_q, max_seqlen_k=max(kv_lens), block_table=table, **kw)


def _one_request(q, k, v, kv_len, geometry, seg, *, q_rows=None, page=256, **kw):
    """One request, paged, with ``q_rows`` query tokens right-aligned at ``kv_len``."""
    rows = q.shape[0] if q_rows is None else q_rows
    kc, vc, table = page_kv(k[:kv_len], v[:kv_len], [kv_len], page)
    return ops.gqa_sm100_fwd(
        q[-rows:].contiguous(), kc, vc, cu([rows]),
        torch.tensor([kv_len], device=DEV, dtype=torch.int32),
        softmax_scale=ops.default_scale(geometry.head_dim), seg_tokens=seg,
        max_seqlen_q=rows, max_seqlen_k=kv_len, block_table=table, **kw)


# --- relations ----------------------------------------------------------------------------------


def rel_fwd_vs_reference(rec: Rec) -> None:
    """Accuracy against the float64 oracle, on both addressing modes and four populations."""
    for g in GEOMETRIES:
        for q_lens, kv_lens in (([512], [512]), ([1], [4096]), ([2239, 17, 384], [2239, 17, 384])):
            seg = _seg(g.head_dim, max(kv_lens))
            for pop in ("normal", "large", "small", "signed_zero"):
                b = make_batch(q_lens, kv_lens, g, 11, population=pop)
                ref_o, ref_l = reference(b)
                for mode, call in (("packed", run_packed), ("paged", run_paged)):
                    out, lse = call(b, seg)
                    rec.checks += 1
                    if not bool(out.isfinite().all()):
                        rec.failures.append(f"{g.tag}/{q_lens}/{pop}/{mode}: non-finite output")
                        continue
                    eo, el = accuracy(out, lse, ref_o, ref_l, b.v)
                    if eo > TOL_OUT or el > TOL_LSE:
                        rec.failures.append(
                            f"{g.tag}/{q_lens}/{pop}/{mode}: out {eo:.5f} lse {el:.5f}")
    rec.note(f"tolerances: out <= {TOL_OUT}, lse <= {TOL_LSE} (normalized / absolute)")


def rel_packed_vs_paged(rec: Rec) -> None:
    """The twin class: the trainer's packed call and the engine's paged call, bit for bit."""
    for g in GEOMETRIES:
        for q_lens, kv_lens in (([1024], [1024]), ([1], [2048]), ([2239, 17, 384], [2239, 17, 384]),
                                ([1] * 8 + [977], [1] * 8 + [977])):
            seg = _seg(g.head_dim, max(kv_lens))
            b = make_batch(q_lens, kv_lens, g, 21)
            po, pl = run_packed(b, seg)
            go, gl = run_paged(b, seg)
            rec.equal(f"{g.tag}/{q_lens}/out", po, go)
            rec.equal(f"{g.tag}/{q_lens}/lse", pl, gl)


def rel_page_layout(rec: Rec) -> None:
    """Page size, page permutation, spare blocks and a poisoned cache tail change no bit."""
    g = GEOMETRIES[0]
    for kv_len in (2048, 2049, 4096 + 7):
        seg = _seg(g.head_dim, 8192)
        b = make_batch([64], [kv_len], g, 31)
        base = run_paged(b, seg, page=256)
        for label, kw in (
            ("page=128", {"page": 128}),
            ("page=512", {"page": 512}),
            ("permuted", {"page": 256, "permute": True}),
            ("spare blocks", {"page": 256, "pad_blocks": 13}),
            ("poisoned tail", {"page": 256, "fill": float("nan"), "permute": True, "pad_blocks": 5}),
        ):
            got = run_paged(b, seg, **kw)
            rec.equal(f"kv={kv_len}/{label}/out", base[0], got[0])
            rec.equal(f"kv={kv_len}/{label}/lse", base[1], got[1])


def rel_execution_plans(rec: Rec) -> None:
    """A row's tree may be folded by one worker or published and merged; the bits are the same.

    The run is what decides how much of a row's canonical tree ONE worker folds in place, and where
    the pending nodes of that fold live -- two TMEM output tiles at one wave and one at two, then the
    64 KiB shared node slot, then L2-resident global slots. Every combination below evaluates the
    same tree in the same order and must return the same bits; ``run=16`` at ``k=8192`` is the whole
    tree in one worker with four backing slots live, and ``waves=2`` runs the same fold with every
    backing slot in global memory.
    """
    g = GEOMETRIES[0]
    #: `([2048], [8192])` is the shape whose uniform tiles sit at n = 13, 14, 15, 16 -- remainders
    #: with two bits set at run 8 -- and `([300], [7000])` puts `seqused_k` off a SEG multiple so the
    #: last segment is partial and the straddling tiles are not tile-aligned.
    for q_lens, kv_lens in (([256], [2048]), ([1], [4096]), ([1, 1, 512], [700, 2048, 512]),
                            ([1], [8192]), ([192], [8192]), ([2048], [8192]), ([300], [7000])):
        seg = _seg(g.head_dim, max(kv_lens))
        b = make_batch(q_lens, kv_lens, g, 41)
        base = run_paged(b, seg, granularity=ops.GRAN_AUTO)
        for label, kw in (
            ("single segments", {"granularity": ops.GRAN_SINGLE}),
            ("aligned pairs", {"granularity": ops.GRAN_PAIRS}),
            ("waves=1", {"waves": 1}),
            ("waves=2", {"waves": 2}),
            ("single + waves=1", {"granularity": ops.GRAN_SINGLE, "waves": 1}),
            ("pairs + waves=2", {"granularity": ops.GRAN_PAIRS, "waves": 2}),
            ("run=1 by cap", {"max_run": 1}),
            ("run=2 by cap", {"max_run": 2}),
            ("run=4 by cap", {"max_run": 4}),
            #: `n mod 8` with two bits set is what `node_size_at` decomposes into three nodes where
            #: `ceil(n / 8)` is two, a plan-capacity bound that was once too small. At k=8192 and
            #: SEG 512 the tiles sit at n = 13..16, so this plan is the one that overflowed.
            ("run=8 by cap, one global slot", {"max_run": 8}),
            ("run=8 at two waves, the overflowing plan", {"waves": 2, "max_run": 8}),
            ("run=16 by cap, two global slots", {"max_run": 16}),
            ("waves=1 + run=16", {"waves": 1, "max_run": 16}),
            ("waves=2 + run=4, global backing", {"waves": 2, "max_run": 4}),
            ("waves=2 + run=8, global backing", {"waves": 2, "max_run": 8}),
            ("waves=2 + run=16, four global slots", {"waves": 2, "max_run": 16}),
            ("waves=2 + single segments", {"waves": 2, "granularity": ops.GRAN_SINGLE}),
            ("waves=2 + run=1", {"waves": 2, "max_run": 1}),
        ):
            got = run_paged(b, seg, **kw)
            rec.equal(f"{q_lens}/{label}/out", base[0], got[0])
            rec.equal(f"{q_lens}/{label}/lse", base[1], got[1])


def rel_batch_composition(rec: Rec) -> None:
    """A request's rows do not depend on its companions or on its slot in the batch."""
    g = GEOMETRIES[0]
    seg = _seg(g.head_dim, 8192)
    lengths = (777, 1, 2048, 33)
    alone = []
    for n in lengths:
        b = make_batch([n], [n], g, 51)
        alone.append((b, run_paged(b, seg)))
    for order in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)):
        q_lens = [lengths[i] for i in order]
        qs, ks, vs = [], [], []
        for i in order:
            qs.append(alone[i][0].q)
            ks.append(alone[i][0].k)
            vs.append(alone[i][0].v)
        q = torch.cat(qs)
        k = torch.cat(ks)
        v = torch.cat(vs)
        kc, vc, table = page_kv(k, v, q_lens, 256)
        out, lse = ops.gqa_sm100_fwd(
            q, kc, vc, cu(q_lens), torch.tensor(q_lens, device=DEV, dtype=torch.int32),
            softmax_scale=ops.default_scale(g.head_dim), seg_tokens=seg,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(q_lens), block_table=table)
        off = 0
        for slot, i in enumerate(order):
            n = lengths[i]
            rec.equal(f"order={order}/slot={slot}/out", alone[i][1][0], out[off:off + n])
            rec.equal(f"order={order}/slot={slot}/lse", alone[i][1][1], lse[off:off + n])
            off += n


def rel_chunked_prefill(rec: Rec) -> None:
    """A token's output is the same in full prefill, at every chunk boundary, and in decode."""
    g = GEOMETRIES[0]
    total = 2048 + 129
    seg = _seg(g.head_dim, 8192)
    q = draw(total, g.h_q, g.head_dim, 61)
    k = draw(total, g.h_kv, g.head_dim, 62)
    v = draw(total, g.h_kv, g.head_dim, 63)
    full_o, full_l = _one_request(q, k, v, total, g, seg)
    for bounds in ([0, 512, 1024, 2048, total], [0, 777, 1000, total], [0, total - 1, total]):
        for lo, hi in zip(bounds, bounds[1:]):
            co, cl = ops.gqa_sm100_fwd(
                q[lo:hi].contiguous(), *page_kv(k[:hi], v[:hi], [hi], 256)[:2],
                cu([hi - lo]), torch.tensor([hi], device=DEV, dtype=torch.int32),
                softmax_scale=ops.default_scale(g.head_dim), seg_tokens=seg,
                max_seqlen_q=hi - lo, max_seqlen_k=hi,
                block_table=page_kv(k[:hi], v[:hi], [hi], 256)[2])
            rec.equal(f"chunk[{lo},{hi})/out", full_o[lo:hi], co)
            rec.equal(f"chunk[{lo},{hi})/lse", full_l[lo:hi], cl)
    for position in (0, 1, 511, 512, 2047, 2048, total - 1):
        do, dl = _one_request(q[: position + 1], k, v, position + 1, g, seg, q_rows=1)
        rec.equal(f"decode@{position}/out", full_o[position : position + 1], do)
        rec.equal(f"decode@{position}/lse", full_l[position : position + 1], dl)


def rel_later_turn_prompt(rec: Rec) -> None:
    """A token generated in one call and re-read as prompt in the next gets the same bits."""
    g = GEOMETRIES[0]
    seg = _seg(g.head_dim, 8192)
    total = 1500
    q = draw(total, g.h_q, g.head_dim, 71)
    k = draw(total, g.h_kv, g.head_dim, 72)
    v = draw(total, g.h_kv, g.head_dim, 73)
    for cut in (1024, 1200, 1499):
        generated = _one_request(q[: cut + 1], k, v, cut + 1, g, seg, q_rows=1)
        # the same token, now inside a prompt of the next turn
        prompt = _one_request(q[:total], k, v, total, g, seg, q_rows=total)
        rec.equal(f"cut={cut}/out", generated[0], prompt[0][cut : cut + 1])
        rec.equal(f"cut={cut}/lse", generated[1], prompt[1][cut : cut + 1])


def rel_repeat(rec: Rec) -> None:
    """Fifty launches of the same call, bit for bit."""
    g = GEOMETRIES[1]
    seg = _seg(g.head_dim, 8192)
    b = make_batch([1, 1, 700, 33], [2048, 4096, 700, 33], g, 81)
    base = run_paged(b, seg)
    for i in range(50):
        got = run_paged(b, seg)
        rec.equal(f"repeat {i}/out", base[0], got[0])
        rec.equal(f"repeat {i}/lse", base[1], got[1])


def rel_cuda_graph(rec: Rec) -> None:
    """A captured graph replays the eager bits."""
    g = GEOMETRIES[0]
    seg = _seg(g.head_dim, 8192)
    b = make_batch([1, 1, 1, 1], [2048, 1024, 4096, 17], g, 91)
    kc, vc, table = page_kv(b.k, b.v, b.kv_lens, 256)
    # every device tensor is built before the capture: a host-to-device copy inside one is refused,
    # and the point of the relation is the kernel's bits, not the harness's allocations.
    cu_q, seqused = b.cu_q, b.seqused
    args = {"softmax_scale": ops.default_scale(g.head_dim), "seg_tokens": seg,
                "max_seqlen_q": 1, "max_seqlen_k": max(b.kv_lens), "block_table": table}
    eager = ops.gqa_sm100_fwd(b.q, kc, vc, cu_q, seqused, **args)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            ops.gqa_sm100_fwd(b.q, kc, vc, cu_q, seqused, **args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.gqa_sm100_fwd(b.q, kc, vc, cu_q, seqused, **args)
    for i in range(5):
        graph.replay()
        torch.cuda.synchronize()
        rec.equal(f"replay {i}/out", eager[0], captured[0])
        rec.equal(f"replay {i}/lse", eager[1], captured[1])


def rel_tails(rec: Rec) -> None:
    """Lengths around every boundary the program has: KV tile, segment, page, query tile."""
    g = GEOMETRIES[0]
    seg = _seg(g.head_dim, 8192)
    boundaries = []
    for base in (seg, 2 * seg, 256, 128, 16):
        boundaries += [base - 1, base, base + 1]
    boundaries = sorted({n for n in boundaries if 1 <= n <= 4096})
    for n in boundaries:
        b = make_batch([n], [n], g, 101)
        ref_o, ref_l = reference(b)
        out, lse = run_paged(b, seg)
        rec.checks += 1
        eo, el = accuracy(out, lse, ref_o, ref_l, b.v)
        if eo > TOL_OUT or el > TOL_LSE or not bool(out.isfinite().all()):
            rec.failures.append(f"n={n}: out {eo:.5f} lse {el:.5f}")
        rec.equal(f"n={n}/packed==paged/out", run_packed(b, seg)[0], out)


def rel_domain_refusal(rec: Rec) -> None:
    """Outside its declared domain the call REFUSES; it does not quietly compute something else."""
    g = GEOMETRIES[0]
    seg = _seg(g.head_dim, 8192)
    b = make_batch([64], [512], g, 121)
    # 128 and 256 are the compiled head dims (ops.HEAD_DIMS). 192 is a real head dim this package
    # does not serve, and the refusal says so rather than reading past the tile. It replaces the
    # "head dim 256 is refused" check, which stopped being true when the two-CTA leaf landed.
    q192 = torch.zeros(64, 8, 192, device=DEV, dtype=torch.bfloat16)
    k192 = torch.zeros(512, 1, 192, device=DEV, dtype=torch.bfloat16)
    # Head dim 256 runs a CTA pair and has no wave count, so the D128 wave override is refused
    # there rather than silently ignored.
    q256 = torch.zeros(64, 2, 256, device=DEV, dtype=torch.bfloat16)
    k256 = torch.zeros(512, 1, 256, device=DEV, dtype=torch.bfloat16)
    for name, call in (
        ("head dim 192", lambda: ops.gqa_sm100_fwd(
            q192, k192, k192, b.cu_q, b.seqused, softmax_scale=0.0625, seg_tokens=seg,
            max_seqlen_q=64, max_seqlen_k=512, cu_seqlens_k=b.cu_k)),
        ("a wave override at head dim 256", lambda: ops.gqa_sm100_fwd(
            q256, k256, k256, b.cu_q, b.seqused, softmax_scale=0.0625, seg_tokens=seg,
            max_seqlen_q=64, max_seqlen_k=512, cu_seqlens_k=b.cu_k, waves=1)),
        ("seg not a multiple of the KV tile", lambda: run_paged(b, 300)),
        ("page not a multiple of the KV tile", lambda: run_paged(b, seg, page=64)),
    ):
        rec.checks += 1
        try:
            call()
        except Exception as exc:
            # The refusal IS the passing outcome here, and its text is what a caller reads, so it is
            # recorded rather than swallowed.
            rec.note(f"{name}: refused -- {str(exc).splitlines()[0][:120]}")
            continue
        rec.failures.append(f"{name}: accepted, but it is outside the declared domain")


def rel_head_dims(rec: Rec) -> None:
    """Head dim 256 runs the two-CTA leaf: the oracle, both addressing modes and every plan.

    This is the claim the battery used to report INCOMPLETE. The D256 leaf is a different KERNEL --
    a cluster of two CTAs, `cta_group::2` UMMA, 128 packed query rows and all 256 output features
    per CTA -- running the SAME numerical program, so what it owes is exactly what the D128 leaf
    owes: the float64 oracle within tolerance, packed == paged bit for bit, and every admissible
    plan and page layout identical.
    """
    for g in GEOMETRIES_D256:
        for q_lens, kv_lens in (([512], [512]), ([1], [4096]),
                                ([2239, 17, 384], [2239, 17, 384])):
            seg = _seg(g.head_dim, max(kv_lens))
            b = make_batch(q_lens, kv_lens, g, 131)
            ref_o, ref_l = reference(b)
            base = run_paged(b, seg)
            rec.checks += 1
            if not bool(base[0].isfinite().all()):
                rec.failures.append(f"{g.tag}/{q_lens}: non-finite output")
                continue
            eo, el = accuracy(base[0], base[1], ref_o, ref_l, b.v)
            if eo > TOL_OUT or el > TOL_LSE:
                rec.failures.append(f"{g.tag}/{q_lens}: out {eo:.5f} lse {el:.5f}")
            packed = run_packed(b, seg)
            rec.equal(f"{g.tag}/{q_lens}/packed==paged/out", packed[0], base[0])
            rec.equal(f"{g.tag}/{q_lens}/packed==paged/lse", packed[1], base[1])
            for label, kw in (
                ("single segments", {"granularity": ops.GRAN_SINGLE}),
                ("aligned pairs", {"granularity": ops.GRAN_PAIRS}),
                ("run=1 by cap", {"max_run": 1}),
                #: The D256 FOLD, forced. A run folds its segments in one CTA pair
                #: through global backing slots; every run must equal run 1 bit for bit.
                ("run=2", {"max_run": 2}),
                ("run=4", {"max_run": 4}),
                ("run=8", {"max_run": 8}),
                ("page=128", {"page": 128}),
                ("poisoned tail", {"page": 512, "fill": float("nan"), "permute": True,
                                   "pad_blocks": 3}),
            ):
                got = run_paged(b, seg, **kw)
                rec.equal(f"{g.tag}/{q_lens}/{label}/out", base[0], got[0])
                rec.equal(f"{g.tag}/{q_lens}/{label}/lse", base[1], got[1])
    rec.note("head dim 256 on the two-CTA leaf, folding runs of 1/2/4/8 segments through global "
             "backing slots, every run bit-identical to run 1")


class Incomplete(Exception):
    """The claim cannot be made here. Reported as INCOMPLETE (exit 2), never as a pass."""


RELATIONS = {
    "fwd_vs_reference": rel_fwd_vs_reference,
    "packed_vs_paged": rel_packed_vs_paged,
    "page_layout": rel_page_layout,
    "execution_plans": rel_execution_plans,
    "batch_composition": rel_batch_composition,
    "chunked_prefill": rel_chunked_prefill,
    "later_turn_prompt": rel_later_turn_prompt,
    "repeat": rel_repeat,
    "cuda_graph": rel_cuda_graph,
    "tails": rel_tails,
    "domain_refusal": rel_domain_refusal,
    "head_dims": rel_head_dims,
}


def registry_covers() -> str | None:
    """The battery qualifies the recipe this package declares, and nothing else."""
    from isoexec.ops.attention.gqa_sm100 import spec

    rows = [i for i in spec.IMPLS if i.op == "attention.core.gqa"]
    if {i.impl_id for i in rows} != {spec.TRAINER_IMPL, spec.PAGED_IMPL}:
        return "gqa_sm100/spec.py does not declare the trainer and paged members of the recipe"
    if any(i.supported_archs != frozenset({"sm100"}) or i.version != spec.RECIPE_VERSION for i in rows):
        return "the SM100 GQA members must be sm100-only at the recipe version"
    if {i.twin_class for i in rows} != {spec.TWIN_CLASS}:
        return "the trainer and paged members must share the package's twin class"
    return None


@pytest.mark.skipif(live_device_arch() != "sm100", reason="needs a live B200 (sm100)")
def test_relations():
    """Every relation, as one pytest case so the GPU lane witnesses the whole gate."""
    covered = registry_covers()
    assert covered is None, covered
    outcomes = [run(n) for n in RELATIONS]
    bad = [o for o in outcomes if o.status in ("fail", "error")]
    assert not bad, "; ".join(f"{o.name}: {o.detail[:300]}" for o in bad)
    incomplete = [o for o in outcomes if o.status == "skip"]
    if incomplete:
        pytest.skip("INCOMPLETE: " + "; ".join(f"{o.name}: {o.detail}" for o in incomplete))


def run(name: str) -> Outcome:
    rec = Rec()
    started = time.time()
    try:
        RELATIONS[name](rec)
    except Incomplete as exc:
        return Outcome(name, "skip", rec.checks, str(exc), time.time() - started)
    except Exception:
        return Outcome(name, "error", rec.checks, traceback.format_exc(limit=6),
                       time.time() - started)
    if rec.failures:
        return Outcome(name, "fail", rec.checks, "; ".join(rec.failures[:4]), time.time() - started)
    return Outcome(name, "pass", rec.checks, "; ".join(rec.notes), time.time() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    arch = live_device_arch()
    if arch != "sm100":
        print(f"INCOMPLETE: live arch is {arch}, this battery proves the sm100 program")
        return 2
    covered = registry_covers()
    if covered is not None:
        print(f"INCOMPLETE: {covered}")
        return 2

    names = args.only or list(RELATIONS)
    outcomes = [run(n) for n in names]
    for o in outcomes:
        print(f"{o.status.upper():8s} {o.name:22s} {o.checks:5d} checks {o.seconds:6.1f}s  "
              f"{o.detail[:200]}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([dataclasses.asdict(o) for o in outcomes], fh, indent=2)
    if any(o.status in ("fail", "error") for o in outcomes):
        return 1
    if any(o.status == "skip" for o in outcomes):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
