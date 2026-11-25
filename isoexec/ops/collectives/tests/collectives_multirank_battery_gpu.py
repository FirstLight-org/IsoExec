#!/usr/bin/env python
"""Real multi-rank GPU battery for every ``collectives`` impl in ``spec.py``, both directions.

Each leg spawns ``--tp`` NCCL ranks (one GPU each) and checks, bitwise:

Forward (the contract: identical bits at every TP/CP size, batch, transport, rank arrival order, graph/eager)
  * ``row_parallel`` ``pik_tree`` / ``pik_tree_rs`` (and the partial + finish split the shared expert uses):
    the TP=``tp`` result equals the single-GPU TP=1 statement of the same fixed-leaf tree on the same global
    operands, so TP2 and TP4 legs agree with each other through it. Shapes are the shipped models'
    row-parallel projections; every M is a row prefix of one global batch, so the TP=1 reference itself is
    checked for batch invariance, and each M is repeated under different rank arrival orders.
  * ``tp_copy``: identity is the input's bytes; ``seq_all_gather`` / ``sp_gather`` / ``sp_scatter`` backward
    gather and ``all_gather`` / KV gather equal ``dist.all_gather_into_tensor`` bytes, over the complete bf16
    encoding space (signed zero, subnormals, infinities, NaN payloads) and fp32 edge values;
    ``cp_gather`` (cp=2) equals the CP=1 global stream.
  * a CUDA graph of the forwards replays to the eager bits.

Backward (deterministic per layout; TP invariance of gradients is not part of the contract)
  * ``tp_copy`` / ``seq_all_gather`` / ``all_gather`` / KV gather: the fixed tree over rank-ordered
    contributors, recomputed locally with ``pik.plan.combine_order`` in fp32 and rounded once.
  * ``sp_gather``: this rank's chunk, unchanged. ``sp_scatter``: the gathered bytes.
  * ``cp_gather``: the two-contributor sum. ``row_parallel`` / ``pik_tree_rs``: repeat-stable.

Legs: pik's P2P symmetric-memory transport (which must engage with zero fallbacks; skipped on an NCCL-only
arch such as sm100, see ``pik_tp_invariant.NCCL_ONLY_ARCHS``), the NCCL transport, and NCCL algorithm /
channel overrides; each over fp32 and bf16 leaves, and leaves in {8, tp} so the bf16
leaf wire (one leaf per rank) is covered too.

    CUDA_VISIBLE_DEVICES=<tp GPUs> python isoexec/ops/collectives/tests/collectives_multirank_battery_gpu.py --tp 2
    CUDA_VISIBLE_DEVICES=<tp GPUs> python isoexec/ops/collectives/tests/collectives_multirank_battery_gpu.py --tp 4
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 4))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

# (K, N) of row-parallel projections in the shipped models: Qwen3.5-35B-A3B o_proj and shared-expert fc2,
# Qwen3.5-9B down_proj, Qwen1.5-MoE-A2.7B shared-expert fc2.
SHAPES = {
    "35b_o_proj": (4096, 2048),
    "35b_shared_fc2": (512, 2048),
    "9b_down_proj": (12288, 4096),
    "a27b_shared_fc2": (5632, 2048),
}
# Decode- through prefill-sized row counts; every one is a multiple of 8 (the reduce-scatter slices rows).
M_POINTS = (8, 256, 2048, 4096)
M_PREFILL = 16384  # added for the narrower-N shapes with --prefill (needs ~8 GiB free per GPU)
JITTERS = (
    lambda world, rank: 0.0,
    lambda world, rank: 0.02 * rank,
    lambda world, rank: 0.02 * (world - 1 - rank),
)
LEGS = (
    ("p2p", {}, False),
    ("nccl", {}, True),
    ("nccl_ring", {"NCCL_ALGO": "allreduce:ring"}, True),
    ("nccl_tree", {"NCCL_ALGO": "allreduce:tree"}, True),
    ("nccl_1ch", {"NCCL_MIN_NCHANNELS": "1", "NCCL_MAX_NCHANNELS": "1"}, True),
    ("nccl_4ch", {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}, True),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _adversarial(shape, seed: int, dev, dtype) -> torch.Tensor:
    """Wide exponent spread so any change of summation order shows. Same seed -> same bits on every rank."""
    g = torch.Generator(device=dev).manual_seed(seed)
    a = torch.randn(*shape, generator=g, device=dev) * torch.exp(torch.randn(*shape, generator=g, device=dev) * 4)
    return a.to(dtype)


def _byte_cases(rank: int, dev) -> list[torch.Tensor]:
    """Payloads that are only preserved by byte movement: every bf16 encoding and fp32 edge values."""
    enc = ((torch.arange(65536, device=dev, dtype=torch.int64) + 733 * rank) % 65536).to(torch.int16)
    edge = torch.tensor(
        [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC12345,
         0x7F800001, 0x7F7FFFFF], device=dev, dtype=torch.int64,
    )
    cases = [enc.view(torch.bfloat16).reshape(512, 128), edge.roll(rank).to(torch.int32).view(torch.float32).reshape(3, 4)]
    for dtype in (torch.bfloat16, torch.float32):
        for i, shape in enumerate(((1, 3), (17, 31), (256, 2048), (4096, 2048))):
            cases.append(_adversarial(shape, 7100 + 31 * rank + i, dev, dtype))
    return cases


def _fold(parts: list[torch.Tensor], out_dtype) -> torch.Tensor:
    """The fixed balanced tree over rank-ordered contributors, in fp32, rounded once."""
    from pik.plan import combine_order

    slots = [p.float() for p in parts]
    for dst, lhs, rhs in combine_order(len(slots)):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0].to(out_dtype)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world: int, leaf_dtype: str, leaves: int, port: int, disable_p2p: bool, graph: bool,
            prefill: bool, quick: bool, out):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    failures: list[str] = []
    checks = [0]

    def check(name: str, cond: bool) -> None:
        checks[0] += 1
        if not cond:
            failures.append(name)

    try:
        from isoexec.ops.collectives.pik_bootstrap import ensure_pik

        ensure_pik()
        import pik.allreduce as PA

        import isoexec.ops.collectives.pik.allreduce as PA_dotted

        check("one pik.allreduce module", PA is PA_dotted)
        if disable_p2p:
            PA.p2p_available = lambda *a, **k: False  # what the trainer-side NCCL transport selection does

        from pik.linear import row_parallel_linear
        from pik.plan import ReductionPlan

        from isoexec.ops.collectives import tp_bytes
        from isoexec.ops.collectives.cp_gather import build_layout, make_cp_attention_core, make_cp_gather
        from isoexec.ops.collectives.tp_tree import make_row_parallel, make_row_parallel_partial, make_row_parallel_rs
        from isoexec.schema import Axis, Placement

        group = dist.group.WORLD
        # cp_gather's CP groups are rank pairs. Built first, while their communicators still fit next to
        # GPUs that other jobs share; at world 2 the pair is the world group itself.
        pairs = [dist.new_group([p, p + 1]) for p in range(0, world, 2)] if world > 2 else [group]
        one = Axis(1, 0, None)
        placement = Placement(
            tp=Axis(world, rank, group), ep=one, dp=one, pp=one, etp=Axis(world, rank, group), cp=one,
            side="engine", leaves=leaves, leaf_dtype=leaf_dtype, sp=False,
        )
        plan = ReductionPlan(num_leaves=leaves, leaf_dtype=torch.bfloat16 if leaf_dtype == "bf16" else torch.float32)
        row_parallel = make_row_parallel(placement)
        row_parallel_rs = make_row_parallel_rs(placement)
        partial = make_row_parallel_partial(placement)
        tp_copy = tp_bytes.make_tp_copy(placement)
        seq_all_gather = tp_bytes.make_seq_all_gather(placement)
        sp_gather = tp_bytes.make_seq_all_gather_replicated(placement)
        sp_scatter = tp_bytes.make_seq_scatter(placement)
        all_gather = tp_bytes.make_all_gather(placement)
        kv_heads = max(1, world // 2)
        kv_gather = tp_bytes.make_kv_gather(placement, kv_heads)
        step = world // kv_heads
        kv_lo, kv_hi = (rank // step) * step, (rank // step + 1) * step

        def nccl_gather(x: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            o = x.new_empty((world * x.shape[0], *x.shape[1:]))
            dist.all_gather_into_tensor(o, x, group=group)
            return o

        # ---- row_parallel: TP=tp result == TP=1 statement, at every batch prefix, under arrival jitter ----
        for name, (k_full, n) in list(SHAPES.items())[: 1 if quick else None]:
            k_local = k_full // world
            m_points = M_POINTS[: 2 if quick else None] + ((M_PREFILL,) if prefill and n <= 2048 else ())
            m_max = max(m_points)
            x_full = _adversarial((m_max, k_full), 11 + k_full, dev, torch.bfloat16)
            w_full = _adversarial((n, k_full), 13 + k_full + n, dev, torch.bfloat16)
            w = w_full[:, rank * k_local : (rank + 1) * k_local].contiguous()
            x_mine = x_full[:, rank * k_local : (rank + 1) * k_local].contiguous()
            ref_max = row_parallel_linear(x_full, w_full, plan=plan, tp_size=1, k_full=k_full, out_dtype=torch.bfloat16)
            for m in m_points:
                ref = row_parallel_linear(x_full[:m], w_full, plan=plan, tp_size=1, k_full=k_full, out_dtype=torch.bfloat16)
                tag = f"{name}_m{m}"
                check(f"tp1 batch-invariant {tag}", _same(ref, ref_max[:m]))
                chunk = m // world
                x = x_mine[:m]
                first = None
                for j, jitter in enumerate(JITTERS):
                    dist.barrier()
                    time.sleep(jitter(world, rank))
                    y = row_parallel(x, w)
                    y_rs = row_parallel_rs(x, w)
                    y_split = partial.finish(partial(x, w))
                    check(f"pik_tree == tp1 {tag} j{j}", _same(y, ref))
                    check(f"pik_tree_rs == tp1 slice {tag} j{j}", _same(y_rs, ref[rank * chunk : (rank + 1) * chunk]))
                    check(f"partial+finish == tp1 {tag} j{j}", _same(y_split, ref))
                    first = first or (y, y_rs)
                    check(f"repeat {tag} j{j}", _same(y, first[0]) and _same(y_rs, first[1]))
            del x_full, w_full, ref_max, x_mine, w, ref, x, y, y_rs, y_split, first
            torch.cuda.empty_cache()

        # row_parallel backward (local GEMMs; pik_tree_rs gathers dY bytes first): repeat-stable.
        k_full, n = SHAPES["35b_o_proj"]
        k_local, m = k_full // world, 256
        w0 = _adversarial((n, k_local), 501 + rank, dev, torch.bfloat16)
        x0 = _adversarial((m, k_local), 502 + rank, dev, torch.bfloat16)
        # pik_tree_rs's dY is this rank's row slice; its backward gathers every rank's slice in rank order.
        dy_rs = [_adversarial((m // world, n), 503 + r, dev, torch.bfloat16) for r in range(world)]
        dy_ar = _adversarial((m, n), 504 + rank, dev, torch.bfloat16)
        for fn, dy, full, label in ((row_parallel, dy_ar, dy_ar, "pik_tree"),
                                    (row_parallel_rs, dy_rs[rank], torch.cat(dy_rs), "pik_tree_rs")):
            grads = []
            for _ in range(2):
                x = x0.clone().requires_grad_()
                ww = w0.clone().requires_grad_()
                fn(x, ww).backward(dy)
                grads.append((x.grad, ww.grad))
            check(f"{label} bwd repeat-stable", _same(grads[0][0], grads[1][0]) and _same(grads[0][1], grads[1][1]))
            check(f"{label} bwd dX == dY @ W", _same(grads[0][0], torch.ops.aten.mm(full, w0)))
            check(f"{label} bwd dW == dY^T @ X", _same(grads[0][1], torch.ops.aten.mm(full.t(), x0)))

        # ---- byte-moving forwards and their adjoints ----
        for i, x in enumerate(_byte_cases(rank, dev)):
            tag = f"case{i}_{tuple(x.shape)}_{x.dtype}"
            want = nccl_gather(x)
            for label, fn in (("seq_all_gather", seq_all_gather), ("sp_gather", sp_gather)):
                check(f"{label} fwd bytes {tag}", _same(fn(x), want))
            check(f"tp_copy fwd bytes {tag}", _same(tp_copy(x), x))
            if x.shape[0] % world == 0:
                chunk = x.shape[0] // world
                check(f"sp_scatter fwd bytes {tag}", _same(sp_scatter(x), x[rank * chunk : (rank + 1) * chunk]))
            if x.dim() == 2:
                cols = want.view(world, *x.shape).permute(1, 0, 2)
                check(f"all_gather fwd bytes {tag}", _same(all_gather(x), cols.reshape(x.shape[0], -1)))
                check(f"kv_gather fwd bytes {tag}", _same(kv_gather(x), cols[:, kv_lo:kv_hi].reshape(x.shape[0], -1)))

        for dtype in (torch.float32, torch.bfloat16):
            t, h = 512, 1024
            s = 9000 + (dtype == torch.bfloat16)
            gs = [_adversarial((t, h), s + r, dev, dtype) for r in range(world)]
            gs_full = [_adversarial((world * t, h), s + 100 + r, dev, dtype) for r in range(world)]
            gs_cols = [_adversarial((t, world * h), s + 200 + r, dev, dtype) for r in range(world)]
            sl = slice(rank * t, (rank + 1) * t)
            x = torch.zeros(t, h, device=dev, dtype=dtype, requires_grad=True)

            def grad(fn, g, inp=None):
                inp = (x if inp is None else inp).detach().clone().requires_grad_()
                fn(inp).backward(g)
                return inp.grad

            for label, fn, g, want in (
                ("tp_copy", tp_copy, gs[rank], _fold(gs, dtype)),
                ("seq_all_gather", seq_all_gather, gs_full[rank], _fold(gs_full, dtype)[sl]),
                ("sp_gather", sp_gather, gs_full[rank], gs_full[rank][sl]),
            ):
                got = grad(fn, g)
                check(f"{label} bwd {dtype}", _same(got, want))
                check(f"{label} bwd repeat {dtype}", _same(grad(fn, g), got))
            # all_gather / KV gather: rank r's gradient is the tree over every peer's column-block r.
            blocks = [g.view(t, world, h).permute(1, 0, 2) for g in gs_cols]
            check(f"all_gather bwd {dtype}", _same(grad(all_gather, gs_cols[rank]), _fold([b[rank] for b in blocks], dtype)))
            kv_cols = [g[:, : step * h] for g in gs_cols]
            peers = []
            for p in range(world):
                lo = (p // step) * step
                buf = torch.zeros(world, t, h, device=dev, dtype=dtype)
                buf[lo : lo + step] = kv_cols[p].reshape(t, step, h).permute(1, 0, 2)
                peers.append(buf[rank])
            check(f"kv_gather bwd {dtype}", _same(grad(kv_gather, kv_cols[rank]), _fold(peers, dtype)))
            # sp_scatter: forward keeps this rank's chunk; its adjoint gathers every rank's complete gradient.
            xs = torch.zeros(world * t, h, device=dev, dtype=dtype)
            check(f"sp_scatter bwd {dtype}", _same(grad(sp_scatter, gs[rank], xs), torch.cat(gs)))

        # ---- cp_gather (cp=2): CP groups are rank pairs ----
        torch.cuda.empty_cache()
        cp_group, cp_rank = pairs[rank // 2], rank % 2
        cp_pl = Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=Axis(2, cp_rank, cp_group), side="trainer", leaves=8)
        cu = [0, 512, 1280, 2048, 2056]  # every sequence a multiple of 2 * cp
        layout = build_layout(cu, 2, cp_rank, dev)
        gather = make_cp_gather(cp_pl)
        core = make_cp_attention_core(lambda q, k, v, cu_seqlens, max_q, scale: q * scale + v, cp_pl)
        cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
        for dtype in (torch.bfloat16, torch.float32):
            for i, shape in enumerate(((cu[-1],), (cu[-1], 4, 128), (cu[-1], 2048))):
                glob = _adversarial(shape, 4400 + i + 10 * (rank // 2), dev, dtype)
                x = glob.index_select(0, layout.mine).clone().requires_grad_()
                y = gather(x, layout)
                check(f"cp_gather fwd == cp1 stream {dtype} {shape}", _same(y, glob))
                g = [_adversarial(shape, 4500 + i + 10 * rank_ + 100 * (rank // 2), dev, dtype) for rank_ in (0, 1)]
                y.backward(g[cp_rank])
                want = (g[0].float() + g[1].float()).to(dtype).index_select(0, layout.mine)
                check(f"cp_gather bwd two-contributor sum {dtype} {shape}", _same(x.grad, want))
            q = _adversarial((cu[-1], 4, 128), 4600, dev, dtype)
            v = _adversarial((cu[-1], 4, 128), 4601, dev, dtype)
            want = (q * 0.5 + v).index_select(0, layout.mine)
            got = core(q.index_select(0, layout.mine), q.index_select(0, layout.mine), v.index_select(0, layout.mine),
                       cu_t, 768, 0.5)
            check(f"cp attention core == cp1 owned rows {dtype}", _same(got, want))

        # ---- CUDA graph: the forwards replay to their eager bits ----
        if graph:
            k_full, n = SHAPES["35b_o_proj"]
            k_local, m = k_full // world, 256
            xg = _adversarial((m, k_local), 801 + rank, dev, torch.bfloat16)
            wg = _adversarial((n, k_local), 802 + rank, dev, torch.bfloat16)
            sg = _adversarial((m, 1024), 803 + rank, dev, torch.bfloat16)
            fwds = (lambda: row_parallel(xg, wg), lambda: row_parallel_rs(xg, wg),
                    lambda: partial.finish(partial(xg, wg)), lambda: sp_gather(sg), lambda: seq_all_gather(sg))
            torch.cuda.synchronize()
            for _ in range(2):
                eager = [f().clone() for f in fwds]  # warms every shape/pool before capture
            torch.cuda.synchronize()
            dist.barrier()
            g_ = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_):
                captured = [f() for f in fwds]
            for rep in range(2):
                xg.copy_(_adversarial((m, k_local), 900 + rep + rank, dev, torch.bfloat16))
                sg.copy_(_adversarial((m, 1024), 950 + rep + rank, dev, torch.bfloat16))
                eager = [f().clone() for f in fwds]
                torch.cuda.synchronize()
                dist.barrier()
                g_.replay()
                torch.cuda.synchronize()
                for idx, (a, b) in enumerate(zip(captured, eager)):
                    check(f"graph replay == eager fwd{idx} rep{rep}", _same(a, b))
            del g_

        torch.cuda.synchronize()
        dist.barrier()
        counts = PA.transport_counts()
        seq = tp_bytes.seq_gather_transport_counts()
        if disable_p2p:
            check(f"nccl leg moved nothing over P2P {counts} {seq}", counts["p2p_calls"] == 0 and seq["p2p_calls"] == 0)
        else:
            check(f"P2P engaged, no fallback {counts} {seq}",
                  counts["p2p_calls"] > 0 and counts["p2p_fallbacks"] == 0
                  and seq["p2p_calls"] > 0 and seq["p2p_fallbacks"] == 0)
        out[rank] = {"failures": failures, "checks": checks[0]}
    finally:
        dist.destroy_process_group()


def run_leg(tp: int, leaf_dtype: str, leaves: int, disable_p2p: bool, env: dict, graph: bool, prefill: bool,
            quick: bool, timeout_s: int) -> dict:
    prior = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ctx = mp.get_context("spawn")
        with ctx.Manager() as mgr:
            out = mgr.dict()
            port = _free_port()
            procs = [ctx.Process(target=_worker, args=(r, tp, leaf_dtype, leaves, port, disable_p2p, graph, prefill, quick, out))
                     for r in range(tp)]
            for p in procs:
                p.start()
            deadline = time.monotonic() + timeout_s
            for p in procs:
                p.join(timeout=max(0, deadline - time.monotonic()))
            if any(p.is_alive() for p in procs):
                for p in procs:
                    p.terminate()
                    p.join(5)
                return {"ok": False, "error": "timed out"}
            codes = [p.exitcode for p in procs]
            res = dict(out)
            if codes != [0] * tp or sorted(res) != list(range(tp)):
                return {"ok": False, "error": f"exit codes {codes}, ranks reporting {sorted(res)}"}
            fails = {r: v["failures"] for r, v in res.items() if v["failures"]}
            return {"ok": not fails, "error": None, "failures": fails, "checks": sum(v["checks"] for v in res.values())}
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True, choices=(2, 4, 8))
    ap.add_argument("--legs", nargs="*", default=[leg[0] for leg in LEGS], choices=[leg[0] for leg in LEGS])
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per leg")
    ap.add_argument("--prefill", action="store_true", help=f"also run M={M_PREFILL} on the N<=2048 shapes")
    ap.add_argument("--quick", action="store_true", help="smoke: one shape, two row counts")
    a = ap.parse_args()
    if torch.cuda.device_count() < a.tp:
        print(f"[collectives-battery] needs {a.tp} visible GPUs, sees {torch.cuda.device_count()}", flush=True)
        return 2
    print(f"[collectives-battery] device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__}", flush=True)
    from isoexec.ops.collectives.pik_tp_invariant import p2p_arch_refusal

    p2p_refusal = p2p_arch_refusal()
    if p2p_refusal and "p2p" in a.legs:
        print(f"[collectives-battery] leg=p2p SKIPPED: {p2p_refusal}", flush=True)
    all_ok, n = True, 0
    for leaves in sorted({8, a.tp}, reverse=True):
        for leaf_dtype in ("fp32", "bf16"):
            for label, env, disable_p2p in LEGS:
                # One leaf per rank (the bf16 leaf wire) is a plan property, not a transport one: two legs suffice.
                if label not in a.legs or (leaves == a.tp and label not in ("p2p", "nccl")):
                    continue
                if not disable_p2p and p2p_refusal:
                    continue
                t0 = time.monotonic()
                r = run_leg(a.tp, leaf_dtype, leaves, disable_p2p, env, graph=label in ("p2p", "nccl"),
                            prefill=a.prefill, quick=a.quick, timeout_s=a.timeout)
                n += 1
                all_ok &= r["ok"]
                print(f"[collectives-battery] tp={a.tp} leaves={leaves} leaf_dtype={leaf_dtype} leg={label}: "
                      f"{'PASS' if r['ok'] else 'FAIL'} checks={r.get('checks')} ({time.monotonic() - t0:.0f}s) "
                      f"{r.get('error') or ''}", flush=True)
                for rank, fails in sorted((r.get("failures") or {}).items()):
                    print(f"[collectives-battery]   rank {rank}: {len(fails)} failures, first: {fails[:8]}", flush=True)
    print(f"[collectives-battery] RESULT tp={a.tp}: {'PASS' if all_ok else 'FAIL'} ({n} legs)", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
