"""Real multi-rank bitwise gate of the BF16 actor on canonical BF16 MegaMoE, one process per GPU:

    torchrun --standalone --nproc-per-node W isoexec/ops/moe/deepseek_v41_sm100/tests/_actor_mega_ep_child.py --out r.json

Real geometry (E = 384 routed + 1 shared BF16 masters, H = 5120, I = 2304, top-6, L = 10). Every rank also holds all
masters for the reference = the grouped-kernel single-rank recipe (``ops.moe_forward``: the bits the trainer and
rollout produce on the grouped kernels). Checked bitwise on every rank:

* EP = 1 MegaMoE (``ops.moe_forward_mega``) == grouped at M (``--ms``; ``--sweep`` every M = 1..4096), skewed routing;
* EP = W MegaMoE on replicated rows (``ep.mega_ep_forward`` on this rank's canonical home slice with the fused NVLink
  gather of every slice) == grouped single-rank at the same M, ranks agree; skewed routing (every route on one
  rank's experts); the partitioned entry (rank r's rows are its own); chunked calls on a small-capacity workspace;
  a 120-call mixed-M stress;
* CUDA graph chains ([2048], [64], [2048, 64, 1024, 8], [1, 3, 4096, 17]) with three changed-input replays each
  (combine-readiness tags and gather tags across replays of one graph node).
Exit status 1 on any mismatch; the JSON receipt lists every failure.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

DEFAULT_MS = "1,2,3,5,8,16,17,31,64,127,128,256,511,512,513,1000,1024,1536,2047,2048,2500,4096"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=DEFAULT_MS)
    ap.add_argument("--skip-ep1", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    W, R = dist.get_world_size(), dist.get_rank()
    dev = torch.device("cuda", local)
    group = dist.group.WORLD

    from isoexec.ops.moe import ep_fixed_order as epf
    from isoexec.ops.moe.deepgemm_sm100.mega_ep import home_slice
    from isoexec.ops.moe.deepseek_v41_sm100 import ep, mega, ops

    E, H, I, K, L = 384, 5120, 2304, 6, 10.0
    KW = dict(top_k=K, routed_scale=1.5, limit=L)
    log = (lambda *a: print(f"[rank {R}]", *a, flush=True)) if R == 0 else (lambda *a: None)

    g = torch.Generator(device=dev).manual_seed(1234)
    gu = torch.empty(E + 1, 2 * I, H, dtype=torch.bfloat16, device=dev)
    dn = torch.empty(E + 1, H, I, dtype=torch.bfloat16, device=dev)
    for e in range(E + 1):       # per-expert fill keeps the fp32 temporaries small
        gu[e].copy_(torch.randn(2 * I, H, generator=g, device=dev) * 0.03)
        dn[e].copy_(torch.randn(H, I, generator=g, device=dev) * 0.03)
    rw = (torch.randn(E, H, generator=g, device=dev) * 0.03).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device=dev) * 0.01
    NL = E // W
    gl = torch.cat((gu[R * NL:(R + 1) * NL], gu[E:])).contiguous()
    dl = torch.cat((dn[R * NL:(R + 1) * NL], dn[E:])).contiguous()
    emap = epf.ExpertMap.contiguous(E, W, shared="home")
    ws = mega.Workspace.prepare(group, dev, E, K, H, I)
    t0 = time.time()
    mega.warm(ws, gl, dl, L)
    log(f"warm EP={W}: {len(ws.compiled)} geometries in {time.time() - t0:.1f}s, cap {ws.cap}")

    def grouped(x, b=rb):
        return ops.moe_forward(x, rw, b, gu, dn, **KW)

    def mega1(x, b=rb):
        return ops.moe_forward_mega(x, rw, b, gu, dn, **KW)

    def replicated(x, b=rb, workspace=ws):
        if W == 1:
            return mega1(x, b)
        M = x.size(0)
        spans = [home_slice(M, W, q) for q in range(W)]
        lo, hi = spans[R]
        return ep.mega_ep_forward(x[lo:hi], rw, b, gl, dl, emap=emap, ws=workspace,
                                  width=max(q1 - q0 for q0, q1 in spans), out_rows=M, **KW)

    def rows(M, seed):
        x = torch.randn(M, H, generator=torch.Generator(device=dev).manual_seed(seed), device=dev).to(torch.bfloat16)
        dist.broadcast(x, 0)
        return x

    failures, errors = [], []

    def check(name, got, want):
        a, b = got.detach().reshape(want.shape), want.detach()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.reshape(-1).view(torch.uint8),
                                                                        b.reshape(-1).view(torch.uint8)):
            n = int((a != b).sum()) if a.shape == b.shape else -1
            failures.append(f"{name}: {n} elements differ (shapes {tuple(a.shape)} vs {tuple(b.shape)})")

    def same_everywhere(name, y):
        got = [torch.empty_like(y) for _ in range(W)]
        dist.all_gather(got, y.contiguous())
        if not all(torch.equal(t, got[0]) for t in got):
            failures.append(f"{name}: ranks disagree")

    try:
        with torch.no_grad():
            for M in [int(m) for m in args.ms.split(",")]:
                x = rows(M, 100 + M)
                want = grouped(x)
                if not args.skip_ep1:
                    check(f"ep1 mega M={M}", mega1(x), want)
                y = replicated(x)
                torch.cuda.synchronize()
                check(f"ep{W} mega M={M}", y, want)
                same_everywhere(f"ep{W} mega M={M}", y)
            if W > 1:      # chunked calls (per-rank capacity 256): every chunk prepped, launched and gathered
                small = mega.Workspace(group, dev, E, 256, K, H, I)
                mega.warm(small, gl, dl, L)
                for M in (5, 600, 1500, 4096):
                    x = rows(M, 77 + M)
                    check(f"ep{W} chunked cap={small.cap} M={M}", replicated(x, workspace=small), grouped(x))
                del small
            g2 = torch.Generator().manual_seed(99)
            for j in range(120):                     # mixed-M stress (tags advance every call, eager)
                M = int(torch.randint(1, 4097, (1,), generator=g2))
                x = rows(M, 5000 + j)
                y = replicated(x)
                if j % 8 == 0:
                    check(f"stress call {j} M={M}", y, grouped(x))
            if args.sweep:
                for M in range(1, 4097):
                    x = rows(M, 20000 + M)
                    want = grouped(x)
                    check(f"sweep ep{W} M={M}", replicated(x), want)
                    if not args.skip_ep1:
                        check(f"sweep ep1 M={M}", mega1(x), want)
                log(f"sweep done: failures so far {len(failures)}")
            for owner in sorted({0, W - 1}):          # every route on one rank's experts
                b = rb.clone()
                b[owner * NL: owner * NL + K] += 50.0
                for M in (683, 2048):
                    x = rows(M, 7 + M)
                    want = grouped(x, b)
                    check(f"skew owner={owner} M={M}", replicated(x, b), want)
                    if not args.skip_ep1:
                        check(f"ep1 skew owner={owner} M={M}", mega1(x, b), want)
            for M in ((5, 1023) if W > 1 else ()):    # partitioned entry: rank r's rows are its own
                x = rows(M, 55 + M)
                lo, hi = home_slice(M, W, R)
                width = max(b - a for a, b in (home_slice(M, W, q) for q in range(W)))
                got = ep.mega_ep_forward(x[lo:hi].contiguous(), rw, rb, gl, dl, emap=emap, ws=ws, width=width, **KW)
                check(f"partitioned M={M}", got, grouped(x)[lo:hi])
        log(f"eager: failures so far {len(failures)}")
    except Exception as exc:
        errors.append(f"eager: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    def capture(fn, xs):
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for xb in xs:
                    fn(xb)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            ys = [fn(xb) for xb in xs]
        torch.cuda.synchronize()
        dist.barrier()
        return gr, ys

    if not errors:
        try:
            with torch.no_grad():
                for name, fn in (("ep%d" % W, replicated),) + ((("ep1", mega1),) if not args.skip_ep1 else ()):
                    for chain in ([2048], [64], [2048, 64, 1024, 8], [1, 3, 4096, 17]):
                        xs = [torch.empty(M, H, dtype=torch.bfloat16, device=dev) for M in chain]
                        for M, xb in zip(chain, xs):
                            xb.copy_(rows(M, 100 + M))
                        gr, ys = capture(fn, xs)
                        for rep in range(3):
                            fresh = [rows(M, 1000 * rep + M + 17 * j) for j, M in enumerate(chain)]
                            for xb, x in zip(xs, fresh):
                                xb.copy_(x)
                            gr.replay()
                            torch.cuda.synchronize()
                            for M, y, x in zip(chain, ys, fresh):
                                check(f"{name} graph chain={chain} M={M} replay {rep}", y, grouped(x))
                        del gr, ys
                        torch.cuda.synchronize()
                        dist.barrier()
            log(f"graphs: failures so far {len(failures)}")
        except Exception as exc:
            errors.append(f"graphs: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    flags = [None] * W
    dist.all_gather_object(flags, (failures, errors))
    ok = all(not f and not e for f, e in flags)
    if R == 0:
        result = {"world": W, "pass": ok, "per_rank": [{"failures": f, "errors": e} for f, e in flags]}
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=1)
        print(json.dumps({"world": W, "pass": ok}), flush=True)
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
