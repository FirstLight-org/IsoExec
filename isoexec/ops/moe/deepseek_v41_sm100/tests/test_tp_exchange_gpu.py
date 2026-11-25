"""NVLink exchanges of the TP-sharded experts on real ranks.

SymmExchange (pull): its gathers (copy-in and in place) are NcclExchange's bytes (eager and replayed CUDA graphs over
many calls), and tp_moe_forward over it is ops.moe_forward bit for bit.

MulticastExchange (Lamport multicast push): at world 2 (and 4 with four GPUs) its gathers are NcclExchange's bytes and
an independently regenerated expectation, for many row counts on both regions and both paths (multicast below the
capacity, pull above), payloads with every special bf16 pattern (NaNs with several payloads, +-0, +-inf, subnormals,
the 32-bit words 0x80000000 / 0x00000000) and all 65536 bf16 patterns in one slice, FP32 payloads with arbitrary bits,
copy-in and in place, many back-to-back calls with varying row counts (stage rotation and cleaning), and CUDA graph
replays past a full cycle of the 65024 tags; the device state (stage, tag counter, calls) is identical on every rank;
tp_moe_forward over it is ops.moe_forward bit for bit. Refusals: no multicast (TORCH_SYMM_MEM_DISABLE_MULTICAST=1) and
a group spanning hosts raise named errors on every rank. A lost peer makes the wait time out, record the failure and
trap (no hang). The test fails, saying so, when torch symmetric memory gives no multicast pointer on this node.

Groups: two groups over the same ranks (world 2) or of one size over different ranks (world 4), one exchange each,
calls interleaved, are NcclExchange's bytes; a setup only some ranks take part in makes the next exchange over a
skewed group raise RendezvousSkewError on every rank (no hang).

Skipped without a two-GPU (four-GPU) lease."""
from __future__ import annotations

import os

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")


def _rank(rank, world, port, out):
    import torch.distributed as dist

    from isoexec.ops.moe.deepseek_v41_sm100 import ops, tp_exchange, tp_experts
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    ok = []
    E, H, I = 256, 1024, 1024
    sx = tp_exchange.SymmExchange(None, torch.device("cuda", rank), max_tokens=512, intermediate=I, hidden=H)
    nx = tp_experts.NcclExchange()
    g = torch.Generator(device="cuda").manual_seed(11 + rank)
    for rows in (1, 7, 300, 3584):
        part = torch.randn(rows, I // world, device="cuda", generator=g).to(torch.bfloat16)
        ok.append(torch.equal(sx.gather_h(part).view(torch.int16), nx.gather_h(part).view(torch.int16)))
        inplace = sx.h_buffer(rows, I // world)                 # the producer writes into the region: no copy
        inplace.copy_(part)
        ok.append(torch.equal(sx.gather_h(inplace).view(torch.int16), nx.gather_h(part).view(torch.int16)))
        if rows <= 512:                                         # the out region holds max_tokens rows
            yo = torch.randn(rows, H // world, device="cuda", generator=g).to(torch.bfloat16)
            yin = sx.out_buffer(rows, H // world)
            yin.copy_(yo)
            ok.append(torch.equal(sx.gather_out(yin).view(torch.int16), nx.gather_out(yo).view(torch.int16)))
    # graphs: several replays alternate the region parity
    part = torch.randn(70, I // world, device="cuda", generator=g).to(torch.bfloat16)
    ref = nx.gather_h(part)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sx.gather_h(part)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        got = sx.gather_h(part)
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        ok.append(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
    # the experts over the NVLink exchange
    gg = torch.Generator(device="cuda").manual_seed(0)
    gu = (torch.randn(E + 1, 2 * I, H, generator=gg, device="cuda") / H ** 0.5).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=gg, device="cuda") / I ** 0.5).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=gg, device="cuda") / H ** 0.5).to(torch.bfloat16)
    rb = torch.randn(E, generator=gg, device="cuda") * 0.01
    gu_r, dn_r = tp_experts.shard_masters(gu, dn, world, rank)
    kw = {"top_k": 6, "routed_scale": 1.5, "limit": 10.0}
    for T in (1, 33, 300):
        x = (torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(T)) / 2).to(
            torch.bfloat16)
        got = tp_experts.tp_moe_forward(x, rw, rb, gu_r, dn_r, exchange=sx, **kw)
        want = ops.moe_forward(x, rw, rb, gu, dn, **kw)
        ok.append(torch.equal(got.view(torch.int16), want.view(torch.int16)))
    torch.save(ok, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


def test_nvlink_exchange_is_the_nccl_bytes_and_the_tp1_program(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    import torch.multiprocessing as mp
    port = 29900 + os.getpid() % 90
    mp.spawn(_rank, args=(2, port, str(tmp_path / "ok")), nprocs=2, join=True)
    for r in range(2):
        res = torch.load(tmp_path / f"ok.{r}")
        assert all(res), (r, res)



# ---------------------------------------------------------------------------------------------------------------------
# MulticastExchange (Lamport multicast push)

_SPECIAL = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7FC1, 0x7F81, 0x7FA5, 0x7FFF, 0xFFFF, 0xFF81,
            0x0001, 0x8001, 0x0040, 0x007F, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x3F80)
_HS, _IS = 640, 288                  # the TP8 slice widths (out [M, 640], h [7M, 288] per rank), any world here


def _i16(v):
    import torch
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def _payload(rows, cols, kind, rank, seed, dtype="bf16"):
    """Rank ``rank``'s slice of call ``seed``: regenerated identically by every rank (the independent expectation)."""
    g = torch.Generator(device="cuda").manual_seed(1000003 * seed + 7919 * rank + 17)
    if dtype == "fp32":
        bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (rows, cols), dtype=torch.int64, device="cuda", generator=g)
        bits = bits.to(torch.int32)
        flat = bits.view(-1)
        sp = torch.tensor([0x7FC00000, 0x7F800001, 0x7FFFFFFF, -1, 0x7F800000, -8388608, 0, -2 ** 31, 1, -2 ** 31 + 1,
                           0x00800000, 0x80000001 - 2 ** 32], dtype=torch.int64, device="cuda").to(torch.int32)
        flat[:min(flat.numel(), sp.numel())] = sp[:min(flat.numel(), sp.numel())]
        return bits.view(torch.float32)
    n = rows * cols
    if kind == "special":
        sp = _i16(torch.tensor(_SPECIAL, dtype=torch.int32, device="cuda"))
        bits = sp[torch.randint(0, len(_SPECIAL), (n,), device="cuda", generator=g)]
        if n >= 8:                    # words 0x80000000 (vLLM's sentinel), 0x00000000, 0xFFFFFFFF, 0x7FC07FC0
            bits[:8] = _i16(torch.tensor([0, 0x8000, 0, 0, 0xFFFF, 0xFFFF, 0x7FC0, 0x7FC0], device="cuda"))
    else:
        bits = _i16(torch.randint(0, 65536, (n,), dtype=torch.int32, device="cuda", generator=g))
        if kind == "all":             # every bf16 pattern once, rotated per rank
            k = min(n, 65536)
            bits[:k] = _i16(torch.arange(65536, dtype=torch.int32, device="cuda").roll(977 * rank + seed))[:k]
    return bits.view(torch.bfloat16).view(rows, cols)


def _same(a, b):
    it = torch.int32 if a.element_size() == 4 else torch.int16
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(it), b.view(it))


def _expect(rows, cols, kind, seed, world, dtype="bf16"):
    return torch.cat([_payload(rows, cols, kind, r, seed, dtype) for r in range(world)], 1)


def _rank_multicast(rank, world, port, out):
    import torch.distributed as dist

    from isoexec.ops.moe.deepseek_v41_sm100 import ops, tp_exchange, tp_experts
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    res = {"rank": rank, "world": world, "checks": []}

    def check(name, ok):
        res["checks"].append((name, bool(ok)))

    dev = torch.device("cuda", rank)
    try:
        mx = tp_exchange.MulticastExchange(None, dev, max_tokens=512, intermediate=_IS * world, hidden=_HS * world,
                                           lamport_max_bytes=512 << 10, timeout_s=60.0)
    except tp_exchange.MulticastUnavailableError as exc:
        res.update(multicast=False, error=str(exc))
        torch.save(res, f"{out}.{rank}")
        dist.barrier()
        dist.destroy_process_group()
        return
    res.update(multicast=True, status=mx.multicast_status)
    nx = tp_experts.NcclExchange()
    seed = [0]

    def one(which, rows, cols, kind, inplace=False, dtype="bf16"):
        seed[0] += 1
        part = _payload(rows, cols, kind, rank, seed[0], dtype)
        x = part
        if inplace:
            x = (mx.h_buffer if which == "h" else mx.out_buffer)(rows, cols)
            x.copy_(part)
        path = mx.path(x, which)
        got = (mx.gather_h if which == "h" else mx.gather_out)(x)
        want = (nx.gather_h if which == "h" else nx.gather_out)(part)
        indep = _expect(rows, cols, kind, seed[0], world, dtype)
        check(f"{which} rows={rows} {kind} {dtype} inplace={inplace} path={path}",
              _same(got, want) and _same(got, indep))
        return path

    # h: multicast up to 910 rows (512 KiB / 576 B), pull above; out: multicast up to 409 rows, pull above
    paths = set()
    for rows in (1, 2, 3, 5, 7, 8, 13, 31, 64, 127, 128, 129, 228, 300, 511, 700, 910, 911, 1500, 3584):
        for kind in ("random", "special") + (("all",) if rows * _IS >= 65536 else ()):
            paths.add(one("h", rows, _IS, kind))
        paths.add(one("h", rows, _IS, "special", inplace=True))
    for rows in (1, 7, 64, 300, 409, 410, 512):
        for kind in ("random", "special"):
            paths.add(one("out", rows, _HS, kind))
        paths.add(one("out", rows, _HS, "special", inplace=True))
    for rows in (1, 300, 910, 1500):                           # FP32 payloads (any bits): 144 x 4 B = one h row
        paths.add(one("h", rows, _IS // 2, "random", dtype="fp32"))
    check("both paths exercised", paths == {"multicast", "pull"})

    # back to back, no host sync, varying row counts (stage rotation; cleaning of a larger previous call)
    seq = [910, 1, 300, 2, 700, 7, 910, 910, 1, 1, 64, 511, 3, 910, 129, 1, 228, 8, 600, 5] * 3
    pend = []
    for i, rows in enumerate(seq):
        seed[0] += 1
        kind = ("random", "special", "all")[i % 3] if rows * _IS >= 65536 else ("random", "special")[i % 2]
        part = _payload(rows, _IS, kind, rank, seed[0])
        pend.append((mx.gather_h(part), rows, kind, seed[0]))
        if i % 4 == 0:
            ry = 1 + (i * 37) % 409
            seed[0] += 1
            pend.append((mx.gather_out(_payload(ry, _HS, "special", rank, seed[0])), ry, "out", seed[0]))
    torch.cuda.synchronize()
    for got, rows, kind, sd in pend:
        cols = _HS if kind == "out" else _IS
        check(f"back-to-back rows={rows} {kind}", _same(got, _expect(rows, cols, "special" if kind == "out" else kind,
                                                                     sd, world)))

    # CUDA graph: varying row counts per replay, multicast and pull mixed; replays past a full tag cycle
    plan = [("h", 1), ("h", 300), ("h", 7), ("h", 910), ("out", 409), ("h", 64), ("h", 1500), ("out", 2), ("h", 2)]
    static = []
    for which, rows in plan:
        cols = _IS if which == "h" else _HS
        static.append(torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda"))

    def load(sd):
        for k, ((which, rows), x) in enumerate(zip(plan, static)):
            x.copy_(_payload(rows, x.size(1), "special" if k % 2 else "random", rank, sd * 100 + k))

    def verify(outs, sd, tag):
        ok = all(_same(o, _expect(rows, x.size(1), "special" if k % 2 else "random", sd * 100 + k, world))
                 for k, (o, (which, rows), x) in enumerate(zip(outs, plan, static)))
        check(f"graph {tag}", ok)

    load(10 ** 6)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (which, _), x in zip(plan, static):
            (mx.gather_h if which == "h" else mx.gather_out)(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        outs = [(mx.gather_h if which == "h" else mx.gather_out)(x) for (which, _), x in zip(plan, static)]
    h_calls = sum(1 for w, r in plan if w == "h" and r * _IS * 2 <= mx.h.cap_bytes)
    replays = 65024 // h_calls + 50                            # every tag of the h region used, then wrapped
    for i in range(replays):
        checked = i < 6 or i % 1000 == 0 or i >= replays - 3
        if checked:
            load(10 ** 6 + i)
        graph.replay()
        if checked:
            torch.cuda.synchronize()
            verify(outs, 10 ** 6 + i, f"replay {i}")
    torch.cuda.synchronize()
    st = torch.stack((mx.h.state[:4], mx.out.state[:4])).to(torch.int64)
    allst = [torch.empty_like(st) for _ in range(world)]
    dist.all_gather(allst, st)
    check("device state (stage, arrivals, tag counter, calls) identical on every rank",
          all(torch.equal(a, allst[0]) for a in allst) and int(st[0, 1]) == 0 and int(st[1, 1]) == 0)
    res["state"] = st.tolist()
    res["h_calls"] = int(st[0, 3])
    check("tag counter wrapped", int(st[0, 3]) > 65024)
    check("no error record", mx.last_error() is None)

    # the experts over the multicast exchange
    E, H, I = 256, 1024, 1024
    ex = tp_exchange.MulticastExchange(None, dev, max_tokens=512, intermediate=I, hidden=H)
    gg = torch.Generator(device="cuda").manual_seed(0)
    gu = (torch.randn(E + 1, 2 * I, H, generator=gg, device="cuda") / H ** 0.5).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=gg, device="cuda") / I ** 0.5).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=gg, device="cuda") / H ** 0.5).to(torch.bfloat16)
    rb = torch.randn(E, generator=gg, device="cuda") * 0.01
    gu_r, dn_r = tp_experts.shard_masters(gu, dn, world, rank)
    kw = {"top_k": 6, "routed_scale": 1.5, "limit": 10.0}
    for T in (1, 33, 300):
        x = (torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(T)) / 2).to(
            torch.bfloat16)
        got = tp_experts.tp_moe_forward(x, rw, rb, gu_r, dn_r, exchange=ex, **kw)
        want = ops.moe_forward(x, rw, rb, gu, dn, **kw)
        check(f"experts T={T}", _same(got, want))
    torch.save(res, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


_PORTS = iter(range(29700 + os.getpid() % 100, 29700 + os.getpid() % 100 + 1000, 7))


def _run(fn, world, tmp_path, *extra):
    import torch.multiprocessing as mp
    port = next(_PORTS)
    mp.spawn(fn, args=(world, port, str(tmp_path / "res"), *extra), nprocs=world, join=True)
    return [torch.load(tmp_path / f"res.{r}", weights_only=False) for r in range(world)]


def _assert_multicast(results):
    for res in results:
        print({k: v for k, v in res.items() if k != "checks"})
        assert res.get("multicast"), (
            "torch symmetric memory gives NO multicast pointer on this node (handle.multicast_ptr == 0): the "
            f"multicast exchange cannot run here; rank {res['rank']}: {res.get('error')}")
        bad = [c for c in res["checks"] if not c[1]]
        assert not bad and res["checks"], (res["rank"], bad)


@pytest.mark.parametrize("world", [2, 4])
def test_multicast_exchange_is_the_nccl_bytes(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires a {world}-GPU lease")
    _assert_multicast(_run(_rank_multicast, world, tmp_path))


def _rank_refusal(rank, world, port, out, mode):
    import torch.distributed as dist
    if mode == "no_multicast":
        os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"] = "1"
    from isoexec.ops.collectives import node_local
    from isoexec.ops.moe.deepseek_v41_sm100 import tp_exchange
    if mode == "hosts" and rank == 1:
        node_local.host_identity = lambda: "another-host/boot"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    want = tp_exchange.MulticastUnavailableError if mode == "no_multicast" else node_local.NodeLocalityError
    got = None
    try:
        tp_exchange.MulticastExchange(None, torch.device("cuda", rank), max_tokens=8, intermediate=_IS * world,
                                      hidden=_HS * world)
    except Exception as exc:  # noqa: BLE001
        got = exc
    torch.save({"rank": rank, "refused": isinstance(got, want), "error": repr(got)}, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["no_multicast", "hosts"])
def test_multicast_exchange_refuses_by_name(mode, tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    for res in _run(_rank_refusal, 2, tmp_path, mode):
        assert res["refused"], res


def _rank_lost_peer(rank, world, port, out):
    """Rank 1 never calls: rank 0's wait must time out, record the failure and trap (a launch error, no hang)."""
    import time

    import torch.distributed as dist

    from isoexec.ops.moe.deepseek_v41_sm100 import tp_exchange
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    try:
        mx = tp_exchange.MulticastExchange(None, torch.device("cuda", rank), max_tokens=8, intermediate=_IS * world,
                                           hidden=_HS * world, timeout_s=2.0)
    except tp_exchange.MulticastUnavailableError as exc:
        torch.save({"rank": rank, "multicast": False, "error": str(exc)}, f"{out}.{rank}")
        dist.barrier()
        dist.destroy_process_group()
        return
    done = f"{out}.0"
    if rank == 0:
        t0 = time.time()
        res = {"rank": 0, "multicast": True, "trapped": False}
        try:
            mx.gather_h(torch.zeros((7, _IS), dtype=torch.bfloat16, device="cuda"))
            torch.cuda.synchronize()
        except RuntimeError as exc:
            res.update(trapped=True, error=str(exc)[:300])
        res.update(seconds=time.time() - t0, record=mx.last_error())
        torch.save(res, done)
    else:
        torch.save({"rank": 1, "multicast": True}, f"{out}.1")
        t0 = time.time()
        while not os.path.exists(done) and time.time() - t0 < 300:
            time.sleep(0.5)
    os._exit(0)                                                 # rank 0's context is lost: no collective teardown


def test_multicast_exchange_lost_peer_traps_with_a_record(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    res = _run(_rank_lost_peer, 2, tmp_path)
    assert res[0].get("multicast"), ("torch symmetric memory gives NO multicast pointer on this node", res[0])
    r0 = res[0]
    print(r0)
    assert r0["trapped"] and r0["seconds"] < 120, r0
    rec = r0["record"]
    assert rec and rec[0]["set"] == 1 and rec[0]["rank"] == 0 and rec[0]["source_rank"] == 1, r0


# ---------------------------------------------------------------------------------------------------------------------
# Groups of one size in one process, and a lone subgroup (``moe.symm_rendezvous``)

def _rank_groups(rank, world, port, out):
    """Two families of groups of one size with different roles (an exchange on each), calls interleaved: every gather
    is the bytes of NcclExchange over the same group and of the regenerated expectation (no aliasing between groups).
    World 2: both families are {0, 1} (two groups over the same ranks). World 4: {0,1},{2,3} beside {0,2},{1,3}.
    Then a setup only some ranks take part in (rank 0 alone at world 2; {0, 3} at world 4): the next exchange over a
    group whose ranks are now at different setup counts raises RendezvousSkewError on EVERY rank (no hang), and the
    exchanges built before keep serving."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    from isoexec.ops.collectives import symm_setup
    from isoexec.ops.moe import symm_rendezvous as sr
    from isoexec.ops.moe.deepseek_v41_sm100 import tp_exchange, tp_experts
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, device_id=dev)
    res = {"rank": rank, "world": world, "checks": []}

    def check(name, ok):
        res["checks"].append((name, bool(ok)))

    def mine(families):                              # every rank creates every group, in one order
        made = [(ranks, dist.new_group(list(ranks))) for ranks in families]
        return next((ranks, g) for ranks, g in made if rank in ranks)

    if world == 2:
        fam_a, fam_b, lone = [(0, 1)], [(0, 1)], (0,)
    else:
        fam_a, fam_b, lone = [(0, 1), (2, 3)], [(0, 2), (1, 3)], (0, 3)
    ranks_a, ga = mine(fam_a)
    ranks_b, gb = mine(fam_b)
    lone_g = dist.new_group(list(lone))
    n = len(ranks_a)
    kw = {"max_tokens": 64, "intermediate": _IS * n, "hidden": _HS * n}
    xa = tp_exchange.SymmExchange(ga, dev, **kw)
    try:
        xb = tp_exchange.MulticastExchange(gb, dev, lamport_max_bytes=64 << 10, **kw)
        res["multicast"] = True
    except tp_exchange.MulticastUnavailableError as exc:  # the property here is group isolation: pull serves too
        xb, res["multicast"] = exc.pull, False
    na, nb = tp_experts.NcclExchange(ga), tp_experts.NcclExchange(gb)
    seed = 0
    for i, rows in enumerate((1, 7, 64, 3, 200, 64, 1, 400, 113, 114)):  # b: multicast <= 113 h rows, pull above
        kind = ("random", "special")[i % 2]
        for name, ranks, ex, nx in (("a", ranks_a, xa, na), ("b", ranks_b, xb, nb)):
            seed += 1
            part = _payload(rows, _IS, kind, rank, seed)
            want = torch.cat([_payload(rows, _IS, kind, r, seed) for r in ranks], 1)
            got, ref = ex.gather_h(part), nx.gather_h(part)    # every rank issues the same calls, whatever the bits
            check(f"{name}{ranks} h rows={rows} path={ex.path(part) if hasattr(ex, 'path') else 'pull'}",
                  _same(got, ref) and _same(got, want))
            if rows <= 64:
                seed += 1
                y = _payload(rows, _HS, kind, rank, seed)
                want = torch.cat([_payload(rows, _HS, kind, r, seed) for r in ranks], 1)
                check(f"{name}{ranks} out rows={rows}", _same(ex.gather_out(y), want))
    counts = [torch.empty(1, dtype=torch.int64, device=dev) for _ in range(world)]
    dist.all_gather(counts, torch.tensor([symm_setup.setup_count()], dtype=torch.int64, device=dev))
    check("every rank at the same setup count", len({int(c) for c in counts}) == 1)

    if rank in lone:                                 # a setup only some ranks take part in
        buf = symm_mem.empty(4096, dtype=torch.uint8, device=dev)
        sr.rendezvous(buf, lone_g)
    torch.cuda.synchronize()
    verdict = "built"
    try:
        tp_exchange.SymmExchange(ga, dev, **kw)
    except sr.RendezvousSkewError:
        verdict = "skew"
    check("a group skewed by a lone setup is refused on every rank", verdict == "skew" and sr.skewed(ga))
    seed += 1
    part = _payload(9, _IS, "special", rank, seed)
    got = [xa.gather_h(part), na.gather_h(part), xb.gather_h(part), nb.gather_h(part)]
    check("exchanges built before the skew keep serving", _same(got[0], got[1]) and _same(got[2], got[3]))
    torch.save(res, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_groups_of_one_size_and_a_lone_subgroup(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires a {world}-GPU lease")
    for res in _run(_rank_groups, world, tmp_path):
        print({k: v for k, v in res.items() if k != "checks"})
        bad = [c for c in res["checks"] if not c[1]]
        assert res["checks"] and not bad, (res["rank"], bad)


def _rank_capacity(rank, world, port, out):
    """An exchange built for ``max_tokens`` used AT max_tokens with a swapped (decode) record: GEMM1 asks the h region
    for ``Tk + tile_n`` rows (before the headroom: ValueError 'slice exceeds the prepared capacity')."""
    import torch.distributed as dist

    from isoexec.ops.moe.deepseek_v41_sm100 import bi_tp, tp_exchange, tp_experts
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, device_id=dev)
    E, H, I, M = 384, 5120, 2304, 8
    g = torch.Generator(device="cuda").manual_seed(0)
    gu = (torch.randn(E + 1, 2 * I, H, device="cuda", generator=g) / 72).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, device="cuda", generator=g) / 48).to(torch.bfloat16)
    gu_r, dn_r = tp_experts.shard_masters(gu, dn, world, rank)
    del gu, dn
    ids = torch.stack([torch.randperm(E, device="cuda", generator=g)[:6] for _ in range(M)]).sort(1).values
    ids = torch.cat((ids, torch.full((M, 1), E, device="cuda")), 1).int().contiguous()
    w = torch.rand(M, 7, device="cuda", generator=g)
    x = (torch.randn(M, H, device="cuda", generator=g) / 4).to(torch.bfloat16)
    want = bi_tp.BiTPExperts(gu_r, dn_r, tp=world, limit=10.0, exchange=tp_experts.NcclExchange()).forward(x, ids, w)
    res = {"rank": rank, "checks": []}
    for name, mk in (("pull", tp_exchange.SymmExchange), ("multicast", tp_exchange.MulticastExchange)):
        ex = bi_tp.BiTPExperts(gu_r, dn_r, tp=world, limit=10.0, exchange=mk(None, dev, max_tokens=M, intermediate=I,
                                                                               hidden=H))
        c1, c2 = ex.record(M, 1), ex.record(M, 2)
        res["checks"].append((f"{name}: a swapped record at M = max_tokens", c1.swap_ab or c2.swap_ab))
        try:
            got = ex.forward(x, ids, w)
            ok = _same(got, want)
        except ValueError as exc:
            ok = repr(exc)[:120]
        res["checks"].append((f"{name}: forward at M = max_tokens == NCCL", ok is True))
    torch.save(res, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


def test_bi_tp_at_capacity_with_a_swapped_record(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    for res in _run(_rank_capacity, 2, tmp_path):
        bad = [c for c in res["checks"] if c[1] is not True]
        assert res["checks"] and not bad, (res["rank"], bad)
