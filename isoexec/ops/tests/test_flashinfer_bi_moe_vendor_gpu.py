"""The vendored FlashInfer batch-invariant BF16 MoE on an SM100 GPU: first call with no external kernel library, no
compiler process and no network; compile refused inside CUDA graph capture; coexistence with the host's FlashInfer in
one process, both load orders. Each relation runs in a child process (fresh caches)."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

REPO = pathlib.Path(__file__).resolve().parents[3]

_PRELUDE = r'''
import os, subprocess, socket, sys, time
BLOCKED = ("flashinfer", "deep_gemm", "flash_mla", "tilelang", "vllm")
MODE = sys.argv[1]

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked in this process")
        return None

if MODE in ("blocked", "capture"):
    sys.meta_path.insert(0, _Block())
    for n in BLOCKED:
        sys.modules[n] = None
SPAWNED = []
_popen = subprocess.Popen.__init__
def _record(self, args, *a, **k):
    SPAWNED.append(args if isinstance(args, str) else " ".join(map(str, args)))
    return _popen(self, args, *a, **k)
subprocess.Popen.__init__ = _record
_os_exec = os.execv
def _no_net(*a, **k):
    raise RuntimeError("network access at first call")
socket.socket.connect = _no_net

import torch
from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import api as bi
from isoexec.ops._vendor.flashinfer_bi_moe.jit import cute_dsl_core

E, H, I, K, T = 17, 512, 256, 7, 64
g = torch.Generator(device="cuda").manual_seed(7)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / 16).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=g) / 16).to(torch.bfloat16)
x = (torch.randn(T, H, device="cuda", generator=g) / 4).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E, device="cuda", generator=g)[:K] for _ in range(T)]).sort(1).values.int()
wts = torch.rand(T, K, device="cuda", generator=g)
W = bi.MoEWeights.prepare(w1, w2, act="swiglu_oai", params=(1.0, 0.0, 10.0))

def run():
    return bi.fused_moe(x, ids, wts, W)
'''

_BLOCKED = _PRELUDE + r'''
t0 = time.time()
y = run()
torch.cuda.synchronize()
first = time.time() - t0
assert torch.isfinite(y.float()).all()
leaked = [m for m in sys.modules if m.split(".")[0] in BLOCKED and sys.modules[m] is not None]
assert not leaked, leaked
compilers = [c for c in SPAWNED if any(t in c for t in ("nvcc", "ptxas", "clang", "gcc", "g++", "cicc"))]
assert not compilers, compilers
print("OK", len(cute_dsl_core.compiled_kernels()), f"{first:.1f}s")
'''

_CAPTURE = _PRELUDE + r'''
s = torch.cuda.Stream()
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph, stream=s):
        run()
except RuntimeError as exc:
    assert "never inside CUDA graph capture" in str(exc), exc
else:
    raise SystemExit("a cold compile inside capture was not refused")
torch.cuda.synchronize()
eager = run().clone()                     # warm-up at "placement": every variant compiles outside capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=s):
    out = run()
graph.replay()
torch.cuda.synchronize()
assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
print("OK")
'''

_COEXIST = _PRELUDE + r'''
def host():
    import flashinfer  # noqa: F401  (the host's copy, whatever version this runtime installs)
    try:
        from flashinfer.fused_moe.batch_invariant import api as hb
    except ImportError:
        return None
    hw = hb.MoEWeights.prepare(w1, w2, act="swiglu_oai", params=(1.0, 0.0, 10.0))
    return hb.fused_moe(x, ids, wts, hw)

if MODE == "vendored-first":
    mine = run().clone(); theirs = host(); again = run()
else:
    theirs = host(); mine = run().clone(); again = run()
torch.cuda.synchronize()
assert torch.equal(mine.view(torch.int16), again.view(torch.int16))
if theirs is not None:              # same upstream program: same bits as the vendored copy, unaffected by each other
    assert torch.equal(theirs.view(torch.int16), mine.view(torch.int16))
print("OK", "host-bi" if theirs is not None else "host-import-only")
'''


def _child(src, mode):
    out = subprocess.run([sys.executable, "-c", src, mode], cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0 and "OK" in out.stdout, (out.stdout[-2000:], out.stderr[-4000:])
    return out.stdout.strip().splitlines()[-1]


def test_first_call_without_external_libraries_or_compiler_processes():
    print(_child(_BLOCKED, "blocked"))


def test_compile_is_refused_inside_capture_and_warm_graphs_replay_the_eager_bits():
    _child(_CAPTURE, "capture")


@pytest.mark.parametrize("order", ["vendored-first", "host-first"])
def test_coexists_with_the_host_flashinfer(order):
    pytest.importorskip("flashinfer")
    print(_child(_COEXIST, order))


@pytest.mark.parametrize("T,k,E", [(1, 7, 385), (8, 7, 385), (36, 7, 385), (1, 6, 8), (40, 6, 64), (256, 1, 385)])
def test_one_cta_plan_is_the_plan(T, k, E):
    """IsoExec's one-CTA decode plan (n <= 256 slots) gives PlanKernel's counts / dest / src, dropped slots included."""
    import torch
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import permute as P
    g = torch.Generator(device="cuda").manual_seed(T * 31 + k)
    ids = torch.randint(-1, E + 1, (T, k), device="cuda", generator=g).int().contiguous()
    n = T * k
    outs = []
    for small in (True, False):
        counts = torch.full((E,), 7, dtype=torch.int32, device="cuda")
        dest = torch.full((n,), 9, dtype=torch.int32, device="cuda")
        src = torch.full((n,), 9, dtype=torch.int32, device="cuda")
        if small:
            dummy = torch.empty(1, dtype=torch.float32, device="cuda")
            P.get_compiled_plan_small()(ids.view(-1), counts, dest, src, dummy, dummy, n, E)
        else:
            P.get_compiled_plan(1, False)(ids.view(-1), counts, dest, src, dest, dest, n, E)
        outs.append((counts, dest, src))
    for a, b in zip(*outs):
        assert torch.equal(a, b)
    # the weighted form: the plan plus api.row_weights's values in one launch
    from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant import api
    w = torch.rand(T, k, device="cuda", generator=g)
    c2, d2, s2, rw = api.permute_plan_weighted(ids, E, w)
    assert torch.equal(c2, outs[1][0]) and torch.equal(d2.view(-1), outs[1][1]) and torch.equal(s2, outs[1][2])
    assert torch.equal(rw, api.row_weights(w, outs[1][2]))
