"""mm_sm100/general: the TMEM base handoff of the 2-CTA plan under stress.

The epilogue reads the TMEM base the allocating warp wrote to shared memory. A handoff that is not ordered by a
releasing barrier lets a reader take whatever word a previous kernel left at that shared-memory address
(``misaligned address`` in tcgen05.ld, or silently wrong rows). The stale word is only non-benign when another
kernel has just used that shared memory and the reader wins the race, so the test interleaves the 2-CTA GEMM with
shared-memory-heavy GEMMs on several streams and in several processes on ONE device, for many launches, and
requires every result to equal the first bit for bit (and no CUDA error)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100", reason="needs an sm100 device"
)

ITERS = int(os.environ.get("ISOEXEC_TMEM_STRESS_ITERS", "20000"))
PROCS = int(os.environ.get("ISOEXEC_TMEM_STRESS_PROCS", "4"))


def _r(*s, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g)


def _stress(iters: int) -> None:
    from isoexec.ops.mm.mm_sm100.general import ops as G

    # the 2-CTA plan (M > 128, long K) with an MN-major B (dgrad layout) and a K-major B (forward layout)
    pair = [(_r(1000, 8192), _r(8192, 640, seed=1)), (_r(1024, 4096, seed=2), _r(1024, 4096, seed=3).t())]
    # shared-memory-heavy neighbours (1-CTA plans, other widths) that leave non-sentinel words behind
    noise = [(_r(512, 2048, seed=4), _r(2048, 256, seed=5)), (_r(64, 64, 512, seed=6), _r(64, 640, 512, seed=7).transpose(1, 2))]
    refs = [G.mm(a, b, cl=2, bn=256) for a, b in pair]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(3)]
    bad, outs = [], []
    for i in range(iters):
        s = streams[i % len(streams)]
        with torch.cuda.stream(s):
            na, nb = noise[i % len(noise)]
            (G.bmm if na.dim() == 3 else G.mm)(na, nb, cl=1, bn=64 if na.dim() == 2 else 128)
            a, b = pair[i % len(pair)]
            outs.append((i, i % len(pair), G.mm(a, b, cl=2, bn=256)))
        if len(outs) == 256 or i == iters - 1:
            torch.cuda.synchronize()
            bad += [j for j, k, o in outs if not torch.equal(o.view(torch.int16), refs[k].view(torch.int16))]
            outs = []
    assert not bad, f"{len(bad)} of {iters} 2-CTA results differ from the first (first at launch {bad[0]})"


def test_two_cta_handoff_one_process():
    _stress(ITERS)


def test_two_cta_handoff_processes_on_one_device():
    """Several processes on the same device (how the V4.1 trainer EP backward first exposed the race)."""
    code = ("import sys; sys.path.insert(0, %r); import torch; "
            "from isoexec.ops.mm.mm_sm100.tests.test_general_tmem_handoff_gpu import _stress; _stress(%d)"
            % (os.getcwd(), ITERS))
    env = dict(os.environ)
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
             for _ in range(PROCS)]
    results = [(p.wait(timeout=1800), p.stdout.read().decode(errors="replace")[-2000:]) for p in procs]
    failed = [out for rc, out in results if rc != 0]
    assert not failed, failed[0]
