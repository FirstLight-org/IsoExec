"""Pass 2 always finishes (README "Forward progress"): graph-replayed stateful prefills with separate
sweep/output launches and 8 output CTAs per SM (most start only as others finish), while a second
prefill kernel runs concurrently on another stream and competes for the same SMs. Slots are reused in
place within a pack and across calls (sequences continue, move into slots freed by earlier calls, start
fresh, are empty). A watchdog kills the child process: a hang fails the test instead of the suite, and
every result must equal the plain eager call bitwise.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import SM90

pytestmark = SM90

CHILD = textwrap.dedent(
    """
    import random, torch
    from isoexec.ops._testing import bitwise_equal
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
    from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import BF, DEV, I32, state_equal

    HK, HV, POOL, N, T = 2, 4, 48, 20, 2600
    rng = random.Random(11)
    g0 = torch.Generator(device=DEV).manual_seed(3)
    A_log = torch.log(torch.empty(HV, device=DEV).uniform_(1, 16, generator=g0))
    dt_bias = torch.randn(HV, device=DEV, generator=g0) * 0.5

    def inputs(seed, T):
        g = torch.Generator(device=DEV).manual_seed(seed)
        r = lambda *s: torch.randn(*s, device=DEV, generator=g).to(BF)
        return [r(T, HK, 128), r(T, HK, 128), r(T, HV, 128), r(T, HV), r(T, HV)]

    def pack(step):
        outs = rng.sample(range(POOL), N)
        ins = [so if rng.random() < 0.6 else -1 for so in outs]   # in place, or fresh in a used slot
        cuts = sorted(rng.choices(range(T + 1), k=N - 1))
        return ins, outs, [0] + cuts + [T]

    eager, graphed, rival = (G.CprState.allocate(POOL, HV, DEV) for _ in range(3))
    static = dict(x=inputs(0, T), si=torch.zeros(N, dtype=I32, device=DEV), so=torch.zeros(N, dtype=I32, device=DEV),
                  cu=torch.linspace(0, T, N + 1, device=DEV).round().to(I32))   # a valid pack
    rx = inputs(99, T)
    rcu = torch.tensor([0, 700, 1400, 2100, T], dtype=I32, device=DEV)
    rslots = torch.arange(4, dtype=I32, device=DEV)

    def ours():
        with G._prefill_mode("two_pass"), G._sweep_schedule(oversubscribe=True):
            return G.prefill(*static["x"], A_log, dt_bias, static["cu"], state=graphed, slot_in=static["si"],
                             slot_out=static["so"])

    def theirs():   # the co-running kernel: a normal prefill of another pool
        return G.prefill(*rx, A_log, dt_bias, rcu, state=rival, slot_in=rslots)

    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    for s in (side, other):
        s.wait_stream(torch.cuda.current_stream())
    snap = graphed.data.clone()
    with torch.cuda.stream(side):
        ours()
    with torch.cuda.stream(other):
        theirs()
    torch.cuda.synchronize()
    graphed.data.copy_(snap)
    g_ours, g_theirs = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_ours):
        out = ours()
    with torch.cuda.graph(g_theirs, stream=other):
        theirs()
    for step in range(12):
        ins, outs, cu = pack(step)
        x = inputs(step + 1, T)
        si, so, cut = (torch.tensor(v, dtype=I32, device=DEV) for v in (ins, outs, cu))
        want = G.prefill(*x, A_log, dt_bias, cut, state=eager, slot_in=si, slot_out=so)
        for t, v in zip(static["x"], x):
            t.copy_(v)
        static["si"].copy_(si); static["so"].copy_(so); static["cu"].copy_(cut)
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            g_theirs.replay()   # competes for the SMs with the replay below
        g_ours.replay()
        torch.cuda.synchronize()
        assert bitwise_equal(out, want), f"step {step}: outputs"
        assert state_equal(eager, graphed, outs), f"step {step}: records"
    print("OK")
    """
)


def test_pass2_finishes_oversubscribed_with_a_corunner():
    torch.cuda.empty_cache()   # the child needs a little device memory next to this process's cache
    try:
        res = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:  # pragma: no cover - the failure this test exists for
        raise AssertionError("pass 2 did not finish within the watchdog (240 s): a forward-progress hang")
    if res.returncode != 0 and "out of memory" in res.stderr:  # pragma: no cover - shared GPU full
        pytest.skip("the child process could not allocate device memory (GPU shared and full)")
    assert res.returncode == 0 and res.stdout.strip().endswith("OK"), res.stdout[-2000:] + res.stderr[-4000:]
