"""CPU-offloaded Adam: every rank must run the same ATen CPU kernels (their bits depend on the capability)."""

import pytest
import torch

from isoexec.runtimes.megatron import optimizer


def test_uniform_capability_passes_and_a_mix_is_refused(monkeypatch):
    import torch.distributed as dist

    mine = str(torch.backends.cpu.get_cpu_capability())
    assert optimizer.require_uniform_cpu_capability() == mine          # no process group: nothing to agree
    peers = [mine, mine, mine]
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: len(peers))

    def gather(out, obj, group=None):
        out[:] = peers

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert optimizer.require_uniform_cpu_capability() == mine
    peers[2] = "DEFAULT" if mine != "DEFAULT" else "AVX2"
    with pytest.raises(RuntimeError, match=r"CPU_ADAM_CAPABILITY_MIXED.*\[2\]"):
        optimizer.require_uniform_cpu_capability()


def test_offloaded_adam_is_run_to_run_and_thread_count_invariant():
    def run(threads):
        before = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            gen = torch.Generator().manual_seed(0)
            p = torch.randn(4097, 33, generator=gen).requires_grad_()
            opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
            for _ in range(3):
                p.grad = torch.randn(p.shape, generator=gen) * 1e-2
                opt.step()
            return p.detach().clone()
        finally:
            torch.set_num_threads(before)

    a, b, c = run(1), run(1), run(4)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)) and torch.equal(a.view(torch.int32), c.view(torch.int32))
