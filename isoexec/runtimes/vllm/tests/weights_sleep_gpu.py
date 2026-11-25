"""Exact level-2 sleep on a real CuMem pool (1 GPU): what survives, what is discarded, and the stash's reuse.

A module built in the ``weights`` pool holds: Parameters (packed into one buffer, as packed projections are), a
registered buffer that aliases them, a content buffer (a temperature-like scalar), a plain tensor no module
registers (an extension's table), and a large uncovered allocation. After ``sleep(2)`` + ``wake_up(["weights"])``
every non-Parameter byte must equal its pre-sleep value; Parameters are rewritten as an update would, after which
the whole pool must be bitwise identical to before. Run twice to cover the stash's reuse; with the hook absent,
the uncovered bytes are undefined. Usage: ``python weights_sleep_gpu.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from vllm.device_allocator.cumem import CuMemAllocator

from isoexec.runtimes.vllm import weights_sleep as ws


def _pool_bytes(alloc):
    dev = torch.device("cuda", torch.cuda.current_device())
    return {p: ws._device_view(p, n, dev).clone() for p, n in ws._weights_allocations(alloc).items()}


def main() -> int:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    alloc = CuMemAllocator.get_instance()
    torch.manual_seed(0)
    model = torch.nn.Module()
    with alloc.use_memory_pool(tag="weights"):
        packed = torch.randn(3 * 4096 * 1024 + 7, dtype=torch.bfloat16, device=device)
        for i in range(3):
            model.register_parameter(f"w{i}", torch.nn.Parameter(packed[i * 4096 * 1024:(i + 1) * 4096 * 1024]
                                                                .view(4096, 1024), requires_grad=False))
        model.register_buffer("packed", packed, persistent=False)  # aliases the Parameters (+7 tail elements)
        model.register_buffer("temperature", torch.tensor([0.7], device=device), persistent=False)
        model.register_parameter("strided", torch.nn.Parameter(torch.randn(256, 512, device=device).t(),
                                                               requires_grad=False))  # non-contiguous
        table = torch.randint(0, 1 << 30, (64 << 20,), dtype=torch.int32, device=device)  # unregistered, 256 MiB
    sources = {n: p.detach().clone() for n, p in model.named_parameters()}  # outside the pool
    ok = True

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(model=model, get_model=lambda: model)

    def stock_sleep(worker, level):  # vLLM 0.33 CuMemBackend.suspend, plus the buffer snapshot it would take
        worker.snapshot = {n: b.cpu().clone() for n, b in worker.model_runner.model.named_buffers()}
        alloc.sleep(offload_tags=("runtime", "cudagraph") if level == 2 else ("weights", "runtime", "cudagraph"))

    ws._isoexec_model = lambda worker: model  # the test model stands in for a VllmModel
    worker = Worker()
    for cycle in range(2):
        before = _pool_bytes(alloc)
        torch.cuda.synchronize()
        ws._exact_sleep(worker, stock_sleep, 2)
        if worker.snapshot:
            print(f"FAIL: vLLM's buffer snapshot was not suppressed: {sorted(worker.snapshot)}")
            ok = False
        alloc.wake_up(["weights"])
        ws._exact_wake(worker, ["weights"])
        if not ws.weights_discarded(model):
            print("FAIL: the model is not marked discarded after a level-2 sleep")
            ok = False
        # Non-Parameter bytes are back before any update ...
        if not torch.equal(model.temperature.cpu(), torch.tensor([0.7])):
            print(f"FAIL cycle {cycle}: temperature {model.temperature.item()}")
            ok = False
        for name, p in model.named_parameters():  # ... then the update rewrites every Parameter
            p.data.copy_(sources[name])
        ws.mark_restored(model)
        after = _pool_bytes(alloc)
        diff = [hex(p) for p in before if p not in after or not torch.equal(before[p], after[p])]
        print(f"cycle {cycle}: {len(before)} weights allocations, stash {worker._isoexec_stash.host.numel() / 2**20:.1f} "
              f"MiB pinned, differing allocations after update: {diff}")
        ok = ok and not diff and set(before) == set(after)

    # Informational: stock level 2 (no stash) leaves the unregistered table undefined (the driver may hand back
    # the same pages, so "not lost" is possible and not a failure).
    torch.cuda.synchronize()
    ref = table.clone()
    alloc.sleep(offload_tags=("runtime", "cudagraph"))
    alloc.wake_up(["weights"])
    lost = not torch.equal(ref, table)
    print(f"informational: stock level 2 changed the unregistered table: {lost}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
