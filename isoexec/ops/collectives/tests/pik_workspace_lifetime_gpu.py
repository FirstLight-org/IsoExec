"""Single-GPU scratch growth, stream lifetime, and capture ownership regression.

Run with an explicitly assigned CUDA_VISIBLE_DEVICES and --out outside the source tree.
No extension builds or GEMM autotuning: exercises the real shared workspace allocator.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from isoexec.ops.collectives.pik_bootstrap import ensure_pik


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ensure_pik()
    from pik import gemm

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    baseline = torch.cuda.memory_allocated()
    for step in range(1, 33):
        gemm._workspace(1, 1, step * 8 * 1024 * 1024, device)
    torch.cuda.synchronize()
    receipt = {
        "sha256": hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest(),
        "steps": 32,
        "final_requested_bytes": 1024**3,
        "active_bytes": sum(t.numel() * t.element_size() for t in gemm._WORKSPACE.values()),
        "retired_count": len(gemm._WORKSPACE_RETIRED),
        "retired_bytes": sum(t.numel() * t.element_size() for t in gemm._WORKSPACE_RETIRED),
        "allocated_delta": torch.cuda.memory_allocated() - baseline,
    }
    assert receipt["retired_count"] == 0, receipt
    assert receipt["allocated_delta"] == 1024**3, receipt
    gemm._WORKSPACE.clear()
    gc.collect()

    # Work queued on a side stream must finish before the eager generation can be reused.
    source = torch.arange(4096, device=device, dtype=torch.float32)
    output = torch.empty_like(source)
    gemm._workspace(1, 1, source.numel(), device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scratch = gemm._workspace(1, 1, source.numel(), device).view_as(source)
        torch.cuda._sleep(10_000_000)
        torch.mul(source, 3, out=scratch)
        output.copy_(scratch)
        del scratch
    # Grow on the original allocation stream while the side stream still uses the old generation.
    gemm._workspace(1, 1, source.numel() * 3, device)
    churn = [torch.full_like(source, -1) for _ in range(32)]
    torch.cuda.synchronize()
    assert torch.equal(output, source * 3)
    del churn
    receipt["side_stream"] = "PASS"

    # A graph must retain its generation after a larger eager request changes the active buffer.
    graph_results = []
    for dtype in (torch.float32, torch.bfloat16):
        source = torch.arange(256, device=device, dtype=dtype)
        output = torch.empty_like(source)
        gemm._workspace(1, 1, source.numel(), device, dtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scratch = gemm._workspace(1, 1, source.numel(), device, dtype).view_as(source)
            torch.mul(source, 2, out=scratch)
            output.copy_(scratch)
        del scratch
        old_capacity = gemm._WORKSPACE[device, dtype].numel()
        grown = gemm._workspace(1, 1, old_capacity + 1, device, dtype)
        grown.fill_(-5)
        del grown
        torch.cuda.synchronize()
        for value in (1, 2, 3):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, torch.full_like(output, value * 2))
        graph_results.append({"dtype": str(dtype), "replays": 3, "status": "PASS"})
        del graph
    assert len(gemm._WORKSPACE_RETIRED) == 2
    receipt["graphs"] = graph_results
    receipt["captured_generations"] = len(gemm._WORKSPACE_RETIRED)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
