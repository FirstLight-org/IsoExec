"""Check released four-layer GLM FP8 projection weights on four SM100 GPUs.

torchrun --standalone --nproc-per-node=4 examples/check_glm53_fp8.py CHECKPOINT MANIFESTS OUTPUT_DIR
Pass each resulting *.inputs.json to project-x's policy_validation/lowbit/glm_fp8_schedule.py --probe.
"""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch
import torch.distributed as dist

from isoexec.ops.mm.glm_fp8_sm100 import ProjectionSchedule, merge_partials, project


def words(tensor, dtype):
    return tensor.detach().cpu().contiguous().view(dtype).tolist()


def check(name, manifest, load, rank, out):
    key = f"model.language_model.layers.{name}.weight"
    host_w, host_s = load(key), load(key + "_scale_inv")
    w, s = host_w.cuda(), host_s.cuda()
    n, k = w.shape
    rng = torch.Generator().manual_seed(k + n)
    x = torch.randn(64, k, dtype=torch.bfloat16, generator=rng).cuda()
    schedule = ProjectionSchedule.from_manifest(manifest)
    if rank == 0:
        y = project(x, w, s, schedule=schedule)
        solo = project(x[63:], w, s, schedule=schedule)
        assert torch.equal(y[63:].view(torch.int32), solo.view(torch.int32)), (name, "batch")
        bf = project(x, w, s, schedule=schedule, dtype=torch.bfloat16)
        assert torch.equal(bf.view(torch.int16), y.bfloat16().view(torch.int16)), (name, "BF16")
    if schedule.max_tp == 4:
        a, b = rank * (k // 4), (rank + 1) * (k // 4)
        local = project(x[:, a:b], w[:, a:b], s[:, a//128:b//128], schedule=schedule, tp_degree=4)
        parts = [torch.empty_like(local) for _ in range(4)]
        dist.all_gather(parts, local)
        if rank == 0:
            merged = merge_partials(torch.stack(parts))
            assert torch.equal(merged.view(torch.int32), y.view(torch.int32)), (name, "TP4")
    if rank == 0:
        cols = [0, 127, 128, n - 1]
        rows = [0, 63]
        probe = {"N": n, "K": k, "cols": cols, "max_tp": schedule.max_tp,
                 "manifest_hash": schedule.manifest_hash,
                 "X": words(x[rows], torch.uint16), "W": host_w.view(torch.uint8)[cols].tolist(),
                 "WS": words(host_s[[c//128 for c in cols]], torch.uint32),
                 "output": words(y[rows][:, cols], torch.uint32)}
        (out / f"{name}.inputs.json").write_text(json.dumps(probe))
        print(f"{name}: batch 1/64 and BF16 root passed" +
              (", physical TP4 passed" if schedule.max_tp == 4 else ""), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifests", type=Path, help="project-x --release-manifests --max-tp 4 output")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        parser.error("the K-sharding comparison requires four local ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        mapping = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        def load(key):
            with safe_open(args.checkpoint / mapping[key], framework="pt") as shard:
                return shard.get_tensor(key)
        manifests = json.loads(args.manifests.read_text())
        if not manifests or {p["reductions"]["y"]["d_max"] for p in manifests.values()} != {1, 4}:
            raise ValueError("expected single-shard and four-shard deployment manifests")
        for name, manifest in manifests.items():
            check(name, manifest, load, rank, args.output_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
