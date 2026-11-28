"""Check released four-layer GLM FP8 projection weights on four SM100 GPUs.

torchrun --standalone --nproc-per-node=4 examples/check_glm53_fp8.py CHECKPOINT OUTPUT_DIR
Pass each resulting *.inputs.json to project-x's policy_validation/lowbit/glm_fp8_schedule.py --probe.
"""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch
import torch.distributed as dist

from isoexec.ops.mm.glm_fp8_sm100 import project


PROJECTIONS = (
    "0.mlp.gate_proj", "0.mlp.down_proj",
    "3.mlp.experts.0.gate_proj", "3.mlp.experts.0.down_proj",
    "3.self_attn.kv_a_proj_with_mqa", "3.self_attn.q_a_proj",
    "3.self_attn.q_b_proj", "3.self_attn.o_proj",
)


def words(tensor, dtype):
    return tensor.detach().cpu().contiguous().view(dtype).tolist()


def check(name, load, rank, out):
    key = f"model.language_model.layers.{name}.weight"
    host_w, host_s = load(key), load(key + "_scale_inv")
    w, s = host_w.cuda(), host_s.cuda()
    n, k = w.shape
    rng = torch.Generator().manual_seed(k + n)
    x = torch.randn(64, k, dtype=torch.bfloat16, generator=rng).cuda()
    leaves = 4 if name == "3.self_attn.o_proj" else 1
    if rank == 0:
        y = project(x, w, s, leaves=leaves, dtype=torch.float32)
        solo = project(x[63:], w, s, leaves=leaves, dtype=torch.float32)
        assert torch.equal(y[63:].view(torch.int32), solo.view(torch.int32)), (name, "batch")
        bf = project(x, w, s, leaves=leaves)
        assert torch.equal(bf.view(torch.int16), y.bfloat16().view(torch.int16)), (name, "BF16")
    if leaves == 4:
        a, b = rank * (k // 4), (rank + 1) * (k // 4)
        local = project(x[:, a:b], w[:, a:b], s[:, a//128:b//128], dtype=torch.float32)
        parts = [torch.empty_like(local) for _ in range(4)]
        dist.all_gather(parts, local)
        if rank == 0:
            merged = (parts[0] + parts[1]) + (parts[2] + parts[3])
            assert torch.equal(merged.view(torch.int32), y.view(torch.int32)), (name, "TP4")
    if rank == 0:
        cols = [0, 127, 128, n - 1]
        rows = [0, 63]
        probe = {"N": n, "K": k, "cols": cols, "leaves": leaves,
                 "X": words(x[rows], torch.uint16), "W": host_w.view(torch.uint8)[cols].tolist(),
                 "WS": words(host_s[[c//128 for c in cols]], torch.uint32),
                 "output": words(y[rows][:, cols], torch.uint32)}
        (out / f"{name}.inputs.json").write_text(json.dumps(probe))
        print(f"{name}: batch 1/64 and BF16 root passed" + (", physical TP4 passed" if leaves == 4 else ""), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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
        for name in PROJECTIONS:
            check(name, load, rank, args.output_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
