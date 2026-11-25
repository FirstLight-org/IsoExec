"""Real CuMem nesting, slot-map growth, CUDA replay and level-1 sleep regression."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.device_allocator.cumem import CuMemAllocator

from isoexec.ops.gdn import RecurrentGDN, gdn_cpr_state
from isoexec.runtimes.vllm.gdn_slot_map import wrap_cache_initialization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    # The directory holding the ``isoexec`` package: the checkout root, or site-packages when installed.
    source_root = Path(__file__).resolve().parents[4]
    if out.is_relative_to(source_root):
        raise ValueError(f"--out must be outside the source tree {source_root}")
    os.environ["ISOEXEC_GDN_SLOT_MAP_SIZE"] = "4"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    allocator = CuMemAllocator.get_instance()
    owner = torch.nn.Module()
    with allocator.use_memory_pool(tag="weights"):
        for name, shape, dtype in (("conv", (6, 3), torch.bfloat16), ("a", (1,), torch.float32),
                                   ("dt", (1,), torch.float32)):
            owner.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device=device)))
        pool = RecurrentGDN(capacity=2, conv_weight=owner.conv, conv_bias=None, A_log=owner.a, dt_bias=owner.dt,
                            num_k_heads=1, head_k_dim=2, num_v_heads=1, head_v_dim=2, device=device,
                            on_alloc=lambda name, tensor: owner.register_buffer(name, tensor, persistent=False))
    old_row = pool._assign(2)
    pool.flush_slot_map()
    gdn_cpr_state.CPR_LAYERS[:] = [pool]
    runner = SimpleNamespace()
    with allocator.use_memory_pool(tag="kv_cache"):
        initialize = wrap_cache_initialization(
            lambda runner, config, is_profiling=False, kv_cache_allocation_context=None: None
        )
        initialize(runner, SimpleNamespace(num_blocks=1024))
    new_row = pool._assign(700)
    pool.flush_slot_map()
    assert pool.slot2row[2].item() == old_row
    last_row = pool._assign(1023)
    pool.flush_slot_map()
    pointer = owner.slot2row.data_ptr()
    tags = [data.tag for base, data in allocator.pointer_to_data.items()
            if base <= pointer < base + data.handle[1]]
    assert tags == ["weights"], tags
    owner.register_buffer("query", torch.tensor([700, 1023, 1024, 9999], dtype=torch.int64, device=device), persistent=False)
    expected = [new_row, last_row, 0, 0]
    for _ in range(3):
        pool._rows(owner.query)
        pool._rows_pair(owner.query)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pool._rows(owner.query)
        fused_long, fused_int = pool._rows_pair(owner.query)
    owner.register_buffer("result", result, persistent=False)
    owner.register_buffer("fused_long", fused_long, persistent=False)
    owner.register_buffer("fused_int", fused_int, persistent=False)
    graph.replay()
    torch.cuda.synchronize()
    assert owner.result.tolist() == owner.fused_long.tolist() == owner.fused_int.tolist() == expected
    allocator.sleep(offload_tags=("weights",))
    allocator.wake_up()
    graph.replay()
    torch.cuda.synchronize()
    assert owner.result.tolist() == owner.fused_long.tolist() == owner.fused_int.tolist() == expected
    assert owner.slot2row.data_ptr() == pointer
    try:
        pool.finalize_slot_map(2048)
    except RuntimeError:
        pass
    else:
        raise AssertionError("growth after finalization must refuse")
    report = {"GATE": "PASS", **runner._isoexec_slot_map_receipt,
              "allocator_tag": tags[0], "level1_sleep_replay_exact": True, "address_preserved": True,
              "last_valid_and_padding_rows": expected}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
