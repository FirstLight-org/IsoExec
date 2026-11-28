"""Replicated-parameter digests: the digest itself, the copy identity, the comparison, and a real 4-rank EP2 model
(each expert shard on two ranks) over gloo, equal and with one drifted copy."""
import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isoexec.runtimes.megatron import replicas


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.int8])
def test_fold64_sees_every_single_bit_and_ignores_chunking(dtype, monkeypatch):
    base = torch.randn(1031).to(dtype)
    digest = replicas.tensor_digest(base)
    monkeypatch.setattr(replicas, "_CHUNK_WORDS", 7)
    assert replicas.tensor_digest(base) == digest
    raw = base.view(torch.uint8).clone()
    for position in (0, 1, 513, raw.numel() - 1):
        for bit in (0, 7):
            flipped = raw.clone()
            flipped[position] ^= 1 << bit
            assert replicas.tensor_digest(flipped.view(dtype)) != digest, (position, bit)


def test_digest_of_an_unaligned_view_equals_the_copy_and_sha256_is_hashlib():
    buffer = torch.randn(64).bfloat16()
    view = buffer[1:30]                       # byte offset 2: not a whole 32-bit word
    assert replicas.tensor_digest(view) == replicas.tensor_digest(view.clone())
    assert replicas.tensor_digest(view, "sha256") == hashlib.sha256(view.clone().view(torch.uint8).numpy().tobytes()).hexdigest()
    with pytest.raises(ValueError, match="digest must be one of"):
        replicas.tensor_digest(view, "md5")


def test_replica_key_names_full_copies_by_name_and_shards_by_tp_coordinate():
    assert replicas.replica_key("a.w", (4, 8), (4, 8), tp_rank=1, tp_size=2) == "a.w"
    assert replicas.replica_key("a.w", (2, 8), (4, 8), tp_rank=1, tp_size=2) == "a.w@tp1/2"
    assert replicas.replica_key("a.w", (4, 8), (4, 8), tp_rank=0, tp_size=2, dp_coordinate=3) == "a.w@dp3"


def _entries(values):
    return [(key, key, torch.nn.Parameter(value, requires_grad=False)) for key, value in values.items()]


def test_comparison_names_the_first_differing_tensor_and_the_ranks_of_each_value():
    norm, shard = torch.ones(4), torch.arange(6.0)
    ranks = [{"norm": norm, "w@tp0/2": shard}, {"norm": norm, "w@tp1/2": shard + 1},
             {"norm": norm, "w@tp0/2": shard}, {"norm": norm.clone(), "w@tp1/2": shard + 1}]
    digests = [replicas.local_digests(_entries(r)) for r in ranks]
    verdict = replicas.compare_replicas(digests)
    assert verdict["equal"] and verdict["holders"] == {2: 2, 4: 1} and verdict["replicated_keys"] == 3
    ranks[3]["norm"] = norm.clone()
    ranks[3]["norm"][2] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    digests = [replicas.local_digests(_entries(r)) for r in ranks]
    with pytest.raises(replicas.ReplicaDivergence, match=r"step 2: replicated param norm differs across its 4 "
                                                         r"holders: ranks \[0, 1, 2\] .*; ranks \[3\]"):
        replicas.check_replicas(None, entries=_entries(ranks[0]), gather=lambda mine: digests, label="step 2")
    verdict = replicas.check_replicas(None, entries=_entries(ranks[0]), gather=lambda mine: digests,
                                      raise_on_mismatch=False)
    assert [m["key"] for m in verdict["mismatches"]] == ["norm"]


def test_main_grad_digests_skip_parameters_without_one():
    with_grad, frozen = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3), requires_grad=False)
    with_grad.main_grad = torch.full((3,), 0.5)
    got = replicas.local_digests([("a", "a", with_grad), ("b", "b", frozen)], what="main_grad")
    assert list(got) == ["a"] and got["a"][1] == "torch.float32"


def test_check_after_optimizer_step_is_off_without_a_multi_rank_group(monkeypatch):
    monkeypatch.setattr(replicas, "check_replicas", lambda *a, **k: pytest.fail("no check on one rank"))
    assert replicas.check_after_optimizer_step([object()]) is None


def _ep2_world4(rank, rendezvous):
    from datetime import timedelta

    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch import distributed as dist
    from torch import fx, nn
    from transformers import PretrainedConfig

    import isoexec.nn as inn
    from isoexec.compiler.partition import StageDefinition
    from isoexec.compiler.source import ModelSource
    from isoexec.ops import _dispatch
    from isoexec.runtimes.megatron import model as native

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        _dispatch.ARCH = "sm90"
        singles = [dist.new_group([i]) for i in range(4)]
        ep_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]          # EP over consecutive ranks
        edp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]         # expert data-parallel replicas
        single = singles[rank]
        group = SimpleNamespace(tp=single, pp=single, cp=single, expt_tp=single, dp=dist.group.WORLD,
                                dp_cp=dist.group.WORLD, ep=ep_groups[rank // 2], expt_dp=edp_groups[rank % 2])
        root, graph = nn.Module(), fx.Graph()
        root.moe = inn.MoE(hidden=8, num_experts=4, top_k=2, intermediate=6, shared_intermediate=0)
        graph.output(graph.call_module("moe", (graph.placeholder("x"),)))
        allocate = native.allocate_tensors
        native.allocate_tensors = lambda module, **kwargs: allocate(module, device=torch.device("cpu"))
        torch.cuda.current_device = lambda: 0
        config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1, num_moe_experts=4,
                                   moe_router_topk=2, expert_model_parallel_size=2, add_bias_linear=False,
                                   bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16)
        model = native.MegatronModel(StageDefinition(fx.GraphModule(root, graph), None),
                                     source=ModelSource("isoexec.models.qwen2_moe"), hf_config=PretrainedConfig(),
                                     config=config, pg_collection=group)
        ep_rank = rank % 2
        with torch.no_grad():
            model.graph_module.moe.router.weight.fill_(0.5)
            for name, expert in model.graph_module.moe.experts.named_children():
                for k, parameter in enumerate(expert.parameters()):
                    parameter.fill_(10 * int(name) + k)
        entries = replicas.model_entries(model)
        keys = sorted(key for key, _, _ in entries)
        assert keys == sorted(["moe.router.weight"] + [f"moe.experts.{e}.{r}" for e in (2 * ep_rank, 2 * ep_rank + 1)
                                                       for r in ("gate", "up", "down")])
        verdict = replicas.check_replicas(model, kind="sha256")
        # router on all four ranks; each of the 12 expert tensors on its two expert-DP replicas
        assert verdict["equal"] and verdict["holders"] == {2: 12, 4: 1}, verdict
        recompute = replicas.check_replicas(model, scope="recompute")
        assert recompute["equal"] and recompute["holders"] == {1: 28}, recompute   # TP1: every rank its own tokens
        if rank == 3:                                  # one copy of expert 3's down projection drifts by one ulp
            with torch.no_grad():
                down = model.graph_module.moe.experts.get_submodule("3").down
                down.view(torch.int16)[0, 0] += 1
        with pytest.raises(replicas.ReplicaDivergence, match=r"moe\.experts\.3\.down differs across its 2 holders: "
                                                             r"ranks \[1\] .*; ranks \[3\]"):
            replicas.check_replicas(model, label="step 1")
    finally:
        dist.destroy_process_group()


def test_ep2_over_four_ranks_compares_each_expert_shard_across_its_replicas(tmp_path):
    torch.multiprocessing.spawn(_ep2_world4, args=((tmp_path / "ep").as_uri(),), nprocs=4, join=True)
