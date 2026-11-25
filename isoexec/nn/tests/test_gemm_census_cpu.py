"""The dense GEMM census read off a constructed component tree.

``fixtures/gemm_census_snapshot.json`` freezes, for the shipped model packages at tiny and released
geometries and every ``(tp, etp)`` in {1,2,4,8}^2, the census the earlier config-table derivation
produced at commit 9aced96c. Those shapes drive the GEMM coverage report and the tile tables on both
runtimes, so any drift is a deliberate re-record.
"""

from __future__ import annotations

import json

import pytest
import torch

from isoexec.compiler.source import ModelSource
from isoexec.nn import GDN, LMHead, MLP, Attention, LatentAttention, MoE, gemm_census
from isoexec.ops.mm import mm_shapes

from isoexec.nn.tests import _packages

SNAPSHOT = _packages.cases()


def _rows(sites) -> list:
    return [[s.name, s.K, s.N, str(s.dtype).removeprefix("torch."), s.axis, s.calls, s.table_candidate]
            for s in sites]


@pytest.fixture(scope="module")
def censuses() -> dict:
    return {name: gemm_census(ModelSource(case["package"]).build_meta(case["config"]))
            for name, case in SNAPSHOT.items()}


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_census_matches_the_frozen_row_table_census(censuses, name):
    census = censuses[name]
    for key, want in SNAPSHOT[name]["sites"].items():
        tp, etp = map(int, key.split("/"))
        try:
            got = _rows(census.sites(tp, etp))
        except Exception as error:  # noqa: BLE001 -- refusals are part of the snapshot
            got = f"ERR: {type(error).__name__}: {error}"
        assert got == want, f"{name} tp={tp} etp={etp}"


def test_census_is_a_function_of_package_and_config(censuses):
    """Trainer and engine each construct the model; equal inputs must give equal censuses."""
    case = SNAPSHOT["qwen3_5_moe/tiny"]
    again = gemm_census(ModelSource(case["package"]).build_meta(json.loads(json.dumps(case["config"]))))
    assert again == censuses["qwen3_5_moe/tiny"]
    assert again.digest() == censuses["qwen3_5_moe/tiny"].digest()
    assert censuses["qwen3_5/tiny"].digest() != censuses["qwen3_5_moe/tiny"].digest()


def test_released_moe_sites_and_union(censuses):
    sites = {s.name: s for s in censuses["qwen3_5_moe/released_35b_a3b"].sites(1, 1)}
    assert sites["moe.router"].dtype is torch.float32 and sites["moe.router"].axis == "none"
    assert (sites["shexp.gate"].K, sites["shexp.gate"].N) == (2048, 1)
    assert (sites["lm_head"].N, sites["lm_head"].calls) == (248320, 1)
    assert not {"mlp.fc1", "moe.fc1", "attn.o", "gdn.out_proj"} & set(sites)  # row-parallel, routed: absent
    union = mm_shapes.census_union(censuses["qwen3_5_moe/released_35b_a3b"], (1, 2, 4, 8), (1, 2, 4, 8))
    assert union[(2048, 9216 // 2, torch.bfloat16)] == "attn.qkv@tp2/etp1"


class _Toy(torch.nn.Module):
    """Every censused component kind in one test-local tree."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention(hidden=32, heads=4, kv_heads=2, head_dim=8, rope_theta=1e4, out_gate=None, qk_norm=None)
        self.gdn = GDN(hidden=32, key_heads=2, value_heads=4, key_head_dim=8, value_head_dim=8, conv_kernel=4)
        self.mla = LatentAttention(hidden=32, heads=6, q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8,
                                   qk_rope_head_dim=4, v_head_dim=8, rope_theta=1e4)
        self.moe = MoE(hidden=32, num_experts=4, top_k=2, intermediate=8, shared_intermediate=8,
                       shared_expert_gate=False)
        self.mlp = MLP(hidden=32, intermediate=24)
        self.head = LMHead(vocab=100, hidden=32, padded_vocab=128)


def test_every_component_kind_contributes_its_sites():
    census = gemm_census(_Toy())
    got = {(s.name, s.K, s.N, s.axis) for s in census.sites(2, 2)}
    assert got == {
        ("attn.qkv", 32, (4 + 2 * 2) * 8 // 2, "tp"),
        ("gdn.in_proj_qkvzba", 32, (2 * 16 + 2 * 32 + 2 * 4) // 2, "tp"),
        ("mla.q_down", 32, 16, "none"),
        ("mla.q_up", 16, 6 * 12 // 2, "tp"),
        ("mla.kv_down", 32, 8 + 4, "none"),
        ("moe.router", 32, 4, "none"),
        ("shexp.fc1", 32, 2 * 8 // 2, "tp"),
        ("mlp.fc1", 32, 2 * 24 // 2, "tp"),
        ("lm_head", 32, 128 // 2, "tp"),
    }
    with pytest.raises(ValueError, match="heads must be divisible by TP=4"):
        census.sites(4, 4)
