"""CPU contract of the SM100 DeepSeek-V4.1 BF16-actor recipe declarations."""
from isoexec.ops.moe.deepseek_v41_sm100 import spec


def test_declarations_cpu():
    assert spec.OP == "moe.deepseek_v41" and {i.op for i in spec.IMPLS} == {spec.OP}
    assert spec.ROUTING == "sqrtsoftplus_bias_select" and spec.FC2_LEAVES == 1
    assert all(i.supported_archs == frozenset({"sm100"}) for i in spec.IMPLS)


def test_exchange_pad_covers_every_swapped_token_tile_cpu():
    """``bi_tp`` asks the exchange's h region for ``Tk + tile_n`` rows under a swapped record; the region's headroom must
    cover the widest swapped token tile the vendored generator can produce."""
    from isoexec.ops._vendor.flashinfer_bi_moe.gemm.batch_invariant import generator
    from isoexec.ops.moe.deepseek_v41_sm100 import tp_exchange
    assert tp_exchange.H_PAD_ROWS >= max((*generator.TILES_N_NARROW, generator.SWAP_WIDE_TILE_N))


def test_the_schedule_is_the_schedule_policy_on_the_deployed_model_cpu():
    """Fails if a segment count, a segment length or a merge of the V4.1 MoE differs from
    ``isoexec/ops/schedule_policy.py`` at the shapes of the DeepSeek-V4.1-Flash configuration the tree ships.

    Reductions and rules: router logits over the hidden axis, rule 3 (one segment); FC1 gate / up over the hidden
    axis and FC2 over the intermediate axis, routed and shared, rule 3 (one segment; no axis of a projection is
    sharded in this recipe, so d_max = 1); the renormaliser over the K selected scores and the combine over the
    selected experts' rows, rule 2 (one segment per selected expert, a chain from +0)."""
    import json
    from pathlib import Path

    from isoexec.ops import schedule_policy as P
    from isoexec.ops.moe.moe_sm100 import ops as base
    from isoexec.ops.moe.router_sm100 import ops as router
    from isoexec.ops.moe.router_sm100 import spec as router_spec

    root = Path(__file__).resolve().parents[5]
    cfg = json.loads((root / "isoexec/models/deepseek_v4_1_flash/tests/fixtures/config.json").read_text())["text_config"]
    hidden, inter, experts, top_k = (cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"],
                                     cfg["num_experts_per_tok"])
    assert (hidden, inter, experts, top_k) == (5120, 2304, 384, 6)
    one = P.reduce(hidden, b=16)
    assert (one.G, one.S, one.merge) == (1, hidden, "tree")
    # router logits: every path takes the policy's count
    assert router.declared_segments(hidden) == router_spec.logit_segments(hidden) == one.G
    assert router.mega_gate_segments() == one.G and router.mega_serves(experts, hidden)
    # FC1 (routed and shared): one chain over the hidden axis; FC2: one chain over the intermediate axis
    assert base.projection_schedule(hidden) == one
    fc2 = base.projection_schedule(inter, spec.FC2_LEAVES)
    assert fc2 == P.reduce(inter, b=16) and (fc2.G, fc2.S) == (1, inter)
    assert base.fc2_leaves(inter, spec.FC2_LEAVES) == spec.FC2_LEAVES == 1
    # the renormaliser and the combine: rule 2
    assert router.RENORM_SCHEDULE == base.SELECTED == P.selected() == P.Segments(1, None, "chain")
    assert router_spec.RENORM_FOLD == "ascending_expert_id_chain_from_zero_plus_1e-20"


def test_the_kernel_sources_run_the_policys_schedule_cpu():
    """Order and seed are code, not parameters: the ported sources carry the rule-2 chain (ascending expert id,
    seeded with +0) for the renormaliser and the combine on every path, the fused router instantiates one segment
    only, and nothing of the 4-segment / half-fold / first-as-is schedule is left. The bits are GPU-gated
    (router_relations_gpu.py, test_route_mega_gpu.py, test_actor_mega_gpu.py, test_bi_tp_gpu.py)."""
    import re
    from pathlib import Path

    ops = Path(__file__).resolve().parents[3]
    select = (ops / "moe/router_sm100/csrc/select.cuh").read_text()
    gate = (ops / "_vendor/deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh").read_text()
    backward = (ops / "moe/router_sm100/csrc/backward.cu").read_text()
    router = (ops / "moe/router_sm100/csrc/router.cu").read_text()
    mega = (ops / "_vendor/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh").read_text()
    route = (ops / "moe/moe_sm100/csrc/route.cu").read_text()
    bi_tp = (ops / "moe/deepseek_v41_sm100/bi_tp.py").read_text()
    for src in (select, gate, backward, route):
        assert "halffold" not in src
    for src in (select, gate):
        assert "__fadd_rn(selected_sum<K>(sw[t], win[t]), 1e-20f)" in src
    assert "selected_sum<K>(sw[0], win[0])" in backward
    # one segment: the only fused instantiations are (CS, SL) = (1, 1); the selection from segment roots and the
    # persistent kernel are instantiated at S = 1; the host takes the count as an argument
    pairs = set(re.findall(r"RT_NS\(\d+, [^,]+, (\d+), (\d+),", router))
    assert pairs == {("1", "1")}, pairs
    assert "select_tree_kernel<MODE_, KT, ET, 1>" in router and "persist_kernel<CE_, MODE_, K_, EPL_, 1, 1>" in router
    assert "const int S = int(segments);" in router and "int declared_segments(" not in router
    # the combine
    assert "first_contribution" not in mega
    kernel = route[route.index("void combine_kernel("):][:3200]
    assert kernel.count("acc[i] = 0.f;") == 2 and "acc[i] = v[0][i]" not in kernel and "load(0, acc)" not in kernel
    assert "from_first=True" not in bi_tp
