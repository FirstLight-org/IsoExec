"""Qwen3 MoE: released geometry, per-expert checkpoint coverage, refusals, routing and parity with
Transformers."""

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM as HFModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from isoexec.compiler.realization.weights import _conversion_mapping, _find_mapping, ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.integrations.verl.model import PACKAGES
from isoexec.models.qwen3.tests.hf_parity import Harness, hf_logits, plain_gamma_read_as_zero_centred, randomize
from isoexec.models.qwen3_moe.model import Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM
from isoexec.nn import Attention, MoE
from isoexec.nn.inspect import inventory

SOURCE = ModelSource("isoexec.models.qwen3_moe")
TINY = dict(hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256, num_experts=8,
            num_experts_per_tok=2, bos_token_id=None, eos_token_id=None)


def configuration(tiny=True):
    raw = json.loads((Path(__file__).parent / "fixtures/qwen3_30b_a3b.json").read_text())
    if tiny:
        raw.update(TINY)
    return raw


def checkpoint_state(hf_state):
    """Released per-expert keys from Transformers' fused expert storage."""
    state = {}
    for name, tensor in hf_state.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            gate, up = tensor.chunk(2, dim=1)
            for index in range(tensor.shape[0]):
                state[f"{prefix}{index}.gate_proj.weight"] = gate[index]
                state[f"{prefix}{index}.up_proj.weight"] = up[index]
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for index in range(tensor.shape[0]):
                state[f"{prefix}{index}.down_proj.weight"] = tensor[index]
        else:
            state[name] = tensor
    return state


def test_released_geometry_and_per_expert_checkpoint_coverage():
    raw = configuration(False)
    assert PACKAGES[raw["model_type"]] == SOURCE.package
    model = SOURCE.build_meta(raw)
    assert isinstance(model, Qwen3MoeForCausalLM) and len(model.layers) == 48
    assert all(p.is_meta for p in model.parameters())
    block = model.layers[0]
    assert isinstance(block, Qwen3MoeDecoderLayer)
    attn, mlp = block.mixer, block.mlp
    assert isinstance(attn, Attention) and isinstance(mlp, MoE)
    assert (attn.heads, attn.kv_heads, attn.head_dim) == (32, 4, 128)
    assert attn.qk_norm == "rms" and attn.out_gate is None and not attn.qkv_bias and attn.rope_theta == 1e6
    assert (mlp.num_experts, mlp.top_k, mlp.intermediate) == (128, 8, 768)
    assert mlp.routing == "softmax_renorm" and mlp.weight_after_fc2 and mlp.shared is None
    assert all(isinstance(layer.mlp, MoE) for layer in model.layers)
    assert model.embed.vocab == model.head.vocab == 151936 and not model.head.tied
    names = inventory(model)
    assert names["layers.0.mixer.q"].shape == (4096, 2048)
    assert names["layers.0.mixer.k"].shape == (512, 2048)
    assert names["layers.47.mlp.experts.127.down"].shape == (2048, 768)
    assert names["layers.0.mlp.router.weight"].shape == (128, 2048)
    config = SOURCE.build_config(raw)
    renamings, converters = _conversion_mapping(SOURCE, config)
    with torch.device("meta"):
        hf = HFModel(Qwen3MoeConfig.from_dict(raw))
    keys = checkpoint_state(hf.state_dict())
    assert len(keys) == 18867, "the released checkpoint's key count"
    mapped = {name for key in keys for name in _find_mapping(key, renamings, converters)[2]}
    assert mapped == set(dict(model.named_parameters()))
    assert ignored_patterns(config) == ()


@pytest.mark.parametrize("key,value", [("norm_topk_prob", False), ("decoder_sparse_step", 2), ("mlp_only_layers", [0]),
                                       ("use_sliding_window", True), ("attention_bias", True)])
def test_unsupported_settings_refuse(key, value):
    raw = configuration()
    raw[key] = value
    with pytest.raises(ValueError):
        SOURCE.build_meta(raw)


def test_softmax_renorm_selects_and_weights_as_transformers_router():
    """HF: softmax over all experts in fp32, top-k, divide by the selected sum. IsoExec: stable top-k on
    the fp32 logits, softmax over the selected. Same experts, same weights (to fp32 rounding)."""
    from isoexec.ops.moe.routing import route
    from isoexec.schema import Mlp

    config = Qwen3MoeConfig.from_dict(configuration(False))
    router = Qwen3MoeTopKRouter(config)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        router.weight.copy_(torch.randn(router.weight.shape, generator=generator) * 0.05)
    hidden = torch.randn(257, config.hidden_size, generator=generator)
    logits, weights, indices = router(hidden)
    spec = Mlp(intermediate_size=config.moe_intermediate_size, num_experts=config.num_experts,
               top_k=config.num_experts_per_tok, routing="softmax_renorm")
    result = route(logits.float(), spec)
    assert torch.equal(result.indices.long(), indices)
    torch.testing.assert_close(result.probs, weights.float(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result.probs.sum(-1), torch.ones(257), rtol=1e-6, atol=1e-6)


def test_logits_match_transformers_and_the_plain_gamma_convention_matters():
    raw = configuration()
    config = SOURCE.build_config(raw)
    hf = HFModel(config).bfloat16().eval()
    randomize(hf, seed=3)
    harness = Harness(SOURCE, raw, checkpoint_state(hf.state_dict()))
    assert harness.loaded == set(dict(harness.model.named_parameters()))
    lengths = [9, 5, 14]
    ids = torch.randint(0, 256, (sum(lengths),), generator=torch.Generator().manual_seed(2))
    expected = hf_logits(hf, ids, lengths)
    actual = harness.logits(ids, lengths)
    assert actual.shape == expected.shape
    error = (actual - expected).abs().max().item()
    assert error < 0.1 * expected.abs().max().item(), error
    undo = plain_gamma_read_as_zero_centred(harness.model)
    try:
        wrong = (harness.logits(ids, lengths) - expected).abs().max().item()
    finally:
        undo()
    assert wrong > 10 * error, (wrong, error)


@pytest.mark.parametrize("side", ["trainer", "engine"])
@pytest.mark.parametrize("tp,ep", [(8, False), (8, True), (2, True)])
def test_rank_slices_at_tp8_with_kv_heads_below_tp_and_ep(side, tp, ep):
    """TP8 over 4 KV heads (the 30B-A3B ratio: below the head granule) and EP over 8 experts: every
    rank's registered slice loads from the released per-expert keys and equals the declared slice."""
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.realization.graph import clone_graph_module
    from isoexec.compiler.realization.storage import allocate_tensors
    from isoexec.compiler.realization.tp import shard_parameter_shapes, slice_weight
    from isoexec.compiler.realization.weights import load_weights
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.text import text_inputs

    raw = configuration()
    raw.update(num_attention_heads=8, num_key_value_heads=4)
    config = SOURCE.build_config(raw)
    author = SOURCE.build_meta(raw).eval()
    examples = text_inputs(author, device="cpu", max_tokens=16, profile="engine" if side == "engine" else "trainer")
    captured = capture(author, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    with torch.device("meta"):
        hf = HFModel(Qwen3MoeConfig.from_dict(raw))
    shapes = checkpoint_state(hf.state_dict())
    weights = {name: torch.arange(torch.tensor(t.shape).prod().item()).reshape(t.shape).float().remainder(113).bfloat16()
               for name, t in shapes.items()}
    experts = raw["num_experts"]
    for rank in (0, tp - 1):
        target = clone_graph_module(define_stage(captured, ((0, raw["num_hidden_layers"]),), pp_rank=0).module)
        remote = set()
        if ep:
            per = experts // tp
            keep = range(rank * per, (rank + 1) * per)
            for index in range(raw["num_hidden_layers"]):
                held = target.get_submodule(f"layers.{index}.mlp.experts")
                for expert in range(experts):
                    if expert not in keep:
                        remote |= {f"layers.{index}.mlp.experts.{expert}.{r}" for r in ("gate", "up", "down")}
                        del held._modules[str(expert)]
        etp = 1 if ep else tp
        shard_parameter_shapes(target, tp_size=tp, side=side, etp_size=etp)
        allocate_tensors(target, device=torch.device("cpu"))
        loaded = load_weights(target, weights.items(), source=SOURCE, hf_config=config, tp_size=tp, tp_rank=rank,
                              side=side, etp_size=etp, remote=remote)
        assert loaded == dict(target.named_parameters()).keys()
        for name, key in (("layers.0.mixer.k", "model.layers.0.self_attn.k_proj.weight"),
                          ("layers.1.mixer.q", "model.layers.1.self_attn.q_proj.weight"),
                          ("layers.0.mixer.k_norm", "model.layers.0.self_attn.k_norm.weight")):
            expected = slice_weight(target, name, weights[key], tp_size=tp, tp_rank=rank, side=side, etp_size=etp)
            assert torch.equal(target.get_parameter(name), expected), name
        if side == "engine" and tp > raw["num_key_value_heads"]:  # below the granule: one whole replicated kv head
            assert target.get_parameter("layers.0.mixer.k").shape == (16, raw["hidden_size"])
