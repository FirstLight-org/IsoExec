"""Real Qwen1.5 authored dataflow, checkpoint mapping, runtime and auxiliary gradients."""

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM as HFModel

from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.weights import load_weights
from isoexec.compiler.source import ModelSource
from isoexec.models.qwen2_moe.model import (
    Qwen2MoeDecoderLayer as Block,
)
from isoexec.models.qwen2_moe.model import (
    Qwen2MoeForCausalLM as Model,
)
from isoexec.nn.inspect import inventory

SOURCE = ModelSource("isoexec.models.qwen2_moe")


def configuration(*, tiny=True):
    raw = json.loads((Path(__file__).parent / "fixtures/qwen1_5_moe_a2_7b.json").read_text())
    if tiny:
        raw.update(hidden_size=16, vocab_size=32, num_hidden_layers=2, num_attention_heads=2,
                   num_key_value_heads=1, moe_intermediate_size=8, shared_expert_intermediate_size=16,
                   num_experts=4, num_experts_per_tok=2)
    return raw


def checkpoint_state(raw):
    """Restore released per-expert keys from Transformers' fused model storage."""
    with torch.device("meta"):
        reference = HFModel(Qwen2MoeConfig.from_dict(raw))
    state = {}
    for name, tensor in reference.state_dict().items():
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


def test_released_model_and_checkpoint_names():
    model = SOURCE.build_meta(configuration(tiny=False))
    assert isinstance(model, Model)
    assert tuple(inspect.signature(model.forward).parameters) == ("input_ids",)
    assert all(tuple(inspect.signature(layer.forward).parameters) == ("hidden",) for layer in model.layers)
    block = model.layers[0]
    assert isinstance(block, Block)
    assert len(model.layers) == 24
    assert len(block.mlp.experts) == 60
    assert block.mlp.weight_after_fc2
    assert block.mlp.routing == "softmax"
    assert block.attn.qkv_bias and block.attn.qk_norm is None
    assert block.attn.out_gate is None
    names = inventory(model)
    assert names["layers.0.mlp.experts.59.down"].shape == (2048, 1408)
    assert all(parameter.is_meta for parameter in model.parameters())
    # Released per-expert checkpoint names load every registered tensor through the native loader.
    raw = configuration()
    tiny = SOURCE.build_meta(raw)
    allocate_tensors(tiny, device=torch.device("cpu"))
    weights = ((name, torch.zeros(tensor.shape, dtype=tensor.dtype)) for name, tensor in checkpoint_state(raw).items())
    loaded = load_weights(tiny, weights, source=SOURCE, hf_config=SOURCE.build_config(raw))
    assert loaded == {name for name, _ in tiny.named_parameters()}


@pytest.mark.parametrize("key,value", [("norm_topk_prob", True), ("use_sliding_window", True)])
def test_incompatible_config_is_not_silently_changed(key, value):
    raw = configuration()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        SOURCE.build_meta(raw)
