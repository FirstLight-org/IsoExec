"""Qwen3 dense: released geometry, checkpoint key coverage, refusals and parity with Transformers."""

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFModel

from isoexec.compiler.realization.weights import _conversion_mapping, _find_mapping, ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.integrations.verl.model import PACKAGES
from isoexec.models.qwen3.model import Qwen3DecoderLayer, Qwen3ForCausalLM
from isoexec.models.qwen3.tests.hf_parity import Harness, hf_logits, plain_gamma_read_as_zero_centred, randomize
from isoexec.nn import MLP, Attention, MoE
from isoexec.nn.inspect import inventory

SOURCE = ModelSource("isoexec.models.qwen3")
TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=256, bos_token_id=None, eos_token_id=None)


def configuration(tiny=True):
    raw = json.loads((Path(__file__).parent / "fixtures/qwen3_8b.json").read_text())
    if tiny:
        raw.update(TINY)
    return raw


def hf_shapes(raw):
    with torch.device("meta"):
        hf = HFModel(Qwen3Config.from_dict(raw))
    return {name: tuple(tensor.shape) for name, tensor in hf.state_dict().items()}


def test_released_geometry_and_hf_key_coverage():
    raw = configuration(False)
    assert PACKAGES[raw["model_type"]] == SOURCE.package
    model = SOURCE.build_meta(raw)
    assert isinstance(model, Qwen3ForCausalLM) and len(model.layers) == 36
    assert all(p.is_meta for p in model.parameters())
    block = model.layers[0]
    assert isinstance(block, Qwen3DecoderLayer) and isinstance(block.mlp, MLP) and not isinstance(block.mlp, MoE)
    attn = block.mixer
    assert isinstance(attn, Attention)
    assert (attn.heads, attn.kv_heads, attn.head_dim) == (32, 8, 128)
    assert attn.qk_norm == "rms" and attn.out_gate is None and not attn.qkv_bias
    assert attn.rope_theta == 1e6 and attn.rotary_fraction == 1.0 and attn.rope_style == "neox"
    assert attn.mrope_section is None and attn.eps == 1e-6
    assert all(norm.convention == "rms" for norm in (block.attn_norm, block.mlp_norm, model.norm))
    assert model.embed.vocab == model.head.padded_vocab == model.head.vocab == 151936
    assert not model.head.tied
    names = inventory(model)
    assert names["layers.0.mixer.q"].shape == (4096, 4096)
    assert names["layers.0.mixer.k"].shape == (1024, 4096)
    assert names["layers.0.mixer.q_norm"].shape == (128,)
    assert names["layers.35.mlp.down"].shape == (4096, 12288)
    assert names["head.weight"].shape == (151936, 4096)
    config = SOURCE.build_config(raw)
    renamings, converters = _conversion_mapping(SOURCE, config)
    shapes = hf_shapes(raw)
    assert len(shapes) == 399, "the released checkpoint's key count"
    mapped = {name for key in shapes for name in _find_mapping(key, renamings, converters)[2]}
    assert mapped == set(dict(model.named_parameters()))
    assert ignored_patterns(config) == ()


def test_tied_embeddings_share_the_table_and_ignore_a_shipped_head():
    raw = configuration()
    raw["tie_word_embeddings"] = True
    model = SOURCE.build_meta(raw)
    assert model.head.tied and model.head.weight is model.embed.weight
    assert ignored_patterns(SOURCE.build_config(raw)) == model._keys_to_ignore_on_load_unexpected


@pytest.mark.parametrize("key,value", [("use_sliding_window", True), ("attention_bias", True), ("hidden_act", "gelu"),
                                       ("rope_scaling", {"rope_type": "yarn", "factor": 4.0,
                                                         "original_max_position_embeddings": 32768})])
def test_unsupported_settings_refuse(key, value):
    raw = configuration()
    raw[key] = value
    with pytest.raises(ValueError):
        SOURCE.build_meta(raw)


def test_logits_match_transformers_and_the_plain_gamma_convention_matters():
    raw = configuration()
    config = SOURCE.build_config(raw)
    hf = HFModel(config).bfloat16().eval()
    randomize(hf)
    harness = Harness(SOURCE, raw, hf.state_dict())
    assert harness.loaded == set(dict(harness.model.named_parameters()))
    lengths = [7, 12, 1]
    ids = torch.randint(0, 256, (sum(lengths),), generator=torch.Generator().manual_seed(1))
    expected = hf_logits(hf, ids, lengths)
    actual = harness.logits(ids, lengths)
    assert actual.shape == expected.shape and actual.dtype == torch.float32
    # HF rounds every intermediate to bf16 (including rms(x) before * gamma); IsoExec keeps fp32
    # inside each op and rounds once, so the two agree to bf16 noise, not bitwise.
    error = (actual - expected).abs().max().item()
    assert error < 0.1 * expected.abs().max().item(), error
    undo = plain_gamma_read_as_zero_centred(harness.model)
    try:
        wrong = (harness.logits(ids, lengths) - expected).abs().max().item()
    finally:
        undo()
    assert wrong > 10 * error, (wrong, error)
