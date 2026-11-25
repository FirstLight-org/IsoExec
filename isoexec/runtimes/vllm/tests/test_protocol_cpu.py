"""Native embedding, position and Mamba queries use the existing model state."""

import pytest
import torch

from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.source import ModelSource
from isoexec.ops import gdn
from isoexec.ops.gdn.shared import gdn_ops
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import _gdn_fixtures, test_definition_cpu

qwen_config = pytest.fixture(test_weights_cpu.qwen_config)
qwen_stage = test_weights_cpu.qwen_stage
hf_config = test_definition_cpu.hf_config
production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


def test_embed_input_ids_calls_the_registered_embedding_without_replacing_weights(
    qwen_stage, hf_config, vllm_config, monkeypatch,
):
    model = VllmModel(qwen_stage,
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)
    embedding = model.body.get_submodule("embed")
    weight = embedding.weight
    ids = torch.tensor([3, 5, 7], dtype=torch.int32)
    hidden = torch.empty(3, 16, dtype=torch.bfloat16, device="meta")
    received = []

    def forward(input_ids):
        received.append(input_ids)
        return hidden

    monkeypatch.setattr(embedding, "forward", forward)

    assert model.embed_input_ids(ids) is hidden
    assert torch.equal(received[0], ids)
    assert embedding.weight is model.head.weight is weight


def test_text_mrope_positions_return_three_rows_and_reject_multimodal_features(
    qwen_stage, hf_config, vllm_config,
):
    model = VllmModel(qwen_stage,
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)

    positions, delta = model.get_mrope_input_positions([3, 5, 7], [])

    assert torch.equal(positions, torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    assert positions.dtype == torch.int64 and delta == 0
    with pytest.raises(ValueError, match="text-only"):
        model.get_mrope_input_positions([3], [object()])


def test_mamba_class_queries_match_registered_gdn_pages(qwen_config, qwen_stage, hf_config, vllm_config, monkeypatch):
    vllm_config.model_config.hf_text_config = qwen_config.get_text_config()
    shape = VllmModel.get_mamba_state_shape_from_config(vllm_config)
    dtype = VllmModel.get_mamba_state_dtype_from_config(vllm_config)
    copies = VllmModel.get_mamba_state_copy_func()

    assert shape == ((3, 48), (4, 4, 4))
    assert dtype == (torch.bfloat16, torch.float32)
    assert copies == (_gdn_fixtures.conv_copy, _gdn_fixtures.ssm_copy)
    model = VllmModel(qwen_stage,
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)
    layer = model.body.get_submodule("layers.0.mixer.core")
    assert layer.get_state_shape() == shape and layer.get_state_dtype() == dtype
    monkeypatch.setattr(gdn_ops, "gdn_cpr_min_pages", lambda: True)
    monkeypatch.setitem(gdn.__dict__, "gdn_cpr_min_pages", lambda: True)
    # Minimal pages apply only to a private-pool provider; native-state GDN (the sm90 default) keeps its
    # real state in these pages, so the class query and the layer both stay at full size.
    from isoexec.runtimes.vllm._gdn import _selected_needs_native_cache

    want = shape if _selected_needs_native_cache() else ((1, 8), (1, 1, 4))
    assert VllmModel.get_mamba_state_shape_from_config(vllm_config) == layer.get_state_shape() == want
