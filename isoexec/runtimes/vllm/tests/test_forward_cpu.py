"""Native inputs reach the body with the captured sequence contract."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.source import ModelSource
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import test_definition_cpu

qwen_stage = test_weights_cpu.qwen_stage
hf_config = test_definition_cpu.hf_config
production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config

pytestmark = pytest.mark.parametrize("qwen_stage", ["engine"], indirect=True)


@pytest.fixture
def model(qwen_stage, hf_config, vllm_config):
    return VllmModel(qwen_stage,
                     source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                     vllm_config=vllm_config)


@pytest.fixture
def forward_context(monkeypatch):
    context = SimpleNamespace(attn_metadata=None)
    native = ModuleType("vllm.forward_context")
    native.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return context


@pytest.fixture
def body_calls(model, monkeypatch):
    calls = []
    guard = model.body.get_submodule("_isoexec_export_input_guard")

    def forward(input_ids, seq):
        guard((input_ids, seq))
        hidden = torch.empty(input_ids.numel(), 16, dtype=torch.bfloat16)
        calls.append((input_ids, seq, hidden))
        return hidden

    monkeypatch.setattr(model.body, "forward", forward)
    return calls


@pytest.mark.parametrize("kind,decode", [("attention", False), ("attention", True), ("gdn", False), ("gdn", True)])
def test_forward_preserves_native_boundaries_positions_and_body_state(model, forward_context, body_calls, kind, decode):
    graph = str(model.body.graph)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    ids = torch.tensor([4] if decode else [4, 5, 6, 7, 8], dtype=torch.int32)
    positions = torch.tensor([19] if decode else [9, 10, 2, 3, 4])
    cu = torch.tensor([0, 1] if decode else [0, 2, 5], dtype=torch.int32)
    if kind == "attention":
        metadata = SimpleNamespace(query_start_loc=cu, max_query_len=1 if decode else 3, max_seq_len=32)
    else:
        metadata = SimpleNamespace(non_spec_query_start_loc=cu, num_decodes=1 if decode else 0,
                                   num_prefills=0 if decode else 2)
    forward_context.attn_metadata = {"native.layer": metadata}

    output = model(ids, positions)

    actual_ids, seq, hidden = body_calls[-1]
    assert output is hidden
    assert actual_ids is seq.input_ids and actual_ids.dtype == torch.int64
    assert torch.equal(actual_ids, ids.long())
    assert torch.equal(seq.positions, positions.expand(3, -1)) and seq.positions.is_contiguous()
    assert seq.cu_seqlens is cu and seq.cu_seqlens_cpu is None
    assert seq.max_q == (1 if decode else 3)
    assert seq.max_kv == (32 if kind == "attention" else 128)
    assert str(model.body.graph) == graph
    for name, parameter in parameters.items():
        assert model.get_parameter(name) is parameter
    assert model.head.weight is model.body.embed.weight


def test_forward_preserves_all_mrope_rows_and_creates_a_fresh_context(model, forward_context, body_calls):
    ids = torch.tensor([4, 5, 6, 7, 8])
    positions = torch.arange(30).reshape(3, 10)[:, 2:7]
    before = positions.clone()
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    forward_context.attn_metadata = {"native.layer": SimpleNamespace(
        query_start_loc=cu, max_query_len=3, max_seq_len=32,
    )}

    model(ids, positions)
    first = body_calls[-1][1]
    first.rope_freqs = torch.empty(5, 2)
    model(ids, positions)
    second = body_calls[-1][1]

    assert first is not second and second.rope_freqs is None
    assert torch.equal(second.positions, before) and second.positions.is_contiguous()
    assert second.positions.storage_offset() == 0
    assert torch.equal(positions, before)


@pytest.mark.parametrize("metadata", [None, {}])
def test_forward_profile_uses_one_request_without_native_metadata(model, forward_context, body_calls, metadata):
    forward_context.attn_metadata = metadata
    ids, positions = torch.tensor([4, 5, 6]), torch.tensor([0, 1, 2])

    output = model(ids, positions)

    _, seq, hidden = body_calls[-1]
    assert output is hidden
    assert torch.equal(seq.cu_seqlens, torch.tensor([0, 3], dtype=torch.int32))
    assert seq.max_q == seq.max_kv == 3


@pytest.mark.parametrize("invalid", ["pipeline", "embeddings", "missing_ids", "metadata_list", "missing_boundaries", "positions"])
def test_forward_rejects_inputs_outside_the_native_token_path(model, forward_context, body_calls, invalid):
    args = {"input_ids": torch.tensor([4, 5]), "positions": torch.tensor([0, 1])}
    if invalid == "pipeline":
        args["intermediate_tensors"] = object()
    elif invalid == "embeddings":
        args["inputs_embeds"] = torch.empty(2, 16)
    elif invalid == "missing_ids":
        args["input_ids"] = None
    elif invalid == "metadata_list":
        forward_context.attn_metadata = [{}, {}]
    elif invalid == "missing_boundaries":
        forward_context.attn_metadata = {"native.layer": SimpleNamespace()}
    else:
        args["positions"] = torch.tensor([0])

    with pytest.raises(ValueError):
        model(**args)

    assert not body_calls
