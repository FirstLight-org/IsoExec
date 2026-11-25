"""CPU contracts for the Model Runner V2 hooks and verl placement used by the verl integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isoexec.runtimes.vllm import vllm_patches as patches


@pytest.fixture(autouse=True)
def reset_v2_census(monkeypatch):
    monkeypatch.setattr(patches, "_V2_STATE",
                        {"calls": 0, "full_rows": 0, "rowinv_owned": 0, "incumbent": 0, "partial": 0, "reported": 0})


def _full_ids(sampled: torch.Tensor, vocab: int) -> torch.Tensor:
    order = torch.stack([torch.randperm(vocab) for _ in range(sampled.numel())])
    return torch.cat([sampled.view(-1, 1), order], dim=1)


def test_v2_complete_gather_uses_the_full_row_producer():
    logits = torch.randn(3, 11)
    rows = torch.log_softmax(logits.double(), dim=-1).float()
    ids = _full_ids(torch.tensor([1, 4, 9]), 11)
    seen = {}

    def rowinv(value, *, src_dtype, reference, temperature):
        seen.update(value=value, temperature=temperature)
        return rows

    wrapped = patches._make_vllm_v2_token_logprobs(
        lambda *_: pytest.fail("vLLM's kernel must not run for a complete gather"), rowinv,
        lambda value: value, lambda: None)
    out = wrapped(logits, ids)
    assert torch.equal(out, rows.gather(-1, ids))
    assert seen["value"] is logits and seen["temperature"] is None
    assert patches._V2_STATE["rowinv_owned"] == 1 and patches._V2_STATE["full_rows"] == 3


def test_v2_sampled_token_gather_uses_the_full_row_producer():
    """``logprobs=0`` (sampled token only) must come from the trainer's producer too, not vLLM's kernel."""
    logits = torch.randn(2, 8)
    rows = torch.log_softmax(logits.double(), dim=-1).float()
    ids = torch.tensor([[3], [5]])
    wrapped = patches._make_vllm_v2_token_logprobs(
        lambda *_: pytest.fail("vLLM's kernel must not score a sampled-token gather"),
        lambda *a, **k: rows, lambda v: v, lambda: None)
    assert torch.equal(wrapped(logits, ids), rows.gather(-1, ids))
    assert patches._V2_STATE["rowinv_owned"] == 1 and patches._V2_STATE["partial"] == 0


def test_v2_malformed_gather_and_decline_keep_vllm_kernel():
    logits = torch.randn(2, 8)
    calls = []

    def original(value, ids):
        calls.append(ids.shape)
        return torch.zeros(ids.shape)

    wrapped = patches._make_vllm_v2_token_logprobs(original, lambda *a, **k: None, lambda v: v, lambda: None)
    wrapped(logits, torch.zeros(3, 1, dtype=torch.int64))  # row count mismatch: not ours to score
    wrapped(logits, _full_ids(torch.tensor([0, 1]), 8))
    assert calls == [(3, 1), (2, 9)]
    assert patches._V2_STATE["partial"] == 1 and patches._V2_STATE["incumbent"] == 1


def test_v2_installer_replaces_the_module_global_once():
    module = SimpleNamespace(compute_token_logprobs=lambda logits, ids: None)
    assert patches._install_vllm_v2_token_logprobs(module, None, None, None)
    first = module.compute_token_logprobs
    assert patches._install_vllm_v2_token_logprobs(module, None, None, None)
    assert module.compute_token_logprobs is first and first._isoexec_v2_full_row_hook


def test_engine_environment_selects_model_runner_v2():
    env = {}
    patches.apply_vllm_isoexec_env(env)
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    from isoexec.runtimes.environment import resolved_environment

    assert resolved_environment("model")["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_verl_replica_world_size_counts_prefill_context_parallel_ranks():
    from verl.workers.config.rollout import rollout_context_parallel_size, rollout_replica_world_size

    config = SimpleNamespace(tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=2,
                             engine_kwargs={"vllm": {"prefill_context_parallel_size": 2,
                                                      "decode_context_parallel_size": 2}})
    assert rollout_context_parallel_size(config) == 2
    assert rollout_replica_world_size(config) == 4
    config.engine_kwargs = {}
    assert rollout_replica_world_size(config) == 2


def test_gdn_v2_driver_view_reads_batch_order_host_state():
    from isoexec.runtimes.vllm import gdn_engine_patch as gdn

    class Tables:
        redirect_writes_to_null_block = False

        def __init__(self):
            self.block_tables = [SimpleNamespace(gpu=torch.zeros(4, 3, dtype=torch.int32))]
            self.num_blocks = SimpleNamespace(np=np.zeros((1, 4), dtype=np.int32))

        def append_block_ids(self, req_index, new_block_ids, overwrite):
            start = 0 if overwrite else self.num_blocks.np[0, req_index]
            self.num_blocks.np[0, req_index] = start + len(new_block_ids[0])

    Tables.append_block_ids = gdn._mirror_block_ids(Tables.append_block_ids)
    tables = Tables()
    tables.append_block_ids(2, ([7],), overwrite=True)
    tables.append_block_ids(0, ([5, 6],), overwrite=True)
    tables.append_block_ids(2, ([8],), overwrite=False)
    runner = SimpleNamespace(
        block_tables=tables, kv_cache_config="kv", vllm_config="cfg",
        req_states=SimpleNamespace(num_computed_tokens_np=np.array([4, 0, 9, 0], dtype=np.int32),
                                   req_id_to_index={"a": 0, "b": 2}),
    )
    batch = SimpleNamespace(req_ids=["b", "a"], idx_mapping_np=np.array([2, 0]))
    view = gdn._V2DriverView(runner, batch)
    assert view.input_batch.num_reqs == 2 and view.input_batch.req_ids == ["b", "a"]
    assert view.input_batch.num_computed_tokens_cpu.tolist() == [9, 4]
    assert view.input_batch.block_table.block_tables[0].block_table.np.tolist() == [[7, 8, 0], [5, 6, 0]]
    assert view.requests == {"a": 0, "b": 2}
