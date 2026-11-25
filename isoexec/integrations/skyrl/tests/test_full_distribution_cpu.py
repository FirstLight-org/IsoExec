from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isoexec.checks import Refusal
from isoexec.integrations.skyrl import scoring


@pytest.fixture(autouse=True)
def full_logprobs(monkeypatch):
    def compute(logits, temperature):
        assert temperature == 1.0
        return logits.contiguous(), torch.tensor(1.0, dtype=torch.float32)

    producer = Mock(side_effect=compute)
    monkeypatch.setattr("isoexec.runtimes.trainer_installs.compute_full_logprobs_for_check", producer)
    return producer


def _case(monkeypatch):
    monkeypatch.setattr(scoring, "_find_isoexec_stage", lambda actor_module: SimpleNamespace())
    wrapper = SimpleNamespace(actor_module=[object()])
    sequences = torch.tensor(
        [
            [10, 11, 12, 1, 2],
            [0, 0, 20, 21, 2],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.float32)
    selected_rows = torch.tensor(
        [
            [-3.0, -1.0, -2.0],
            [-4.0, -3.0, -1.0],
            [-5.0, -4.0, -2.0],
        ],
        dtype=torch.float32,
    )
    logits = torch.zeros((1, 8, 3), dtype=torch.float32)
    logits[0, torch.tensor([2, 3, 6])] = selected_rows
    rollout_rows = torch.zeros((2, 2, 3), dtype=torch.float32)
    rollout_rows[mask > 0] = selected_rows
    rollout_logprobs = torch.zeros((2, 2), dtype=torch.float32)
    sampled_tokens = sequences[:, -2:][mask > 0]
    rollout_logprobs[mask > 0] = selected_rows.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    data = {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "rollout_full_logprobs": rollout_rows,
        "loss_mask": mask,
        "rollout_logprobs": rollout_logprobs,
        "sub_seq_lengths_list": None,
    }
    packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 5, 8], dtype=torch.int32))
    return wrapper, logits, data, packed


def test_combined_skyrl_mapping_uses_isoexec_full_row_comparator(monkeypatch, full_logprobs):
    wrapper, logits, data, packed = _case(monkeypatch)

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3
    assert torch.equal(full_logprobs.call_args.args[0], data["rollout_full_logprobs"][data["loss_mask"] > 0])


def test_combined_skyrl_mapping_accepts_empty_selected_microbatch(monkeypatch):
    wrapper, logits, data, packed = _case(monkeypatch)
    data["loss_mask"].zero_()

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 0


def test_combined_skyrl_mapping_rejects_one_bit_row_difference(monkeypatch):
    wrapper, logits, data, packed = _case(monkeypatch)
    data["rollout_full_logprobs"][0, 1, 0] = torch.nextafter(
        data["rollout_full_logprobs"][0, 1, 0],
        torch.tensor(0.0),
    )

    with pytest.raises(Refusal, match="full_distribution.bits"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})


# --- on-disk evidence (ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR) -------------------------------------


def _evidence_env(monkeypatch, tmp_path):
    import numpy as np

    from isoexec.integrations.skyrl import evidence

    monkeypatch.setattr(evidence.FullDistributionEvidence, "_instance", None)
    monkeypatch.setenv(evidence.ENV_VAR, str(tmp_path / "rows"))
    return evidence, np


def _read_events(path):
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_evidence_sink_records_exact_rows_keys_and_alignment(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _case(monkeypatch)

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3

    rank_dir = tmp_path / "rows" / "rank0000"
    saved = np.load(rank_dir / "rows-000000.npz")
    selected = data["rollout_full_logprobs"][data["loss_mask"] > 0].numpy()
    assert saved["trainer"].dtype == np.float32 and saved["engine"].dtype == np.float32
    assert (saved["trainer"].view(np.int32) == selected.view(np.int32)).all()
    assert (saved["engine"].view(np.int32) == selected.view(np.int32)).all()
    # row 0 of sample 0 predicts token 1 from prefix [10, 11, 12]; row 1 predicts 2 from [10, 11, 12, 1];
    # sample 1 (left padded) predicts 2 from the real prefix [20, 21].
    assert saved["sample_index"].tolist() == [0, 0, 1]
    assert saved["response_token_index"].tolist() == [0, 1, 1]
    assert saved["token_id"].tolist() == [1, 2, 2]
    assert saved["prefix_length"].tolist() == [3, 4, 2]
    assert saved["prefix_digest"].tolist() == [
        evidence.prefix_digest([10, 11, 12]),
        evidence.prefix_digest([10, 11, 12, 1]),
        evidence.prefix_digest([20, 21]),
    ]
    assert (saved["sampled_logprob_bits"] == saved["engine_sampled_bits"]).all()
    assert (saved["trainer_sampled_bits"] == saved["engine_sampled_bits"]).all()

    events = _read_events(rank_dir / "events.jsonl")
    assert [e["event"] for e in events] == ["compare"]
    assert events[0]["rows"] == 3 and events[0]["vocab_size"] == 3
    assert events[0]["first_mismatch"] is None
    assert events[0]["sampled_bits_agree"] == 3
    assert events[0]["sample_index_scope"] == "local"
    assert events[0]["temperature_float32_bits"] == 0x3F800000
    assert events[0]["context"] == {"global_step": None, "weight_version": None, "num_actions": None}


def test_evidence_sink_saves_rows_and_first_mismatch_before_refusing(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _case(monkeypatch)
    data["global_step"] = 4
    data["rollout_full_logprobs"][0, 1, 0] = torch.nextafter(
        data["rollout_full_logprobs"][0, 1, 0],
        torch.tensor(0.0),
    )

    with pytest.raises(Refusal, match="full_distribution.bits: row=1 token=0 .* mismatch_count=1"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})

    rank_dir = tmp_path / "rows" / "rank0000"
    saved = np.load(rank_dir / "rows-000000.npz")
    assert saved["trainer"].shape == (3, 3)
    assert (saved["trainer"].view(np.int32) != saved["engine"].view(np.int32)).sum() == 1
    events = _read_events(rank_dir / "events.jsonl")
    assert [e["event"] for e in events] == ["compare", "refusal"]
    assert events[0]["first_mismatch"]["selected_index"] == 1
    assert events[0]["first_mismatch"]["first_token"] == 0
    assert events[0]["first_mismatch"]["mismatch_count"] == 1
    assert events[0]["context"]["global_step"] == 4
    assert events[1]["message"].startswith("full_distribution.bits: row=1 token=0")


def test_evidence_sink_uses_forwarded_global_sample_indices(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _case(monkeypatch)
    data["sample_indices"] = torch.tensor([7, 9], dtype=torch.long)

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3

    rank_dir = tmp_path / "rows" / "rank0000"
    saved = np.load(rank_dir / "rows-000000.npz")
    assert saved["sample_index"].tolist() == [7, 7, 9]
    assert _read_events(rank_dir / "events.jsonl")[0]["sample_index_scope"] == "global"


def test_evidence_sink_records_early_refusals(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _case(monkeypatch)
    data["rollout_full_logprobs"] = None

    with pytest.raises(Refusal, match="rollout rows, sampled logprobs, and loss mask are required"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})

    events = _read_events(tmp_path / "rows" / "rank0000" / "events.jsonl")
    assert [e["event"] for e in events] == ["refusal"]
    assert not list((tmp_path / "rows" / "rank0000").glob("rows-*.npz"))


def test_evidence_sink_is_inert_without_the_environment_variable(monkeypatch, tmp_path):
    from isoexec.integrations.skyrl import evidence

    monkeypatch.setattr(evidence.FullDistributionEvidence, "_instance", None)
    monkeypatch.delenv(evidence.ENV_VAR, raising=False)
    wrapper, logits, data, packed = _case(monkeypatch)

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3
    assert not list(tmp_path.iterdir())


# --- tensor-parallel vocabulary slices ------------------------------------------------------------


class _ShardStage:
    """A fake last stage that owns vocabulary slice [start, end) of V=3, like one TP rank."""

    def __init__(self, start, end):
        self.start, self.end = start, end

    def compute_full_logprobs_for_check_shard(self, logits, temperature):
        assert temperature == 1.0
        return logits[:, self.start : self.end].contiguous(), torch.tensor(1.0), (self.start, self.end)


def _shard_case(monkeypatch, start, end):
    wrapper, logits, data, packed = _case(monkeypatch)
    monkeypatch.setattr(scoring, "_find_isoexec_stage", lambda actor_module: _ShardStage(start, end))
    return wrapper, logits, data, packed


def test_tp_rank_compares_only_its_vocabulary_slice(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _shard_case(monkeypatch, 0, 2)

    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3

    saved = np.load(tmp_path / "rows" / "rank0000" / "rows-000000.npz")
    assert saved["trainer"].shape == (3, 2) and saved["engine"].shape == (3, 2)
    assert int(saved["vocab_start"]) == 0 and int(saved["vocab_end"]) == 2
    # tokens 1, 2, 2 are sampled: token 1 is owned by this slice, token 2 is not.
    assert saved["sampled_owner"].tolist() == [True, False, False]
    assert saved["trainer_sampled_bits"].tolist()[1:] == [0xFFFFFFFF, 0xFFFFFFFF]
    assert saved["trainer_sampled_bits"][0] == saved["engine_sampled_bits"][0]
    events = _read_events(tmp_path / "rows" / "rank0000" / "events.jsonl")
    assert events[0]["vocab_slice"] == [0, 2] and events[0]["vocab_size"] == 2


def test_tp_rank_ignores_a_divergence_outside_its_slice_and_refuses_one_inside(monkeypatch, tmp_path):
    evidence, np = _evidence_env(monkeypatch, tmp_path)
    wrapper, logits, data, packed = _shard_case(monkeypatch, 0, 2)
    # Vocabulary entry 2 belongs to the other rank's slice [2, 3): this rank must not see it.
    data["rollout_full_logprobs"][0, 1, 2] = torch.nextafter(data["rollout_full_logprobs"][0, 1, 2], torch.tensor(0.0))
    assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3

    wrapper, logits, data, packed = _shard_case(monkeypatch, 2, 3)
    data["rollout_full_logprobs"][0, 1, 2] = torch.nextafter(data["rollout_full_logprobs"][0, 1, 2], torch.tensor(0.0))
    with pytest.raises(Refusal, match="full_distribution.bits: row=1 token=2 .* mismatch_count=1"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})


def test_tp_slice_must_fit_the_engine_vocabulary(monkeypatch):
    wrapper, logits, data, packed = _shard_case(monkeypatch, 1, 5)
    with pytest.raises(Refusal, match="vocabulary slice"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})


def _native_tp_case(monkeypatch, rank):
    """A native last stage on TP rank ``rank`` of 2 over V=4; its logits are that rank's two columns."""
    wrapper, logits, data, packed = _case(monkeypatch)
    logits = torch.cat((logits, torch.full((*logits.shape[:2], 1), -6.0)), dim=-1)
    data["rollout_full_logprobs"] = torch.cat((data["rollout_full_logprobs"], torch.full((2, 2, 1), -6.0)), dim=-1)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    monkeypatch.setattr(scoring, "_find_isoexec_stage",
                        lambda actor_module: SimpleNamespace(pg_collection=SimpleNamespace(tp=group)))
    return wrapper, logits[..., 2 * rank : 2 * rank + 2], data, packed, group


def test_native_tp_rank_checks_its_vocabulary_columns(monkeypatch):
    producer = Mock(side_effect=lambda logits, temperature, **kwargs: (logits.contiguous(), torch.tensor(1.0)))
    monkeypatch.setattr("isoexec.runtimes.trainer_installs.compute_full_logprobs_for_check_shard", producer)
    # Sampled tokens are 1, 2, 2, so columns 0 and 3 are checked only by the full-row comparison.
    for rank, own, other in ((0, 0, 3), (1, 3, 0)):
        wrapper, logits, data, packed, group = _native_tp_case(monkeypatch, rank)
        assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3
        assert producer.call_args.kwargs == {"vocab_start_index": 2 * rank, "vocab_end_index": 2 * rank + 2,
                                             "group": group}
        rows = data["rollout_full_logprobs"]
        rows[0, 1, other] = torch.nextafter(rows[0, 1, other], torch.tensor(0.0))
        assert scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed}) == 3
        rows[0, 1, own] = torch.nextafter(rows[0, 1, own], torch.tensor(0.0))
        with pytest.raises(Refusal, match=f"full_distribution.bits: row=1 token={own} "):
            scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})


def test_native_nonfinal_pp_stage_refuses_the_check(monkeypatch):
    wrapper, logits, data, packed = _case(monkeypatch)
    monkeypatch.setattr(scoring, "_find_isoexec_stage", lambda actor_module: SimpleNamespace(post_process=False))
    with pytest.raises(Refusal, match="last PP stage"):
        scoring.verify_full_distribution(wrapper, logits, {**data, "packed_seq_params": packed})


# ---- context parallelism: a row belongs to the CP rank that holds its predictor token -------------------


class _CpStage:
    def __init__(self, rank: int):
        # A native stage carries Megatron's process groups; the CP group tells the rank its share of the rows.
        self.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2, rank=lambda: rank))


def _cp_case(monkeypatch, rank: int):
    """Two sequences of 6 and 5 real tokens, padded to 8 each (tp=1, cp=2: multiples of 4), three
    response tokens each. Global logits carry a unique row per packed position; each CP rank is handed
    only its zigzag slice of them."""
    from isoexec.ops.collectives.cp_layout import local_to_global

    monkeypatch.setattr(scoring, "_find_isoexec_stage", lambda actor_module: _CpStage(rank))
    sequences = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 7, 8, 9, 10, 11]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.long)
    mask = torch.ones((2, 3), dtype=torch.float32)
    cu = [0, 8, 16]
    global_logits = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3) / 7.0
    # predictor of (sample s, response col c) sits at packed index cu[s] + (real_len - 3 + c) - 1
    rollout_rows = torch.zeros((2, 3, 3), dtype=torch.float32)
    predictors = {}
    for s, real in enumerate((6, 5)):
        for c in range(3):
            g = cu[s] + (real - 3 + c) - 1
            predictors[(s, c)] = g
            rollout_rows[s, c] = global_logits[g]
    mine = local_to_global(cu, 2, rank)
    data = {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "rollout_full_logprobs": rollout_rows,
        "loss_mask": mask,
        "rollout_logprobs": torch.zeros((2, 3), dtype=torch.float32),
        "sub_seq_lengths_list": None,
    }
    packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor(cu, dtype=torch.int32))
    wrapper = SimpleNamespace(actor_module=[object()])
    owned = sorted(key for key, g in predictors.items() if g in mine)
    return wrapper, global_logits[mine].unsqueeze(0), data, packed, owned


def test_context_parallel_ranks_verify_disjoint_rows_that_tile_the_selection(monkeypatch):
    counts, owned_all = [], []
    for rank in (0, 1):
        wrapper, local_logits, data, packed, owned = _cp_case(monkeypatch, rank)
        counts.append(scoring.verify_full_distribution(wrapper, local_logits, {**data, "packed_seq_params": packed}))
        owned_all.append(owned)
        assert counts[-1] == len(owned)
    assert sum(counts) == 6 and not set(owned_all[0]) & set(owned_all[1])
    assert all(owned_all), "the fixture must give BOTH ranks rows, or the test proves nothing about one of them"


def test_a_context_parallel_rank_refuses_a_flip_in_its_own_row_and_ignores_the_other_ranks(monkeypatch):
    wrapper, local_logits, data, packed, owned = _cp_case(monkeypatch, 1)
    other = _cp_case(monkeypatch, 0)[4]
    monkeypatch.setattr(scoring, "_find_isoexec_stage", lambda actor_module: _CpStage(1))
    s, c = other[0]
    data["rollout_full_logprobs"][s, c, 1] += 1.0  # a row rank 0 owns: rank 1 must not see it
    assert scoring.verify_full_distribution(wrapper, local_logits, {**data, "packed_seq_params": packed}) == len(owned)
    s, c = owned[0]
    data["rollout_full_logprobs"][s, c, 1] += 1.0
    with pytest.raises(Refusal, match="full_distribution.bits"):
        scoring.verify_full_distribution(wrapper, local_logits, {**data, "packed_seq_params": packed})


def test_context_parallel_logits_of_the_wrong_length_refuse(monkeypatch):
    wrapper, local_logits, data, packed, _ = _cp_case(monkeypatch, 0)
    with pytest.raises(Refusal, match="context-parallel layout"):
        scoring.verify_full_distribution(wrapper, local_logits[:, :-1], {**data, "packed_seq_params": packed})
