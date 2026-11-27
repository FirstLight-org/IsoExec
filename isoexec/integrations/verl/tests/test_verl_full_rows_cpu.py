"""CPU contracts for verl full-row exchange, comparison, slot selection and lifecycle gates."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from isoexec.checks import Refusal
from isoexec.integrations.verl import full_distribution as fd


def _rows(n: int, vocab: int, seed: int = 0) -> np.ndarray:
    logits = torch.randn(n, vocab, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.log_softmax(logits, dim=-1).float().numpy().astype("<f4")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(fd.ENV_VAR, str(tmp_path))
    return fd.run_root()


def test_sample_key_separates_prompt_and_response():
    assert fd.sample_key([1, 2], [3]) != fd.sample_key([1], [2, 3])
    assert fd.sample_key([1, 2], [3]) == fd.sample_key([1, 2], [3])


def test_rows_round_trip_by_token_identity_and_version(root):
    rows = _rows(3, 64)
    ref = fd.write_rows(root, version=0, prompt_ids=[5, 6], response_ids=[7, 8, 9], rows=rows, request_id="r0")
    assert ref["shape"] == [3, 64] and ref["prompt_len"] == 2
    loaded = fd.read_rows(root, version=0, prompt_ids=[5, 6], response_ids=[7, 8, 9])
    assert loaded.tobytes() == rows.tobytes()
    with pytest.raises(fd.RowExchangeRefusal, match="version 1"):
        fd.read_rows(root, version=1, prompt_ids=[5, 6], response_ids=[7, 8, 9])
    with pytest.raises(fd.RowExchangeRefusal):
        fd.read_rows(root, version=0, prompt_ids=[5], response_ids=[6, 7, 8])


def test_republishing_a_key_requires_identical_bytes(root):
    rows = _rows(2, 32)
    fd.write_rows(root, version=0, prompt_ids=[1], response_ids=[2, 3], rows=rows, request_id="a")
    fd.write_rows(root, version=0, prompt_ids=[1], response_ids=[2, 3], rows=rows.copy(), request_id="b")
    sidecar = json.loads(fd.rows_path(root, 0, fd.sample_key([1], [2, 3])).with_suffix(".json").read_text())
    assert sidecar["request_ids"] == ["a", "b"]
    changed = rows.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(0))
    with pytest.raises(fd.RowExchangeRefusal, match="different bytes"):
        fd.write_rows(root, version=0, prompt_ids=[1], response_ids=[2, 3], rows=changed, request_id="c")


def test_write_refuses_incomplete_or_non_fp32_rows(root):
    with pytest.raises(fd.RowExchangeRefusal):
        fd.write_rows(root, version=0, prompt_ids=[1], response_ids=[2, 3], rows=_rows(1, 8), request_id="x")
    with pytest.raises(fd.RowExchangeRefusal):
        fd.write_rows(root, version=0, prompt_ids=[1], response_ids=[2], rows=_rows(1, 8).astype(np.float64),
                      request_id="x")


def test_row_stats_accepts_distributions_and_refuses_degenerate_rows():
    stats = fd.row_stats(torch.from_numpy(_rows(4, 128)))
    assert stats.rows == 4 and stats.max_abs_logsumexp <= fd.LOGSUMEXP_TOLERANCE
    assert stats.min_distinct_non_integer >= fd.MIN_DISTINCT_NON_INTEGER
    unnormalized = torch.from_numpy(_rows(2, 128)) + 1.0
    with pytest.raises(Refusal, match="logsumexp"):
        fd.row_stats(unnormalized)
    uniform = torch.full((1, 128), float(np.log(1 / 128)), dtype=torch.float32)
    with pytest.raises(Refusal, match="distinct"):
        fd.row_stats(uniform)
    with pytest.raises(Refusal):
        fd.row_stats(torch.from_numpy(_rows(1, 128)).double())


def test_compare_slices_reports_the_first_differing_bit():
    engine = torch.from_numpy(_rows(3, 64))
    trainer = engine.clone()
    assert fd.compare_slices(trainer, engine, vocab_start=32).first_bad_row is None
    trainer[1, 5] = torch.nextafter(trainer[1, 5], torch.tensor(0.0))
    result = fd.compare_slices(trainer, engine, vocab_start=32)
    assert (result.first_bad_row, result.first_token, result.mismatch_count) == (1, 37, 1)
    trainer[2, 0] = trainer[2, 0] + 0.5
    result = fd.compare_slices(trainer, engine, vocab_start=32)
    assert result.mismatched_rows == 2 and result.max_abs_diff == pytest.approx(0.5)


def test_select_response_slots_maps_packed_predictors_to_response_columns():
    from isoexec.integrations.verl.engine import SAMPLE_STRIDE, select_response_slots

    lengths, responses = [5, 3], [2, 1]  # prompts of 3 and 2 tokens
    codes = []
    for sample, length in enumerate(lengths):
        codes += [sample * SAMPLE_STRIDE + position + 1 for position in range(length)]
        codes += [0] * (2 - length % 2 if length % 2 else 0)  # TP=2 padding
    slots, samples, columns = select_response_slots(torch.tensor(codes, dtype=torch.int64), lengths, responses)
    # Sample 0: positions 2 and 3 predict response tokens at positions 3 and 4 (columns 0, 1).
    # Sample 1: position 1 predicts its only response token at position 2 (column 0).
    assert slots.tolist() == [2, 3, 7]
    assert samples.tolist() == [0, 0, 1]
    assert columns.tolist() == [0, 1, 0]


def _publish(root, version, prompt, response, vocab=16):
    return fd.write_rows(root, version=version, prompt_ids=prompt, response_ids=response,
                         rows=_rows(len(response), vocab, seed=len(prompt)), request_id="r")


def _compare_event(root, rank, key, version, columns, vocab_slice, engine_vocab=16, bad=None):
    fd.append_receipt(root, f"compare-rank{rank:04d}.jsonl", {
        "event": "compare", "version": version, "key": key, "response_columns": columns,
        "vocab_slice": list(vocab_slice), "engine_vocab": engine_vocab, "first_bad_row": bad, "rank": rank,
    })


def test_coverage_requires_every_row_once_per_tiling_vocabulary_slice(root):
    from isoexec.integrations.verl.trainer import verify_coverage

    ref = _publish(root, 0, [1, 2], [3, 4, 5])
    _compare_event(root, 0, ref["key"], 0, [0, 1, 2], (0, 8))
    with pytest.raises(Refusal, match="tile"):
        verify_coverage(root, 0)
    _compare_event(root, 1, ref["key"], 0, [0, 2], (8, 16))
    with pytest.raises(Refusal, match="exactly 1 time"):
        verify_coverage(root, 0)
    _compare_event(root, 1, ref["key"], 0, [1], (8, 16))
    assert verify_coverage(root, 0) == {"samples": 1, "rows": 3, "vocab": [16]}


def test_coverage_refuses_unscored_samples_and_mismatches(root):
    from isoexec.integrations.verl.trainer import verify_coverage

    first = _publish(root, 1, [1], [2])
    _publish(root, 1, [9], [8])
    _compare_event(root, 0, first["key"], 1, [0], (0, 16))
    with pytest.raises(Refusal, match="missing 1"):
        verify_coverage(root, 1)
    _compare_event(root, 2, first["key"], 1, [0], (0, 16), bad=0)
    with pytest.raises(Refusal, match="mismatch"):
        verify_coverage(root, 1)


def test_acks_and_optimizer_gates(root):
    from isoexec.integrations.verl.trainer import verify_acks, verify_optimizer
    from isoexec.integrations.verl.worker_extension import ack_directory

    from isoexec.ops.attention.target_batch import schedule_record

    directory = ack_directory(root, 0)
    directory.mkdir(parents=True)

    def ack(rank, **changed):
        (directory / f"replica0-dp0-rank{rank}.json").write_text(json.dumps({
            "replica_rank": 0, "dp_rank": 0, "rank": rank, "version": 0,
            "runner_class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner", "parameter_sha256": f"h{rank}",
            "attention_schedule": schedule_record(), **changed,
        }))

    for rank in range(2):
        ack(rank)
    # the training workers have not recorded the schedule they bound: the engines cannot be shown to match them
    with pytest.raises(Refusal, match="attention_schedule.missing: 2 training workers"):
        verify_acks(root, 0, 2)
    from isoexec.integrations.verl.schedule import record_trainer_schedule

    for rank in range(2):
        record_trainer_schedule(root, rank)
    assert verify_acks(root, 0, 2)["workers"] == 2
    with pytest.raises(Refusal):
        verify_acks(root, 0, 4)
    # an engine worker that bound another attention schedule (another target batch) is refused at the sync
    other = schedule_record()
    other["mla_sm100"] = {**other["mla_sm100"], "target_batch": 64, "seg_tokens": 5120}
    ack(1, attention_schedule=other)
    with pytest.raises(Refusal, match="attention_schedule.mla_sm100.seg_tokens.*rank 1"):
        verify_acks(root, 0, 2)
    ack(1, attention_schedule=None)
    with pytest.raises(Refusal, match="attention_schedule.missing"):
        verify_acks(root, 0, 2)
    ack(1)
    for rank in range(2):
        fd.append_receipt(root, f"optimizer-rank{rank:04d}.jsonl", {
            "event": "optimizer_step", "step": 1, "grad_norm": 1.5, "changed": True, "rank": rank,
        })
    assert verify_optimizer(root, 2) == {"ranks": 2, "grad_norm": [1.5]}
    fd.append_receipt(root, "optimizer-rank0000.jsonl", {
        "event": "optimizer_step", "step": 2, "grad_norm": 1.0, "changed": True, "rank": 0,
    })
    with pytest.raises(Refusal, match="exactly one"):
        verify_optimizer(root, 2)


def test_coverage_counts_identical_samples_once_each(root):
    from isoexec.integrations.verl.trainer import verify_coverage

    rows = _rows(2, 16, seed=7)
    for request in ("a", "b"):
        ref = fd.write_rows(root, version=0, prompt_ids=[1, 2], response_ids=[3, 4], rows=rows, request_id=request)
    _compare_event(root, 0, ref["key"], 0, [0, 1], (0, 16))
    with pytest.raises(Refusal, match="exactly 2 time"):
        verify_coverage(root, 0)
    _compare_event(root, 1, ref["key"], 0, [0, 1], (0, 16))
    assert verify_coverage(root, 0) == {"samples": 2, "rows": 4, "vocab": [16]}


def test_receipt_requires_census_from_every_acknowledged_worker(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    run = tmp_path / "run"
    receipts, acks, logs = run / "evidence" / "receipts", run / "evidence" / "acks" / "v1", run / "logs"
    for directory in (receipts, acks, logs):
        directory.mkdir(parents=True)
    (receipts / "terminal.json").write_text(json.dumps({"exit_code": 0}))
    for rank in range(4):
        (acks / f"replica0-dp0-rank{rank}.json").write_text(json.dumps({"pid": 100 + rank, "pp_rank": 0, "version": 1}))
    lines = [f"[ISOEXEC-V2-CUDAGRAPH] pid=100 captures=3 replays=8",
             f"[ISOEXEC-SAMPLER-LOGPROBS-V2] pid=100 hook: calls=4 full_rows=9 rowinv_owned=4 incumbent=0 partial=0"]
    (logs / "main_ppo.log").write_text("\n".join(lines))
    script = Path(__file__).resolve().parents[4] / "examples" / "verl_mrv2" / "collect_receipt.py"
    out = tmp_path / "receipt.json"
    subprocess.run([sys.executable, str(script), "--run-dir", str(run), "--case", "2", "--out", str(out)], check=True,
                   capture_output=True)
    receipt = json.loads(out.read_text())
    assert receipt["passed"] is False
    assert receipt["items"]["cuda_graph_capture_replay"]["status"] == "missing"
    assert receipt["items"]["cuda_graph_capture_replay"]["detail"]["missing_pids"] == ["101", "102", "103"]


def test_learning_signal_reports_groups_without_reward_spread():
    from isoexec.integrations.verl.trainer import learning_signal

    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
    flat = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    none = learning_signal(flat, torch.zeros(4, 3), mask, ["a", "a", "b", "b"])
    assert none["groups"] == 2 and none["groups_with_reward_spread"] == 0 and none["advantage_abs_max"] == 0.0
    scores = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    advantages = torch.tensor([[0.7, 0.7, 9.0], [-0.7, -0.7, -0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    some = learning_signal(scores, advantages, mask, ["a", "a", "b", "b"])
    # Padding columns (mask 0) never count, so the 9.0 is ignored.
    assert some["groups_with_reward_spread"] == 1 and some["advantage_abs_max"] == pytest.approx(0.7)
    assert some["reward_mean"] == pytest.approx(0.25)
    assert some["group_rewards"] == {"a": [1.0, 0.0], "b": [0.0, 0.0]}
    with pytest.raises(Refusal, match="non-finite"):
        learning_signal(scores, advantages.masked_fill(mask.bool(), float("nan")), mask, ["a", "a", "b", "b"])


def test_batch_learning_signal_reads_uid_from_a_padded_tensordict():
    from tensordict import NonTensorData, NonTensorStack, TensorDict

    from isoexec.integrations.verl.trainer import batch_learning_signal

    def nested(rows):
        return torch.nested.nested_tensor([torch.tensor(row, dtype=torch.float32) for row in rows], layout=torch.jagged)

    data = TensorDict({
        "uid": NonTensorStack(*(NonTensorData(uid) for uid in ["a", "a", "b"])),
        "response_mask": nested([[1, 1], [1, 1, 1], [1]]),
        "rm_scores": nested([[0, 1], [0, 0, 0], [0]]),
        "advantages": nested([[0.7, 0.7], [-0.7, -0.7, -0.7], [0]]),
    }, batch_size=[3])
    signal = batch_learning_signal(data.to_padded_tensor())
    assert signal["group_rewards"] == {"a": [1.0, 0.0], "b": [0.0]}
    assert signal["groups_with_reward_spread"] == 1 and signal["advantage_abs_max"] == pytest.approx(0.7)


def test_trainer_rows_use_the_rollout_temperature(monkeypatch):
    """The engine scores logits / T, so the trainer's comparison rows must use the same T, not 1.0."""
    from isoexec.integrations.verl import engine

    seen = []

    class Model:
        def compute_full_logprobs_for_check_shard(self, logits, temperature):
            seen.append(temperature)
            return torch.log_softmax(logits / temperature, dim=-1), temperature, (0, logits.shape[1])

    monkeypatch.setattr(engine, "_unwrap", lambda module: Model())
    logits = torch.randn(3, 8)
    rows = engine._trainer_full_rows(object(), logits, 0, 8, None, 0.8)
    assert seen == [0.8]
    assert torch.equal(rows, torch.log_softmax(logits / 0.8, dim=-1))
