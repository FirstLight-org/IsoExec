"""verl's synchronous PPO trainer with IsoExec's one-update qualification lifecycle.

verl's sync trainer synchronizes W0, generates, rescores, updates once and synchronizes W1, then
stops: it never generates or rescores with W1. This subclass keeps that native order and adds the
missing tail, plus host-side gates that read only compact receipts:

    W0 sync -> every inference worker acks -> rollout -> Megatron rescoring compares every row
    -> coverage gate -> one optimizer update -> W1 sync -> acks -> W1 rollout -> rescoring
    compares every row -> coverage gate -> stop

Selected with ``trainer.v1.trainer_mode=isoexec_sync`` after ``actor_rollout_ref.model.external_lib``
imports this module in the task runner.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

from isoexec.checks import Refusal

from . import full_distribution as fd

TRAINER_MODE = "isoexec_sync"
MULTI_STEP_TRAINER_MODE = "isoexec_sync_multistep"


def validate_trainer_config(config: Any) -> None:
    """The one-update qualification lifecycle; anything else is refused before workers start."""
    trainer, actor = config.trainer, config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    from .rollout import refuse_engine_context_parallel

    refuse_engine_context_parallel(rollout)  # before any worker starts
    required = {
        "trainer.total_training_steps": (trainer.total_training_steps, 1),
        "trainer.val_before_train": (bool(trainer.get("val_before_train", True)), False),
        "trainer.critic_warmup": (trainer.get("critic_warmup", 0), 0),
        "actor_rollout_ref.actor.ppo_epochs": (actor.ppo_epochs, 1),
        "actor_rollout_ref.actor.ppo_mini_batch_size": (actor.ppo_mini_batch_size, config.data.train_batch_size),
        "actor_rollout_ref.rollout.calculate_log_probs": (bool(rollout.calculate_log_probs), True),
        "algorithm.rollout_correction.bypass_mode": (
            bool((config.algorithm.get("rollout_correction") or {}).get("bypass_mode", False)), False),
        "trainer.v1.sync.parameter_sync_step": (
            int(((trainer.v1.get("sync") or {}).get("parameter_sync_step", 1))), 1),
    }
    for name, (value, expected) in required.items():
        if value != expected:
            raise ValueError(f"IsoExec qualification requires {name}={expected!r}, got {value!r}")
    if trainer.get("test_freq", -1) > 0 or trainer.get("save_freq", -1) > 0:
        raise ValueError("IsoExec qualification requires trainer.test_freq<=0 and trainer.save_freq<=0")
    if rollout.n < 1 or config.data.get("gen_batch_size") not in (None, config.data.train_batch_size):
        raise ValueError("IsoExec qualification requires gen_batch_size == train_batch_size")


def validate_multistep_trainer_config(config: Any) -> None:
    """N training steps of native sync GRPO; every step's rollout is rescored and compared."""
    trainer, actor = config.trainer, config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    from .rollout import refuse_engine_context_parallel

    refuse_engine_context_parallel(rollout)  # before any worker starts
    required = {
        "trainer.critic_warmup": (trainer.get("critic_warmup", 0), 0),
        "actor_rollout_ref.rollout.calculate_log_probs": (bool(rollout.calculate_log_probs), True),
        "algorithm.rollout_correction.bypass_mode": (
            bool((config.algorithm.get("rollout_correction") or {}).get("bypass_mode", False)), False),
        "trainer.v1.sync.parameter_sync_step": (
            int(((trainer.v1.get("sync") or {}).get("parameter_sync_step", 1))), 1),
    }
    for name, (value, expected) in required.items():
        if value != expected:
            raise ValueError(f"IsoExec multi-step training requires {name}={expected!r}, got {value!r}")
    if trainer.get("save_freq", -1) > 0:
        raise ValueError("IsoExec multi-step training requires trainer.save_freq<=0 (checkpointing is not qualified)")
    if config.data.train_batch_size % actor.ppo_mini_batch_size:
        raise ValueError("IsoExec multi-step training requires ppo_mini_batch_size to divide train_batch_size")
    if rollout.n < 1 or config.data.get("gen_batch_size") not in (None, config.data.train_batch_size):
        raise ValueError("IsoExec multi-step training requires gen_batch_size == train_batch_size")
    mismatch_policy(config)


#: ``isoexec.on_mismatch``: what a trainer/engine sampled-logprob bit difference does. ``refuse`` (default)
#: stops the run, as qualification requires. ``record`` logs the agreement metrics and continues, for
#: measurement runs that report the bitwise-equal token fraction (e.g. a configuration under evaluation).
MISMATCH_POLICIES = ("refuse", "record")


def mismatch_policy(config: Any) -> str:
    from omegaconf import OmegaConf

    policy = OmegaConf.select(config, "isoexec.on_mismatch", default="refuse")
    if policy not in MISMATCH_POLICIES:
        raise ValueError(f"isoexec.on_mismatch must be one of {MISMATCH_POLICIES}, got {policy!r}")
    return policy


def sampled_logprob_agreement(old_log_probs, rollout_log_probs, response_mask) -> dict[str, Any]:
    """Bitwise agreement of trainer and engine sampled-token logprobs over the response mask."""
    import torch

    mask = response_mask.to(torch.bool)
    trainer = old_log_probs.to(torch.float32)[mask].contiguous()
    engine = rollout_log_probs.to(torch.float32)[mask].contiguous()
    if trainer.shape != engine.shape:
        raise Refusal(f"sampled_logprobs.shape: trainer {tuple(trainer.shape)} engine {tuple(engine.shape)}")
    differ = trainer.view(torch.int32) != engine.view(torch.int32)
    return {"tokens": int(mask.sum()), "samples": int(mask.shape[0]), "mismatched_tokens": int(differ.sum()),
            "max_abs_diff": float((trainer - engine).abs().max()) if trainer.numel() else 0.0}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_acks(root: Path, version: int, expected: int, trainer_ranks: int | None = None) -> dict[str, Any]:
    """Every one of the ``expected`` engine workers acknowledged ``version``, and they and the ``trainer_ranks``
    training workers (default: as many as engine workers, the colocated layout) bound one attention schedule."""
    from .worker_extension import ack_directory

    directory = ack_directory(root, version)
    acks = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))] if directory.exists() else []
    identities = {(ack["replica_rank"], ack["dp_rank"], ack["rank"]) for ack in acks}
    if len(acks) != expected or len(identities) != expected or any(ack["version"] != version for ack in acks):
        raise Refusal(f"weight_sync.acks: version {version} has {len(acks)} worker acks, expected {expected}")
    # The parties that compute bound one attention schedule (the optional target batch among its inputs): every
    # training worker's record against every engine worker's. This process (verl's task runner) computes no
    # attention, so its own environment is not compared. A party launched with another declaration computes
    # other bits and is refused here, at version 0 before the first rollout and again at every later version.
    from .schedule import require_same_schedules

    require_same_schedules(root, acks, expected if trainer_ranks is None else trainer_ranks)
    return {"workers": len(acks), "runner_classes": sorted({ack["runner_class"] for ack in acks}),
            "parameter_sha256": {f"r{a['replica_rank']}d{a['dp_rank']}k{a['rank']}": a["parameter_sha256"] for a in acks}}


def verify_coverage(root: Path, version: int) -> dict[str, Any]:
    """Every published row of ``version`` was compared once per vocabulary slice, and every slice set
    covers the engine vocabulary exactly. Returns compact counts; refuses any gap or mismatch."""
    published: dict[str, dict[str, Any]] = {}
    directory = root / "rows" / f"v{version}"
    for sidecar in sorted(directory.glob("*.json")) if directory.exists() else []:
        record = json.loads(sidecar.read_text())
        published[record["key"]] = record
    if not published:
        raise Refusal(f"full_distribution.coverage: no rows were published at version {version}")
    columns: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    vocab: dict[str, int] = {}
    for path in sorted((root / "receipts").glob("compare-rank*.jsonl")):
        for event in _jsonl(path):
            if event.get("event") != "compare" or event["version"] != version:
                continue
            if event["first_bad_row"] is not None:
                raise Refusal(f"full_distribution.bits: rank {event['rank']} recorded a mismatch at version {version}")
            start, end = event["vocab_slice"]
            columns[(event["key"], start, end)].extend(event["response_columns"])
            vocab[event["key"]] = event["engine_vocab"]
    compared_keys = {key for key, _, _ in columns}
    if compared_keys != set(published):
        missing, extra = sorted(set(published) - compared_keys), sorted(compared_keys - set(published))
        raise Refusal(f"full_distribution.coverage: version {version} missing {len(missing)} samples, "
                      f"unexpected {len(extra)}: {missing[:3]} {extra[:3]}")
    rows = samples = 0
    for key, record in published.items():
        slices = sorted((start, end) for k, start, end in columns if k == key)
        width = record["shape"][1]
        if vocab[key] != width or slices[0][0] != 0 or slices[-1][1] != width or any(
                previous[1] != current[0] for previous, current in zip(slices, slices[1:])):
            raise Refusal(f"full_distribution.coverage: sample {key} vocabulary slices {slices} do not tile [0, {width})")
        # Identical prompt+completion requests share one key; each must be compared once per slice.
        request_ids = record.get("request_ids", [])
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise Refusal(f"full_distribution.coverage: sample {key} has invalid request_ids {request_ids!r}")
        count = len(request_ids)
        expected = Counter({column: count for column in range(record["response_len"])})
        for start, end in slices:
            got = Counter(columns[(key, start, end)])
            if got != expected:
                raise Refusal(f"full_distribution.coverage: sample {key} slice [{start}, {end}) compared "
                              f"{sum(got.values())} rows; expected each of {record['response_len']} response rows "
                              f"exactly {count} time(s)")
        samples += count
        rows += record["response_len"] * count
    return {"samples": samples, "rows": rows, "vocab": sorted({r["shape"][1] for r in published.values()})}


def verify_optimizer_steps(root: Path, expected_ranks: int, first: int, last: int) -> dict[str, Any]:
    """Every trainer rank recorded exactly the native updates ``first..last`` (1-based, inclusive)."""
    events = [event for path in sorted((root / "receipts").glob("optimizer-rank*.jsonl")) for event in _jsonl(path)
              if first <= event["step"] <= last]
    per_rank = Counter(event["rank"] for event in events)
    expected = last - first + 1
    if len(per_rank) != expected_ranks or any(count != expected for count in per_rank.values()):
        raise Refusal(f"optimizer: expected updates {first}..{last} on each of {expected_ranks} ranks, "
                      f"found {dict(per_rank)}")
    if not all(event["changed"] for event in events):
        raise Refusal("optimizer: a trainer rank's local parameters did not change across an update")
    return {"ranks": len(per_rank), "updates": expected,
            "grad_norm": sorted({event["grad_norm"] for event in events if event["step"] == last})}


def verify_optimizer(root: Path, expected_ranks: int) -> dict[str, Any]:
    events = [event for path in sorted((root / "receipts").glob("optimizer-rank*.jsonl")) for event in _jsonl(path)]
    ranks = {event["rank"] for event in events}
    if len(events) != expected_ranks or len(ranks) != expected_ranks or any(event["step"] != 1 for event in events):
        raise Refusal(f"optimizer: expected exactly one native update on each of {expected_ranks} ranks, "
                      f"found {len(events)} events on {len(ranks)} ranks")
    if not all(event["changed"] for event in events):
        raise Refusal("optimizer: a trainer rank's local parameters did not change across the update")
    return {"ranks": len(ranks), "grad_norm": sorted({event["grad_norm"] for event in events})}


def learning_signal(scores, advantages, response_mask, uids) -> dict[str, Any]:
    """Per-group rewards and advantage size of the batch the one optimizer update consumes.

    GRPO normalizes rewards within each prompt group, so groups with identical rewards contribute
    zero advantages; with the reference equal to the actor at step 0 the KL term is zero too.
    """
    import torch

    mask = response_mask.to(torch.bool)
    # verl's outcome reward is the sum of the token-level scores (the reward sits on the last token).
    rewards = scores.to(torch.float32).sum(dim=-1)
    valid_advantages = advantages.to(torch.float32).masked_select(mask)
    if not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(valid_advantages).all()):
        raise Refusal("learning_signal: non-finite rewards or advantages before the optimizer update")
    groups: dict[str, list[float]] = defaultdict(list)
    for uid, reward in zip(uids, rewards.tolist(), strict=True):
        groups[str(uid)].append(reward)
    return {
        "samples": int(rewards.numel()), "groups": len(groups), "group_rewards": dict(groups),
        "groups_with_reward_spread": sum(len(set(values)) > 1 for values in groups.values()),
        "reward_mean": float(rewards.mean()) if rewards.numel() else 0.0,
        "advantage_abs_max": float(valid_advantages.abs().max()) if valid_advantages.numel() else 0.0,
    }


def batch_learning_signal(padded) -> dict[str, Any]:
    """learning_signal over a padded TensorDict read back from TransferQueue."""
    # Read uid as verl's own _compute_advantage does: pop returns the NonTensorStack; indexing returns
    # its raw LinkedList, which has no tolist().
    uids = padded.pop("uid").tolist()
    return learning_signal(padded["rm_scores"], padded["advantages"], padded["response_mask"], uids)


@register_trainer(TRAINER_MODE)
class IsoExecPPOTrainerSync(PPOTrainerSync):
    """Native sync GRPO plus the W1 rollout, rescoring and receipt gates."""

    def __init__(self, config):
        from omegaconf import open_dict

        validate_trainer_config(config)
        # verl chose this class by name; its sync branches (replay buffer, dataloader refill, weight-sync
        # cadence) test trainer_mode == "sync". Present that mode so the native sync paths apply unchanged.
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config=config)
        if self.trainer_mode != "sync":
            raise RuntimeError("IsoExec trainer requires verl's native sync trainer mode")
        self._isoexec_version = 0
        self._isoexec_summary: dict[str, Any] = {}

    @property
    def _isoexec_root(self) -> Path:
        return fd.run_root()

    def _isoexec_expected_workers(self) -> int:
        from .rollout import expected_workers
        from verl.workers.config import RolloutConfig  # noqa: F401  (validated config type)
        from verl.utils.config import omega_conf_to_dataclass

        rollout = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout)
        per_replica = expected_workers(rollout)
        world = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        if world % per_replica:
            raise Refusal(f"rollout replica size {per_replica} does not divide the trainer world {world}")
        return world

    def _isoexec_record(self, name: str, payload: dict[str, Any]) -> None:
        self._isoexec_summary[name] = payload
        fd.append_receipt(self._isoexec_root, "lifecycle.jsonl", {"event": name, **payload})

    def on_init_end(self):
        super().on_init_end()
        self._isoexec_record("w0_sync", verify_acks(self._isoexec_root, 0, self._isoexec_expected_workers()))

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        version = self._isoexec_version
        coverage = verify_coverage(self._isoexec_root, version)
        self._isoexec_record(f"w{version}_compare", coverage)
        shutil.rmtree(self._isoexec_root / "rows" / f"v{version}")
        metrics[f"isoexec/w{version}_rows_compared"] = coverage["rows"]
        metrics[f"isoexec/w{version}_samples_compared"] = coverage["samples"]
        return batch

    def _compute_advantage(self, batch, metrics):
        import transfer_queue as tq
        from verl import DataProto

        batch = super()._compute_advantage(batch, metrics)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["uid", "response_mask", "rm_scores", "advantages"])
        signal = batch_learning_signal(DataProto(batch=data.to_padded_tensor()).batch)
        self._isoexec_record("learning_signal", signal)
        metrics["isoexec/groups_with_reward_spread"] = signal["groups_with_reward_spread"]
        if signal["advantage_abs_max"] == 0.0:
            raise Refusal("learning_signal: all advantages selected by response_mask are zero; this qualification "
                          f"requires a nonzero GRPO advantage before its single update ({signal})")
        return batch

    def on_step_end(self):
        root = self._isoexec_root
        world = self._isoexec_expected_workers()
        optimizer = verify_optimizer(root, world)
        self._isoexec_record("optimizer", optimizer)
        super().on_step_end()  # W1 sync
        self._isoexec_version = 1
        w1 = verify_acks(root, 1, world)
        w0 = self._isoexec_summary["w0_sync"]
        unchanged = [name for name, digest in w1["parameter_sha256"].items() if w0["parameter_sha256"].get(name) == digest]
        if unchanged:
            raise Refusal(f"weight_sync: inference workers {unchanged} hold the same parameters at W0 and W1")
        self._isoexec_record("w1_sync", w1)
        probe_metrics: dict[str, Any] = {}
        self._isoexec_w1_rollout(probe_metrics)
        metrics = {
            "isoexec/optimizer_updates": 1,
            "isoexec/grad_norm": optimizer["grad_norm"][0],
            "isoexec/w0_rows_compared": self._isoexec_summary["w0_compare"]["rows"],
            "isoexec/w1_rows_compared": self._isoexec_summary["w1_compare"]["rows"],
            "isoexec/inference_workers_acked": world,
            **{key: value for key, value in probe_metrics.items() if key.startswith("isoexec/")},
        }
        self._pending_sync_metrics = {**(getattr(self, "_pending_sync_metrics", None) or {}), **metrics}
        self._isoexec_record("complete", {"passed": True, "metrics": metrics})

    def _isoexec_w1_rollout(self, metrics: dict[str, Any]) -> None:
        """Generate once with W1 and rescore it; no reward, advantage or update follows."""
        import transfer_queue as tq

        self._add_batch_to_generate()
        batch, _ = self.replay_buffer.sample(
            global_steps=self.global_steps, partition_id="train", batch_size=self.config.data.train_batch_size,
        )
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self.checkpoint_manager.sleep_replicas()
        batch = self._balance_batch(batch, metrics=metrics)
        batch = self._compute_old_log_prob(batch, metrics)
        tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)


@register_trainer(MULTI_STEP_TRAINER_MODE)
class IsoExecPPOTrainerSyncMultiStep(IsoExecPPOTrainerSync):
    """Native sync GRPO for N steps; each step's W_k rollout is rescored against full rows.

    Per step k (rollout with weight version k-1): rescoring compares every published row, then the
    native updates run, the W_k sync is acknowledged by every inference worker and must change their
    parameters. Validation generations publish no rows (nothing rescores them).
    """

    def __init__(self, config):  # noqa: D107 - same as the parent, with the N-step validator
        from omegaconf import open_dict

        validate_multistep_trainer_config(config)
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        PPOTrainerSync.__init__(self, config=config)
        if self.trainer_mode != "sync":
            raise RuntimeError("IsoExec trainer requires verl's native sync trainer mode")
        self._isoexec_version = 0
        self._isoexec_summary: dict[str, Any] = {}
        data, actor = config.data, config.actor_rollout_ref.actor
        self._isoexec_updates_per_step = (data.train_batch_size // actor.ppo_mini_batch_size) * actor.ppo_epochs
        self._isoexec_updates_done = 0

    def _compute_old_log_prob(self, batch, metrics):
        from .full_distribution import full_rows_enabled

        batch.extra_info["isoexec_weight_version"] = self._isoexec_version
        if full_rows_enabled():  # debug qualification: every FP32 vocabulary row, through the row exchange
            return super()._compute_old_log_prob(batch, metrics)
        import transfer_queue as tq
        from verl import DataProto

        batch = PPOTrainerSync._compute_old_log_prob(self, batch, metrics)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["old_log_probs", "rollout_log_probs", "response_mask"])
        padded = DataProto(batch=data.to_padded_tensor()).batch
        agreement = sampled_logprob_agreement(padded["old_log_probs"], padded["rollout_log_probs"],
                                              padded["response_mask"])
        self._isoexec_record(f"w{self._isoexec_version}_compare", {**agreement, "rows": agreement["tokens"]})
        metrics["isoexec/sampled_tokens_compared"] = agreement["tokens"]
        metrics["isoexec/sampled_tokens_mismatched"] = agreement["mismatched_tokens"]
        metrics["isoexec/sampled_tokens_max_abs_diff"] = agreement["max_abs_diff"]
        if agreement["mismatched_tokens"]:
            if mismatch_policy(self.config) == "refuse":
                raise Refusal(f"sampled_logprobs.bits: weight version {self._isoexec_version}: {agreement}")
            print(f"[ISOEXEC-VERL] isoexec.on_mismatch=record: weight version {self._isoexec_version}: {agreement}",
                  flush=True)
        return batch

    def _compute_advantage(self, batch, metrics):
        import transfer_queue as tq
        from verl import DataProto

        batch = PPOTrainerSync._compute_advantage(self, batch, metrics)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["uid", "response_mask", "rm_scores", "advantages"])
        signal = batch_learning_signal(DataProto(batch=data.to_padded_tensor()).batch)
        signal.pop("group_rewards")
        self._isoexec_record(f"w{self._isoexec_version}_learning_signal", signal)
        metrics["isoexec/groups_with_reward_spread"] = signal["groups_with_reward_spread"]
        return batch

    def on_step_end(self):
        root = self._isoexec_root
        world = self._isoexec_expected_workers()
        first = self._isoexec_updates_done + 1
        last = self._isoexec_updates_done + self._isoexec_updates_per_step
        optimizer = verify_optimizer_steps(root, world, first, last)
        self._isoexec_updates_done = last
        previous = self._isoexec_version
        PPOTrainerSync.on_step_end(self)  # native W_k sync
        self._isoexec_version = previous + 1
        acks = verify_acks(root, self._isoexec_version, world)
        before = self._isoexec_summary["w0_sync" if previous == 0 else f"w{previous}_sync"]
        unchanged = [name for name, digest in acks["parameter_sha256"].items()
                     if before["parameter_sha256"].get(name) == digest]
        if unchanged:
            raise Refusal(f"weight_sync: inference workers {unchanged} hold the same parameters at "
                          f"W{previous} and W{self._isoexec_version}")
        self._isoexec_record(f"w{self._isoexec_version}_sync", acks)
        compare = self._isoexec_summary[f"w{previous}_compare"]  # full rows or sampled tokens
        self._pending_sync_metrics = {
            **(getattr(self, "_pending_sync_metrics", None) or {}),
            "isoexec/weight_version": self._isoexec_version,
            "isoexec/optimizer_updates": last,
            "isoexec/rows_compared": compare["rows"],
            "isoexec/samples_compared": compare.get("samples", 0),
            "isoexec/inference_workers_acked": world,
        }
        self._isoexec_record("step_complete", {"weight_version": self._isoexec_version, "updates": last,
                                               "grad_norm": optimizer["grad_norm"]})

    def on_validate_begin(self):
        (self._isoexec_root / fd.SUSPEND_ROWS).parent.mkdir(parents=True, exist_ok=True)
        (self._isoexec_root / fd.SUSPEND_ROWS).touch()
        super().on_validate_begin()

    def on_validate_end(self):
        super().on_validate_end()
        (self._isoexec_root / fd.SUSPEND_ROWS).unlink(missing_ok=True)
        stray = self._isoexec_root / "rows" / f"v{self._isoexec_version}"
        if stray.exists():
            raise Refusal(f"full_distribution: validation at weight version {self._isoexec_version} published rows")


__all__ = ["IsoExecPPOTrainerSync", "IsoExecPPOTrainerSyncMultiStep", "MISMATCH_POLICIES", "MULTI_STEP_TRAINER_MODE", "TRAINER_MODE", "batch_learning_signal", "learning_signal", "validate_trainer_config", "verify_acks",
           "verify_coverage", "verify_optimizer", "verify_optimizer_steps", "validate_multistep_trainer_config", "mismatch_policy"]
