"""Expand verl's ``isoexec.enable=true`` into every setting the IsoExec integration needs.

verl calls :func:`apply` in the driver before validating its config. One switch selects the three
IsoExec components together, so a run cannot be half IsoExec:

- ``actor_rollout_ref.model.external_lib`` imports :mod:`isoexec.integrations.verl.register` in the
  task runner and every worker (IsoExec Megatron engine, ``isoexec_sync`` trainer);
- ``rollout.agent.custom_async_server`` selects the IsoExec vLLM Model Runner V2 server;
- ``trainer.v1.trainer_mode=isoexec_sync`` selects the one-update qualification lifecycle.

It also sets the settings those components require (raw full-vocabulary logprobs, CUDA graphs on,
recomputed old log probs, no fused kernels). Chunked prefill and prefix caching keep the caller's
values (verl's default: on) when the GDN composition resumes them exactly -- the native recurrent
composition from vLLM's state blocks, CPR from its private pool -- and are forced off otherwise
(speculative decoding, other compositions); see :func:`_state_features`. Under CPR, an enabled feature
turns on the engine flags it needs (``ISOEXEC_GDN_CHUNKED_PREFILL``, ``ISOEXEC_GDN_CPR_APC``), here and
in every Ray worker through ``ray_kwargs.ray_init.runtime_env.env_vars``.
Each value it changes is printed. Parallel layout, model, data, sampling and the lifecycle limits
(``total_training_steps`` and friends, checked by the trainer) stay the caller's.
"""

from __future__ import annotations

import os
from typing import Any

REGISTER = "isoexec.integrations.verl.register"
SERVER_PATH = "pkg://isoexec.integrations.verl.rollout"
SERVER_NAME = "IsoExecvLLMHttpServer"
TRAINER_MODE = "isoexec_sync"
MULTI_STEP_TRAINER_MODE = "isoexec_sync_multistep"

#: Required values by dotted key. Every key exists in verl's ppo_trainer config.
REQUIRED: dict[str, Any] = {
    "actor_rollout_ref.rollout.agent.custom_async_server.path": SERVER_PATH,
    "actor_rollout_ref.rollout.agent.custom_async_server.name": SERVER_NAME,
    "trainer.v1.trainer_mode": TRAINER_MODE,
    "trainer.use_v1": True,
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.calculate_log_probs": True,
    "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
    "actor_rollout_ref.rollout.enforce_eager": False,
    "algorithm.rollout_correction.bypass_mode": False,
    "actor_rollout_ref.model.use_fused_kernels": False,
    "actor_rollout_ref.model.use_remove_padding": True,
    "actor_rollout_ref.actor.megatron.use_mbridge": True,
    "actor_rollout_ref.ref.megatron.use_mbridge": True,
    "trainer.val_before_train": False,
}


def _external_libs(value: Any) -> list[str]:
    if value is None:
        return [REGISTER]
    libs = [str(item) for item in value] if not isinstance(value, str) else [value]
    return libs if REGISTER in libs else [*libs, REGISTER]


def _async_scheduling_safe() -> bool:
    """Whether the rollout engine's GDN state stays exact under vLLM async scheduling.

    Async scheduling prepares step N+1 on the host while step N runs on the device. Both GDN state homes
    are safe under that overlap on Model Runner V2 (the only runner the IsoExec engine runs on):

    * recurrent with native state: the state lives in vLLM's mamba blocks, indexed from vLLM's metadata;
    * CPR with its private pool: the lifecycle driver reads only the scheduler's own host output for the
      step it prepares (request ids, block-table slot ids, scheduled and computed token counts, finished
      and preempted ids, which V2 retires in ``finish_requests`` before that step's claims), never a device
      result; every pool mutation it causes (row claims, slot-map and LRU-stamp uploads, fresh-record
      writes, resyncs inside the decode launch) is enqueued on the compute stream after step N's kernels, so a
      row released while step N still reads it is rewritten only once step N is done.
    """
    from isoexec.core.gdn_kernel_env import gdn_kernel_mode
    from isoexec.ops.gdn.shared.gdn_recurrent_state import native_state_enabled

    kernel = gdn_kernel_mode()
    if kernel == "recurrent":
        return native_state_enabled()
    return kernel == "cpr"


#: verl rollout keys of the engine features whose GDN state must resume exactly.
STATE_FEATURE_KEYS = {
    "enable_chunked_prefill": "actor_rollout_ref.rollout.enable_chunked_prefill",
    "enable_prefix_caching": "actor_rollout_ref.rollout.enable_prefix_caching",
}


def _speculative(config: Any) -> bool:
    """True when the rollout runs speculative decoding (verl MTP rollout or an explicit vLLM config)."""
    from omegaconf import OmegaConf

    mtp = OmegaConf.select(config, "actor_rollout_ref.rollout.mtp", default=None)
    if mtp is not None and bool(mtp.get("enable", False)) and bool(mtp.get("enable_rollout", False)):
        return True
    return OmegaConf.select(config, "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config",
                            default=None) is not None


def _state_features(config: Any) -> tuple[frozenset[str], str]:
    """``(admitted, reason)`` from the engine gate's own predicate (``gdn_engine_patch``)."""
    from isoexec.runtimes.vllm.gdn_engine_patch import resumable_engine_args_for

    return resumable_engine_args_for(speculative=_speculative(config))


def _forward_state_feature_flags(config: Any, enabled: list[str]) -> list[str]:
    """Turn on the engine flags the GDN composition needs for ``enabled`` (here, for the driver's own checks)
    and forward them to every Ray worker; a flag the caller turned off explicitly is refused."""
    from omegaconf import OmegaConf

    from isoexec.runtimes.vllm.gdn_engine_patch import enable_state_features, state_feature_env

    enable_state_features(enabled)
    changes = []
    for name, value in state_feature_env(enabled).items():
        key = f"ray_kwargs.ray_init.runtime_env.env_vars.{name}"
        old = OmegaConf.select(config, key, default=None)
        if old is not None and str(old).strip().lower() in ("0", "false", "no", ""):
            raise ValueError(f"isoexec.enable=true: {', '.join(enabled)} under the CPR GDN composition needs "
                             f"{name}={value}, but {key}={old!r}")
        if old is None:
            OmegaConf.update(config, key, os.environ.get(name, value), force_add=True)
            changes.append(f"{key}: None -> {os.environ.get(name, value)!r} ({', '.join(enabled)} under CPR)")
    return changes


def apply(config: Any) -> list[str]:
    """Write the IsoExec settings into ``config`` in place; return ``key: old -> new`` for each change."""
    from omegaconf import OmegaConf, open_dict

    if config.get("model_engine", "megatron") not in (None, "megatron") and "model_engine" in config:
        raise ValueError(f"isoexec.enable=true requires model_engine=megatron, got {config.model_engine!r}")
    changes = []
    multi_step = bool(OmegaConf.select(config, "isoexec.multi_step", default=False))
    required = dict(REQUIRED)
    if multi_step:
        # N-step training (trainer.IsoExecPPOTrainerSyncMultiStep) compares every step's rollout and
        # allows validation, which publishes no rows.
        required["trainer.v1.trainer_mode"] = MULTI_STEP_TRAINER_MODE
        del required["trainer.val_before_train"]
    with open_dict(config):
        for key, value in required.items():
            old = OmegaConf.select(config, key, default=None)
            if old != value:
                OmegaConf.update(config, key, value, force_add=True)
                changes.append(f"{key}: {old!r} -> {value!r}")
        libs = OmegaConf.select(config, "actor_rollout_ref.model.external_lib", default=None)
        libs = OmegaConf.to_container(libs) if OmegaConf.is_config(libs) else libs
        current = [] if libs is None else ([libs] if isinstance(libs, str) else [str(item) for item in libs])
        merged = _external_libs(libs)
        if merged != current:
            OmegaConf.update(config, "actor_rollout_ref.model.external_lib", merged if len(merged) > 1 else merged[0],
                             force_add=True)
            changes.append(f"actor_rollout_ref.model.external_lib: {libs!r} -> {merged!r}")
        admitted, reason = _state_features(config)
        enabled = []
        for feature, key in STATE_FEATURE_KEYS.items():
            old = OmegaConf.select(config, key, default=None)
            if feature in admitted:
                if old:  # the caller's value (verl's default: on) stands
                    enabled.append(feature)
                continue
            if old is not False:
                OmegaConf.update(config, key, False, force_add=True)
                changes.append(f"{key}: {old!r} -> False ({reason})")
        changes += _forward_state_feature_flags(config, enabled)
        key = "actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling"
        old = OmegaConf.select(config, key, default=None)
        # Async scheduling is on wherever the GDN state stays exact under it (_async_scheduling_safe).
        want = _async_scheduling_safe()
        if old is not want:
            OmegaConf.update(config, key, want, force_add=True)
            changes.append(f"{key}: {old!r} -> {want}")
    print("[ISOEXEC-VERL] isoexec.enable=true set: " + ("; ".join(changes) if changes else "nothing changed"), flush=True)
    return changes


__all__ = ["MULTI_STEP_TRAINER_MODE", "REGISTER", "REQUIRED", "SERVER_NAME", "SERVER_PATH", "STATE_FEATURE_KEYS",
           "TRAINER_MODE", "apply"]
