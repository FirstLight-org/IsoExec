"""Compose verl's config with the launcher's overrides, expand isoexec.enable and run every IsoExec validator.

No Ray, no GPU. Hydra's ``--cfg job`` prints a config without running ``main``, so it would skip the
expansion and validation; this checker runs them. It mirrors verl's own driver order (expand, then
validate) and the engine budgets verl's worker copies in before building the engine.

Usage: python examples/verl_mrv2/dry_check.py <hydra overrides...>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(overrides: list[str]) -> None:
    import hydra
    import verl
    from verl.utils.config import omega_conf_to_dataclass

    from isoexec.integrations.verl import config as isoexec_config
    from isoexec.integrations.verl.engine import capture_bound, validate_engine_config
    from isoexec.integrations.verl.rollout import expected_workers, validate_rollout_config
    from isoexec.integrations.verl.trainer import validate_trainer_config

    with hydra.initialize_config_dir(config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    if not config.isoexec.enable:
        raise SystemExit("dry check requires isoexec.enable=true")
    isoexec_config.apply(config)
    validate_trainer_config(config)
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    validate_rollout_config(rollout)
    engine = omega_conf_to_dataclass(config.actor_rollout_ref.actor).engine
    engine.max_token_len_per_gpu = config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
    engine.infer_max_token_len_per_gpu = config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
    engine.use_dynamic_bsz = config.actor_rollout_ref.actor.use_dynamic_bsz
    validate_engine_config(engine, config.actor_rollout_ref.model)
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    per_replica = expected_workers(rollout)
    if world % per_replica:
        raise SystemExit(f"rollout replica size {per_replica} does not divide the trainer world {world}")
    print(json.dumps({"dry_check": "PASS", "trainer_mode": config.trainer.v1.trainer_mode,
                      "use_v1": config.trainer.use_v1, "rollout_workers_per_replica": per_replica,
                      "replicas": world // per_replica, "capture_bound": capture_bound(engine),
                      "logprobs_mode": rollout.logprobs_mode, "calculate_log_probs": rollout.calculate_log_probs,
                      "custom_server": [rollout.agent.custom_async_server.path, rollout.agent.custom_async_server.name]},
                     sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
