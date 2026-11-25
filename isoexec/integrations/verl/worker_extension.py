"""vLLM worker extension for IsoExec models served by verl's colocated rollout.

verl's colocated weight sync streams named tensors to each vLLM worker over its bucketed IPC
channel. This extension keeps that transport and loads the stream through ``VllmModel``'s native
update lifecycle (``start_weight_update`` -> logical-name ``load_weights`` -> coverage-checked
``finish_weight_update``). After each complete update the worker publishes an acknowledgement
naming its weight version, runner class and loaded-parameter digest; the rollout server opens
generation for a version only after every worker of its replica has acknowledged it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def ack_directory(root: Path, version: int) -> Path:
    return root / "acks" / f"v{version}"


def worker_identity(worker: Any) -> dict[str, Any]:
    """This worker's coordinates in vLLM's own rank groups."""
    from vllm.distributed import parallel_state as ps

    config = worker.model_runner.vllm_config.parallel_config
    identity = {
        "rank": int(worker.rank),
        "local_rank": int(worker.local_rank),
        "dp_rank": int(getattr(config, "data_parallel_rank", 0) or 0),
        "tp_rank": int(ps.get_tp_group().rank_in_group),
        "pp_rank": int(ps.get_pp_group().rank_in_group),
        "replica_rank": int(os.environ.get("VERL_REPLICA_RANK", "0")),
    }
    for name, getter in (("pcp_rank", "get_pcp_group"), ("dcp_rank", "get_dcp_group"), ("ep_rank", "get_ep_group")):
        group = getattr(ps, getter, None)
        try:
            identity[name] = int(group().rank_in_group) if group is not None else 0
        except (AssertionError, AttributeError):
            identity[name] = 0
    return identity


def local_parameter_sha256(model: torch.nn.Module) -> str:
    """The worker's W0/W1 receipt digest (device-side in production runs; see ``digest``)."""
    from .digest import parameter_digest

    return parameter_digest(model.named_parameters())


class IsoExecWorkerExtension(vLLMColocateWorkerExtension):
    """Native IsoExec updates and acknowledgements on verl's colocated transport."""

    _isoexec_version = -1

    def _isoexec_model(self):
        from isoexec.runtimes.vllm.model import VllmModel

        model = self.model_runner.get_model()
        if not isinstance(model, VllmModel):
            raise RuntimeError(f"IsoExec worker extension requires a VllmModel, got {type(model).__name__}")
        return model

    def monkey_patch_model(self, vocab_size: int, banned_token_ids=None):
        """Keep the model's complete distribution: masking logits would change every full row.

        ``VllmModel.compute_logits`` already trims vocabulary padding. Out-of-tokenizer entries and
        placeholder tokens stay in the distribution the trainer scores as well.
        """
        model = self._isoexec_model()
        self._isoexec_record({
            "event": "logit_mask_skipped", "tokenizer_vocab": int(vocab_size),
            "banned_token_ids": len(banned_token_ids or ()), "model_class": type(model).__name__,
        }, "workers.jsonl")

    def isoexec_runtime_identity(self) -> dict[str, Any]:
        """Report the runner, model and import paths this worker actually uses."""
        import vllm

        import isoexec

        runner = self.model_runner
        return {
            **worker_identity(self),
            "runner_class": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "model_class": f"{type(runner.get_model()).__module__}.{type(runner.get_model()).__qualname__}",
            "use_v2_model_runner": bool(getattr(runner.vllm_config, "use_v2_model_runner", False)),
            "isoexec_file": os.path.abspath(isoexec.__file__),
            "vllm_file": os.path.abspath(vllm.__file__),
            "vllm_version": vllm.__version__,
            "pid": os.getpid(),
        }

    def update_weights_from_ipc(self, peft_config: dict | None = None, base_sync_done=False, use_shm: bool = False):
        """Receive one complete update into the registered model, then acknowledge its version."""
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if peft_config:
            raise RuntimeError("IsoExec worker extension does not support LoRA updates")
        if self.device is None:
            self.device = torch.device(f"cuda:{self.local_rank}")
        model = self._isoexec_model()
        received = 0
        with torch.device(self.device):
            model.start_weight_update()

            def on_bucket_received(weights: list[tuple[str, torch.Tensor]], is_last: bool) -> None:
                nonlocal received
                with torch.device(self.device):
                    model.load_weights(weights)
                received += len(weights)

            receiver = BucketedWeightReceiver(zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm)
            receiver.receive_weights(on_bucket_received=on_bucket_received)
            model.finish_weight_update()
        torch.cuda.synchronize(self.device)
        self._isoexec_version += 1
        ack = {
            **self.isoexec_runtime_identity(),
            "event": "weight_ack", "version": self._isoexec_version, "tensors_received": received,
            "parameter_sha256": local_parameter_sha256(model), "wall_time": time.time(),
        }
        from .full_distribution import run_root

        directory = ack_directory(run_root(), self._isoexec_version)
        directory.mkdir(parents=True, exist_ok=True)
        name = f"replica{ack['replica_rank']}-dp{ack['dp_rank']}-rank{ack['rank']}.json"
        tmp = directory / f".{name}.tmp"
        tmp.write_text(json.dumps(ack, sort_keys=True))
        os.replace(tmp, directory / name)
        return ack

    def _isoexec_record(self, payload: dict[str, Any], name: str) -> None:
        from .full_distribution import append_receipt, run_root

        append_receipt(run_root(), name, {**payload, **worker_identity(self)})


__all__ = ["IsoExecWorkerExtension", "ack_directory", "local_parameter_sha256", "worker_identity"]
