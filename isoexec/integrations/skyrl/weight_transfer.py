"""Select IsoExec receive engines through vLLM's existing plugin registry."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import entry_points
import json
import os
from typing import Any

PLUGINS = "isoexec,isoexec_skyrl"


def require_plugins() -> None:
    """A source checkout alone cannot install entry points in spawned workers."""
    expected = {"isoexec": "isoexec.runtimes.vllm.vllm_plugin:register",
                "isoexec_skyrl": "isoexec.integrations.skyrl.weight_transfer:register"}
    entries = entry_points(group="vllm.general_plugins")
    for name, target in expected.items():
        matches = [entry.value for entry in entries if entry.name == name]
        if matches != [target]:
            raise RuntimeError(f"Reinstall IsoExec in this environment: vllm.general_plugins entry point {name!r} "
                               f"must resolve uniquely to {target!r}, found {matches!r}")


def register() -> None:
    """Register lazy receivers in every opted-in vLLM process, without CUDA init."""
    if os.environ.get("VLLM_ISOEXEC_REGISTERED") != "1":
        return
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    for name, cls in (("isoexec_nccl", "IsoExecNCCLWeightTransferEngine"),
                      ("isoexec_ipc", "IsoExecIPCWeightTransferEngine")):
        if name not in WeightTransferEngineFactory._registry:
            WeightTransferEngineFactory.register_engine(name, "isoexec.integrations.skyrl.weight_receiver", cls)


def configure(args: Any) -> None:
    """Keep SkyRL's selected transport, replacing only its receive implementation."""
    from vllm.config import WeightTransferConfig

    config = getattr(args, "weight_transfer_config", None)
    if config is None:
        return
    require_plugins()
    register()
    os.environ["VLLM_PLUGINS"] = PLUGINS
    if isinstance(config, str):
        config = json.loads(config)
    if isinstance(config, dict):
        config = WeightTransferConfig(**config)
    if not isinstance(config, WeightTransferConfig):
        raise ValueError("IsoExec requires a native weight_transfer_config")
    names = {"skyrl_nccl": "isoexec_nccl", "skyrl_ipc": "isoexec_ipc"}
    if config.backend not in names:
        raise ValueError(f"IsoExec SkyRL receiver does not support {config.backend!r}")
    args.weight_transfer_config = replace(config, backend=names[config.backend])
