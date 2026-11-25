"""Exercise native registry selection and receive lifecycle without CUDA transport."""

from types import SimpleNamespace as NS
from importlib.metadata import EntryPoint
from typing import Any, cast

import pytest
import torch

pytest.importorskip("vllm.distributed.weight_transfer.factory")

from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from isoexec.integrations.skyrl import weight_transfer


@pytest.mark.parametrize("backend,native", [("ipc", IPCWeightTransferEngine), ("nccl", NCCLWeightTransferEngine)])
@pytest.mark.parametrize("failure", [None, "receive", "finish"])
def test_native_transport_is_retained_and_imports_are_closed(monkeypatch, backend, native, failure):
    monkeypatch.setattr(weight_transfer, "entry_points", lambda **kwargs: [
        EntryPoint(name="isoexec", value="isoexec.runtimes.vllm.vllm_plugin:register", group="vllm.general_plugins"),
        EntryPoint(name="isoexec_skyrl", value="isoexec.integrations.skyrl.weight_transfer:register", group="vllm.general_plugins"),
    ])
    monkeypatch.setenv("VLLM_ISOEXEC_REGISTERED", "1")
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", dict(WeightTransferEngineFactory._registry))
    original = WeightTransferConfig(backend="skyrl_" + backend)
    args = NS(weight_transfer_config=original)
    weight_transfer.configure(args)
    assert original.backend == "skyrl_" + backend
    assert args.weight_transfer_config.backend == "isoexec_" + backend
    events = []

    class Model(torch.nn.Module):
        def start_weight_update(self):
            events.append("start")

        def finish_weight_update(self):
            events.append("finish")
            if failure == "finish":
                raise RuntimeError("finish failed")

    model = Model()
    engine = WeightTransferEngineFactory.create_engine(args.weight_transfer_config,
        cast(Any, NS(model_config=None, parallel_config=None)), torch.device("cpu"), model)
    assert isinstance(engine, native)
    assert engine.supports_draft_weight_update is False
    cast(Any, engine)._packed_importer = NS(close=lambda: events.append("close"))

    def receive(self, payload):
        assert self.model is model and payload is update
        events.append("native_receive")
        if failure == "receive":
            raise RuntimeError("receive failed")

    monkeypatch.setattr(native, "receive_weights", receive)
    update = object()
    engine.start_weight_update()
    if failure == "receive":
        with pytest.raises(RuntimeError, match="receive failed"):
            engine.receive_weights(update)
        assert events == ["start", "native_receive", "close"]
    else:
        engine.receive_weights(update)
        if failure == "finish":
            with pytest.raises(RuntimeError, match="finish failed"):
                engine.finish_weight_update()
        else:
            engine.finish_weight_update()
        assert events == ["start", "native_receive", "finish", "close"]


@pytest.mark.parametrize("backend", ["isoexec_nccl", "nccl", "delta", "sharded_rdt"])
def test_only_skyrls_push_receivers_are_replaced(monkeypatch, backend):
    monkeypatch.setattr(weight_transfer, "require_plugins", lambda: None)
    monkeypatch.setenv("VLLM_ISOEXEC_REGISTERED", "1")
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", dict(WeightTransferEngineFactory._registry))
    with pytest.raises(ValueError, match="does not support"):
        weight_transfer.configure(NS(weight_transfer_config=WeightTransferConfig(backend=backend)))


def test_registration_does_not_change_a_nonisoexec_process(monkeypatch):
    monkeypatch.delenv("VLLM_ISOEXEC_REGISTERED", raising=False)
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", {})
    weight_transfer.register()
    assert WeightTransferEngineFactory._registry == {}


def test_missing_installed_entry_point_refuses_before_backend_selection(monkeypatch):
    monkeypatch.setattr(weight_transfer, "entry_points", lambda **kwargs: [])
    config = WeightTransferConfig(backend="skyrl_ipc")
    args = NS(weight_transfer_config=config)
    with pytest.raises(RuntimeError, match="Reinstall IsoExec"):
        weight_transfer.configure(args)
    assert args.weight_transfer_config is config and config.backend == "skyrl_ipc"
