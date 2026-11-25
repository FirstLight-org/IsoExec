"""Native configuration/rank metadata proof; no process groups, CUDA or scheduling."""

from importlib import metadata as installed_metadata
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from isoexec.compiler.tests.test_capture_cpu import Model


def test_actual_worker_admission_reaches_deterministic_installer_before_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real configuration/adapter caller; stop at installer, never run a CUDA kernel."""
    framework = pytest.importorskip("skyrl.train.config.config")
    integration = pytest.importorskip("isoexec.integrations.skyrl.megatron")
    pytest.importorskip("megatron.core.transformer.transformer_config")
    from isoexec.integrations.skyrl.config import SkyRLConfig
    from isoexec.runtimes import trainer_installs

    events: list[Any] = []
    class InstallerReached(Exception):
        pass
    def install(path: str, *, source: Any = None, raw_config: Any = None,
                deterministic_attention_backward: bool = False) -> dict[str, Any]:
        events.append(("install", path, deterministic_attention_backward))
        raise InstallerReached
    monkeypatch.setattr(trainer_installs, "install_trainer_kernels", install)
    cfg = framework.TrainerConfig()
    cfg.policy.model.path = "checkpoint-unread-before-installer"
    cfg.policy.megatron_config = framework.MegatronConfig(tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2, expert_tensor_parallel_size=1)
    setattr(cfg, "_isoexec_config", SkyRLConfig(engine_tp=2, model_descriptor={"model_package": "author_package"}))
    # The existing TrainerConfig channel carries the driver's model descriptor.
    cfg.gradient_checkpointing = False
    monkeypatch.setattr(integration.model_folder, "source_config", lambda *args: (object(), {}))
    worker = SimpleNamespace(cfg=cfg, _isoexec_section_name="policy")
    for request in (None, False, True):
        kwargs = {} if request is None else {"deterministic_mode": request}
        cfg.policy.megatron_config.transformer_config_kwargs = kwargs
        events.clear()
        integration.validate_model_options(cfg.policy.megatron_config)
        with pytest.raises(InstallerReached):
            integration.build_module(worker, wrap_with_ddp=False, ddp_config=None, lora_config=None, bf16=True)
        assert events == [("install", cfg.policy.model.path, request is True)]
        assert not hasattr(worker, "actor_module")
    for invalid in (None, 1, "true"):
        cfg.policy.megatron_config.transformer_config_kwargs = {"deterministic_mode": invalid}
        events.clear()
        with pytest.raises(ValueError, match="deterministic_mode must be a bool"):
            integration.build_module(worker, wrap_with_ddp=False, ddp_config=None, lora_config=None, bf16=True)
        assert not events and not hasattr(worker, "actor_module")
    cfg.policy.megatron_config.transformer_config_kwargs = {"hidden_dropout": 0.1}
    with pytest.raises(ValueError, match="does not support.*hidden_dropout"):
        integration.validate_model_options(cfg.policy.megatron_config)
    assert not torch.cuda.is_initialized()


def test_actual_native_config_mesh_admits_dense_etp1_and_preserves_expert_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real SDK computes different dense/expert grids at native ETP=1."""
    native = pytest.importorskip("megatron.core.parallel_state")
    framework = pytest.importorskip("skyrl.train.config.config")
    integration = pytest.importorskip("isoexec.integrations.skyrl.megatron")

    # The megatron-core the pinned SkyRL lock installs (environments/default.toml).
    assert installed_metadata.version("megatron-core") == "0.21.0+ed9c2c8c1"
    mc = framework.MegatronConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
                                 expert_model_parallel_size=1, expert_tensor_parallel_size=1)
    dense_grid = native.RankGenerator(tp=2, ep=1, dp=1, pp=2, cp=1, order="tp-cp-ep-dp-pp")
    expert_grid = native.RankGenerator(tp=1, ep=1, dp=2, pp=2, cp=1, order="tp-cp-ep-dp-pp")
    assert dense_grid.world_size == expert_grid.world_size == 4
    assert dense_grid.get_ranks("tp") == [[0, 1], [2, 3]]
    assert dense_grid.get_ranks("dp-cp") == [[0], [1], [2], [3]]
    assert expert_grid.get_ranks("tp") == [[0], [1], [2], [3]]
    assert expert_grid.get_ranks("dp") == [[0, 1], [2, 3]]
    assert dense_grid.get_ranks("pp") == expert_grid.get_ranks("pp") == [[0, 2], [1, 3]]

    integration.validate_mesh(None, mc, None)
    # Dense DDP's real native owner is the dense DP group: no expert buffers.
    buffers = pytest.importorskip("megatron.core.distributed.param_and_grad_buffer")
    parameters = list(Model().parameters())
    for parameter in parameters:
        setattr(parameter, "allreduce", True)
    assert all(not key.is_expert_parallel
               for key in buffers.group_params_for_buffers(parameters, grad_reduce_in_fp32=True))
    assert mc.expert_tensor_parallel_size == 1
    # The native mesh admits ETP in (1, TP) at EP=1 and only ETP=1 above it.
    integration.validate_mesh(None, framework.MegatronConfig(tensor_model_parallel_size=2,
        expert_model_parallel_size=1, expert_tensor_parallel_size=2), None)
    with pytest.raises(ValueError, match="expert_tensor_parallel_size=2 must be one of \\(1,\\)"):
        integration.validate_mesh(None, framework.MegatronConfig(tensor_model_parallel_size=2,
            expert_model_parallel_size=2, expert_tensor_parallel_size=2), None)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("tp, cp, ep, expert_dp", [(1, 2, 2, 1), (2, 2, 2, 2), (2, 2, 4, 1), (1, 2, 1, 2)])
def test_context_parallel_ranks_are_expert_peers_and_the_mesh_admits_cp_with_ep(
    monkeypatch: pytest.MonkeyPatch, tp: int, cp: int, ep: int, expert_dp: int,
) -> None:
    """Megatron carves expert groups from TP x CP x DP, so a CP group's ranks dispatch to each other
    like DP replicas; the admission counts CP in that mesh and refuses nothing for the cut itself."""
    native = pytest.importorskip("megatron.core.parallel_state")
    framework = pytest.importorskip("skyrl.train.config.config")
    integration = pytest.importorskip("isoexec.integrations.skyrl.megatron")

    world = tp * cp
    dense = native.RankGenerator(tp=tp, ep=1, dp=1, pp=1, cp=cp, order="tp-cp-ep-dp-pp")
    expert = native.RankGenerator(tp=1, ep=ep, dp=world // ep, pp=1, cp=1, order="tp-cp-ep-dp-pp")
    tp_groups, ep_groups = dense.get_ranks("tp"), expert.get_ranks("ep")
    # Every EP group is a union of whole TP groups and, at EP > TP, spans the CP cut.
    assert all(any(set(t) <= set(e) for e in ep_groups) for t in tp_groups)
    if ep > tp:
        assert all(set(c) & set(e) for c in dense.get_ranks("cp") for e in ep_groups)
    assert len(expert.get_ranks("dp")[0]) == expert_dp

    def group_of(groups):
        return next(g for g in groups if 0 in g)

    monkeypatch.setattr(integration, "mpu", SimpleNamespace(
        get_data_parallel_world_size=lambda with_context_parallel=False: 1,
        get_expert_data_parallel_world_size=lambda: expert_dp,
        get_expert_data_parallel_rank=lambda: 0,
        get_expert_model_parallel_group=lambda: ("ep", tuple(group_of(ep_groups))),
        get_tensor_model_parallel_group=lambda: ("tp", tuple(group_of(tp_groups))),
    ))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(group[1]))
    integration.validate_mesh(None, framework.MegatronConfig(
        tensor_model_parallel_size=tp, context_parallel_size=cp, expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1), None)
