"""Stream registered Megatron weights to native vLLM logical names (vLLM's WeightSource contract).

Framework-neutral: the verl and SkyRL integrations both import it from here."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from typing import Any

import torch
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

from isoexec.compiler.realization.tp import gather_weight, global_shape
from isoexec.sync.pp_gather import broadcast_stage_stream, stage_schedule
from isoexec.sync.weights import gather_expert_stack
from isoexec.nn import LMHead


class IsoExecWeightSource(WeightSource):
    """A repeatable stream of full tensors from the current registered Parameters.

    ``metadata`` and ``iter_full()`` (vLLM 0.33's WeightSource contract) use the same logical names, order, dtypes and shapes.
    Tied parameters are emitted once. Every model-parallel rank must iterate in
    the same order. Each EP-held expert role is streamed as every global expert, in order.
    The caller must finish optimizer writes before transfer begins.
    """

    def __init__(self, module: Any, dtype: torch.dtype) -> None:
        from megatron.core.utils import unwrap_model

        from isoexec.runtimes.megatron.model import MegatronModel

        if dtype != torch.bfloat16:
            raise ValueError("IsoExec weight sync requires torch.bfloat16")
        models = unwrap_model(list(module) if isinstance(module, (list, tuple)) else [module])
        if len(models) != 1 or not isinstance(models[0], MegatronModel):
            raise ValueError("IsoExec weight sync requires one native MegatronModel")
        self._module = models[0].graph_module
        self._groups = models[0].pg_collection
        self._tp_group = self._groups.tp
        self._pp_group = self._groups.pp
        self._head_name = next((str(node.target) for node in self._module.graph.nodes
                                if node.op == "call_module"
                                and isinstance(self._module.get_submodule(str(node.target)), LMHead)), None)

    def _parameters(self) -> Iterator[tuple[str, str, torch.nn.Parameter]]:
        prefix = self._head_name + "." if self._head_name is not None else None
        for logical_name, parameter in self._module.named_parameters():
            # The final stage's PP-tied head replicates the first stage's embedding.
            if getattr(parameter, "shared", False):
                continue
            name = ("head." + logical_name.removeprefix(prefix) if prefix and logical_name.startswith(prefix)
                    else "body." + logical_name)
            yield name, logical_name, parameter

    def _local_entries(self) -> list[tuple[str, torch.dtype, tuple[int, ...], str | tuple[str, str, int]]]:
        """Describe this stage's stream with each entry's source: a logical name, or for EP-held experts the
        ``(MoE path, role, global expert)`` row of the stack one gather over the EP group builds."""
        tp_size = self._tp_group.size()
        entries: list = []
        roles = set()
        for name, logical_name, parameter in self._parameters():
            # Megatron marks EP-held experts with allreduce=False.
            if getattr(parameter, "allreduce", True):
                entries.append((name, parameter.dtype,
                                global_shape(self._module, logical_name, parameter, tp_size=tp_size, side="trainer"),
                                logical_name))
                continue
            expert, _, role = logical_name.rpartition(".")
            moe_path = expert.rpartition(".")[0].rpartition(".")[0]
            if (moe_path, role) in roles:
                continue
            roles.add((moe_path, role))
            prefix = name[: len(name) - len(logical_name)]
            # EP>1 holds whole experts (ETP=1), so each global expert has the local shape.
            entries += [(f"{prefix}{moe_path}.experts.{index}.{role}", parameter.dtype, tuple(parameter.shape),
                         (moe_path, role, index))
                        for index in range(self._module.get_submodule(moe_path).num_experts)]
        return entries

    def _stage_metadata(self, entries) -> list[list[tuple[str, torch.dtype, tuple[int, ...]]]]:
        local = [(name, dtype, shape) for name, dtype, shape, _ in entries]
        if self._pp_group.size() == 1:
            return [local]
        stages: list = [None] * self._pp_group.size()
        torch.distributed.all_gather_object(stages, local, group=self._pp_group)
        return stages

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        """Describe the complete stream without copying parameter data.

        vLLM 0.33 requires a cached property: engines read it every round, and the stream's names,
        dtypes and shapes are fixed for the model's life. With PP > 1 the first read is a collective
        over the PP group, as before."""
        stages = self._stage_metadata(self._local_entries())
        stage_schedule(stages)  # a name advertised by two stages refuses
        return [ParamMeta(name, dtype, shape) for stage in stages for name, dtype, shape in stage]

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Gather TP shards and EP-held experts, and broadcast each stage's weights from that stage, in metadata
        order on every rank."""
        entries = self._local_entries()
        device = next(self._module.parameters()).device

        def local_stream() -> Iterator[tuple[str, torch.Tensor]]:
            stack, stack_role = None, None
            for name, _, _, source in entries:
                if isinstance(source, str):
                    parameter = self._module.get_parameter(source).detach()
                    yield name, gather_weight(self._module, source, parameter, tp_group=self._tp_group, side="trainer")
                    continue
                moe_path, role, index = source
                if stack_role != (moe_path, role):
                    # Each EP rank holds a contiguous ascending expert range, so one gather orders them globally.
                    held = self._module.get_submodule(moe_path).held_experts()
                    stack = gather_expert_stack([expert.get_parameter(role).detach() for expert in held],
                                                self._groups.ep)
                    stack_role = (moe_path, role)
                yield name, stack[index]

        if self._pp_group.size() == 1:
            yield from local_stream()
            return
        schedule = stage_schedule(self._stage_metadata(entries))
        for _, name, tensor in broadcast_stage_stream(schedule, local_stream(), group=self._pp_group, device=device):
            yield name, tensor


__all__ = ["IsoExecWeightSource"]
