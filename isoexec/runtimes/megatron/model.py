"""Megatron native runtime model built from an IsoExec stage definition.

``MegatronModel`` interface:
- owns an isolated executable FX GraphModule
- binds trainer ops
- materializes states (allocating parameters/buffers, loading weights)
- exposes the native Megatron model interface
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import make_viewless_tensor

from isoexec.compiler.realization.graph import clone_graph_module, drop_auxiliary_outputs
from isoexec.compiler.realization.pp import prepare_pipeline_graph, uses_mrope
from isoexec.compiler.realization.sp import prepare_sequence_parallel_graph
from isoexec.compiler.realization.storage import allocate_tensors, take_host_resident_buffers
from isoexec.compiler.realization.tp import set_head_output_shape, set_local_dims, shard_parameter_shapes
from isoexec.compiler.realization.weights import load_weights
from isoexec.compiler.outputs import model_outputs
from isoexec.compiler.partition import StageDefinition
from isoexec.nn import LMHead, MoE
from isoexec.nn.component import Component
from isoexec.ops.collectives.tp_bytes import bind_seq_scatter

from ._binding import (
    bind_trainer_ops,
    placement_from_groups,
    set_expert_parallel_attributes,
    set_partial_gradient_sums,
    set_sequence_parallel_attributes,
    set_tensor_parallel_attributes,
)
from ._inputs import packed_sequence_context

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import PretrainedConfig

    from isoexec.compiler.source import ModelSource


def node_shared_host_buffers(graph_module, taken, *, checkpoint: str | None, pg_collection):
    """Place the taken host-resident buffers (``HOST_RESIDENT_BUFFERS``) in ONE shared host copy per node for the
    ranks of this pipeline stage (``runtimes/vllm/host_tables.NodeSharedHostBuffers``, collective over the stage's
    TP x DP x CP group): the host's lowest rank creates and fills a shared memory segment named by the
    checkpoint identity and layout, the others map it read-only; it is unlinked once every co-located rank has
    mapped it and registered with CUDA in pieces after the load (``finish``). Returns the buffers and the
    logical names of the taken buffers, or ``(None, {})`` without host-resident buffers (the path every other
    model takes, unchanged)."""
    if not taken:
        return None, {}
    from isoexec.runtimes.vllm.host_tables import NodeSharedHostBuffers, read_only_registration_supported

    group = getattr(pg_collection, "tp_dp_cp", None)
    if group is None:
        from megatron.core import parallel_state
        group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    if group is not None and group.size() == 1:
        group = None
    owners = {id(module): name for name, module in graph_module.named_modules()}
    names = {(id(child), name): f"{owners[id(child)]}.{name}".lstrip(".") for child, name, _ in taken}
    device = torch.device("cuda", torch.cuda.current_device())
    pp = pg_collection.pp
    buffers = NodeSharedHostBuffers(taken, checkpoint=checkpoint, stage=f"trainer-pp{pp.rank()}/{pp.size()}",
                                    group=group, read_only_register=read_only_registration_supported(device))
    return buffers, names


class MegatronModel(MegatronModule):
    """An isolated Megatron native runtime stage with allocated TP-local state.

    Construction clones ``definition.module`` and preserves local parameter
    ties, logical names, tensor dtypes, and module training modes. Parameters
    and meta buffers remain uninitialized until weight loading.

    The caller must call ``install_trainer_kernels`` in the current worker
    process before constructing this model.

    Args:
        definition: Local stage definition containing the source graph.
        source: Author package supplying checkpoint conversions.
        hf_config: Root HF configuration for weight conversion.
        config: Megatron configuration with TP, PP and CP sizes matching
            ``pg_collection``, BF16 parameters, and CUDA graphs disabled. CP
            greater than one shards each packed batch: contiguously at any degree
            when every component has a contiguous CP form (``cp_layout``), else
            zigzag at CP=2 (#190's sm100 path). Either way every rank's forward
            rows equal CP=1 bit for bit. SP requires TP greater than one. EP greater than
            one requires MoE layers and holds whole experts (ETP=1).
            Full uniform recompute checkpoints complete authored decoder layers,
            including normalization and residual paths. ``recompute_num_layers``
            sets the number of consecutive local layers in each region. Other
            recompute granularities and block recomputation are unsupported.
        pg_collection: Process groups with native TP, PP, and expert groups.
        cp_layout: The CP row layout the host shards packed batches with, when
            it does so itself (SkyRL: ``"zigzag"``); ``None`` lets the model
            choose (``runtimes/context_parallel.select_cp_layout``) and the host
            follow ``self.cp_layout`` (verl).

    Attributes:
        graph_module: FX graph module owning local computation and registered
            model states.
        pre_process: Whether this is the first PP stage.
        post_process: Whether this is the last PP stage.
        placement: Placement the MoE and context-parallel bindings use, or None.
        aux_outputs: Names of the auxiliary outputs the PP1 graph returns.
        aux_loss_injector: Optional training-forward hook receiving logits,
            auxiliary outputs, and packed sequence parameters.
        score_temperature: Temperature the last stage divides its logits by,
            as an FP32 tensor divide (the engine's scoring temperature). At
            1.0 the logits are returned as computed.
    """

    # Megatron schedule reads model_type even for PP1; GPTModel also sets it:
    # https://github.com/NVIDIA/Megatron-LM/blob/3249e400/megatron/core/models/gpt/gpt_model.py#L185
    model_type = ModelType.encoder_or_decoder

    def __init__(
        self,
        definition: StageDefinition,
        *,
        source: ModelSource,
        hf_config: PretrainedConfig,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        cp_layout: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        cp_group = getattr(pg_collection, "cp", None)
        if (1 if cp_group is None else cp_group.size()) != config.context_parallel_size:
            raise ValueError("MegatronModel construction requires matching CP config and group size")
        if not config.bf16 or config.params_dtype != torch.bfloat16:
            raise ValueError("MegatronModel construction requires BF16 parameters")
        if config.cuda_graph_impl != "none":
            raise ValueError("MegatronModel construction requires cuda_graph_impl='none'")
        if pg_collection.pp.size() != config.pipeline_model_parallel_size:
            raise ValueError("MegatronModel construction requires matching PP config and group size")
        if pg_collection.tp.size() != config.tensor_model_parallel_size:
            raise ValueError("MegatronModel construction requires matching TP config and group size")
        ep = getattr(pg_collection, "ep", None)
        if (1 if ep is None else ep.size()) != config.expert_model_parallel_size:
            raise ValueError("MegatronModel construction requires matching EP config and group size")
        if config.sequence_parallel and pg_collection.tp.size() == 1:
            raise ValueError("MegatronModel construction requires TP>1 with SP")
        if config.recompute_granularity is not None and (
            config.recompute_granularity != "full"
            or config.recompute_method != "uniform"
            or type(config.recompute_num_layers) is not int
            or config.recompute_num_layers < 1
        ):
            raise ValueError("Native activation checkpointing requires full uniform recompute with a positive layer span")
        if any(not parameter.is_meta for parameter in definition.module.parameters()):
            raise ValueError("MegatronModel construction requires meta parameters")

        from isoexec.core import load_timers

        lap = load_timers.Lap()
        super().__init__(config)
        self.source = source
        self.hf_config = hf_config
        self.pg_collection = pg_collection
        self.pre_process = pg_collection.pp.rank() == 0
        self.post_process = pg_collection.pp.rank() == pg_collection.pp.size() - 1
        self._pipeline_input: torch.Tensor | None = None

        # StageDefinition.module must remain read-only.
        self.graph_module = clone_graph_module(definition.module)
        if pg_collection.pp.size() > 1:
            # Pipeline stages cross one hidden tensor; auxiliary outputs, such as
            # MoE router scores, stay local only at PP1.
            drop_auxiliary_outputs(self.graph_module)
            prepare_pipeline_graph(self.graph_module)
        output = next(node for node in self.graph_module.graph.nodes if node.op == "output")
        _, aux = model_outputs(output.args[0]) if pg_collection.pp.size() == 1 else (None, {})
        self.aux_outputs = tuple(aux)
        self.aux_loss_injector = None
        self.score_temperature = 1.0
        self.last_aux: dict[str, torch.Tensor] = {}
        # Rollout metadata (``isoexec.schema.rollout``): read by components that declare it; the integration
        # sets ``forward_context`` (packed_seq_params -> context manager) around a microbatch schedule.
        self.reads_rollout_requests = any(isinstance(module, Component) and module.READS_ROLLOUT_REQUESTS
                                          for module in self.graph_module.modules())
        self.forward_context = None
        tp_size = pg_collection.tp.size()
        sp = config.sequence_parallel
        has_moe = any(isinstance(module, MoE) for module in self.graph_module.modules())
        cp = config.context_parallel_size
        bound = any(isinstance(module, Component) and module.PLACEMENT_BOUND for module in self.graph_module.modules())
        placement = (placement_from_groups(pg_collection, sp=sp, has_moe=has_moe)
                     if has_moe or cp > 1 or bound else None)
        self.placement = placement
        # Expert tensors split over ETP, which is TP at EP=1 and one at EP>1.
        self._etp_size = tp_size if placement is None else placement.etp.size
        shard_parameter_shapes(self.graph_module, tp_size=tp_size, side="trainer", etp_size=self._etp_size)
        set_local_dims(self.graph_module, tp_size=tp_size)
        set_head_output_shape(self.graph_module)
        if sp:
            scatter = bind_seq_scatter(tp_group=pg_collection.tp, site="seq_scatter")
            prepare_sequence_parallel_graph(self.graph_module, tp_size=tp_size, scatter=scatter.fn)
        self._cp = cp
        #: The CP row layout this model binds and verl must shard with (``None`` at CP=1).
        self.cp_layout = None
        if cp > 1:
            from isoexec.runtimes.context_parallel import select_cp_layout

            # A host that shards its own rows (SkyRL: zigzag) states the layout; otherwise the model picks.
            self.cp_layout = select_cp_layout(self.graph_module, cp=cp, required=cp_layout)
        cp_bind_group = pg_collection.cp if self.cp_layout == "contiguous" else None
        # TODO(jiangjiawei1103): Review binding details.
        bind_trainer_ops(self.graph_module, tp_group=pg_collection.tp, sp=sp, placement=placement,
                         cp_layout=self.cp_layout or "contiguous")
        if cp_bind_group is not None:
            from isoexec.runtimes.context_parallel import dist_transport_factory

            # Create every CP peer pair's communicator now, together, rather than inside a wavefront.
            dist_transport_factory(cp_bind_group)().warmup(torch.device("cuda", torch.cuda.current_device()))
        # Residual join + pre-norm pairs run as one fused kernel: the same pass, on the same captured
        # graph, as the engine (bit-identical forward and backward), before checkpoint regions copy it.
        from isoexec.compiler.realization.fusion import fuse_residual_norms

        self.fused_residual_norms = fuse_residual_norms(self.graph_module)
        # Loading skips other stages' names and resolves tied names to the local Parameter. MoE binding
        # removes experts other EP ranks own, so their names are skipped the same way.
        held = set(dict(self.graph_module.named_parameters(remove_duplicate=False)))
        self._remote_parameters = definition.remote_parameters | {
            name for name, _ in definition.module.named_parameters(remove_duplicate=False) if name not in held
        }
        self._parameter_aliases = dict(definition.parameter_aliases)
        # TODO(jiangjiawei1103): DDP/optimizer setup must collect registered Parameters after all
        # construction-time parameter changes.
        # Tables a component keeps in host memory (read through a device mapping) are not device tensors.
        host_resident = take_host_resident_buffers(self.graph_module)
        lap("model_init.bind")
        allocate_tensors(self.graph_module, device=torch.device("cuda", torch.cuda.current_device()))
        lap("model_init.allocate")
        # One host copy per node shared by the stage's ranks (the engine's form), not a private copy per rank.
        self._host_buffers, self._host_buffer_names = node_shared_host_buffers(
            self.graph_module, host_resident, checkpoint=checkpoint_path, pg_collection=pg_collection)
        lap("model_init.host_segment")
        if tp_size > 1:
            set_tensor_parallel_attributes(self.graph_module)
            self.partial_gradient_sums = tuple(set_partial_gradient_sums(self.graph_module, pg_collection.tp))
        else:
            self.partial_gradient_sums = ()
        if sp:
            set_sequence_parallel_attributes(self.graph_module)
        if placement is not None and placement.ep.size > 1:
            set_expert_parallel_attributes(self.graph_module)
        self.training = self.graph_module.training
        self._activation_checkpoint_regions: tuple = ()
        self.activation_checkpoint_layers: tuple[tuple[str, ...], ...] = ()
        if config.recompute_granularity is not None:
            from ._recompute import checkpoint_layers

            self._activation_checkpoint_regions = checkpoint_layers(
                self.graph_module, span=config.recompute_num_layers,
            )
            self.activation_checkpoint_layers = tuple(region.layer_paths for region in self._activation_checkpoint_regions)
        self._shared_parameter_name = definition.pp_shared_parameter
        if self._shared_parameter_name is None:
            # Capture guarantees one terminal LMHead call; PP1 can resolve
            # the local embedding/head tie from that head.
            for node in self.graph_module.graph.nodes:
                if node.op != "call_module":
                    continue
                head = self.graph_module.get_submodule(str(node.target))
                if isinstance(head, LMHead):
                    if head.tied:
                        self._shared_parameter_name = f"{node.target}.weight"
                    break
        if self._shared_parameter_name is not None:
            # The tied embedding/output weight also receives the lookup's gradient through autograd, which
            # a fused wgrad (``grad_added_to_main_grad``) would make Megatron's hook drop: keep it unfused.
            from isoexec.ops.mm.wgrad import OPT_OUT_ATTR

            setattr(self.graph_module.get_parameter(self._shared_parameter_name), OPT_OUT_ATTR, True)
        self._mrope = uses_mrope(self.graph_module)
        if definition.pp_shared_parameter is not None:
            # Megatron's embedding group all-reduces gradients of the replicas tied across PP stages.
            weight = self.shared_embedding_or_output_weight()
            weight.shared_embedding = True
            weight.is_embedding_or_output_parameter = True
            if self.post_process:
                weight.shared = True
        lap("model_init.bind")

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Toggle constructed layer regions without changing parameter/state ownership."""
        if type(enabled) is not bool:
            raise ValueError("Activation checkpointing toggle must be a bool")
        if enabled and not self._activation_checkpoint_regions:
            raise ValueError("Construct the model with full uniform recompute before enabling checkpointing")
        for region in self._activation_checkpoint_regions:
            region.enabled = enabled

    def set_input_tensor(self, input_tensor: torch.Tensor | list[torch.Tensor | None] | None) -> None:
        """Set input tensor instead of forward()'s input.

        Retain this microbatch's ``[T, 1, hidden]`` pipeline activation, or this
        rank's ``[T // TP, 1, hidden]`` chunk under SP; the first stage accepts
        only ``None`` or ``[None]``.

        For details, please see:
        https://github.com/NVIDIA/Megatron-LM/blob/aa7a2f34/megatron/core/transformer/transformer_block.py#L480-L488
        """
        if isinstance(input_tensor, list):
            if len(input_tensor) != 1:
                raise ValueError("MegatronModel expects exactly one pipeline input")
            input_tensor = input_tensor[0]
        if self.pre_process and input_tensor is not None:
            raise ValueError("The first Megatron stage does not accept a pipeline activation")
        self._pipeline_input = input_tensor

    @property
    def share_embeddings_and_output_weights(self) -> bool:
        """Whether the input embedding and LM head weights are tied."""
        return self._shared_parameter_name is not None

    def shared_embedding_or_output_weight(self) -> torch.nn.Parameter | None:
        """Return the current registered shared embedding/head weight."""
        if self._shared_parameter_name is None:
            return None
        return self.graph_module.get_parameter(self._shared_parameter_name)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Compute and return logits for tokens packed into one row.

        Args:
            input_ids: First-stage token IDs shaped ``[1, T]``, where T is this
                rank's number of tokens in the pack (its zigzag rows under CP).
            position_ids: Optional token positions. Positions restart at zero
                for each sequence if omitted.

        Returns:
            FP32 logits shaped ``[1, T, vocab]`` without vocabulary padding at
            TP1, or this rank's vocabulary shard with padding masked to -inf.
            Nonfinal PP stages return hidden states shaped ``[T, 1, hidden]``,
            or this rank's ``[T // TP, 1, hidden]`` chunk under SP.
        """
        first_stage_ids = input_ids is not None or not self.pre_process
        if not first_stage_ids or packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("MegatronModel requires first-stage input_ids and PackedSeqParams(qkv_format='thd')")
        if attention_mask is not None:
            raise ValueError("Packed MegatronModel requires attention_mask=None")
        if packed_seq_params.local_cp_size not in (None, 1) or packed_seq_params.cp_group is not None:
            raise ValueError("MegatronModel serves static context parallelism only (no dynamic CP groups)")
        mode = getattr(packed_seq_params, "cp_partition_mode", None)
        if self._cp > 1 and mode is not None and mode != self.cp_layout:
            raise ValueError(f"MegatronModel shards CP rows {self.cp_layout!r}ly; got cp_partition_mode={mode!r} "
                             f"(verl: cp_layout={self.cp_layout!r})")
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv_padded
        if cu_kv is None:
            cu_kv = packed_seq_params.cu_seqlens_kv
        if cu is None or cu_kv is None or (cu_kv is not cu and not torch.equal(cu_kv, cu)):
            raise ValueError("MegatronModel requires matching self-attention q and kv boundaries")
        incoming, self._pipeline_input = self._pipeline_input, None
        if not self.pre_process:
            if incoming is None:
                raise ValueError("Megatron PP requires a [T, 1, hidden] activation for this packed batch")
            cu = cu.to(device=incoming.device)
        max_q, max_kv = packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv
        contiguous = self.cp_layout == "contiguous"
        if contiguous:
            cu, max_q, max_kv = self._context_parallel_boundaries(cu, input_ids, incoming)
            if position_ids is None:
                position_ids = self._cp_positions(cu)
        # TODO(jiangjiawei1103): Review and think about if we really need seq ctx input into gm.
        seq = packed_sequence_context(
            input_ids if self.pre_process else None,
            position_ids,
            cu,
            max_q=max_q,
            max_kv=max_kv,
            mrope=self._mrope,
            # The zigzag layout registers itself on the global boundaries; the contiguous plan is registered above.
            cp=None if self.placement is None or contiguous else self.placement.cp,
        )
        tokens, tp = int(seq.positions.shape[-1]), self.pg_collection.tp.size()
        if self.config.sequence_parallel and tokens % tp:
            raise ValueError(f"SP requires the packed token count to divide TP={tp}, got T={tokens}; "
                             "SkyRL pads each packed sequence to a multiple of TP")
        context = self.forward_context(packed_seq_params) if self.forward_context is not None else nullcontext()
        with context:
            if self.__dict__.get("_first_forward_timed"):
                return self._forward_graph(input_ids, seq, packed_seq_params, incoming, tokens, tp)
            # The process's first forward (lazy kernel loads and JIT included) is a cold-start component; one
            # device synchronisation closes it.
            self.__dict__["_first_forward_timed"] = True
            from isoexec.core import load_timers

            with load_timers.phase("trainer.first_forward"):
                out = self._forward_graph(input_ids, seq, packed_seq_params, incoming, tokens, tp)
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            load_timers.mark("first_forward")
            return out

    def _forward_graph(self, input_ids, seq, packed_seq_params, incoming, tokens, tp):
        if self.pg_collection.pp.size() == 1:
            logits, aux = model_outputs(self.graph_module(input_ids=seq.input_ids, seq=seq))
            if self.aux_loss_injector is not None and torch.is_grad_enabled():
                logits = self.aux_loss_injector(logits, aux, packed_seq_params)
            return self._tempered(logits).unsqueeze(0)
        hidden = None
        if not self.pre_process:
            # The previous stage must send this packed batch's [T, 1, hidden] stream, or this rank's
            # [T // TP, 1, hidden] chunk under SP.
            rows = tokens // tp if self.config.sequence_parallel else tokens
            expected = (rows, 1, self.config.hidden_size)
            if tuple(incoming.shape) != expected:
                raise ValueError(f"Megatron PP requires a {list(expected)} activation for this packed batch, "
                                 f"got {list(incoming.shape)}")
            hidden = incoming.squeeze(1)
        result = self.graph_module(input_ids=seq.input_ids, seq=seq, hidden_states=hidden)
        if self.post_process:
            return self._tempered(result).unsqueeze(0)
        # Native schedules free sent outputs in place, which a view of another tensor does not allow.
        return make_viewless_tensor(result.unsqueeze(1), requires_grad=True, keep_graph=True)

    def _tempered(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_temperature == 1.0:
            return logits
        from isoexec.ops.logprobs.temperature import temperature_divide

        return temperature_divide(logits.float(), self.score_temperature)
    def _context_parallel_boundaries(self, cu: torch.Tensor, input_ids: torch.Tensor | None,
                                     incoming: torch.Tensor | None) -> tuple[torch.Tensor, int, int]:
        """This rank's local boundaries, with the microbatch's contiguous CP plan registered on them.

        ``cu`` holds the GLOBAL padded boundaries (verl's ``cu_seqlens_q_padded``); this rank holds rows
        ``[rank*T/cp, (rank+1)*T/cp)`` of that stream.
        """
        from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan

        group = self.pg_collection.cp
        plan = ContiguousPlan.build([int(v) for v in cu.tolist()], group.size(), group.rank())
        rows = (input_ids.shape[-1] if self.pre_process else
                (incoming.shape[0] * (self.pg_collection.tp.size() if self.config.sequence_parallel else 1)))
        if rows != plan.local:
            raise ValueError(f"context parallelism: this rank holds {rows} rows, the contiguous layout of "
                             f"{plan.total} packed rows over cp={plan.cp} gives {plan.local}")
        device = input_ids.device if input_ids is not None else incoming.device
        local = plan.device_tensors(device)["local_cu"]
        register_plan(local, plan)
        return local, plan.max_q, max(plan.max_k, plan.max_q)

    @staticmethod
    def _cp_positions(local_cu: torch.Tensor) -> torch.Tensor:
        from isoexec.ops.collectives.cp_contiguous import plan_for

        plan = plan_for(local_cu)
        return torch.tensor(plan.positions(), dtype=torch.int64, device=local_cu.device).unsqueeze(0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF weights into the registered tensors.

        The runtime supplies an iterator over the model's checkpoint weights.
        Weight copies completed before an error are not restored. With PP or
        EP, weights of other stages or EP ranks are skipped and tied names load
        this stage's Parameter.

        Returns:
            Loaded logical names qualified by ``graph_module``.
        """
        tp = self.pg_collection.tp
        # Node-shared host buffers: only the host's leader writes them; the others skip their names.
        host = self._host_buffers
        shared = set() if host is None else host.skip_names(lambda module, name: self._host_buffer_names[id(module), name])
        logical_names = load_weights(self.graph_module, weights, source=self.source, hf_config=self.hf_config,
                                     tp_size=tp.size(), tp_rank=tp.rank(), side="trainer",
                                     remote=self._remote_parameters | shared, aliases=self._parameter_aliases,
                                     etp_size=self._etp_size)
        # Collective: the leader publishes the copy, every rank verifies, registers it in pieces and maps it.
        if host is not None and not host.ready:
            from isoexec.core import load_timers

            with load_timers.phase("weights.host_finish"):
                host.finish(torch.device("cuda", torch.cuda.current_device()))
            # the names the leader loaded count as loaded on every rank (same bytes)
            logical_names = set(logical_names) | shared
        return {f"graph_module.{name}" for name in logical_names}
