"""vLLM native runtime model built from an IsoExec stage definition.

``VllmModel`` interface:
- owns an isolated FX GraphModule split into body and head
- binds inference ops
- materializes states (allocating parameters/buffers, loading weights)
- exposes the native vLLM model interfaces
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn

from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.pp import prepare_pipeline_graph, uses_mrope
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tp import (
    global_shape,
    set_local_dims,
    shard_parameter_shapes,
    slice_weight,
)
from isoexec.compiler.realization.weights import load_weights
from isoexec.compiler.partition import StageDefinition
from isoexec.nn import Embedding, MoE
from isoexec.runtimes.engine_ops import gather_vocab
from isoexec.storage import attach_packed_experts, attach_packed_proj, prepare_packed_storage, refresh_packed_storage

from ._binding import (
    PIECEWISE_CUDAGRAPH_MODES,
    SUPPORTED_CUDAGRAPH_MODES,
    bind_attention_cores,
    bind_body_ops,
    bind_gdn_cores,
    bind_head_ops,
    cudagraph_mode,
    placement_from_groups,
)
from ._definition import _split_body_and_head
from ._inputs import engine_sequence_context, select_sequence_metadata

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
    from vllm.sequence import IntermediateTensors

    from isoexec.compiler.source import ModelSource


def _registered_name(name: str) -> str:
    """Qualify a stage graph name by ``body`` or ``head`` as ``VllmModel`` registers it."""
    return name if name.startswith("head.") else f"body.{name}"


def _require_breakable_piecewise(vllm_config: VllmConfig, mode: str) -> None:
    """Refuse PIECEWISE graphs that would not run through vLLM's breakable capture.

    The breaks in ``graph_breaks`` only exist under Model Runner V2's breakable CUDA graphs; any
    other PIECEWISE runner would capture the composition-dependent cores into the graph.
    """
    import vllm.envs as envs

    if not envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
        raise ValueError(f"VllmModel cudagraph_mode={mode} requires VLLM_USE_BREAKABLE_CUDAGRAPH=1")
    if not getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError(f"VllmModel cudagraph_mode={mode} requires Model Runner V2 (VLLM_USE_V2_MODEL_RUNNER=1)")


class VllmModel(nn.Module):
    """An isolated vLLM native runtime stage with allocated TP-local state.

    Construction clones ``definition.module``, splits it into body and head,
    and preserves local parameter ties, tensor dtypes, and module training
    modes. Body logical names gain the ``body.`` prefix; the final PP stage's
    output head is registered as ``head``. Parameters and meta buffers remain uninitialized
    until weight loading.

    Attention cores use registered native KV cache layers; GDN cores use
    registered CPR pool buffers and native request metadata.

    The caller must construct this model within vLLM's configuration and
    allocation context. State is allocated on the worker's current CUDA device.

    Args:
        definition: Local stage definition containing the source graph.
        source: Author package supplying checkpoint conversions.
        hf_config: Root HF configuration for weight conversion.
        vllm_config: vLLM configuration requiring TP and PP sizes matching the
            native groups, BF16 parameters, compilation mode 0 with
            cudagraph_mode NONE, FULL_DECODE_ONLY or FULL_AND_PIECEWISE (breakable
            graphs on Model Runner V2), and quantization off. It
            also supplies the KV length upper bound.
        prefix: Prefix of native layer names registered in vLLM's forward
            context.

    Attributes:
        body: FX stage graph returning hidden states.
        head: Final PP stage's LMHead, or None; it shares the body's embedding
            parameter when tied.
    """

    def __init__(
        self,
        definition: StageDefinition,
        *,
        source: ModelSource,
        hf_config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("VllmModel construction requires BF16 parameters")
        compilation_mode = getattr(vllm_config.compilation_config, "mode", 0) or 0
        if getattr(compilation_mode, "value", compilation_mode) != 0:
            raise ValueError("VllmModel construction requires compilation mode 0")
        # Uniform decode replays FULL graphs. Mixed batches either run eagerly (FULL_DECODE_ONLY) or
        # replay breakable PIECEWISE graphs whose attention and GDN cores are eager breaks
        # (graph_breaks); a FULL graph over a prefill would freeze GDN's host boundaries.
        mode = cudagraph_mode(vllm_config)
        from .registration import engine_environment, requested_cudagraph_mode

        requested = requested_cudagraph_mode(vllm_config)
        if requested is not None and requested != mode:
            # vLLM rewrites a mode it cannot run (FULL_AND_PIECEWISE without breakable graphs becomes
            # FULL_DECODE_ONLY) with only a log line; a run must not proceed under another mode.
            env = " ".join(f"{k}={v}" for k, v in engine_environment(requested).items())
            raise ValueError(f"VllmModel: cudagraph_mode={requested} was requested but vLLM built the engine "
                             f"with {mode}; the engine process lacks {env or 'nothing (no environment is required)'}")
        if mode not in SUPPORTED_CUDAGRAPH_MODES:
            raise ValueError(f"VllmModel construction requires cudagraph_mode in {SUPPORTED_CUDAGRAPH_MODES}, "
                             f"got {mode}")
        if mode in PIECEWISE_CUDAGRAPH_MODES:
            _require_breakable_piecewise(vllm_config, mode)
        if vllm_config.quant_config is not None:
            raise ValueError("VllmModel construction requires quantization off")
        if getattr(vllm_config.model_config, "enable_sleep_mode", False) is True:
            # Level-2 sleep discards the Parameters (every update rewrites them) and keeps every other
            # weights-pool byte; it needs the worker hooks before the first sleep. With the flag off this
            # still installs a guard that sleeps an IsoExec model at level 1 when level 2 is requested
            # (the level may be chosen by another process whose environment differs).
            from .weights_sleep import install_exact_sleep_hooks

            install_exact_sleep_hooks()
        if any(not parameter.is_meta for parameter in definition.module.parameters()):
            raise ValueError("VllmModel construction requires meta parameters")

        from vllm.distributed import get_pp_group, get_tp_group

        super().__init__()
        self.source = source
        self.hf_config = hf_config
        self.vllm_config = vllm_config
        tp_group = get_tp_group().device_group
        if tp_group.size() != vllm_config.parallel_config.tensor_parallel_size:
            raise ValueError("VllmModel construction requires matching TP config and group size")
        pp_group = get_pp_group().device_group
        if pp_group.size() != vllm_config.parallel_config.pipeline_parallel_size:
            raise ValueError("VllmModel construction requires matching PP config and group size")
        has_moe = any(isinstance(module, MoE) for module in definition.module.modules())
        ep_group = None
        if getattr(vllm_config.parallel_config, "enable_expert_parallel", False):
            if not has_moe:
                raise ValueError("VllmModel construction with enable_expert_parallel=True requires MoE layers")
            from vllm.distributed import get_ep_group

            ep_group = get_ep_group().device_group
        self._tp_group = tp_group
        self._ep_size = 1 if ep_group is None else ep_group.size()
        # Expert tensors are whole on their owner under EP (ETP=1); otherwise they split over TP.
        self._etp_size = 1 if self._ep_size > 1 else None
        if tp_group.size() > 1:
            # One transport decision per engine process, before any collective is bound.
            from isoexec.ops.collectives.pik_tp_invariant import select_engine_transport

            self.pik_transport_intent = select_engine_transport("ENGINE")

        # StageDefinition.module must remain read-only. self.body reuses graph
        # module object, so don't set self.graph_module.
        graph_module = clone_graph_module(definition.module)
        self.body, self.head = _split_body_and_head(graph_module)
        if pp_group.size() > 1:
            prepare_pipeline_graph(self.body)
        placement = (placement_from_groups(tp_group=tp_group, pp_group=pp_group, ep_group=ep_group)
                     if has_moe else None)
        self._pruned_experts: frozenset[str] = frozenset()
        if placement is not None and placement.ep.size > 1:
            # Keep only this rank's experts before allocation; their global keys stay unchanged.
            before = {name for name, _ in self.body.named_parameters(remove_duplicate=False)}
            for module in self.body.modules():
                if isinstance(module, MoE):
                    module.narrow_held(placement, site="body")
            after = {name for name, _ in self.body.named_parameters(remove_duplicate=False)}
            self._pruned_experts = frozenset(before - after)
        # Shard meta shapes before allocation so only TP-local storage is allocated.
        shard_parameter_shapes(self, tp_size=tp_group.size(), side="engine", etp_size=self._etp_size)
        set_local_dims(self, tp_size=tp_group.size())
        # Allocate both children together to preserve the embedding/head tie.
        allocate_tensors(self, device=torch.device("cuda", torch.cuda.current_device()))
        # TODO(jiangjiawei1103): Review binding details.
        # Body ops precede cores: they declare the Attention/GDN core and GDN conv
        # slots unbound, which would overwrite cores bound earlier.
        bind_body_ops(self.body, tp_group=tp_group, placement=placement)
        if self.head is not None:
            bind_head_ops(self.head, tp_group=tp_group)
        # Residual join + pre-norm pairs run as one fused kernel: the same pass, on the same captured
        # graph, as the trainer (bit-identical), before CUDA graph capture records the launches.
        from isoexec.compiler.realization.fusion import fuse_residual_norms

        self.fused_residual_norms = fuse_residual_norms(self.body)
        # Prepare packed aliases inside the weight allocation context, before graph capture.
        # Checkpoint loads and native updates then write directly into the packed storage.
        attach_packed_experts(self.body)
        attach_packed_proj(self.body)
        prepare_packed_storage(self.body)
        # Cores follow allocation: GDN state pools borrow the GDN weights, and
        # allocation replaces the meta Parameters they would otherwise capture.
        bind_attention_cores(self.body, vllm_config=vllm_config, prefix=prefix)
        bind_gdn_cores(self.body, vllm_config=vllm_config, prefix=prefix)
        self.training = self.body.training
        self._mrope = uses_mrope(self.body)
        self._embedding_name = next(
            (
                str(node.target)
                for node in self.body.graph.nodes
                if node.op == "call_module" and isinstance(self.body.get_submodule(str(node.target)), Embedding)
            ),
            None,
        )
        # Stage names loading skips or resolves: HF loading uses the stage graph's names, and native
        # updates qualify them by body or head as this model registers them.
        # Experts other EP ranks own are skipped by name, like other stages' parameters.
        self._remote_parameters = definition.remote_parameters | self._pruned_experts

        self._parameter_aliases = dict(definition.parameter_aliases)
        self._remote_updates = frozenset(_registered_name(name) for name in self._remote_parameters)
        self._update_aliases = {_registered_name(name): _registered_name(local)
                                for name, local in definition.parameter_aliases.items()}
        self._loaded_weights: set[nn.Parameter] | None = None
        self._weight_update_failed = False
        # With verification on, an update carries every Parameter twice: the second copy is compared with the
        # bytes the first one wrote, and the update finishes only when every Parameter was compared.
        settings = (getattr(vllm_config, "additional_config", None) or {}).get("isoexec") or {}
        self._verify_weight_updates = bool(settings.get("verify_weight_updates", False))
        self._compared_weights: set[nn.Parameter] = set()
        # The vocab-parallel sampling head is built and probed here, agreed over TP, not at the first step.
        from .vllm_patches import prepare_v2_shard_head

        prepare_v2_shard_head(self)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding_name is None:
            raise ValueError("Only the first PP stage embeds token IDs")
        return self.body.get_submodule(self._embedding_name)(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> IntermediateTensors:
        """Allocate the native profiling buffer for incoming hidden states."""
        from vllm.sequence import IntermediateTensors

        return IntermediateTensors({"hidden_states": torch.empty(
            batch_size, self.vllm_config.model_config.get_hidden_size(), dtype=dtype, device=device,
        )})

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        """Return text positions on three MRoPE rows and a zero position delta."""
        if mm_features:
            raise ValueError("VllmModel supports text-only MRoPE positions")
        positions = torch.arange(len(input_tokens), dtype=torch.int64)
        return positions.unsqueeze(0).expand(3, -1), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        """Return native Mamba page dtypes from the model and cache configuration."""
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return native Mamba page shapes matching the selected CPR page policy."""
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        from isoexec.ops.gdn.shared.gdn_ops import GDN_CPR_MIN_STATE_SHAPES, gdn_cpr_min_pages

        from ._gdn import _selected_needs_native_cache

        # The same rule as GDNStateLayer.get_state_shape: minimal pages only for a provider that keeps
        # its state in a private pool. A provider whose state IS vLLM's mamba cache needs full pages.
        if gdn_cpr_min_pages() and not _selected_needs_native_cache():
            return GDN_CPR_MIN_STATE_SHAPES
        config = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return the native conv and SSM page copy functions."""
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator

        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Compute body forward and return hidden states.

        Request boundaries come from ForwardContext. Metadata-free profiling
        represents one request. Nonfirst PP stages consume ``hidden_states``
        from IntermediateTensors; nonfinal stages return them for the next
        stage. Precomputed input embeddings are unsupported.
        """
        if inputs_embeds is not None:
            raise ValueError("VllmModel forward requires inputs_embeds=None")
        if self._embedding_name is not None and (input_ids is None or intermediate_tensors is not None):
            raise ValueError("The first vLLM stage requires input_ids and no pipeline activations")
        if self._embedding_name is None and intermediate_tensors is None:
            raise ValueError("A nonfirst vLLM stage requires pipeline activations")
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("VllmModel forward requires one native attention metadata mapping")
            metadata = select_sequence_metadata(metadata)
        # TODO(jiangjiawei1103): Review and think about if we really need seq ctx input into gm.
        seq = engine_sequence_context(
            input_ids, positions, metadata, max_model_len=self.vllm_config.model_config.max_model_len, mrope=self._mrope
        )
        if self.vllm_config.parallel_config.pipeline_parallel_size == 1:
            return self.body(input_ids=seq.input_ids, seq=seq)
        hidden = None if intermediate_tensors is None else intermediate_tensors["hidden_states"]
        result = self.body(input_ids=seq.input_ids, seq=seq, hidden_states=hidden)
        if self.head is not None:
            return result
        from vllm.sequence import IntermediateTensors

        return IntermediateTensors({"hidden_states": result})

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Return FP32 logits over the vocabulary, or None on a nonfinal PP stage."""
        if self.head is None:
            return None
        hidden_states = self.head.slots.lm_head_slice(hidden_states)
        logits = self.head.head_output(hidden_states)
        tp_group = self._tp_group
        # The head GEMM already rounds to BF16. Gather those bytes before
        # widening, preserving output bits without doubling collective traffic.
        if tp_group.size() > 1:
            logits = gather_vocab(logits, tp_group, out_dtype=torch.float32)  # gather + widen: one copy
        return self.head.trim_to_vocab(logits.to(torch.float32))

    def isoexec_shard_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """This TP rank's bf16 vocab shard of the selected rows' logits, for the vocab-parallel sampling head
        (vllm_patches.patch_vllm_v2_shard_sampler), or None when the head cannot run shard-local: a nonfinal
        PP stage, padded vocabulary columns, or a vocab/TP geometry the leaf recipe does not split."""
        head = self.head
        if head is None or head.padded_vocab != head.vocab:
            return None
        from isoexec.ops.logprobs import supports

        tp_group = self._tp_group
        world = 1 if tp_group is None else tp_group.size()
        if not supports(head.vocab, world):
            return None
        return head.head_output(head.slots.lm_head_slice(hidden_states))

    def start_weight_update(self) -> None:
        """Begin one native update using registered logical names.

        The native caller must stop generation before this call. A new start
        may replace a failed update; overlapping active updates are rejected.
        """
        if self._loaded_weights is not None and not self._weight_update_failed:
            raise ValueError("A weight update is already active")
        self._loaded_weights = set()
        self._compared_weights = set()
        self._weight_update_failed = False

    def finish_weight_update(self) -> None:
        """Finish only after every registered Parameter has been received.

        With ``verify_weight_updates`` in ``additional_config["isoexec"]``, also only after every Parameter
        was compared with a second copy in the same update, and then print a PASS line for this rank.
        Failure leaves the update active and generation must remain stopped.
        Already copied weights are not rolled back.
        """
        if self._loaded_weights is None:
            raise ValueError("No weight update is active")
        if self._weight_update_failed:
            raise ValueError("Weight update failed; start a complete replacement update")
        missing = {name for name, parameter in self.named_parameters() if parameter not in self._loaded_weights}
        if missing:
            self._weight_update_failed = True
            raise ValueError(f"Weight update is missing parameters: {sorted(missing)}")
        if self._verify_weight_updates:
            uncompared = sorted(name for name, parameter in self.named_parameters()
                                if parameter not in self._compared_weights)
            if uncompared:
                self._weight_update_failed = True
                raise ValueError(f"Weight update verification: {len(uncompared)} Parameters were never compared "
                                 f"with a second copy: {uncompared[:8]}")
        try:
            # Captured graphs retain packed addresses across in-place weight updates.
            refresh_packed_storage(self.body)
        except BaseException:
            self._weight_update_failed = True
            raise
        self._loaded_weights = None
        # Every Parameter holds the update's bytes again: a forward after a level-2 sleep is defined.
        from .weights_sleep import mark_restored

        mark_restored(self)
        if self._verify_weight_updates:
            parameters = [parameter for _, parameter in self.named_parameters()]
            receipt = {"parameters": len(parameters),
                       "bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
                       "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}
            print("[ISOEXEC-WEIGHT-SYNC] PASS " + json.dumps(receipt, sort_keys=True), flush=True)

    @torch.no_grad()
    def _load_parameter_updates(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._weight_update_failed:
            raise ValueError("Weight update failed; start a complete replacement update")
        loaded = self._loaded_weights
        assert loaded is not None
        names = set()
        try:
            # With PP, updates address every stage's logical names; only local Parameters are written.
            for logical_name, tensor in weights:
                if logical_name in self._remote_updates:
                    continue
                local_name = self._update_aliases.get(logical_name, logical_name)
                try:
                    parameter = self.get_parameter(local_name)
                except AttributeError as error:
                    raise ValueError(f"{logical_name}: logical name is not registered") from error
                self._copy_parameter_update(local_name, parameter, tensor, loaded)
                names.add(local_name)
        except BaseException:
            self._weight_update_failed = True
            raise
        return names

    def _copy_parameter_update(
        self, logical_name: str, parameter: nn.Parameter, tensor: torch.Tensor, loaded: set[nn.Parameter],
    ) -> None:
        """Copy this rank's slice of one full logical tensor into its local Parameter."""
        tp_group = self._tp_group
        expected = global_shape(self, logical_name, parameter, tp_size=tp_group.size(), side="engine",
                                etp_size=self._etp_size)
        if tensor.is_meta or tuple(tensor.shape) != expected or tensor.dtype != parameter.dtype:
            raise ValueError(f"{logical_name}: update requires the full tensor shape and registered dtype")
        tensor = slice_weight(self, logical_name, tensor, tp_size=tp_group.size(), tp_rank=tp_group.rank(),
                              side="engine", etp_size=self._etp_size)
        if parameter in loaded:
            incoming = tensor.contiguous().reshape(-1).view(torch.uint8)
            current = parameter.detach().to(device=tensor.device).contiguous().reshape(-1).view(torch.uint8)
            if not torch.equal(incoming, current):
                raise ValueError(f"{logical_name}: conflicting values for a shared Parameter")
            self._compared_weights.add(parameter)
        else:
            parameter.copy_(tensor)
        loaded.add(parameter)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF weights, or logical names during a native update.

        The runtime supplies an iterator over the model's checkpoint weights.
        Weight copies completed before an error are not restored. Native updates
        receive full tensors in logical-name order with matching dtypes; with PP,
        names cover every stage and only local Parameters are written. Coverage
        and tied values are checked across all chunks in that update.

        Returns:
            Loaded logical names qualified by ``body`` or ``head``. 
        """
        if self._loaded_weights is not None:
            return self._load_parameter_updates(weights)
        # HF mapping yields stage names (e.g., "layers.0.mixer.q"), but this model
        # registers body tensors under "body." (e.g., "body.layers.0.mixer.q").
        stage = nn.Module()
        stage._parameters, stage._buffers = self.body._parameters, self.body._buffers
        stage._modules = {**self.body._modules, **({"head": self.head} if self.head is not None else {})}
        tp_group = self._tp_group
        logical_names = load_weights(stage, weights, source=self.source, hf_config=self.hf_config,
                                     tp_size=tp_group.size(), tp_rank=tp_group.rank(), side="engine",
                                     remote=self._remote_parameters, aliases=self._parameter_aliases,
                                     etp_size=self._etp_size)
        names = {_registered_name(name) for name in logical_names}
        self._note_reload_after_discard(names)
        return names

    def _note_reload_after_discard(self, names: set[str]) -> None:
        """A checkpoint reload after a level-2 sleep (vLLM ``reload_weights``) restores the weights too.

        The model leaves the discarded state once the reload (in one call or several) has written every local
        Parameter; a partial reload keeps refusing forwards.
        """
        from .weights_sleep import mark_restored, weights_discarded

        if not weights_discarded(self):
            self.__dict__.pop("_reload_written", None)
            return
        written = self.__dict__.setdefault("_reload_written", set())
        for name in names:
            try:
                written.add(id(self.get_parameter(name)))
            except AttributeError:  # a buffer, or an alias of a Parameter registered under another name
                continue
        if all(id(p) in written for p in self.parameters()):
            self.__dict__.pop("_reload_written", None)
            mark_restored(self)
