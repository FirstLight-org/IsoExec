"""DeepSeek V4.1 Flash text tower: the tech report's Causal Encoder-Decoder (Fig. 3) with IsoExec components.

``embed -> CausalEncoder (every token) -> EncoderOutput (the handoff) -> Decoder: entry (the decoder global-KV
projection, every token) + blocks (the decoder segments' rows) -> final collapse + RMSNorm -> LMHead (FP32)``.

The CED prefill program is part of the model's identity (``config.CED_PREFILL_MODES``): ``bounded_replay``
(release; the decoder on the last ``replay_window`` prompt tokens and on generated tokens) or ``full_decoder``
(reference parity gates). The trainer streams ``program_identity()`` with every weight update and the engine
refuses a mismatch (``isoexec.sync.identity``). The vision tower and the
DSpark rows are excluded by declaration (``CedGeometry.excluded``).
"""
from __future__ import annotations

import torch
from torch import nn

import isoexec.nn as inn
from isoexec.compiler.capture.component_capture import register_capture_component

from .admission import admit_pipeline
from .block import DecoderBlock, DecoderEntry, EncoderBlock, V41Layer
from .config import CED_PREFILL_MODES, RECIPE, RELEASE_CED_PREFILL, DeepseekV41Config, ced_geometry, program_identity
from .embed import DeepSeekV41Embed
from .head import DeepSeekV41HeadStage
from .program import ced_states

#: The package's config class: ``ModelSource.build_config`` parses with it whatever class a runtime has
#: registered for ``deepseek_v41`` (vLLM ships its own, with the text tower flattened).
CONFIG_CLASS = DeepseekV41Config

# Blocks and the embedding take the packed sequence context on the engine (positions, request boundaries).
register_capture_component(DeepSeekV41Embed, sequence_context=True)
register_capture_component(EncoderBlock, sequence_context=True)
register_capture_component(DecoderEntry, sequence_context=True)
register_capture_component(DecoderBlock, sequence_context=True)
register_capture_component(DeepSeekV41HeadStage)


class CausalEncoder(nn.Module):
    """The causal encoder (layers ``[0, b)``), on every token."""

    def __init__(self, cfg, *, recipe: str = RECIPE, device="meta") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(EncoderBlock(cfg, k.layer, recipe=recipe, device=device) for k in cfg.encoder)
        # Cross-layer attention state inside each CSA2 group (FULL -> its REUSE layers), fixed by the config.
        self.state_routes = tuple((block.consumes(), block.produces()) for block in self.blocks)


class Decoder(nn.Module):
    """The decoder (layers ``[b, L)``): ``entry`` projects the decoder global KV from the handoff for every token
    and opens the decoder forward; ``blocks`` compute the decoder segments' rows."""

    def __init__(self, cfg, *, recipe: str = RECIPE, device="meta") -> None:
        super().__init__()
        self.entry = DecoderEntry(cfg, recipe=recipe, device=device)
        self.blocks = nn.ModuleList(DecoderBlock(cfg, k.layer, recipe=recipe, device=device) for k in cfg.decoder)

    def forward(self, x, pre, out, post, comb, seq=None):
        """The handoff (the encoder's carry) -> the decoder's carry at every token (decoder rows computed)."""
        x, pre, out, post, comb, *kv = self.entry(x, pre, out, post, comb, seq=seq)
        for block in self.blocks:
            x, pre, out, post, comb = block(x, pre, out, post, comb, *kv, seq=seq)
        return x, pre, out, post, comb


class DeepseekV41ForCausalLM(nn.Module):
    """The CED program over packed tokens: ``input_ids [T]`` (or ``[1, T]``) -> FP32 logits at the decoder rows.

    The recipe is ``sm100_bf16_actor``: every FP8/FP4 checkpoint tensor is decoded exactly to a BF16 master at
    load, and rollout and trainer run the same components and the same program (``ced_prefill``).
    """

    def __init__(self, hf_config, *, device: str | torch.device = "meta", ced_prefill: str = RELEASE_CED_PREFILL,
                 native: bool = True) -> None:
        super().__init__()
        if ced_prefill not in CED_PREFILL_MODES:
            raise ValueError(f"ced_prefill must be one of {CED_PREFILL_MODES}, got {ced_prefill!r}")
        self.config = hf_config
        cfg = self.cfg = ced_geometry(hf_config, native=native)
        self.recipe = RECIPE
        self.ced_prefill = ced_prefill
        self.embed = DeepSeekV41Embed(cfg, device=device)
        self.encoder = CausalEncoder(cfg, recipe=RECIPE, device=device)
        self.decoder = Decoder(cfg, recipe=RECIPE, device=device)
        self.norm = DeepSeekV41HeadStage(cfg, device=device)
        self.head = inn.LMHead(vocab=cfg.vocab_size, hidden=cfg.hidden_size, logits_dtype="fp32", device=device)
        for member in self.members():
            member.ced_prefill = ced_prefill

    # ---------------------------------------------------------------------------------- structure
    def layers(self) -> list[V41Layer]:
        """Every backbone layer's block in layer order (the decoder FULL layer's block holds its query side)."""
        return [*self.encoder.blocks, *self.decoder.blocks]

    def members(self) -> list:
        """The placed components that admit layouts: every block and the decoder entry."""
        return [*self.encoder.blocks, self.decoder.entry, *self.decoder.blocks]

    def program_identity(self) -> dict:
        return program_identity(self.cfg, self.ced_prefill)

    def program_recorded(self) -> dict:
        from .config import program_recorded
        return program_recorded()

    def state_classes(self):
        """The definition's state classes (owner, record, lifetime, crosses calls), ``program.ced_states``."""
        return ced_states(self.cfg)

    def declare_pipeline_stages(self, stage_starts) -> None:
        """Before placement under PP: the stage starts (validated against the admissible cuts)."""
        admit_pipeline(self.cfg, stage_starts)
        for member in self.members():
            member.stage_starts = tuple(stage_starts)

    def pipeline_layers(self):
        """The pipeline units: the encoder blocks, then the decoder as one unit (no cut inside it)."""
        return [*self.encoder.blocks, self.decoder]

    def pipeline_assignments(self, units: int, pp: int) -> tuple[tuple[int, int], ...]:
        """The vLLM factory's stage split (author hook, ``runtimes/vllm/factory.stage_assignments``): native vLLM's
        layout of the backbone layers (``get_pp_indices`` over ``num_hidden_layers``, honouring
        ``VLLM_PP_LAYER_PARTITION`` in layers per stage, as native), admitted and mapped onto the pipeline units.
        At production depth the PP2 default is the CED cut (20 / 20); a cut inside the decoder or a CSA2 group is
        refused by name here, before anything is built."""
        from .admission import unit_ranges, vllm_layer_partition
        if units != len(self.pipeline_layers()):
            raise ValueError(f"the stage split takes {len(self.pipeline_layers())} pipeline units, got {units}")
        return unit_ranges(self.cfg, vllm_layer_partition(self.cfg, pp))

    def forward(self, input_ids: torch.Tensor, seq=None) -> torch.Tensor:
        """Packed tokens -> FP32 logits: the engine's ``[T]`` -> ``[T, V]`` (vLLM selects its sampled rows from
        the leading token axis: the head runs only on those, after the body/head split), the trainer's ``[1, T]``
        -> ``[1, T, V]``. The decoder rows are the step's (engine) or the rollout segments' (trainer,
        ``training_ced.decoder_segments``); logits of other tokens carry no meaning and are never read."""
        packed = input_ids.dim() == 1
        if packed:
            input_ids = input_ids.view(1, -1)
        # seq goes to every component explicitly: this forward takes seq itself, so the capture binds the root
        # sequence context to this parameter and does not supply an omitted one (the embed's Engram hash needs it)
        embedded = self.embed(input_ids, seq=seq)
        x, pre = embedded[0], embedded[1]
        hashes = embedded[2] if len(embedded) > 2 else None
        out = post = comb = None
        state: dict[str, torch.Tensor] = {}
        for block, (consumes, produces) in zip(self.encoder.blocks, self.encoder.state_routes):
            given = {key: state[key] for key in consumes}
            x, pre, out, post, comb, *produced = block(x, pre, out, post, comb, hashes, seq=seq, **given)
            state.update(zip(produces, produced))
        # the handoff: the encoder carry; decoder.entry settles it (EncoderOutput) and projects the global KV
        x, pre, out, post, comb = self.decoder(x, pre, out, post, comb, seq=seq)
        h = self.norm(x, pre, out, post, comb)
        if packed:
            h = h.view(-1, h.shape[-1])
        return self.head(h)

    # ---------------------------------------------------------------------------------- engine (vLLM)
    @staticmethod
    def engine_settings() -> dict:
        """vLLM settings of the native V4.1 cache layers (``engine.bind_row_caches``)."""
        from .engine import BLOCK_SIZE, KV_CACHE_DTYPE
        return {"kv_cache_dtype": KV_CACHE_DTYPE, "block_size": BLOCK_SIZE, "enable_chunked_prefill": True,
                "attention_backend": None}

    @staticmethod
    def engine_model_state_cls():
        """vLLM's V4.1 model state (the Engram token-history window) with this package's step checks."""
        from .engine_state import model_state_class
        return model_state_class()

    # ---------------------------------------------------------------------------------- trainer (Megatron)
    @staticmethod
    def bind_trainer_state(module: nn.Module, checkpoint_path: str) -> dict:
        """After the trainer's weights load: the tokenizer's compressed token map and the host-resident
        Engram tables mapped into the device address space (``training.bind_trainer_state``)."""
        from .training import bind_trainer_state
        return bind_trainer_state(module, checkpoint_path)

    def prepare_static_ep(self, max_home_rows: int) -> None:
        """Prepare fixed-capacity EP routes once, outside capture, with the same value on every EP peer."""
        for layer in self.layers():
            layer.prepare_static_ep(max_home_rows)

    def use_static_ep(self, enabled: bool) -> bool:
        return any([layer.use_static_ep(enabled) for layer in self.layers()])

    def prepare_expert_workspace(self) -> bool:
        """Engine, EP > 1: the canonical BF16 MegaMoE workspace of every block (collective, after the weights
        load, outside capture, on every EP rank). Returns whether the blocks run MegaMoE."""
        return all([layer.prepare_expert_workspace() for layer in self.layers()])

    def refresh_derived_weights(self) -> int:
        from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import refresh_derived_weights
        return sum(layer.refresh_derived_weights() for layer in self.layers()) + refresh_derived_weights(
            self.decoder.entry)


__all__ = ["CONFIG_CLASS", "CausalEncoder", "Decoder", "DeepseekV41ForCausalLM"]
