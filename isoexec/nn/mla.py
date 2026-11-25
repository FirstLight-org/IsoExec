"""Absorbed multi-head latent attention over a replicated latent stream. Rotary lane order is
checkpoint data, and local up-projection heads follow tensor parallelism.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.core.rope import RotaryTables, rotary_freqs
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component, PlacementError
from isoexec.schema import Placement, SequenceContext, ShardAxis, StateSpec, col, replicated, row, torch_dtype
from isoexec.schema.op_metadata import MLAMetadata

#: Family policy, not an argument: every projection and latent gamma of this kind is held bf16.
HELD_DTYPE = "bf16"

#: How the two latent gammas are held. ``rms`` holds gamma itself; ``rms_zero_centered`` holds
#: ``gamma - 1`` and the kernel applies ``1 + w``. It selects the ``norms.mla`` implementation at
#: placement, so the convention and the arithmetic cannot disagree.
NORM_CONVENTIONS = ("rms", "rms_zero_centered")

#: The rope lane layouts a checkpoint can store. ``interleaved`` means adjacent ``(real, imag)``
#: pairs, which are gathered into ``neox`` (rotate-half) coordinates before the rotation runs.
ROPE_STYLES = ("neox", "interleaved")


class LatentAttention(Component):
    """Absorbed latent attention over packed ``[T, H]`` tokens, registered at global shape in
    checkpoint order.

    ``hidden`` is the residual width and ``heads`` is the query-head count. ``q_lora_rank`` and
    ``kv_lora_rank`` size the normalized latent streams; ``qk_nope_head_dim`` and
    ``qk_rope_head_dim`` divide the query/key width into unrotated and rotary lanes.
    ``v_head_dim`` sizes the output contraction.

    ``rope_theta`` sets the rotary base and ``rope_style`` describes checkpoint lane order.
    ``norm_convention`` states how latent gammas are held, with ``eps`` inside the square root.
    ``device`` defaults to meta.
    """

    TENSOR_PLAN_KIND = "attn"

    kv_up: torch.Tensor

    # Latents are replicated, so no KV gather is needed. Resolve the two latent
    # norms separately because their widths can select different implementations.
    SLOT_OPS = MappingProxyType(
        {
            "mm": "mm",
            "q_latent_norm": "norms.mla",
            "kv_latent_norm": "norms.mla",
            "rope": "rope.apply",
            "core": "attention.core.mla",
            "row_parallel": "collectives.row_parallel",
            "tp_copy": "collectives.tp_copy",
        }
    )
    stateful_op_id = SLOT_OPS["core"]
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)
    #: ``forward`` returns its ``row_parallel`` slot's result unmodified (the residual+norm fusion pass may defer
    #: that reduction into the consuming pre-norm; compiler/realization/fusion.py).
    ROW_PARALLEL_OUTPUT = True

    # Only the GQA context-parallel gather is qualified; MLA has no supported CP form.
    NO_CONTEXT_PARALLEL = "only attention.core.gqa has a qualified cp_gather; the absorbed latent core has none"

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_theta: float,
        rope_style: str = "neox",
        norm_convention: str = "rms",
        eps: float = 1e-5,
        device: torch.device | str = "meta",
    ) -> None:
        widths = (hidden, heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
        if min(widths) < 1:
            raise ValueError(f"LatentAttention: every width must be >= 1, got {widths}")
        if rope_style not in ROPE_STYLES:
            raise ValueError(f"LatentAttention: rope_style must be one of {ROPE_STYLES}, got {rope_style!r}")
        if norm_convention not in NORM_CONVENTIONS:
            raise ValueError(
                f"LatentAttention: norm_convention must be one of {NORM_CONVENTIONS}, got {norm_convention!r}"
            )
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.qk_nope_head_dim, self.qk_rope_head_dim = qk_nope_head_dim, qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta, self.rope_style = float(rope_theta), rope_style
        self.norm_convention, self.eps = norm_convention, float(eps)
        # WHY: scale is portable semantics of this authored invocation. Materialize its final value
        # once so forward and ModelContract read the exact same float; preserve the existing expression.
        self.spec = MLAMetadata(
            num_q_heads=heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_theta=self.rope_theta,
            rope_style=rope_style,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        #: Local activation head count; the global one until apply_placement narrows it.
        self.heads_local = heads

        dtype = torch_dtype(HELD_DTYPE)
        shapes: tuple[tuple[str, tuple[int, ...]], ...] = (
            ("q_down", (q_lora_rank, hidden)),
            ("q_norm", (q_lora_rank,)),
            ("q_up", (heads * self.head_dim, q_lora_rank)),
            ("kv_down", (kv_lora_rank + qk_rope_head_dim, hidden)),
            ("kv_norm", (kv_lora_rank,)),
            ("kv_up", (heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)),
            ("o", (hidden, heads * v_head_dim)),
        )
        for role, shape in shapes:
            self.register_parameter(role, nn.Parameter(torch.empty(shape, dtype=dtype, device=device)))

        # Angles describe NeoX rotation after checkpoint lanes are gathered into halves.
        self._rotary = RotaryTables(
            rotary_dim=qk_rope_head_dim,
            style="neox",
            theta=self.rope_theta,
            fn=lambda positions: rotary_freqs(positions, self.rope_theta, qk_rope_head_dim),
        )


    @property
    def head_dim(self) -> int:
        """The query/key width one head carries: the un-rotated half plus the rotated half."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        """The final authored scale stored in canonical op metadata."""
        return self.spec.scale

    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Replicate down projections and latent gammas; shard up projections by head. Latent gamma
        gradients are complete on every rank, so summing over TP would overcount them.
        """
        h = self.heads
        return {
            "q_down": replicated(),
            "q_norm": replicated(),
            "q_up": col(units=h),
            "kv_down": replicated(),
            "kv_norm": replicated(),
            "kv_up": col(units=h),
            "o": row(),
        }

    def state_specs(self) -> tuple[StateSpec, ...]:
        """Declare one replicated ``[c_kv | k_pe]`` cache entry per token. Absorbed attention reads
        latent values and up-projection weights instead of materializing per-head keys and
        values.
        """
        return (
            StateSpec(
                "kv_latent",
                (self.kv_lora_rank + self.qk_rope_head_dim,),
                HELD_DTYPE,
                "per_token",
                "all",
                shard=replicated(),
            ),
        )

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "mla",
            "hidden": self.hidden,
            "heads": self.heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rope_style": self.rope_style,
            "norm_convention": self.norm_convention,
            "eps": self.eps,
            "held_dtype": HELD_DTYPE,
        }


    def _place(self, placement: Placement) -> None:
        site = self.site
        tp = placement.tp.size
        if placement.sp:
            # SP is unsupported: the TP wire follows down projections and would gather
            # latents while head views still use the local token count.
            raise PlacementError(
                f"{site}: this MLA row has no sequence-parallel form. Its column-parallel wire sits after "
                f"the replicated down projections, so an SP gather there would not match the token count "
                f"the head views are built from; place this model with sp=False."
            )
        if self.heads % tp:
            raise PlacementError(
                f"{site}: {self.heads} query heads do not divide tp={tp}; an MLA row's up projections "
                f"split by whole heads and a fractional shard has no absorbed core"
            )
        self.heads_local = self.heads // tp

        # Placement movement, from the Placement this component receives once.
        seams = self.seams
        self.slots.tp_copy = seams.tp_copy()
        self.slots.row_parallel = seams.row_parallel()

        # Stateless arithmetic, through the family facades. Arch selection happens here.
        self.slots.mm = self.bind_op("mm", site=f"{site}.projections")
        # Width selects an admitted kernel; model configuration already records the shape.
        def latent_norm(role: str, width: int) -> BoundOp:
            return self.bind_op(
                "norms.mla",
                site=f"{site}.{role}",
                inputs={"eps": self.eps},
                convention=self.norm_convention,
                facts={"width": width},
            )

        self.slots.q_latent_norm = latent_norm("q_norm", self.q_lora_rank)
        self.slots.kv_latent_norm = latent_norm("kv_norm", self.kv_lora_rank)
        self.slots.rope = self.bind_op(
            "rope.apply",
            site=f"{site}.positional",
            # The kernel rotates NeoX coordinates; facts() records the original checkpoint lane order.
            inputs={"style": "neox", "rotary_dim": self.qk_rope_head_dim, "theta": self.rope_theta},
            facts={
                "style": "neox",
                "rotary_dim": self.qk_rope_head_dim,
                "theta": self.rope_theta,
                "mrope_section": None,
                "mrope_interleaved": False,
                "qk_norm": None,
            },
        )

        # Engine state. The paged core reads the host's latent cache, an object this component must
        # never see, so the slot stays unbound and the adapter's attach_core fills it.
        if placement.side == "engine":
            self.slots.declare_unbound("core", "the engine adapter calls attach_core with its cache-aware core")
        else:
            self.slots.core = self.bind_op(
                "attention.core.mla",
                site=f"{site}.core",
                inputs={"scale": self.scale, "head_dim": self.head_dim, "kv_lora_rank": self.kv_lora_rank},
            )

    def attach_core(self, resolved: BoundOp) -> None:
        """Attach a cache-aware ``attention.core.mla`` binding with provenance. Refuse replacement
        of an existing core.
        """
        self.require_placed("attach_core")
        declared = self.slots.op_of("core")
        if resolved.op != declared:
            raise PlacementError(
                f"{self.site}.core takes {declared!r}, not {resolved.op!r}: a core attached under another "
                f"op id would digest as that op and be compared against the wrong peer entry"
            )
        if "core" not in self.slots.unfilled():
            raise PlacementError(
                f"{self.site}.core is already filled by {self.slots.resolved()['core'].provider}; "
                f"attach_core fills an unbound core, it does not replace a resolved one"
            )
        self.slots.core = resolved

    def bind_stateful_op(self, op_id: str, binding: BoundOp) -> None:
        if op_id != self.stateful_op_id:
            raise PlacementError(f"{self.site}: expected stateful op {self.stateful_op_id!r}, got {op_id!r}")
        if not isinstance(binding, BoundOp) or binding.op != op_id:
            raise PlacementError(f"{self.site}: stateful op {op_id!r} requires a matching BoundOp")
        self.attach_core(binding)


    def forward(self, x: torch.Tensor, seq: SequenceContext) -> torch.Tensor:
        """Map ``[T, H]`` through the replicated latent streams. Apply the TP gradient wire after
        latent norms, where the up projections branch into head shards.
        """
        t, heads, mm = x.shape[0], self.heads_local, self.slots.mm
        nope, rope_dim, v_dim = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        tp_copy = self.slots.tp_copy

        q_latent = self.slots.q_latent_norm(mm(x, self.q_down), self.q_norm, self.eps)
        q = mm(tp_copy(q_latent), self.q_up).view(t, heads, self.head_dim)
        q_nope, q_pe = q.split((nope, rope_dim), dim=-1)

        kv = mm(x, self.kv_down)
        c, k_pe = kv.split((self.kv_lora_rank, rope_dim), dim=-1)
        c = tp_copy(self.slots.kv_latent_norm(c, self.kv_norm, self.eps))
        k_pe = tp_copy(k_pe).unsqueeze(1)

        q_pe, k_pe = self._rotate(q_pe, k_pe, seq)

        # The absorbed form: the core never materializes per-head keys or values, it reads the
        # latent and the two halves of the up-projection weight.
        weight = self.kv_up.view(heads, nope + v_dim, self.kv_lora_rank)
        out = self.slots.core(
            q_nope, q_pe, c, k_pe.squeeze(1), weight[:, :nope], weight[:, nope:], seq, self.scale
        )
        return self.slots.row_parallel(out.reshape(t, heads * v_dim), self.o)

    def _rotate(self, q_pe: torch.Tensor, k_pe: torch.Tensor, seq: SequenceContext):
        """Let the selected rotation binding handle checkpoint lane order and angle tables."""
        return self.slots.rope(
            q_pe, k_pe, None, style="neox", rotary_dim=self.qk_rope_head_dim,
            seq=seq, rotary=self._rotary, input_pairs=self.rope_style,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, heads={self.heads}, q_lora={self.q_lora_rank}, "
            f"kv_lora={self.kv_lora_rank}, qk={self.qk_nope_head_dim}+{self.qk_rope_head_dim}, "
            f"v={self.v_head_dim}, rope_style={self.rope_style!r}, norm={self.norm_convention!r}"
        )


__all__ = ["HELD_DTYPE", "NORM_CONVENTIONS", "ROPE_STYLES", "LatentAttention"]
