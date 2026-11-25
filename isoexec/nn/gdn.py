"""Gated delta-net mixer with packed projections and placement-resolved stages. Convolution shards
follow separate q/k/v head segments; recurrent state resumes only at kernel-defined chunk
boundaries, and context parallelism re-cuts the mixer by heads (``collectives.cp_heads``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from isoexec.core.bindings import BoundOp, no_movement
from isoexec.core.linear import packed_view, split_columns, split_widths
from isoexec.nn._slots import Unbound
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component, PlacementError
from isoexec.schema import Placement, SequenceContext, ShardAxis, StateSpec, col, replicated, row, torch_dtype
from isoexec.schema.op_metadata import GDNMetadata

# Parameters are held bf16. A_log is fp32 in HF; the output gamma is loaded
# from plain fp32 gamma as bf16 (gamma - 1), matching trainer-held values.
HELD_DTYPE = "bf16"

# The held gamma convention selects the matching gated output-norm implementation.
OUT_NORMS = ("gated_rms_zero_centered", "gated_rms")

# Every core must own q/k l2 normalization and gating.
CORE_FUSED_OPS = ("gdn.l2norm", "gdn.gating")


class GDN(Component):
    """Gated delta-net mixer over packed tokens, with parameters registered at global shape in
    checkpoint order.

    ``key_heads`` and ``key_head_dim`` size q/k; ``value_heads`` and ``value_head_dim`` size
    v/z. Value heads must be a multiple of key heads. ``conv_kernel`` is the causal q/k/v filter
    width, ``out_norm`` states the held gamma convention, and ``eps`` is inside its square root.
    ``hidden`` is the residual width; ``device`` defaults to meta. The core owns its chunk
    length.
    """

    TENSOR_PLAN_KIND = "gdn"

    # Convolution is a separate slot only when the bound core does not own it.
    SLOT_OPS = MappingProxyType(
        {
            "mm": "mm",
            "conv": "conv.short",
            "core": "gdn.core",
            "out_norm": "norms.gated_out",
            "row_parallel": "collectives.row_parallel",
            "tp_copy": "collectives.tp_copy",
            "cp_heads": "collectives.cp_heads",
        }
    )
    stateful_op_id = SLOT_OPS["core"]
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)
    #: ``forward`` returns its ``row_parallel`` slot's result unmodified (the residual+norm fusion pass may defer
    #: that reduction into the consuming pre-norm; compiler/realization/fusion.py).
    ROW_PARALLEL_OUTPUT = True

    # The recurrent state and the conv halo cannot cross a token cut, so CP re-cuts by heads.
    CONTEXT_PARALLEL = (
        "collectives.cp_heads around the convolution and the core: one all-to-all hands every rank the whole "
        "packed sequence for 1/cp of the heads, the per-head program runs unchanged, and a second all-to-all "
        "returns this rank's tokens for every head; the adjoints are the reverse all-to-alls"
    )

    #: The name the packed ``q|k|v|z|beta|alpha`` operand is registered under (non-persistent buffer).
    packed_proj_name = "in_proj_packed"

    def __init__(
        self,
        *,
        hidden: int,
        key_heads: int,
        value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel: int,
        out_norm: str = "gated_rms_zero_centered",
        output_activation: str = "silu",
        eps: float = 1e-6,
        device: torch.device | str = "meta",
    ) -> None:
        sizes = (hidden, key_heads, value_heads, key_head_dim, value_head_dim, conv_kernel)
        if min(sizes) < 1:
            raise ValueError(
                f"GDN: hidden/key_heads/value_heads/key_head_dim/value_head_dim/conv_kernel must be >= 1, got {sizes}"
            )
        if value_heads % key_heads:
            raise ValueError(f"GDN: {value_heads} value heads is not a whole multiple of {key_heads} key heads")
        if out_norm not in OUT_NORMS:
            raise ValueError(f"GDN: out_norm must be one of {OUT_NORMS}, got {out_norm!r}")
        if output_activation not in ("silu", "sigmoid"):
            raise ValueError(f"GDN: output_activation must be silu or sigmoid, got {output_activation!r}")
        super().__init__()
        self.hidden, self.key_heads, self.value_heads = hidden, key_heads, value_heads
        self.key_head_dim, self.value_head_dim, self.conv_kernel = key_head_dim, value_head_dim, conv_kernel
        self.eps = float(eps)
        # The convention cannot be `self.out_norm`: that name belongs to the gamma parameter, which
        # is what the forward hands the kernel. The suffix says which of the two a reader has.
        self.out_norm_convention = out_norm
        self.output_activation = output_activation
        self.spec = GDNMetadata(
            num_key_heads=key_heads,
            num_value_heads=value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel=conv_kernel,
        )
        # Local activation geometry; the globals until apply_placement narrows them.
        self.key_heads_local, self.value_heads_local = key_heads, value_heads
        self.key_dim_local, self.value_dim_local = self.key_dim, self.value_dim

        dtype, kd, vd = torch_dtype(HELD_DTYPE), self.key_dim, self.value_dim
        shapes: tuple[tuple[str, tuple[int, ...]], ...] = (
            ("q", (kd, hidden)),
            ("k", (kd, hidden)),
            ("v", (vd, hidden)),
            ("z", (vd, hidden)),
            ("beta", (value_heads, hidden)),
            ("alpha", (value_heads, hidden)),
            ("conv_w", (2 * kd + vd, 1, conv_kernel)),
            ("A_log", (value_heads,)),
            ("dt_bias", (value_heads,)),
            ("out_norm", (value_head_dim,)),
            ("o", (hidden, vd)),
        )
        for role, shape in shapes:
            self.register_parameter(role, nn.Parameter(torch.empty(shape, dtype=dtype, device=device)))

        #: The free-alias hook for the fused in-projection operand. ``None`` until a build path calls
        #: ``ops/mm/packed_proj.attach``; ``None`` from the hook means "fall back to ``packed_view``".
        self.packed_proj: Callable[[], torch.Tensor | None] | None = None

    @property
    def key_dim(self) -> int:
        """The global q/k channel width. The local one is ``key_dim_local``."""
        return self.key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        """The global v/z channel width. The local one is ``value_dim_local``."""
        return self.value_heads * self.value_head_dim


    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Shard q/k by key heads and v/z by value heads. Declare ``conv_w`` as separate q, k and v
        segments because their head counts can differ. The replicated output gamma receives a TP
        sum of partial weight gradients.
        """
        hk, hv = self.key_heads, self.value_heads
        colk, colv = col(units=hk), col(units=hv)
        return {
            "q": colk,
            "k": colk,
            "v": colv,
            "z": colv,
            "beta": colv,
            "alpha": colv,
            "conv_w": col(segments=((self.key_dim, hk), (self.key_dim, hk), (self.value_dim, hv))),
            "A_log": colv,
            "dt_bias": colv,
            "out_norm": replicated(reduce="sum_over_tp"),
            "o": row(),
        }

    def state_specs(self) -> tuple[StateSpec, ...]:
        """Declare separate q/k/v convolution tails and fp32 recurrent state. ``reuse="align"``
        requires chunk-boundary resume points on the same per-request grid as training.
        """
        hk, hv, dk, dv, w = (
            self.key_heads,
            self.value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel,
        )
        colk, colv = col(units=hk), col(units=hv)
        return (
            StateSpec("conv_q", (hk, dk, w), "bf16", "per_request", "align", shard=colk),
            StateSpec("conv_k", (hk, dk, w), "bf16", "per_request", "align", shard=colk),
            StateSpec("conv_v", (hv, dv, w), "bf16", "per_request", "align", shard=colv),
            StateSpec("ssm", (hv, dk, dv), "fp32", "per_request", "align", shard=colv),
        )

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "gdn",
            "hidden": self.hidden,
            "key_heads": self.key_heads,
            "value_heads": self.value_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "conv_kernel": self.conv_kernel,
            "out_norm": self.out_norm_convention,
            "output_activation": self.output_activation,
            "eps": self.eps,
            "held_dtype": HELD_DTYPE,
        }


    def _place(self, placement: Placement) -> None:
        site = self.require_placed("_place")
        self._set_local_dims(placement)

        # Placement movement, from the Placement this component receives once. There is no gather
        # seam: see _set_local_dims for what that costs a trainer.
        seams = self.seams
        self.slots.tp_copy = seams.tp_copy()
        self.slots.row_parallel = seams.row_parallel()
        # The CP region re-cuts this TP rank's heads: the in-projection columns it receives are local.
        self.slots.cp_heads = seams.cp_heads(
            key_heads=self.key_heads_local, value_heads=self.value_heads_local,
            key_head_dim=self.key_head_dim, value_head_dim=self.value_head_dim,
        )

        # Stateless arithmetic, through the family facades. Arch selection happens here.
        self.slots.mm = self.bind_op("mm", site=f"{site}.in_proj")
        self.slots.out_norm = self.bind_op(
            "norms.gated_out",
            site=f"{site}.out_norm",
            inputs={"eps": self.eps},
            convention=self.out_norm_convention,
            # The out-norm normalizes over one value head, so the width the norm family has to
            # admit is the value head's, not the row's output width. It is a fact and not an
            # input: see ``isoexec.ops.table.OpTable.bind``.
            facts={"width": self.value_head_dim, "activation": self.output_activation},
        )

        # The engine adapter supplies the state-bearing core and settles convolution ownership.
        if placement.side == "engine":
            self.slots.declare_unbound("core", "the engine adapter calls attach_core with its cache-aware core")
            self.slots.declare_unbound(
                "conv",
                "attach_core settles it: a core that reads the paged conv window computes conv.short "
                "in-kernel and owns it, and otherwise attach_core fills this slot",
            )
        else:
            self._resolve_core(site)

    def _set_local_dims(self, placement: Placement) -> None:
        """Derive local activation heads and widths. Engines replicate whole heads below the
        granule; trainers refuse that layout because GDN has no head-reassembly seam.
        """
        tp, side = placement.tp.size, placement.side
        hk = self._local_units("key", self.key_heads, tp, side)
        hv = self._local_units("value", self.value_heads, tp, side)
        if hv % hk:
            raise PlacementError(f"{self.site}: local value heads {hv} not a multiple of local key heads {hk}")
        self.key_heads_local, self.value_heads_local = hk, hv
        self.key_dim_local, self.value_dim_local = hk * self.key_head_dim, hv * self.value_head_dim

    def _local_units(self, role: str, units: int, tp: int, side: str) -> int:
        if tp <= units:
            if units % tp:
                raise PlacementError(f"{self.site}: {units} {role} heads do not divide tp={tp}")
            return units // tp
        if tp % units:
            raise PlacementError(f"{self.site}: tp={tp} is not a multiple of the {units} declared {role} heads")
        if side == "trainer":
            raise PlacementError(
                f"{self.site}: tp={tp} is below the {units} {role} heads of a GDN row, and a GDN row has no "
                f"gather seam -- there is nothing that could reassemble a head split across "
                f"{tp // units} ranks, and replicating the weight would leave those ranks each holding a "
                f"different partial gradient for the same bytes, reduced nowhere. Refusing rather than "
                f"planning a split whose activation geometry no seam restores"
            )
        return 1  # an engine holds no gradients: tp // units ranks replicate the same whole head


    def _resolve_core(self, site: str) -> None:
        """Resolve the trainer core and read its fused-operation ownership. Every core owns q/k l2
        normalization and gating; convolution ownership is binding-specific.
        """
        op = self.slots.op_of("core")
        resolved = self.bind_op(op, site=f"{site}.core")
        # Read complete fusion ownership from this bound callable, including its own op.
        owns = dict(resolved.owns) or {op: resolved.impl}
        self._refuse_unowned_stages(owns)
        # No inputs: every operand of the delta rule is a tensor the forward hands it, and the one
        # scalar it has -- the chunk length -- is the kernel's pin. An empty record is the statement.
        self.slots.core = BoundOp(resolved.op, resolved.impl, resolved.version, resolved.provider, {}, resolved.fn, owns=owns)
        self._settle_conv(site, owns)

    def _refuse_unowned_stages(self, owns: Mapping[str, str]) -> None:
        missing = [op for op in CORE_FUSED_OPS if op not in owns]
        if missing:
            raise PlacementError(
                f"{self.site}.core: {owns.get(self.slots.op_of('core'), '?')!r} does not claim {missing}; "
                f"the GDN core normalizes and gates IN-KERNEL, so a core that does not own those op ids "
                f"leaves them with no owner in the composition and nothing to compare against the peer"
            )

    def _settle_conv(self, site: str, owns: Mapping[str, str]) -> None:
        """Bind the convolution, or record that the core computes it."""
        op = self.slots.op_of("conv")
        if op in owns:
            self.slots.declare_unbound(
                "conv",
                f"nothing does: the {owns[op]} core reads the paged conv window and computes {op} in-kernel",
                required=False,
            )
        else:
            # Resolve a separate convolution only after the core has declared its ownership.
            self.slots.conv = self.bind_op(op, site=f"{site}.conv")

    def attach_core(self, resolved: BoundOp) -> None:
        """Attach a state-bearing ``gdn.core`` binding with provenance, refusing an existing core.
        Use the attached ownership metadata to settle the convolution slot.
        """
        site = self.require_placed("attach_core")
        if "core" not in self.slots.unfilled():
            raise PlacementError(
                f"{self.site}.core is already filled by {self.slots.resolved()['core'].provider}; "
                f"attach_core fills an unbound core, it does not replace a resolved one"
            )
        owns = dict(resolved.owns)
        self._refuse_unowned_stages(owns)
        self.slots.core = resolved
        self._settle_conv(site, owns)

    def bind_stateful_op(self, op_id: str, binding: BoundOp) -> None:
        """Validate op_id and the BoundOp, then delegate installation to attach_core()."""
        if op_id != self.stateful_op_id:
            raise PlacementError(f"{self.site}: expected stateful op {self.stateful_op_id!r}, got {op_id!r}")
        if not isinstance(binding, BoundOp) or binding.op != op_id:
            raise PlacementError(f"{self.site}: stateful op {op_id!r} requires a matching BoundOp")
        self.attach_core(binding)


    def forward(self, x: torch.Tensor, seq: SequenceContext) -> torch.Tensor:
        xc = self.slots.tp_copy(x)
        hk, hv = self.key_heads_local, self.value_heads_local
        dk, dv = self.key_head_dim, self.value_head_dim
        kd, vd = self.key_dim_local, self.value_dim_local
        weights = self.packed_proj_members()
        w = self.packed_proj() if self.packed_proj is not None else None
        if w is None:  # no hook installed, or the alias could not be verified this forward
            w = packed_view(*weights)
        # one GEMM; the operands are adjacent column views of its output, never copies.
        proj = self.slots.mm(xc, w)
        heads = self.slots.cp_heads
        if heads is not no_movement:
            # Head-wise context parallelism: the region runs _mix on every token of this rank's heads.
            q, k, v, z, beta, alpha = split_columns(proj, weights)
            o = heads(proj, self._mix, self.conv_w, self.A_log, self.dt_bias, seq)
            y = self.slots.out_norm(o, z.unflatten(-1, (hv, dv)), self.out_norm, self.eps)
            return self.slots.row_parallel(y, self.o)
        conv = self.slots.conv
        # The slot is the ownership record: an Unbound conv means the core computes it in-kernel,
        # which _settle_conv established from the resolution rather than from the side.
        if not isinstance(conv, Unbound):
            # The conv reads the adjacent q|k|v columns as ONE window of the GEMM output (no cat): the same
            # values in the same order, and its gradient lands in that window's columns as the cat's did.
            widths = tuple(int(t.shape[0]) for t in weights)
            qkv, z, beta, alpha = split_widths(proj, (sum(widths[:3]), *widths[3:]))
            qkv = conv(qkv, self.conv_w, seq.cu_seqlens)
            q, k, v = split_widths(qkv, (kd, kd, vd))
            q, k, v = q.unflatten(-1, (hk, dk)), k.unflatten(-1, (hk, dk)), v.unflatten(-1, (hv, dv))
        else:  # the core owns the conv: q/k/v stay the adjacent column views of the one GEMM
            q, k, v, z, beta, alpha = split_columns(proj, weights)
            q, k, v = q.unflatten(-1, (hk, dk)), k.unflatten(-1, (hk, dk)), v.unflatten(-1, (hv, dv))
        metadata = {} if seq.cu_seqlens_cpu is None else {"cu_seqlens_cpu": seq.cu_seqlens_cpu}
        o = self.slots.core(q, k, v, alpha, beta, self.A_log, self.dt_bias, seq.cu_seqlens, **metadata)
        y = self.slots.out_norm(o, z.unflatten(-1, (hv, dv)), self.out_norm, self.eps)
        return self.slots.row_parallel(y, self.o)


    def _mix(self, qkv, beta, alpha, conv_w, A_log, dt_bias, seq):
        """Convolution and delta rule over packed ``qkv`` columns of whole heads (any head subset)."""
        hv, dk, dv = beta.shape[1], self.key_head_dim, self.value_head_dim
        hk = hv * self.key_heads // self.value_heads
        kd = hk * dk
        qkv = self.slots.conv(qkv, conv_w, seq.cu_seqlens)
        q = qkv[:, :kd].unflatten(-1, (hk, dk))
        k = qkv[:, kd : 2 * kd].unflatten(-1, (hk, dk))
        v = qkv[:, 2 * kd :].unflatten(-1, (hv, dv))
        metadata = {} if seq.cu_seqlens_cpu is None else {"cu_seqlens_cpu": seq.cu_seqlens_cpu}
        return self.slots.core(q, k, v, alpha, beta, A_log, dt_bias, seq.cu_seqlens, **metadata)

    def packed_proj_roles(self) -> tuple[str, ...]:
        """The in_proj members, in the order ``packed_view`` concatenates them."""
        return ("q", "k", "v", "z", "beta", "alpha")

    def packed_proj_members(self) -> tuple[torch.Tensor, ...]:
        """The live parameters those roles name, read per call rather than cached: DDP may re-map
        ``param.data`` after wrap, so a module reads the parameter at the moment it uses it (A41)."""
        return tuple(getattr(self, role) for role in self.packed_proj_roles())

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, key_heads={self.key_heads}, value_heads={self.value_heads}, "
            f"key_head_dim={self.key_head_dim}, value_head_dim={self.value_head_dim}, "
            f"conv_kernel={self.conv_kernel}, out_norm={self.out_norm_convention!r}"
        )


__all__ = ["CORE_FUSED_OPS", "GDN", "HELD_DTYPE", "OUT_NORMS"]
