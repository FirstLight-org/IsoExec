"""Softmax grouped-query attention over packed tokens. Trainer KV weights split below the head
granule, so ``kv_gather`` reconstructs whole activation heads; engines replicate those weights
instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch
from torch import nn

from isoexec.core.bindings import BoundOp
from isoexec.core.linear import split_columns
from isoexec.core.rope import resolve_rope
from isoexec.nn import _seams
from isoexec.schema.token_layout import TokenLayoutContract, TokenLayoutEffect
from isoexec.nn.component import Component, PlacementError
from isoexec.schema import Placement, SequenceContext, ShardAxis, StateSpec, col, replicated, row, torch_dtype
from isoexec.schema.op_metadata import GQAMetadata

#: Family policy, not an argument: every projection and gamma of this kind is held bf16.
HELD_DTYPE = "bf16"

#: The q/k norm conventions this kind admits, and the output-gate functions. ``rms_zero_centered``
#: holds ``gamma - 1`` and applies ``1 + w`` (Qwen3.5); ``rms`` holds gamma itself (Qwen3). The two
#: read the same bytes two ways, so the convention is recorded on the binding and selects the kernel.
QK_NORMS = ("rms_zero_centered", "rms", None)
OUT_GATES = ("sigmoid", "swish", None)
ROPE_STYLES = ("neox", "interleaved")


class Attention(Component):
    """Softmax grouped-query attention over a packed ``[T, H]`` input.

    Parameters register at global shape in checkpoint order on ``device`` (meta by default);
    placement resolves kernels and local activation geometry. ``heads`` must divide into groups
    of ``kv_heads``.

    ``head_dim`` sets the softmax scale. ``rope_theta``, ``rotary_fraction`` and ``rope_style``
    define the rotation; ``mrope_section`` and ``mrope_interleaved`` define multimodal
    coordinates. ``qkv_bias`` controls projection biases, ``qk_norm`` states the held gamma
    convention, ``eps`` is inside the norm square root, and ``out_gate`` selects the optional
    output gate.
    """

    TENSOR_PLAN_KIND = "attn"

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor | None
    q_bias: torch.Tensor
    k_bias: torch.Tensor
    v_bias: torch.Tensor
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None

    # The q/k norm slot stays unfilled when one binding owns normalization and rotation.
    SLOT_OPS = MappingProxyType(
        {
            "mm": "mm",
            "qk_norm": "norms.qk",
            "qk_rope": "rope.apply",
            "core": "attention.core.gqa",
            "gate": "attention.gate",
            "row_parallel": "collectives.row_parallel",
            "tp_copy": "collectives.tp_copy",
            "kv_gather": "collectives.all_gather",
            "cp_gather": "collectives.cp_gather",
        }
    )
    stateful_op_id = SLOT_OPS["core"]
    SLOTS = tuple(SLOT_OPS)
    NATIVE_LAYOUT = TokenLayoutContract(TokenLayoutEffect.DECODER)
    #: ``forward`` returns its ``row_parallel`` slot's result unmodified (the residual+norm fusion pass may defer
    #: that reduction into the consuming pre-norm; compiler/realization/fusion.py).
    ROW_PARALLEL_OUTPUT = True

    # CP gathers global packed tokens for the core, then returns local rows.
    # SequenceContext carries global cu_seqlens and per-sequence positions for local tokens.
    CONTEXT_PARALLEL = (
        "collectives.cp_gather in front of the core: it gathers every CP rank's tokens into the global "
        "packed stream, keeps this rank's output rows, and the adjoint sums the two contributors"
    )

    #: The name the packed ``q|k|v|gate`` operand is registered under (non-persistent buffer).
    packed_proj_name = "qkv_packed"

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rotary_fraction: float = 1.0,
        rope_style: str = "neox",
        mrope_section: tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
        qkv_bias: bool = False,
        qk_norm: str | None = "rms_zero_centered",
        eps: float = 1e-6,
        out_gate: str | None = "sigmoid",
        device: torch.device | str = "meta",
    ) -> None:
        if min(hidden, heads, kv_heads, head_dim) < 1:
            raise ValueError(
                f"Attention: hidden/heads/kv_heads/head_dim must be >= 1, got {(hidden, heads, kv_heads, head_dim)}"
            )
        if heads % kv_heads:
            raise ValueError(f"Attention: {heads} query heads is not a whole multiple of {kv_heads} kv heads")
        if qk_norm not in QK_NORMS:
            raise ValueError(f"Attention: qk_norm must be one of {QK_NORMS}, got {qk_norm!r}")
        if out_gate not in OUT_GATES:
            raise ValueError(f"Attention: out_gate must be one of {OUT_GATES}, got {out_gate!r}")
        if rope_style not in ROPE_STYLES:
            raise ValueError(f"Attention: rope_style must be one of {ROPE_STYLES}, got {rope_style!r}")
        super().__init__()
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.rope_theta, self.rotary_fraction, self.rope_style = float(rope_theta), float(rotary_fraction), rope_style
        self.mrope_section, self.mrope_interleaved = mrope_section, bool(mrope_interleaved)
        self.qkv_bias, self.qk_norm, self.out_gate = bool(qkv_bias), qk_norm, out_gate
        self.eps = float(eps)
        # WHY: scale is portable semantics of this authored invocation. Materialize its final value
        # once so forward and ModelContract read the exact same float; preserve the existing expression.
        self.spec = GQAMetadata(
            num_q_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=self.rope_theta,
            partial_rotary=self.rotary_fraction,
            rope_style=rope_style,
            mrope_section=mrope_section,
            mrope_interleaved=self.mrope_interleaved,
            qk_norm=qk_norm,
            qkv_bias=self.qkv_bias,
            out_gate=out_gate,
            scale=1.0 / math.sqrt(head_dim),
        )
        # Local activation geometry; the globals until apply_placement narrows them.
        self.heads_local, self.kv_heads_local = heads, kv_heads

        dtype, q_dim, kv_dim = torch_dtype(HELD_DTYPE), heads * head_dim, kv_heads * head_dim
        shapes: list[tuple[str, tuple[int, ...]]] = [
            ("q", (q_dim, hidden)),
            ("k", (kv_dim, hidden)),
            ("v", (kv_dim, hidden)),
        ]
        if out_gate is not None:
            shapes.append(("gate", (q_dim, hidden)))
        if qkv_bias:
            shapes += [("q_bias", (q_dim,)), ("k_bias", (kv_dim,)), ("v_bias", (kv_dim,))]
        if qk_norm is not None:
            shapes += [("q_norm", (head_dim,)), ("k_norm", (head_dim,))]
        shapes.append(("o", (hidden, q_dim)))
        for role, shape in shapes:
            self.register_parameter(role, nn.Parameter(torch.empty(shape, dtype=dtype, device=device)))
        if out_gate is None:
            self.gate = None
        if qk_norm is None:  # the stage takes the weights per call; absent means "no norm"
            self.q_norm = self.k_norm = None

        # Build MRoPE angles through the shared resolver from this component's fields.
        self._rotary = resolve_rope(self.spec)
        #: The free-alias hook for the fused q|k|v|gate operand. ``None`` until a build path calls
        #: ``ops/mm/packed_proj.attach``; ``None`` from the hook means "four separate GEMMs".
        self.packed_proj: Callable[[], torch.Tensor | None] | None = None

    @property
    def scale(self) -> float:
        """The final authored scale stored in canonical op metadata."""
        return self.spec.scale


    def tensor_axes(self) -> Mapping[str, ShardAxis]:
        """Declare head-sharded projections and the row-parallel output axis. Per-head q/k gammas
        are replicated but receive a TP sum of partial weight gradients.
        """
        axes: dict[str, ShardAxis] = {"q": col(), "k": col(units=self.kv_heads), "v": col(units=self.kv_heads)}
        if self.out_gate is not None:
            axes["gate"] = col()
        if self.qkv_bias:
            axes["q_bias"] = col()
            axes["k_bias"] = col(units=self.kv_heads)
            axes["v_bias"] = col(units=self.kv_heads)
        if self.qk_norm is not None:
            axes["q_norm"] = replicated(reduce="sum_over_tp")
            axes["k_norm"] = replicated(reduce="sum_over_tp")
        axes["o"] = row()
        return axes

    def state_specs(self) -> tuple[StateSpec, ...]:
        """Declare one paged KV geometry shared by keys and values."""
        return (
            StateSpec(
                "kv",
                (self.kv_heads, self.head_dim),
                HELD_DTYPE,
                "per_token",
                "all",
                shard=col(units=self.kv_heads),
            ),
        )

    def facts(self) -> Mapping[str, Any]:
        return {
            "kind": "attn",
            "hidden": self.hidden,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rotary_fraction": self.rotary_fraction,
            "rotary_dim": self._rotary.rotary_dim,
            "rope_style": self.rope_style,
            "mrope_section": self.mrope_section,
            "mrope_interleaved": self.mrope_interleaved,
            "qkv_bias": self.qkv_bias,
            "qk_norm": self.qk_norm,
            "eps": self.eps,
            "out_gate": self.out_gate,
            "held_dtype": HELD_DTYPE,
        }


    def _place(self, placement: Placement) -> None:
        site = self.require_placed("_place")
        self.set_local_heads(placement.tp.size)

        # Placement movement, from the Placement this component receives once.
        seams = self.seams
        self.slots.tp_copy = seams.tp_copy()
        self.slots.kv_gather = seams.kv_gather(kv_heads=self.kv_heads)

        # Stateless arithmetic, through the family facades. Arch selection happens here.
        mm = self.bind_op("mm", site=f"{site}.qkv_proj")
        self.slots.mm = mm
        # Not a function of tp: the o_proj K-reduction is the placement's leaf plan at every degree.
        self.slots.row_parallel = seams.row_parallel()
        if self.out_gate is not None:
            # ``gate`` is both: the two gate impls are different functions, so the kind selects,
            # and it is a decision the peer must have made the same way, so it is recorded too.
            self.slots.gate = self.bind_op(
                "attention.gate",
                site=f"{site}.out_gate",
                inputs={"gate": self.out_gate},
                facts={"gate": self.out_gate},
            )
        else:
            self.slots.declare_unbound("gate", "nothing does: this row has no output gate", required=False)
        self._resolve_qk_rope(site)

        # Engine state. The paged core reads the host's KV cache, an object this component must
        # never see, so the slot stays unbound and the adapter's attach_core fills it.
        core = None
        if placement.side == "engine":
            self.slots.declare_unbound("core", "the engine adapter calls attach_core with its cache-aware core")
        else:
            core = self.bind_op(
                "attention.core.gqa", site=f"{site}.core", inputs={"scale": self.scale, "head_dim": self.head_dim}
            )
        # The context-parallel seam last, because it is a function of the core this side resolved:
        # the gather and the core that performs it are one statement, so one call states both. At
        # cp=1 the gather is the recorded identity and the core comes back untouched.
        self.slots.cp_gather, core = seams.cp_attention(core)
        if core is not None:
            self.slots.core = core

    def set_local_heads(self, tp: int) -> None:
        """Derive local activation heads at ``tp``, including whole heads reconstructed by ``kv_gather``."""
        if self.heads % tp:
            raise PlacementError(f"{self.site}: {self.heads} query heads do not divide tp={tp}")
        heads_local = self.heads // tp
        if tp <= self.kv_heads:
            if self.kv_heads % tp:
                raise PlacementError(f"{self.site}: {self.kv_heads} kv heads do not divide tp={tp}")
            kv_local = self.kv_heads // tp
        else:
            # Below the unit granule: whatever a side does with the weight rows, the number of whole
            # kv heads this rank consumes is one -- a trainer splits and `kv_gather` puts the head
            # back together, an engine replicates the head outright.
            if tp % self.kv_heads:
                raise PlacementError(f"{self.site}: tp={tp} is not a multiple of the {self.kv_heads} declared kv heads")
            kv_local = 1
        if heads_local % kv_local:
            raise PlacementError(
                f"{self.site}: local q heads {heads_local} not a multiple of local kv heads {kv_local}"
            )
        self.heads_local, self.kv_heads_local = heads_local, kv_local

    def _resolve_qk_rope(self, site: str) -> None:
        """Resolve q/k normalization before rotation to read fusion ownership from the binding.
        Build one callable for both stages and leave any subsumed slot explicitly unfilled.
        """
        norm = None
        if self.qk_norm is None:
            self.slots.declare_unbound("qk_norm", "nothing does: this row has no q/k norm", required=False)
        else:
            norm = self.bind_op("norms.qk", site=f"{site}.qk_norm", inputs={"eps": self.eps}, convention=self.qk_norm)
            if "rope.apply" in norm.owns:
                self._fused_qk_rope(norm)
                return
            self.slots.qk_norm = norm
        norm_fn = None if norm is None else self.slots.qk_norm
        rope = self.bind_op(
            "rope.apply",
            site=f"{site}.positional",
            inputs={"style": self._rotary.style, "rotary_dim": self._rotary.rotary_dim, "theta": self._rotary.theta},
            # The rope constants are inputs and facts: a table-driven rope kernel builds its cos/sin
            # table from them at resolution, so the family needs them to choose. ``qk_norm`` is a
            # fact alone -- it is how the family knows whether its fused kernel had anything to fuse.
            facts={
                "style": self._rotary.style,
                "rotary_dim": self._rotary.rotary_dim,
                "theta": self._rotary.theta,
                "mrope_section": self.mrope_section,
                "mrope_interleaved": self.mrope_interleaved,
                "qk_norm": self.qk_norm,
            },
        )
        rotary, rope_fn = self._rotary, rope.fn

        def stage(q, k, wq, wk, eps, seq):
            if norm_fn is not None:
                q = norm_fn(q, wq, eps)
                k = norm_fn(k, wk, eps)
            return rope_fn(
                q, k, None, style=rotary.style, rotary_dim=rotary.rotary_dim, seq=seq, rotary=rotary
            )

        # Record every operation computed by this fused or composed callable.
        owns = {"rope.apply": rope.impl}
        if norm is not None:
            owns["norms.qk"] = norm.impl
        self.slots.qk_rope = BoundOp(rope.op, rope.impl, rope.version, rope.provider, rope.inputs, stage, owns=owns)

    def _fused_qk_rope(self, norm: BoundOp) -> None:
        """Wrap a norm binding that owns ``rope.apply``. Record both operations under the fused
        implementation and pass this row's MRoPE layout as data.
        """
        fused, rotary = norm.fn, self._rotary
        spec = SimpleNamespace(mrope_section=self.mrope_section, mrope_interleaved=self.mrope_interleaved)

        def stage(q, k, wq, wk, eps, seq):
            return fused(q, k, wq, wk, eps, seq=seq, rotary=rotary, spec=spec)

        owns = dict(norm.owns)
        self.slots.declare_unbound(
            "qk_norm",
            f"nothing does: the {norm.impl} kernel normalizes q and k and rotates them in one stage, "
            f"so the q/k norm has no binding of its own and the stage owns both op ids",
            required=False,
        )
        self.slots.qk_rope = BoundOp(norm.op, norm.impl, norm.version, norm.provider, norm.inputs, stage, owns=owns)

    def attach_core(self, resolved: BoundOp) -> None:
        """Attach a cache-aware ``attention.core.gqa`` binding with provenance. Refuse replacement
        of an existing core, including a trainer core resolved at placement.
        """
        self.require_placed("attach_core")
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
        hq, hk, d = self.heads_local, self.kv_heads_local, self.head_dim
        kv_gather, mm = self.slots.kv_gather, self.slots.mm
        xc = self.slots.tp_copy(x)
        # The token count the mixer works on is the seam's output: under sequence parallelism the
        # input is this rank's chunk and tp_copy is the sequence all-gather.
        T = xc.shape[0]
        w = self.packed_proj() if self.packed_proj is not None else None
        if w is None:  # no hook installed, a kv_gather seam, or the alias could not be verified
            q = mm(xc, self.q).view(T, hq, d)
            k = kv_gather(mm(xc, self.k)).view(T, hk, d)
            v = kv_gather(mm(xc, self.v)).view(T, hk, d)
            gate = mm(xc, self.gate).view(T, hq, d) if self.gate is not None else None
        else:  # one GEMM; q/k/v/gate are adjacent column views of its output, never copies
            parts = split_columns(mm(xc, w), self.packed_proj_members())
            q, k, v = parts[0].view(T, hq, d), parts[1].view(T, hk, d), parts[2].view(T, hk, d)
            gate = parts[3].view(T, hq, d) if self.gate is not None else None
        if self.qkv_bias:
            q = q + self.q_bias.view(hq, d)
            k = k + kv_gather(self.k_bias.view(1, -1)).view(hk, d)
            v = v + kv_gather(self.v_bias.view(1, -1)).view(hk, d)
        q, k = self.slots.qk_rope(q, k, self.q_norm, self.k_norm, self.eps, seq)
        o = self.slots.core(q, k, v, seq.cu_seqlens, seq.max_q, self.scale)
        if gate is not None:
            o = self.slots.gate(o, gate)
        return self.slots.row_parallel(o.reshape(T, -1), self.o)


    def packed_proj_members(self) -> tuple[torch.Tensor, ...]:
        """The projection parameters the fused GEMM concatenates, in column order."""
        return (self.q, self.k, self.v) if self.gate is None else (self.q, self.k, self.v, self.gate)

    def packed_proj_roles(self) -> tuple[str, ...]:
        """Return projection roles eligible for packing; exclude projections requiring KV gathering."""
        gather = self.slots.resolved().get("kv_gather")
        if gather is None or gather.impl != _seams.NO_MOVEMENT:
            return ()
        return ("q", "k", "v") if self.gate is None else ("q", "k", "v", "gate")

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, heads={self.heads}, kv_heads={self.kv_heads}, head_dim={self.head_dim}, "
            f"qk_norm={self.qk_norm!r}, out_gate={self.out_gate!r}, qkv_bias={self.qkv_bias}"
        )


__all__ = ["HELD_DTYPE", "OUT_GATES", "QK_NORMS", "ROPE_STYLES", "Attention"]
