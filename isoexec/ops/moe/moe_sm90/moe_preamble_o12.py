"""Engine shared-expert SwiGLU kernel preserving bf16 rounding boundaries.
The shared-expert binding can defer its fc2 reduction to owner combination; width-specific admission checks
fall back to eager before admission or under unsupported conditions."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from isoexec.core.triton_nonftz import div_rn as _nonftz_div_rn

# Pinned tiles, never autotuned.
_GLU_BLOCK_T = 32
# BLOCK_F=64 matches the production TP8 shared width; keep it fixed per process, independent of token count.
_GLU_BLOCK_F = 64
_GLU_WARPS = 4


if HAVE_TRITON:

    @triton.jit
    def _glu_kernel(
        x_ptr,  # [T, 2F] bf16, contiguous
        out_ptr,  # [T, F] bf16, contiguous
        T,
        F,
        OFFSET,  # fp32 scalar, config.glu_linear_offset
        BLOCK_T: tl.constexpr,
        BLOCK_F: tl.constexpr,
        # Gate-only FTZ controls; production uses (DIV_FTZ, DIV_FORM)=(0,0).
        # The sigmoid-shaped numerator exposes subnormal results in the actual kernel.
        DIV_FTZ: tl.constexpr = False,
        DIV_FORM: tl.constexpr = 0,
    ):
        pid_t = tl.program_id(0).to(tl.int64)
        pid_f = tl.program_id(1).to(tl.int64)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
        cols = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)[None, :]
        mask = (rows < T) & (cols < F)

        two_f = 2 * F
        g = tl.load(x_ptr + rows * two_f + cols, mask=mask, other=0.0).to(tl.float32)
        lin = tl.load(x_ptr + rows * two_f + F + cols, mask=mask, other=0.0).to(tl.float32)

        # 1. F.silu on bf16: fp32 x / (1 + exp(-x)), then round. `g * -1.0`, never `-g`.
        num = g if DIV_FORM == 0 else 1.0
        den = 1.0 + libdevice.exp(g * -1.0)
        if DIV_FTZ:
            s = libdevice.div_rn(num, den).to(tl.bfloat16)
        else:
            s = _nonftz_div_rn(num, den).to(tl.bfloat16)
        # 2. x_linear + offset: fp32 add, then round. NOT a no-op at 0.0: (-0.0)+0.0 == +0.0.
        lb = (lin + OFFSET).to(tl.bfloat16)
        # 3. the product: both bf16 operands promote to fp32, multiply, round.
        h = (s.to(tl.float32) * lb.to(tl.float32)).to(tl.bfloat16)

        tl.store(out_ptr + rows * F + cols, h, mask=mask)


def fused_shared_glu(
    x: torch.Tensor, offset: float, *, _div_ftz: bool = False, _div_form: int = 0, _fp_fusion: bool = False
) -> torch.Tensor:
    """``silu(x[..., :F]) * (x[..., F:] + offset)`` in one launch, bitwise-equal to the eager chain.

    ``x`` is ``linear_fc1``'s contiguous ``[..., 2F]`` bf16 output. The underscored keywords are
    GATE-ONLY controls (see ``_glu_kernel``); production never passes them.
    """
    assert x.is_contiguous(), "the chunk halves are read by stride; a non-contiguous fc1 output must fall back"
    assert x.dtype == torch.bfloat16
    lead = x.shape[:-1]
    two_f = x.shape[-1]
    assert two_f % 2 == 0
    f = two_f // 2
    x2 = x.view(-1, two_f)
    t = x2.shape[0]
    out = torch.empty((t, f), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(t, _GLU_BLOCK_T), triton.cdiv(f, _GLU_BLOCK_F))
    _glu_kernel[grid](
        x2,
        out,
        t,
        f,
        float(offset),
        BLOCK_T=_GLU_BLOCK_T,
        BLOCK_F=_GLU_BLOCK_F,
        DIV_FTZ=bool(_div_ftz),
        DIV_FORM=int(_div_form),
        num_warps=_GLU_WARPS,
        enable_fp_fusion=bool(_fp_fusion),
    )
    return out.view(*lead, f)


# The engine GLU epilogue preserves separate SiLU, offset-add and product rounds.
# Admit per (width,dtype) by comparing live int16 bits; a refusal keeps the eager path.
# No collective depends on the choice, so admission is rank-local.
_GLU_ADMIT: dict = {}
_GLU_COUNTS = {"calls": 0, "served": 0, "admitted": 0, "rejected": 0, "eager": 0, "capture_skips": 0}
_GLU_OFFSET = 0.0  # mcore glu_linear_offset on every target profile; the eager chain spells it `+ 0.0`


def engine_glu_counts() -> dict:
    return {**_GLU_COUNTS, "widths": {str(k): v for k, v in _GLU_ADMIT.items()}}


def _bits16(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def engine_shared_glu_epilogue(eager_epilogue):
    """``fused_shared_glu`` with ``eager_epilogue`` (``host.shared_glu_epilogue``) as reference and
    fallback. Takes the kernel only for a contiguous bf16 ``[..., 2F]`` CUDA operand without grad,
    and only after the width's admission compare passed; under CUDA-graph capture an unadmitted
    width runs eager (the compare needs a host sync) and is admitted the next time it is seen
    eagerly, which vLLM's per-size warmup guarantees before capture."""

    def epilogue(inter: torch.Tensor) -> torch.Tensor:
        _GLU_COUNTS["calls"] += 1
        if (
            not HAVE_TRITON
            or not inter.is_cuda
            or inter.dtype != torch.bfloat16
            or not inter.is_contiguous()
            or inter.shape[-1] % 2
            or (torch.is_grad_enabled() and inter.requires_grad)
        ):
            _GLU_COUNTS["eager"] += 1
            return eager_epilogue(inter)
        key = (int(inter.shape[-1]) // 2, str(inter.dtype).split(".")[-1])
        verdict = _GLU_ADMIT.get(key)
        if verdict is False:
            _GLU_COUNTS["eager"] += 1
            return eager_epilogue(inter)
        if verdict is None:
            if torch.cuda.is_current_stream_capturing():
                _GLU_COUNTS["capture_skips"] += 1
                _GLU_COUNTS["eager"] += 1
                return eager_epilogue(inter)
            ref = eager_epilogue(inter)
            try:
                got = fused_shared_glu(inter, _GLU_OFFSET)
                ok = got.shape == ref.shape and got.dtype == ref.dtype and bool(torch.equal(_bits16(ref), _bits16(got)))
                why = "" if ok else "BIT PATTERNS DIFFER"
            except Exception as exc:  # noqa: BLE001 -- the eager chain is always the answer
                ok, why = False, f"kernel raised {exc!r}"
            _GLU_ADMIT[key] = ok
            if ok:
                _GLU_COUNTS["admitted"] += 1
                print(
                    f"[ISOEXEC-MOE] fused shared-expert glu ADMITTED at F={key[0]} ({key[1]}): int16 bit patterns "
                    "of silu(x_glu) * (x_linear + 0.0) (3 ATen launches) == fused_shared_glu (1 launch) on the live "
                    "operand. Served from here on for this width; moe_preamble_o12.engine_glu_counts().",
                    flush=True,
                )
            else:
                _GLU_COUNTS["rejected"] += 1
                print(
                    f"[ISOEXEC-MOE] fused shared-expert glu REFUSED at F={key[0]} ({key[1]}): {why}. This width "
                    "runs the eager chain from now on -- the result is unaffected, the two launches are not saved.",
                    flush=True,
                )
            _GLU_COUNTS["eager"] += 1
            return ref  # bit-identical to `got` when admitted; the reference either way
        _GLU_COUNTS["served"] += 1
        return fused_shared_glu(inter, _GLU_OFFSET)

    epilogue.__name__ = epilogue.__qualname__ = "engine_shared_glu_epilogue"
    return epilogue


# The engine's ``moe.shared_expert``: the eager region, able to hand its fc2 tree to the combine.


def engine_shared_expert_owner(eager_fn, *, mm=None, gate_mm=None):
    """Run the eager shared expert with an admitted one-launch GLU epilogue.
    A partial_only row_parallel seam stages fc2 leaves and returns SharedExpertPartial with its finish callback;
    ordinary seams finish eagerly. Forward tp_copy unchanged so gradient-bearing callers retain its adjoint.
    ``mm`` / ``gate_mm`` are the fc1 and gate GEMMs ``eager_fn`` uses, for the staged path; None is the
    trainer's ``mm``."""
    import inspect

    from isoexec.ops.collectives.trainer import _identity, row_parallel_trainer
    from isoexec.ops.mm.trainer import mm_trainer
    from isoexec.ops.moe.carriers import SharedExpertPartial
    from isoexec.ops.moe.eager import shared_expert_gate_score, shared_expert_glu, shared_glu_epilogue

    glu = engine_shared_glu_epilogue(shared_glu_epilogue)
    fc1_mm = mm_trainer if mm is None else mm
    score_mm = mm_trainer if gate_mm is None else gate_mm
    try:
        eager_takes_glu = "glu_epilogue" in inspect.signature(eager_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - a callable with no introspectable signature
        eager_takes_glu = False

    def shared(x, gate, up, down, expert_gate, *, row_parallel=row_parallel_trainer, tp_copy=_identity):
        if not getattr(row_parallel, "partial_only", False):
            kw = {"row_parallel": row_parallel, "tp_copy": tp_copy}
            if eager_takes_glu:
                kw["glu_epilogue"] = glu
            return eager_fn(x, gate, up, down, expert_gate, **kw)
        from isoexec.ops.moe.moe_sm90.moe_pik_combine_owner import stage_shared_partial

        h = shared_expert_glu(tp_copy(x), gate, up, epilogue=glu, mm=fc1_mm)
        T, H = int(h.shape[0]), int(down.shape[0])
        # ``out=`` is the combine's own symmetric staging when it grants one, and ``None`` -- the
        # allocating path -- whenever it cannot prove the bytes are the ones it would have copied.
        dst = stage_shared_partial(T, H, row_parallel.wire_dtype(), row_parallel.group, x.device)
        part = row_parallel(h, down, out=dst)
        # The gate GEMM contracts the FULL hidden width on every rank and reads the raw stream, the
        # same line ``shared_expert_megatron`` draws: its input gradient is already complete.
        g = shared_expert_gate_score(x, expert_gate, mm=score_mm) if expert_gate is not None else None
        return SharedExpertPartial(partial=part, gate=g, finish=row_parallel.finish)

    shared.__name__ = shared.__qualname__ = "engine_shared_expert_owner"
    shared.accepts_partial_row_parallel = True  # type: ignore[attr-defined]
    return shared
