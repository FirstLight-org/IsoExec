"""Resolve attention and RoPE implementations without loading kernels at module import."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import integration, kernel_package, provider_of, refuse, resolution

#: Output-gate kinds and their registered implementations.
GATES: dict[str, tuple[str, str]] = {
    "sigmoid": ("compiled_sigmoid_gate_fp32", "attention_gate_sigmoid"),
    "swish": ("eager_swish", "attention_gate_swish"),
}

#: MLA recipes differ in bits; both hosts must use the same selection.
CORE_MLA_FA3_IMPL = "fa3_absorbed_ns1"
CORE_MLA_SEGTREE_IMPL = "sm90_absorbed_segtree"


def core_mla_impl(seg_tokens: int | None) -> str:
    """Select the shared trainer/engine MLA recipe from the model's seg_tokens.
    None selects FA3; a segment span selects the tree recipe and changes its rounding schedule."""
    from isoexec.ops._dispatch import current_arch
    if current_arch() == "sm100":
        if seg_tokens is None:
            raise refuse("attention.core.mla", site="core_mla_impl", asked={"seg_tokens": None},
                         on_offer="seg_tokens = mla_sm100.segment_span(), the schedule policy's span at "
                                  "mla_sm100/spec.py's declarations (ISOEXEC_TARGET_BATCH, the longest context)",
                         why="FA3 absorbed MLA is SM90-only; SM100 uses its own SEG-tree recipe.")
        return "sm100_absorbed_segtree"
    return CORE_MLA_FA3_IMPL if seg_tokens is None else CORE_MLA_SEGTREE_IMPL


def _rope(site: str, *, style, rotary_dim, theta, mrope_section, mrope_interleaved, qk_norm):
    """Resolve the selected architecture's standalone rotation kernel."""
    package = kernel_package("rope", op="rope.apply", site=site)
    integ = integration(package)
    if qk_norm is not None and getattr(integ, "qk_norm_rope", None) is not None:
        raise refuse(
            "rope.apply",
            site=site,
            asked={"qk_norm": qk_norm},
            on_offer="norms.qk",
            why=(
                f"ops/{package} ships ONE kernel for the q/k norm and the rotation and declares it on "
                f"both op ids, so a row with a q/k norm has no standalone rotation to bind -- resolve "
                f"'norms.qk' and read 'rope.apply' off the binding's explicit owns metadata."
            ),
        )
    if style != "neox":
        raise refuse(
            "rope.apply", site=site, asked={"style": style}, on_offer=("neox",),
            why=f"ops/{package} binds rotate-half rows only.",
        )
    constants = (float(theta), int(rotary_dim), mrope_section, bool(mrope_interleaved))
    return resolution(
        "rope.apply", integ.ROPE_IMPL, integ.rope_apply_for(constants),
        provider=provider_of(package), site=site,
    )


def resolve(
    op: str,
    *,
    site: str,
    gate: str | None = None,
    style: str | None = None,
    rotary_dim: int | None = None,
    theta: float | None = None,
    mrope_section: tuple[int, ...] | None = None,
    mrope_interleaved: bool = False,
    qk_norm: str | None = None,
    seg_tokens: int | None = None,
    **unused: Any,
):
    """The attention kernel serving ``op`` on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=("gate", "style", "rotary_dim", "theta",
                                                            "mrope_section", "mrope_interleaved", "qk_norm",
                                                            "seg_tokens"))
    if op == "attention.core.gqa":
        # The trainer's packed core; the engine binds the same package's paged core (runtimes/vllm).
        package = kernel_package("gqa", op=op, site=site)
        integ = integration(package)
        return resolution(op, integ.TRAINER_IMPL, integ.trainer_core, provider=provider_of(package), site=site)

    if op == "attention.core.mla":
        impl = core_mla_impl(seg_tokens)
        if impl == CORE_MLA_FA3_IMPL:
            from isoexec.ops.attention.mla_trainer import mla_core_absorbed

            return resolution(op, impl, mla_core_absorbed, provider="ops.attention", site=site)
        from isoexec.ops.attention.mla_trainer import mla_core_segtree
        from isoexec.ops._dispatch import current_arch

        if current_arch() == "sm100":
            # SM100: the span is the schedule policy's output at the deployment's declarations (the optional
            # target batch among them), on both hosts. A model config whose seg_tokens was written under another
            # declaration is refused here, at resolution, before either party runs a token.
            from isoexec.ops.attention.mla_sm100.fwd import segment_span

            if type(seg_tokens) is not int or seg_tokens != segment_span():
                raise refuse(op, site=site, asked={"seg_tokens": seg_tokens}, on_offer=segment_span(),
                             why="on SM100 seg_tokens is isoexec.ops.schedule_policy's span at "
                                 "mla_sm100/spec.py's declarations (mla_sm100.segment_span()), not a caller's choice.")
        elif type(seg_tokens) is not int or seg_tokens <= 0 or seg_tokens % 128:
            # Reject invalid spans before recording them in the execution digest.
            raise refuse(op, site=site, asked={"seg_tokens": seg_tokens},
                         on_offer="a positive multiple of 128",
                         why="seg_tokens is the span the SEG-tree merges over, not a capacity knob.")

        return resolution(op, impl, mla_core_segtree(seg_tokens, arch=current_arch()), provider="ops.attention", site=site,
                          inputs={"seg_tokens": int(seg_tokens)})

    if op == "attention.gate":
        if gate not in GATES:
            raise refuse(op, site=site, asked={"gate": gate}, on_offer=tuple(GATES),
                         why="the two gate impls are different functions, so neither substitutes for the other.")
        impl, symbol = GATES[gate]
        from isoexec.ops.attention import attention_gate

        return resolution(op, impl, getattr(attention_gate, symbol), provider="ops.attention", site=site)

    if op == "rope.apply":
        return _rope(site, style=style, rotary_dim=rotary_dim, theta=theta, mrope_section=mrope_section,
                     mrope_interleaved=mrope_interleaved, qk_norm=qk_norm)

    raise refuse(op, site=site, asked={},
                 on_offer=("attention.core.gqa", "attention.core.mla", "attention.gate", "rope.apply"),
                 why="the attention family states these op ids and no others.")


__all__ = ["CORE_MLA_FA3_IMPL", "CORE_MLA_SEGTREE_IMPL", "GATES",
           "core_mla_impl", "resolve"]
