"""Trainer bindings for FA3 absorbed MLA and the SM90 segmented-tree recipe.
The recipes have different rounding schedules; resolve.core_mla_impl selects one for both hosts."""

from __future__ import annotations


def _packed_lens(seq):
    """The per-sequence lengths of a packed batch, as host ints."""
    cu = seq.cu_seqlens.tolist()
    return tuple(b - a for a, b in zip(cu, cu[1:]))


def mla_core_absorbed(q_nope, q_pe, c, k_pe, uk, uv, seq, scale):
    """Trainer binding to the FA3 absorbed forward/unabsorbed backward (both through gqa_sm90)."""
    from isoexec.ops.attention.gqa_sm90.ops import flash_attn_paged, varlen_attention
    from isoexec.ops.attention.mla_absorbed import absorbed_mla
    # Trainer pack metadata is currently GPU-resident. This correctness-first
    # path synchronizes once per call; engine adapters bind their paged core.
    return absorbed_mla(q_nope, q_pe, c, k_pe, uk, uv, seq_lens=_packed_lens(seq), scale=scale,
                        flash_attn=flash_attn_paged, varlen_attn=varlen_attention)


def mla_core_segtree(seg_tokens: int, *, arch: str = "sm90"):
    """Bind the SM90 absorbed kernel with seg_tokens fixed at resolution.
    Its call signature matches mla_core_absorbed, but its rounding schedule differs."""
    if arch == "sm100":
        def core(q_nope, q_pe, c, k_pe, uk, uv, seq, scale):
            from isoexec.ops.attention.mla_sm100.ops import mla_sm100_attention
            return mla_sm100_attention(q_nope, q_pe, c, k_pe, uk, uv,
                                       seq_lens=_packed_lens(seq), scale=scale, seg_tokens=seg_tokens)
        core.seg_tokens = seg_tokens
        return core
    if arch != "sm90":
        raise ValueError(f"absorbed MLA is not available on {arch!r}")
    def core(q_nope, q_pe, c, k_pe, uk, uv, seq, scale):
        from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

        return mla_sm90_attention(q_nope, q_pe, c, k_pe, uk, uv,
                                  seq_lens=_packed_lens(seq), scale=scale, seg_tokens=seg_tokens)

    core.seg_tokens = seg_tokens
    return core


__all__ = ["mla_core_absorbed", "mla_core_segtree"]
