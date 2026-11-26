"""The domain of the sparse attention as an explicit argument of one launch.

    O[r, h, :] = bf16( sum over dom(r) of softmax(scale * q[r, h] . key) * value(key) ),  normaliser += exp(sink[h]) if a sink is given

    dom(r) = window(r) ++ selected(r)        each part an ordered key list of fixed capacity, -1 = an empty position

    domain              window capacity W   selected capacity k   sink      model
    window              > 0                 0                     either    SWA: DeepSeek V4.1 ratio-0 layers
    window + selected   > 0                 > 0                   yes       SWA + CSA: DeepSeek V4.1 compressed layers
    selected            0                   > 0                   no        DSA: V3.2 / GLM-5

W, k (multiples of the 64-position block; 0 allowed, not both) and the sink are read off the arguments: there is one
entry, :func:`sparse_attention`, and no code path per model. Every domain is the same program: blocks of 64 list
positions in list order, one continued chain (the tile reducer has no merge), an empty position neutral. A part's
optional per-row length says how many leading positions are occupied; the blocks past it hold only empty positions
and are not run (a step program: same bits, ``tests/test_domains_gpu.py``).

Which LEAF runs is a recipe, chosen by name and never by shape:

* ``leaf="dualrail"`` (``sm100_sparse_dualrail[_sink]_v1``): keys 512 wide (value = the key row), at most 640 list
  positions in all, BF16 rows or the V4.1 FP8 records (window and selected may then be two caches);
* ``leaf="segmented"`` (``sm100_sparse_segtree_v1`` / ``_v2`` with ``recipe=2``): the selected-only domain without a
  sink, lists of up to 2048 ascending positions, BF16 keys 512 wide or 576 wide with the value in the first 512
  (the V3.2 / GLM-5 layout), or the V4.1 FP8 cache through a block table (512 only).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Domain", "domain_of", "sparse_attention"]

BLOCK = 64


@dataclass(frozen=True)
class Domain:
    """Capacities, in list positions, of the two parts of the key list, and whether the normaliser has a sink."""

    window: int
    selected: int
    sink: bool

    def __post_init__(self):
        if self.window < 0 or self.selected < 0 or self.window % BLOCK or self.selected % BLOCK:
            raise ValueError(f"capacities must be non-negative multiples of {BLOCK} (got {self.window}, {self.selected})")
        if self.window + self.selected == 0:
            raise ValueError("the domain is empty: window and selected capacities are both 0")

    @property
    def name(self) -> str:
        return "window+selected" if self.window and self.selected else "window" if self.window else "selected"


def _part(part, what):
    if part is None:
        return None, None, 0
    kv, lists = part
    if lists.dim() != 2 or lists.dtype != torch.int32:
        raise ValueError(f"{what} lists: int32 [rows, capacity]")
    return kv, lists, int(lists.shape[1])


def domain_of(window=None, selected=None, sink=None) -> Domain:
    """The domain a launch with these arguments has."""
    return Domain(_part(window, "window")[2], _part(selected, "selected")[2], sink is not None)


def sparse_attention(q, softmax_scale, *, window=None, selected=None, sink=None, window_length=None,
                     selected_length=None, leaf: str = "dualrail", out=None, lse=None, **leaf_options):
    """``(out bf16 [R, H, 512], lse fp32 [R, H])`` of the sparse attention over ``window ++ selected``.

    ``window`` / ``selected``: ``(keys, lists)`` or None; ``lists`` int32 ``[R, capacity]`` of key rows, -1 for an
    empty position, occupied positions first. ``window_length`` / ``selected_length``: optional int32 ``[R]``, the
    occupied prefix of each part (blocks past it are not run). ``sink``: fp32 ``[H]`` or None.
    ``leaf_options`` go to the leaf unchanged (``rope``, ``rope_q``, ``rope_o`` of the dual-rail leaf; ``block_table``,
    ``bt_row``, ``recipe``, ``groups`` of the segmented leaf)."""
    dom = domain_of(window, selected, sink)
    wkv, wl, _ = _part(window, "window")
    skv, sl, _ = _part(selected, "selected")
    if leaf == "dualrail":
        from .dualrail import dualrail_attn

        if dom.name == "window+selected":
            if wkv.dtype == torch.bfloat16 or skv.dtype == torch.bfloat16:
                if wkv is not skv:
                    raise ValueError("BF16 keys: both parts must index rows of ONE [S, 512] tensor (pass the same tensor)")
                lists = torch.cat((wl, sl), dim=1)
                if window_length is not None or selected_length is not None:
                    raise ValueError("per-part lengths need the two-list form (FP8 records); one BF16 list has one length")
                return dualrail_attn(q, wkv, lists, softmax_scale, sink=sink, out=out, lse=lse, **leaf_options)
            return dualrail_attn(q, wkv, wl, softmax_scale, sink=sink, topk_length=window_length, extra_kv=skv,
                                 extra_lists=sl, extra_topk_length=selected_length, out=out, lse=lse, **leaf_options)
        kv, lists, length = (wkv, wl, window_length) if dom.window else (skv, sl, selected_length)
        return dualrail_attn(q, kv, lists, softmax_scale, sink=sink, topk_length=length, out=out, lse=lse, **leaf_options)
    if leaf == "segmented":
        from .fwd import sparse_attn_prefill

        if dom.name != "selected" or dom.sink:
            raise ValueError(f"the segmented leaf serves the selected-only domain without a sink (got {dom.name}, "
                             f"sink {'on' if dom.sink else 'off'}); the sink form is sink.sparse_attention_sink")
        return sparse_attn_prefill(q, skv, sl, softmax_scale, topk_length=selected_length, out=out, lse=lse, **leaf_options)
    raise ValueError(f"leaf must be 'dualrail' or 'segmented' (got {leaf!r})")
