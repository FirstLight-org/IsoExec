"""Adapters installing SM90 convolution at both trainer and engine call sites.
The recipe differs from the vendored Triton pair, so both hosts must switch together; install() is idempotent."""

from __future__ import annotations

import torch

from isoexec.ops.gdn.conv_sm90 import ops

_COUNTS = {"varlen": 0, "update": 0, "train": 0}
_INSTALLED = {}


def stats() -> dict:
    return dict(_COUNTS)


def causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    num_computed_tokens=None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """Drop-in for vLLM's ``causal_conv1d_fn``: ``x`` is the channel-last ``[D, T]`` view; returns ``[D, T]``."""
    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        raise NotImplementedError("conv_sm90: mamba prefix-cache block addressing is not supported")
    _COUNTS["varlen"] += 1
    xt = x.transpose(0, 1)
    n = query_start_loc.shape[0] - 1
    if cache_indices is None:
        cache_indices = torch.arange(n, dtype=torch.int32, device=x.device)
    if has_initial_state is None:
        has_initial_state = torch.zeros(n, dtype=torch.bool, device=x.device)
    y = ops.varlen(
        xt,
        weight,
        bias,
        cu_seqlens=query_start_loc,
        activation=activation,
        conv_state=conv_states,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        null_block_id=null_block_id if null_block_id is not None else -1,
    )
    return y.transpose(0, 1)


def _packed_rows(x) -> bool:
    """Rows of exactly ``D`` elements, unit column stride: a buffer of its own (the cat's output).

    A one-row column window of the in_proj output reads as ``is_contiguous()`` (size-1 dims are
    ignored) yet its row stride is the GEMM width; it must never be written in place.
    """
    return x.dim() == 2 and x.stride(-1) == 1 and x.stride(0) == x.shape[1]


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    """Drop-in for vLLM's single-token ``causal_conv1d_update`` (in place into ``x``)."""
    if num_accepted_tokens is not None or query_start_loc is not None or block_idx_last_scheduled_token is not None:
        raise NotImplementedError("conv_sm90: speculative / varlen / APC decode forms are not supported")
    if x.ndim != 2:
        raise NotImplementedError("conv_sm90: decode update takes [N, D]")
    _COUNTS["update"] += 1
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    idx = conv_state_indices
    if idx is None:
        idx = torch.arange(x.shape[0], dtype=torch.int32, device=x.device)
    # In place for a contiguous x (vLLM's contract). A row-strided x -- the q|k|v column window of the
    # in_proj GEMM output, handed over without a cat -- is read in place and written to a fresh
    # contiguous output, so the GEMM output's other columns and its storage are never written.
    return ops.update(
        x, conv_state, weight, bias, activation=activation, cache_indices=idx,
        null_block_id=null_block_id if null_block_id is not None else -1, out=x if _packed_rows(x) else None,
    )


def gdn_native_conv(x, weight, bias, *, cu_seqlens, activation="silu"):
    """Trainer: the ``gdn_ops.gdn_native_conv`` signature (x [T, D], weight [D, W])."""
    _COUNTS["train"] += 1
    return ops.conv(x, weight, bias, cu_seqlens=cu_seqlens, activation=activation)


#: ``x`` may be a row window of a wider tensor (unit column stride, 16-byte aligned rows): read in place.
gdn_native_conv.reads_row_windows = True
causal_conv1d_fn.reads_row_windows = True


#: The registered impl id on ``conv.short`` (``ops/gdn/conv_sm90/spec.py``).
IMPL = "sm90_cuda_v1"


def install() -> bool:
    """Rebind IsoExec's conv call sites to the SM90 kernels. Idempotent; returns True once active."""
    if _INSTALLED:
        return True
    from isoexec.ops.gdn.shared import gdn_ops
    from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as vendored

    _INSTALLED["fn"] = vendored.causal_conv1d_fn
    _INSTALLED["update"] = vendored.causal_conv1d_update
    _INSTALLED["native"] = gdn_ops.gdn_native_conv
    vendored.causal_conv1d_fn = causal_conv1d_fn
    vendored.causal_conv1d_update = causal_conv1d_update
    gdn_ops.gdn_native_conv = gdn_native_conv
    print("[ISOEXEC-CONV-SM90] conv.short -> conv_sm90 v1 (engine update/prefill + trainer fwd/bwd)", flush=True)
    return True


def uninstall() -> None:
    if not _INSTALLED:
        return
    from isoexec.ops.gdn.shared import gdn_ops
    from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as vendored

    vendored.causal_conv1d_fn = _INSTALLED.pop("fn")
    vendored.causal_conv1d_update = _INSTALLED.pop("update")
    gdn_ops.gdn_native_conv = _INSTALLED.pop("native")
