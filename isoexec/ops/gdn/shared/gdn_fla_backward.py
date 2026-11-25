"""Load the vendored FLA 0.5.2 chunk gated-delta-rule (fused Triton backward) from ``vendored_fla_bwd``.

The source lives in IsoExec (``ops/gdn/shared/vendored_fla_bwd``, provenance in its PROVENANCE.json);
nothing here imports the ``fla`` package or touches ``sys.modules['fla']``.
ISOEXEC_GDN_FLA_BACKWARD controls gdn_ops autograd paths; the SM90 trainer core always uses this backward."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_fla_chunk = None  # memoized vendored chunk_gated_delta_rule (fwd+bwd), pinned and warmed


def fla_backward_enabled() -> bool:
    """Read at call time; default off, in which case the reference VJP is used."""
    return os.environ.get("ISOEXEC_GDN_FLA_BACKWARD") == "1"


def _import_vendored(name: str):
    """``fla.ops.<name>`` of the vendored tree, the resolver ``pin_fla_backward_configs`` expects."""
    import importlib

    return importlib.import_module(f"{__package__}.vendored_fla_bwd.ops.{name}")


def get_fla_chunk_backward():
    """Return the vendored FLA ``chunk_gated_delta_rule`` (with its fused backward), pinned once.

    ``inspect.unwrap(result).__globals__`` is the namespace of ``vendored_fla_bwd/ops/gated_delta_rule/chunk.py``;
    ``gdn_cpr_sm90/fla_backward.py`` reads the backward's stages from it.
    """
    global _fla_chunk
    if _fla_chunk is not None:
        return _fla_chunk

    from .gdn_batch_invariant import pin_fla_backward_configs, tree_autotuners
    from .vendored_fla_bwd import __version__
    from .vendored_fla_bwd.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

    # Pin before the warmup's first launch so no unpinned candidate is ever benchmarked.
    pin_fla_backward_configs(_import_vendored, all_autotuners=lambda: tree_autotuners(f"{__package__}.vendored_fla_bwd"))
    _warmup(chunk_gated_delta_rule)
    _fla_chunk = chunk_gated_delta_rule
    logger.info("[isoexec-gdn] loaded vendored FLA fused backward (v%s)", __version__)
    return _fla_chunk


def _warmup(chunk) -> None:
    """One tiny fwd+bwd at load time, as the previous loader did.

    It no longer has an import job to do. It is kept because the kernels left unpinned autotune once
    per key, and none of their keys include T or the varlen layout. For example,
    ``chunk_gated_delta_rule_fwd_kkt_solve_kernel`` is keyed on (H, HV, K, BC), and this launch has the
    same key as the GQA-expanded H=HV=8, K=128 calls of ``fla_chunk_vjp``, so the warmup decides that
    kernel's config for them. Dropping it would move the decision to the first real call and could
    change which candidate runs.

    Runs under ``enable_grad`` because this is called from inside a backward pass, where autograd is
    otherwise disabled and the warmup forward would record no graph.
    """
    import torch

    if not torch.cuda.is_available():
        return
    dev, dt = "cuda", torch.bfloat16
    with torch.enable_grad():
        q = torch.nn.functional.normalize(torch.randn(1, 8, 1, 128, device=dev), dim=-1).to(dt).requires_grad_(True)
        k = torch.nn.functional.normalize(torch.randn(1, 8, 1, 128, device=dev), dim=-1).to(dt).requires_grad_(True)
        v = torch.randn(1, 8, 1, 128, device=dev, dtype=dt, requires_grad=True)
        g = torch.nn.functional.logsigmoid(torch.randn(1, 8, 1, device=dev) + 4.0).to(dt).requires_grad_(True)
        beta = torch.rand(1, 8, 1, device=dev, dtype=dt, requires_grad=True)
        o, _ = chunk(q, k, v, g, beta, scale=None, use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        torch.autograd.grad(o, [q, k, v, g, beta], torch.ones_like(o))


def fla_chunk_vjp(q, k, v, g, beta, do, initial_state, cu_seqlens, chunk_size):
    """Gradient of the chunk gated-delta-rule w.r.t. (q, k, v, g, beta) via FLA's fused Triton backward.

    Follows the pinned forward's conventions: q/k are already L2-normed and GQA-expanded, beta is
    already sigmoid'd, and ``scale=None`` means ``K**-0.5``. ``initial_state`` is treated as a
    constant. Returns grads cast back to each input's dtype.
    """
    import torch

    chunk = get_fla_chunk_backward()
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
        o, _ = chunk(
            leaves[0],
            leaves[1],
            leaves[2],
            leaves[3],
            leaves[4],
            scale=None,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
        )
        grads = torch.autograd.grad(o, leaves, do.to(o.dtype))
    return [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
