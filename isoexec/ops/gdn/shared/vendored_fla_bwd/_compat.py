"""The upstream FLA names the vendored backward tree imports but does not copy.

Two kinds of names, both reached through one-line import rewrites recorded in PROVENANCE.json:

* ``dispatch``: FLA's backend-dispatch decorator (``fla.ops.backends``, re-exported by
  ``fla.ops.common.backends`` and ``fla.modules.backends``). Upstream picks a TileLang /
  Ascend / intra-card-CP implementation at call time unless ``FLA_DISABLE_BACKEND_DISPATCH=1``,
  in which case ``dispatch(op)(func)`` returns ``func`` unchanged. The loader this tree replaces
  always set that variable before importing FLA, so the Triton kernels here were always the ones
  that ran. The identity decorator below is that behaviour, with nothing left to configure.
* Context parallelism (``fla.ops.cp``): ``FLACPContext`` and the four CP pre/post-processing
  stages. ``chunk.py`` evaluates ``FLACPContext | None`` in its signatures and calls the stages
  only when ``cp_context is not None``. IsoExec never passes a CP context, so the type is a bare
  marker and the stages refuse loudly instead of computing something.
"""

from __future__ import annotations


def dispatch(operation: str):
    """``fla.ops.backends.dispatch`` with ``FLA_DISABLE_BACKEND_DISPATCH=1``: return the Triton function."""
    del operation

    def decorator(func):
        return func

    return decorator


class FLACPContext:
    """Type marker for ``fla.ops.cp.FLACPContext``. Context parallelism is not vendored."""


def _cp_unsupported(name: str):
    def stage(*args, **kwargs):
        raise NotImplementedError(
            f"vendored_fla_bwd: {name} belongs to FLA's context-parallel path (fla.ops.cp), which IsoExec "
            "does not vendor; call chunk_gated_delta_rule with cp_context=None"
        )

    stage.__name__ = name
    return stage


chunk_gated_delta_rule_bwd_dhu_pre_process = _cp_unsupported("chunk_gated_delta_rule_bwd_dhu_pre_process")
chunk_gated_delta_rule_fwd_h_pre_process = _cp_unsupported("chunk_gated_delta_rule_fwd_h_pre_process")
compress_h0 = _cp_unsupported("compress_h0")
expand_h0 = _cp_unsupported("expand_h0")

__all__ = [
    "FLACPContext",
    "chunk_gated_delta_rule_bwd_dhu_pre_process",
    "chunk_gated_delta_rule_fwd_h_pre_process",
    "compress_h0",
    "dispatch",
    "expand_h0",
]
