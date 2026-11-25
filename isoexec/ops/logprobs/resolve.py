"""Resolve the head, row-selection and log-softmax stages, with architecture selection only for log-softmax."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import integration, kernel_package, provider_of, refuse, resolution

HEAD_IMPL = "eager"
SLICE_IMPL = "identity"


def resolve(op: str, *, site: str, **unused: Any):
    """The head GEMM, the row selection, or the log-softmax, on the live arch."""
    if unused:
        raise refuse(op, site=site, asked=unused, on_offer=(),
                     why="the scoring stages take no caller-supplied fact; the TP group is a seam, not an input.")

    if op == "logprobs.head":
        from isoexec.ops._dispatch import current_arch
        if current_arch() == "sm100":
            from isoexec.ops.mm.resolve import resolve as resolve_mm
            mm = resolve_mm("mm", site=site)
            def head(x, weight):
                return mm.fn(x, weight)
            return resolution(op, HEAD_IMPL, head, provider="ops.logprobs", site=site,
                              owns={"mm": mm.impl}, inputs=mm.inputs)
        from isoexec.ops.logprobs.eager import head_bf16

        return resolution(op, HEAD_IMPL, head_bf16, provider="ops.logprobs", site=site)

    if op == "logprobs.lm_head_slice":
        from isoexec.ops.logprobs.eager import lm_head_slice_identity

        return resolution(op, SLICE_IMPL, lm_head_slice_identity, provider="ops.logprobs", site=site)

    if op == "logprobs.log_softmax":
        package = kernel_package("logprobs", op=op, site=site)
        integ = integration(package)
        from isoexec.ops.logprobs.eager import log_softmax_rowinv

        impl = integ.IMPL
        def log_softmax(*args, **kwargs):
            if integ.IMPL != impl:
                raise RuntimeError("logprobs recipe changed after placement; rebuild the bound operation")
            return log_softmax_rowinv(*args, **kwargs, _sampled_logprobs=integ.sampled_logprobs,
                                      _sampled_entropy=getattr(integ, "sampled_logprobs_entropy", None))
        return resolution(op, impl, log_softmax, provider=provider_of(package), site=site)

    raise refuse(op, site=site, asked={},
                 on_offer=("logprobs.head", "logprobs.lm_head_slice", "logprobs.log_softmax"),
                 why="the logprobs family states these op ids and no others.")


__all__ = ["HEAD_IMPL", "SLICE_IMPL", "resolve"]
