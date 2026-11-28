"""Resolve the head, row-selection and log-softmax stages, with architecture selection only for log-softmax."""

from __future__ import annotations

from typing import Any

from isoexec.ops._resolve import integration, kernel_package, provider_of, refuse, resolution

HEAD_IMPL = "eager"
SLICE_IMPL = "identity"


HEAD_FP32_IMPL = "fp32_root"


def _head_fp32_root(x, weight):
    """``x @ weight^T`` as the FP32 root of mm_sm100's BF16 GEMM tree (no BF16 rounding of the logits), with
    its backward; rows are independent, so a vocabulary shard's columns are the full head's."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100.ops import linear
    lead = x.shape[:-1]
    # the head keeps mm_sm100's plans at decode rows too (no DeepGEMM segment GEMMs for the vocabulary-wide GEMM)
    return linear(x.reshape(-1, x.shape[-1]), weight, root_fp32=True, decode_tree=False).view(*lead, weight.shape[0])


def resolve(op: str, *, site: str, logits: str | None = None, **unused: Any):
    """The head GEMM, the row selection, or the log-softmax, on the live arch. ``logits="fp32"`` selects
    the head GEMM that returns its FP32 root (an ``LMHead(logits_dtype="fp32")``), served on SM100."""
    if logits is not None and logits != "bf16":
        from isoexec.ops._dispatch import current_arch
        if op != "logprobs.head" or logits != "fp32" or current_arch() != "sm100" or unused:
            raise refuse(op, site=site, asked={"logits": logits, **unused}, on_offer=(),
                         why="the FP32-root head GEMM is the SM100 head's; no other stage takes a logits fact.")
        return resolution(op, HEAD_FP32_IMPL, _head_fp32_root, provider="ops.logprobs", site=site)
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


__all__ = ["HEAD_FP32_IMPL", "HEAD_IMPL", "SLICE_IMPL", "resolve"]
