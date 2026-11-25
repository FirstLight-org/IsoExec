"""vLLM Triton causal_conv1d kernels vendored for conv.short.
The copy is header + rewrite(upstream) under PROVENANCE.json; the shared vendored_fla/_provenance.py
checker pins it to its recorded vendored_sha256."""

from isoexec.ops.gdn.conv_sm90.vendored.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
