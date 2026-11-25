"""Trainer entry for the SM100 native recurrent GDN core.

Forward is ``gdn_ops.gdn_native_core``: the engine's own kernel over a zero scratch state, so trainer
and engine compute one function the same way. Backward is FLA's chunk VJP chained through an eager
rebuild of the in-kernel l2norm/gating (``ISOEXEC_GDN_FLA_BACKWARD=1``), else the fp32 reference VJP.
"""

from __future__ import annotations

import torch

#: The registered impl id on ``gdn.core`` (``spec.py``).
IMPL = "fla_native_recurrent_v1"
#: The ops the fused recurrence also computes.
_OWNED = ("gdn.l2norm", "gdn.gating")


def gdn_recurrent_sm100_core(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, cu_seqlens_cpu=None, **unused):
    """Packed varlen trainer forward, zero initial state. ``q, k`` raw ``[T, hk, 128]``, ``v`` ``[T, hv, 128]``,
    ``a, b`` ``[T, hv]``. Returns ``[T, hv, 128]``."""
    del cu_seqlens_cpu, unused
    from isoexec.ops.gdn.shared.gdn_ops import gdn_native_core

    o = gdn_native_core(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), a, b, A_log, dt_bias,
                        cu_seqlens=cu_seqlens.to(torch.int32))
    return o.squeeze(0)
