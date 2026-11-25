"""Engine state for the SM100 native recurrent GDN core: vLLM's mamba cache pages hold the fp32 state.

The state layer (``runtimes/vllm/_gdn.GDNStateLayer``) builds a :class:`RecurrentGDN` on the
(conv, ssm) views vLLM binds into each layer, so vLLM sizes, allocates, evicts and sleeps the state
exactly as for its native GDN layer. Nothing lives outside vLLM's KV cache, which is what lets Model
Runner V2 admit, evict and sleep these requests like any other. Decode is one-token recurrent and
capturable; a chunk split at any token boundary is exact, so chunked prefill and align-mode prefix
caching resume exactly.
"""

from __future__ import annotations

import os

from isoexec.ops.gdn.shared.gdn_recurrent_state import RecurrentGDN, build_recurrent_gdn

_NOTED = False


def _say(msg: str) -> None:
    """fd 1, not ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-GDN-RECURRENT-SM100] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def refuse_incompatible_flags() -> None:
    """Recurrent mode, vLLM-native kernels on both sides and vLLM-owned state, nothing CPR-scoped."""
    global _NOTED
    from isoexec.ops.gdn.shared.gdn_recurrent_state import native_recurrent_flag_problems

    bad = native_recurrent_flag_problems()
    if bad:
        raise RuntimeError("[isoexec-gdn] the selected GDN core is ops/gdn/gdn_recurrent_sm100 "
                           "(ISOEXEC_GDN_KERNEL=recurrent), and:\n  - " + "\n  - ".join(bad))
    if not _NOTED:
        _NOTED = True
        _say("engine core = gdn.core:fla_native_recurrent_v1 (ops/gdn/gdn_recurrent_sm100), state in vLLM mamba pages")


def build_recurrent_sm100_gdn(*, max_num_seqs: int, chunk_size: int = 64, ssm_state=None, conv_state=None,
                              **kw) -> RecurrentGDN:
    """``ssm_state``/``conv_state`` are the layer's bound vLLM mamba views (both required)."""
    del chunk_size
    refuse_incompatible_flags()
    if ssm_state is None or conv_state is None:
        raise RuntimeError("[isoexec-gdn] gdn_recurrent_sm100 needs vLLM's bound mamba cache (ssm, conv) views")
    return build_recurrent_gdn(max_num_seqs=max_num_seqs, ssm_state=ssm_state, conv_state=conv_state, **kw)
