"""Memoize the device-derived half of torch's ``_native`` RMSNorm admission predicate."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIG: dict = {}
_SUPPORTED: dict = {}
_SMEM: dict = {}
_COUNTS = {
    "hits": 0,  # predicate evaluations that reused a cached device fact
    "misses": 0,  # first evaluation per (device, dtype) -- bounded by the device count
    "bypass_no_index": 0,  # index-less device: fell through to the uncached original
}


def native_norm_memo_enabled() -> bool:
    return os.environ.get("ISOEXEC_NATIVE_NORM_MEMO", "0") == "1"


def native_rmsnorm_memo_counts() -> dict:
    out = dict(_COUNTS)
    out["installed"] = _INSTALLED
    out["enabled"] = native_norm_memo_enabled()
    out["device_keys"] = sorted({str(k) for k in _SUPPORTED} | {str(k) for k in _SMEM})
    return out


def _reset_for_tests() -> None:
    _SUPPORTED.clear()
    _SMEM.clear()
    for k in _COUNTS:
        _COUNTS[k] = 0


def _device_key(device):
    """``(type, index)``, or ``None`` when the index is implicit (never cache that)."""
    idx = getattr(device, "index", None)
    if idx is None:
        return None
    return (device.type, idx)


def install_native_rmsnorm_memo() -> bool:
    """Rebind ``_is_supported`` / ``_smem_budget_bytes`` to memoized wrappers. Idempotent."""
    global _INSTALLED
    if not native_norm_memo_enabled():
        return False
    if _INSTALLED:
        return True

    try:
        from torch._native.ops.norm import rmsnorm_impl as R  # type: ignore
    except Exception as exc:  # noqa: BLE001 -- no _native on this torch: nothing to memoize
        logger.info("[ISOEXEC-NORM] native RMSNorm memo skipped: %s", exc)
        return False

    orig_supported = getattr(R, "_is_supported", None)
    orig_smem = getattr(R, "_smem_budget_bytes", None)
    if orig_supported is None or orig_smem is None:
        logger.warning(
            "[ISOEXEC-NORM] native RMSNorm memo skipped: torch._native.ops.norm.rmsnorm_impl has no "
            "_is_supported/_smem_budget_bytes (torch layout changed). Predicate left untouched."
        )
        return False

    def _is_supported(input):  # noqa: A002 -- torch's parameter name, kept verbatim
        key = _device_key(input.device)
        if key is None:
            _COUNTS["bypass_no_index"] += 1
            return orig_supported(input)
        k = (key, input.dtype)
        hit = _SUPPORTED.get(k)
        if hit is None:
            _COUNTS["misses"] += 1
            hit = _SUPPORTED[k] = bool(orig_supported(input))
        else:
            _COUNTS["hits"] += 1
        return hit

    def _smem_budget_bytes(device):
        key = _device_key(device)
        if key is None:
            _COUNTS["bypass_no_index"] += 1
            return orig_smem(device)
        hit = _SMEM.get(key)
        if hit is None:
            _COUNTS["misses"] += 1
            hit = _SMEM[key] = int(orig_smem(device))
        else:
            _COUNTS["hits"] += 1
        return hit

    _ORIG["_is_supported"] = orig_supported
    _ORIG["_smem_budget_bytes"] = orig_smem
    R._is_supported = _is_supported
    R._smem_budget_bytes = _smem_budget_bytes
    _INSTALLED = True
    print(
        "[ISOEXEC-NORM] native RMSNorm admission memo INSTALLED: the two DEVICE-derived halves of "
        "torch's _fused_rms_norm_cond (get_device_capability, get_device_properties) are now "
        "evaluated once per (device, dtype) instead of once per call. Every live guard "
        "(alignment, contiguity, COW, numel, shape, weight dtype/device) still runs on every "
        "call, so the boolean -- and therefore the kernel and its arguments -- cannot move. "
        "Read native_rmsnorm_memo_counts()['hits'], not this banner.",
        flush=True,
    )
    return True


def revert_native_rmsnorm_memo() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    from torch._native.ops.norm import rmsnorm_impl as R  # type: ignore

    R._is_supported = _ORIG["_is_supported"]
    R._smem_budget_bytes = _ORIG["_smem_budget_bytes"]
    _INSTALLED = False
