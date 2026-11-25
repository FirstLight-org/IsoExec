"""Pin GDN launch geometry that participates in the IsoExec arithmetic contract.

FLA chunk kernels otherwise autotune independently in the trainer and engine processes, which can select
different reduction orders. ``pin_fla_autotune_configs`` selects the same declared config in both runtimes;
``pin_gdn_rmsnorm_rows_per_block`` fixes the gated-norm tile height to one row so its reduction order cannot
depend on batch size. Both are on by default, and ``ISOEXEC_GDN_CONFIG_INDEX`` must match across runtimes.

TWO TREES, ON PURPOSE (kernel-ownership.md 1, 5.4). The kernels we run are ``ops/gdn/shared/vendored_fla``,
vendored so an upstream patch cannot move our rounding schedule under us. But the engine process still
launches vLLM's OWN copies -- ``QwenGatedDeltaNetAttention._warmup_prefill_kernels`` calls
``chunk_gated_delta_rule`` with dummy tensors during V1 profiling, and our ``_forward_core`` rebind does
not reach it. An unpinned autotuner there benchmarks under whatever memory is left and leaves a racy
config cached. So the pin walks the vendored tree, which is the contract, and then vLLM's tree when it is
importable, which is hygiene. The upstream half is the INVERTED direction of the runtime-import ledger --
we patch the runtime to install OUR determinism -- and vendoring can never remove it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

# Every autotuned Triton kernel reachable from `chunk_gated_delta_rule`. Missing entries are
# tolerated (vLLM moves kernels between versions) but reported: an unpinned kernel silently
# reintroduces per-process autotuning.
_FLA_KERNELS: tuple[tuple[str, str], ...] = (
    ("chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel"),
    ("solve_tril", "solve_tril_16x16_kernel"),
    ("solve_tril", "merge_16x16_to_32x32_inverse_kernel"),
    ("solve_tril", "merge_16x16_to_64x64_inverse_kernel"),
    ("wy_fast", "recompute_w_u_fwd_kernel"),
    ("chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ("chunk_o", "chunk_fwd_kernel_o"),
    ("cumsum", "chunk_local_cumsum_scalar_kernel"),
    ("cumsum", "chunk_local_cumsum_vector_kernel"),
    # The q/k l2 norm the chunk forward and the CPR prep launch; its tile height is autotuned too.
    ("l2norm", "l2norm_fwd_kernel"),
    ("l2norm", "l2norm_fwd_kernel1"),
)

_pinned = False


def gdn_pin_enabled() -> bool:
    return os.environ.get("ISOEXEC_GDN_PIN_CONFIGS", "1") == "1"


def _config_index() -> int:
    return int(os.environ.get("ISOEXEC_GDN_CONFIG_INDEX", "0"))


def _import_vendored(module_name: str):
    """One kernel module out of our own tree. These configs are the ones that decide our bits."""
    return importlib.import_module(f"{__package__}.vendored_fla.{module_name}")


def _import_upstream(module_name: str):
    """One kernel module out of vLLM's tree.

    The package name is spelled out as a LITERAL on purpose. ``checks/lint.py`` resolves a
    dynamic import down to the constant head of its f-string, so building this name from a variable
    would hide a live runtime binding from the ledger -- and hiding it is exactly the failure the
    ledger exists to prevent. This import is category (c): the inverted direction, where we reach
    into the runtime to install OUR determinism. Vendoring cannot remove it.
    """
    return importlib.import_module(f"vllm.third_party.flash_linear_attention.ops.{module_name}")


def _autotuner(importer, module_name: str, kernel_name: str):
    """Return the Autotuner behind a (possibly Heuristics-wrapped) Triton kernel, or None."""
    try:
        mod = importer(module_name)
    except Exception as e:  # pragma: no cover - upstream tree absent (trainer without vLLM)
        logger.info("[isoexec-gdn] no %s (%s)", module_name, e)
        return None
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        return None
    # triton.heuristics wraps triton.autotune wraps JITFunction; unwrap until we find `.configs`.
    obj = kernel
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "cache"):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def _config_identity(kernel_name: str, cfg) -> str:
    """The launch identity of one pinned config -- the thing that decides this kernel's bits."""
    return f"{kernel_name}:{cfg.kwargs}/w{cfg.num_warps}/s{cfg.num_stages}"


def _walk_tree(importer, idx: int, *, pin: bool) -> tuple[list[str], list[str]]:
    """Select config ``idx`` for every kernel in one tree; mutate it iff ``pin``.

    ORDER-INDEPENDENT BY CONSTRUCTION, which is what lets the attestation be read at any time.
    Pinning collapses ``at.configs`` to ``[chosen]``, so a later read-only walk re-selects that same
    single entry -- ``configs[idx] if idx < len(configs) else configs[0]`` lands on it whether idx is
    0 or out of range. The survey therefore returns the identity the process WILL pin before the pin
    runs, and the identity it DID pin afterwards, and they are the same string.
    """
    chosen_ids, missing = [], []
    for module_name, kernel_name in _FLA_KERNELS:
        at = _autotuner(importer, module_name, kernel_name)
        if at is None:
            missing.append(f"{module_name}.{kernel_name}")
            continue
        configs = list(at.configs)
        if not configs:
            missing.append(f"{module_name}.{kernel_name}(no configs)")
            continue
        chosen = configs[idx] if idx < len(configs) else configs[0]
        if pin:
            at.configs = [chosen]
            at.cache.clear()
        chosen_ids.append(_config_identity(kernel_name, chosen))
    return chosen_ids, missing


def _pin_tree(importer, idx: int) -> tuple[list[str], list[str]]:
    return _walk_tree(importer, idx, pin=True)


def fla_pin_fingerprint() -> str:
    """The cross-side attestation of the GDN autotune pin: what THIS process actually runs.

    Folded into ``enforce.install_attestation_digest`` and compared at the weight-sync handshake, so
    a trainer and an engine that pinned different FLA configs REFUSE instead of quietly computing
    different reduction orders. Before this existed the module docstring's "must match across
    runtimes" was enforced by nothing: the config index is not a contract pin, `gdn.core`'s
    ``{"kernel": "cpr"}`` is a literal on both sides rather than a readback, and the one mechanism
    that would have caught it -- ``install_attestation_digest`` -- had no caller (STATUS 25.1).

    VENDORED TREE ONLY. Upstream's copies are hygiene and legitimately differ: a trainer without
    vLLM pins none of them, so folding them in would refuse every correct colocated pair.

    Read-only and side-effect free, so it may be called before or after the pin -- see
    ``_walk_tree``. That matters: the trainer can defer its pin to the first GDN launch, which is
    after the first weight sync, and an order-dependent fact would refuse a correct pair.
    """
    import hashlib
    import json

    if not gdn_pin_enabled():
        # A distinct value, not an absence: autotune left ON is exactly the state that makes the
        # kernels nondeterministic, and a pair where only one side did that must refuse.
        return "gdn.fla_autotune=OFF(ISOEXEC_GDN_PIN_CONFIGS=0)"
    idx = _config_index()
    chosen, missing = _walk_tree(_import_vendored, idx, pin=False)
    payload = {"index": idx, "configs": sorted(chosen), "missing": sorted(missing)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"gdn.fla_autotune=idx{idx}/n{len(chosen)}/miss{len(missing)}/sha{digest}"


# THE AUTOTUNE PIN AS A CROSS-SIDE FACT. The pin is a property of the process, carried by the
# environment (`ISOEXEC_GDN_CONFIG_INDEX`, `ISOEXEC_GDN_PIN_CONFIGS`) that both sides inherit, not an
# obligation anyone discharges. `fla_pin_fingerprint` is what the GDN install logs.


def _upstream_importable() -> bool:
    """True iff vLLM is installed in this interpreter.

    ``find_spec`` on the TOP-LEVEL name only: it resolves a spec without executing anything, so a
    trainer process with no vLLM skips the upstream half in one cheap call instead of nine failed
    imports, and a process that does have vLLM still imports exactly the nine leaf kernel modules
    the pin needs -- not ``fla.ops.__init__``, which pulls in kernels we never launch.
    """
    try:
        return importlib.util.find_spec("vllm") is not None
    except Exception:  # pragma: no cover - a broken/partial vLLM install
        return False


def pin_fla_autotune_configs() -> int:
    """Pin every FLA chunk kernel to one statically-chosen config. Idempotent; returns the count.

    The count is the number of VENDORED autotuners pinned -- the ones whose configs decide our
    bits. Upstream's copies are pinned too when vLLM is importable (see the module docstring) and
    reported separately, because failing to pin them is a performance and hygiene problem, not a
    contract violation.
    """
    global _pinned

    if _pinned:
        return 0
    if not gdn_pin_enabled():
        print(
            "[ISOEXEC-GDN] ISOEXEC_GDN_PIN_CONFIGS=0 -> leaving Triton autotune ON. "
            "GDN kernels are then nondeterministic and NOT batch-invariant (baseline A/B only).",
            flush=True,
        )
        return 0

    idx = _config_index()
    pinned, missing = _pin_tree(_import_vendored, idx)
    if missing:
        # Loud, not fatal: an unpinned kernel means per-process autotuning is back for that op.
        # In the vendored tree it is also a bug report -- the module is ours and should be there.
        logger.warning("[isoexec-gdn] could not pin vendored: %s", ", ".join(missing))

    up_pinned, up_missing = ([], [])
    if _upstream_importable():
        up_pinned, up_missing = _pin_tree(_import_upstream, idx)
        if up_missing:
            logger.info("[isoexec-gdn] could not pin upstream vLLM: %s", ", ".join(up_missing))

    _pinned = True
    print(
        f"[ISOEXEC-GDN] pinned {len(pinned)} FLA autotune configs (index={idx}) in the VENDORED tree "
        f"-> deterministic, batch-invariant GDN. Unpinned: {len(missing)}. "
        f"vLLM's own tree: {len(up_pinned)} pinned (its profiling warmup launches it), "
        f"{len(up_missing)} unpinned" + ("" if up_pinned or up_missing else " -- not importable here"),
        flush=True,
    )
    logger.info("[isoexec-gdn] pinned configs: %s", "; ".join(pinned))
    return len(pinned)


# Note (Alexander): FLA backward autotune is off for these kernels. Their candidates are not
# bitwise-equal, so the bits depended on autotune timing (cold-cache runs picked different
# configs), and dqkwg's num_warps=2 candidates spill ~27 KB/thread, leaving ~6.5 GiB of CUDA
# stack reserved for the rest of the process. Each pin is the config autotune selects, verified
# bit-identical at the 35B trainer shapes. Architectures without measured pins keep autotune.
_FLA_BACKWARD_PINS: dict[str, dict[tuple[str, str], dict]] = {
    "sm90": {
        ("common.chunk_o", "chunk_bwd_kernel_dqkwg"): {"num_warps": 4, "num_stages": 3},
        ("common.chunk_delta_h", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"):
            {"num_warps": 4, "num_stages": 3, "BV": 32},
        ("gated_delta_rule.wy_fast", "prepare_wy_repr_bwd_kernel"): {"num_warps": 4, "num_stages": 3},
        ("utils.cumsum", "chunk_local_cumsum_scalar_kernel"): {"num_warps": 1, "num_stages": 3},
    },
    # Measured on B200 at Qwen3.6-27B's trainer shape (48 GQA-expanded heads, one 10240-token sequence): the
    # config FLA's autotuner picks there. Autotune keys include the head count, so with autotune left on a
    # context-parallel rank (24 heads) picked different configs (dhu BV 32 vs 64, kkt_solve BK 64 vs 32) and
    # different bits. Every other autotuner of the tree is pinned to its first candidate (_FLA_BACKWARD_PIN_ALL).
    "sm100": {
        ("common.chunk_delta_h", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"):
            {"num_warps": 4, "num_stages": 4, "BV": 64},
        ("common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"):
            {"num_warps": 2, "num_stages": 2, "BV": 32},
        ("common.chunk_o", "chunk_bwd_kernel_dqkwg"): {"num_warps": 4, "num_stages": 2},
        ("common.chunk_o", "chunk_bwd_kernel_dv_local"): {"num_warps": 2, "num_stages": 3},
        ("common.chunk_o", "chunk_fwd_kernel_o"): {"num_warps": 8, "num_stages": 3, "BK": 128, "BV": 128},
        ("gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel"):
            {"num_warps": 1, "num_stages": 3, "BK": 32},
        ("gated_delta_rule.wy_fast", "recompute_w_u_fwd_kernel"): {"num_warps": 2, "num_stages": 3},
        ("utils.cumsum", "chunk_local_cumsum_scalar_kernel"): {"num_warps": 1, "num_stages": 3},
    },
}

#: Architectures whose backward tree is pinned WHOLE: an autotuner without a measured pin takes its first
#: candidate, so no kernel of the trainer's GDN backward ever autotunes and the config is a function of
#: the kernel alone -- not of the shape key, the cache, the timing or the process.
_FLA_BACKWARD_PIN_ALL = frozenset({"sm90", "sm100"})


def tree_autotuners(package: str) -> dict:
    """``{"<module under ops/>.<kernel>": Autotuner}`` for every autotuned kernel of a vendored FLA tree."""
    import pkgutil

    from triton.runtime.autotuner import Autotuner

    root = importlib.import_module(package)
    found: dict = {}
    for info in pkgutil.walk_packages(root.__path__, f"{package}."):
        module = importlib.import_module(info.name)
        short = info.name[len(package) + 1:].removeprefix("ops.")
        for name, obj in vars(module).items():
            for _ in range(4):
                if isinstance(obj, Autotuner):
                    if not any(at is obj for at in found.values()):
                        found[f"{short}.{name}"] = obj
                    break
                obj = getattr(obj, "fn", None)
                if obj is None:
                    break
    return found


def pin_fla_backward_configs(import_module, arch: str | None = None, *, all_autotuners=None) -> list[str]:
    """Pin the measured FLA backward kernels of the vendored backward tree; return the pinned identities.

    ``import_module(name)`` must resolve ``<tree>.ops.<name>`` in the tree the backward will run, which is
    ``vendored_fla_bwd`` (``gdn_fla_backward._import_vendored``). Raises if a measured pin no longer
    matches a candidate, so a re-vendor at a new FLA revision cannot silently drop it. On an architecture
    in ``_FLA_BACKWARD_PIN_ALL``, ``all_autotuners()`` (``{name: Autotuner}``) lists the rest of the tree,
    and each of those is pinned to its first candidate.
    """
    if arch is None:
        from isoexec.core.arch import live_device_arch

        arch = live_device_arch()
    table = _FLA_BACKWARD_PINS.get(arch)
    if not table:
        print(f"[ISOEXEC-GDN] UNPINNED: no measured FLA backward configs for {arch}; FLA backward "
              "autotunes here (bits may depend on the autotune cache); the local-memory guard "
              "still restores the stack limit after each sweep.", flush=True)
        return []
    pinned = []
    for (module_name, kernel_name), want in table.items():
        at = _autotuner(import_module, module_name, kernel_name)
        if at is None:
            raise RuntimeError(f"[isoexec-gdn] measured backward pin {module_name}.{kernel_name} has no autotuner "
                               "in the vendored FLA tree; re-measure the pin for this FLA revision")
        match = [c for c in at.configs
                 if all(getattr(c, k, c.kwargs.get(k)) == v for k, v in want.items())]
        if not match:
            raise RuntimeError(f"[isoexec-gdn] no {kernel_name} candidate matches the measured pin {want}; "
                               f"candidates: {[_config_identity(kernel_name, c) for c in at.configs]}")
        at.configs = [match[0]]
        at.cache.clear()
        pinned.append(_config_identity(kernel_name, match[0]))
    if arch in _FLA_BACKWARD_PIN_ALL:
        if all_autotuners is None:
            raise RuntimeError(f"[isoexec-gdn] {arch} pins the whole FLA backward tree; pass all_autotuners")
        for name, at in sorted(all_autotuners().items()):
            if len(at.configs) > 1:
                at.configs = [at.configs[0]]
                at.cache.clear()
                pinned.append(_config_identity(name.rsplit(".", 1)[-1], at.configs[0]))
    print(f"[ISOEXEC-GDN] pinned {len(pinned)} FLA backward configs for {arch}: {pinned}", flush=True)
    return pinned


_norm_pinned = False


def pin_gdn_rmsnorm_rows_per_block() -> bool:
    """Make ``RMSNormGated`` (the GDN output norm) batch-invariant. Idempotent.

    ``layernorm_guard.layer_norm_fwd`` derives its tile height from the row count M, and the kernel
    reduces a ``[ROWS_PER_BLOCK, BLOCK_N]`` tile with ``tl.sum(x, axis=1)``. The tile shape therefore
    decides the order of the fp32 reduction and the last bit of ``rstd``, so decode (small M) and
    prefill (large M) can disagree on the same input row. Pinning the height to one row removes the
    M dependence; the cost is a taller grid at prefill on a memory-bound kernel.

    NOT vendored, deliberately. The module we patch is the one vLLM's own ``RMSNormGated`` calls
    (``layers.layernorm.RMSNormGated.forward_cuda`` -> ``fla.ops.layernorm_guard.rmsnorm_fn``), and
    the engine keeps that norm: only ``_forward_core`` is rebound. This is the inverted direction --
    we install OUR numerics into the runtime -- so a vendored copy would pin nothing and the import
    is permanent, category (c) of the runtime-import ledger.
    """
    global _norm_pinned

    if _norm_pinned:
        return True
    try:
        from vllm.third_party.flash_linear_attention.ops import layernorm_guard as lg
    except Exception as e:  # pragma: no cover
        logger.warning("[isoexec-gdn] cannot pin RMSNormGated tile height: %s", e)
        return False

    # A correctness pin, not a tuning knob: any other value reintroduces M-dependent rounding.
    rows = 1
    lg.calc_rows_per_block = lambda M, device: rows
    _norm_pinned = True
    print(
        f"[ISOEXEC-GDN] pinned RMSNormGated tile height to {rows} row(s) -> batch-invariant "
        "gated norm (was M-dependent: 1 row at decode, 4 at prefill)",
        flush=True,
    )
    return True


