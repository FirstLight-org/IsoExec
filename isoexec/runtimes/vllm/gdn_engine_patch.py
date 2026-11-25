"""Integrate the canonical IsoExec GDN state machine with vLLM.

Routes GDN model calls through the selected IsoExec core and supplies scheduler-owned lifecycle
metadata. vLLM-visible Mamba pages are slot identities for private state, not state storage, and
engine args that would advance that state on their own terms are refused at initialization.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


# Engagement count for native vLLM GDN calls.
CALL_COUNT = 0

# vLLM's reserved null KV/state block id (never allocated to a request).
NULL_BLOCK_ID = 0


# vLLM features that would read/advance ssm_state outside chunk-consistent decode's control.
INCOMPATIBLE_ENGINE_ARGS = (
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "speculative_config",
)


def native_gdn_composition() -> bool:
    """True iff GDN runs the native recurrent composition on vLLM's own state blocks.

    Recurrent kernel, vLLM's native fused kernels and the state in vLLM's mamba pages: prefill, a
    chunked-prefill continuation and a prefix-cache hit all start from the state vLLM itself placed in
    the request's block (``prefill_has_initial_state`` + the align-mode pre-copy), so a split at any
    token boundary is exact. Reads only the environment; importable without CUDA or vLLM.
    """
    from isoexec.ops.gdn.shared.gdn_ops import gdn_native_kernels_enabled, recurrent_mode
    from isoexec.ops.gdn.shared.gdn_recurrent_state import native_state_enabled

    return recurrent_mode() and gdn_native_kernels_enabled() and native_state_enabled()


def cpr_pool_composition() -> bool:
    """True iff GDN runs CPR on its private record pool (cpr kernel, native kernels, no native state).

    Its state is not in vLLM's mamba blocks, so it resumes a chunked-prefill continuation from the row it
    carries (``ISOEXEC_GDN_CHUNKED_PREFILL=1``) and a prefix-cache hit from a boundary checkpoint it scanned
    itself (``ISOEXEC_GDN_CPR_APC``, ``runtimes/vllm/cpr_apc.py``). Environment only.
    """
    from isoexec.ops.gdn.shared.gdn_ops import cpr_mode, gdn_native_kernels_enabled
    from isoexec.ops.gdn.shared.gdn_recurrent_state import native_state_enabled

    return cpr_mode() and gdn_native_kernels_enabled() and not native_state_enabled()


#: Engine features whose GDN state must resume exactly: from vLLM's own blocks (native composition) or from
#: the private pool's carried rows and boundary checkpoints (CPR).
STATE_RESUMING_ENGINE_ARGS = ("enable_chunked_prefill", "enable_prefix_caching")

#: Engine flags the CPR composition needs per feature (``state_feature_env``).
_CPR_FEATURE_FLAGS = {
    "enable_chunked_prefill": {"ISOEXEC_GDN_CHUNKED_PREFILL": "1"},
    # A hit is resumed as a chunked-prefill continuation from the adopted checkpoint.
    "enable_prefix_caching": {"ISOEXEC_GDN_CHUNKED_PREFILL": "1", "ISOEXEC_GDN_CPR_APC": "shm"},
}


def resumable_engine_args() -> frozenset[str]:
    """The :data:`STATE_RESUMING_ENGINE_ARGS` the selected GDN composition runs bitwise.

    The integrations (verl, ``register_native``) keep the caller's values for exactly these, and turn on
    the engine flags each needs (:func:`state_feature_env`); every other composition gets both off. The
    native composition resumes from vLLM's blocks; CPR from its pool (rows and boundary checkpoints).
    """
    return frozenset(STATE_RESUMING_ENGINE_ARGS) if native_gdn_composition() or cpr_pool_composition() \
        else frozenset()


def state_feature_env(features) -> dict[str, str]:
    """The engine flags the selected composition needs to run ``features`` (empty when it needs none)."""
    if not cpr_pool_composition():
        return {}
    env: dict[str, str] = {}
    for feature in features:
        env.update(_CPR_FEATURE_FLAGS.get(feature, {}))
    return env


def enable_state_features(features, environ=None) -> dict[str, str]:
    """Export :func:`state_feature_env` into ``environ`` (default ``os.environ``); return what it set.

    A flag the caller set to an enabling value stands; one the caller set to an off value (``0``) contradicts
    the feature and is refused, naming both.
    """
    environ = os.environ if environ is None else environ
    set_: dict[str, str] = {}
    for name, value in state_feature_env(features).items():
        current = (environ.get(name) or "").strip()
        if current.lower() in ("0", "false", "no"):
            wanted = sorted(f for f in features if name in _CPR_FEATURE_FLAGS.get(f, {}))
            raise ValueError(f"[isoexec-gdn] {', '.join(wanted)} under the CPR GDN composition needs {name}={value}, "
                             f"but {name}={current} is set; unset it or turn the feature off")
        if not current:
            environ[name] = value
            set_[name] = value
    return set_


def resumable_engine_args_for(*, speculative: bool) -> tuple[frozenset[str], str]:
    """``(features to allow, why the rest are off)`` for an integration choosing engine defaults.

    ``speculative`` says the deployment runs speculative decoding, which no GDN composition admits;
    with it on, chunked prefill and prefix caching are turned off too rather than left to fail later.
    """
    if speculative:
        return frozenset(), "speculative decoding is refused for GDN models"
    allowed = resumable_engine_args()
    if allowed:
        return allowed, ""
    from isoexec.ops.gdn.shared.gdn_ops import gdn_kernel_mode, gdn_native_kernels_enabled
    from isoexec.ops.gdn.shared.gdn_recurrent_state import native_state_enabled

    return allowed, (
        f"GDN composition ISOEXEC_GDN_KERNEL={gdn_kernel_mode()!r} NATIVE_KERNELS="
        f"{int(gdn_native_kernels_enabled())} NATIVE_STATE={int(native_state_enabled())} keeps state no resume "
        "path covers; recurrent + NATIVE_KERNELS=1 + NATIVE_STATE=1 resumes from vLLM's blocks, cpr + "
        "NATIVE_KERNELS=1 + NATIVE_STATE=0 from its own pool"
    )


def admitted_engine_args() -> frozenset[str]:
    """Which :data:`INCOMPATIBLE_ENGINE_ARGS` the selected GDN composition admits. Environment only.

    Speculative decoding is never admitted. The native composition admits chunked prefill and prefix
    caching (:func:`native_gdn_composition`). A private pool admits chunked prefill with its resume flag
    (a continuation chunk resumes from the carried row), and CPR also prefix caching with the boundary store
    on (``ISOEXEC_GDN_CPR_APC``, which needs the resume flag too).
    """
    from isoexec.ops.gdn.shared.gdn_cpr_state import cpr_apc_enabled
    from isoexec.ops.gdn.shared.gdn_ops import cpr_mode, recurrent_mode
    from isoexec.ops.gdn.shared.gdn_recurrent_state import chunked_prefill_enabled

    admitted = set(resumable_engine_args()) if native_gdn_composition() else set()
    if (recurrent_mode() or cpr_mode()) and chunked_prefill_enabled():
        admitted.add("enable_chunked_prefill")
        if cpr_pool_composition() and cpr_apc_enabled():
            admitted.add("enable_prefix_caching")
    return frozenset(admitted)


def assert_engine_args_compatible(kwargs: dict, *, capturing: bool) -> None:
    """Raise if any engine arg is incompatible with chunk-consistent GDN decode.

    ``kwargs`` carries the three state-reinterpreting engine args (:data:`INCOMPATIBLE_ENGINE_ARGS`),
    plus ``mamba_cache_mode`` when known. ``capturing`` says whether this deployment replays CUDA
    graphs; it is passed in rather than sniffed out of ``kwargs["enforce_eager"]`` because
    ``enforce_eager`` is only vLLM's *request*, which ``VllmConfig.__post_init__`` then folds into
    ``compilation_config.cudagraph_mode`` -- and the mode is what the runner dispatches on. Callers
    derive it from the resolved mode.

    Loud on purpose: each of these silently reinterprets ``ssm_state`` and would produce a
    plausible-looking, non-bitwise rollout.
    """
    from ...ops.gdn import cpr_mode, recurrent_mode
    from ...ops.gdn import refuse_incompatible_flags

    # Validate provider-specific switches before constructing the engine or pool.
    refuse_incompatible_flags()

    admitted = admitted_engine_args()
    incompatible = tuple(k for k in INCOMPATIBLE_ENGINE_ARGS if k not in admitted)
    bad = [k for k in incompatible if kwargs.get(k)]
    # A prefix-cache hit resumes the state vLLM saved at a block boundary. Only "align" saves state
    # there (and only at chunk ends, which is all this composition writes); vLLM selects it whenever
    # prefix caching is on for a hybrid model, so anything else is a configuration drift.
    mode = kwargs.get("mamba_cache_mode")
    if kwargs.get("enable_prefix_caching") and mode is not None and mode != "align":
        bad.append(f"enable_prefix_caching with mamba_cache_mode={mode!r} (needs 'align')")
    # CUDA graphs are compatible: chunk decode is shape-static and sync-free (fixed cu grid,
    # persistent index buffers, masked chunk roll), and recurrent decode is static by construction.
    if cpr_mode() and capturing:
        # Graphs under cpr ride on the lazy resync driver, so decode() stays pure device work and
        # the boundary resync runs host-driven between replays. Installed here (idempotent) because
        # this assert runs before the engine is built.
        try:
            install_cpr_lazy_driver()
        except Exception as e:  # noqa: BLE001 - converted to the loud config error below
            bad.append(f"CUDA graph capture under cpr: lazy resync driver failed ({e})")
    elif capturing and not recurrent_mode() and os.environ.get("ISOEXEC_GDN_PADDED_DECODE") != "1":
        bad.append("CUDA graph capture without ISOEXEC_GDN_PADDED_DECODE=1")
    if bad:
        why = (
            "These features require state reuse or rollback that the selected GDN pool does not support"
        )
        raise ValueError(
            f"[isoexec-gdn] ISOEXEC_GDN=1 but the engine was configured with {bad}. {why}. "
            "Disable the listed unsupported engine features for this GDN model."
        )


def lift_gdn_batch_invariance_veto() -> None:
    """Let ``VLLM_BATCH_INVARIANT=1`` coexist with GDN. Idempotent.

    vLLM's veto on GDN_ATTN would also block the softmax layers. The backend getter is cached, so
    this must run before the backend is first resolved (KV-cache spec collection).
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)


# Per-forward host-side step data (decode/prefill slot ids and offsets), computed once and shared by
# every GDN layer: each `.tolist()` on engine metadata is a GPU sync, and paying it per layer
# serializes decode. Fed by the lazy driver from scheduler-owned host arrays; falls back to a device
# read on direct recurrent calls and the first forward.
_DRIVER_HOST_STEP = None
_DRIVER_HOST_STEP_STATS = {"served": 0, "fallback": 0}


def _build_driver_host_step(req_ids, scheduled, nct, slots) -> dict | None:
    """Reproduce GDN metadata's decode/prefill split from scheduler-owned host arrays.

    Admission is stricter than vLLM's builder: every row must schedule >= 1 token and the batch must
    be decode(1-token) then prefill(>1-token). Anything else returns ``None``.
    """

    if scheduled is None:
        return None
    # Recognise the common all-decode step early. The ``all`` keeps the strict contract: a synthetic
    # {0, 2} dictionary must not be mistaken for decode merely because its total is two.
    n = len(req_ids)
    if len(scheduled) == n and all(int(q) == 1 for q in scheduled.values()):
        dec = slots[:n]
        dec_slots = dec.tolist() if hasattr(dec, "tolist") else [int(v) for v in dec]
        return {
            "key": (n, 0, n),
            "dec_slots": dec_slots,
            "pre_slots": [],
            "pre_has_init": [],
            "qsl": [0],
        }
    qlens = [int(scheduled.get(rid, 0)) for rid in req_ids]
    if any(q <= 0 for q in qlens):
        return None
    n_dec = next((i for i, q in enumerate(qlens) if q != 1), len(qlens))
    if any(q == 1 for q in qlens[n_dec:]):
        return None
    pre_lens = qlens[n_dec:]
    qsl = [0]
    for q in pre_lens:
        qsl.append(qsl[-1] + q)
    return {
        "key": (n_dec, len(qlens) - n_dec, sum(qlens)),
        "dec_slots": [int(v) for v in slots[:n_dec]],
        "pre_slots": [int(v) for v in slots[n_dec:]],
        "pre_has_init": [bool(v > 0) for v in nct[n_dec:]],
        "qsl": qsl,
    }


def _driver_prefill_slot_positions(step: dict | None, nct) -> list[tuple[int, int]] | None:
    """Scheduler positions for only the positively-identified prefill suffix.

    ``None`` means the split was not proven and callers must invalidate their position mirror; an
    empty list is a proven pure-decode step.
    """

    if step is None:
        return None
    n_dec, n_pre, _ = step["key"]
    pre_slots = step["pre_slots"]
    if len(pre_slots) != n_pre or len(nct) < n_dec + n_pre:
        return None
    return [(int(pre_slots[j]), int(nct[n_dec + j])) for j in range(n_pre)]


def _step_cpu(md, n_dec: int, n_pre: int) -> dict:
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    # Under NATIVE_STATE the metadata alias is OFF by design: each mamba KV-cache group's layers keep
    # their state in that group's own vLLM pages, so each group's metadata (own slot ids) is the
    # right one. Cache the host step per metadata object then, and never take slot ids from the
    # driver feed (that one is built from a single group's block table).
    per_group = not gdn_alias_active() and _group_alias_mode() == "1"
    if per_group:
        cache = getattr(ctx, "_isoexec_gdn_step_cpu_by_md", None)
        if cache is None:
            cache = {}
            try:
                ctx._isoexec_gdn_step_cpu_by_md = cache
            except Exception:
                pass
        step = cache.get(id(md))
        if step is not None and step.get("md") is md and step.get("key") == (n_dec, n_pre, md.num_actual_tokens):
            return step
        step = None
    else:
        step = getattr(ctx, "_isoexec_gdn_step_cpu", None)
    if step is not None and step.get("key") == (n_dec, n_pre, md.num_actual_tokens):
        # Identity guard: this cache is shared across layers, which is sound only if every layer got
        # the SAME metadata object. A mismatch means multiple KV-cache groups with no alias active.
        if step.get("md") is not md:
            raise RuntimeError(
                "[isoexec-gdn] two GDN layers received DIFFERENT metadata objects in one forward: "
                "vLLM has split the mamba layers across multiple KV-cache groups (each with its "
                "own block table / state-slot ids) and no canonical-group metadata alias is "
                "active. Host bookkeeping would track one group's ids while each layer's decode "
                "gathers its own group's -- the unmapped ids fold onto null row 0 and every "
                "request shares one state row. cpr runs "
                "install the alias via resolve_gdn_groups (lazy driver); other modes must not run "
                "this geometry."
            )
        return step

    key = (n_dec, n_pre, md.num_actual_tokens)
    step = {"key": key, "md": md}
    fed = _DRIVER_HOST_STEP
    use_fed = fed is not None and fed.get("key") == key
    if use_fed:
        _DRIVER_HOST_STEP_STATS["served"] += 1
    else:
        _DRIVER_HOST_STEP_STATS["fallback"] += 1
    fed_slots = use_fed and not per_group
    if n_dec:
        step["dec_slots"] = fed["dec_slots"] if fed_slots else md.non_spec_state_indices_tensor[:n_dec].tolist()
    if n_pre:
        # One D2H for the whole step: every layer's `_continuation_mask` needs this same mask, and
        # the check below reads this host copy rather than paying its own sync.
        hi = md.prefill_has_initial_state
        step["pre_has_init"] = (
            fed["pre_has_init"] if use_fed else (None if hi is None else [bool(v) for v in hi.tolist()])
        )
        # A prefill normally starts at position 0, so an initial state means prefix caching or
        # chunked prefill snuck on -- except for a resume-flagged chunked-prefill continuation.
        if step["pre_has_init"] is not None and any(step["pre_has_init"]):
            # Under the native composition an initial state is routine (both continuations and
            # prefix-cache hits resume from vLLM's state blocks); with the private-pool resume flag a
            # continuation resumes the carried row. The raise guards every other composition.
            if "enable_chunked_prefill" not in admitted_engine_args():
                raise RuntimeError(
                    "[isoexec-gdn] a prefill carries an initial state (prefix caching or chunked "
                    "prefill is on). ssm_state is a chunk-boundary state under this patch and cannot "
                    "be resumed."
                )
        pre_slots = fed["pre_slots"] if fed_slots else md.prefill_state_indices.tolist()
        # Block 0 is vLLM's reserved null block: dummy capture/warmup batches route every row there
        # (BlockTables.get_dummy_block_tables) and no live request ever owns it.
        if n_dec and (set(pre_slots) & set(step["dec_slots"])) - {NULL_BLOCK_ID}:
            raise RuntimeError("[isoexec-gdn] a slot is being prefilled and decoded in one batch")
        step["pre_slots"] = pre_slots
        # offsets INTO the packed prefill region. The fed values are the cumulative scheduler query
        # lengths; fallback reads the tensor the metadata builder uploaded from those same values.
        step["qsl"] = fed["qsl"] if use_fed else md.prefill_query_start_loc.tolist()
        # Build causal-conv's sequence metadata once for the whole forward; every GDN layer sees the
        # same metadata object. None here leaves causal_conv1d_fn's own internal builder in charge.
        from ...ops.gdn import causal_conv1d_metadata

        step["conv_metadata"] = causal_conv1d_metadata(
            md.prefill_query_start_loc,
            cu_host=step["qsl"] if use_fed else None,
            include_launch_args=False,
        )
    if per_group:
        cache[id(md)] = step
        return step
    try:
        ctx._isoexec_gdn_step_cpu = step
    except Exception:
        pass  # uncacheable context: every layer recomputes (correct, just not fast)
    return step


def _decode_slots(md, n_dec: int):
    """One decode-slots view per metadata object per forward, so every layer sharing ``md`` passes the SAME
    tensor and the per-slots rows are derived once (gdn_recurrent_state.RecurrentGDN._rows_pair).

    Scoped to the forward context: a cached view (and the rows derived from it) never outlives the forward
    that built it, even if vLLM reuses a metadata object across steps.
    """
    from vllm.forward_context import get_forward_context

    try:
        ctx = get_forward_context()
    except Exception:  # pragma: no cover - outside a forward
        return md.non_spec_state_indices_tensor[:n_dec]
    cache = getattr(ctx, "_isoexec_decode_slots", None)
    if cache is None:
        cache = {}
        try:
            ctx._isoexec_decode_slots = cache
        except Exception:
            return md.non_spec_state_indices_tensor[:n_dec]
    hit = cache.get(id(md))
    if hit is not None and hit[0] is md and hit[1] == n_dec:
        return hit[2]
    view = md.non_spec_state_indices_tensor[:n_dec]
    if gdn_alias_active():
        # Every GDN layer of this forward resolves to this one metadata object and so decodes on this
        # very view: a state may resolve all layers' rows from it in one launch (gdn_cpr_sm100).
        try:
            view._isoexec_step_shared = True
        except (AttributeError, TypeError):
            pass
    cache[id(md)] = (md, n_dec, view)
    return view


@torch.no_grad()
def recurrent_core(rg, md, mixed_qkv, a, b):
    """Run one GDN layer's core for a vLLM batch: ``GDNAttentionMetadata`` -> RecurrentGDN.

    ``mixed_qkv`` is PRE-conv, ``a`` the gate input and ``b`` the beta input; returns ``[T, Hv, Dv]``.
    A pure-decode step touches no host data, so it captures into a FULL CUDA graph directly:
    ``_step_cpu`` syncs and is consulted only for batches with a prefill, which are never captured --
    under breakable PIECEWISE graphs this whole call is an eager break (``graph_breaks``) that reruns
    on every replay with that step's metadata.
    """
    global CALL_COUNT
    CALL_COUNT += 1
    if md.spec_sequence_masks is not None:
        raise RuntimeError(
            "[isoexec-gdn] speculative decoding advances ssm_state several tokens at a time with the "
            "spec kernels. Disable spec decode for bitwise IsoExec."
        )

    n_dec, n_pre = md.num_decodes, md.num_prefills
    if n_dec == 0 and n_pre == 0:
        return None
    if md.num_decode_tokens != n_dec:
        raise RuntimeError(f"[isoexec-gdn] expected 1 token per decode, got {md.num_decode_tokens} for {n_dec}")

    T = md.num_actual_tokens
    x, a, b = mixed_qkv[:T], a[:T], b[:T]

    # Pure decode returns the core's own output tensor: ``decode`` already allocates it fresh,
    # contiguous and at the right shape, so staging it through ``out`` would be a pure D2D copy.
    dec_slots = _decode_slots(md, n_dec) if n_dec else None
    if n_pre == 0:
        return rg.decode(dec_slots, x, a, b)

    out = torch.empty(T, rg.num_v_heads, rg.head_v_dim, dtype=x.dtype, device=x.device)

    # Non-spec token order is decode-first, then prefill (see GDNAttentionMetadata builder).
    if n_dec:
        out[:n_dec] = rg.decode(dec_slots, x[:n_dec], a[:n_dec], b[:n_dec])
    if n_pre:
        step = _step_cpu(md, n_dec, n_pre)  # also runs the initial-state / slot-overlap checks
        # The recurrent state's native prefill caches its per-step host constants in ``step`` (one pinned
        # upload for all layers); a subclass that overrides prefill does not take the argument.
        extra = {"step_cache": step} if getattr(type(rg).prefill, "_takes_step_cache", False) else {}
        out[n_dec:] = rg.prefill(
            md.prefill_state_indices,
            step["pre_slots"],
            x[n_dec:],
            a[n_dec:],
            b[n_dec:],
            step["qsl"],
            has_initial_state=step["pre_has_init"],
            conv_metadata=step.get("conv_metadata"),
            prefill_query_start_loc=md.prefill_query_start_loc,
            **extra,
        )
    return out


def gdn_layer_core(state, md, mixed_qkv, a, b):
    """Run one GDN layer's core, whichever mode built ``state``.

    Single owner of the alpha/beta compaction decision: the native core materialises them inside its
    fused q/k/v split, every other mode gets them compacted here.
    """
    from ...ops.gdn import defer_ab

    if not defer_ab(state):
        a, b = a.contiguous(), b.contiguous()
    return recurrent_core(state, md, mixed_qkv, a, b)


def gdn_metadata(prefix: str):
    """This layer's ``GDNAttentionMetadata``, or None during a vLLM profiling run.

    Every layer resolves to the FIRST GDN layer's metadata object so the stack shares one slot-id
    space. The alias must live here, not in a driver hook: CUDA-graph capture bakes in the address
    of the buffer read here. With a single mamba group it is identical to a per-prefix fetch.
    """
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if not _GDN_CANON_PREFIX:
        _GDN_CANON_PREFIX.append(prefix)  # first GDN fetch of the process == first GDN layer
    md = get_forward_context().attn_metadata
    if md is None:
        return None
    assert isinstance(md, dict)
    key = _GDN_CANON_PREFIX[0] if gdn_alias_active() else prefix
    if key not in md:
        # Model Runner V2 profile/autotune dummy runs build metadata for softmax attention only and
        # leave every Mamba-family layer out (model_runner.execute_model, "cheap warmup path"). That
        # is the metadata-free case. A step that has GDN metadata but not this layer's is an error.
        if not any(isinstance(value, GDNAttentionMetadata) for value in md.values()):
            return None
        raise KeyError(f"{key}: GDN metadata is missing while other GDN layers have it")
    md = md[key]
    assert isinstance(md, GDNAttentionMetadata)
    return md


# Mamba-group geometry: where the GDN state slot id lives in the block table.
_MAMBA_GROUP: list[int] = []  # resolved lazily from kv_cache_config; [-1] = no mamba group
_ALIGN_BS: list[int] = []  # mamba block size under align mode; [0] = fixed column 0


def _mamba_group_index(runner) -> int:
    if not _MAMBA_GROUP:
        gi = -1
        try:
            from vllm.v1.kv_cache_interface import MambaSpec

            for j, grp in enumerate(runner.kv_cache_config.kv_cache_groups):
                if isinstance(grp.kv_cache_spec, MambaSpec):
                    gi = j
                    break
        except Exception:  # pragma: no cover - config shape drift fails loud below
            pass
        _MAMBA_GROUP.append(gi)
    gi = _MAMBA_GROUP[0]
    if gi < 0:
        raise RuntimeError("[isoexec-gdn] lazy driver: no MambaSpec kv-cache group found")
    return gi


def _align_block_size(runner) -> int:
    """Mamba block size when vLLM is in ALIGN cache mode, else 0 (fixed column 0).

    Align mode gathers the block covering the LAST token of the step, so a request's slot id rotates
    every ``block_size`` tokens; reading column 0 there would hand the pool a stale slot.
    """
    if not _ALIGN_BS:
        bs = 0
        try:
            if runner.vllm_config.cache_config.mamba_cache_mode == "align":
                gi = _mamba_group_index(runner)
                bs = int(runner.kv_cache_config.kv_cache_groups[gi].kv_cache_spec.block_size)
        except Exception:  # pragma: no cover - older vLLM without mamba_cache_mode
            bs = 0
        _ALIGN_BS.append(bs)
    return _ALIGN_BS[0]


# Multi-group guard: one slot-id space for all GDN layers. vLLM can split the mamba layers across
# KV-cache groups, each with its own block table, at which point unmapped state ids fold onto null
# row 0 and every request silently shares one state row. Under cpr the pages carry no state, so all
# layers can safely read the canonical (first) GDN layer's ids. Not applied under NATIVE_STATE,
# where each layer's state really does live in its own group's blocks.
_GDN_CANON_PREFIX: list = []  # [first GDN layer's prefix], self-registered at the first fetch
_gdn_groups_resolved = False
_slot_space_checked = False  # one-shot: vLLM's num_blocks fits the slot map (checked in _drive)


def _group_alias_mode() -> str:
    """ISOEXEC_GDN_GROUP_ALIAS: '1' (default, alias to the canonical mamba group's metadata) or
    '0' (refuse a multi-group geometry)."""
    mode = os.environ.get("ISOEXEC_GDN_GROUP_ALIAS", "1").strip().lower() or "1"
    if mode not in ("0", "1"):
        raise ValueError(f"ISOEXEC_GDN_GROUP_ALIAS={mode!r}: expected '1' (alias) or '0' (refuse multi-group)")
    return mode


def gdn_alias_active() -> bool:
    """True iff GDN metadata fetches canonicalize to the first GDN layer's metadata object.

    Default ON; OFF under NATIVE_STATE and in the '0' mode.
    """
    if _group_alias_mode() != "1":
        return False
    from ...ops.gdn import native_state_enabled

    return not native_state_enabled()


def resolve_gdn_groups(runner) -> None:
    """Validate vLLM's mamba KV-cache grouping ONCE, at the driver's first step, and banner it.

    The fail-closed half of the alias (which itself lives in ``gdn_metadata``): it proves the
    canonical GDN layer belongs to the same MambaSpec group ``_slot_ids`` reads block ids from, and
    raises rather than logs if not. The banner goes to fd 1; vLLM filters INFO out of subprocesses.
    """
    global _gdn_groups_resolved
    if _gdn_groups_resolved:
        return
    _gdn_groups_resolved = True
    from vllm.v1.kv_cache_interface import MambaSpec

    groups = [
        (j, g) for j, g in enumerate(runner.kv_cache_config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    if len(groups) <= 1:
        return  # one slot-id space by construction; the alias is the identity
    sizes = [len(g.layer_names) for _, g in groups]
    mode = _group_alias_mode()
    if mode == "0":
        raise RuntimeError(
            f"[isoexec-gdn] vLLM split the mamba layers across {len(groups)} KV-cache groups "
            f"(sizes {sizes}) and ISOEXEC_GDN_GROUP_ALIAS=0 refuses to run that geometry. "
            "Each group carries its own block table, so without the canonical-group metadata "
            "alias the GDN layers disagree on state slot ids and decode folds the unmapped ones "
            "onto null row 0 -- silent corruption. Unset the flag (or set "
            "1) to run with the alias."
        )
    gi = _mamba_group_index(runner)
    gi_names = list(runner.kv_cache_config.kv_cache_groups[gi].layer_names)
    if not gdn_alias_active():
        try:
            os.write(
                1,
                f"[ISOEXEC-GDN-GROUPS] pid={os.getpid()} {len(groups)} mamba KV-cache groups "
                f"(sizes {sizes}); NATIVE_STATE: per-group metadata, each group's state lives in "
                "its own vLLM mamba pages (no alias).\n".encode(),
            )
        except Exception:  # pragma: no cover - fd 1 closed
            pass
        return
    canon = _GDN_CANON_PREFIX[0] if _GDN_CANON_PREFIX else gi_names[0]
    if _GDN_CANON_PREFIX and canon not in gi_names:
        raise RuntimeError(
            f"[isoexec-gdn] the canonical GDN layer {canon!r} (first metadata fetch of the "
            f"process) is NOT in the driver's slot-id source group {gi} ({gi_names[:3]}...). The "
            "metadata alias and _slot_ids would read DIFFERENT block tables -- the exact "
            "disagreement the alias exists to remove. vLLM's grouping/order changed; fix "
            "_mamba_group_index / the canonical registration together."
        )
    try:
        os.write(
            1,
            f"[ISOEXEC-GDN-GROUPS] pid={os.getpid()} {len(groups)} mamba KV-cache groups "
            f"(sizes {sizes}); slot-id source = group {gi}; metadata for ALL GDN layers "
            f"canonicalized to '{canon}' since the first forward (capture included), so the "
            "stack keeps one slot-id space (the pages carry no state -- only slot ids).\n".encode(),
        )
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def _slot_ids(runner, sched_out, n: int, nct):
    """This step's GDN state slot id per request row, matching what the metadata builder will use."""
    import numpy as np

    gi = _mamba_group_index(runner)
    bt = runner.input_batch.block_table.block_tables[gi].block_table.np
    bs = _align_block_size(runner)
    if bs <= 0:
        return bt[:n, 0]
    ns = np.zeros(n, dtype=np.int64)
    nst = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
    if nst:
        ids = runner.input_batch.req_ids
        for i in range(n):
            ns[i] = int(nst.get(ids[i], 0))
    seq = np.asarray(nct[:n], dtype=np.int64) + ns
    col = np.clip((seq - 1) // bs, 0, bt.shape[1] - 1)
    return bt[np.arange(n), col]


_DRIVER_SLOT_BY_REQ: dict[str, int] = {}


def _remember_slots_before_layers(req_ids, slots) -> None:
    """Remember the first prefill's slot names before the private GDN pools exist.

    The driver runs before the first forward, when ``CPR_LAYERS`` is still empty; retaining these
    names lets the next chunk rebind a row when Mamba ALIGN rotates to a new block.
    """

    for rid, slot in zip(req_ids, slots, strict=True):
        _DRIVER_SLOT_BY_REQ.setdefault(str(rid), int(slot))


def release_request(req_id, layers) -> None:
    """Return a request's private GDN rows to every pool's free list (host bookkeeping only).

    Keyed by the scheduler's request id, the one name that stays fixed while its slot id may rotate.
    A request that never prefilled holds nothing, so releasing it is a no-op.
    """
    old = _DRIVER_SLOT_BY_REQ.pop(str(req_id), None)
    if old is not None:
        for ly in layers:
            ly.release_slot(old)


def _sync_slot_lifecycle(runner, sched_out, n: int, nct, slots, layers) -> None:
    """Maintain request->slot ownership during chunked prefill.

    Mamba ALIGN may rotate a live request's block id between chunks, and the private CPR state must
    follow that rename without moving the row. Unscheduled requests keep their rows; finished and
    preempted ones are released. Host bookkeeping only, before the forward and the device-map flush.
    """

    live = getattr(runner, "requests", None)
    resumed = getattr(getattr(sched_out, "scheduled_cached_reqs", None), "resumed_req_ids", None) or ()
    finished = getattr(sched_out, "finished_req_ids", None) or ()
    preempted = getattr(sched_out, "preempted_req_ids", None) or ()

    # Steady-decode fast path: under mamba cache mode ``none`` column zero is a lifetime-stable slot
    # name, so with no lifecycle transition the full release/rebind pass is an identity.
    scheduled = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
    total_scheduled = getattr(sched_out, "total_num_scheduled_tokens", None) if sched_out is not None else None
    pure_decode = (
        n > 0
        and scheduled is not None
        and len(scheduled) == n
        and (int(total_scheduled) == n if total_scheduled is not None else all(int(q) == 1 for q in scheduled.values()))
    )
    nct_positive = bool((nct[:n] > 0).all()) if hasattr(nct, "dtype") else all(int(v) > 0 for v in nct[:n])
    if (
        pure_decode
        and not resumed
        and not finished
        and not preempted
        and live is not None
        and len(_DRIVER_SLOT_BY_REQ) == len(live)
        and _align_block_size(runner) <= 0
        and nct_positive
    ):
        # When the scheduler names lifecycle deltas explicitly, empty transition sets plus equal
        # live/owned counts prove it. Otherwise compare pairs, catching same-cardinality replacement.
        new_reqs = getattr(sched_out, "scheduled_new_reqs", None)
        if new_reqs is not None and not new_reqs:
            return
        if new_reqs is None:
            req_ids = runner.input_batch.req_ids
            for i in range(n):
                if _DRIVER_SLOT_BY_REQ.get(str(req_ids[i])) != int(slots[i]):
                    break
            else:
                return

    ids = [str(rid) for rid in runner.input_batch.req_ids[:n]]
    live_ids = set(ids) if live is None else {str(rid) for rid in live}
    resumed_ids = {str(rid) for rid in resumed}
    finished_ids = {str(rid) for rid in finished}
    preempted_ids = {str(rid) for rid in preempted}

    def release(rid: str) -> None:
        release_request(rid, layers)

    # Explicit scheduler transitions precede liveness inference and rebinding: that ordering covers
    # preemption (the runner keeps the request object) and the finished+new-same-id edge.
    for rid in finished_ids | preempted_ids:
        release(rid)
    for rid in list(_DRIVER_SLOT_BY_REQ):
        if rid not in live_ids:
            release(rid)
    for rid in resumed_ids:
        release(rid)

    for i, rid in enumerate(ids):
        slot = int(slots[i])
        old = _DRIVER_SLOT_BY_REQ.get(rid)
        # A reset to zero without a resumed_req_ids surface is still a fresh prefill.
        if old is not None and int(nct[i]) == 0 and rid not in resumed_ids:
            release(rid)
            old = None
        if old is not None and old != slot:
            for ly in layers:
                ly.rebind_slot(old, slot)
        _DRIVER_SLOT_BY_REQ[rid] = slot


_lazy_driver_installed = False


def _drive(runner, sched_out=None) -> None:
    """Fire pending boundary resyncs from the scheduler's host state, with zero device syncs.

    ``CommonAttentionMetadata.seq_lens_cpu`` must not be used here: its getter is an implicit
    D2H sync.
    """
    import numpy as np

    global _DRIVER_HOST_STEP
    _DRIVER_HOST_STEP = None  # fail closed on every early return below; this step must re-arm it

    from ...ops.gdn import (
        CPR_LAYERS,
    )

    # Before the layer-existence check: the KV-group geometry must be resolved before the first
    # forward builds any layer state, and on step 1 the layers do not exist yet.
    resolve_gdn_groups(runner)

    ib = runner.input_batch
    n = ib.num_reqs
    nct = ib.num_computed_tokens_cpu[:n]  # tokens consumed BEFORE this step, host numpy
    # A zero-row output can still carry a request's only finished/preempted transition, so it
    # must still reach lifecycle release and the device-map flush below.
    slots = (
        _slot_ids(runner, sched_out, n, nct) if n else np.empty(0, dtype=np.int64)
    )  # honours mamba align mode's rotating column
    from ...ops.gdn.shared.gdn_cpr_state import cpr_apc_enabled

    apc_on = cpr_apc_enabled()
    if apc_on:
        from .cpr_apc import track_requests

        track_requests(sched_out)
    if not CPR_LAYERS or not CPR_LAYERS[0].lazy_resync:
        # The first prefill reaches the driver before the private GDN pools exist; remember the
        # slot names its forward will assign so the next chunk can rebind.
        if n:
            _remember_slots_before_layers(ib.req_ids[:n], slots)
        return
    C = CPR_LAYERS[0].chunk_size

    # Latch scheduler lifecycle ownership before adoption can call `_assign`: from here on a
    # mapped row is never an LRU victim.
    if not CPR_LAYERS[0]._driver_managed:
        for ly in CPR_LAYERS:
            ly._driver_managed = True
        # A latch follows a pool built or reset empty (CPR sleep resets it): the driver forgets every
        # request whose row the pool no longer holds, so its map names exactly the pool's owners.
        held = CPR_LAYERS[0]._slot2row
        for rid in [rid for rid, slot in _DRIVER_SLOT_BY_REQ.items() if slot not in held]:
            del _DRIVER_SLOT_BY_REQ[rid]

    _sync_slot_lifecycle(runner, sched_out, n, nct, slots, CPR_LAYERS)

    # Prefix caching (cpr_apc): adopt the checkpoint of every admitted hit (a request first scheduled with
    # computed tokens and no row) and capture this step's boundary checkpoints, before any row is required,
    # claimed or published below.
    if apc_on and n:
        from .cpr_apc import worker_apc

        worker_apc(runner, CPR_LAYERS).step(ib.req_ids[:n], nct, slots, CPR_LAYERS)

    # Position mirror over the prefill suffix only. If the split cannot be validated, clear each
    # mirror so a continuation falls back to the exact D2H read.
    from ...ops.gdn import _HOST_BOOKKEEPING_STATS

    if n:
        nst = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
        _DRIVER_HOST_STEP = _build_driver_host_step(ib.req_ids[:n], nst, nct, slots)
        slot_pos = _driver_prefill_slot_positions(_DRIVER_HOST_STEP, nct)
        if slot_pos is None:
            for ly in CPR_LAYERS:
                ly.note_slot_positions(None)
            _HOST_BOOKKEEPING_STATS["driver_position_layer_calls"] += len(CPR_LAYERS)
        elif slot_pos:
            for ly in CPR_LAYERS:
                ly.note_slot_positions(slot_pos)
            nlayers = len(CPR_LAYERS)
            _HOST_BOOKKEEPING_STATS["driver_position_layer_calls"] += nlayers
            _HOST_BOOKKEEPING_STATS["driver_position_rows"] += nlayers * len(slot_pos)

    _HOST_BOOKKEEPING_STATS["driver_steps"] += 1

    # Fail-closed slot audit: on the device a padding lane and a live-but-unmapped slot both
    # fold onto row 0, but every id in `slots` belongs to a live request, so a miss is real.
    l0 = CPR_LAYERS[0]
    if not l0._native:
        global _slot_space_checked
        if not _slot_space_checked:
            nb = int(runner.kv_cache_config.num_blocks or 0)
            if nb >= l0.slot2row.numel():
                raise RuntimeError(
                    f"[isoexec-gdn] vLLM minted {nb} KV blocks but the GDN slot map holds only "
                    f"{l0.slot2row.numel()} entries including null padding -- block ids past the map would fold onto "
                    "null row 0 at decode. The resolved KV geometry changed after slot-map "
                    "initialization; rebuild the engine before capturing the new geometry."
                )
            _slot_space_checked = True
        # Every request past its first scheduled token must hold a mapped row before the forward
        # reads the device map; a miss is unrecoverable and would otherwise be silent.
        for i in range(n):
            if nct[i] > 0:
                l0.require_row(int(slots[i]))

    # Claim this step's prefill rows for the whole stack at once (pools that read a PrefillPlan), so
    # each GDN layer's prefill makes no host->device copies of its own. Only a positively identified
    # prefill suffix is claimed; anything else leaves each layer to claim its own rows.
    from ...ops.gdn import assign_prefill_rows, flush_slot_maps

    assign_prefill_rows(CPR_LAYERS, _DRIVER_HOST_STEP["pre_slots"] if _DRIVER_HOST_STEP is not None else ())

    # Publish the step's slot-map edits while still on the host and outside capture. Rebinds
    # above only touched the numpy mirror, so this must sit above every return path below.
    flush_slot_maps(CPR_LAYERS)

    if n == 0:
        return

    hits = np.nonzero((nct > 0) & (nct % C == 0))[0]
    if hits.size == 0:
        return
    # The batched resync dedups per (row, pos), so prefill-serviced boundaries are no-ops.
    pairs = [(int(slots[i]), int(nct[i])) for i in hits]
    from ...ops.gdn import layer_batched_resync

    layer_batched_resync(CPR_LAYERS, pairs)


class _V2Batch:
    """The ``input_batch`` fields ``_drive`` reads, rebuilt from V2 state for one scheduled batch."""

    def __init__(self, runner, batch_req_state) -> None:
        import numpy as np

        index = np.asarray(batch_req_state.idx_mapping_np)
        self.req_ids = list(batch_req_state.req_ids)
        self.num_reqs = len(self.req_ids)
        self.num_computed_tokens_cpu = runner.req_states.num_computed_tokens_np[index].astype(np.int64)
        self.block_table = _V2BlockTables(host_block_tables(runner.block_tables), index)


class _V2Rows:
    """One KV-cache group's block-table rows for this batch, gathered on first read: the driver reads one
    group (the mamba group's slot ids), so the other groups' [n, max_blocks] rows are never copied."""

    def __init__(self, mirror, index) -> None:
        self._mirror, self._index, self._np = mirror, index, None

    @property
    def np(self):
        if self._np is None:
            self._np = self._mirror[self._index]
        return self._np


class _V2BlockTables:
    def __init__(self, mirrors, index) -> None:
        from types import SimpleNamespace

        self.block_tables = [SimpleNamespace(block_table=_V2Rows(mirror, index)) for mirror in mirrors]


class _V2DriverView:
    """Present a V2 runner's scheduled step through the attributes ``_drive`` reads."""

    def __init__(self, runner, batch_req_state) -> None:
        self.kv_cache_config = runner.kv_cache_config
        self.vllm_config = runner.vllm_config
        self.requests = dict(runner.req_states.req_id_to_index)
        self.input_batch = _V2Batch(runner, batch_req_state)


def host_block_tables(block_tables):
    """Host mirror of V2 ``BlockTables`` rows, maintained by ``_mirror_block_ids``."""
    mirror = getattr(block_tables, "_isoexec_host_block_tables", None)
    if mirror is None:
        raise RuntimeError("[isoexec-gdn] V2 block tables have no host mirror; install the driver before KV init")
    return mirror


def _mirror_block_ids(append):
    """Record each request row's block IDs on the host as V2 stages them for the device."""
    import numpy as np
    from functools import wraps

    @wraps(append)
    def wrapped(self, req_index, new_block_ids, overwrite):
        result = append(self, req_index, new_block_ids, overwrite)
        mirror = self.__dict__.get("_isoexec_host_block_tables")
        if mirror is None:
            mirror = [np.zeros(tuple(table.gpu.shape), dtype=np.int64) for table in self.block_tables]
            self._isoexec_host_block_tables = mirror
        for group, block_ids in enumerate(new_block_ids):
            end = int(self.num_blocks.np[group, req_index])
            start = end - len(block_ids)
            if overwrite:
                mirror[group][req_index, :] = 0
            ids = [0] * len(block_ids) if getattr(self, "redirect_writes_to_null_block", False) else block_ids
            mirror[group][req_index, start:end] = ids
        return result

    setattr(wrapped, "_isoexec_host_mirror", True)
    return wrapped


def _install_v2_driver() -> bool:
    """Drive CPR resyncs from Model Runner V2's scheduled batch, before its inputs are prepared."""
    # vLLM is pinned to a release that ships Model Runner V2 (checks/dependencies.py); a missing
    # module must fail here rather than leave the V2 engine without its CPR driver.
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from .gdn_slot_map import wrap_cache_initialization

    if not getattr(BlockTables.append_block_ids, "_isoexec_host_mirror", False):
        BlockTables.append_block_ids = _mirror_block_ids(BlockTables.append_block_ids)
    setattr(V2Runner, "initialize_kv_cache", wrap_cache_initialization(V2Runner.initialize_kv_cache))
    orig_finish = V2Runner.finish_requests

    def finish_requests(self, scheduler_output, *args, **kw):
        # V2 retires finished, aborted and preempted requests here, at the head of every execute_model,
        # including steps that schedule no tokens and so never reach prepare_inputs (the warmup's cleanup
        # step, a step that only finishes or aborts requests). Releasing their rows here keeps the pool's
        # owners exactly the runner's live requests at every step.
        from ...ops.gdn import CPR_LAYERS

        ended = set(scheduler_output.finished_req_ids or ()) | set(scheduler_output.preempted_req_ids or ())
        for req_id in sorted(ended):
            release_request(req_id, CPR_LAYERS)
        return orig_finish(self, scheduler_output, *args, **kw)

    setattr(V2Runner, "finish_requests", finish_requests)
    orig_prepare = V2Runner.prepare_inputs

    def prepare_inputs(self, scheduler_output, batch_req_state, *args, **kw):
        # Called once per real (non-dummy) batch after request state, block tables and computed
        # counts reflect this step, and before attention metadata is built.
        _drive(_V2DriverView(self, batch_req_state), scheduler_output)
        return orig_prepare(self, scheduler_output, batch_req_state, *args, **kw)

    setattr(V2Runner, "prepare_inputs", prepare_inputs)
    return True


def install_cpr_lazy_driver() -> bool:
    """Install the CPR LAZY-resync driver on Model Runner V2's step boundaries.

    The driver runs on the host once per step, outside any CUDA graph, so it is the one place a
    boundary resync can run while decode() stays capturable. A missed resync would move logprobs
    silently, so the driver never fails soft -- any exception aborts the step.
    """
    global _lazy_driver_installed
    if _lazy_driver_installed:
        return True
    if not _install_v2_driver():
        raise RuntimeError("[isoexec-gdn] the CPR driver needs vLLM Model Runner V2 (vllm.v1.worker.gpu)")
    _lazy_driver_installed = True
    logger.info("[isoexec-gdn] CPR LAZY resync driver installed on Model Runner V2 prepare_inputs "
                "(host-state only, no device syncs)")
    return True
