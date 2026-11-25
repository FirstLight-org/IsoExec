"""Engine state for this family: the vLLM-facing state machine over a :class:`~.ops.CprState` pool."""

from __future__ import annotations

import os

import torch

from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as conv
from isoexec.ops.gdn.gdn_cpr_sm100 import ops
from isoexec.ops.gdn.gdn_cpr_sm100._extension import extension
from isoexec.ops.gdn.shared.gdn_cpr_sleep import alloc_cpr_arena, register_layer
from isoexec.ops.gdn.shared.gdn_cpr_state import CPR_LAYERS, _dev_idx, cpr_apc_enabled
from isoexec.ops.gdn.shared.gdn_ops import gdn_kernel_mode, gdn_native_kernels_enabled
from isoexec.ops.gdn.shared.gdn_decode_group import DecodeGroup
from isoexec.ops.gdn.shared.gdn_recurrent_state import RecurrentGDN, chunked_prefill_enabled, native_state_enabled

_NOTED = False


def _say(msg: str) -> None:
    """fd 1, not ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-GDN-CPR-SM100] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def refuse_incompatible_flags() -> None:
    """Require the CPR branch, in-kernel preparation, a private pool, and no APC store."""
    global _NOTED
    bad: list[str] = []
    mode = gdn_kernel_mode()
    if mode != "cpr":
        bad.append(f"ISOEXEC_GDN_KERNEL={mode!r}: the engine binding rides the cpr branch; set "
                   "ISOEXEC_GDN_KERNEL=cpr (or ISOEXEC_GDN_KERNEL=recurrent, which selects "
                   "ops/gdn/gdn_recurrent_sm100)")
    if not gdn_native_kernels_enabled():
        bad.append("ISOEXEC_GDN_NATIVE_KERNELS=0: this family's core takes the RAW post-conv q/k (l2norm "
                   "and gating are in-kernel); there is no eager-prep variant")
    if native_state_enabled():
        bad.append("ISOEXEC_GDN_NATIVE_STATE=1: the pool is private (vLLM's mamba pages are a slot-id "
                   "source only); state in vLLM's pages is the recurrent core's, selected by "
                   "ISOEXEC_GDN_KERNEL=recurrent")
    if cpr_apc_enabled():
        bad.append("ISOEXEC_GDN_CPR_APC: no boundary-state store is published or adopted by this family, "
                   "so a prefix-cache hit would resume state this pool never scanned")
    if bad:
        raise RuntimeError("[isoexec-gdn] the selected GDN core is ops/gdn/gdn_cpr_sm100, and:\n  - "
                           + "\n  - ".join(bad))
    if not _NOTED:
        _NOTED = True
        _say("engine core = gdn.core:sm100_cpr_v1 (ops/gdn/gdn_cpr_sm100), in-kernel boundary resync; "
             f"chunked-prefill resume {'ON' if chunked_prefill_enabled() else 'OFF (ISOEXEC_GDN_CHUNKED_PREFILL=0)'}")


def build_cpr_sm100_gdn(*, max_num_seqs: int, chunk_size: int, **kw) -> CprSm100GDN:
    """Build a :class:`CprSm100GDN` sized by the scheduler's concurrency cap (``chunk_size`` is carried for
    the shared lifecycle driver; this family's chunk is fixed at 64 by its recipe)."""
    return CprSm100GDN(capacity=max_num_seqs, chunk_size=chunk_size, **kw)


class CprSm100GDN(RecurrentGDN):
    """RecurrentGDN whose core is ``ops/gdn/gdn_cpr_sm100`` and whose ``ssm_state`` is a record pool."""

    #: Read by ``gdn_cpr_state.layer_batched_resync``: the resync is inside the decode launch.
    needs_boundary_resync = False
    #: Read by ``runtimes/vllm/_gdn.py``: a pure-decode step hands over the q|k|v column window of the
    #: in_proj output (no cat); the conv update reads it row-strided and writes a fresh output.
    reads_strided_decode_qkv = True
    #: Read by ``gdn_fused_split.defer_ab``: alpha/beta may arrive as strided column views. The decode
    #: kernel reads them strided; the prefill split compacts them in its own launch.
    defers_ab = True
    #: Read by ``gdn_recurrent_state.assign_prefill_rows``: prefill takes the step's PrefillPlan.
    consumes_prefill_plan = True
    #: Set by :meth:`DecodeGroup.add` (runtimes/vllm/_gdn.py groups one model's pools).
    decode_group: DecodeGroup | None = None

    def __init__(self, *, capacity: int, chunk_size: int, ssm_state=None, conv_state=None, **kw):
        if chunk_size != ops.C:
            raise ValueError(f"[isoexec-gdn] gdn_cpr_sm100 requires chunk_size={ops.C}, got {chunk_size}")
        if ssm_state is not None or conv_state is not None:
            raise ValueError("[isoexec-gdn] the gdn_cpr_sm100 state machine owns its state pool; native "
                             "vLLM state is not supported. Unset ISOEXEC_GDN_NATIVE_STATE.")
        refuse_incompatible_flags()
        super().__init__(capacity=capacity, **kw)
        self.chunk_size = int(chunk_size)
        rows = capacity + 1   # row 0 = null, as in the parent pool
        dev = self.ssm_state.device
        HV, K, V = self.num_v_heads, self.head_k_dim, self.head_v_dim
        if (K, V) != (128, 128):
            raise ValueError(f"[isoexec-gdn] gdn_cpr_sm100 is a head_k_dim = head_v_dim = 128 family, got "
                             f"({K}, {V})")
        for name, t in (("A_log", self.A_log), ("dt_bias", self.dt_bias)):
            if t.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"[isoexec-gdn] gdn_cpr_sm100 reads {name} natively in fp32 or bf16, got "
                                 f"{t.dtype}; a per-call cast would allocate inside the captured decode graph")
        nbytes = ops.record_layout()["bytes"]
        specs = [("ssm_state", (rows, HV, nbytes), torch.uint8),
                 ("conv_state", tuple(self.conv_state.shape), self.conv_state.dtype)]
        arena = alloc_cpr_arena(specs, dev)
        if arena is None:
            self._flat_arena = None
            self._publish("ssm_state", torch.zeros(rows, HV, nbytes, dtype=torch.uint8, device=dev))
        else:
            self._publish("_flat_arena", arena.pop("_flat"))
            for name, t in arena.items():
                self._publish(name, t)
            register_layer(self, self._flat_arena, capacity)
        self.records = ops.CprState(self.ssm_state)
        self._publish("_null32", torch.full((), -1, dtype=torch.int32, device=dev))
        self.lazy_resync = True
        self._driver_managed = False
        self._row_pos: dict[int, int] = {}
        extension()   # compile at bind, never under CUDA-graph capture
        CPR_LAYERS.append(self)

    # -- shared-driver surface ------------------------------------------
    def note_slot_positions(self, slot_pos) -> None:
        """Latch scheduler lifecycle ownership; positions live in the records, so none are mirrored."""
        self._driver_managed = True
        if slot_pos is None:
            self._row_pos.clear()
            return
        for slot, pos in slot_pos:
            row = self._slot2row.get(int(slot))
            if row is not None:
                self._row_pos[row] = int(pos)

    def host_resync(self, slot_pos) -> int:
        """Nothing to resynchronise on the host (the launch replaces boundary states itself)."""
        del slot_pos
        return 0

    def _on_row_released(self, row: int) -> None:
        self._row_pos.pop(int(row), None)

    # -- public API ----------------------------------------------------
    @torch.no_grad()
    def prefill(self, slots, slots_cpu, x, a, b, qsl, has_initial_state=None, conv_metadata=None,
                prefill_query_start_loc=None):
        """Packed varlen prefill for N prompts; returns ``o [T, HV, DV]``."""
        N = len(slots_cpu)
        plan, is_cont = self._claim_prefill(slots_cpu, has_initial_state)
        rows = plan.rows
        # Built once per step and shared by every layer (the pools are lockstepped).
        rows32 = plan.device("rows32", lambda: _dev_idx(rows, torch.int32, x.device))
        cu = prefill_query_start_loc if prefill_query_start_loc is not None else _dev_idx(qsl, torch.int32,
                                                                                         x.device)
        has = plan.device(("has", tuple(is_cont)), lambda: _dev_idx(is_cont, torch.bool, x.device))
        y = conv.causal_conv1d_fn(x.transpose(0, 1), self.conv_weight, self.conv_bias,
                                  activation=self.activation, conv_states=self.conv_state,
                                  has_initial_state=has, cache_indices=rows32, query_start_loc=cu,
                                  metadata=conv_metadata).transpose(0, 1)
        q, k, v, a, b = self._split_qkv_raw(y, a, b)
        # A padding lane (``null_slot``, row 0: capture and warmup dummies) reads and writes no record.
        slot_out = rows32 if all(rows) else plan.device(
            "slot_out", lambda: torch.where(rows32 > 0, rows32, self._null32))
        slot_in = plan.device(("slot_in", tuple(is_cont)), lambda: torch.where(has, rows32, self._null32))
        return ops.prefill(q, k, v, a, b, self.A_log, self.dt_bias, cu[: N + 1], state=self.records,
                           slot_in=slot_in, slot_out=slot_out)

    @torch.no_grad()
    def decode(self, slots, x, a, b):
        """One decode step for N requests; returns ``o [N, HV, DV]``. Host-free and shaped by N alone, so
        it captures into a CUDA graph as-is.

        ``decode_rows`` resolves ``rows32 = slot2row[clamp(slots)]``, the core's lane map
        ``where(rows32 > 0, rows32, -1)`` (the slot map's null row 0 -- unknown slots, padded replay
        lanes -- is this family's NULL lane) and the LRU stamp (``clock += 1; last_used[rows] = clock``),
        for every pool of the group in one launch when it can (:meth:`_group_rows`). ``x`` may be the
        row-strided q|k|v column window of the in_proj output: the conv reads it in place and writes a
        packed output. The core reads q/k/v as row-strided column views of that output and alpha/beta
        as the caller's views; it takes every stride, so nothing is copied.
        """
        rows32, lane32 = self._group_rows(slots) or _decode_rows(slots, [self])[0]
        y = conv.causal_conv1d_update(x, self.conv_state, self.conv_weight, self.conv_bias, self.activation,
                                      conv_state_indices=rows32)
        T = y.shape[0]
        kd = self.num_k_heads * self.head_k_dim
        q = y[:, :kd].view(T, self.num_k_heads, self.head_k_dim)
        k = y[:, kd:2 * kd].view(T, self.num_k_heads, self.head_k_dim)
        v = y[:, 2 * kd:].view(T, self.num_v_heads, self.head_v_dim)
        return ops.decode(q, k, v, a, b, self.A_log, self.dt_bias, self.records, lane32)

    def _group_rows(self, slots):
        """This pool's ``(rows32, lane32)`` from its group's one-launch bookkeeping, or ``None``.

        The results live on the slot tensor (one per forward: ``gdn_engine_patch._decode_slots``), so
        they never outlive the step that computed them. A round starts when no member has an entry
        left; a member whose entry is already taken while others still hold theirs (a pool decoded
        twice on one slot tensor) gets ``None`` and stamps itself, so no pool is ever stamped twice.
        """
        group = self.decode_group
        # Only a slot tensor every member decodes on this step (``_decode_slots`` marks the canonical
        # one) may carry a round: pre-stamping a member that then decodes on another tensor would stamp
        # it twice.
        if group is None or len(group.pools) < 2 or not getattr(slots, "_isoexec_step_shared", False):
            return None
        cache = getattr(slots, "_isoexec_cpr_group_rows", None)
        if cache is not None and cache[0] is group:
            entry = cache[1].pop(id(self), None)
            if entry is not None or cache[1]:
                return entry
        entries: dict = {}
        try:
            slots._isoexec_cpr_group_rows = (group, entries)
        except (AttributeError, TypeError):
            return None  # nowhere to keep the round: every pool stamps itself
        entries.update(zip((id(p) for p in group.pools), _decode_rows(slots, group.pools)))
        return entries.pop(id(self))


def _decode_rows(slots, pools):
    """``[(rows32, lane32)]`` for ``pools`` on one step's ``slots``, stamping each pool once, in one launch."""
    n = slots.numel()
    rows = [torch.empty(n, dtype=torch.int32, device=slots.device) for _ in pools]
    lanes = [torch.empty(n, dtype=torch.int32, device=slots.device) for _ in pools]
    extension().decode_rows(slots, [p.slot2row for p in pools], rows, lanes, [p._clock for p in pools],
                            [p.last_used for p in pools])
    return list(zip(rows, lanes))
