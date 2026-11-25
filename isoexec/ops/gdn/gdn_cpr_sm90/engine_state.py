"""Engine state for this family: the vLLM-facing state machine over a :class:`~.ops.CprState` pool.

Shared state ownership supplies the slot map, convolution window and installed
``causal_conv1d_fn`` / ``causal_conv1d_update`` adapters, padding contract, and
discard-at-sleep arena. This provider owns the 137,984-byte CPR record (format 3) per
(row, value head): the fp32 boundary state encoded exactly as its bf16 image, low 16 bits and a tie bit,
open-chunk rows and position.
Chunk-boundary resync happens inside the decode launch for CUDA graph capture.

This module connects the provider to ``gdn.shared`` and the convolution package;
the arithmetic kernels do not import the shared engine layer.
"""

from __future__ import annotations

import os

import torch

from isoexec.ops.gdn.conv_sm90.vendored import causal_conv1d as conv
from isoexec.ops.gdn.shared.gdn_cpr_sleep import alloc_cpr_arena, register_layer
from isoexec.ops.gdn.shared.gdn_cpr_state import CPR_LAYERS, _dev_idx, cpr_apc_enabled
from isoexec.ops.gdn.shared.gdn_decode_group import DecodeGroup
from isoexec.ops.gdn.shared.gdn_ops import gdn_kernel_mode, gdn_native_kernels_enabled
from isoexec.ops.gdn.shared.gdn_recurrent_state import RecurrentGDN, chunked_prefill_enabled, native_state_enabled
from isoexec.ops.gdn.gdn_cpr_sm90 import ops
from isoexec.ops.gdn.gdn_cpr_sm90._extension import extension

_NOTED = False


def _say(msg: str) -> None:
    """fd 1, not ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-GDN-CPR-SM90] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def refuse_incompatible_flags() -> None:
    """Require the CPR branch, in-kernel preparation and a private pool; the boundary store (prefix caching)
    rides the chunked-prefill resume, so ``ISOEXEC_GDN_CPR_APC`` requires ``ISOEXEC_GDN_CHUNKED_PREFILL=1``."""
    global _NOTED
    bad: list[str] = []
    mode = gdn_kernel_mode()
    if mode != "cpr":
        bad.append(f"ISOEXEC_GDN_KERNEL={mode!r}: the engine binding rides the cpr branch; set ISOEXEC_GDN_KERNEL=cpr")
    if not gdn_native_kernels_enabled():
        bad.append("ISOEXEC_GDN_NATIVE_KERNELS=0: this family's core takes the RAW post-conv q/k (l2norm and "
                   "gating are in-kernel); there is no eager-prep variant")
    if native_state_enabled():
        bad.append("ISOEXEC_GDN_NATIVE_STATE=1: the pool is private (vLLM's mamba pages are a slot-id source only)")
    if cpr_apc_enabled() and not chunked_prefill_enabled():
        bad.append("ISOEXEC_GDN_CPR_APC without ISOEXEC_GDN_CHUNKED_PREFILL=1: a prefix-cache hit resumes the "
                   "adopted boundary record as a chunked-prefill continuation, which this pool only does with "
                   "the resume flag on")
    if bad:
        raise RuntimeError("[isoexec-gdn] the selected GDN core is ops/gdn/gdn_cpr_sm90, and:\n  - " + "\n  - ".join(bad))
    if not _NOTED:
        _NOTED = True
        _say("engine core = gdn.core:cuda_cpr_v1 (ops/gdn/gdn_cpr_sm90), in-kernel boundary resync; chunked-prefill "
             f"resume {'ON' if chunked_prefill_enabled() else 'OFF (ISOEXEC_GDN_CHUNKED_PREFILL=0)'}; prefix-cache "
             f"boundary store {'ON' if cpr_apc_enabled() else 'OFF'}")


def build_cpr_sm90_gdn(*, max_num_seqs: int, chunk_size: int, **kw) -> CprSm90GDN:
    """Build a :class:`CprSm90GDN` sized by the scheduler's concurrency cap (``chunk_size`` is carried for the
    shared lifecycle driver; this family's chunk is fixed at 64 by its recipe)."""
    return CprSm90GDN(capacity=max_num_seqs, chunk_size=chunk_size, **kw)


class CprSm90GDN(RecurrentGDN):
    """RecurrentGDN whose core is ``ops/gdn/gdn_cpr_sm90`` and whose ``ssm_state`` is a record pool."""

    #: Read by ``gdn_cpr_state.layer_batched_resync``: the resync is inside the decode launch.
    needs_boundary_resync = False
    #: Read by ``gdn_recurrent_state.assign_prefill_rows``: prefill takes the step's PrefillPlan.
    consumes_prefill_plan = True
    #: Read by ``runtimes/vllm/_gdn.py``: a pure-decode step hands over the q|k|v column window of the
    #: in_proj output (no cat); the conv update reads it row-strided and writes a fresh output.
    reads_strided_decode_qkv = True
    #: Read by ``gdn_fused_split.defer_ab``: alpha/beta may arrive as strided column views. The decode
    #: kernel reads them strided; the prefill split compacts them in its own launch.
    defers_ab = True
    #: Set by :meth:`DecodeGroup.add` (runtimes/vllm/_gdn.py groups one model's pools).
    decode_group: DecodeGroup | None = None

    def __init__(self, *, capacity: int, chunk_size: int, ssm_state=None, conv_state=None, **kw):
        if chunk_size != ops.C:
            raise ValueError(f"[isoexec-gdn] gdn_cpr_sm90 requires chunk_size={ops.C}, got {chunk_size}")
        if ssm_state is not None or conv_state is not None:
            raise ValueError("[isoexec-gdn] the gdn_cpr_sm90 state machine owns its state pool; native vLLM state "
                             "is not supported. Unset ISOEXEC_GDN_NATIVE_STATE.")
        refuse_incompatible_flags()
        super().__init__(capacity=capacity, **kw)
        self.chunk_size = int(chunk_size)
        rows = capacity + 1   # row 0 = null, as in the parent pool
        dev = self.ssm_state.device
        HV, K, V = self.num_v_heads, self.head_k_dim, self.head_v_dim
        if (K, V) != (128, 128):
            raise ValueError(f"[isoexec-gdn] gdn_cpr_sm90 is a head_k_dim = head_v_dim = 128 family, got ({K}, {V})")
        for name, t in (("A_log", self.A_log), ("dt_bias", self.dt_bias)):
            if t.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"[isoexec-gdn] gdn_cpr_sm90 reads {name} natively in fp32 or bf16, got {t.dtype}; "
                                 "a per-call cast would allocate inside the captured decode graph")
        layout = ops.record_layout()
        nbytes = layout["bytes"]
        if (layout["format"], nbytes) != (ops.RECORD_FORMAT, ops.RECORD_BYTES):
            raise RuntimeError(f"[isoexec-gdn] gdn_cpr_sm90 record format {layout['format']} ({nbytes} B) does not match "
                               f"ops.RECORD_FORMAT {ops.RECORD_FORMAT} ({ops.RECORD_BYTES} B): a stale extension build")
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
        # measure the decode split per batch size once per process, before any graph captures a decode
        ops.tune_decode_split(HV, self.num_k_heads, dev)
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
        """Nothing to resynchronise on the host (the decode launch replaces boundary states itself)."""
        del slot_pos
        return 0

    def _on_row_released(self, row: int) -> None:
        self._row_pos.pop(int(row), None)

    # -- public API ------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, slots, slots_cpu, x, a, b, qsl, has_initial_state=None, conv_metadata=None,
                prefill_query_start_loc=None):
        """Packed varlen prefill for N prompts; returns ``o [T, HV, DV]``. A fresh prompt takes the NULL
        input slot (a negative slot reads nothing, which is the fresh record, empty prompts included); a
        continuation takes its own row and resumes from its record."""
        N = len(slots_cpu)
        dev = x.device
        plan, is_cont = self._claim_prefill(slots_cpu, has_initial_state)
        rows = plan.rows
        # Built once per step and shared by every layer (the pools are lockstepped).
        rows32 = plan.device("rows32", lambda: _dev_idx(rows, torch.int32, dev))
        cu = prefill_query_start_loc
        if cu is None:
            cu = _dev_idx(qsl, torch.int32, dev)
        has = plan.device(("has", tuple(is_cont)), lambda: _dev_idx(is_cont, torch.bool, dev))
        # Installation may happen after this state module is imported. Resolve the
        # live adapter so trainer and engine always execute the installed recipe.
        y = conv.causal_conv1d_fn(x.transpose(0, 1), self.conv_weight, self.conv_bias, activation=self.activation,
                             conv_states=self.conv_state, has_initial_state=has, cache_indices=rows32,
                             query_start_loc=cu, metadata=conv_metadata).transpose(0, 1)
        q, k, v, a, b = self._split_qkv_raw(y, a, b)
        # A padding lane (``null_slot``, row 0: capture and warmup dummies) reads and writes no record.
        slot_out = rows32 if all(rows) else plan.device(
            "slot_out", lambda: torch.where(rows32 > 0, rows32, self._null32))
        slot_in = plan.device(("slot_in", tuple(is_cont)), lambda: torch.where(has, rows32, self._null32))
        return ops.prefill(q, k, v, a, b, self.A_log, self.dt_bias, cu[: N + 1], state=self.records,
                           slot_in=slot_in, slot_out=slot_out,
                           cu_seqlens_cpu=torch.as_tensor(list(qsl[: N + 1]), dtype=torch.int64))

    @torch.no_grad()
    def decode(self, slots, x, a, b):
        """One decode step for N requests; returns ``o [N, HV, DV]``. Host-free and shaped by N alone, so
        it captures into a CUDA graph as-is.

        Rows, NULL lanes and the LRU stamp come from the decode group's one ``decode_rows`` launch per step
        (or this pool's own); q/k/v are column views of the conv output and alpha/beta strided columns of the
        in_proj output, which the core reads in place."""
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
    """``[(rows32, lane32)]`` for ``pools`` on one step's ``slots``, stamping each pool once, in one launch
    (csrc/rows.cu): rows = slot2row[clamp(slots)], lane32 = rows or NULL (-1) for the null row, clock += 1,
    last_used[rows] = clock -- the per-layer chain it replaces, as index arithmetic."""
    n = slots.numel()
    rows = [torch.empty(n, dtype=torch.int32, device=slots.device) for _ in pools]
    lanes = [torch.empty(n, dtype=torch.int32, device=slots.device) for _ in pools]
    extension().decode_rows(slots, [p.slot2row for p in pools], rows, lanes, [p._clock for p in pools],
                            [p.last_used for p in pools])
    return list(zip(rows, lanes))


# -- prefix-cache boundary records ------------------------------------------------------------------------
# At a position p that is a multiple of the chunk (64), a record's open chunk is empty: only rows [0, p mod 64)
# of its buffers are meaningful (csrc/state.cuh), so the state is exactly the boundary state's encoded bytes
# [sh | lo | tie] (bytes [0, kb) of the record) plus pos = p, and the request's conv window. A boundary
# checkpoint is those bytes for every layer; installing them on another row and setting pos = p makes that row
# indistinguishable, to every kernel, from a row that scanned the same p tokens itself.


def boundary_bytes() -> int:
    """Bytes of one (row, head) record that carry a chunk-boundary state: the encoded S (``sh``, ``lo``, ``tie``)."""
    layout = ops.record_layout()
    if not (layout["sh"] == 0 and layout["lo"] < layout["tie"] < layout["kb"]):
        raise RuntimeError(f"[isoexec-gdn] gdn_cpr_sm90 record layout {layout} no longer starts with the encoded "
                           "boundary state; the prefix-cache boundary record must be redefined")
    return int(layout["kb"])


@torch.no_grad()
def capture_boundaries(layers: list[CprSm90GDN], rows: list[int], positions: list[int]):
    """Boundary checkpoints of ``rows`` across the whole stack, one per row: ``[(rec, conv)]`` with ``rec``
    uint8 ``[L, HV, boundary_bytes()]`` and ``conv`` ``[L, *conv_state.shape[1:]]``, fresh tensors.

    Every position must be a positive multiple of the chunk. Enqueued on the current stream after every kernel
    already issued for these rows, so it reads the state they leave."""
    if any(p <= 0 or p % ops.C for p in positions):
        raise ValueError(f"[isoexec-gdn] boundary checkpoints exist only at positive multiples of {ops.C}: {positions}")
    if not rows:
        return []
    l0 = layers[0]
    dev = l0.ssm_state.device
    kb = boundary_bytes()
    m = len(rows)
    rows_t = _dev_idx(rows, torch.long, dev)
    rec = torch.empty(m, len(layers), l0.num_v_heads, kb, dtype=torch.uint8, device=dev)
    win = torch.empty(m, len(layers), *l0.conv_state.shape[1:], dtype=l0.conv_state.dtype, device=dev)
    for j, ly in enumerate(layers):
        rec[:, j] = ly.ssm_state[:, :, :kb].index_select(0, rows_t)
        win[:, j] = ly.conv_state.index_select(0, rows_t)
    return [(rec[i].clone(), win[i].clone()) for i in range(m)] if m > 1 else [(rec[0], win[0])]


@torch.no_grad()
def install_boundaries(layers: list[CprSm90GDN], rows: list[int], positions: list[int], checkpoints) -> None:
    """Make each row of ``rows`` the state its checkpoint was captured from: boundary bytes, conv window and
    position (every copy of ``pos``) in every layer. The rows' other record bytes are past ``pos mod 64 == 0``
    and are never read."""
    if any(p <= 0 or p % ops.C for p in positions):
        raise ValueError(f"[isoexec-gdn] boundary checkpoints exist only at positive multiples of {ops.C}: {positions}")
    if not rows:
        return
    l0 = layers[0]
    dev = l0.ssm_state.device
    kb = boundary_bytes()
    for rec, win in checkpoints:
        if rec.shape != (len(layers), l0.num_v_heads, kb) or win.shape[0] != len(layers):
            raise RuntimeError(f"[isoexec-gdn] a boundary checkpoint of shape {tuple(rec.shape)} does not fit this "
                               f"stack ({len(layers)} layers, {l0.num_v_heads} heads, {kb} bytes)")
    rows_t = _dev_idx(rows, torch.long, dev)
    rec = torch.stack([c[0] for c in checkpoints], dim=1)    # [L, M, HV, kb]
    win = torch.stack([c[1] for c in checkpoints], dim=1)    # [L, M, ...]
    pos = _dev_idx(positions, torch.int32, dev).view(-1, 1, 1).expand(len(rows), l0.num_v_heads, ops.NPOS)
    for j, ly in enumerate(layers):
        ly.ssm_state[rows_t, :, :kb] = rec[j]
        ly.conv_state[rows_t] = win[j]
        ly.records.pos[rows_t] = pos
        for row, p in zip(rows, positions):
            ly._row_pos[int(row)] = int(p)


# The engine reaches these through the pool's class (runtimes/vllm/cpr_apc.py), never by importing this family.
CprSm90GDN.capture_boundaries = staticmethod(capture_boundaries)
CprSm90GDN.install_boundaries = staticmethod(install_boundaries)
