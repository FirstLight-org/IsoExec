"""SM100 GLM Kimi Delta Attention core (recipe ``sm100_kda_v1``, csrc/recipe.cuh).

``forward`` evaluates the chunk recipe for a packed varlen call: q, k, v, log-decay ``[T, H, 128]``, beta
``[T, H]``, int32 ``cu_seqlens [N + 1]``, optional fp32 initial state ``[N, H, 128, 128]``. Chunks are
anchored at each sequence's first token of the call. Two launches after the integer plan: phase 1 over
every (chunk, head) publishes the state-independent rows to a token-indexed workspace, phase 2 walks each
(sequence, head, 64-column value slice) through its chunks with the fp32 state in registers.

Glue in this file is allocation, views and dtype/shape checks only.
"""

from __future__ import annotations

import torch

from isoexec.ops.kda import _checks
from isoexec.ops.kda.glm_sm100 import record_layout
from isoexec.ops.kda.glm_sm100._extension import extension
from isoexec.ops.kda.glm_sm100.spec import CHUNK, CHUNK_LENGTH, HEAD_DIM

_GATE_DTYPES = (torch.float32, torch.bfloat16)

#: CUDA's limit of gridDim.y. The state pass (forward, prefill), the backward state pass, the decode row kernel and
#: record_state put one sequence (decode lane, slot) per grid row on that axis: a call with more is refused here,
#: before any launch, instead of launching nothing.
MAX_SEQUENCES = 65535


def _check_count(n: int, what: str):
    if n > MAX_SEQUENCES:
        raise ValueError(f"kda glm_sm100: {n} {what} in one call, more than the {MAX_SEQUENCES} the kernels' grid "
                         f"y axis holds; split the call")


def _check(q, k, v, g, beta, cu_seqlens, initial_state):
    if not q.is_cuda:
        raise ValueError("kda glm_sm100 requires CUDA tensors")
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM or k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        raise ValueError(f"kda glm_sm100 expects q, k, v, log-decay [T, H, {HEAD_DIM}]")
    if beta.shape != q.shape[:2]:
        raise ValueError("kda glm_sm100 expects beta [T, H]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("kda glm_sm100 requires bf16 q, k, v")
    if g.dtype not in _GATE_DTYPES or beta.dtype not in _GATE_DTYPES:
        raise ValueError("kda glm_sm100 requires fp32 or bf16 log-decay and beta")
    if cu_seqlens is not None and (cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1 or cu_seqlens.device != q.device):
        raise ValueError("kda glm_sm100 requires int32 cu_seqlens on the input device")
    for t in (q, k, v, g, beta):
        if t.device != q.device:
            raise ValueError("kda glm_sm100 requires all inputs on one device")
    n = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1   # None: one token per sequence
    if initial_state is not None:
        if initial_state.shape != (n, q.shape[1], HEAD_DIM, HEAD_DIM) or initial_state.dtype != torch.float32:
            raise ValueError(f"kda glm_sm100 initial_state must be fp32 [{n}, {q.shape[1]}, 128, 128]")


def max_chunks(T: int, nseq: int) -> int:
    """Upper bound of a call's chunk count: every sequence adds at most one partial chunk."""
    return T // CHUNK_LENGTH + nseq


def make_plan(cu_seqlens: torch.Tensor, T: int, records=None, slot_in=None, H: int = 0):
    nseq = cu_seqlens.shape[0] - 1
    maxch = max_chunks(T, nseq) + (nseq if records is not None else 0)
    buf = torch.empty(1 + 5 * maxch + 2 * nseq + 1, dtype=torch.int32, device=cu_seqlens.device)
    if nseq > 0:
        extension().plan(cu_seqlens, buf, maxch, records, slot_in, H)
    else:
        buf.zero_()
    return buf, maxch


def workspace(T: int, H: int, maxch: int, device):
    bf = dict(dtype=torch.bfloat16, device=device)
    return dict(W=torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device), U=torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device),
                QG=torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device), KD=torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device),
                MQ=torch.empty(T, H, CHUNK, dtype=torch.bfloat16, device=device),
                GL=torch.empty(max(maxch, 1), H, HEAD_DIM, dtype=torch.float32, device=device))


def forward(q, k, v, g, beta, cu_seqlens, initial_state=None, *, output_final_state=False, saves=None,
            records=None, slot_in=None, slot_out=None):
    """``(o [T, H, 128] bf16, final_state [N, H, 128, 128] fp32 or None)``.

    ``saves`` (backward recompute only) is a dict that receives the workspace, the plan, S^ at every chunk
    start and the deltas.
    """
    _check_count(cu_seqlens.shape[0] - 1, "sequences")
    _check(q, k, v, g, beta, cu_seqlens, initial_state)
    T, H, _ = q.shape
    nseq = cu_seqlens.shape[0] - 1
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    plan, maxch = make_plan(cu_seqlens, T, records, slot_in, H)
    # the workspace has a row for every chunk row this call evaluates: with records a continuing
    # sequence's first chunk re-evaluates up to 63 rows it takes from its record
    ws = workspace(T + (63 * nseq if records is not None else 0), H, maxch, q.device)
    carry = (torch.empty(nseq, H, extension().carry_bytes(), dtype=torch.uint8, device=q.device)
             if records is not None else None)
    o = torch.empty_like(v)
    ht = (torch.empty(nseq, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=q.device)
          if output_final_state else None)
    h0 = initial_state.contiguous() if initial_state is not None else None
    save_S = save_d = save_X = save_KK = save_Xu = save_Kg = None
    if saves is not None:
        bf, f32 = dict(dtype=torch.bfloat16, device=q.device), dict(dtype=torch.float32, device=q.device)
        save_S = torch.empty(max(maxch, 1), H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
        save_d = torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
        save_X, save_KK = torch.empty(T, H, CHUNK, dtype=torch.float32, device=q.device), torch.empty(T, H, CHUNK, dtype=torch.float32, device=q.device)
        save_Xu, save_Kg = torch.empty(T, H, CHUNK, dtype=torch.bfloat16, device=q.device), torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if nseq > 0:
        extension().fwd(q, k, v, g, beta, cu_seqlens, plan, maxch, ws["W"], ws["U"], ws["QG"], ws["KD"], ws["MQ"],
                        ws["GL"], h0, ht, o, save_S, save_d, True, True,
                        save_X, save_KK, save_Xu, save_Kg, records, slot_in, slot_out, carry)
    elif ht is not None:
        ht.zero_()
    if saves is not None:
        saves.update(ws, plan=plan, maxch=maxch, S=save_S, d=save_d, X=save_X, KK=save_KK, Xu=save_Xu,
                     Kg=save_Kg)
    return o, ht


def decode(q, k, v, g, beta, state, *, out_state=None):
    """One token per sequence: q, k, v, g ``[N, H, 128]``, beta ``[N, H]``, fp32 ``state [N, H, 128, 128]``
    (None: zero). Returns ``(o [N, H, 128], new_state)``; ``out_state`` may be ``state`` itself (in place).

    Bit for bit ``forward`` of N one-token sequences (README "Relations")."""
    N = q.shape[0]
    _check(q, k, v, g, beta, None, state)
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if state is not None and not state.is_contiguous():
        raise ValueError("kda glm_sm100 decode requires a contiguous state")
    if out_state is None:
        out_state = torch.empty(N, q.shape[1], HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=q.device)
    elif out_state.shape != (N, q.shape[1], HEAD_DIM, HEAD_DIM) or out_state.dtype != torch.float32 \
            or not out_state.is_contiguous():
        raise ValueError("kda glm_sm100 decode out_state must be contiguous fp32 [N, H, 128, 128]")
    o = torch.empty_like(v)
    H, bf = q.shape[1], dict(dtype=torch.bfloat16, device=q.device)
    rows = [torch.empty(N, H, HEAD_DIM, dtype=torch.bfloat16, device=q.device) for _ in range(4)] + [
        torch.empty(N, H, 2, dtype=torch.bfloat16, device=q.device), torch.empty(N, H, HEAD_DIM, dtype=torch.float32, device=q.device)]
    extension().decode(q, k, v, g, beta, state, out_state, o, rows)
    return o, out_state


def backward(q, k, v, g, beta, cu_seqlens, initial_state, do, dht=None):
    """Gradients ``(dq, dk, dv, dg, dbeta, dh0)`` of ``forward`` (with the final state returned) given the
    output gradient ``do [T, H, 128]`` and the final-state gradient ``dht [N, H, 128, 128]`` (None: zero).

    The forward intermediates are recomputed by the forward kernels themselves; every reduction is a
    fixed-order chain, no kernel uses atomics, so the gradients are run-to-run deterministic."""
    _check_count(cu_seqlens.shape[0] - 1, "sequences")
    _check(q, k, v, g, beta, cu_seqlens, initial_state)
    T, H, _ = q.shape
    nseq = cu_seqlens.shape[0] - 1
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    saves = {}
    forward(q, k, v, g, beta, cu_seqlens, initial_state, output_final_state=False, saves=saves)
    maxch, dev = saves["maxch"], q.device
    f32 = dict(dtype=torch.float32, device=dev)
    bf = dict(dtype=torch.bfloat16, device=dev)
    iw = [torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev), torch.empty(max(maxch, 1), H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev),
          torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev), torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev), torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev),
          torch.empty(T, H, CHUNK, dtype=torch.float32, device=dev), torch.empty(max(maxch, 1), H, HEAD_DIM, dtype=torch.float32, device=dev),
          torch.empty(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev), torch.empty(T, H, CHUNK, dtype=torch.float32, device=dev), torch.empty(T, H, dtype=torch.float32, device=dev)]
    out = [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(g), torch.empty_like(beta),
           torch.empty(nseq, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev)]
    fw = [saves[n] for n in ("W", "U", "QG", "KD", "MQ", "GL", "S", "d", "X", "KK", "Xu", "Kg")]
    do = do.to(torch.bfloat16).contiguous()
    if dht is not None:
        dht = dht.to(torch.float32).contiguous()
    if nseq > 0:
        extension().bwd(q, k, v, g, beta, cu_seqlens, saves["plan"], maxch, fw, do, dht, iw, out)
    return tuple(out)



# ------------------------------------------------------------------------------------------- engine records
class KdaState:
    """A pool of engine records ``[slots, H, rec_bytes]`` (csrc/record.cuh): chunks anchored at absolute
    position 0, the boundary state plus the open chunk's rows. A zeroed record is a fresh sequence at
    position 0 with a zero state; ``load_state`` seeds a slot with an initial state at position 0."""

    def __init__(self, slots: int, H: int, device="cuda"):
        self.H = H
        if extension().rec_bytes() != record_layout.REC_BYTES:
            raise RuntimeError(f"record_layout.py ({record_layout.REC_BYTES} B) disagrees with the built csrc/record.cuh "
                               f"({extension().rec_bytes()} B)")
        self.data = torch.zeros(slots, H, record_layout.REC_BYTES, dtype=torch.uint8, device=device)

    def load_state(self, slot: int, h0: torch.Tensor):
        """Position 0 with fp32 state ``h0 [H, 128, 128]`` (S = h0 as the state planes, no open rows)."""
        rec = self.data[slot]
        rec.zero_()
        sh, sl, sc = record_layout.pack_planes(h0.to(device=rec.device, dtype=torch.float32))
        L = record_layout.REC
        rec[:, L["SH"]:L["SH"] + record_layout.SH_BYTES].copy_(sh)
        rec[:, L["SL"]:L["SL"] + record_layout.SL_BYTES].copy_(sl)
        rec[:, L["SC"]:L["SC"] + record_layout.SC_BYTES].copy_(sc)

    def positions(self) -> torch.Tensor:
        return self.data[:, 0, -16:-12].contiguous().view(torch.int32).view(-1)


def _check_state(state, H: int, device):
    if not isinstance(state, KdaState):
        raise ValueError("kda glm_sm100: state must be a KdaState pool")
    t = state.data
    if (not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.dim() != 3 or t.size(1) != H
            or t.size(2) != record_layout.REC_BYTES or t.device != device):
        raise ValueError(f"kda glm_sm100: state must be a uint8 [slots, {H}, {record_layout.REC_BYTES}] KdaState pool "
                         f"on {device}")


def _slots(name, x, n, device, capacity: int, null_slot: int | None = None, *, padding=True):
    return _checks.slots("kda glm_sm100", name, x, n, device, capacity, null_slot, padding=padding)


def _check_out(out, N: int, H: int, device):
    if out is None:
        return torch.empty(N, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if (not isinstance(out, torch.Tensor) or out.shape != (N, H, HEAD_DIM) or out.dtype != torch.bfloat16
            or out.device != device or not out.is_contiguous()):
        raise ValueError(f"kda glm_sm100: out must be a contiguous bf16 [{N}, {H}, {HEAD_DIM}] tensor on {device}")
    return out


def _check_workspace(ws, N: int, H: int, device):
    if ws is None:
        return decode_workspace(N, H, device)
    want = [((N, H, HEAD_DIM), torch.bfloat16), ((N, H, HEAD_DIM), torch.bfloat16), ((N, H, CHUNK), torch.bfloat16),
            ((N, H, CHUNK), torch.bfloat16), ((N, H), torch.int32)]
    if not (isinstance(ws, (list, tuple)) and len(ws) == len(want) and all(
            isinstance(t, torch.Tensor) and t.shape == shape and t.dtype == dt and t.device == device
            and t.is_contiguous() for t, (shape, dt) in zip(ws, want))):
        raise ValueError(f"kda glm_sm100: workspace must be decode_workspace({N}, {H}, {device}): contiguous "
                         f"{[(list(sh), str(dt).split('.')[-1]) for sh, dt in want]}")
    return list(ws)


def prefill(q, k, v, g, beta, cu_seqlens, state: KdaState, slot_in, slot_out=None):
    """Chunked prefill against engine records: sequence n continues record ``slot_in[n]`` (negative: a
    fresh sequence at position 0) and leaves its record in ``slot_out[n]`` (default ``slot_in``; negative:
    discarded). Outputs ``o [T, H, 128]`` are bit for bit those of one ``forward`` over each whole
    sequence from position 0, wherever the calls split it (README "Relations")."""
    nseq = cu_seqlens.shape[0] - 1
    _check_count(nseq, "sequences")
    _check(q, k, v, g, beta, cu_seqlens, None)
    _check_state(state, q.shape[1], q.device)
    cap = state.data.shape[0]
    si = _slots("slot_in", slot_in, nseq, q.device, cap)
    so = si if slot_out is None else _slots("slot_out", slot_out, nseq, q.device, cap)
    o, _ = forward(q, k, v, g, beta, cu_seqlens, None, records=state.data, slot_in=si, slot_out=so)
    return o


def decode_workspace(N: int, H: int, device) -> list:
    """The row kernel's hand-off to the state pass for ``N`` lanes (reusable across steps; a CUDA graph
    keeps one)."""
    return [torch.empty(N, H, HEAD_DIM, dtype=torch.bfloat16, device=device),
            torch.empty(N, H, HEAD_DIM, dtype=torch.bfloat16, device=device),
            torch.empty(N, H, CHUNK, dtype=torch.bfloat16, device=device),
            torch.empty(N, H, CHUNK, dtype=torch.bfloat16, device=device),
            torch.empty(N, H, dtype=torch.int32, device=device)]


def decode_step(q, k, v, g, beta, state: KdaState, slots, *, null_slot: int = -1, out=None, workspace=None):
    """One token per sequence against engine records (q, k, v, g ``[N, H, 128]``, beta ``[N, H]``): row
    ``pos mod 64`` of each record's open chunk, evaluated incrementally, appended; the chunk closes into
    the boundary state at row 63. Bit for bit the prefill of the same tokens (README "Relations").

    Lanes whose slot is negative or ``null_slot`` (an engine's null block, e.g. vLLM's ``NULL_BLOCK_ID``)
    are padding: output +0, no record byte touched. No host synchronization and no data-dependent launch
    shape, so the call is CUDA-graph capturable (``out`` / ``workspace`` may be preallocated)."""
    N = q.shape[0]
    _check_count(N, "decode lanes")
    _check(q, k, v, g, beta, None, None)
    H = q.shape[1]
    _check_state(state, H, q.device)
    sl = _slots("slots", slots, N, q.device, state.data.shape[0], int(null_slot))
    o = _check_out(out, N, H, q.device)
    hand = _check_workspace(workspace, N, H, q.device)
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    extension().decode_rec(q, k, v, g, beta, state.data, sl, o, hand, int(null_slot))
    return o


def decode_step_fused(q, k, v, f, b, dt_bias, A_log, lower_bound, state: KdaState, slots, *, null_slot: int = -1,
                      out=None, workspace=None):
    """:func:`decode_step` with the forget-gate and beta stages evaluated inside the row kernel, reading the raw
    bf16 projections ``f [N, H*128]`` (``integration.forget_gate`` input) and ``b [N, H]``
    (``integration.sigmoid_beta`` input), and with ``q, k, v [N, H, 128]`` allowed to be token rows of one
    wider tensor (128-contiguous heads, one common token stride). Same operations, same order: bit for bit
    ``decode_step(q, k, v, forget_gate(f, ...), sigmoid_beta(b), ...)``, two launches fewer and no
    contiguous copies."""
    N, H = q.shape[0], q.shape[1]
    _check_count(N, "decode lanes")
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"kda glm_sm100 expects q, k, v [N, H, {HEAD_DIM}]")
    if f.shape != (N, H * HEAD_DIM) or b.shape != (N, H) or f.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("decode_step_fused expects raw bf16 projections f [N, H*128] and b [N, H]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("kda glm_sm100 requires bf16 q, k, v")
    if dt_bias.numel() != H * HEAD_DIM or A_log.numel() != H:
        raise ValueError(f"decode_step_fused expects dt_bias [{H * HEAD_DIM}] and A_log [{H}]")
    if not q.is_cuda or any(t.device != q.device for t in (k, v, f, b, dt_bias, A_log)):
        raise ValueError("kda glm_sm100 requires all inputs on one CUDA device")
    _check_state(state, H, q.device)
    sl = _slots("slots", slots, N, q.device, state.data.shape[0], int(null_slot))
    o = _check_out(out, N, H, q.device)
    hand = _check_workspace(workspace, N, H, q.device)
    extension().decode_rec_fused(q, k, v, f, b, dt_bias.float().contiguous(), A_log.float().contiguous(),
                                 float(lower_bound), state.data, sl, o, hand, int(null_slot))
    return o


def record_state(state: KdaState, slots) -> torch.Tensor:
    """The fp32 state ``[N, H, 128, 128]`` the records stand for: bit for bit ``forward``'s final state of
    the same tokens as one sequence."""
    if not isinstance(state, KdaState):
        raise ValueError("kda glm_sm100: state must be a KdaState pool")
    n = torch.as_tensor(slots).numel() if not isinstance(slots, torch.Tensor) else slots.numel()
    _check_count(n, "slots")
    dev = state.data.device
    if dev.type != "cuda":
        raise ValueError("kda glm_sm100: record_state requires a KdaState pool on a CUDA device")
    _check_state(state, state.H, dev)
    cap = state.data.shape[0]
    sl = _slots("slots", slots, n, dev, cap, padding=False)   # every slot is read: no padding lanes
    if n and isinstance(slots, torch.Tensor) and slots.is_cuda and not torch.cuda.is_current_stream_capturing():
        lo, hi = (int(x) for x in torch.aminmax(sl))        # off the decode path: this read synchronizes
        if lo < 0 or hi >= cap:
            raise ValueError(f"kda glm_sm100: slots out of range for a pool of {cap} slots (no padding lanes here)")
    out = torch.empty(sl.numel(), state.H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=state.data.device)
    if sl.numel():
        extension().record_state(state.data, sl, out)
    return out
