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


def _slots(x, n, device):
    return torch.as_tensor(x, dtype=torch.int32, device=device).reshape(n).contiguous()


def prefill(q, k, v, g, beta, cu_seqlens, state: KdaState, slot_in, slot_out=None):
    """Chunked prefill against engine records: sequence n continues record ``slot_in[n]`` (negative: a
    fresh sequence at position 0) and leaves its record in ``slot_out[n]`` (default ``slot_in``; negative:
    discarded). Outputs ``o [T, H, 128]`` are bit for bit those of one ``forward`` over each whole
    sequence from position 0, wherever the calls split it (README "Relations")."""
    nseq = cu_seqlens.shape[0] - 1
    _check_count(nseq, "sequences")
    si = _slots(slot_in, nseq, q.device)
    so = si if slot_out is None else _slots(slot_out, nseq, q.device)
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
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    o = torch.empty_like(v) if out is None else out
    hand = decode_workspace(N, q.shape[1], q.device) if workspace is None else workspace
    extension().decode_rec(q, k, v, g, beta, state.data, _slots(slots, N, q.device), o, hand,
                           int(null_slot))
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
    o = torch.empty((N, H, HEAD_DIM), dtype=torch.bfloat16, device=q.device) if out is None else out
    hand = decode_workspace(N, H, q.device) if workspace is None else workspace
    extension().decode_rec_fused(q, k, v, f, b, dt_bias.float().contiguous(), A_log.float().contiguous(),
                                 float(lower_bound), state.data, _slots(slots, N, q.device), o, hand,
                                 int(null_slot))
    return o


def record_state(state: KdaState, slots) -> torch.Tensor:
    """The fp32 state ``[N, H, 128, 128]`` the records stand for: bit for bit ``forward``'s final state of
    the same tokens as one sequence."""
    sl = torch.as_tensor(slots, dtype=torch.int32, device=state.data.device).reshape(-1).contiguous()
    _check_count(sl.numel(), "slots")
    out = torch.empty(sl.numel(), state.H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=state.data.device)
    if sl.numel():
        extension().record_state(state.data, sl, out)
    return out
