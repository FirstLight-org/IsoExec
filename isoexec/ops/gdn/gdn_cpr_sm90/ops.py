"""SM90 Gated DeltaNet core, chunk-parallel with recurrent decode: one arithmetic identity for decode,
(chunked) prefill and trainer forward.

Arithmetic version 1 is the chunk form of ``csrc/recipe.cuh``: chunks of 64 tokens aligned to absolute
sequence position, tensor-core contractions with a declared chain structure, an fp32 boundary state.
Prefill evaluates all rows of all chunks in parallel (with one sequential pass over chunks for the
boundary states); decode evaluates the one row a new token adds and, when that row completes a chunk,
replaces the boundary state in the same launch. Every per-token quantity is row-separable, so a token's
output and the state it leaves depend only on the sequence's tokens -- never on batch composition,
packing, split points, execution plan, or whether the tokens arrived by prefill or by decode.

Tensors (engine and trainer use the same kernels):
  q, k  [T, hk, 128] bf16, raw (l2 normalization is in-kernel)
  v     [T, hv, 128] bf16
  a, b  [T, hv]      bf16 (gate and beta logits; gating is in-kernel)
  A_log, dt_bias [hv] fp32 or bf16 (read natively), anything else is widened once
  state :class:`CprState` pool; slots are int32 rows, negative = NULL (reads nothing / writes nothing)
  out   [T, hv, 128] bf16
hv must be a multiple of hk; head counts need not be powers of two.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from isoexec.ops.gdn.gdn_cpr_sm90._extension import extension

D = 128
C = 64
NPOS = 8   # position copies per record (csrc/state.cuh)
#: Bytes of one (slot, value head) record (csrc/state.cuh ``Rec::BYTES``), known without building the extension
#: because vLLM sizes its mamba pages from it before any kernel is loaded; checked against the build at bind.
RECORD_BYTES = 137984
#: The state record format (csrc/state.cuh ``Rec::FORMAT``): 3 = the fp32 boundary state encoded exactly as its
#: bf16 rounding, low 16 bits and a tie bit (1 = fp32 + bf16 image, 2 = fp32 only).
RECORD_FORMAT = 3
SCALE = 1.0 / math.sqrt(D)

# Count launch submissions (including graph capture), not device-side graph replays.
# These are this core's witnesses; the shared engine bookkeeping counters cannot prove its use.
_CALLS = {"prefill": 0, "trainer_forward": 0, "decode": 0}


def census():
    return {"served": sum(_CALLS.values()), **_CALLS}


@lru_cache(None)
def record_layout() -> dict:
    """Byte offsets of the sections of one (slot, head) state record (``csrc/state.cuh``)."""
    return dict(extension().record_layout())


def encode_state(x: torch.Tensor):
    """fp32 ``x`` [..., n] (n a multiple of 8) -> (sh int16 = rn(x) bits, lo int16 = low 16 bits, tie uint8 [..., n/8]):
    the record-format-3 code (csrc/state.cuh ``s_encode``), exact for every finite or infinite value."""
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    b = x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    lo, hi = u & 0xFFFF, u >> 16
    t = ((lo == 0x8000) & (hi != b)).to(torch.int64)
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    tie = (t.reshape(*t.shape[:-1], -1, 8) << shifts).sum(-1).to(torch.uint8)
    as16 = lambda v: torch.where(v >= 0x8000, v - 0x10000, v).to(torch.int16)
    return as16(b), as16(lo), tie


def decode_state(b: torch.Tensor, lo: torch.Tensor, tie_bits: torch.Tensor) -> torch.Tensor:
    """The exact inverse of :func:`encode_state` (csrc/state.cuh ``s_decode``): int16 ``b``/``lo`` and 0/1 ``tie_bits``."""
    b = b.to(torch.int64) & 0xFFFF
    lo = lo.to(torch.int64) & 0xFFFF
    hi = torch.where(lo < 0x8000, b, torch.where(lo > 0x8000, b - 1, b - tie_bits)) & 0xFFFF
    u = (hi << 16) | lo
    return torch.where(u >= 0x80000000, u - 0x100000000, u).to(torch.int32).view(torch.float32)


@lru_cache(None)
def _s_index() -> torch.Tensor:
    """Storage index of S element (v, d) in the fp32 mma A-fragment image (csrc/state.cuh ``s_index``)."""
    v = torch.arange(D).view(D, 1)
    d = torch.arange(D).view(1, D)
    lane = 4 * (v % 8) + (d // 2) % 4
    reg = (v // 8) % 2 + 2 * ((d // 8) % 2)
    return ((((v // 16) * 8 + d // 16) * 32 + lane) * 8 + reg * 2 + d % 2).reshape(-1)


@lru_cache(None)
def _kb_index() -> torch.Tensor:
    """Storage offset of k^ element (i, d) in a Wide<64> image (csrc/tile.cuh): two 64-column halves of
    64 rows, chunk c of row i at chunk position c ^ (i & 7)."""
    i = torch.arange(C).view(C, 1)
    d = torch.arange(D).view(1, D)
    return ((d >> 6) * C * 64 + i * 64 + (((d & 63) >> 3) ^ (i & 7)) * 8 + (d & 7)).reshape(-1)


@lru_cache(None)
def _cols_index() -> torch.Tensor:
    """Storage offset of element (v, i) in a Cols<128> image (csrc/tile.cuh): 16-row slabs, each chunk-major."""
    v = torch.arange(D).view(D, 1)
    i = torch.arange(C).view(1, C)
    return (((((v >> 4) << 3) + (i >> 3)) * 16 + (v & 15)) * 8 + (i & 7)).reshape(-1)


@dataclass(frozen=True)
class CprState:
    """The recurrent state pool: one record per (slot, value head).

    A record is the fp32 boundary state ``S`` [128 v, 128 d] of the open chunk (its bf16 image
    ``sh`` = rn(S) is derived by every reader, not stored), the open chunk's rows (``kb`` k^ -- bf16 [64 i, 128 d]; ``vb`` v and ``db`` delta
    -- bf16 value-major [128 v, 64 i]; these three are stored as the shared-memory tile images decode
    reads, and their properties return logical copies; ``xb`` rows of (I+L)^-1 -- fp32 [64, 64]; ``pb`` the open 16-row
    block's rows of P -- fp32 [16, 64];
    ``gb``/``bb`` the cumulative log decay and beta -- fp32 [64]) and ``pos``, the number of tokens the
    slot has consumed. Only rows ``[0, pos % 64)`` of the buffers are meaningful. A slot is fresh when
    ``pos == 0`` and ``S == 0``; a NULL input slot stands for exactly that.
    """

    data: torch.Tensor   # uint8 [P, hv, record bytes]

    @classmethod
    def allocate(cls, rows: int, hv: int, device, *, zero: bool = True) -> CprState:
        nbytes = record_layout()["bytes"]
        make = torch.zeros if zero else torch.empty
        return cls(make(rows, hv, nbytes, device=device, dtype=torch.uint8))

    @property
    def rows(self) -> int:
        return self.data.size(0)

    @property
    def hv(self) -> int:
        return self.data.size(1)

    def _section(self, name: str, nbytes: int, dtype, shape):
        off = record_layout()[name]
        return self.data[:, :, off:off + nbytes].view(dtype).view(self.rows, self.hv, *shape)

    @property
    def S(self) -> torch.Tensor:
        """fp32 [P, hv, 128, 128] boundary state (value-major; a copy). Stored ENCODED (record format 3,
        csrc/state.cuh): ``sh`` = rn(S), the low 16 bits ``lo`` and a tie bit, in the A-fragment image order; this
        is the exact decode (``decode_state``). Write through :meth:`set_S`."""
        return decode_state(self._raw("sh", torch.int16), self._raw("lo", torch.int16), self._tie_bits())[
            :, :, _s_index().to(self.data.device)].view(self.rows, self.hv, D, D)

    def _raw(self, name: str, dtype) -> torch.Tensor:
        return self._section(name, D * D * 2, dtype, (D * D,))

    def _tie_bits(self) -> torch.Tensor:
        """[P, hv, 128 x 128] int64 tie bit of each image element (bit i % 8 of byte i // 8)."""
        tie = self._section("tie", D * D // 8, torch.uint8, (D * D // 8,)).to(torch.int64)
        shifts = torch.arange(8, device=self.data.device, dtype=torch.int64)
        return ((tie.unsqueeze(-1) >> shifts) & 1).reshape(self.rows, self.hv, D * D)

    def set_S(self, idx, value: torch.Tensor) -> None:
        """Store fp32 [n, hv, 128, 128] ``value`` as the boundary state of rows ``idx`` (encoded, exactly)."""
        inv = torch.argsort(_s_index()).to(self.data.device)
        img = value.to(torch.float32).reshape(*value.shape[:-2], D * D)[..., inv]
        b, lo, tie = encode_state(img)
        self._raw("sh", torch.int16)[idx] = b
        self._raw("lo", torch.int16)[idx] = lo
        self._section("tie", D * D // 8, torch.uint8, (D * D // 8,))[idx] = tie

    @property
    def sh(self) -> torch.Tensor:
        """bf16 [P, hv, 128, 128]: S^ = rn(S), the operand every decode step reads (a copy; stored as the mma
        A-fragment image)."""
        raw = self._raw("sh", torch.bfloat16)
        return raw[:, :, _s_index().to(self.data.device)].view(self.rows, self.hv, D, D)

    def _image(self, name: str, index: torch.Tensor, shape) -> torch.Tensor:
        """A bf16 buffer stored as a shared-memory tile image, gathered into its logical layout (a copy)."""
        raw = self._section(name, C * D * 2, torch.bfloat16, (C * D,))
        return raw[:, :, index.to(raw.device)].view(self.rows, self.hv, *shape)

    @property
    def kb(self) -> torch.Tensor:
        """bf16 [P, hv, 64 i, 128 d] (a copy; stored as a Wide<64> tile image)."""
        return self._image("kb", _kb_index(), (C, D))

    @property
    def vb(self) -> torch.Tensor:
        """bf16 [P, hv, 128 v, 64 i] (a copy; stored as a Cols<128> image: 16-row slabs, chunk-major)."""
        return self._image("vb", _cols_index(), (D, C))

    @property
    def db(self) -> torch.Tensor:
        """bf16 [P, hv, 128 v, 64 i] (a copy; stored as a Cols<128> image: 16-row slabs, chunk-major)."""
        return self._image("db", _cols_index(), (D, C))

    @property
    def xb(self) -> torch.Tensor:
        return self._section("xb", C * C * 4, torch.float32, (C, C))

    @property
    def pb(self) -> torch.Tensor:
        """fp32 [P, hv, 16, 64]: rows of P for the open 16-row block (row 16B + r at r)."""
        return self._section("pb", 16 * C * 4, torch.float32, (16, C))

    @property
    def gb(self) -> torch.Tensor:
        return self._section("gb", C * 4, torch.float32, (C,))

    @property
    def bb(self) -> torch.Tensor:
        return self._section("bb", C * 4, torch.float32, (C,))

    @property
    def pos(self) -> torch.Tensor:
        """int32 [P, hv, NPOS] tokens consumed: NPOS equal copies (one per decode CTA of a head, so the
        CTAs never order against each other). Assigning through this view sets every copy."""
        return self._section("pos", 4 * NPOS, torch.int32, (NPOS,))

    def reset(self, slots: torch.Tensor) -> None:
        """Make ``slots`` fresh (zero boundary state, position 0). Stale buffer rows need no clearing."""
        idx = slots.long()
        for name in ("sh", "lo"):
            self._raw(name, torch.int16)[idx] = 0
        self._section("tie", D * D // 8, torch.uint8, (D * D // 8,))[idx] = 0
        self.pos[idx] = 0

    def load_initial(self, slots: torch.Tensor, S0: torch.Tensor) -> None:
        """Start ``slots`` at position 0 from the fp32 [n, hv, 128, 128] initial states ``S0``."""
        idx = slots.long()
        self.set_S(idx, S0)
        self.pos[idx] = 0


def _check_inputs(q, k, v, a, b):
    for name, t in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"gdn_cpr_sm90: {name} must be a CUDA tensor")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"gdn_cpr_sm90: {name} must be bfloat16, got {t.dtype}")
    if q.dim() != 3 or k.shape != q.shape or q.size(2) != D:
        raise ValueError(f"gdn_cpr_sm90: q/k must be [T, hk, {D}], got {tuple(q.shape)} / {tuple(k.shape)}")
    if v.dim() != 3 or v.size(0) != q.size(0) or v.size(2) != D:
        raise ValueError(f"gdn_cpr_sm90: v must be [T, hv, {D}], got {tuple(v.shape)}")
    hk, hv = q.size(1), v.size(1)
    if hk <= 0 or hv % hk:
        raise ValueError(f"gdn_cpr_sm90: value heads ({hv}) must be a multiple of key heads ({hk})")
    if a.shape != (q.size(0), hv) or b.shape != a.shape:
        raise ValueError(f"gdn_cpr_sm90: a/b must be [T, hv], got {tuple(a.shape)} / {tuple(b.shape)}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        # rows are streamed with 16-byte asynchronous copies
        if t.stride(2) != 1 or t.data_ptr() % 16 or t.stride(0) % 8 or t.stride(1) % 8:
            raise ValueError(f"gdn_cpr_sm90: {name} rows must be contiguous and 16-byte aligned")


def _params(A_log, dt_bias, hv, device):
    """fp32 and bf16 parameters are both read natively; anything else is widened once, here."""
    keep = (torch.float32, torch.bfloat16)
    A, dt = A_log.detach(), dt_bias.detach()
    A = A.contiguous() if (A.dtype in keep and A.device == device) else A.to(device=device, dtype=torch.float32).contiguous()
    dt = dt.contiguous() if (dt.dtype in keep and dt.device == device) else dt.to(device=device, dtype=torch.float32).contiguous()
    if A.dtype != dt.dtype:
        A, dt = A.float().contiguous(), dt.float().contiguous()
    if A.shape != (hv,) or dt.shape != (hv,):
        raise ValueError(f"gdn_cpr_sm90: A_log/dt_bias must be [{hv}]")
    return A, dt


def _check_out(out, n, hv, dev):
    if out is None:
        return torch.empty(n, hv, D, device=dev, dtype=torch.bfloat16)
    if (out.shape != (n, hv, D) or out.dtype != torch.bfloat16 or out.stride(2) != 1 or out.data_ptr() % 16
            or out.stride(0) % 8 or out.stride(1) % 8):
        raise ValueError("gdn_cpr_sm90: out must be bf16 [n, hv, 128] with contiguous, 16-byte aligned rows")
    return out


def _check_state(state, hv, dev):
    if not isinstance(state, CprState):
        raise ValueError("gdn_cpr_sm90: state must be a CprState pool")
    t = state.data
    if (t.dtype != torch.uint8 or t.dim() != 3 or t.size(1) != hv or t.size(2) != record_layout()["bytes"]
            or t.device != dev or not t.is_contiguous()):
        raise ValueError(f"gdn_cpr_sm90: state must be a contiguous uint8 [P, {hv}, {record_layout()['bytes']}] "
                         f"CprState pool on {dev}")


def _check_slots(name, slots, n, dev):
    if not (isinstance(slots, torch.Tensor) and slots.dtype == torch.int32 and slots.numel() == n
            and slots.device == dev):
        raise ValueError(f"gdn_cpr_sm90: {name} must be int32 [{n}] on {dev}")
    return slots.contiguous()


def _check_cu(cu_seqlens, T):
    if not (isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.is_cuda and cu_seqlens.dtype == torch.int32
            and cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 1):
        raise ValueError("gdn_cpr_sm90: cu_seqlens must be an int32 CUDA tensor [N + 1]")
    n = cu_seqlens.numel() - 1
    if n == 0 and T > 0:
        raise ValueError(f"gdn_cpr_sm90: {T} tokens but no sequences (cu_seqlens has one entry)")
    return n


def _max_chunks(cu_seqlens_cpu, n, T, stateful):
    """The workspace bound (chunks). From the host copy of cu_seqlens, if given, the pack's own count
    (stateful: each sequence may start anywhere in its open chunk); else -1, the bound for any lengths
    (T / 64 + 2N, prefill.cu). A host copy is validated: it must describe the same pack."""
    if cu_seqlens_cpu is None:
        return -1
    if not (isinstance(cu_seqlens_cpu, torch.Tensor) and cu_seqlens_cpu.device.type == "cpu"
            and cu_seqlens_cpu.numel() == n + 1):
        raise ValueError(f"gdn_cpr_sm90: cu_seqlens_cpu must be a CPU tensor [{n + 1}]")
    cu = cu_seqlens_cpu.numpy().astype(np.int64)   # host arithmetic: a few numpy ops, not torch dispatches
    lens = cu[1:] - cu[:-1]
    if cu[0] != 0 or cu[-1] != T or (lens < 0).any():
        raise ValueError(f"gdn_cpr_sm90: cu_seqlens_cpu must rise from 0 to T = {T}")
    reach = lens + (C - 1 if stateful else 0)   # the open chunk's rows before the call, at most C - 1
    return int(np.where(lens > 0, (reach + C - 1) // C, 0).sum())


def _sweep_rows(sweep_rows):
    if sweep_rows is None:
        return 0   # the kernel chooses (csrc/sweep.cu)
    if sweep_rows not in (16, 32):
        raise ValueError(f"gdn_cpr_sm90: sweep_rows must be 16, 32 or None, got {sweep_rows!r}")
    return int(sweep_rows)


# Test hook: bit 0 is a compatibility no-op after separating the sweep/output launches;
# bit 1 oversubscribes the output grid, and bit 2 chains all sequences in one sweep group.
# Results are bitwise identical under every schedule.
_SCHEDULE = 0


@contextlib.contextmanager
def _sweep_schedule(*, on_demand=False, oversubscribe=False, one_group=False):
    global _SCHEDULE
    before, _SCHEDULE = _SCHEDULE, int(on_demand) | 2 * int(oversubscribe) | 4 * int(one_group)
    try:
        yield
    finally:
        _SCHEDULE = before


# Test hook: force "two_pass", "fused", or "prepared"; None uses choose_prefill_mode.
_MODE = None


@contextlib.contextmanager
def _prefill_mode(mode):
    global _MODE
    if mode not in (None, "two_pass", "fused", "prepared"):
        raise ValueError(f"gdn_cpr_sm90: prefill mode {mode!r}")
    before, _MODE = _MODE, mode
    try:
        yield
    finally:
        _MODE = before


def choose_prefill_mode(n, T, hv, cu_seqlens_cpu=None):
    """Choose an execution plan without inspecting device data: 0 two-pass, 1 fully fused, 2 prepared.

    On the measured H100 grid, parallel WY followed by the prepared state/output walk wins for
    balanced, multi-chunk packs with at least 64 sequence-head items. Single-chunk packs and few long
    sequences keep the two-pass path. Without host lengths, or outside the measured length range,
    keep that general path as well. Fully fused remains available to the private test/tuning hook;
    it has no broad crossover advantage over prepared on the measured grid.
    """
    if _MODE is not None:
        return {"two_pass": 0, "fused": 1, "prepared": 2}[_MODE]
    if n < 16 or n * hv < 64 or T <= n * C or T > n * 1280:
        return 0
    if not (isinstance(cu_seqlens_cpu, torch.Tensor) and cu_seqlens_cpu.device.type == "cpu"
            and cu_seqlens_cpu.ndim == 1 and cu_seqlens_cpu.numel() == n + 1):
        return 0
    lengths = np.diff(cu_seqlens_cpu.numpy().astype(np.int64))
    # A long outlier leaves a sequence-owned CTA running after its peers finish. Do not infer a
    # balanced pack from its average, and avoid charging two launches to mostly single-chunk items.
    if lengths.min() <= C or lengths.max() * n > 2 * T:
        return 0
    return 2


def prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, state=None, slot_in=None, slot_out=None, out=None,
            max_seqlen=None, sweep_rows=None, cu_seqlens_cpu=None):
    """Packed varlen prefill. Returns ``out`` [T, hv, 128] bf16 and updates ``state[slot_out]``.

    Sequence ``n`` (tokens ``cu_seqlens[n] .. cu_seqlens[n+1]``) continues from ``state[slot_in[n]]``
    (NULL = fresh) and leaves its state in ``state[slot_out[n]]`` (defaults to ``slot_in``; NULL =
    discard). Without ``state`` every sequence starts fresh and nothing is written back (trainer). A pack
    with no tokens still moves its sequences' slots (``slot_out`` holds the state afterwards, as for any
    other pack). A slot written by one sequence must
    not be read by another sequence of the same call.

    ``cu_seqlens_cpu`` (optional, the host copy of ``cu_seqlens``) sizes the workspace to the pack's
    chunks; without it the workspace is sized for any lengths. ``max_seqlen`` is accepted for API parity
    with existing callers and is not needed. ``sweep_rows`` (16 or 32 value rows per sequential CTA; None chooses)
    is an execution choice; results are bitwise identical.
    """
    del max_seqlen
    rows = _sweep_rows(sweep_rows)
    _check_inputs(q, k, v, a, b)
    T, hv = q.size(0), v.size(1)
    dev = q.device
    n = _check_cu(cu_seqlens, T)
    A, dt = _params(A_log, dt_bias, hv, dev)
    out = _check_out(out, T, hv, dev)
    if n <= 0 or (T == 0 and state is None):
        return out
    if state is None:
        if slot_in is not None or slot_out is not None:
            raise ValueError("gdn_cpr_sm90: slots given without a state pool")
        extension().prefill(q, k, v, a, b, A, dt, cu_seqlens.contiguous(), None, None, None, out, SCALE, rows,
                            _max_chunks(cu_seqlens_cpu, n, T, False), [], _SCHEDULE,
                            choose_prefill_mode(n, T, hv, cu_seqlens_cpu))
        _CALLS["prefill"] += 1
        return out
    _check_state(state, hv, dev)
    slot_in = _check_slots("slot_in", slot_in, n, dev)
    slot_out = slot_in if slot_out is None else _check_slots("slot_out", slot_out, n, dev)
    extension().prefill(q, k, v, a, b, A, dt, cu_seqlens.contiguous(), state.data, slot_in, slot_out, out, SCALE,
                        rows, _max_chunks(cu_seqlens_cpu, n, T, True), [], _SCHEDULE,
                        choose_prefill_mode(n, T, hv, cu_seqlens_cpu))
    _CALLS["prefill"] += 1
    return out


def trainer_forward(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, save_states=False, cu_seqlens_cpu=None):
    """Stateless prefill with backward intermediates in FLA's layouts; output bits match ``prefill``.

    Returns ``(out, saved)``; ``saved`` holds ``qn``, ``kn`` [T, hk, 128] bf16 (l2-normalized, ``qn``
    without the attention scale), ``q_rstd``, ``k_rstd`` [T, hk] fp32 (inverse norms), ``v`` [T, hv, 128]
    (the values, contiguous), ``A`` [T, hv, 64] bf16 ((I + L)^-1 row of each token in its chunk), ``w``
    [T, hv, 128] bf16, ``h`` [NT, hv, 128, 128] bf16 (``S^`` at each chunk start, value-major = FLA's
    ``state_v_first``; the sequences' chunks back to back, NT allocated as ``T // 64 + N``) and ``v_new``
    [T, hv, 128] bf16 (the deltas). ``u`` is not saved: the backward does not read it.
    The default ``save_states=False`` leaves out ``w``, ``h`` and ``v_new`` (empty tensors); backward recomputes
    them from what is kept; that keeps about 40% of the full saves' memory at hk = 4, hv = 8.
    """
    _check_inputs(q, k, v, a, b)
    T, hk, hv = q.size(0), q.size(1), v.size(1)
    dev = q.device
    n = _check_cu(cu_seqlens, T)
    A, dt = _params(A_log, dt_bias, hv, dev)
    out = _check_out(None, T, hv, dev)
    saved = dict(qn=torch.empty(T, hk, 128, device=dev, dtype=torch.bfloat16), kn=torch.empty(T, hk, 128, device=dev, dtype=torch.bfloat16),
                 q_rstd=torch.empty(T, hk, device=dev, dtype=torch.float32),
                 k_rstd=torch.empty(T, hk, device=dev, dtype=torch.float32),
                 v=torch.empty(T, hv, 128, device=dev, dtype=torch.bfloat16), A=torch.empty(T, hv, 64, device=dev, dtype=torch.bfloat16),
                 **{name: torch.empty(*(shape if save_states else (0,)), device=dev, dtype=torch.bfloat16) for name, shape in (
                     ("w", (T, hv, 128)), ("h", (T // 64 + max(n, 0), hv, 128, 128)),
                     ("v_new", (T, hv, 128)))})
    if n > 0 and T > 0:
        extension().prefill(q, k, v, a, b, A, dt, cu_seqlens.contiguous(), None, None, None, out, SCALE, 0,
                            _max_chunks(cu_seqlens_cpu, n, T, False), list(saved.values()), _SCHEDULE,
                            choose_prefill_mode(n, T, hv, cu_seqlens_cpu))
        _CALLS["trainer_forward"] += 1
    return out, saved


_DECODE_SPLITS = (1, 2, 4, 8)


def _heuristic_split(num_seqs: int, hv: int) -> int:
    heads = num_seqs * hv
    if heads <= 64:
        return 4
    if heads <= 128:
        return 2
    return 1


#: Measured best split per (device, hv, hk): sorted [(max sequences, split)] (``tune_decode_split``).
_SPLIT_TABLE: dict = {}
_TUNE_NS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512)


def choose_decode_split(num_seqs: int, hv: int, hk: int | None = None) -> int:
    """CTAs per head for a decode step (execution only; every split is bitwise identical).

    The split measured fastest on this device for the step's sequence count (``tune_decode_split``, run when a
    pool binds), else a heuristic: few heads in flight want a head's rows spread over several SMs; from ~128
    heads the grid fills the machine and repeating the key-side work per CTA adds overhead."""
    if hk is not None and torch.cuda.is_available():
        table = _SPLIT_TABLE.get((torch.cuda.current_device(), hv, hk))
        if table:
            for n_max, split in table:
                if num_seqs <= n_max:
                    return split
            return table[-1][1]
    return _heuristic_split(num_seqs, hv)


def tune_decode_split(hv: int, hk: int, device=None, *, reps: int = 8) -> list:
    """Measure every decode split at a grid of sequence counts on synthetic records whose open chunks sit at
    every row (a boundary resync included), and keep the fastest per count. Execution only: every split gives
    the same bits, so the choice may differ between devices or ranks. Runs once per (device, hv, hk), outside
    graph capture (pools call it when they bind). ``ISOEXEC_GDN_CPR_DECODE_TUNE=0`` keeps the heuristic."""
    import os

    if os.environ.get("ISOEXEC_GDN_CPR_DECODE_TUNE", "1").lower() in ("0", "false", "no"):
        return []
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    key = (dev.index, hv, hk)
    if key in _SPLIT_TABLE:
        return _SPLIT_TABLE[key]
    if torch.cuda.is_current_stream_capturing():
        return []
    g = torch.Generator(device=dev).manual_seed(0)
    n_top = _TUNE_NS[-1]
    r = lambda *shape: (torch.randn(*shape, device=dev, generator=g, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    q, k, v = r(n_top, hk, D), r(n_top, hk, D), r(n_top, hv, D)
    a, b = r(n_top, hv), r(n_top, hv)
    A_log = torch.zeros(hv, device=dev, dtype=torch.float32)
    dt_bias = torch.zeros(hv, device=dev, dtype=torch.float32)
    state = CprState.allocate(n_top + 1, hv, dev)
    state.pos[1:] = (torch.arange(n_top, device=dev, dtype=torch.int32) % C).view(n_top, 1, 1)
    out = torch.empty(n_top, hv, D, device=dev, dtype=torch.bfloat16)
    table = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for n in _TUNE_NS:
        slots = torch.arange(1, n + 1, device=dev, dtype=torch.int32)
        best = None
        for split in _DECODE_SPLITS:
            args = (q[:n], k[:n], v[:n], a[:n], b[:n], A_log, dt_bias, state, slots)
            decode(*args, out=out[:n], split=split)   # warm (and every slot advances: open rows vary)
            start.record()
            for _ in range(reps):
                decode(*args, out=out[:n], split=split)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            if best is None or ms < best[0]:
                best = (ms, split)
        table.append((n, best[1]))
    del q, k, v, a, b, state, out
    _SPLIT_TABLE[key] = table
    return table


def decode(q, k, v, a, b, A_log, dt_bias, state, slot_in, *, slot_out=None, out=None, split=None):
    """One token per sequence: q/k [N, hk, 128], v [N, hv, 128], a/b [N, hv]. Returns out [N, hv, 128] bf16.

    A single launch with no workspace and no host interaction (capturable in a CUDA graph, NULL lanes
    included). When a sequence's token completes its chunk, the same launch replaces its boundary state.
    ``split`` (CTAs per head: 1, 2, 4 or 8) is an execution choice; results are bitwise identical.
    """
    _check_inputs(q, k, v, a, b)
    N, hv = q.size(0), v.size(1)
    dev = q.device
    A, dt = _params(A_log, dt_bias, hv, dev)
    _check_state(state, hv, dev)
    slot_in = _check_slots("slot_in", slot_in, N, dev)
    slot_out = slot_in if slot_out is None else _check_slots("slot_out", slot_out, N, dev)
    out = _check_out(out, N, hv, dev)
    if N == 0:
        return out
    split = choose_decode_split(N, hv, q.size(1)) if split is None else split
    if split not in _DECODE_SPLITS:
        raise ValueError(f"gdn_cpr_sm90: unsupported decode split {split}")
    extension().decode(q, k, v, a, b, A, dt, state.data, slot_in, slot_out, out, SCALE, split)
    _CALLS["decode"] += 1
    return out
