"""SM100 Gated DeltaNet core, chunk form with recurrent decode: one arithmetic identity for decode, (chunked) prefill and trainer forward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from isoexec.ops.gdn.gdn_cpr_sm100._extension import extension

D = 128
C = 64
NPOS = 8   # position copies per record (csrc/state.cuh)
SCALE = 1.0 / math.sqrt(D)

#: PHASE-1 EXECUTION PLANS, a bitmask, EXECUTION ONLY. Bit 0 is the ten-tile KK / KQ deal balanced 3/3/2/2
#: across sub-partitions; bit 1 is the compile-time phase-1 widths (the X earlier-block rounds templated on
#: their block, the T16 tree streamed in groups of four, compile-time trip counts on the copy and xu / xw
#: loops). Bit 2 is the explicit `@p` predication of the phase-2 chains, and it is a DECODE bit: it is
#: worth 3.8-6.0% on a decode step and it costs the prefill kernel 1,256 bytes of extra local frame, so
#: `slab_step`'s `PRED` parameter is true only where the decode kernel instantiates it.
#: `PLAN_DEFAULT` (-1) lets the kernel take the DEPLOYMENT-FIXED RULE: **a prefill runs plan 3** (both
#: phase-1 choices, 1.03x to 1.07x on every prefill row) and **a decode step plan 4** (the predication,
#: NEITHER phase-1 choice). The two phase-1 choices make a decode step's own phases cheaper and still
#: make the STEP slower -- 11.63 us at plan 4 against 12.27 at plan 7 at B = 1, and 78.3 against 79.9 at
#: B = 32, while WINNING 2.3% at B = 256 -- and the cause is not settled (csrc/core.cu and README.md's
#: "Remaining gaps" 0e; the if-conversion explanation this file used to give is withdrawn). A call is a
#: prefill or a decode at its call site, so THIS rule is not request-varying; every plan value produces
#: the SAME BITS and tests/test_identity_gpu.py is the gate, which is what makes selecting on speed
#: legitimate. Nothing in production names a plan.
#: THE PAIRED BF16 NARROWING, which is worth more than both of these together, is not a plan: it wins on
#: both call kinds, and its equality with the scalar form is proved exhaustively over all 2^32 float bit
#: patterns rather than gated by an A/B (csrc/recipe.cuh::rn2).
PLAN_DEFAULT = -1
#: `prepared_round4` is BIT 4 on top of `prepared`: the same two-launch structure with ROUND 4's launch A
#: -- the one-launch kernel's 194,816-byte shared-memory layout (one CTA per SM) and the three-deep
#: dependent global prologue. It is this pass's BEFORE column, and it exists in the shipped build for the
#: same reason `round2` and the forced plans do: a CUPTI number is only comparable to one taken in the
#: same rotation cycle, so the previous shipped kernel has to run beside the new one in one process.
#: `prepared_prekernel` is BIT 5: the (chunk, KEY head) pre-kernel, built and gated and NOT shipped --
#: it measured a net loss (`_use_prekernel`). The shipped `prepared` is the two-CTA launch A without it.
#: `prepared_nofuse` is BIT 6: launch B with its delta and output-seed chains as two k-loops rather than
#: one (`slab.cuh`, `FUSE`), which is the fused form's before column.
#: `rec` is BIT 8 and it is the SHIPPED decode step: the RECURRENT form of `csrc/decode.cu`, which
#: evaluates row `j = pos mod 64` of the chunk arithmetic incrementally out of the record's already-computed
#: rows [0, j) instead of asking the 64-row chunk kernel for one row and predicating the rest away. It is
#: the same recipe (`recipe.cuh`) and `tests/test_identity_gpu.py` gates it bit-identical to the chunk form
#: on a seventy-step stream, on every open-chunk offset, and against a prefill over the same tokens.
#: `rec_r1` / `rec_r2` / `rec_r4` / `rec_r8` force the VALUE-ROW SPLIT (CTAs per head) that
#: `choose_decode_split` otherwise picks from the call's shape; `rec` alone takes the rule. Every decode
#: plan WITHOUT bit 8 -- 0, 1, 3, 4, 7 -- runs the CHUNK-FORM decode of `csrc/core.cu`, which is the
#: previous shipped kernel and stays in the build as this pass's before column and as the bit-identity
#: reference. A prefill ignores bit 8.
#: `prepared_split2` / `prepared_split4` are BITS 7-8 on top of `prepared`: launch B's (sequence, value
#: head) split across 2 or 4 CTAs along the VALUE COLUMNS, which is the fifth review's top item
#: (`csrc/prepared.cu`, `SPLIT`). Code 0 is the rule, 1 is two-way, 2 is four-way.
PLANS = {"round2": 0, "deal": 1, "deal+widths": 3, "pred": 4, "deal+widths+pred": 7,          "prepared": 11, "prepared_round4": 27, "prepared_prekernel": 43, "prepared_nofuse": 75,
         "prepared_split2": 139, "prepared_split4": 267,
         "rec": 4096, "rec_r1": 12288, "rec_r2": 20480, "rec_r4": 28672, "rec_r8": 36864}

#: Bit 12 of a decode plan: the recurrent step. Bits 13-15 force its value-row split (0 = the rule).
REC = 4096
_REC_SPLIT_SHIFT = 13
#: Bits 7-8 of a prepared prefill plan: launch B's value-column split across CTAs (0 = the rule).
_PREP_SPLIT_SHIFT = 7

#: The PREPARED TWO-LAUNCH prefill is plan BIT 3, and it is a launch STRUCTURE rather than a phase-1
#: choice: launch A evaluates `wy_chunk` for every (chunk, value head) in parallel across the whole
#: machine, launch B walks each sequence's chunks with the fp32 state in registers and runs `slab_step`.
#: Bits 0 and 1 still name launch A's phase-1 plan, so the built value is 11 = prepared + deal + widths.
#: It exists because a one-launch kernel's time is (chunks per sequence) x (full-chunk latency) with only
#: `nseq * hv` CTAs busy, which is the floor the single-sequence rows sit on; the prepared mode's phase 1
#: instead fills the machine. It is selected by `_use_prepared`, a rule on the call's SHAPE -- tokens,
#: sequences, value heads -- and never on its contents. Note that this is NOT the "not request-varying"
#: argument `PLANS` rests on: sequence count and token count ARE batch composition, so the same prompt
#: alone and beside seven others takes a different launch structure. What makes that acceptable is only
#: that the two structures are BIT-IDENTICAL, which puts the rule inside claim (d) rather than exempt
#: from it; `tests/test_identity_gpu.py` gates it on every relation and that gate is the whole licence.
PREPARED = 11

#: How many bytes of workspace the prepared mode may ask for. The hand-off is ~72 KB per (chunk, value
#: head) and the whole call's chunks are live at once, so `T = 32768` at 16/32 heads is 1.2 GB. Past this
#: the one-launch kernel runs instead; nothing is windowed, because a window would serialize launch A
#: against launch B and the rows this mode exists for are the ones with the fewest sequences.
PREP_BUDGET_BYTES = 4 << 30


@lru_cache(None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@lru_cache(None)
def _decode_blocks_per_sm() -> int:
    """`cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the recurrent decode kernel, from the hardware."""
    return int(record_layout()["decode_rec_blocks_per_sm"])


def _use_prekernel(hk: int, hv: int) -> bool:
    """Check whether the key-head pre-kernel can run."""
    return hv > hk


def _prep_sizes(T: int, nseq: int, hk: int, hv: int, prek: bool | None = None):
    """(chunk-slot bound, hand-off stride, hand-off bytes, key hand-off bytes, carry bytes)."""
    lay = record_layout()
    maxch = T // C + 2 * max(nseq, 1)
    if prek is None:
        prek = _use_prekernel(hk, hv)
    stride = lay["prep_hand_short"] if prek else lay["prep_hand"]
    handk = maxch * hk * lay["prep_handk"] if prek else 0
    return maxch, stride, maxch * hv * stride, handk, max(nseq, 1) * hv * lay["prep_carry"]


#: The two-structure cost model's per-chunk constants, measured on a leased B200 over 65 (geometry, T,
#: nseq) points in one interleaved rotation each (`_use_prepared`). `t1` is head-count INDEPENDENT, which
#: is the physical invariant this package uses as a contamination test (README.md, "Caveats").
_T1_US = 13.44
_TA2_US = 15.47
_TB_US = 4.05
_TFIX_US = 8.28


def _use_prepared(T: int, nseq: int, hk: int, hv: int, save: bool, device, maxlen: int | None = None) -> bool:
    """CHUNKS PER SEQUENCE, and after round 5 that is the WHOLE rule: the CTA bound is gone."""
    if save or T <= 0 or nseq <= 0:
        return False
    #: THE RULE READS THE LONGEST SEQUENCE WHEN THE HOST KNOWS IT, AND THE MEAN ONLY WHEN IT DOES NOT.
    #: `T / nseq` is the mean, and the mean is wrong for exactly the shape this mode exists for: a pack
    #: like `[8192] + 100 x [64]` has a mean of 144 tokens, so the old rule DECLINED it and the
    #: 8,192-token prompt walked 128 serial chunks on one CTA while the other hundred finished in one
    #: wave. `prefill` already accepted `cu_seqlens_cpu` and deleted it; taking the maximum from it costs
    #: no device synchronization, because it is a HOST tensor. Without it the mean stands, which is the
    #: behaviour every previous pass measured. The fifth Fable review found this (README.md, gap 0g).
    nch = -(-(maxlen if maxlen else (T // nseq)) // C)
    if nch < 4:
        return False                       # under four chunks a sequence the split is a coin flip
    _, _, hand, handk, carry = _prep_sizes(T, nseq, hk, hv)
    if hand + handk + carry > PREP_BUDGET_BYTES:
        return False
    #: The model above, evaluated. It is redundant with `T >= 4 C nseq` on every shape measured -- the
    #: chunk-count guard is the binding one -- and it is here because the two-structure model is what the
    #: rule MEANS, so a machine with a different SM count gets the rule and not the constant.
    sms = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    waves = -(-nseq * hv // sms)
    one = waves * nch * _T1_US
    prep = -(-nseq * nch * hv // (2 * sms)) * _TA2_US + waves * nch * _TB_US + _TFIX_US
    return prep < one


def _prep_split(nseq: int, hv: int, device) -> int:
    """CTAs per (sequence, value head) in LAUNCH B of the prepared mode: 1, 2 or 4 (``csrc/prepared.cu``)."""
    items = max(nseq, 1) * max(hv, 1)
    sms = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    return 2 if 2 * items <= sms else 1


def _plan(plan):
    """`None` -> the deployment-fixed rule; otherwise a value the kernel actually instantiates."""
    if plan is None:
        return PLAN_DEFAULT
    if plan not in PLANS.values():
        raise ValueError(f"gdn_cpr_sm100: unknown execution plan {plan}; built plans are "
                         f"{sorted(PLANS.values())} ({PLANS})")
    return plan

#: Launch submissions (including graph capture), not device-side graph replays: this core's witnesses.
_CALLS = {"prefill": 0, "trainer_forward": 0, "decode": 0}


def census():
    return {"served": sum(_CALLS.values()), **_CALLS}


@lru_cache(None)
def record_layout() -> dict:
    """Byte offsets of the sections of one (slot, head) state record (``csrc/state.cuh``)."""
    return dict(extension().record_layout())


@lru_cache(None)
def _tile_index(rows: int, cols: int) -> torch.Tensor:
    """Storage offset of element (r, col) in a ``Tile<rows, cols>`` image (``csrc/tile.cuh``):
    chunk c of row r at chunk position ``c ^ (r & 7)``."""
    ch = cols // 8
    r = torch.arange(rows).view(rows, 1)
    col = torch.arange(cols).view(1, cols)
    return ((r * ch + (((col >> 3) ^ (r & 7)))) * 8 + (col & 7)).reshape(-1)


@dataclass(frozen=True)
class CprState:
    """The recurrent state pool: one record per (slot, value head)."""

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
        """fp32 [P, hv, 128 v, 128 d] boundary state."""
        return self._section("S", D * D * 4, torch.float32, (D, D))

    def _sh_raw(self) -> torch.Tensor:
        return self._section("sh", D * D * 2, torch.bfloat16, (D * D,))

    def _image(self, name: str, rows: int, cols: int) -> torch.Tensor:
        raw = self._section(name, rows * cols * 2, torch.bfloat16, (rows * cols,))
        idx = _tile_index(rows, cols).to(raw.device)
        return raw[:, :, idx].view(self.rows, self.hv, rows, cols)

    @property
    def sh(self) -> torch.Tensor:
        """bf16 [P, hv, 128 v, 128 d]: rn(S), the operand every chain reads (a copy of a tile image)."""
        return self._image("sh", D, D)

    def set_sh(self, idx: torch.Tensor, value: torch.Tensor) -> None:
        """Store bf16 [n, hv, 128, 128] ``value`` as the S^ tile image of rows ``idx``."""
        inv = torch.argsort(_tile_index(D, D)).to(self.data.device)
        self._sh_raw()[idx] = value.reshape(*value.shape[:-2], D * D)[..., inv]

    @property
    def kb(self) -> torch.Tensor:
        """bf16 [P, hv, 64 i, 128 d] (a copy of a tile image)."""
        return self._image("kb", C, D)

    @property
    def vb(self) -> torch.Tensor:
        """bf16 [P, hv, 64 i, 128 v] (a copy of a tile image)."""
        return self._image("vb", C, D)

    @property
    def db(self) -> torch.Tensor:
        """bf16 [P, hv, 64 i, 128 v] (a copy of a tile image)."""
        return self._image("db", C, D)

    @property
    def xb(self) -> torch.Tensor:
        return self._section("xb", C * C * 4, torch.float32, (C, C))

    @property
    def pb(self) -> torch.Tensor:
        """fp32 [P, hv, 16, 64]: rows of P for the open 16-row block (row 16B + r at r).

        At open-chunk offset ``j = pos % 64`` with ``B = j // 16 >= 1`` the record defines rows
        ``[0, j % 16)`` and, in them, columns ``[0, 16 B)``; the kernels read nothing else. A call stores the
        rest of each row it writes as +0. Rows at or past ``j % 16`` are stale rows of earlier blocks, like
        the stale rows of every other open-chunk section."""
        return self._section("pb", 16 * C * 4, torch.float32, (16, C))

    @property
    def gb(self) -> torch.Tensor:
        return self._section("gb", C * 4, torch.float32, (C,))

    @property
    def bb(self) -> torch.Tensor:
        return self._section("bb", C * 4, torch.float32, (C,))

    @property
    def pos(self) -> torch.Tensor:
        """int32 [P, hv, NPOS] tokens consumed: NPOS equal copies (one per CTA of the widest split, so
        the CTAs never order against each other). Assigning through this view sets every copy."""
        return self._section("pos", 4 * NPOS, torch.int32, (NPOS,))

    def reset(self, slots: torch.Tensor) -> None:
        """Make ``slots`` fresh (zero boundary state, position 0). Stale buffer rows need no clearing."""
        idx = slots.long()
        self.S[idx] = 0.0
        self._sh_raw()[idx] = 0
        self.pos[idx] = 0

    def load_initial(self, slots: torch.Tensor, S0: torch.Tensor) -> None:
        """Start ``slots`` at position 0 from the fp32 [n, hv, 128, 128] initial states ``S0``."""
        idx = slots.long()
        S0 = S0.to(torch.float32)
        self.S[idx] = S0
        self.set_sh(idx, S0.to(torch.bfloat16))   # round-to-nearest-even, the recipe's rn()
        self.pos[idx] = 0


def _check_inputs(q, k, v, a, b):
    for name, t in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"gdn_cpr_sm100: {name} must be a CUDA tensor")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"gdn_cpr_sm100: {name} must be bfloat16, got {t.dtype}")
    if q.dim() != 3 or k.shape != q.shape or q.size(2) != D:
        raise ValueError(f"gdn_cpr_sm100: q/k must be [T, hk, {D}], got {tuple(q.shape)} / {tuple(k.shape)}")
    if v.dim() != 3 or v.size(0) != q.size(0) or v.size(2) != D:
        raise ValueError(f"gdn_cpr_sm100: v must be [T, hv, {D}], got {tuple(v.shape)}")
    hk, hv = q.size(1), v.size(1)
    if hk <= 0 or hv % hk:
        raise ValueError(f"gdn_cpr_sm100: value heads ({hv}) must be a multiple of key heads ({hk})")
    if a.shape != (q.size(0), hv) or b.shape != a.shape:
        raise ValueError(f"gdn_cpr_sm100: a/b must be [T, hv], got {tuple(a.shape)} / {tuple(b.shape)}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.stride(2) != 1 or t.data_ptr() % 16 or t.stride(0) % 8 or t.stride(1) % 8:
            raise ValueError(f"gdn_cpr_sm100: {name} rows must be contiguous and 16-byte aligned")


def _params(A_log, dt_bias, hv, device):
    keep = (torch.float32, torch.bfloat16)
    A, dt = A_log.detach(), dt_bias.detach()
    A = A.contiguous() if (A.dtype in keep and A.device == device) else A.to(device=device, dtype=torch.float32).contiguous()
    dt = dt.contiguous() if (dt.dtype in keep and dt.device == device) else dt.to(device=device, dtype=torch.float32).contiguous()
    if A.dtype != dt.dtype:
        A, dt = A.float().contiguous(), dt.float().contiguous()
    if A.shape != (hv,) or dt.shape != (hv,):
        raise ValueError(f"gdn_cpr_sm100: A_log/dt_bias must be [{hv}]")
    return A, dt


def _check_out(out, n, hv, dev):
    if out is None:
        return torch.empty(n, hv, D, device=dev, dtype=torch.bfloat16)
    if (out.shape != (n, hv, D) or out.dtype != torch.bfloat16 or out.stride(2) != 1 or out.data_ptr() % 16
            or out.stride(0) % 8 or out.stride(1) % 8):
        raise ValueError("gdn_cpr_sm100: out must be bf16 [n, hv, 128] with contiguous, 16-byte aligned rows")
    return out


def _check_state(state, hv, dev):
    if not isinstance(state, CprState):
        raise ValueError("gdn_cpr_sm100: state must be a CprState pool")
    t = state.data
    if (t.dtype != torch.uint8 or t.dim() != 3 or t.size(1) != hv or t.size(2) != record_layout()["bytes"]
            or t.device != dev or not t.is_contiguous()):
        raise ValueError(f"gdn_cpr_sm100: state must be a contiguous uint8 [P, {hv}, "
                         f"{record_layout()['bytes']}] CprState pool on {dev}")


def _check_slots(name, slots, n, dev):
    if not (isinstance(slots, torch.Tensor) and slots.dtype == torch.int32 and slots.numel() == n
            and slots.device == dev):
        raise ValueError(f"gdn_cpr_sm100: {name} must be int32 [{n}] on {dev}")
    return slots.contiguous()


def _max_len_of(cu_seqlens_cpu, n: int) -> int | None:
    """The longest sequence of the pack from the HOST copy of ``cu_seqlens``, or ``None``."""
    if cu_seqlens_cpu is None or n <= 0:
        return None
    try:
        if isinstance(cu_seqlens_cpu, torch.Tensor):
            if cu_seqlens_cpu.is_cuda or cu_seqlens_cpu.numel() < n + 1:
                return None
            cu = cu_seqlens_cpu[:n + 1].to(torch.int64)
            return int((cu[1:] - cu[:-1]).max().item())
        cu = [int(x) for x in cu_seqlens_cpu]
        if len(cu) < n + 1:
            return None
        return max(cu[i + 1] - cu[i] for i in range(n))
    except (TypeError, ValueError, RuntimeError):
        return None


def _check_cu(cu_seqlens, T):
    # Note: the final entry is NOT compared with T. cu_seqlens lives on the device and reading it here
    # would be a synchronization on the engine's hot path; the sm90 package makes the same choice. A
    # caller that passes a cu_seqlens describing a different token count gets out-of-bounds reads or
    # untouched output rows, and that is on the caller. In the PREPARED mode the out-of-bounds read
    # lands on the hand-off workspace rather than on q/k/v, because the grid is bounded by the host's
    # `T // 64 + 2 nseq` and a sequence longer than `T` claims chunk slots past it -- the same caller
    # error with a different victim, named here because the fourth Fable review asked for it to be.
    if not (isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.is_cuda and cu_seqlens.dtype == torch.int32
            and cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 1):
        raise ValueError("gdn_cpr_sm100: cu_seqlens must be an int32 CUDA tensor [N + 1]")
    n = cu_seqlens.numel() - 1
    if n == 0 and T > 0:
        raise ValueError(f"gdn_cpr_sm100: {T} tokens but no sequences (cu_seqlens has one entry)")
    return n


def prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, state=None, slot_in=None, slot_out=None, out=None,
            max_seqlen=None, sweep_rows=None, cu_seqlens_cpu=None,
            plan=None):
    """Prefill packed variable-length sequences."""
    del max_seqlen, sweep_rows
    _check_inputs(q, k, v, a, b)
    T, hv = q.size(0), v.size(1)
    dev = q.device
    n = _check_cu(cu_seqlens, T)
    maxlen = _max_len_of(cu_seqlens_cpu, n)
    A, dt = _params(A_log, dt_bias, hv, dev)
    out = _check_out(out, T, hv, dev)
    if n <= 0 or (T == 0 and state is None):
        return out
    pl = _plan(plan)
    prep = ((pl & 8) != 0 if plan is not None
            else _use_prepared(int(T), n, int(q.size(1)), hv, False, dev, maxlen))
    if state is None:
        if slot_in is not None or slot_out is not None:
            raise ValueError("gdn_cpr_sm100: slots given without a state pool")
        sdata = sin = sout = None
    else:
        _check_state(state, hv, dev)
        slot_in = _check_slots("slot_in", slot_in, n, dev)
        slot_out = slot_in if slot_out is None else _check_slots("slot_out", slot_out, n, dev)
        sdata, sin, sout = state.data, slot_in, slot_out
    cu = cu_seqlens.contiguous()
    if prep:
        # WHICH FORM, and therefore WHICH WORKSPACE SHAPE. The pre-kernel makes the per-(chunk, value head)
        # record the SHORT one, so a forced plan that switches the pre-kernel off (bits 4 and 5) must get
        # the FULL stride or launch A's `k^` / `q^` publish would run past the end of a row.
        eff = (PREPARED | (({2: 1, 4: 2}.get(_prep_split(n, hv, dev), 0)) << _PREP_SPLIT_SHIFT)
               if plan is None else pl)
        prek = (eff & 0x20) != 0 and _use_prekernel(int(q.size(1)), hv)
        maxch, stride, _, hkb, _ = _prep_sizes(int(T), n, int(q.size(1)), hv, prek)
        lay = record_layout()
        hand = torch.empty(maxch, hv, stride, device=dev, dtype=torch.uint8)
        handk = (torch.empty(maxch, int(q.size(1)), lay["prep_handk"], device=dev, dtype=torch.uint8)
                 if hkb else None)
        carry = torch.empty(n, hv, lay["prep_carry"], device=dev, dtype=torch.uint8)
        # `prep_table_ints` ints per chunk slot (the plan kernel's `ChunkRec`, read as two int4), then
        # `base[n + 1]`. `torch.empty` gives at least 16-byte alignment, which the int4 read needs and
        # `prefill_prepared` checks.
        table = torch.empty(lay["prep_table_ints"] * maxch + n + 1, device=dev, dtype=torch.int32)
        extension().prefill_prepared(q, k, v, a, b, A, dt, cu, sdata, sin, sout, out, SCALE, hand, handk,
                                     carry, table, eff & 0x1ff)
    else:
        extension().prefill(q, k, v, a, b, A, dt, cu, sdata, sin, sout, out, SCALE, [], None, pl)
    _CALLS["prefill"] += 1
    return out


def trainer_forward(q, k, v, a, b, A_log, dt_bias, cu_seqlens, *, save_states=False, cu_seqlens_cpu=None):
    """Stateless prefill with backward intermediates in FLA's layouts; output bits match ``prefill``.

    ``saved["h"]`` is allocated for the worst case, ``T // 64 + N`` chunks, because the exact count
    ``K = sum(ceil(len_n / 64))`` is only known on the device. The kernel writes rows ``[0, K)`` (FLA's
    chunk order) and the backward reads exactly ``h[:K]``; rows ``[K, T // 64 + N)`` are NOT written and
    hold whatever the allocation held, so they differ from run to run. Compare ``h[:K]``. (Zeroing them
    would cost a clear of up to ``N`` chunk states, 32 KiB per value head each, on every forward.)"""
    del cu_seqlens_cpu
    _check_inputs(q, k, v, a, b)
    T, hk, hv = q.size(0), q.size(1), v.size(1)
    dev = q.device
    n = _check_cu(cu_seqlens, T)
    A, dt = _params(A_log, dt_bias, hv, dev)
    out = _check_out(None, T, hv, dev)
    saved = dict(qn=torch.empty(T, hk, D, device=dev, dtype=torch.bfloat16),
                 kn=torch.empty(T, hk, D, device=dev, dtype=torch.bfloat16),
                 q_rstd=torch.empty(T, hk, device=dev, dtype=torch.float32),
                 k_rstd=torch.empty(T, hk, device=dev, dtype=torch.float32),
                 v=torch.empty(T, hv, D, device=dev, dtype=torch.bfloat16),
                 A=torch.empty(T, hv, C, device=dev, dtype=torch.bfloat16),
                 **{name: torch.empty(*(shape if save_states else (0,)), device=dev, dtype=torch.bfloat16)
                    for name, shape in (("w", (T, hv, D)), ("h", (T // C + max(n, 0), hv, D, D)),
                                        ("v_new", (T, hv, D)))})
    if n > 0 and T > 0:
        cb = torch.empty(n, device=dev, dtype=torch.int32) if save_states else None
        extension().prefill(q, k, v, a, b, A, dt, cu_seqlens.contiguous(), None, None, None, out, SCALE,
                            list(saved.values()), cb, PLAN_DEFAULT)
        _CALLS["trainer_forward"] += 1
    return out, saved


_DECODE_SPLITS = (1, 2, 4, 8)


def choose_decode_split(num_seqs: int, hv: int, device=None) -> int:
    """CTAs PER HEAD for a recurrent decode step, and it is a real choice now rather than API parity."""
    items = max(int(num_seqs), 1) * max(int(hv), 1)
    idx = device.index if (device is not None and device.index is not None) else None
    sms = _sm_count(idx if idx is not None else torch.cuda.current_device())
    return 2 if 2 * items <= sms * _decode_blocks_per_sm() else 1


def decode(q, k, v, a, b, A_log, dt_bias, state, slot_in, *, slot_out=None, out=None, split=None,
           plan=None):
    """One token per sequence: q/k [N, hk, 128], v [N, hv, 128], a/b [N, hv]."""
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
    if split is not None and split not in _DECODE_SPLITS:
        raise ValueError(f"gdn_cpr_sm100: unsupported decode split {split}")
    pl = _plan(plan)
    #: The value-row split is an EXECUTION choice on the call's shape. A forced plan may pin it (bits
    #: 9-11); otherwise the caller's `split` wins and, failing that, the rule.
    forced = (pl >> _REC_SPLIT_SHIFT) & 7 if pl >= 0 else 0
    if forced:
        r = 1 << (forced - 1)
    elif split is not None:
        r = int(split)
    else:
        r = choose_decode_split(N, hv, dev)
    # `plan`: diagnostic only. Overrides the deployment-fixed phase-1 EXECUTION plan (see PLANS below).
    # Every value produces bit-identical output -- tests/test_identity_gpu.py gates it -- so this is how the
    # two schedules are timed in ONE process and one provider rotation, a CUPTI number being comparable
    # only to one taken in the same rotation cycle.
    extension().decode(q, k, v, a, b, A, dt, state.data, slot_in, slot_out, out, SCALE, pl, r)
    _CALLS["decode"] += 1
    return out
