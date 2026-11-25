"""Backends, request pools, paged batch assembly and raw-bit comparisons for MLA conformance.
Each assembly copies the same logical request tensors, varying packing, companions, chunking and page placement."""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import Callable, Optional

import torch

from isoexec.ops.attention.mla_sm90.ops import kernel_fwd, kernel_plan

LATENT, ROPE, WIDTH, PAGE = 512, 64, 576, 64
SCALE = 1.0 / math.sqrt(256.0)
DTYPE = torch.bfloat16
# Stock FlashMLA comparator: a built checkout of https://github.com/deepseek-ai/FlashMLA (opt-in).
FLASHMLA_HEAD = os.environ.get("ISOEXEC_FLASHMLA_PATH", "")


class Unsupported(Exception):
    """The backend cannot express this call (e.g. FlashMLA with ragged query lengths)."""


# bit comparison

_INT = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_INT[t.dtype]) if t.dtype in _INT else t.contiguous()


@dataclasses.dataclass
class Diff:
    """Result of one bitwise comparison. ``detail`` names the first differing element."""

    name: str
    equal: bool
    shape: tuple
    n_diff: int = 0
    detail: str = ""

    def as_dict(self):
        return dataclasses.asdict(self)


def bit_diff(name: str, a: torch.Tensor, b: torch.Tensor, labels: Optional[list] = None) -> Diff:
    """Compare raw bits; on mismatch report the first differing index (row label, head, column)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return Diff(name, False, tuple(a.shape), -1, f"shape/dtype mismatch {tuple(a.shape)}/{a.dtype} vs "
                                                    f"{tuple(b.shape)}/{b.dtype}")
    ne = bits(a) != bits(b)
    n = int(ne.sum())
    if n == 0:
        return Diff(name, True, tuple(a.shape))
    idx = tuple(int(i) for i in ne.nonzero()[0].tolist())
    row = idx[0] if idx else None
    rows_bad = int(ne.reshape(ne.shape[0], -1).any(-1).sum()) if ne.ndim else 1
    label = labels[row] if labels is not None and row is not None and row < len(labels) else row
    detail = (f"first diff at index {idx} (row {label}); a={a[idx].item()!r} b={b[idx].item()!r}; "
              f"{n} elements in {rows_bad}/{ne.shape[0] if ne.ndim else 1} rows differ")
    return Diff(name, False, tuple(a.shape), n, detail)


# requests and batch assembly


@dataclasses.dataclass
class Request:
    kv: torch.Tensor  # [L, 576] bf16
    q: torch.Tensor  # [L, h, 576] bf16
    tag: str = ""

    @property
    def length(self) -> int:
        return self.kv.shape[0]


def make_request(length: int, heads: int, seed: int, device="cuda", tag="",
                 q_scale: float = 1.0, kv_scale: float = 1.0) -> Request:
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = (torch.randn(length, WIDTH, generator=g) * kv_scale).to(device, DTYPE)
    q = (torch.randn(length, heads, WIDTH, generator=g) * q_scale).to(device, DTYPE)
    return Request(kv, q, tag or f"L{length}/s{seed}")


@dataclasses.dataclass
class Slice:
    req: Request
    q_start: int
    q_end: int
    used: Optional[int] = None  # KV length; defaults to q_end (right-aligned)

    def __post_init__(self):
        if self.used is None:
            self.used = self.q_end
        if self.used - (self.q_end - self.q_start) != self.q_start:
            raise ValueError("slices are right-aligned: q_start must equal used - q_len")

    @property
    def q_len(self):
        return self.q_end - self.q_start


def decode(req: Request, p: int, **kw) -> Slice:
    return Slice(req, p, p + 1, **kw)


def full(req: Request, **kw) -> Slice:
    return Slice(req, 0, req.length, **kw)


@dataclasses.dataclass
class Batch:
    q: torch.Tensor
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    max_seqlen_q: int
    row_ranges: list  # per slice: (start, end) in packed T
    labels: list  # per packed row: "tag@pos"
    slices: list
    page_of: list  # per slice: list of physical page ids used

    def rows(self, i: int, out: torch.Tensor) -> torch.Tensor:
        s, e = self.row_ranges[i]
        return out[s:e]

    def labels_of(self, i: int) -> list:
        s, e = self.row_ranges[i]
        return self.labels[s:e]


def assemble(slices, *, page_perm_seed: Optional[int] = None, fill: str = "zeros", extra_pages: int = 0,
             table_width: Optional[int] = None, table_pad: str = "zero", max_seqlen_q: Optional[int] = None,
             device="cuda") -> Batch:
    """Pack slices into one paged batch.

    fill: what occupies cache slots beyond each slice's ``used`` inside its last page and every
      page nobody references: ``zeros``, ``future`` (the request's real later tokens), ``nan``,
      ``inf``, ``ninf``, ``garbage`` (huge finite values).
    table_pad: block-table entries past a slice's used pages: ``zero`` (page 0), ``poison`` (a page
      filled per ``fill``), ``last`` (the last valid page id).
    extra_pages: unreferenced physical pages appended to the cache (filled per ``fill``).
    page_perm_seed: shuffle physical page ids (non-contiguous tables, like an engine).
    """
    heads = slices[0].req.q.shape[1] if slices else 1
    counts = [(s.used + PAGE - 1) // PAGE for s in slices]
    poison_page = 1 if table_pad == "poison" else 0
    total = sum(counts) + extra_pages + poison_page
    total = max(total, 1)
    ids = list(range(total))
    if page_perm_seed is not None:
        g = torch.Generator().manual_seed(page_perm_seed)
        ids = [ids[i] for i in torch.randperm(total, generator=g).tolist()]
    fill_value = {"zeros": 0.0, "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"),
                  "garbage": 6.0e4, "future": 0.0}[fill]
    cache = torch.full((total, PAGE, 1, WIDTH), fill_value, device=device, dtype=DTYPE)
    width = table_width if table_width is not None else max([1] + counts)
    table = torch.zeros(len(slices), width, dtype=torch.int32)
    page_of, cursor = [], 0
    for i, (s, n) in enumerate(zip(slices, counts)):
        mine = ids[cursor:cursor + n]
        cursor += n
        page_of.append(mine)
        if n:
            table[i, :n] = torch.tensor(mine, dtype=torch.int32)
        if width > n:
            pad = {"zero": 0, "poison": ids[total - 1] if poison_page else 0, "last": mine[-1] if mine else 0}[table_pad]
            table[i, n:] = pad
        src = s.req.kv
        flat_idx = torch.tensor(mine, dtype=torch.long, device=device)
        if n:
            block = cache[flat_idx].view(n * PAGE, WIDTH)
            block[: s.used] = src[: s.used]
            if fill == "future":
                extra = min(src.shape[0], n * PAGE) - s.used
                if extra > 0:
                    block[s.used : s.used + extra] = src[s.used : s.used + extra]
            cache[flat_idx] = block.view(n, PAGE, 1, WIDTH)
    q_parts, labels, ranges, t = [], [], [], 0
    for s in slices:
        q_parts.append(s.req.q[s.q_start:s.q_end])
        labels += [f"{s.req.tag}@{p}" for p in range(s.q_start, s.q_end)]
        ranges.append((t, t + s.q_len))
        t += s.q_len
    q = torch.cat(q_parts) if q_parts else torch.zeros(0, heads, WIDTH, device=device, dtype=DTYPE)
    cu = torch.tensor([0] + [r[1] for r in ranges], dtype=torch.int32, device=device)
    return Batch(
        q=q.contiguous(), kv_cache=cache, block_table=table.to(device),
        cu_seqlens_q=cu,
        seqused_k=torch.tensor([s.used for s in slices], dtype=torch.int32, device=device),
        max_seqlen_q=max_seqlen_q if max_seqlen_q is not None else max([0] + [s.q_len for s in slices]),
        row_ranges=ranges, labels=labels, slices=list(slices), page_of=page_of,
    )


# backends


class Backend:
    """A forward (and optionally backward) kernel under test, over the DESIGN.md calling convention."""

    name = "abstract"
    has_seg = False  # consumes seg_tokens (the deployment segment size)
    has_overrides = False
    has_backward = False

    def __init__(self, seg_tokens: int = 512):
        self.seg_tokens = seg_tokens

    def fwd(self, b: Batch, exec_override=None):  # -> (out [T,h,512], lse [T,h] or None)
        raise NotImplementedError

    def overrides(self) -> list:
        return []


class SM90Backend(Backend):
    name = "mla_sm90"
    has_seg = True

    def __init__(self, seg_tokens: int = 512):
        super().__init__(seg_tokens)
        from isoexec.ops.attention.mla_sm90 import fwd as fwd_mod
        from isoexec.ops.attention.mla_sm90.ops import kernel_fwd  # noqa: F401  (the role-free call convention)

        self._fwd_mod = fwd_mod
        self._fwd = fwd_mod.mla_sm90_fwd
        self.has_overrides = True
        try:
            from isoexec.ops.attention import gqa_sm90
            from isoexec.ops.attention.mla_sm90.bwd import mla_sm90_attention_grads  # noqa: F401

            self.has_backward = gqa_sm90.available()
        except Exception:  # the gqa_sm90 FA3 build is unavailable
            pass

    def fwd(self, b: Batch, exec_override=None, seg_tokens=None):
        """``seg_tokens`` overrides the deployment SEG for one call: used ONLY by the sensitivity emulation of
        a phase-dependent program, never by a conformance relation."""
        kw = {}
        if exec_override is not None:
            ov = dict(exec_override)
            plan_mode = ov.pop("_plan", None)
            if plan_mode is not None:
                # Not a kernel override: exercise the caller-built plan paths (exact, or fixed capacity as a
                # captured CUDA-graph region would build it) with the default execution.
                cap = None if plan_mode == "exact" else {
                    "max_seqlen_k": int(b.seqused_k.max()) + (0 if plan_mode == "capacity" else 4096),
                    "num_tokens": b.q.shape[0] + (0 if plan_mode == "capacity" else 777)}
                kw["plan"] = kernel_plan(b.cu_seqlens_q, b.seqused_k, self.seg_tokens, b.q.shape[1], b.max_seqlen_q,
                                         capacity=cap, exec_override=ov or None)
            else:
                kw["exec_override"] = ov
        return kernel_fwd(b.q, b.kv_cache, b.block_table, b.cu_seqlens_q, b.seqused_k, SCALE,
                          seg_tokens if seg_tokens is not None else self.seg_tokens, b.max_seqlen_q, **kw)

    def identity(self) -> dict:
        """Digests of the kernel sources in use, so a result is attributable to one build of each stream."""
        import hashlib
        from pathlib import Path

        ident = {}
        try:
            ident["fwd_profile_digest"] = self._fwd_mod.profile_digest()
        except Exception as exc:  # pragma: no cover
            ident["fwd_profile_digest"] = f"unavailable: {exc!r}"
        h = hashlib.sha256()
        root = Path(self._fwd_mod.__file__).resolve().parent
        for p in [root / "bwd.py", root / "fwd.py", root / "ops.py"]:
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        ident["fwd_bwd_source_digest"] = h.hexdigest()[:16]
        ident["fwd_signature"] = "role-free"
        return ident

    def overrides(self) -> list:
        """Execution overrides the forward stream publishes (``EXEC_OVERRIDE_VARIANTS`` or a function)."""
        for attr in ("exec_override_variants", "EXEC_OVERRIDE_VARIANTS"):
            v = getattr(self._fwd_mod, attr, None)
            if v is not None:
                return list(v(self.seg_tokens) if callable(v) else v) + PLAN_VARIANTS
        # fwd.py's documented keys (mla_sm90_plan: granularity, num_parts, reverse, item_overhead, fold_overhead);
        # the plan refuses unknown keys, so a variant can never be silently ignored.
        return DEFAULT_OVERRIDES + PLAN_VARIANTS


DEFAULT_OVERRIDES = [
    {"granularity": "single"},
    {"granularity": "whole"},
    {"granularity": "pairs"},
    {"granularity": "mixed"},
    {"num_parts": 1},
    {"num_parts": 2},
    {"num_parts": 7},
    {"num_parts": 64},
    {"num_parts": 1000},
    {"reverse": True},
    {"reverse": True, "num_parts": 3},
    {"item_overhead": 0, "fold_overhead": 0},
    {"item_overhead": 100000},
]
PLAN_VARIANTS = [{"_plan": "exact"}, {"_plan": "capacity"}, {"_plan": "capacity_loose"},
                 {"_plan": "capacity_loose", "num_parts": 1000}]


class FlashMLABackend(Backend):
    """Stock FlashMLA SM90 dense decode kernel. Uniform query length per call only."""

    def __init__(self, seg_tokens: int = 512, source: str = "head"):
        super().__init__(seg_tokens)
        if source == "head":
            if not FLASHMLA_HEAD:
                raise ImportError("flashmla_head needs ISOEXEC_FLASHMLA_PATH set to a built FlashMLA checkout")
            if FLASHMLA_HEAD not in sys.path:
                sys.path.insert(0, FLASHMLA_HEAD)
            from flash_mla import flash_mla_with_kvcache, get_mla_metadata
            import flash_mla

            if not flash_mla.__file__.startswith(FLASHMLA_HEAD):
                raise ImportError(f"flash_mla resolved to {flash_mla.__file__}, not the HEAD build")
            self.name = "flashmla_head"
        else:
            from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata

            self.name = "flashmla_vllm"
        self._fn, self._meta = flash_mla_with_kvcache, get_mla_metadata

    def fwd(self, b: Batch, exec_override=None):
        B = b.block_table.shape[0]
        lens = set(torch.diff(b.cu_seqlens_q).tolist())
        if B == 0 or len(lens) != 1:
            raise Unsupported("FlashMLA needs B >= 1 and a uniform query length")
        s_q = lens.pop()
        if s_q == 0:
            # Measured: stock FlashMLA (HEAD ba89a34) never returns for s_q = 0 with B > 0.
            raise Unsupported("FlashMLA hangs on s_q = 0 (T = 0 with B > 0); not called")
        h = b.q.shape[1]
        meta, _ = self._meta()
        out, lse = self._fn(b.q.view(B, s_q, h, WIDTH), b.kv_cache, b.block_table, b.seqused_k, LATENT, meta, None,
                            softmax_scale=SCALE, causal=True)
        return out.reshape(B * s_q, h, LATENT), lse.permute(0, 2, 1).reshape(B * s_q, h)


class FA3Backend(Backend):
    """FA3's absorbed-MLA forward. ``num_splits=1`` is IsoExec's own build (gqa_sm90, the production
    kernel); ``num_splits=0`` is FA3's split heuristic, which gqa_sm90 compiles out, so that incumbent
    comes from vLLM's own FA3 build (``vllm.vllm_flash_attn``, fa_version=3) as a comparator only."""

    def __init__(self, seg_tokens: int = 512, num_splits: int = 1):
        super().__init__(seg_tokens)
        if num_splits == 1:
            from isoexec.ops.attention.gqa_sm90 import ops

            ops.forward_kernel(64, LATENT)
            self._fn = self._gqa_sm90
        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            self._vllm = flash_attn_varlen_func
            self._fn = self._vllm_fa3
        self.num_splits = num_splits
        self.name = f"fa3_ns{num_splits}"

    @staticmethod
    def _gqa_sm90(b: Batch):
        from isoexec.ops.attention.gqa_sm90 import ops

        kv = b.kv_cache
        return ops.paged_fwd(b.q[..., LATENT:].contiguous(), kv[..., LATENT:], kv[..., :LATENT], b.cu_seqlens_q,
                             b.seqused_k, b.block_table, b.max_seqlen_q, None, SCALE,
                             q_v=b.q[..., :LATENT].contiguous())

    def _vllm_fa3(self, b: Batch):
        kv = b.kv_cache
        return self._vllm(b.q[..., LATENT:].contiguous(), kv[..., LATENT:], kv[..., :LATENT], b.max_seqlen_q,
                          b.cu_seqlens_q, int(b.seqused_k.max()), seqused_k=b.seqused_k,
                          q_v=b.q[..., :LATENT].contiguous(), softmax_scale=SCALE, causal=True,
                          window_size=[-1, 0], block_table=b.block_table, return_softmax_lse=True,
                          num_splits=self.num_splits, fa_version=3)

    def fwd(self, b: Batch, exec_override=None):
        if b.block_table.shape[0] == 0:
            raise Unsupported("FA3 needs B >= 1")
        out, lse = self._fn(b)[:2]
        return out, lse.transpose(0, 1).contiguous()


BACKEND_FACTORIES: dict[str, Callable[[int], Backend]] = {
    "mla_sm90": lambda seg: SM90Backend(seg),
    "flashmla_head": lambda seg: FlashMLABackend(seg, "head"),
    "flashmla_vllm": lambda seg: FlashMLABackend(seg, "vllm"),
    "fa3_ns0": lambda seg: FA3Backend(seg, 0),
    "fa3_ns1": lambda seg: FA3Backend(seg, 1),
}


def load_backend(name: str, seg_tokens: int = 512):
    """(backend, None) or (None, reason)."""
    try:
        return BACKEND_FACTORIES[name](seg_tokens), None
    except Exception as exc:  # import failures become a SKIP, never a pass
        return None, f"{type(exc).__name__}: {exc}"


def cuda_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
