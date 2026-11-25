"""Shared fixtures for the gqa_sm100 GPU checks: request pools, paged layouts, and the two call sites.

Nothing here decides a numerical parameter. ``SEG`` comes from ``ops.seg_for``; the wave count, the
granularity and the page size are execution knobs the relations sweep.
"""

from __future__ import annotations

import dataclasses

import torch

from isoexec.ops.attention.gqa_sm100 import ops
from isoexec.ops.attention.gqa_sm100.reference import gqa_reference_fwd

BF16 = torch.bfloat16
DEV = "cuda"


@dataclasses.dataclass(frozen=True)
class Geometry:
    """A per-rank head geometry the op must serve."""

    tag: str
    h_q: int
    h_kv: int
    head_dim: int


#: The head-dim-128 geometries. One CTA per work item.
GEOMETRIES_D128 = (
    Geometry("d128 Hq8/Hkv1", 8, 1, 128),
    Geometry("d128 Hq16/Hkv2", 16, 2, 128),
)

#: The head-dim-256 geometries (the Qwen3.5 layers at TP 8, 4 and 1). One CTA PAIR per work item,
#: 128 packed query rows each.
GEOMETRIES_D256 = (
    Geometry("d256 Hq2/Hkv1", 2, 1, 256),
    Geometry("d256 Hq4/Hkv1", 4, 1, 256),
    Geometry("d256 Hq16/Hkv2", 16, 2, 256),
)

#: Every geometry the op serves. The relations that sweep this compare BOTH head dims against the
#: oracle and across addressing modes; the plan and page-layout relations pin ``GEOMETRIES[0]``
#: because what they vary is orthogonal to the head dim.
GEOMETRIES = GEOMETRIES_D128 + GEOMETRIES_D256


def draw(n: int, h: int, d: int, seed: int, population: str = "normal") -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(n, h, d, device=DEV, dtype=torch.float32, generator=g)
    if population == "large":
        x = x * 64.0
    elif population == "small":
        x = x * 2.0**-9
    elif population == "signed_zero":
        mask = torch.randint(0, 3, x.shape, device=DEV, generator=g) == 0
        sign = torch.randint(0, 2, x.shape, device=DEV, generator=g).to(torch.float32) * 2 - 1
        x = torch.where(mask, torch.zeros_like(x) * sign, x)
    return x.to(BF16)


def cu(lengths) -> torch.Tensor:
    out = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEV)
    out[1:] = torch.cumsum(torch.tensor(list(lengths), device=DEV, dtype=torch.int32), 0)
    return out


def page_kv(k, v, kv_lens, page: int, *, permute: bool = False, pad_blocks: int = 0, fill=None):
    """Lay packed K/V out as a paged cache plus a block table -- the engine's KV memory.

    ``permute`` scatters each sequence's pages through the cache so the block table's layout is
    provably invisible to the arithmetic; ``fill`` poisons every slot the table does not name.
    """
    h_kv, d = k.shape[1], k.shape[2]
    per = [(n + page - 1) // page for n in kv_lens]
    total = sum(per) + pad_blocks
    if fill is None:
        kc = torch.zeros(total, page, h_kv, d, device=k.device, dtype=k.dtype)
        vc = torch.zeros(total, page, h_kv, d, device=v.device, dtype=v.dtype)
    else:
        kc = torch.full((total, page, h_kv, d), fill, device=k.device, dtype=k.dtype)
        vc = torch.full((total, page, h_kv, d), fill, device=v.device, dtype=v.dtype)
    order = torch.randperm(total, device="cpu").tolist() if permute else list(range(total))
    table = torch.zeros(len(kv_lens), max(per), device=k.device, dtype=torch.int32)
    nxt = off = 0
    for i, n in enumerate(kv_lens):
        for j in range(per[i]):
            block = order[nxt]
            nxt += 1
            table[i, j] = block
            take = min(page, n - j * page)
            kc[block, :take] = k[off : off + take]
            vc[block, :take] = v[off : off + take]
            # the rest of a partial page is left at the fill value: masked keys never reach the
            # arithmetic, which is exactly what `fill` checks.
            off += take
    return kc, vc, table


@dataclasses.dataclass
class Batch:
    """One call's worth of inputs, in both the packed and the paged addressing mode."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_lens: tuple
    kv_lens: tuple
    geometry: Geometry
    page: int

    @property
    def cu_q(self):
        return cu(self.q_lens)

    @property
    def cu_k(self):
        return cu(self.kv_lens)

    @property
    def seqused(self):
        return torch.tensor(list(self.kv_lens), device=DEV, dtype=torch.int32)


def make_batch(q_lens, kv_lens, geometry: Geometry, seed: int, *, page: int = 256,
               population: str = "normal") -> Batch:
    q = draw(sum(q_lens), geometry.h_q, geometry.head_dim, seed, population)
    k = draw(sum(kv_lens), geometry.h_kv, geometry.head_dim, seed + 1, population)
    v = draw(sum(kv_lens), geometry.h_kv, geometry.head_dim, seed + 2, population)
    return Batch(q, k, v, tuple(q_lens), tuple(kv_lens), geometry, page)


def run_packed(batch: Batch, seg: int, **kw):
    """The trainer's call site: a packed KV cache with cu_seqlens_k."""
    return ops.gqa_sm100_fwd(
        batch.q, batch.k, batch.v, batch.cu_q, batch.seqused,
        softmax_scale=ops.default_scale(batch.geometry.head_dim), seg_tokens=seg,
        max_seqlen_q=max(batch.q_lens), max_seqlen_k=max(batch.kv_lens),
        cu_seqlens_k=batch.cu_k, **kw,
    )


def run_paged(batch: Batch, seg: int, *, page: int | None = None, permute: bool = False,
              pad_blocks: int = 0, fill=None, **kw):
    """The engine's call site: a paged KV cache with a block table."""
    page = page or batch.page
    kc, vc, table = page_kv(batch.k, batch.v, batch.kv_lens, page, permute=permute,
                            pad_blocks=pad_blocks, fill=fill)
    return ops.gqa_sm100_fwd(
        batch.q, kc, vc, batch.cu_q, batch.seqused,
        softmax_scale=ops.default_scale(batch.geometry.head_dim), seg_tokens=seg,
        max_seqlen_q=max(batch.q_lens), max_seqlen_k=max(batch.kv_lens),
        block_table=table, **kw,
    )


def reference(batch: Batch):
    return gqa_reference_fwd(
        batch.q, batch.k, batch.v, batch.cu_q, batch.seqused,
        ops.default_scale(batch.geometry.head_dim), cu_seqlens_k=batch.cu_k,
    )


def bits(t: torch.Tensor) -> torch.Tensor:
    """An integer view, so a comparison is bitwise and NaN compares equal to itself."""
    if t.dtype is torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype is torch.float32:
        return t.view(torch.int32)
    return t


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def first_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    ia, ib = bits(a).reshape(-1), bits(b).reshape(-1)
    bad = (ia != ib).nonzero()
    if bad.numel() == 0:
        return "equal"
    i = int(bad[0])
    return (f"{bad.numel()}/{ia.numel()} differ; first at flat {i}: "
            f"{a.reshape(-1)[i].item()!r} vs {b.reshape(-1)[i].item()!r}")


def accuracy(got_out, got_lse, ref_out, ref_lse, v: torch.Tensor):
    """Normalized output error and absolute lse error against the float64 oracle."""
    scale = float(v.to(torch.float64).abs().max().clamp(min=1e-30))
    o = float((got_out.to(torch.float64) - ref_out).abs().max()) / scale
    finite = torch.isfinite(ref_lse)
    l = float((got_lse.to(torch.float64)[finite] - ref_lse[finite]).abs().max()) if finite.any() else 0.0
    return o, l


__all__ = [
    "BF16",
    "DEV",
    "GEOMETRIES",
    "Batch",
    "Geometry",
    "accuracy",
    "bit_equal",
    "bits",
    "cu",
    "draw",
    "first_diff",
    "make_batch",
    "page_kv",
    "reference",
    "run_packed",
    "run_paged",
]
