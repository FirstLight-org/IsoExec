"""Shape record and row sweep for the vendored DeepGEMM JIT, so a prebuilt cache holds EVERY kernel a run can need.

The vendored kernels are JIT-compiled per (shape, launch configuration), and the configuration is chosen per call
from the row count M (tile sizes, swap-AB, cluster): which kernels a run compiles depends on which M its scheduler
happens to produce, so a cache filled by one run can miss a kernel another run of the same model needs (seen:
``bhr_hdr_bhd`` at a decode row count the recording run never formed). The weight shapes are fixed by the model and
layout; only M varies. Hence:

- ``note(kind, ...)``: every IsoExec entry point to a DeepGEMM GEMM records its static signature (weight shape,
  segment / leaf count, grad or inference mode) ONCE per process into ``$ISOEXEC_PREBUILT_DIR/dg-shapes.jsonl``
  while a prebuilt cache is being recorded (``ISOEXEC_PREBUILT_DIR`` set, ``ISOEXEC_PREBUILT_ONLY`` not). Otherwise
  it is one boolean test.
- ``sweep(root, max_m)``: on a GPU, calls the same entry point for every recorded signature at every M in
  ``1..max_m`` on zero operands, so every configuration the heuristics can pick is compiled into the cache.

Values never matter (zeros); the entry points are the production ones, so their launch policy (shape table, swap-AB
policy, early launch under inference mode) is exactly what a run uses. MegaMoE compiles its own geometries at
placement (``moe/deepseek_v41_sm100/mega.warm``) and is not swept here.
"""
from __future__ import annotations

import json
import os

_STATE = {"on": None, "seen": set()}


def _recording() -> bool:
    if _STATE["on"] is None:
        _STATE["on"] = bool(os.environ.get("ISOEXEC_PREBUILT_DIR")) and os.environ.get("ISOEXEC_PREBUILT_ONLY") != "1"
    return _STATE["on"]


def note(kind: str, **dims) -> None:
    """Record one DeepGEMM call signature (no tensors: shapes, dtypes and switches only)."""
    if not _recording():
        return
    import torch

    dims["inference"] = bool(torch.is_inference_mode_enabled())
    key = (kind, tuple(sorted(dims.items())))
    if key in _STATE["seen"]:
        return
    _STATE["seen"].add(key)
    path = os.path.join(os.environ["ISOEXEC_PREBUILT_DIR"], "dg-shapes.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:                     # one short line per write: O_APPEND keeps lines whole
            f.write(json.dumps({"kind": kind, **dims}, sort_keys=True) + "\n")
    except OSError:
        pass


def shapes(root) -> list[dict]:
    path = os.path.join(str(root), "dg-shapes.jsonl")
    if not os.path.exists(path):
        return []
    rows = {json.dumps(json.loads(line), sort_keys=True) for line in open(path) if line.strip()}
    return [json.loads(r) for r in sorted(rows)]


def _call(row: dict, m: int, pool: dict, device):
    import torch

    def zeros(*shape, dtype=torch.bfloat16):
        key = (shape[1:], dtype)
        need = shape[0]
        buf = pool.get(key)
        if buf is None or buf.shape[0] < need:
            buf = pool[key] = torch.zeros((need, *shape[1:]), dtype=dtype, device=device)
        return buf[:need]

    kind = row["kind"]
    if kind == "gemm":
        from .bf16 import _gemm
        _gemm(zeros(m, row["k"]), zeros(row["n"], row["k"]))
    elif kind == "bhr":
        from .ops import bf16_grouped_bhr_hdr_bhd
        g, r, d = row["g"], row["r"], row["d"]
        og = zeros(g * m, d).view(g, m, d)
        out = torch.empty((m, g, r), dtype=torch.bfloat16, device=device)
        bf16_grouped_bhr_hdr_bhd(og.permute(1, 0, 2), zeros(g * r, d).view(g, r, d), out)
    elif kind in ("seg_partials", "seg_linear", "seg_leaf"):
        from . import segtree
        x, w = zeros(m, row["k"]), zeros(row["n"], row["k"])
        if kind == "seg_partials":
            segtree.partials(x, w, row["segments"], noswap=row["noswap"])
        elif kind == "seg_linear":
            segtree.linear(x, w, row["segments"], out_dtype=getattr(torch, row["out_dtype"]), noswap=row["noswap"])
        else:
            segtree.leaf_linear(x, w, row["leaves"], out_dtype=getattr(torch, row["out_dtype"]))
    elif kind == "gate":
        from isoexec.ops.moe.router_sm100.ops import route_mega
        e, h = row["e"], row["h"]
        kw = {k: row[k] for k in ("routing", "top_k", "n_group", "topk_group", "routed_scale", "num_shared",
                                  "num_split_k", "num_mma_ctas")}
        bias = torch.zeros(e, dtype=getattr(torch, row["bias"]), device=device) if row.get("bias") else None
        route_mega(zeros(m, h), zeros(e, h), expert_bias=bias, **kw)
    else:
        raise ValueError(f"unknown DeepGEMM signature kind {kind!r}")


def sweep(root, max_m: int, *, max_m_grad: int | None = None, device=None, log=print) -> dict:
    """Call every recorded signature at M = 1..max_m (grad-mode signatures up to ``max_m_grad``, default max_m)."""
    import contextlib

    import torch

    device = device or torch.device("cuda", torch.cuda.current_device())
    rows = shapes(root)
    done = {}
    for row in rows:
        limit = max_m if row.get("inference") else (max_m_grad or max_m)
        mode = torch.inference_mode() if row.get("inference") else contextlib.nullcontext()
        pool: dict = {}
        with mode, torch.no_grad():
            for m in range(1, limit + 1):
                _call(row, m, pool, device)
        torch.cuda.synchronize(device)
        pool.clear()
        torch.cuda.empty_cache()           # outputs of every M otherwise pile up in the caching allocator (130 GiB seen)
        done[json.dumps(row, sort_keys=True)] = limit
        log(f"[dg-sweep] {row} M=1..{limit}")
    return done


__all__ = ["note", "shapes", "sweep"]
