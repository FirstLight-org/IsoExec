"""Measure which configuration is fastest per (shape, M bucket) and write ``tuned.json``.

PURE SCHEDULING: every configuration computes the same bits (the battery gates it), so this table moves
speed only. It is not part of the recipe identity (``segments.table_digest`` does not read it), and two
processes with different tables still agree bit for bit.

    python -m isoexec.ops.mm.mm_sm90.tune --model /path/to/model --tps 1,4 [--out tuned.json]

Timing is graph-replayed (one CUDA graph of ``reps`` launches, median of replays), the way a serving
engine runs decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import ops
from . import segments as seg

M_GRID = (1, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384)


def gtime(fn, reps: int = 20, iters: int = 5) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def _candidates(kind: str, M: int) -> list[str]:
    if kind == "node":
        names = seg.configs(out_fp32=1, min_levels=1)
        names += [ops.SPLIT_PREFIX + n for n in _candidates("leaves", M)]
        return names
    else:
        names = seg.configs(out_fp32=0, exact_levels=0)
    out = []
    for n in names:
        c = seg._CONFIGS[n]
        if c.swap and M > 256:
            continue
        if M >= 1024 and (c.block_m * c.block_n) < 128 * 128:
            continue
        if M <= 64 and not c.swap and c.block_m * c.block_n > 64 * 128:
            continue
        out.append(n)
    return out


def tune_shape(kind: str, K: int, N: int, leaf_k: int = 0, ms=M_GRID, max_bytes: float = 3e9,
               verbose: bool = True) -> list[list]:
    """``[[M, config, us, torch_us], ...]`` for one shape; ``kind`` is ``mm``, ``leaves`` or ``node``."""
    ids = ops.config_ids()
    ext = ops.extension()
    w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    rows = []
    for M in ms:
        if M * N * 2 * max(1, K // max(leaf_k, 1) if kind == "leaves" else 1) > max_bytes:
            continue
        x = torch.randn(M, K, device="cuda").bfloat16()
        reps = 5 if M * N > 2e8 else 20
        t_torch = gtime(lambda: torch.mm(x, w.t()), reps=reps)
        best = None
        times: dict = {}
        for name in _candidates(kind, M):
            try:
                if kind == "mm":
                    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                    cid = ids[name]
                    t = gtime(lambda: ext.run(cid, x, w, out, M, N, K, 1, N, K, K, M * N, 0, False, 0), reps=reps)
                elif kind == "leaves":
                    m = K // leaf_k
                    lv = torch.empty(m, M, N, device="cuda", dtype=torch.bfloat16)
                    t = gtime(lambda: ops.leaves(x, w, leaf_k, lv, config=name))
                else:
                    nd = torch.empty(M, N, device="cuda", dtype=torch.float32)
                    t = gtime(lambda: ops.leaf_node(x, w, leaf_k, nd, config=name))
            except Exception:  # noqa: BLE001 -- a config may not support the shape; it is just not a candidate
                continue
            times[name] = t
            if best is None or t < best[1]:
                best = (name, t)
        if best is not None and kind == "mm" and M >= 2048:
            # large M: also order the tiles (swizzle x raster) of the two best persistent configurations
            top = sorted(times.items(), key=lambda kv: kv[1])[:2]
            for name, _t in top:
                if seg._CONFIGS[name].schedule == "ws":
                    continue
                cid = ids[name]
                for swz in (1, 2, 4, 8):
                    for ras in (0, 1, 2):
                        if (swz, ras) == (1, 0):
                            continue
                        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                        t = gtime(lambda: ext.run(cid, x, w, out, M, N, K, 1, N, K, K, M * N, 0, False, 0, swz, ras),
                                  reps=reps)
                        if t < best[1]:
                            best = (f"{name}@s{swz}r{ras}", t)
        del x
        if best is not None:
            rows.append([M, best[0], round(best[1], 2), round(t_torch, 2)])
            if verbose:
                print(f"  {kind} K={K} N={N} leaf_k={leaf_k} M={M}: {best[0]} {best[1]:.2f}us "
                      f"(torch.mm {t_torch:.2f}us, {100 * (best[1] / t_torch - 1):+.0f}%)", flush=True)
    return rows


def model_shapes(model: str, tps: list[int], leaves: int = 8) -> dict:
    """The census (column-parallel) shapes and the row-parallel leaf shapes of a model."""
    from isoexec.integrations.verl.model import source_config
    from isoexec.runtimes.trainer_installs import model_gemm_census

    source, raw = source_config(model)
    census = model_gemm_census(source, raw)
    text = raw.get("text_config", raw)
    H = int(text["hidden_size"])
    row_k = {int(text["intermediate_size"])}
    if "num_attention_heads" in text and "head_dim" in text:
        row_k.add(int(text["num_attention_heads"]) * int(text["head_dim"]))
    if "linear_num_value_heads" in text:
        row_k.add(int(text["linear_num_value_heads"]) * int(text["linear_value_head_dim"]))
    mm, lv = set(), set()
    for tp in tps:
        for site in census.sites(tp, tp):
            if seg.supported(site.K, site.N):
                mm.add((site.K, site.N))
        for k_full in row_k:
            if k_full % leaves == 0 and seg.execution_supported(k_full // tp, H, k_full // leaves):
                lv.add((k_full // tp, H, k_full // leaves))
    return {"mm": sorted(mm), "leaves": sorted(lv)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tps", default="1,4")
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--out", default=str(seg.TUNED_PATH))
    a = ap.parse_args()
    ops.require_sm90()
    shapes = model_shapes(a.model, [int(t) for t in a.tps.split(",")], a.leaves)
    path = Path(a.out)
    table = json.loads(path.read_text()) if path.is_file() else {}
    table.setdefault("schema", "isoexec_mm_sm90_tuned_v1")
    table["device"] = torch.cuda.get_device_name()
    for kind in ("mm", "leaves", "node"):
        table.setdefault(kind, {})
    for K, N in shapes["mm"]:
        table["mm"][f"{K},{N}"] = tune_shape("mm", K, N)
        path.write_text(json.dumps(table, indent=1))
    for K, N, lk in shapes["leaves"]:
        table["leaves"][f"{K},{N},{lk}"] = tune_shape("leaves", K, N, lk, ms=[m for m in M_GRID if m <= 8192])
        table["node"][f"{K},{N},{lk}"] = tune_shape("node", K, N, lk, ms=[m for m in M_GRID if m <= 2048])
        path.write_text(json.dumps(table, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
