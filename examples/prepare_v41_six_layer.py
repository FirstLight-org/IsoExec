"""Prepare the six-layer BF16 checkpoint for the four-B200 SkyRL GRPO experiment."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile

import torch

from isoexec.core.checkpoint_quant import decode_checkpoint_weight

LAYERS = [0, 1, 14, 20, 24, 25]
CHAT_DEFAULT = ('{%- if thinking_mode is not defined and enable_thinking is not defined -%}'
                '{%- set thinking_mode = "chat" -%}{%- endif -%}\n')
GLOBAL_KEEP = ("embed.weight", "norm.weight", "head.weight")
DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_start", "image_end", "image_newline")
TABLE_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")
DTYPE_SIZE = {"F64": 8, "F32": 4, "I64": 8, "I32": 4, "BF16": 2, "F16": 2, "I16": 2, "I8": 1, "U8": 1,
              "F8_E4M3": 1, "F8_E8M0": 1, "BOOL": 1}
SHARD_BYTES = 4 << 30
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, 8 + n, raw


def source_tensors(src):
    """{name: {file, dtype, shape, offset (absolute), nbytes}} from every shard listed in the index."""
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for fname in sorted(set(index.values())):
        header, base, _ = read_header(src / fname)
        for name, v in header.items():
            if index.get(name) != fname or name in out:
                raise ValueError(f"{name} has an inconsistent shard assignment in {fname}")
            s, e = v["data_offsets"]
            out[name] = {"file": fname, "dtype": v["dtype"], "shape": list(v["shape"]), "offset": base + s,
                         "nbytes": e - s}
    missing = set(index) - set(out)
    if missing:
        raise SystemExit(f"index names tensors absent from the shards: {sorted(missing)[:5]}")
    return out


def most_recent(ids, i):
    c = [s for s in ids if s <= i]
    return c[-1] if c else None


def subset_text_config(tc, sel):
    """Rewrite ``text_config`` for production layers ``sel`` (renumbered 0..n-1); refuse unfaithful cuts."""
    if sel != sorted(set(sel)) or not sel or sel[-1] >= tc["num_hidden_layers"] or sel[0] < 0:
        raise ValueError("selection must be ascending unique backbone layer ids")
    remap = {p: r for r, p in enumerate(sel)}
    ratios = tc["compress_ratios"]
    for p in sel:
        if ratios[p] not in (0, 1, 2):
            raise ValueError(f"layer {p}: compress ratio {ratios[p]} not in (0, 1, 2)")
        if ratios[p]:
            for key in ("kv_source_layer_ids", "index_source_layer_ids"):
                s = most_recent(tc[key], p)
                if s is None or s not in remap:
                    raise ValueError(f"layer {p} reads {key[:-10]} from layer {s}, which is not kept")
                if ratios[s] != ratios[p]:
                    raise ValueError(f"layer {p} and its {key[:-10]} {s} differ in ratio")
                # After renumbering, the most recent kept source of the row must still be that source.
                if most_recent([remap[i] for i in tc[key] if i in remap], remap[p]) != remap[s]:
                    raise ValueError(f"layer {p}: renumbering changes its {key[:-10]}")
    cand = tc["candidate_source_layer_id"]
    for p in sel:
        if p in tc["index_source_layer_ids"] and p not in tc["kv_source_layer_ids"] and 0 <= cand < p:
            if cand not in remap:
                raise ValueError(f"Reindex layer {p} needs candidate source {cand}")
    released = list(tc["engram_layer_ids"])
    kept_engram = [p for p in released if p in remap]
    # Hash primes are drawn in Engram order: a faithful subset keeps a prefix of the released Engram layers
    # (the multipliers are then fixed by engram_hash_layer_ids; primes stay those of the release).
    if kept_engram != released[: len(kept_engram)]:
        raise ValueError(f"kept Engram layers {kept_engram} are not a prefix of the released {released}")
    kv_rows = [remap[i] for i in tc["kv_source_layer_ids"] if i in remap]
    if kv_rows and any(remap[p] >= max(kv_rows) for p in kept_engram):
        raise ValueError("an Engram row at or after vLLM's CED cut (last KV source) turns the replay off")
    out = dict(tc)
    out["num_hidden_layers"] = len(sel)
    out["compress_ratios"] = [ratios[p] for p in sel]
    out["kv_source_layer_ids"] = kv_rows
    out["index_source_layer_ids"] = [remap[i] for i in tc["index_source_layer_ids"] if i in remap]
    out["candidate_source_layer_id"] = remap.get(cand, -1)
    sizes = dict(zip(released, tc["engram_num_embeddings"]))
    out["engram_layer_ids"] = [remap[p] for p in kept_engram]
    out["engram_num_embeddings"] = [sizes[p] for p in kept_engram]
    out["engram_hash_layer_ids"] = list(kept_engram)
    out["num_nextn_predict_layers"] = 0
    out["dspark_target_layer_ids"] = []
    return out


def plan_weights(src_t, sel):
    """Output tensors of one form: [{name, src, kind, scale, dtype, shape, nbytes, table}]."""
    remap = {p: r for r, p in enumerate(sel)}
    out = []
    for name in sorted(src_t):
        m = LAYER_RE.match(name)
        if m:
            p = int(m.group(1))
            if p not in remap:
                continue
            new = f"layers.{remap[p]}.{m.group(2)}"
        elif name in GLOBAL_KEEP:
            new = name
        elif name.startswith(DROP_PREFIXES):
            continue
        else:
            raise ValueError(f"unclassified source tensor {name}")
        t = src_t[name]
        table = bool(TABLE_RE.match(name))
        rec = {"name": new, "src": name, "kind": "copy", "scale": None, "dtype": t["dtype"],
               "shape": t["shape"], "nbytes": t["nbytes"], "table": table}
        if not table:
            base = name[: -len(".scale")] if name.endswith(".scale") else None
            if base is not None and base + ".weight" in src_t and src_t[base + ".weight"]["dtype"] in ("F8_E4M3", "I8"):
                continue  # consumed by the decode of its weight
            if t["dtype"] in ("F8_E4M3", "I8"):
                sname = name[: -len(".weight")] + ".scale"
                if not name.endswith(".weight") or sname not in src_t:
                    raise ValueError(f"{name}: quantized tensor without a scale")
                s = src_t[sname]
                rows, cols = t["shape"]
                if s["dtype"] != "F8_E8M0":
                    raise ValueError(f"{sname}: scale dtype {s['dtype']}")
                if t["dtype"] == "F8_E4M3":
                    kind, shape = "fp8_32x32", [rows, cols]
                    want = [math.ceil(rows / 32), math.ceil(cols / 32)]
                else:
                    kind, shape = "mxfp4_1x32", [rows, 2 * cols]
                    want = [rows, 2 * cols // 32]
                    if (2 * cols) % 32:
                        raise ValueError(f"{name}: E2M1 K not a multiple of 32")
                if s["shape"] != want:
                    raise ValueError(f"{sname}: shape {s['shape']} != {want} for {kind}")
                rec.update(kind=kind, scale=sname, dtype="BF16", shape=shape, nbytes=2 * shape[0] * shape[1])
        out.append(rec)
    names = [r["name"] for r in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate output names")
    for g in GLOBAL_KEEP:
        if g not in names:
            raise ValueError(f"{g} missing from the source")
    return out


def row_key(name):
    if name == "embed.weight":
        return (-1, name)
    m = LAYER_RE.match(name)
    if m:
        return (int(m.group(1)), name)
    return (1 << 30, name)


def assign_shards(recs):
    """Each Engram table gets its own file; the rest is packed in row order into ~4 GiB shards."""
    tables = {}
    for r in recs:
        if r["table"]:
            row = LAYER_RE.match(r["name"]).group(1)
            tables.setdefault(f"engram-layers.{row}.safetensors", []).append(r)
    rest = sorted((r for r in recs if not r["table"]), key=lambda r: row_key(r["name"]))
    groups, cur, size = [], [], 0
    for r in rest:
        if cur and size + r["nbytes"] > SHARD_BYTES:
            groups.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += r["nbytes"]
    if cur:
        groups.append(cur)
    shards = {f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors": g for i, g in enumerate(groups)}
    shards.update(tables)
    return shards


def shard_layout(recs):
    """safetensors layout: tensors ordered by dtype width (descending) then name, header padded to 8 bytes."""
    order = sorted(recs, key=lambda r: (-DTYPE_SIZE[r["dtype"]], r["name"]))
    header, pos = {}, 0
    for r in order:
        header[r["name"]] = {"dtype": r["dtype"], "shape": r["shape"], "data_offsets": [pos, pos + r["nbytes"]]}
        pos += r["nbytes"]
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return order, struct.pack("<Q", len(raw)) + raw, pos


def write_shard(path, tensors, records, inputs):
    order, header, size = shard_layout(records)
    with path.open("wb") as out:
        out.write(header)
        for record in order:
            tensor = tensors[record["src"]]
            src = inputs[tensor["file"]]
            src.seek(tensor["offset"])
            if record["kind"] == "copy":
                remaining = tensor["nbytes"]
                while remaining:
                    block = src.read(min(CHUNK, remaining))
                    if not block:
                        raise EOFError(record["src"])
                    out.write(block)
                    remaining -= len(block)
                continue
            # Fetch weight before scale as the same file can contain both.
            raw = bytearray(src.read(tensor["nbytes"]))
            scale = tensors[record["scale"]]
            src_scale = inputs[scale["file"]]
            src_scale.seek(scale["offset"])
            raw_scale = bytearray(src_scale.read(scale["nbytes"]))
            scales = torch.frombuffer(raw_scale, dtype=torch.float8_e8m0fnu).view(scale["shape"])
            fp4 = record["kind"] == "mxfp4_1x32"
            weights = torch.frombuffer(raw, dtype=torch.int8 if fp4 else torch.float8_e4m3fn).view(tensor["shape"])
            # Both block encodings align on a 32-row boundary. Chunk the decode to bound transient memory.
            for first in range(0, len(weights), 256):
                after = min(first + 256, len(weights))
                scale_rows = scales[first:after] if fp4 else scales[first // 32:math.ceil(after / 32)]
                values = decode_checkpoint_weight(weights[first:after], scale_rows, "deepseek_fp4" if fp4 else "deepseek_fp8")
                out.write(values.bfloat16().contiguous().view(torch.uint8).numpy())
        if out.tell() != len(header) + size:
            raise ValueError(f"unexpected payload size in {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="released DeepSeek-V4.1-Flash checkpoint")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="new directory for the six-layer BF16 checkpoint")
    action.add_argument("--plan", action="store_true", help="read the config and shard headers only")
    args = parser.parse_args()
    src = args.source
    config = json.loads((src / "config.json").read_text())
    config["text_config"] = subset_text_config(config["text_config"], LAYERS)
    config.pop("quantization_config", None)
    config["isoexec_subset"] = {"layer_sources": LAYERS, "layout": "physical"}
    tensors = source_tensors(src)
    records = plan_weights(tensors, LAYERS)
    shards = assign_shards(records)
    total = sum(r["nbytes"] for r in records)
    print(f"Layers {LAYERS}, {len(records):,} tensors, {total:,} payload bytes", flush=True)
    if args.plan:
        return
    dst = args.out
    if dst.exists():
        parser.error(f"output already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dst.parent).free <= total:
        parser.error(f"checkpoint payload requires {total:,} bytes of free space")
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".v41-") as tmp, ExitStack() as files:
        staging = Path(tmp) / "checkpoint"
        staging.mkdir()
        filenames = {tensors[key]["file"] for r in records for key in (r["src"], r["scale"]) if key}
        inputs = {name: files.enter_context((src / name).open("rb")) for name in filenames}
        index = {}
        for name, group in shards.items():
            write_shard(staging / name, tensors, group, inputs)
            index.update({r["name"]: name for r in group})
            print(name, flush=True)
        for name in ("tokenizer.json", "LICENSE"):
            shutil.copyfile(src / name, staging / name)
        for name in ("encoding", "inference"):
            if (src / name).is_dir():
                shutil.copytree(src / name, staging / name)
        tokenizer = json.loads((src / "tokenizer_config.json").read_text())
        tokenizer["chat_template"] = CHAT_DEFAULT + (src / "chat_template.jinja").read_text()
        for name, content in (("config.json", config), ("tokenizer_config.json", tokenizer),
                              ("model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": index})):
            (staging / name).write_text(json.dumps(content, indent=2) + "\n")
        staging.rename(dst)


if __name__ == "__main__":
    main()
