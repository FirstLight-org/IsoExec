"""Digests of the trainer's replicated parameters, compared across every rank that holds a copy.

A parameter held in full by several ranks (a TP-replicated tensor, an expert shard under EP < ranks, any
tensor of a data-parallel replica) is updated by each holder's own optimizer. The copies stay bitwise equal
only if every holder receives bitwise equal gradients: either an all-reduce makes them equal, or each holder
recomputes them with a backward that gives the same bits on every rank. The engine receives one copy while
each trainer rank's forward reads its own, so a drift breaks engine == trainer logprobs.

Identity of a copy: the logical name of a tensor held at its full shape (every holder has the whole tensor);
the logical name plus the TP coordinate of a tensor held as a TP / expert-TP shard. EP-held experts carry
global ids in their names. With ``scope="recompute"`` the data-parallel coordinate (the expert data-parallel
one for ``allreduce=False`` tensors) is added, so only holders that saw the same tokens are compared (used for
gradients under the distributed optimizer, whose reduce-scatter leaves only each rank's shard complete).

Digests: ``"sha256"`` (host, exact; for gates) and ``"fold64"`` (device, two 64-bit modular sums of the
32-bit words times distinct odd position multipliers: any change of a single word changes it, integer
arithmetic so the value does not depend on reduction order; for run-time checks).
"""
from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch

DIGESTS = ("fold64", "sha256")
SCOPES = ("replicas", "recompute")
_CHUNK_WORDS = 1 << 24
_HOST_CHUNK_BYTES = 1 << 28


def _signed(value: int) -> int:
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >> 63 else value


_M1, _A1 = _signed(0x9E3779B97F4A7C15), _signed(0xD1B54A32D192ED03)
_M2, _A2 = _signed(0xBF58476D1CE4E5B9), _signed(0x94D049BB133111EB)


class ReplicaDivergence(RuntimeError):
    """Copies of one replicated tensor differ across the ranks that hold them."""


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat


def tensor_digest(tensor: torch.Tensor, kind: str = "fold64") -> str:
    """Digest of ``tensor``'s raw bytes (dtype and shape are compared separately)."""
    if kind not in DIGESTS:
        raise ValueError(f"digest must be one of {DIGESTS}, got {kind!r}")
    data = _bytes(tensor)
    if kind == "sha256":
        h = hashlib.sha256()
        for start in range(0, data.numel(), _HOST_CHUNK_BYTES):
            h.update(data[start:start + _HOST_CHUNK_BYTES].cpu().numpy().tobytes())
        return h.hexdigest()
    n = data.numel()
    if n % 4 or data.storage_offset() % 4:
        data = torch.cat((data, data.new_zeros((-n) % 4)))
    words = data.view(torch.int32)
    acc1 = torch.zeros((), dtype=torch.int64, device=data.device)
    acc2 = torch.zeros((), dtype=torch.int64, device=data.device)
    for start in range(0, words.numel(), _CHUNK_WORDS):
        w = words[start:start + _CHUNK_WORDS].to(torch.int64) & 0xFFFFFFFF
        idx = torch.arange(start, start + w.numel(), dtype=torch.int64, device=data.device)
        acc1 += (w * ((idx * _M1 + _A1) | 1)).sum()
        acc2 += (w * (((idx ^ _A2) * _M2) | 1)).sum()
    h1, h2 = (int(v) & ((1 << 64) - 1) for v in torch.stack((acc1, acc2)).tolist())
    return f"{h1:016x}{h2:016x}:{n}"


def replica_key(logical_name: str, local_shape: Sequence[int], full_shape: Sequence[int], *, tp_rank: int,
                tp_size: int, dp_coordinate: int | None = None) -> str:
    """The identity every copy of one tensor shares (see the module docstring)."""
    key = logical_name
    if tuple(local_shape) != tuple(full_shape):
        key += f"@tp{tp_rank}/{tp_size}"
    if dp_coordinate is not None:
        key += f"@dp{dp_coordinate}"
    return key


def _unwrap(module: Any) -> list[Any]:
    from isoexec.runtimes.megatron.model import MegatronModel

    out = []
    for m in (module if isinstance(module, (list, tuple)) else [module]):
        while not isinstance(m, MegatronModel) and hasattr(m, "module"):
            m = m.module
        if not isinstance(m, MegatronModel):
            raise TypeError(f"replica digests need MegatronModel stages, got {type(m).__name__}")
        out.append(m)
    return out


def model_entries(module: Any, *, scope: str = "replicas") -> list[tuple[str, str, torch.nn.Parameter]]:
    """``(replica key, logical name, Parameter)`` of every registered Parameter of this rank's stages."""
    from isoexec.compiler.realization.tp import global_shape

    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    entries = []
    for model in _unwrap(module):
        groups = model.pg_collection
        tp = groups.tp
        etp = getattr(model, "_etp_size", tp.size())
        for logical, parameter in model.graph_module.named_parameters():
            if getattr(parameter, "shared", False):
                continue   # a PP-tied replica of another stage's tensor; Megatron's embedding group keeps it
            full = global_shape(model.graph_module, logical, parameter, tp_size=tp.size(), side="trainer",
                                etp_size=etp)
            coordinate = None
            if scope == "recompute":
                expert = not getattr(parameter, "allreduce", True)
                group = getattr(groups, "expt_dp", None) if expert else (getattr(groups, "dp_cp", None)
                                                                           or getattr(groups, "dp", None))
                coordinate = 0 if group is None else group.rank()
            entries.append((replica_key(logical, parameter.shape, full, tp_rank=tp.rank(), tp_size=tp.size(),
                                        dp_coordinate=coordinate), logical, parameter))
    return entries


def local_digests(entries: Iterable[tuple[str, str, torch.nn.Parameter]], *, what: str = "param",
                  kind: str = "fold64") -> dict[str, tuple[str, str, tuple[int, ...]]]:
    """``{key: (digest, dtype, shape)}`` of each entry's parameter (``what="param"``) or its Megatron
    ``main_grad`` (``what="main_grad"``; entries without one are skipped)."""
    out = {}
    for key, _, parameter in entries:
        tensor = parameter if what == "param" else getattr(parameter, "main_grad", None)
        if tensor is None:
            continue
        out[key] = (tensor_digest(tensor, kind), str(tensor.dtype), tuple(tensor.shape))
    return out


def compare_replicas(gathered: Sequence[Mapping[str, tuple]], *, ranks: Sequence[int] | None = None) -> dict:
    """Compare the ranks' ``local_digests``. Returns the verdict: key counts by number of holders, and every
    key whose copies differ with the ranks of each distinct value, in this order of first appearance."""
    ranks = list(range(len(gathered))) if ranks is None else list(ranks)
    holders: dict[str, dict[tuple, list[int]]] = {}
    for rank, digests in zip(ranks, gathered, strict=True):
        for key, value in digests.items():
            holders.setdefault(key, {}).setdefault(tuple(value), []).append(rank)
    by_count: dict[int, int] = {}
    mismatches = []
    for key, values in holders.items():
        n = sum(len(r) for r in values.values())
        by_count[n] = by_count.get(n, 0) + 1
        if len(values) > 1:
            mismatches.append({"key": key, "copies": n,
                               "values": [{"ranks": r, "digest": v[0], "dtype": v[1], "shape": list(v[2])}
                                          for v, r in values.items()]})
    replicated = sum(c for n, c in by_count.items() if n > 1)
    return {"keys": len(holders), "replicated_keys": replicated, "holders": dict(sorted(by_count.items())),
            "mismatches": mismatches, "equal": not mismatches}


def _all_gather(group) -> Callable[[Any], list]:
    import torch.distributed as dist

    def gather(obj):
        out = [None] * dist.get_world_size(group)
        dist.all_gather_object(out, obj, group=group)
        return out
    return gather


def check_replicas(module: Any, *, what: str = "param", kind: str = "fold64", scope: str = "replicas",
                   group=None, gather: Callable[[Any], list] | None = None, entries=None,
                   raise_on_mismatch: bool = True, label: str = "") -> dict:
    """Collective over ``group`` (default: the world): digest this rank's copies, gather every rank's digests,
    compare each replicated tensor across its holders. Raises ``ReplicaDivergence`` naming the first differing
    tensor and the ranks of each value (``raise_on_mismatch``); returns the verdict with the time spent."""
    t0 = time.perf_counter()
    entries = model_entries(module, scope=scope) if entries is None else entries
    mine = local_digests(entries, what=what, kind=kind)
    t1 = time.perf_counter()
    gathered = (gather or _all_gather(group))(mine)
    verdict = compare_replicas(gathered)
    verdict.update(what=what, kind=kind, scope=scope, label=label, digest_s=t1 - t0,
                   gather_s=time.perf_counter() - t1,
                   local_bytes=sum(p.numel() * p.element_size() for _, _, p in entries
                                   if what == "param" or getattr(p, "main_grad", None) is not None))
    if raise_on_mismatch and not verdict["equal"]:
        first = verdict["mismatches"][0]
        raise ReplicaDivergence(f"{label + ': ' if label else ''}replicated {what} {first['key']} differs across "
                                f"its {first['copies']} holders: " + "; ".join(
                                    f"ranks {v['ranks']} {v['digest'][:16]}" for v in first["values"])
                                + f" ({len(verdict['mismatches'])} tensors differ)")
    return verdict


def check_after_optimizer_step(modules: Any) -> dict | None:
    """The run-time check at weight-sync time: every trainer rank calls it after each optimizer step (before
    the weights are handed to the engine). ``ISOEXEC_REPLICA_CHECK_EVERY`` = N checks every N-th call
    (default 1; 0 disables). No-op without a process group of more than one rank."""
    import torch.distributed as dist

    every = int(os.environ.get("ISOEXEC_REPLICA_CHECK_EVERY", "1"))
    if every <= 0 or not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    _STEP[0] += 1
    if _STEP[0] % every:
        return None
    return check_replicas(modules, kind="fold64", label=f"optimizer step {_STEP[0]}")


_STEP = [0]

__all__ = ["DIGESTS", "SCOPES", "ReplicaDivergence", "check_after_optimizer_step", "check_replicas",
           "compare_replicas", "local_digests", "model_entries", "replica_key", "tensor_digest"]
