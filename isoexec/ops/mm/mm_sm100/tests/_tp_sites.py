"""Every dense GEMM site of the released Qwen3.5 geometries, with the slices a TP placement gives a rank.

Read off the models' own meta builds (``isoexec/models``, the fixture configs) and their declared
``tensor_axes``, so a site's logical shape and its shard rule are the ones placement uses, not a
restatement of them. One instance per module kind: every layer of a kind presents the same shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MODELS = Path(__file__).resolve().parents[4] / "models"

#: Released geometry -> (author package, effective HF config).
GEOMETRIES = {
    "qwen3.5-0.8b": ("isoexec.models.qwen3_5", _MODELS / "qwen3_5/tests/fixtures/qwen3_5_0_8b.json"),
    "qwen3.5-9b": ("isoexec.models.qwen3_5", _MODELS / "qwen3_5/tests/fixtures/qwen3_5_9b.json"),
    "qwen3.5-35b-a3b": ("isoexec.models.qwen3_5_moe", _MODELS / "qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json"),
}

TPS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Member:
    role: str
    shape: tuple[int, int]  # the LOGICAL weight, [N, K]
    axis: object  # its ShardAxis


@dataclass(frozen=True)
class Site:
    """One GEMM: ``x[M, K] @ cat(members)[N, K].T``.

    ``kind`` is ``column`` (members shard N, one GEMM over the rank's concatenated row blocks),
    ``row`` (one member, K sharded, combined by the fixed leaf tree) or ``replicated``.
    """

    geometry: str
    name: str
    kind: str
    members: tuple[Member, ...]
    dtype: str = "bf16"

    @property
    def K(self) -> int:
        return self.members[0].shape[1]


#: (module class name, site name, kind, member roles). Column members are listed in the order the
#: module concatenates them for one GEMM (``packed_proj_roles`` / ``packed_view``). Matched by class
#: name: ``ops/`` may not import the component tier.
KINDS = (
    ("GDN", "gdn.in_proj", "column", ("q", "k", "v", "z", "beta", "alpha")),
    ("GDN", "gdn.out", "row", ("o",)),
    ("Attention", "attn.q", "column", ("q",)),
    ("Attention", "attn.k", "column", ("k",)),
    ("Attention", "attn.v", "column", ("v",)),
    ("Attention", "attn.gate", "column", ("gate",)),
    ("Attention", "attn.qkv_gate(packed)", "column", ("q", "k", "v", "gate")),
    ("Attention", "attn.o", "row", ("o",)),
    ("MLP", "mlp.gate", "column", ("gate",)),
    ("MLP", "mlp.up", "column", ("up",)),
    ("MLP", "mlp.down", "row", ("down",)),
    ("_Shared", "shared.gate_up", "column", ("gate", "up")),
    ("_Shared", "shared.down", "row", ("down",)),
    ("_Shared", "shared.expert_gate", "replicated", ("expert_gate",)),
    ("_Router", "router", "replicated", ("weight",)),
    ("LMHead", "lm_head", "column", ("weight",)),
)


def sites(geometry: str) -> list[Site]:
    """The GEMM sites of one released geometry, from its meta build."""
    import torch

    from isoexec.compiler.source import ModelSource

    package, config = GEOMETRIES[geometry]
    model = ModelSource(package).build_meta(json.loads(config.read_text()))
    out, seen = [], set()
    for cls, name, kind, roles in KINDS:
        module = next((m for m in model.modules() if type(m).__name__ == cls), None)
        if module is None or name in seen:
            continue
        seen.add(name)
        axes = module.tensor_axes()
        members = tuple(Member(role, tuple(getattr(module, role).shape), axes[role]) for role in roles
                        if getattr(module, role, None) is not None)
        if len(members) != len(roles):
            continue  # an optional projection this geometry does not have
        dtype = "fp32" if getattr(module, roles[0]).dtype is torch.float32 else "bf16"
        out.append(Site(geometry, name, kind, members, dtype))
    return out


def column_ranges(member: Member, tp: int, rank: int) -> tuple[tuple[int, int], ...]:
    """The rows of ``member``'s logical weight (its output columns) this rank holds."""
    from isoexec.runtimes.parallel import colwise_ranges

    axis = member.axis
    return colwise_ranges(member.shape[0], tp, rank, stride=axis.stride, units=axis.units,
                          below_units="replicate")


def row_range(site: Site, tp: int, rank: int) -> tuple[int, int]:
    """The K slice a row-parallel rank holds."""
    from isoexec.runtimes.parallel import colwise_ranges

    (lo, hi), = colwise_ranges(site.K, tp, rank)
    return lo, hi


def local_shape(site: Site, tp: int, rank: int) -> tuple[int, int]:
    """``(K, N)`` of the one GEMM this rank issues for ``site``."""
    if site.kind == "row":
        lo, hi = row_range(site, tp, rank)
        return hi - lo, site.members[0].shape[0]
    if site.kind == "replicated":
        return site.K, sum(m.shape[0] for m in site.members)
    return site.K, sum(hi - lo for m in site.members for lo, hi in column_ranges(m, tp, rank))
