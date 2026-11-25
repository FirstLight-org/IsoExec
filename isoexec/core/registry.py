"""Declare operation implementations, recipe versions, input domains, and numerical equivalence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

DEBUG_PROOF_REVISION = "d2c49d5cfcbed55c0d4a3b4e6875ddab0e4f35e6"
DEBUG_PROOF_ROOT = f"https://github.com/zanderjiang/IsoExec/blob/{DEBUG_PROOF_REVISION}/"
TWIN_PROOFS: Dict[str, tuple] = {
    "mm.dense.segtree": ("isoexec/ops/mm/mm_sm100/tests/mm_sm100_battery_gpu.py",),
    "mm.dense.segtree.sm90": (
        "isoexec/ops/mm/mm_sm90/tests/test_mm_sm90_gpu.py",
        "isoexec/ops/mm/mm_sm90/tests/test_leaves_gpu.py",
        "isoexec/ops/mm/mm_sm90/tests/test_integration_gpu.py",
    ),
    "moe.expert_mlp.segment_tree_glu": ("isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py",),
    "attention.gqa_sm90_segsplit_v1": (
        "isoexec/ops/attention/gqa_sm90/tests/test_seam_gpu.py",
        "isoexec/ops/attention/gqa_sm90/tests/test_segsplit_gpu.py",
        "isoexec/ops/attention/gqa_sm90/tests/test_ops_gpu.py",
    ),
    "attention.gqa_sm100_segtree_v2": (
        "isoexec/ops/attention/gqa_sm100/tests/test_seam_gpu.py",
        "isoexec/ops/attention/gqa_sm100/tests/test_relations_gpu.py",
        "isoexec/ops/attention/gqa_sm100/tests/test_ops_gpu.py",
    ),
    "gdn.cpr": (
        "isoexec/ops/gdn/gdn_cpr_sm100/tests/test_identity_gpu.py",
        "isoexec/ops/gdn/gdn_cpr_sm100/tests/test_splits_and_slots_gpu.py",
        "isoexec/ops/gdn/gdn_cpr_sm100/tests/test_composition_gpu.py",
        "isoexec/ops/gdn/gdn_cpr_sm100/tests/test_mma_facts_gpu.py",
        DEBUG_PROOF_ROOT + "examples/validation/gate_vllm_gdn.py",
        "isoexec/ops/gdn/shared/tests/gdn_native_kernel_parity_test_gpu.py",
    ),
    "moe.route.sorted_topk": (
        "isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py",
        "isoexec/ops/moe/moe_sm90/tests/moe_router_fused_o2_gpu.py",
    ),
    "moe.expert_mlp.sm90_wgmma_v1": (
        "isoexec/ops/moe/moe_sm90/tests/test_sm90.py",
        "isoexec/ops/moe/moe_sm90/tests/test_integration.py",
        DEBUG_PROOF_ROOT + "examples/validation/gate_ep.py",
    ),
    "collectives.row_parallel.pik_tree": (
        "isoexec/ops/collectives/tests/test_tree_reduce_scatter_cpu.py",
        "isoexec/ops/collectives/tests/test_seq_parallel_cpu.py",
    ),
    "moe.combine.korder_fp32": (
        "isoexec/ops/moe/moe_sm100/tests/moe_relations_gpu.py",
        "isoexec/ops/moe/moe_sm100/tests/test_ep_korder_gpu.py",
        DEBUG_PROOF_ROOT + "examples/validation/gate_ep.py",
        "isoexec/ops/moe/moe_sm90/tests/test_combine_fold_cpu.py",
        "isoexec/ops/moe/moe_sm90/tests/test_owner_combine_unified_cpu.py",
    ),
    "logprobs.row_select": (
        "isoexec/ops/logprobs/tests/test_row_select_biteq_cpu.py",
        "isoexec/ops/logprobs/tests/logprobs_head_biteq_gpu.py",
        DEBUG_PROOF_ROOT + "examples/validation/gate_vllm_engine.py",
    ),
}


def runnable_batteries(twin_class: str) -> tuple:
    """Return local test paths for an equivalence class, excluding archived evidence links."""
    return tuple(p for p in TWIN_PROOFS[twin_class] if not p.startswith(DEBUG_PROOF_ROOT))


class RegistryError(ValueError):
    """Raised on any registry-vocabulary violation."""


class _Any:
    """An unconstrained input domain whose value is supplied by the caller."""

    __slots__ = ()

    def __repr__(self) -> str:  # appears verbatim in the refusal message
        return "ANY"


ANY = _Any()


class OneOf:
    """An input whose domain is an enumerated set."""

    __slots__ = ("values",)

    def __init__(self, *values) -> None:
        self.values = tuple(values)

    def __contains__(self, value) -> bool:
        return any(input_values_equal(v, value) for v in self.values)

    def __repr__(self) -> str:
        return f"OneOf{self.values!r}"


def input_values_equal(a, b) -> bool:
    """Equality for a declared domain member vs the value a binding supplied."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(input_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(input_values_equal(a[k], b[k]) for k in a)
    return a == b


@dataclass(frozen=True)
class ImplSpec:
    """One implementation of an op, at a pinned version."""

    impl_id: str
    # Bump the version whenever the arithmetic or rounding schedule changes.
    version: int
    supported_archs: frozenset
    inputs: Dict[str, object] = field(default_factory=dict)
    op: str = field(kw_only=True)
    twin_class: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if not self.supported_archs:
            raise RegistryError(f"impl {self.impl_id!r} must declare at least one supported arch")
        if self.twin_class and self.twin_class not in TWIN_PROOFS:
            raise RegistryError(
                f"impl {self.impl_id!r} declares twin_class {self.twin_class!r}, which TWIN_PROOFS does "
                f"not name a battery for; a twin claim is a test that passes, not a string "
                f"(declared classes: {sorted(TWIN_PROOFS)})"
            )
        bad = sorted(k for k, v in self.inputs.items() if not _is_legal_declaration(v))
        if bad:
            raise RegistryError(
                f"impl {self.impl_id!r} declares input(s) {bad} with a value that is neither ANY nor "
                f"OneOf(...); an input is a key a caller supplies AND the domain it may supply it in. "
                f"A bare literal states neither: if the value is the kernel's, it is not an input and "
                f"belongs beside the kernel, and if a caller does supply it, write OneOf(value)"
            )


def _is_legal_declaration(value) -> bool:
    """Accept only explicit ANY or OneOf input domains."""
    return value is ANY or isinstance(value, OneOf)


@dataclass
class OpSpec:
    """Registry grouping of implementations for one logical operation."""

    name: str
    impls: Dict[str, ImplSpec] = field(default_factory=dict)


class Registry:
    """Operation declarations grouped by operation and implementation ID."""

    def __init__(self) -> None:
        self._ops: Dict[str, OpSpec] = {}

    def add(self, *impls: ImplSpec) -> "Registry":
        for impl in impls:
            if not impl.op:
                raise RegistryError(
                    f"impl {impl.impl_id!r} declares no op; a flat declaration names the op it serves, "
                    f"since nothing else in the list says what this impl implements"
                )
            op = self._ops.get(impl.op)
            if op is None:
                op = self._ops[impl.op] = OpSpec(name=impl.op)
            if impl.impl_id in op.impls:
                raise RegistryError(f"op {op.name!r} already has impl {impl.impl_id!r}")
            op.impls[impl.impl_id] = impl
        return self

    @classmethod
    def from_impls(cls, impls: Iterable[ImplSpec]) -> "Registry":
        """A registry holding exactly ``impls`` -- the whole build, for one package or for a test."""
        return cls().add(*impls)

    def get_op(self, name: str) -> OpSpec:
        if name not in self._ops:
            raise RegistryError(f"unknown op {name!r}")
        return self._ops[name]

    def has_op(self, name: str) -> bool:
        return name in self._ops

    @property
    def ops(self) -> Dict[str, OpSpec]:
        return dict(self._ops)

    def assert_twin_classes_single_op(self) -> None:
        """Require each equivalence class to contain implementations of one operation only."""
        members: dict[str, dict[str, list[str]]] = {}
        for op in self._ops.values():
            for impl in op.impls.values():
                if impl.twin_class:
                    members.setdefault(impl.twin_class, {}).setdefault(op.name, []).append(impl.impl_id)
        for cls, by_op in sorted(members.items()):
            if len(by_op) > 1:
                spread = "; ".join(f"{name}: {sorted(ids)}" for name, ids in sorted(by_op.items()))
                raise RegistryError(
                    f"twin class {cls!r} spans {len(by_op)} op ids ({spread}); a twin class is the set of "
                    f"impls OF ONE OP proven bitwise-equal, and TWIN_PROOFS[{cls!r}] is a battery for that "
                    f"claim. An impl that also computes a neighbouring op says so per site, in the "
                    f"binding's `owns`, not by joining that op into this class"
                )

    def lint(self) -> List[str]:
        """Return cross-declaration consistency errors; ImplSpec validates individual records."""
        problems: List[str] = []
        try:
            self.assert_twin_classes_single_op()
        except RegistryError as exc:
            problems.append(str(exc))
        return problems
