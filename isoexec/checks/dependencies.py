"""Check supported dependency ranges and report versions covered by numerical tests. Accepted
versions may exceed the verified set; widening support requires reviewing the library interfaces
used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from isoexec.checks import Refusal


@dataclass(frozen=True)
class Dependency:
    """One library IsoExec's numerics rest on."""

    name: str
    # The surface IsoExec touches. This is what a human re-reads when widening the range.
    uses: str
    # Inclusive bounds IsoExec consents to run on, as comparable tuples.
    accepted: tuple[tuple[int, ...], tuple[int, ...]]
    # Where the test gate has actually been run. A subset of ``accepted``, often a single point.
    verified: tuple[tuple[int, ...], ...]
    # Reads the running version, or returns None when the library is absent.
    read: Callable[[], tuple[int, ...] | None]
    # True when IsoExec cannot run at all without it.
    required: bool = True
    # When set, an out-of-range version is reported rather than refused, for the reason given:
    # only some compositions reach this library's numerics.
    advisory: str | None = None
    # Numerical dependencies require exact versions; ABI dependencies may admit compatible
    # ranges.
    kind: str = "abi"


def _torch_version() -> tuple[int, ...] | None:
    try:
        import torch
    except ImportError:
        return None
    return tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2] if p.isdigit())


def _nccl_version() -> tuple[int, ...] | None:
    try:
        import torch

        # NCCL config layouts are versioned by minor release; match the bounds’ tuple length.
        return tuple(torch.cuda.nccl.version())[:2]
    except BaseException:  # noqa: BLE001 -- absent CUDA is a legitimate answer here
        return None


def _patch_version(module: str, *, distribution: str | None = None) -> Callable[[], tuple[int, ...] | None]:
    """Read exact patch versions from installed metadata without importing the compiler DSL.
    Numerical pins distinguish patch releases.
    """

    def read() -> tuple[int, ...] | None:
        from importlib import metadata

        try:
            raw = metadata.version(distribution or module)
        except BaseException:  # noqa: BLE001
            return None
        parts = raw.split("+")[0].split(".")[:3]
        return tuple(int(p) for p in parts if p.isdigit()) or None

    return read


def _module_version(module: str) -> Callable[[], tuple[int, ...] | None]:
    def read() -> tuple[int, ...] | None:
        try:
            mod = __import__(module)
        except BaseException:  # noqa: BLE001
            return None
        raw = getattr(mod, "__version__", "")
        parts = raw.split("+")[0].split(".")[:2]
        return tuple(int(p) for p in parts if p.isdigit()) or None

    return read


# The libraries whose behaviour reaches IsoExec's numbers. Ordered by how badly a mismatch bites.
DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency(
        name="nccl",
        uses="ncclConfig_t's first seven fields (through maxCTAs), written by "
             "runtimes/vllm/nccl_channels.py to bound channels per communicator",
        accepted=((2, 26), (2, 30)),
        verified=((2, 29), (2, 30)),
        read=_nccl_version,
    ),
    Dependency(
        name="torch",
        uses="aten op semantics, the cuBLAS workspace, and the autograd graph the trainer's "
             "reductions ride on",
        accepted=((2, 13), (2, 14)),
        verified=((2, 14),),
        read=_torch_version,
    ),
    Dependency(
        name="vllm",
        uses="the engine surface: sampler hooks, model loader, attention backend",
        accepted=((0, 33, 0), (0, 33, 0)),
        verified=((0, 33, 0),),
        read=_patch_version("vllm"),
        required=False,
    ),
    Dependency(
        name="nvidia-cutlass-dsl",
        uses="the reduction behind torch.nn.functional.rms_norm, which the RMSNorm references in "
             "ops/norms/norms_sm90/reference.py compute, and the compiler of the vendored FlashAttention-4 SM100 "
             "attention backward (ops/_vendor/flash_attn_cute)",
        # 4.5.2 was the measured point (4.6.2 changed width-128 reduction rounding). vLLM 0.33's
        # flashinfer 0.7.2 does not import under 4.5.2 (no cute.FastDivmodDivisorV2), so the lock's
        # 4.8.0 is installed. torch 2.14 enables its _native CuTe DSL overrides only for cutedsl
        # 4.4.1/4.4.2/4.5.2, so under 4.8.0 eager rms_norm runs ATen's kernel instead.
        # The vendored FA4 SM100 backward (#161) was qualified on B200 at 4.5.2; with the CuTe DSL
        # 4.8 API edits in its PROVENANCE.md, its out/dQ/dK/dV at 4.8.0 equal the 4.5.2 bits at head
        # dims 128 and 256 (it is outside the cross-host contract; see advisory).
        accepted=((4, 8, 0), (4, 8, 0)),
        verified=((4, 8, 0),),
        read=_patch_version("nvidia_cutlass_dsl", distribution="nvidia-cutlass-dsl"),
        required=False,
        kind="numerics",
        advisory="only the eager torch RMSNorm expressions reach this reduction (the shipped "
                 "compositions bind the arch norm kernels), and the SM100 attention backward it "
                 "compiles is outside the cross-host contract",
    ),
    Dependency(
        name="triton",
        uses="the sm90 kernels' JIT; its codegen decides the bits those kernels produce",
        accepted=((3, 8), (3, 8)),
        verified=((3, 8),),
        read=_module_version("triton"),
        required=False,
    ),
)


def _refuse_range_on_numerics() -> None:
    """A ``numerics`` dependency declared as a range is a bug in this table, not a deployment error."""
    for dep in DEPENDENCIES:
        if dep.kind == "numerics" and dep.accepted[0] != dep.accepted[1]:
            raise Refusal(
                f"dependencies.{dep.name}: declared kind='numerics' with a RANGE "
                f"{dep.accepted}. A library IsoExec's bits depend on is interchangeable across "
                f"exactly zero versions; state one, or justify the kind."
            )


def survey() -> list[dict]:
    """Every dependency's running version beside what is accepted and verified. Never raises."""
    _refuse_range_on_numerics()
    rows = []
    for dep in DEPENDENCIES:
        found = dep.read()
        low, high = dep.accepted
        rows.append({
            "name": dep.name,
            "found": found,
            "accepted": dep.accepted,
            "verified": dep.verified,
            "in_accepted": found is not None and low <= found <= high,
            "in_verified": found in dep.verified,
            "required": dep.required,
        })
    return rows


def check(*, host: str = "") -> list[dict]:
    """Refuse a dependency outside its accepted range, BEFORE a host spends minutes starting up.

    The refusal names the surface, not just the number, because a deployment that pins torch cannot
    act on "requires 2.14"; it can act on knowing which behaviour IsoExec relies on.
    """
    rows = survey()
    where = f" ({host})" if host else ""
    for row, dep in zip(rows, DEPENDENCIES):
        if row["found"] is None:
            if dep.required:
                raise Refusal(f"dependency.{dep.name}: not importable{where}, and IsoExec needs it for {dep.uses}")
            continue
        if not row["in_accepted"] and dep.advisory:
            print(
                f"[ISOEXEC-DEPENDENCIES] {dep.name}={_fmt(row['found'])} is outside "
                f"{_fmt(dep.accepted[0])}..{_fmt(dep.accepted[1])}; not refused because "
                f"{dep.advisory}. IsoExec uses {dep.uses}.",
                flush=True,
            )
            continue
        if not row["in_accepted"]:
            low, high = dep.accepted
            raise Refusal(
                f"dependency.{dep.name}: {_fmt(row['found'])} is outside the accepted range "
                f"{_fmt(low)}..{_fmt(high)}{where}. IsoExec uses {dep.uses}. Confirm that surface is "
                f"unchanged in this release, then widen DEPENDENCIES in checks/dependencies.py -- "
                f"widening it because a run failed is how a silent bit difference gets in."
            )
    return rows


def _fmt(version: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in version)


def unverified(rows: list[dict] | None = None) -> list[str]:
    """Accepted-but-not-gate-verified dependencies: the gap a receipt should carry."""
    return [r["name"] for r in (rows if rows is not None else survey())
            if r["found"] is not None and r["in_accepted"] and not r["in_verified"]]


__all__ = ["DEPENDENCIES", "Dependency", "check", "survey", "unverified"]
