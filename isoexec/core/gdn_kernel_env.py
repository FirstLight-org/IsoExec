"""The GDN delta-rule kernel selector: one vocabulary, one default, one parser.

Both the declaration site (``models/qwen3_5.build``) and the read sites (``ops/gdn/shared/gdn_ops``)
must parse ``ISOEXEC_GDN_KERNEL`` identically, so parsing lives here.
"""

from __future__ import annotations

import os

KERNEL_ENV = "ISOEXEC_GDN_KERNEL"
# Trainer-only ablation override: the contract pins ONE gdn.core kernel for all four sites, so a
# contract build refuses when this asks for a trainer/engine split.
TRAINER_KERNEL_ENV = "ISOEXEC_GDN_TRAINER_KERNEL"

# "cpr": chunkwise-parallel-recurrent (parallel within a chunk, recurrent across chunk boundaries).
KERNELS = ("cpr", "recurrent")
DEFAULT_KERNEL = "recurrent"

# Retired spellings, refused by name so an old launch script fails loudly instead of falling back.
_RENAMED = {"chunk_synced": "cpr"}


def parse_gdn_kernel(value, env: str = KERNEL_ENV) -> str:
    """Normalize one env value to a kernel name. Unknown refuses; empty/absent gives the default."""
    name = (value or "").strip().lower()
    if not name:
        return DEFAULT_KERNEL
    if name not in KERNELS:
        successor = f" {name!r} was renamed to {_RENAMED[name]!r}." if name in _RENAMED else ""
        raise ValueError(
            f"{env}={value!r} names no delta-rule kernel; the vocabulary is {list(KERNELS)}.{successor} "
            f"Refusing rather than falling back: a typo that falls back is the same typo on both "
            f"runtimes, so both derive the same wrong composition and the handshake agrees."
        )
    return name


def gdn_kernel_mode() -> str:
    """The delta-rule kernel this process runs, from the environment. Read at call time."""
    return parse_gdn_kernel(os.environ.get(KERNEL_ENV), KERNEL_ENV)


def requested_gdn_kernel() -> str | None:
    """The kernel the environment names explicitly, or None when unset.

    Package selection reads this rather than :func:`gdn_kernel_mode`: an arch whose default package is
    not the parser's default keeps its default until a deployment names another kernel.
    """
    raw = os.environ.get(KERNEL_ENV, "")
    return parse_gdn_kernel(raw, KERNEL_ENV) if raw.strip() else None


def gdn_trainer_kernel_override() -> str | None:
    """The trainer-only kernel override, or None when unset (the supported configuration)."""
    raw = os.environ.get(TRAINER_KERNEL_ENV, "")
    return parse_gdn_kernel(raw, TRAINER_KERNEL_ENV) if raw.strip() else None
