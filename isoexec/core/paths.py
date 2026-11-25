"""Host locations resolved from the environment, never hard-coded: per-user caches and the CUDA toolkit.

Caches: an explicit env override, else ``$XDG_CACHE_HOME`` (else ``~/.cache``).

The managed environment sets ``XDG_CACHE_HOME`` inside its root, so every default here lands in the
managed tree without per-cache configuration. Paths are computed, never created.
"""

from __future__ import annotations

import os
from pathlib import Path


def user_cache_dir(*parts: str, env: str | None = None) -> Path:
    """Return ``$<env>`` when set and non-empty, else ``$XDG_CACHE_HOME/<parts...>`` (else ``~/.cache/<parts...>``).

    The override names the complete location: ``parts`` are not appended to it.
    """
    if env:
        override = os.environ.get(env)
        if override:
            return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").joinpath(*parts)


def pik_cache_dir() -> Path:
    """PIK's cache root (generated kernels, autotune ledger, cuBLASLt extension): ``$PIK_CACHE`` or ``<cache>/pik``."""
    return user_cache_dir("pik", env="PIK_CACHE")


