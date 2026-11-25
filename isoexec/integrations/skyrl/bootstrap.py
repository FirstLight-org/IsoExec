"""Driver configuration for the opt-in SkyRL integration."""

from __future__ import annotations

from typing import Any

from .config import resolve


def validate_config(cfg: Any) -> Any:
    """Validate and resolve IsoExec at the end of SkyRL's ``validate_cfg``, in the driver."""
    from .full_distribution_config import validate_full_distribution_config

    validate_full_distribution_config(cfg)
    return resolve(cfg)
