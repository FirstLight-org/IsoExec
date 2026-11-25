"""Validate dependencies, model facts, weight bytes, and numerical agreement. Failures raise Refusal
with the mismatched field; no check reads environment settings.
"""

from __future__ import annotations


class Refusal(RuntimeError):
    """An admission failure whose message identifies the mismatched field."""


__all__ = ["Refusal"]
