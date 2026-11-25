"""Architecture-neutral GDN declarations combining the selected recurrence and short-convolution packages.
IMPLS resolves the architecture lazily when read."""

from typing import TYPE_CHECKING

from isoexec.ops._dispatch import family_impls

if TYPE_CHECKING:  # never executed: it names what the lazy ``__getattr__`` below otherwise hides.
    from isoexec.core.registry import ImplSpec

    IMPLS: tuple[ImplSpec, ...]

__all__ = ("IMPLS",)


__getattr__ = family_impls(__name__, "gdn", "conv")
