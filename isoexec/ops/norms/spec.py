"""Architecture-neutral norm declarations, resolving the selected package lazily when IMPLS is read."""

from typing import TYPE_CHECKING

from isoexec.ops._dispatch import family_impls

if TYPE_CHECKING:  # never executed: it names what the lazy ``__getattr__`` below otherwise hides.
    from isoexec.core.registry import ImplSpec

    IMPLS: tuple[ImplSpec, ...]

__all__ = ("IMPLS",)


__getattr__ = family_impls(__name__, "norms")
