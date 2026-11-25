"""Architecture-neutral GQA attention API with lazy kernel providers.
Attention owns the op; the gqa dispatch entry selects its architecture-specific package (gqa_sm90,
gqa_sm100). The two packages' ``ops.varlen_attention`` signatures differ; the arch-neutral entry
points are ``trainer_core`` and ``paged_core``."""

from isoexec.ops._dispatch import facade as _facade

_EXPORTS = {
    "PAGED_IMPL": ("integration", "PAGED_IMPL"),
    "TRAINER_IMPL": ("integration", "TRAINER_IMPL"),
    "integration": ("integration", None),
    "ops": ("ops", None),
    "paged_core": ("integration", "paged_core"),
    "trainer_core": ("integration", "trainer_core"),
    "varlen_attention": ("ops", "varlen_attention"),
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = _facade("gqa", _EXPORTS, globals())
