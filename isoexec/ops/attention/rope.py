"""Architecture-neutral RoPE API with lazy kernel providers.
Attention owns the operations; the rope dispatch entry selects their architecture-specific package."""

from isoexec.ops._dispatch import facade as _facade

_EXPORTS = {
    "RECIPE_VERSION": ("ops", "RECIPE_VERSION"),
    "RopeTable": ("ops", "RopeTable"),
    "build_table": ("ops", "build_table"),
    "integration": ("integration", None),
    "ops": ("ops", None),
    "qk_norm_rope": ("ops", "qk_norm_rope"),
    "reference": ("reference", None),
    "rope": ("ops", "rope"),
    "rope_apply_eager": ("rope_reference", "rope_apply_eager"),
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = _facade("rope", _EXPORTS, globals())
