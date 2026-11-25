"""Architecture-neutral short-convolution API with lazy kernel providers.
The GDN family owns its configuration and the conv dispatch entry selects the kernel package."""

from isoexec.ops._dispatch import facade as _facade

_EXPORTS = {
    "RECIPE_VERSION": ("ops", "RECIPE_VERSION"),
    "bwd_plan": ("ops", "bwd_plan"),
    "conv": ("ops", "conv"),
    "fwd_plan": ("ops", "fwd_plan"),
    "integration": ("integration", None),
    "ops": ("ops", None),
    "update": ("ops", "update"),
    "varlen": ("ops", "varlen"),
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = _facade("conv", _EXPORTS, globals())
