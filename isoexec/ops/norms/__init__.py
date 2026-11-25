"""Architecture-neutral norms API; importing the facade does not load kernel providers."""

from isoexec.ops._dispatch import facade as _facade

_EXPORTS = {
    "RECIPE_VERSION": ("ops", "RECIPE_VERSION"),
    "add_rms_norm": ("ops", "add_rms_norm"),
    "gated_rms_norm": ("ops", "gated_rms_norm"),
    "install_native_rmsnorm_memo": ("native_rmsnorm_memo", "install_native_rmsnorm_memo"),
    "integration": ("integration", None),
    "ops": ("ops", None),
    "recipe": ("ops", "recipe"),
    "rms_norm": ("ops", "rms_norm"),
    "supported_width": ("ops", "supported_width"),
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = _facade("norms", _EXPORTS, globals())
