"""Registry, binding, architecture, and flag metadata. Lazy exports keep component construction
free of architecture probes.
"""

from importlib import import_module

# Lazy export map keeps package imports free of architecture probes.
_EXPORTS = {
    "ANY": "registry",
    "ARCH": "arch",
    "FLAGS": "flags",
    "Flag": "flags",
    "ImplSpec": "registry",
    "NUMERICS_SWITCHES": "flags",
    "OneOf": "registry",
    "OpSpec": "registry",
    "Registry": "registry",
    "RegistryError": "registry",
    "SIDE_LOCAL_SWITCHES": "flags",
    "TWIN_PROOFS": "registry",
    "actor_forwarding_list": "flags",
    "detect_arch": "arch",
    "malformed_sides": "flags",
    "runnable_batteries": "registry",
}


__all__ = [
    "ARCH",
    "detect_arch",
    "Registry",
    "OpSpec",
    "ImplSpec",
    "ANY",
    "OneOf",
    "RegistryError",
    "TWIN_PROOFS",
    "runnable_batteries",
    "FLAGS",
    "Flag",
    "NUMERICS_SWITCHES",
    "SIDE_LOCAL_SWITCHES",
    "actor_forwarding_list",
    "malformed_sides",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
