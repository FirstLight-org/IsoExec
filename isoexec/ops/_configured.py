"""Bind native kernel entry points with immutable numerical settings and lazy imports."""

from __future__ import annotations

import importlib
import inspect

from isoexec.ops._resolve import refuse, registry, resolution


def configured(op, impl, *, site, module, symbol, inputs=None, kwargs=None, check=None, forbidden=()):
    """Pin settings in both the execution record and the callable's actual arguments.

    Kernel loading is deferred until execution. A caller cannot override a numerical setting
    after placement, including by supplying it positionally.
    """
    recorded, fixed = dict(inputs or {}), dict(kwargs or {})
    spec = registry().get_op(op).impls[impl]
    from isoexec.core.registry import OneOf

    for key, domain in spec.inputs.items():
        if key in recorded and isinstance(domain, OneOf) and recorded[key] not in domain:
            raise refuse(op, site=site, asked={key: recorded[key]}, on_offer=domain)

    fn = signature = None

    def call(*args, **kw):
        nonlocal fn, signature
        for key in forbidden:
            if key in kw:
                raise ValueError(f"{site}: {key} is not an execution option for {op}")
        if fn is None:
            fn = getattr(importlib.import_module(module), symbol)
            signature = inspect.signature(fn)
        bound = signature.bind_partial(*args, **kw)
        for key, value in fixed.items():
            if key in bound.arguments and bound.arguments[key] != value:
                raise ValueError(f"{site}: {key} is fixed at placement to {value!r}")
            bound.arguments[key] = value
        if check is not None:
            check(bound.arguments)
        return fn(*bound.args, **bound.kwargs)

    return resolution(op, impl, call, provider=module.rsplit('.', 1)[0].removeprefix('isoexec.'),
                      site=site, inputs=recorded)
