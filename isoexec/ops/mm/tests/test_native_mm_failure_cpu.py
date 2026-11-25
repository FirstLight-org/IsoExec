"""Native backward failures must not silently select another numeric provider."""

import pytest
import torch
from isoexec.ops.mm import mm_fwd_scope as scope


def _case(operation):
    if operation == "bmm":
        return scope.make_scoped_bmm_impl(), (torch.empty(2, 3, 4), torch.empty(2, 4, 5)), {}
    mm, addmm = scope.make_scoped_impls()
    a, b = torch.empty(3, 4), torch.empty(4, 5)
    if operation == "mm":
        return mm, (a, b), {}
    return addmm, (torch.empty(5), a, b), {"alpha": 2, "beta": 3}


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
def test_native_failure_preserves_original_exception(monkeypatch, operation):
    monkeypatch.setattr(scope, "use_batch_invariant", lambda: False)
    failure = RuntimeError("injected native backend failure")

    def fail(*args, **kwargs):
        raise failure

    def forbidden(*args, **kwargs):
        pytest.fail("native failure selected an alternate numerical provider")

    monkeypatch.setattr(scope, "_cublas_" + operation, fail)
    monkeypatch.setattr(scope.V, operation + "_batch_invariant", forbidden)
    function, args, kwargs = _case(operation)
    with pytest.raises(RuntimeError, match="injected native backend failure") as caught:
        function(*args, **kwargs)
    assert caught.value is failure


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
@pytest.mark.parametrize("forward", [False, True])
def test_successful_routing_and_arguments_unchanged(monkeypatch, operation, forward):
    monkeypatch.setattr(scope, "use_batch_invariant", lambda: forward)
    sentinel = object()
    calls = []

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def forbidden(*args, **kwargs):
        pytest.fail("wrong numerical provider selected")

    monkeypatch.setattr(scope, "_cublas_" + operation, forbidden if forward else selected)
    monkeypatch.setattr(scope.V, operation + "_batch_invariant", selected if forward else forbidden)
    function, args, kwargs = _case(operation)
    assert function(*args, **kwargs) is sentinel
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][0], args))
    assert calls[0][1] == kwargs
