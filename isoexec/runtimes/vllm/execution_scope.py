"""The runtime's hold on the execution scopes of a served model's collectives.

A placement-bound component may bind a collective whose transport is serialized by its owning runtime (the
symmetric-memory tree reducer, ``collectives/detreduce_sm100``): every reduction, eager or replayed from a
captured graph, runs inside the transport's execution scope, and a graph records the transport only when
its capture runs inside the scope's capture lifecycle. ``VllmModel.execution_scope`` enters the scopes its
components declare (``engine_execution_scopes``); this module makes vLLM's V2 model runner hold them:

* ``capture_model``: the capture lifecycle (``capture=True``) around every warmup and capture, and an inner
  scope entered right after vLLM's ``graph_capture`` switches to its capture stream, before the first
  ``torch.cuda.graph`` begins (the transport adopts the capture stream while no capture is running);
* ``execute_model``: the runtime scope around every eager forward and every graph replay.

The patch is installed once per process when a model with such scopes is built, and only acts for a runner
whose model has ``execution_scope``; other models run as before.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps

_installed: list = []


def _scope(runner, *, capture: bool = False):
    model = getattr(runner, "model", None)
    scope = getattr(model, "execution_scope", None)
    return scope(capture=capture) if callable(scope) else nullcontext()


def wrap_execute(original):
    @wraps(original)
    def execute_model(runner, *args, **kwargs):
        with _scope(runner):
            return original(runner, *args, **kwargs)

    execute_model._isoexec_orig = original
    return execute_model


def wrap_capture(original, capture_module):
    @wraps(original)
    def capture_model(runner, *args, **kwargs):
        with _scope(runner, capture=True):
            inner = capture_module.graph_capture

            @contextmanager
            def graph_capture(*capture_args, **capture_kwargs):
                with inner(*capture_args, **capture_kwargs) as context, _scope(runner):
                    yield context

            capture_module.graph_capture = graph_capture
            try:
                return original(runner, *args, **kwargs)
            finally:
                capture_module.graph_capture = inner

    capture_model._isoexec_orig = original
    return capture_model


def install_runner_execution_scope() -> bool:
    """Make vLLM's V2 model runner hold its model's execution scopes (idempotent). Returns False when the V2
    runner is not available (its models then enter the scopes per call and capture the scope-free form)."""
    if _installed:
        return True
    try:
        from vllm.v1.worker.gpu import cudagraph_utils
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return False
    GPUModelRunner.execute_model = wrap_execute(GPUModelRunner.execute_model)
    GPUModelRunner.capture_model = wrap_capture(GPUModelRunner.capture_model, cudagraph_utils)
    _installed.append(GPUModelRunner)
    return True


__all__ = ["install_runner_execution_scope", "wrap_capture", "wrap_execute"]
