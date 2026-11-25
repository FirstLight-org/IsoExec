"""Normal-import SkyRL seam proof, not native CUDA/Transformer Engine qualification."""

import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest


@pytest.fixture(scope="module")
def framework(request):
    pytest.importorskip("skyrl.train.config")
    before = dict(sys.modules)
    patch = pytest.MonkeyPatch()

    class UnavailableTE:
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("CPU seam test cannot execute Transformer Engine")

    try:
        triton_spec = importlib.util.find_spec("triton")
        if triton_spec is not None:
            triton = importlib.import_module("triton")
            if "macos-stub" in getattr(triton, "__version__", ""):
                # The installed macOS import-only placeholder has no runtime;
                # correct Torch's CPU capability probe, never emulate a kernel.
                assert importlib.util.find_spec("triton.runtime") is None
                capability = importlib.import_module("torch.utils._triton")
                original = capability.has_triton_package
                for module in tuple(sys.modules.values()):
                    if module is not None and vars(module).get("has_triton_package") is original:
                        patch.setattr(module, "has_triton_package", lambda: False)
                with patch.context() as unavailable_runtime:
                    unavailable_runtime.setitem(sys.modules, "triton", None)
                    importlib.import_module("torch._dynamo")
                    importlib.import_module("torch._inductor.codecache")
        pytest.importorskip("triton.runtime")
        pytest.importorskip("megatron.core")
        importlib.import_module("megatron.core.extensions.transformer_engine")
        importlib.import_module("megatron.core.models.gpt")
        if importlib.util.find_spec("transformer_engine") is None:
            # Only three unavailable SDK annotation types are supplied. Real
            # SkyRL/MCore/Bridge bodies are normally imported without extraction.
            for name in ("transformer_engine", "transformer_engine.pytorch"):
                sdk = ModuleType(name)
                sdk.__path__ = []
                sdk.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                setattr(sdk, "__version__", "0.0.0")
                patch.setitem(sys.modules, name, sdk)
            te = sys.modules["transformer_engine.pytorch"]
            setattr(te, "LayerNorm", UnavailableTE)
            setattr(te, "TransformerLayer", UnavailableTE)
            setattr(te, "ops", SimpleNamespace(Sequential=UnavailableTE))
            setattr(sys.modules["transformer_engine"], "pytorch", te)
        module_name = getattr(request, "param", "skyrl.backends.skyrl_train.workers.megatron.megatron_worker")
        worker = pytest.importorskip(module_name)
        integration = importlib.import_module("isoexec.integrations.skyrl.megatron")
        source = Path(worker.__file__).resolve()
        assert source.parts[-2:] == ("megatron", module_name.rsplit(".", 1)[1] + ".py")
        if module_name.endswith("megatron_worker"):
            assert Path(worker.MegatronPolicyWorkerBase.optim_step.__code__.co_filename).resolve() == source
        yield worker, integration
    finally:
        patch.undo()
        # Clear SDK consumers so annotation-only HAVE_TE state cannot leak.
        # Torch's AOT/Inductor modules register process-global cache artifacts;
        # deleting and reimporting them can register the same artifact twice.
        # Keep those modules alive for subsequent capture tests in this process.
        for name in tuple(sys.modules):
            if name not in before and name.startswith(("skyrl.", "megatron.", "isoexec.integrations.skyrl.megatron",
                                                       "isoexec.integrations.skyrl.model_wrapper",
                                                       "isoexec.integrations.skyrl.workers")):
                module = sys.modules.pop(name)
                parent, _, child = name.rpartition(".")
                package = sys.modules.get(parent)
                if package is not None and getattr(package, child, None) is module:
                    delattr(package, child)


def _stepping_worker(cls, events, *, step=None):
    """A policy worker of ``cls`` holding just what SkyRL's ``optim_step`` reads."""
    megatron_config = SimpleNamespace(expert_nvfp4=False)
    optimizer = SimpleNamespace(
        step=step or (lambda: (events.append("native_step"), 2.0, None)),
        zero_grad=lambda: events.append("optimizer_zero"),
    )
    worker = object.__new__(cls)
    worker.cfg = SimpleNamespace(log_mfu=False, policy=SimpleNamespace(megatron_config=megatron_config))
    worker.optimizer = optimizer
    worker.scheduler = SimpleNamespace(step=lambda count: events.append("scheduler"))
    worker.model = SimpleNamespace(run_pending_grad_sync=lambda: events.append("finalize"))
    # SkyRL builds the strategy from the worker's ``strategy_cls``.
    worker.strategy = object.__new__(cls.strategy_cls)
    worker.actor_module = [SimpleNamespace(zero_grad_buffer=lambda: events.append("clear"))]
    worker._micro_batches_accumulated = 4
    return worker


@pytest.mark.parametrize("isoexec", [False, True])
def test_isoexec_strategy_finishes_gathers_inside_skyrls_optimizer_step(framework, monkeypatch, isoexec):
    worker_module, _ = framework
    from isoexec.integrations.skyrl import workers

    events = []
    cls = workers.IsoExecPolicyWorker if isoexec else worker_module.MegatronPolicyWorkerBase
    worker = _stepping_worker(cls, events)
    monkeypatch.setattr(workers, "finish_optimizer_step", lambda model: events.append("isoexec_finish"))
    assert worker.optim_step() == 2.0
    # The gathers finish before SkyRL's worker clears the DDP gradient buffers.
    stepped = ["finalize", "native_step", "scheduler", "optimizer_zero"]
    assert events == stepped + (["isoexec_finish"] if isoexec else []) + ["clear"]
    assert worker._micro_batches_accumulated == 0


def test_a_failed_optimizer_step_is_not_finished(framework, monkeypatch):
    from isoexec.integrations.skyrl import workers

    def failed():
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(workers, "finish_optimizer_step", lambda model: pytest.fail("no completion after failure"))
    worker = _stepping_worker(workers.IsoExecPolicyWorker, [], step=failed)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        worker.optim_step()


def test_real_inference_only_worker_refuses_before_finalize(framework):
    from isoexec.integrations.skyrl import workers

    worker = object.__new__(workers.IsoExecPolicyWorker)
    worker.cfg, worker.optimizer = SimpleNamespace(log_mfu=False), None
    with pytest.raises(RuntimeError, match="no optimizer"):
        worker.optim_step()
