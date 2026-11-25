"""IsoExec's Megatron actors on SkyRL's real trainer, worker and experiment classes."""

import importlib.util
import subprocess
import sys
from typing import Any, cast

import pytest

from isoexec.integrations.skyrl.tests.test_optimizer_update_cpu import framework  # noqa: F401


def test_registered_training_method_survives_fresh_process(framework, tmp_path):  # noqa: F811
    import ray.cloudpickle

    from isoexec.integrations.skyrl.workers import PolicyWorker

    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is None or te_spec.origin is None:
        pytest.skip("Fresh-process actor imports require the installed native SDK, not annotation stubs")

    # A same-process pickle round trip reuses cloudpickle's class identity cache
    # and misses a local base class captured separately by Ray's method wrapper.
    payload = tmp_path / "policy-actor.pkl"
    serializer = cast(Any, ray.cloudpickle)
    payload.write_bytes(serializer.dumps(cast(Any, PolicyWorker).__ray_metadata__.modified_class))
    script = '''
from pathlib import Path
import sys
import ray.cloudpickle
from isoexec.integrations.skyrl import aux_batch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase

actor_class = ray.cloudpickle.loads(Path(sys.argv[1]).read_bytes())
worker = object.__new__(actor_class)
for enabled in (False, True):
    worker._isoexec_aux_batches_installed = enabled
    def native(self, value, *, fail=False):
        assert self is worker
        assert aux_batch._ACTIVE.get() == ([] if enabled else None)
        if fail:
            raise RuntimeError("native training failed")
        return value
    MegatronPolicyWorkerBase.forward_backward = native
    assert worker.forward_backward(7) == 7
    assert aux_batch._ACTIVE.get() is None
    try:
        worker.forward_backward(9, fail=True)
    except RuntimeError as error:
        assert str(error) == "native training failed"
    else:
        raise AssertionError("native exception was swallowed")
    assert aux_batch._ACTIVE.get() is None
'''
    result = subprocess.run([sys.executable, "-c", script, str(payload)], check=False, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_isoexec_actors_are_skyrls_megatron_workers(framework):  # noqa: F811
    import ray.cloudpickle

    native, _ = framework
    from isoexec.integrations.skyrl import workers

    for actor, base in ((workers.PolicyWorker, native.MegatronPolicyWorkerBase),
                        (workers.RefWorker, native.MegatronRefWorkerBase)):
        serializer = cast(Any, ray.cloudpickle)
        restored = serializer.loads(serializer.dumps(actor))
        assert issubclass(restored.__ray_metadata__.modified_class, base)


@pytest.mark.parametrize("section", ["policy", "ref"])
def test_isoexec_workers_fill_the_places_skyrls_init_model_leaves_to_a_subclass(framework, monkeypatch, section):  # noqa: F811
    from types import SimpleNamespace

    from isoexec.integrations.skyrl import workers
    from isoexec.runtimes.megatron import residency

    native, _ = framework
    cls = workers.IsoExecPolicyWorker if section == "policy" else workers.IsoExecRefWorker
    base = native.MegatronPolicyWorkerBase if section == "policy" else native.MegatronRefWorkerBase
    assert cls._isoexec_section_name == section
    # The strategy completes the optimizer step, so only the policy replaces SkyRL's.
    assert (cls.strategy_cls is workers.IsoExecMegatronStrategy) == (section == "policy")
    assert cls.init_model is base.init_model or section == "policy"

    worker = object.__new__(cls)
    worker.cfg = SimpleNamespace(policy=SimpleNamespace(inference_only_init=False, megatron_config="policy-mc"),
                                 ref=SimpleNamespace(megatron_config="ref-mc"))
    configured = []
    monkeypatch.setattr(workers, "init_configs", lambda w, path, mc: configured.append((w, path, mc)))
    worker.init_configs("/checkpoint", "megatron-config", {}, {}, bf16=True, language_model_only=True)
    assert configured == [(worker, "/checkpoint", "megatron-config")]

    built = []
    monkeypatch.setattr(workers, "build_module", lambda w, **kwargs: built.append(kwargs) or ["module"])
    monkeypatch.setattr(residency, "baseline", lambda: 4096)
    assert worker.make_megatron_module(wrap_with_ddp=False, ddp_config=None, bf16=True) == ["module"]
    assert built == [{"wrap_with_ddp": False, "ddp_config": None, "lora_config": None, "bf16": True}]
    assert worker._isoexec_stack_baseline == 4096

    worker.actor_module, worker.optimizer, worker.policy_loss_fn = ["module"], None, "loss"
    wrapped = []
    monkeypatch.setattr(workers, "build_wrapper", lambda **kwargs: wrapped.append(kwargs) or "wrapper")
    assert worker._build_model_wrapper() == "wrapper"
    expected = dict(config=worker.cfg, actor_module=["module"], is_vlm=False)
    if section == "policy":
        worker._isoexec_optimizer = "optimizer"
        assert worker._build_optimizer() == "optimizer"
        expected.update(megatron_config="policy-mc", training=True, actor_optimizer=None, policy_loss_fn="loss")
    else:
        expected.update(megatron_config="ref-mc", training=False)
    assert wrapped == [expected]


def test_isoexec_worker_offload_gives_back_local_memory(framework, monkeypatch):  # noqa: F811
    native, _ = framework
    from isoexec.integrations.skyrl import workers
    from isoexec.runtimes.megatron import residency

    events = []
    monkeypatch.setattr(native.MegatronPolicyWorkerBase, "offload_to_cpu",
                        lambda self, **kwargs: events.append(("offload", kwargs)))
    monkeypatch.setattr(residency, "release_local_memory", lambda limit, reason: events.append(("release", limit)))
    worker = object.__new__(workers.IsoExecPolicyWorker)
    worker._isoexec_stack_baseline = 1024
    worker.offload_to_cpu(offload_optimizer=False, offload_model=True)
    assert events == [("offload", {"offload_optimizer": False, "offload_model": True}), ("release", 1024)]
