"""Native SkyRL padding and iteration retain row identity without a loss-mask heuristic."""

import sys
from typing import cast

import pytest
import torch

from isoexec.integrations.skyrl import aux_batch
from isoexec.integrations.skyrl.tests.test_optimizer_update_cpu import framework  # noqa: F401


def test_padding_identity_survives_native_token_batching(framework, monkeypatch):  # noqa: F811
    batches = pytest.importorskip("skyrl.backends.skyrl_train.training_batch")
    iterators = pytest.importorskip("skyrl.backends.skyrl_train.workers.worker_utils")
    original = batches.pad_training_input_batch
    for name, module in tuple(sys.modules.items()):
        if module is not None and name.startswith("skyrl."):
            for attribute, value in tuple(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, attribute, value)
    monkeypatch.setattr(iterators.BaseBatchIterator, "batch_to_experience",
                        staticmethod(iterators.BaseBatchIterator.batch_to_experience))
    aux_batch.install_padding_tracking()
    aux_batch.install_padding_tracking()  # Repeated config resolution must not add a second wrapper.
    real = batches.TrainingInputBatch({
        "sequences": torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]),
        # All three are real, even when the policy masks their responses.
        "loss_mask": torch.zeros(3, 2),
    })
    real.metadata = {"response_length": 2}
    padded = batches.pad_training_input_batch(real, 2)
    assert padded[aux_batch.VALID_ROWS].tolist() == [True, True, True, False, False]
    nested = batches.pad_training_input_batch(padded, 1)
    assert nested[aux_batch.VALID_ROWS].tolist() == [True, True, True, False, False, False]
    # DP slicing preserves exactly the same tensor field before local packing.
    assert [part[aux_batch.VALID_ROWS].tolist() for part in nested.chunk(3)] == [
        [True, True, True], [False, False, False]]

    orders = []

    def native_forward_backward(data):
        iterator = iterators.TokenBasedBatchIterator(data, max_tokens_per_microbatch=4)
        orders.append(iterator._microbatches)
        microbatches = []
        for batch in iterator:
            experience = iterators.BaseBatchIterator.batch_to_experience(batch)
            microbatches.append({"attention_mask": experience.attention_mask})
        max_rows = max(batch["attention_mask"].shape[0] for batch in microbatches)
        for batch in microbatches:
            rows = batch["attention_mask"].shape[0]
            batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.zeros(max_rows - rows, 3)))
        return aux_batch.annotate_microbatches(microbatches)

    from isoexec.integrations.skyrl.workers import IsoExecPolicyWorker
    native, _ = framework
    monkeypatch.setattr(native.MegatronPolicyWorkerBase, "forward_backward",
                        lambda self, data: native_forward_backward(data))
    worker = object.__new__(IsoExecPolicyWorker)
    aux_batch.install_worker(worker)
    aux_batch.install_worker(worker)
    for data in (padded, real):
        # Ray captures the class method; an instance wrapper must not be needed.
        result = cast(list[dict[str, torch.Tensor]], type(worker).forward_backward(worker, data))
        for microbatch, indices in zip(result, orders[-1], strict=True):
            expected = [index < 3 for index in indices]
            expected += [False] * (microbatch["attention_mask"].shape[0] - len(indices))
            assert microbatch[aux_batch.VALID_ROWS].tolist() == expected
        assert sum(int(batch[aux_batch.VALID_ROWS].sum()) for batch in result) == 3
        assert aux_batch._ACTIVE.get() is None


def test_native_training_exception_clears_row_metadata(framework, monkeypatch):  # noqa: F811
    iterators = pytest.importorskip("skyrl.backends.skyrl_train.workers.worker_utils")
    monkeypatch.setattr(iterators.BaseBatchIterator, "batch_to_experience",
                        staticmethod(iterators.BaseBatchIterator.batch_to_experience))

    def failed(*args, **kwargs):
        assert aux_batch._ACTIVE.get() == []
        raise RuntimeError("native schedule failed")

    from isoexec.integrations.skyrl.workers import IsoExecPolicyWorker
    native, _ = framework
    monkeypatch.setattr(native.MegatronPolicyWorkerBase, "forward_backward", failed)
    worker = object.__new__(IsoExecPolicyWorker)
    aux_batch.install_worker(worker)
    with pytest.raises(RuntimeError, match="native schedule failed"):
        worker.forward_backward(None)
    assert aux_batch._ACTIVE.get() is None


def test_aux_training_refuses_an_unregistered_worker():
    assert aux_batch._ACTIVE.get() is None
    with pytest.raises(RuntimeError, match="registered IsoExec policy worker"):
        aux_batch.annotate_microbatches([{"attention_mask": torch.ones(1, 2)}], required=True)


def test_registered_actor_preserves_native_calls_when_aux_is_not_enabled(framework, monkeypatch):  # noqa: F811
    from isoexec.integrations.skyrl.workers import IsoExecPolicyWorker

    def native_forward_backward(self, value):
        assert aux_batch._ACTIVE.get() is None
        return value

    native, _ = framework
    monkeypatch.setattr(native.MegatronPolicyWorkerBase, "forward_backward", native_forward_backward)
    worker = object.__new__(IsoExecPolicyWorker)
    assert worker.forward_backward(7) == 7
