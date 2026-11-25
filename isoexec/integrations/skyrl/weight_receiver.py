"""Native vLLM transport with IsoExec model update completion and IPC cleanup."""

from typing import Any, cast

import torch
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine


class _NativeModelUpdate(WeightTransferEngine):
    """Retain vLLM's receive path; use registered model names without HF reload."""

    supports_draft_weight_update = False

    def _close_imports(self) -> None:
        importer = getattr(self, "_packed_importer", None)
        if importer is not None:
            importer.close()

    def start_weight_update(self) -> None:
        with torch.device(self.device):
            cast(Any, self.model).start_weight_update()

    def receive_weights(self, update_info: Any) -> None:
        try:
            with torch.device(self.device):
                # The other base in the concrete class supplies the transport.
                cast(Any, super()).receive_weights(update_info)
        except BaseException:
            self._close_imports()
            raise

    def finish_weight_update(self) -> None:
        from skyrl.backends.skyrl_train.weight_sync.weight_receivers import empty_cuda_cache_rocm

        try:
            with torch.device(self.device):
                cast(Any, self.model).finish_weight_update()
        finally:
            self._close_imports()
        empty_cuda_cache_rocm()


class IsoExecNCCLWeightTransferEngine(_NativeModelUpdate, NCCLWeightTransferEngine):
    """NCCL receives into the registered IsoExec model."""


class IsoExecIPCWeightTransferEngine(_NativeModelUpdate, IPCWeightTransferEngine):
    """IPC receives into the registered model and releases imported buffers."""
