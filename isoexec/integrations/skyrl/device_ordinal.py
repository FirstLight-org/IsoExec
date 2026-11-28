"""Select each IsoExec SkyRL actor's CUDA device by its position in CUDA_VISIBLE_DEVICES."""

import os
from typing import Any

NOSET_VISIBLE_DEVICES = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"


def visible_device_ordinal(gpu_id: Any, visible: str | None) -> int:
    """CUDA ordinal of the Ray-assigned GPU ``gpu_id`` in a process whose devices are ``visible``.

    Under ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` (set by the IsoExec bootstrap) every actor keeps
    the node's ``CUDA_VISIBLE_DEVICES`` and Ray names the assigned GPU by its entry in that list. The
    CUDA ordinal is the entry's position; the entry equals the ordinal only when every GPU is visible.
    """
    if visible is None:
        return int(gpu_id)
    entries = [entry.strip() for entry in visible.split(",")]
    try:
        return entries.index(str(gpu_id).strip())
    except ValueError:
        raise RuntimeError(
            f"Ray assigned GPU {gpu_id!r}, which is not listed in CUDA_VISIBLE_DEVICES={visible!r}"
        ) from None


def actor_local_rank(gpu_ids: list[Any], environ: Any = os.environ) -> str | None:
    """The ``LOCAL_RANK`` the actor must select, or ``None`` when Ray already remaps devices."""
    if not environ.get(NOSET_VISIBLE_DEVICES):
        return None
    if len(gpu_ids) != 1:
        raise RuntimeError(f"an IsoExec SkyRL actor owns exactly one GPU, Ray assigned {gpu_ids!r}")
    return str(visible_device_ordinal(gpu_ids[0], environ.get("CUDA_VISIBLE_DEVICES")))


class VisibleDeviceOrdinal:
    """Mixin: replace the physical id SkyRL writes to ``LOCAL_RANK`` with its visible-device position.

    SkyRL's actor constructor sets ``LOCAL_RANK = ray.get_gpu_ids()[0]`` and selects the device later, in
    ``init_worker_process_group``, so the override runs right after the native constructor. With a GPU
    subset such as ``CUDA_VISIBLE_DEVICES=2,5`` the physical id is otherwise an invalid ordinal.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import ray

        local_rank = actor_local_rank(ray.get_gpu_ids())
        if local_rank is not None:
            os.environ["LOCAL_RANK"] = local_rank
