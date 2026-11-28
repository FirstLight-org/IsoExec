"""The IsoExec policy actor selects its GPU by position in the inherited CUDA_VISIBLE_DEVICES."""

import os

import pytest

from isoexec.integrations.skyrl.device_ordinal import (
    NOSET_VISIBLE_DEVICES,
    actor_local_rank,
    VisibleDeviceOrdinal,
    visible_device_ordinal,
)


@pytest.mark.parametrize(
    ("gpu_id", "visible", "ordinal"),
    [
        (5, "5", 0),
        ("5", "5", 0),
        (5, "3,5,6", 1),
        (6, "3,5,6", 2),
        (3, "3,5,6", 0),
        ("5", " 3, 5 ,6", 1),
        (2, "0,1,2,3", 2),
        ("GPU-b", "GPU-a,GPU-b", 1),
        (5, None, 5),
    ],
)
def test_ordinal_is_the_position_in_visible_devices(gpu_id, visible, ordinal):
    assert visible_device_ordinal(gpu_id, visible) == ordinal


@pytest.mark.parametrize("visible", ["3,5,6", "5", ""])
def test_gpu_outside_visible_devices_refuses(visible):
    with pytest.raises(RuntimeError, match="not listed in CUDA_VISIBLE_DEVICES"):
        visible_device_ordinal(4, visible)


def test_local_rank_follows_noset_and_visible_devices():
    assert actor_local_rank([5], {"CUDA_VISIBLE_DEVICES": "5"}) is None  # Ray remapped the actor
    assert actor_local_rank([5], {NOSET_VISIBLE_DEVICES: "1", "CUDA_VISIBLE_DEVICES": "5"}) == "0"
    assert actor_local_rank(["6"], {NOSET_VISIBLE_DEVICES: "1", "CUDA_VISIBLE_DEVICES": "3,5,6"}) == "2"
    assert actor_local_rank([5], {NOSET_VISIBLE_DEVICES: "1"}) == "5"  # node exposes every GPU
    with pytest.raises(RuntimeError, match="exactly one GPU"):
        actor_local_rank([3, 5], {NOSET_VISIBLE_DEVICES: "1", "CUDA_VISIBLE_DEVICES": "3,5,6"})


@pytest.mark.parametrize(
    ("visible", "gpu_id", "expected"), [("5", 5, "0"), ("3,5,6", 5, "1"), ("3,5,6", 6, "2"), ("2,5", 2, "0")]
)
def test_policy_actor_replaces_the_physical_local_rank(monkeypatch, visible, gpu_id, expected):
    import ray

    class NativeActor:
        def __init__(self):
            # SkyRL's DistributedTorchRayActor under NOSET: the physical id becomes LOCAL_RANK.
            os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])

    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [gpu_id])
    monkeypatch.setenv(NOSET_VISIBLE_DEVICES, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("LOCAL_RANK", "unset")
    type("Actor", (VisibleDeviceOrdinal, NativeActor), {})()
    assert os.environ["LOCAL_RANK"] == expected


def test_policy_actor_keeps_ray_remapped_local_rank(monkeypatch):
    import ray

    class NativeActor:
        def __init__(self):
            os.environ["LOCAL_RANK"] = "0"

    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [5])
    monkeypatch.delenv(NOSET_VISIBLE_DEVICES, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    type("Actor", (VisibleDeviceOrdinal, NativeActor), {})()
    assert os.environ["LOCAL_RANK"] == "0"
