"""Local reverse-scan ABI and launch recipe; native numerics require H100."""

from types import SimpleNamespace
from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward as backward


def chunk_local_cumsum_scalar_kernel():
    pass


def chunk_local_cumsum():
    pass


def _kernel(monkeypatch, *, warps=1):
    jit = SimpleNamespace(arg_names=list(backward._REVERSE_ARGS), fn=chunk_local_cumsum_scalar_kernel)
    tuner = SimpleNamespace(configs=[SimpleNamespace(num_warps=warps, num_stages=3, num_ctas=1, kwargs={})],
                            fn=jit, cache={"old": object()})
    heuristics = SimpleNamespace(arg_names=list(backward._REVERSE_ARGS), fn=tuner,
                                 values={"HAS_SCALE": object(), "IS_VARLEN": object()})
    monkeypatch.setitem(chunk_local_cumsum.__globals__, "chunk_local_cumsum_scalar_kernel", heuristics)
    return jit, tuner


def test_native_recipe_unwrap_does_not_mutate_autotuner(monkeypatch):
    jit, tuner = _kernel(monkeypatch)
    old_configs, old_cache = tuner.configs, dict(tuner.cache)
    assert backward._reverse_kernel(chunk_local_cumsum) is jit
    assert tuner.configs is old_configs and tuner.cache == old_cache
    identity = backward.reverse_cumsum_identity()
    assert identity["recipe"] == "fla_scalar_reverse_bt64_w1_s3_v1"
    assert identity["num_warps"] == 1 and identity["num_stages"] == 3


def test_moved_native_kernel_abi_refuses(monkeypatch):
    jit, _ = _kernel(monkeypatch)
    jit.arg_names.remove("REVERSE")
    with pytest.raises(backward.FlaMoved, match="launch ABI"):
        backward._reverse_kernel(chunk_local_cumsum)


def test_absent_native_config_refuses(monkeypatch):
    _kernel(monkeypatch, warps=2)
    with pytest.raises(backward.FlaMoved, match="declared native"):
        backward._reverse_kernel(chunk_local_cumsum)


def test_reverse_scan_refuses_cpu_before_kernel_launch():
    namespace = SimpleNamespace(reverse_cumsum_kernel=None)
    with pytest.raises(ValueError, match="contiguous CUDA FP32"):
        backward.reverse_cumsum(namespace, torch.empty(1, 38, 8), torch.tensor([0, 38]),
                                torch.tensor([[0, 0]]))


def test_wrong_source_refuses(monkeypatch):
    jit, _ = _kernel(monkeypatch)
    jit.fn = lambda: None
    with pytest.raises(backward.FlaMoved, match="different native source"):
        backward._reverse_kernel(chunk_local_cumsum)


def test_launch_selects_input_device_and_restores_caller(monkeypatch):
    events = []
    current = [torch.device("cuda:0")]
    device = torch.device("cuda:1")

    class Metadata:
        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.device = shape, dtype, device
            self.ndim = len(shape)

        def is_contiguous(self):
            return True

        def numel(self):
            return self.shape[0]

        def __len__(self):
            return self.shape[0]

    @contextmanager
    def selected(target):
        old = current[0]
        current[0] = target
        events.append(("enter", target))
        try:
            yield
        finally:
            current[0] = old
            events.append(("exit", old))

    class Kernel:
        def __getitem__(self, grid):
            assert current[0] == device
            events.append(("grid", grid))

            def launch(**kwargs):
                assert current[0] == device
                assert {name: kwargs[name] for name in ("BT", "REVERSE", "HEAD_FIRST", "HAS_SCALE", "IS_VARLEN",
                                                        "num_warps", "num_stages", "num_ctas")} == {
                    "BT": 64, "REVERSE": True, "HEAD_FIRST": False, "HAS_SCALE": False, "IS_VARLEN": True,
                    "num_warps": 1, "num_stages": 3, "num_ctas": 1}
                events.append(("launch", current[0]))
                return SimpleNamespace(metadata=SimpleNamespace(num_warps=1, num_stages=3, num_ctas=1),
                                       hash="actual-compiled-kernel-hash")

            return launch

    def allocate(value):
        assert current[0] == device
        events.append(("allocate", current[0]))
        return value

    monkeypatch.setattr(torch.cuda, "device", selected)
    monkeypatch.setattr(torch, "empty_like", allocate)
    value = Metadata((1, 38, 8), torch.float32)
    result = cast(Any, backward.reverse_cumsum)(SimpleNamespace(reverse_cumsum_kernel=Kernel()), value,
                                    Metadata((2,), torch.int64), Metadata((1, 2), torch.int64))
    assert result is value and current[0] == torch.device("cuda:0")
    assert [name for name, _ in events] == ["enter", "allocate", "grid", "launch", "exit"]


def test_compiled_geometry_and_hash_are_checked():
    compiled = SimpleNamespace(metadata=SimpleNamespace(num_warps=2, num_stages=3, num_ctas=1), hash="real")
    with pytest.raises(backward.FlaMoved, match="geometry/hash"):
        backward._compiled_reverse(compiled)
    compiled.metadata.num_warps = 1
    del compiled.hash
    compiled.metadata.hash = "wrong-owner"
    with pytest.raises(backward.FlaMoved, match="geometry/hash"):
        backward._compiled_reverse(compiled)
