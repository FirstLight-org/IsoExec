"""CPU proofs for `mm_shapes.live_context()`'s resolution order.

The dict this returns decides which census the coverage banner -- the only in-process statement of
which GEMM sites the installed provider serves -- and the provider's probe budget are computed against.
A host registers the context once at install time (the reference host registers `tp=1 etp=1
side=TRAINER`) and nothing re-registers it when a runtime's parallel state comes up afterwards, so
a registration that wins outright makes every TP>1 engine worker report `tp=1` for the life of the
process. These tests pin the order that fixes that: live parallel state, then environment, then the
registration -- with the census and `model_path` still taken from the registration, since no
runtime carries them.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from isoexec.ops.mm import mm_shapes

CENSUS = mm_shapes.GemmCensus((mm_shapes.GemmSpec("attn.qkv", 16, 32, mm_shapes.dtype_of("bf16"), "tp"),))


@pytest.fixture(autouse=True)
def _isolate_live_context(monkeypatch):
    monkeypatch.setattr(mm_shapes, "_LIVE", None, raising=False)
    for var in ("ISOEXEC_MODEL_PATH", "ISOEXEC_ENGINE_TP", "ISOEXEC_ENGINE_EP"):
        monkeypatch.delenv(var, raising=False)
    yield
    mm_shapes._LIVE = None


def test_environment_parallelism_beats_a_stale_registration():
    # The defect, in one assertion: a TP=2 engine worker whose host registered tp=1 at install time.
    mm_shapes.set_live_context(census=CENSUS, model_path="/ckpt", tp=1, etp=1, side="TRAINER")
    import os

    os.environ["ISOEXEC_ENGINE_TP"] = "2"
    try:
        ctx = mm_shapes.live_context()
    finally:
        del os.environ["ISOEXEC_ENGINE_TP"]
    assert ctx == {"model_path": "/ckpt", "census": CENSUS, "tp": 2, "etp": 2, "side": "ENGINE"}


def test_the_registration_still_answers_when_no_runtime_and_no_environment_does():
    # A single-process trainer gate initialises no parallel state at all; the registration is then
    # the only source, and must still be used rather than yielding None.
    mm_shapes.set_live_context(census=CENSUS, model_path="/ckpt", tp=1, etp=1, side="TRAINER")
    assert mm_shapes.live_context() == {"model_path": "/ckpt", "census": CENSUS, "tp": 1, "etp": 1, "side": "TRAINER"}


def test_the_registered_model_path_survives_a_runtime_derived_parallelism():
    # `model_path` is the one field no runtime can be asked for, so it must come from the
    # registration even when tp/etp/side do not.
    mm_shapes.set_live_context(census=CENSUS, model_path="/registered", tp=1, etp=1, side="TRAINER")
    import os

    os.environ["ISOEXEC_MODEL_PATH"] = "/from-env"
    os.environ["ISOEXEC_ENGINE_TP"] = "8"
    try:
        ctx = mm_shapes.live_context()
    finally:
        del os.environ["ISOEXEC_MODEL_PATH"]
        del os.environ["ISOEXEC_ENGINE_TP"]
    assert ctx["model_path"] == "/registered"
    assert ctx["tp"] == 8


def test_no_model_path_anywhere_is_still_an_honest_none():
    assert mm_shapes.live_context() is None


def test_a_registered_census_needs_no_model_path():
    mm_shapes.set_live_context(census=CENSUS, tp=1, etp=1, side="TRAINER")
    assert mm_shapes.live_context() == {"model_path": "", "census": CENSUS, "tp": 1, "etp": 1, "side": "TRAINER"}


def _fake_megatron_mpu(monkeypatch, *, initialized, tp=None, etp=None, etp_raises=False):
    """Install a `megatron.core.parallel_state` with the three entry points `live_context` calls.

    The runtime branch is the one the fix is ABOUT -- an engine or trainer worker whose parallel
    state came up after the host registered its context -- and it is the branch a CPU test can only
    reach by injecting the module, because no parallel state can be initialised without a GPU
    world. Everything `live_context` touches on this path is a plain function call, so the
    injection exercises the real code rather than a re-implementation of it.
    """
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.parallel_state")
    mpu.model_parallel_is_initialized = lambda: initialized
    mpu.get_tensor_model_parallel_world_size = lambda: tp

    def _etp():
        if etp_raises:  # older megatron: no expert-tensor-parallel accessor at all
            raise AttributeError("get_expert_tensor_parallel_world_size")
        return etp

    mpu.get_expert_tensor_parallel_world_size = _etp
    core.parallel_state = mpu
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)
    # vLLM is consulted first; make sure its absence is what the trainer path actually sees.
    monkeypatch.setitem(sys.modules, "vllm.distributed", None)


def test_a_live_megatron_parallel_state_beats_both_the_environment_and_the_registration(monkeypatch):
    # The whole point of the fix, in one assertion: a trainer worker at TP=4/ETP=2 whose host
    # registered tp=1 at install time and whose environment still carries an engine's TP=2.
    _fake_megatron_mpu(monkeypatch, initialized=True, tp=4, etp=2)
    mm_shapes.set_live_context(census=CENSUS, model_path="/ckpt", tp=1, etp=1, side="TRAINER")
    monkeypatch.setenv("ISOEXEC_ENGINE_TP", "2")

    assert mm_shapes.live_context() == {"model_path": "/ckpt", "census": CENSUS, "tp": 4, "etp": 2, "side": "TRAINER"}


def test_a_live_parallel_state_without_an_expert_accessor_falls_back_to_etp_equals_tp(monkeypatch):
    # `get_expert_tensor_parallel_world_size` is not in every megatron; the census still needs an
    # etp, and tp is the only honest answer when the runtime cannot be asked.
    _fake_megatron_mpu(monkeypatch, initialized=True, tp=8, etp_raises=True)
    mm_shapes.set_live_context(census=CENSUS, model_path="/ckpt", tp=1, etp=1, side="TRAINER")

    assert mm_shapes.live_context() == {"model_path": "/ckpt", "census": CENSUS, "tp": 8, "etp": 8, "side": "TRAINER"}


def test_an_uninitialised_parallel_state_does_not_claim_the_process(monkeypatch):
    # megatron imported but never initialised is the driver process and every CPU gate. It must not
    # answer `tp=None` as if it had, and it must not stop the environment from being consulted.
    _fake_megatron_mpu(monkeypatch, initialized=False)
    mm_shapes.set_live_context(census=CENSUS, model_path="/ckpt", tp=1, etp=1, side="TRAINER")
    monkeypatch.setenv("ISOEXEC_ENGINE_TP", "2")

    assert mm_shapes.live_context() == {"model_path": "/ckpt", "census": CENSUS, "tp": 2, "etp": 2, "side": "ENGINE"}

    monkeypatch.delenv("ISOEXEC_ENGINE_TP")
    assert mm_shapes.live_context() == {"model_path": "/ckpt", "census": CENSUS, "tp": 1, "etp": 1, "side": "TRAINER"}


def test_a_live_parallel_state_answers_even_with_no_registration_at_all(monkeypatch):
    # `model_path` from the environment, parallelism from the runtime: a worker that was never
    # registered still reports its live parallelism rather than None; with no census registered,
    # the banner then names the missing census instead of inventing one.
    _fake_megatron_mpu(monkeypatch, initialized=True, tp=2, etp=2)
    monkeypatch.setenv("ISOEXEC_MODEL_PATH", "/from-env")

    assert mm_shapes.live_context() == {"model_path": "/from-env", "census": None, "tp": 2, "etp": 2, "side": "TRAINER"}
