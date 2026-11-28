"""Cold-start timers: accumulate, emit one stable record per process, stay off when disabled."""

import json

import pytest

from isoexec.core import load_timers


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(load_timers, "_TIMERS", {})
    monkeypatch.setattr(load_timers, "_STATE", {"emitted": False, "side": None, "atexit": False})
    monkeypatch.delenv("ISOEXEC_LOAD_TIMERS", raising=False)
    monkeypatch.delenv("ISOEXEC_DG_JIT_CACHE_DIR", raising=False)


def test_phases_accumulate_and_emit_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ISOEXEC_LOAD_TIMERS_DIR", str(tmp_path))
    with load_timers.phase("engine.load_model"):
        pass
    with load_timers.phase("engine.load_model"):
        pass
    lap = load_timers.Lap()
    lap("model_init.capture")
    load_timers.mark("ready")
    record = load_timers.emit("engine", note="x")
    assert record["timers"]["engine.load_model"]["n"] == 2
    assert {"model_init.capture", "process.ready", "process.emit"} <= set(record["timers"])
    assert load_timers.emit("engine") is None
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith(load_timers.MARKER)]
    assert len(lines) == 1
    parsed = json.loads(lines[0].split(" ", 1)[1])
    assert parsed["side"] == "engine" and parsed["note"] == "x"
    files = list(tmp_path.glob("engine-*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["timers"] == parsed["timers"]


def test_disabled_records_nothing(monkeypatch):
    monkeypatch.setenv("ISOEXEC_LOAD_TIMERS", "0")
    with load_timers.phase("engine.load_model"):
        pass
    assert load_timers.snapshot() == {} and load_timers.emit("engine") is None


def test_first_call_times_only_the_first_and_emits(monkeypatch, capsys):
    calls = []
    wrapped = load_timers.first_call(lambda x: calls.append(x) or x, "trainer.first_backward", emit_side="trainer")
    assert wrapped(1) == 1 and wrapped(2) == 2 and calls == [1, 2]
    assert load_timers.snapshot()["trainer.first_backward"]["n"] == 1
    assert sum(line.startswith(load_timers.MARKER) for line in capsys.readouterr().err.splitlines()) == 1


def test_dg_jit_compiles_are_read_from_new_cache_entries(tmp_path, monkeypatch):
    import os
    import time

    entry = tmp_path / "cache" / "kernel.abc"
    entry.mkdir(parents=True)
    now = time.time()
    old = tmp_path / "cache" / "kernel.old"          # compiled before this process: a hit, not counted
    old.mkdir()
    for name, at in (("kernel.cu", now - 50), ("kernel.cubin", now - 40), (".committed", now + 5)):
        (old / name).write_text("")
        os.utime(old / name, (at, at))
    for name, at in (("kernel.cu", now), ("kernel.cubin", now + 2.5), (".committed", now + 2.6)):
        (entry / name).write_text("")
        os.utime(entry / name, (at, at))
    monkeypatch.setenv("ISOEXEC_DG_JIT_CACHE_DIR", f"{tmp_path}:/elsewhere")
    monkeypatch.setattr(load_timers, "process_start_wall", lambda: now - 10)
    load_timers._dg_jit_compiles()
    assert load_timers.snapshot()["dg_jit.compile"] == {"s": 2.5, "n": 1}
