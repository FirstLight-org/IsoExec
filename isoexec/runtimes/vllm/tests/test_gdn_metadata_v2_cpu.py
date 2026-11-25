"""GDN metadata lookup under Model Runner V2 dummy runs, which omit Mamba-family layers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

gdn_attn = pytest.importorskip("vllm.v1.attention.backends.gdn_attn")


def _lookup(monkeypatch, metadata, prefix="body.layers.0.mixer.core"):
    from isoexec.runtimes.vllm import gdn_engine_patch as patch

    monkeypatch.setattr(patch, "_GDN_CANON_PREFIX", [])
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    return patch.gdn_metadata(prefix)


def test_v2_warmup_without_any_gdn_metadata_is_metadata_free(monkeypatch):
    assert _lookup(monkeypatch, {"body.layers.3.mixer.core": object()}) is None
    assert _lookup(monkeypatch, None) is None


def test_missing_layer_with_other_gdn_metadata_still_refuses(monkeypatch):
    other = gdn_attn.GDNAttentionMetadata.__new__(gdn_attn.GDNAttentionMetadata)
    monkeypatch.setattr("isoexec.runtimes.vllm.gdn_engine_patch.gdn_alias_active", lambda: False)
    with pytest.raises(KeyError, match="missing while other GDN layers"):
        _lookup(monkeypatch, {"body.layers.1.mixer.core": other})


def test_v2_finished_and_preempted_requests_release_their_private_gdn_rows(monkeypatch):
    # V2 retires finished and preempted requests at the head of execute_model, also on steps that schedule
    # no tokens and never reach prepare_inputs (warmup cleanup); their rows must return to the pool there.
    import isoexec.ops.gdn as gdn
    from isoexec.runtimes.vllm import gdn_engine_patch as patch

    block_table = pytest.importorskip("vllm.v1.worker.gpu.block_table")
    model_runner = pytest.importorskip("vllm.v1.worker.gpu.model_runner")
    runner_cls = model_runner.GPUModelRunner
    for cls, name in ((runner_cls, "finish_requests"), (runner_cls, "initialize_kv_cache"),
                      (runner_cls, "prepare_inputs"), (block_table.BlockTables, "append_block_ids")):
        monkeypatch.setattr(cls, name, getattr(cls, name))  # restored on teardown
    seen = []
    monkeypatch.setattr(runner_cls, "finish_requests", lambda self, out: seen.append(out))

    class Pool:
        def __init__(self):
            self.released = []

        def release_slot(self, slot):
            self.released.append(slot)

    pools = [Pool(), Pool()]
    monkeypatch.setattr(gdn, "CPR_LAYERS", pools)
    monkeypatch.setattr(patch, "_DRIVER_SLOT_BY_REQ", {"done": 7, "preempted": 8, "live": 9})
    assert patch._install_v2_driver()

    out = SimpleNamespace(finished_req_ids={"done", "never-prefilled"}, preempted_req_ids={"preempted"},
                          total_num_scheduled_tokens=0)
    runner_cls.finish_requests(object(), out)
    assert seen == [out]
    assert [p.released for p in pools] == [[7, 8], [7, 8]]
    assert patch._DRIVER_SLOT_BY_REQ == {"live": 9}
