"""Entry-point validation of the KDA operators that needs no GPU (no build).

Inputs that the kernels would dereference on the host's address space or past an allocation are refused with
ValueError, as ``gdn_cpr_sm100`` refuses them (``_check_inputs`` / ``_check_state`` / ``_check_slots``). The
extensions are replaced by a stub that raises on any use, so a missing check fails the test and never launches.
"""

from __future__ import annotations

import pytest
import torch


class _Launched(AssertionError):
    pass


class _Stub:
    def __getattr__(self, name):
        raise _Launched(f"the call reached the extension ({name}) without being refused")


@pytest.fixture
def chunk(monkeypatch):
    from isoexec.ops.kda.glm_sm100 import integration, ops

    monkeypatch.setattr(ops, "extension", lambda: _Stub())
    monkeypatch.setattr(integration, "extension", lambda: _Stub())
    return ops


@pytest.fixture
def token(monkeypatch):
    from isoexec.ops.kda.glm_rec_sm100 import ops

    monkeypatch.setattr(ops, "extension", lambda: _Stub())
    return ops


def _rows(n, H=2, device="cpu"):
    q, k, v = (torch.zeros(n, H, 128, dtype=torch.bfloat16, device=device) for _ in range(3))
    return q, k, v, torch.zeros(n, H, 128, device=device), torch.zeros(n, H, dtype=torch.bfloat16, device=device)


# ------------------------------------------------------------------------------- slot indices (shared)
def test_host_slots_are_range_checked_against_the_pool_cpu():
    from isoexec.ops.kda import _checks

    ok = _checks.slots("t", "slots", [0, 3, -1, 7], 4, "cpu", capacity=4, null_slot=7)   # -1, 7: padding
    assert ok.dtype == torch.int32 and ok.tolist() == [0, 3, -1, 7] and ok.is_contiguous()
    for bad in ([0, 4, 1, 2], [9, 0, 1, 2]):
        with pytest.raises(ValueError, match="out of range for a pool of 4 slots"):
            _checks.slots("t", "slots", bad, 4, "cpu", capacity=4, null_slot=7)
    with pytest.raises(ValueError, match="out of range"):       # no padding lanes: a negative slot is an error
        _checks.slots("t", "slots", [0, -1], 2, "cpu", capacity=4, padding=False)
    with pytest.raises(ValueError, match="must hold 3 slot indices, got 2"):
        _checks.slots("t", "slots", [0, 1], 3, "cpu", capacity=4)
    with pytest.raises(ValueError, match="integer slot indices"):
        _checks.slots("t", "slots", torch.zeros(2), 2, "cpu", capacity=4)
    assert _checks.slots("t", "slots", torch.tensor([1, 2], dtype=torch.int64), 2, "cpu", capacity=4).dtype == torch.int32


# ------------------------------------------------------------------------ host tensors never reach a kernel
def test_token_reducer_refuses_host_tensors_cpu(token):
    q, k, v, g, b = _rows(3)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    state = torch.zeros(4, 2, 128, 128)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    f = torch.zeros(3, 256, dtype=torch.bfloat16)
    calls = {
        "forward": lambda: token.forward(q, k, v, g, b, cu),
        "forward_train": lambda: token.forward_train(q, k, v, g, b, cu),
        "prefill": lambda: token.prefill(q, k, v, g, b, cu, state, [-1, -1], [0, 1]),
        "prefill_fused": lambda: token.prefill_fused(q, k, v, f, b, torch.zeros(256), torch.zeros(2), -5.0, cu, state,
                                                     [-1, -1], [0, 1]),
        "decode_step": lambda: token.decode_step(q, k, v, g, b, state, slots),
        "decode_step_fused": lambda: token.decode_step_fused(q, k, v, f, b, torch.zeros(256), torch.zeros(2), -5.0,
                                                             state, slots),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="CUDA"):
            call()


def test_chunk_reducer_refuses_host_tensors_cpu(chunk):
    from isoexec.ops.kda.glm_sm100 import integration, record_layout

    q, k, v, g, b = _rows(3)
    st = chunk.KdaState.__new__(chunk.KdaState)
    st.H, st.data = 2, torch.zeros(4, 2, record_layout.REC_BYTES, dtype=torch.uint8)
    f = torch.zeros(3, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        chunk.decode_step_fused(q, k, v, f, b, torch.zeros(256), torch.zeros(2), -5.0, st, [0, 1, 2])
    with pytest.raises(ValueError, match="CUDA"):
        chunk.record_state(st, [0, 1])
    with pytest.raises(ValueError, match="CUDA"):
        integration.forget_gate(f, torch.zeros(256), torch.zeros(2), -5.0)
    with pytest.raises(ValueError, match="CUDA"):
        integration.gated_rms_norm(q, k, torch.ones(128, dtype=torch.bfloat16), 1e-6)
