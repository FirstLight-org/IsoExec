"""More sequences than a grid axis holds are refused on the host, before any launch (no GPU, no build).

The chunk reducer's state pass (forward, prefill), backward state pass, decode row kernel and ``record_state`` put one
sequence (decode lane, slot) per row of grid axis y; the token reducer's kernel puts them on axis z. CUDA caps both
axes at 65535, and a launch over the cap launches nothing: the output stays uninitialised. Each entry point now
refuses such a call before it allocates or launches anything.

No call here can reach a kernel: the extensions are replaced by a stub that raises on any use, and every tensor is on
the CPU. A missing check therefore fails the test (the stub or the CUDA-device check raises instead of the limit
refusal); it can never launch.
"""

from __future__ import annotations

import pytest
import torch

LIMIT = 65535
REFUSED = "grid . axis holds"


class _Launched(AssertionError):
    pass


class _Stub:
    def __getattr__(self, name):
        raise _Launched(f"the call reached the extension ({name}) without being refused")


@pytest.fixture
def chunk(monkeypatch):
    from isoexec.ops.kda.glm_sm100 import ops

    monkeypatch.setattr(ops, "extension", lambda: _Stub())
    return ops


@pytest.fixture
def token(monkeypatch):
    from isoexec.ops.kda.glm_rec_sm100 import ops

    monkeypatch.setattr(ops, "extension", lambda: _Stub())
    return ops


def _rows(n, H=1):
    q, k, v = (torch.zeros(n, H, 128, dtype=torch.bfloat16) for _ in range(3))
    return q, k, v, torch.zeros(n, H, 128), torch.zeros(n, H, dtype=torch.bfloat16)


def _records(ops, slots=1, H=1):
    from isoexec.ops.kda.glm_sm100 import record_layout

    st = ops.KdaState.__new__(ops.KdaState)        # no build: the constructor asks the extension for rec_bytes
    st.H, st.data = H, torch.zeros(slots, H, record_layout.REC_BYTES, dtype=torch.uint8)
    return st


def _not_the_limit(call):
    """``call`` is not refused by the limit (it fails later, on the CPU tensors or at the stub)."""
    with pytest.raises((ValueError, RuntimeError, _Launched)) as e:
        call()
    assert "axis holds" not in str(e.value), e.value


@pytest.mark.parametrize("n", [LIMIT + 1])
def test_chunk_reducer_refuses_more_sequences_than_grid_y_holds_cpu(chunk, n):
    q, k, v, g, b = _rows(0)
    cu = torch.zeros(n + 1, dtype=torch.int32)
    with pytest.raises(ValueError, match=REFUSED):
        chunk.forward(q, k, v, g, b, cu, None)
    with pytest.raises(ValueError, match=REFUSED):
        chunk.backward(q, k, v, g, b, cu, None, q)
    with pytest.raises(ValueError, match=REFUSED):
        chunk.prefill(q, k, v, g, b, cu, _records(chunk), [-1] * n)
    q, k, v, g, b = _rows(n)
    with pytest.raises(ValueError, match=REFUSED):
        chunk.decode_step(q, k, v, g, b, _records(chunk), torch.zeros(n, dtype=torch.int32))
    f, braw = torch.zeros(n, 128, dtype=torch.bfloat16), torch.zeros(n, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=REFUSED):
        chunk.decode_step_fused(q, k, v, f, braw, torch.zeros(128), torch.zeros(1), -5.0, _records(chunk),
                                torch.zeros(n, dtype=torch.int32))
    with pytest.raises(ValueError, match=REFUSED):
        chunk.record_state(_records(chunk), torch.zeros(n, dtype=torch.int32))


def test_chunk_reducer_takes_exactly_the_limit_cpu(chunk):
    q, k, v, g, b = _rows(0)
    cu = torch.zeros(LIMIT + 1, dtype=torch.int32)
    _not_the_limit(lambda: chunk.forward(q, k, v, g, b, cu, None))
    q, k, v, g, b = _rows(LIMIT)
    _not_the_limit(lambda: chunk.decode_step(q, k, v, g, b, _records(chunk), torch.zeros(LIMIT, dtype=torch.int32)))


@pytest.mark.parametrize("n", [LIMIT + 1])
def test_token_reducer_refuses_more_sequences_than_grid_z_holds_cpu(token, n):
    q, k, v, g, b = _rows(0)
    cu = torch.zeros(n + 1, dtype=torch.int32)
    state = torch.zeros(1, 1, 128, 128)
    for call in (lambda: token.forward(q, k, v, g, b, cu, None),
                 lambda: token.forward_train(q, k, v, g, b, cu, None),
                 lambda: token.prefill(q, k, v, g, b, cu, state, [-1] * n, [-1] * n),
                 lambda: token.prefill_fused(q, k, v, q.view(0, 128), b, torch.zeros(128), torch.zeros(1), -5.0, cu,
                                             state, [-1] * n, [-1] * n)):
        with pytest.raises(ValueError, match=REFUSED):
            call()
    q, k, v, g, b = _rows(n)
    slots = torch.zeros(n, dtype=torch.int32)
    with pytest.raises(ValueError, match=REFUSED):
        token.decode_step(q, k, v, g, b, state, slots)
    with pytest.raises(ValueError, match=REFUSED):
        token.decode_step_fused(q, k, v, torch.zeros(n, 128, dtype=torch.bfloat16), b, torch.zeros(128),
                                torch.zeros(1), -5.0, state, slots)


def test_token_reducer_takes_exactly_the_limit_cpu(token):
    q, k, v, g, b = _rows(LIMIT)
    _not_the_limit(lambda: token.decode_step(q, k, v, g, b, torch.zeros(1, 1, 128, 128),
                                             torch.zeros(LIMIT, dtype=torch.int32)))
