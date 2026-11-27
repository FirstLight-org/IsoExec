"""Entry-point validation of the KDA operators on device tensors.

``decode_step`` wrote through a caller's ``out`` / ``workspace`` of any size, slot indices were never compared with
the pool (``record_state`` dereferenced a negative slot), and the forget gate handed ``dt_bias`` / ``A_log`` of any
dtype to a kernel that reads fp32. Each is now refused or converted on the host, as ``gdn_cpr_sm100`` does.

No refused call can reach a kernel: every refusal is asserted with the extensions replaced by a stub that raises on
use (a missing check fails at the stub). The two tests that then call the real kernels do so only after the stubbed
call has shown the check runs first.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_sm100.tests._fixtures import require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


class _Launched(AssertionError):
    pass


class _Recorder:
    """Records the first extension call (name, args) and refuses to run it."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            raise _Launched(f"the call reached the extension ({name})")
        return call


@pytest.fixture
def stub(monkeypatch):
    from isoexec.ops.kda.glm_rec_sm100 import ops as rec_ops
    from isoexec.ops.kda.glm_sm100 import integration, ops

    rec = _Recorder()
    for mod in (ops, integration, rec_ops):
        monkeypatch.setattr(mod, "extension", lambda: rec)
    return rec


N, H, CAP = 3, 2, 4


def _rows(n=N, h=H):
    q, k, v = (torch.zeros(n, h, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    return q, k, v, torch.zeros(n, h, 128, device="cuda"), torch.zeros(n, h, dtype=torch.bfloat16, device="cuda")


def _pool(h=H, cap=CAP):
    from isoexec.ops.kda.glm_sm100 import ops, record_layout

    st = ops.KdaState.__new__(ops.KdaState)        # no extension call
    st.H, st.data = h, torch.zeros(cap, h, record_layout.REC_BYTES, dtype=torch.uint8, device="cuda")
    return st


def _reaches_the_kernel(stub, call):
    with pytest.raises(_Launched):
        call()
    return stub.calls[-1]


# --------------------------------------------------------------------------------------- chunk reducer
def test_decode_step_refuses_a_short_or_mistyped_out(stub):
    from isoexec.ops.kda.glm_sm100 import ops

    q, k, v, g, b = _rows()
    sl = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    for out in (torch.empty(N - 1, H, 128, dtype=torch.bfloat16, device="cuda"),
                torch.empty(N, H, 128, device="cuda"),
                torch.empty(N, H, 256, dtype=torch.bfloat16, device="cuda")[..., :128],
                torch.empty(N, H, 128, dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match="out must be"):
            ops.decode_step(q, k, v, g, b, _pool(), sl, out=out)
        with pytest.raises(ValueError, match="out must be"):
            ops.decode_step_fused(q, k, v, torch.zeros(N, H * 128, dtype=torch.bfloat16, device="cuda"), b,
                                  torch.zeros(H * 128, device="cuda"), torch.zeros(H, device="cuda"), -5.0, _pool(),
                                  sl, out=out)
    good = torch.empty(N, H, 128, dtype=torch.bfloat16, device="cuda")
    name, args = _reaches_the_kernel(stub, lambda: ops.decode_step(q, k, v, g, b, _pool(), sl, out=good))
    assert name == "decode_rec" and args[7] is good
    assert not stub.calls[:-1]


def test_decode_step_refuses_a_short_workspace(stub):
    from isoexec.ops.kda.glm_sm100 import ops

    q, k, v, g, b = _rows()
    sl = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    for ws in (ops.decode_workspace(N - 1, H, "cuda"), ops.decode_workspace(N, H, "cuda")[:4],
               ops.decode_workspace(N, H + 1, "cuda")):
        with pytest.raises(ValueError, match="workspace must be"):
            ops.decode_step(q, k, v, g, b, _pool(), sl, workspace=ws)
    ws = ops.decode_workspace(N, H, "cuda")
    name, args = _reaches_the_kernel(stub, lambda: ops.decode_step(q, k, v, g, b, _pool(), sl, workspace=ws))
    assert all(a is w for a, w in zip(args[8], ws))
    assert not stub.calls[:-1]


def test_slots_are_checked_against_the_pool(stub):
    from isoexec.ops.kda.glm_sm100 import ops

    q, k, v, g, b = _rows()
    with pytest.raises(ValueError, match="out of range for a pool of 4 slots"):
        ops.decode_step(q, k, v, g, b, _pool(), [0, 4, 1])
    with pytest.raises(ValueError, match="must hold 3 slot indices"):
        ops.decode_step(q, k, v, g, b, _pool(), [0, 1])
    with pytest.raises(ValueError, match="state must be"):
        ops.decode_step(q, k, v, g, b, _pool(h=H + 1), [0, 1, 2])
    cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="out of range"):
        ops.prefill(q, k, v, g, b, cu, _pool(), [-1, -1, -1], [0, 1, 9])
    with pytest.raises(ValueError, match="out of range"):
        ops.record_state(_pool(), [0, -1])                                       # no padding lanes in record_state
    with pytest.raises(ValueError, match="out of range"):
        ops.record_state(_pool(), torch.tensor([0, -1], dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="out of range"):
        ops.record_state(_pool(), torch.tensor([4], dtype=torch.int32, device="cuda"))
    assert not stub.calls
    # padding lanes (negative, null_slot) and in-range slots go through
    _reaches_the_kernel(stub, lambda: ops.decode_step(q, k, v, g, b, _pool(), [-1, 3, 0], null_slot=0))
    _reaches_the_kernel(stub, lambda: ops.record_state(_pool(), [3, 0]))


def test_forget_gate_widens_its_parameters_to_fp32_and_keeps_fp32_ones(stub):
    from isoexec.ops.kda.glm_sm100 import integration

    f = torch.zeros(5, H * 128, dtype=torch.bfloat16, device="cuda")
    dt, A = torch.randn(H * 128, device="cuda"), torch.randn(H, device="cuda")
    _, args = _reaches_the_kernel(stub, lambda: integration.forget_gate(f, dt, A, -5.0))
    assert args[1].dtype == torch.float32 and args[1].data_ptr() == dt.data_ptr()      # fp32: the same tensor
    assert args[2].dtype == torch.float32 and args[2].data_ptr() == A.data_ptr()
    for dtype in (torch.bfloat16, torch.float16):
        _, args = _reaches_the_kernel(stub, lambda: integration.forget_gate(f, dt.to(dtype), A.to(dtype), -5.0))
        assert args[1].dtype == torch.float32 and args[2].dtype == torch.float32
        assert torch.equal(args[1], dt.to(dtype).float()) and args[1].is_contiguous()
    with pytest.raises(ValueError, match="dt_bias"):
        integration.forget_gate(f, dt[:128], A, -5.0)
    with pytest.raises(ValueError, match="bf16 or fp32 projection"):
        integration.forget_gate(f.half(), dt, A, -5.0)
    with pytest.raises(ValueError, match="weight"):
        integration.gated_rms_norm(f.view(5, H, 128), f.view(5, H, 128), torch.ones(128, dtype=torch.float16,
                                                                                    device="cuda"), 1e-6)


def test_forget_gate_of_low_precision_parameters_is_that_of_their_fp32_widening(stub, monkeypatch):
    """The real kernel, once the stubbed call above has shown the conversion happens before the launch."""
    from isoexec.ops.kda.glm_sm100 import integration
    from isoexec.ops.kda.glm_sm100._extension import extension

    f = torch.randn(7, H * 128, device="cuda").bfloat16()
    dt, A = torch.randn(H * 128, device="cuda").bfloat16(), torch.randn(H, device="cuda").bfloat16()
    _, args = _reaches_the_kernel(stub, lambda: integration.forget_gate(f, dt, A, -5.0))
    assert args[1].dtype == torch.float32 and args[2].dtype == torch.float32
    monkeypatch.setattr(integration, "extension", extension)
    a = integration.forget_gate(f, dt, A, -5.0)
    b = integration.forget_gate(f, dt.float(), A.float(), -5.0)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


# --------------------------------------------------------------------------------------- token reducer
def test_token_reducer_checks_out_state_and_slots(stub):
    from isoexec.ops.kda.glm_rec_sm100 import ops

    q, k, v, g, b = _rows()
    state = torch.zeros(CAP, H, 128, 128, device="cuda")
    sl = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="out must be"):
        ops.decode_step(q, k, v, g, b, state, sl, out=torch.empty(N - 1, H, 128, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="state must be"):
        ops.decode_step(q, k, v, g, b, state.cpu(), sl)
    with pytest.raises(ValueError, match="state must be"):
        ops.decode_step(q, k, v, g, b, state[:, :1], sl)
    with pytest.raises(ValueError, match="out of range"):
        ops.decode_step(q, k, v, g, b, state, [0, 1, 4])
    cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="out of range"):
        ops.prefill(q, k, v, g, b, cu, state, [-1, -1, -1], [0, 1, 5])
    assert not stub.calls
    _reaches_the_kernel(stub, lambda: ops.decode_step(q, k, v, g, b, state, [-1, 3, 0], null_slot=0))
    f = torch.zeros(N, H * 128, dtype=torch.bfloat16, device="cuda")
    dt = torch.randn(H * 128, device="cuda")
    _, args = _reaches_the_kernel(stub, lambda: ops.decode_step_fused(q, k, v, f, b, dt.bfloat16(), dt[:H].bfloat16(),
                                                                      -5.0, state, sl))
    assert args[5].dtype == torch.float32 and args[6].dtype == torch.float32      # the fused kernel reads fp32
    _, args = _reaches_the_kernel(stub, lambda: ops.decode_step_fused(q, k, v, f, b, dt, dt[:H].contiguous(), -5.0,
                                                                      state, sl))
    assert args[5].data_ptr() == dt.data_ptr()                                    # fp32: the same tensor
