"""Model Runner V2's decode-sampler hook draws and scores bit for bit what the decode sampler does.

Same logits, per-row temperature, score temperature, default CUDA generator seed and step offset:
the V2 ``Sampler.__call__`` wrapper must return the kernel's own ``decode_step`` tokens, logprobs,
ranks and top-1, in V2's output contract (int64 tokens [rows, 1], int64 logprob token ids).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="requires SM100 (B200)"
)

from isoexec.runtimes.vllm import vllm_patches as patches  # noqa: E402

V = 248320
ROWS = 7


def _fs():
    from isoexec.ops.logprobs.logprobs_sm100 import sampler

    return sampler


def _v2_sampler_and_batch(temperature, k):
    """A V2 sampler whose request slots are the batch rows reversed, with a plain decode batch."""
    rows = temperature.numel()
    slots = rows + 3
    order = np.arange(rows)[::-1].copy()
    temp = np.zeros(slots, dtype=np.float32)
    temp[order] = temperature.cpu().numpy()
    dev = temperature.device
    states = SimpleNamespace(
        vocab_size=V,
        temperature=SimpleNamespace(np=temp, gpu=torch.from_numpy(temp).to(dev)),
        top_k=SimpleNamespace(np=np.full(slots, V, dtype=np.int32)),
        top_p=SimpleNamespace(np=np.ones(slots, dtype=np.float32)),
        min_p=SimpleNamespace(np=np.zeros(slots, dtype=np.float32)),
        seeds_set=np.zeros(slots, dtype=bool),
    )
    sampler = SimpleNamespace(
        sampling_states=states, logprobs_mode="raw_logprobs", num_speculative_tokens=1, trace_replay_state=None,
        return_sampling_mask=False, compute_nans=True, needs_logits_processing=np.zeros(slots, dtype=bool),
        _isoexec_processed=np.zeros(slots, dtype=bool),
        get_logprobs_dims=lambda idx: None if k is None else (k, 0),
        req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.ones(slots, dtype=torch.int32, device=dev))),
    )
    idx = torch.from_numpy(order).to(dev)
    batch = SimpleNamespace(
        idx_mapping_np=order, idx_mapping=idx, expanded_idx_mapping=idx, num_reqs=rows,
        seq_lens=torch.full((rows,), 9, dtype=torch.int32, device=dev),
        cu_num_logits=torch.arange(rows + 1, dtype=torch.int32, device=dev),
        num_computed_tokens_np=np.full(rows, 8), num_scheduled_tokens=np.ones(rows, dtype=np.int64),
    )
    return sampler, batch


def _logits(dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(ROWS, V, device="cuda", generator=g) * 4
    x[1, : V // 3] = -float("inf")
    x[2] = 1.5
    return x.to(dtype).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [None, 0, 1])
@pytest.mark.parametrize("temps", ["mixed", "greedy"])
@pytest.mark.parametrize("score_temp", [None, 0.7])
def test_v2_hook_is_the_decode_sampler_draw(dtype, k, temps, score_temp):
    from vllm.v1.outputs import LogprobsTensors

    fs = _fs()
    x = _logits(dtype, 3)
    greedy = temps == "greedy"
    if greedy:
        temperature = torch.zeros(ROWS, device="cuda")
    else:
        temperature = torch.tensor([1.0, 0.6, 0.0, 1.3, 1.0, 1e-6, 0.9], device="cuda")
    st = None if score_temp is None else torch.tensor(score_temp, device="cuda")

    v2 = patches._make_vllm_v2_fused_sampler_call(
        lambda *a: pytest.fail("V2 fell back"), SimpleNamespace, lambda: fs, lambda: st,
    )
    sampler, batch = _v2_sampler_and_batch(temperature, k)

    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    offset = gen.get_offset()
    temp, seed, step = patches._fused_draw(fs, x, None if greedy else temperature, greedy)
    token, lp = fs.decode_step(x, temp, seed, step, k, st)
    advanced = gen.get_offset()
    gen.set_offset(offset)
    b = v2(sampler, x, batch)
    assert gen.get_offset() == advanced  # V2 consumed the same step offset
    assert advanced == offset + (0 if greedy else fs.OFFSET_STEP)

    assert b.sampled_token_ids.dtype == torch.int64 and b.sampled_token_ids.shape == (ROWS, 1)
    assert torch.equal(b.sampled_token_ids.view(-1), token.to(torch.int64))
    assert b.num_nans.tolist() == [0] * ROWS
    assert b.num_sampled.tolist() == [1] * ROWS and b.num_rejected.tolist() == [0] * ROWS
    if k is None:
        assert lp is None and b.logprobs_tensors is None
        return
    la, lb = LogprobsTensors(*lp), b.logprobs_tensors
    assert lb.logprob_token_ids.dtype == torch.int64 and lb.logprob_token_ids.shape == (ROWS, 1 + k)
    assert torch.equal(lb.logprob_token_ids, la.logprob_token_ids.to(torch.int64))
    assert torch.equal(lb.logprob_token_ids[:, 0:1], b.sampled_token_ids)
    assert lb.logprobs.dtype == torch.float32 and torch.equal(_bits(lb.logprobs), _bits(la.logprobs))
    assert lb.selected_token_ranks.dtype == torch.int64
    assert torch.equal(lb.selected_token_ranks, la.selected_token_ranks)
    assert lb.cu_num_generated_tokens is None and lb.cu_num_generated_tokens_tensor is None
    assert torch.isfinite(lb.logprobs).all()
