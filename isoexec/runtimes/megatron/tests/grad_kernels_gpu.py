"""Pinned-TE regression and allocation battery: run directly with the GPU interpreter."""

import json

import torch

from isoexec.runtimes.megatron.grad_kernels import (
    install_mixed_dtype_grad_kernels,
)


def main() -> None:
    from megatron.core.optimizer import clip_grads

    device = torch.device("cuda", 0)
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    # BF16 first makes the old kernel read only a prefix of the fp32 storage, so this negative
    # control exposes dtype corruption without reading beyond the shorter bf16 allocation.
    controls = [
        torch.full((128,), 0.00314159, dtype=dtype, device=device)
        for dtype in (torch.bfloat16, torch.float32)
    ]
    wrong, _ = clip_grads.multi_tensor_applier(
        clip_grads.l2_norm_impl, flag, [controls], False
    )
    reference = (
        torch.stack(
            [g.to(dtype=torch.float64, device=device).square().sum() for g in controls]
        )
        .sum()
        .sqrt()
    )
    assert not torch.isclose(wrong.double(), reference, rtol=1e-5, atol=1e-8).all()
    install_mixed_dtype_grad_kernels()
    installed = clip_grads.multi_tensor_applier
    install_mixed_dtype_grad_kernels()
    assert clip_grads.multi_tensor_applier is installed

    checked = 0
    for sizes in ((1, 1), (128, 257), (65536, 1024)):
        for dtypes in (
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float32),
        ):
            for factor in (0.0, 0.25, 0.9375):
                values = [
                    torch.full((size,), 0.00314159, dtype=dtype, device=device)
                    for size, dtype in zip(sizes, dtypes, strict=True)
                ]
                expected = (
                    torch.stack(
                        [
                            g.to(dtype=torch.float64, device=device).square().sum()
                            for g in values
                        ]
                    )
                    .sum()
                    .sqrt()
                )
                got, _ = installed(clip_grads.l2_norm_impl, flag, [values], False)
                torch.testing.assert_close(
                    got.double().reshape(()), expected, rtol=2e-6, atol=1e-8
                )
                before = [v.clone() for v in values]
                pointers = [v.data_ptr() for v in values]
                coefficient = torch.full(
                    (1,), factor, dtype=torch.float32, device=device
                )
                installed(
                    clip_grads.multi_tensor_scale_tensor_impl,
                    flag,
                    [values, values],
                    coefficient,
                )
                assert [v.data_ptr() for v in values] == pointers
                assert all(
                    torch.equal(v, b * factor)
                    for v, b in zip(values, before, strict=True)
                )
                checked += 1

    values = [
        torch.full((64 * 1024 * 1024,), 0.125, dtype=torch.bfloat16, device=device),
        torch.full((1024,), 0.25, dtype=torch.float32, device=device),
    ]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    norm, _ = installed(clip_grads.l2_norm_impl, flag, [values], False)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    assert torch.isfinite(norm).all()
    assert extra < 8 * 1024 * 1024, (
        f"gradient-buffer-sized temporary allocation: {extra} bytes"
    )
    print(
        json.dumps(
            {
                "cases": checked,
                "negative_control_detected": True,
                "gradient_bytes": sum(v.numel() * v.element_size() for v in values),
                "norm_peak_extra_bytes": extra,
            }
        )
    )
    print(
        f"RESULT: PASS {checked} dtype/norm/clipping cases; original corruption detected; no gradient-sized copy"
    )


if __name__ == "__main__":
    main()
