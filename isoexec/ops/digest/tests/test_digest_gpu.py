"""Multi-tensor digest kernel vs the torch reference (bitwise), determinism, and timing.

Run: ``CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. $PY isoexec/ops/digest/tests/test_digest_gpu.py [--bench]``
"""

from __future__ import annotations

import sys
import time

import torch

from isoexec.ops.digest import digest_terms, digest_terms_reference


def _skip() -> bool:
    return not torch.cuda.is_available()


def test_kernel_equals_reference():
    if _skip():
        return
    g = torch.Generator(device="cuda").manual_seed(0)
    ts = [torch.randn(n, device="cuda", generator=g).to(dt) for n, dt in
          ((1, torch.bfloat16), (7, torch.float32), (4099, torch.bfloat16), ((1 << 20) + 13, torch.bfloat16),
           (3 * (1 << 20), torch.float32), (64, torch.float16))]
    ts.append(torch.arange(9, dtype=torch.uint8, device="cuda"))           # odd byte count
    ts.append(torch.randn(256, 384, device="cuda").t())                    # non-contiguous
    ts.append(torch.randn(2 * 4096 + 8, device="cuda").to(torch.bfloat16)[3:4099])  # unaligned base
    got = digest_terms(ts)
    assert got == digest_terms_reference(ts), "kernel digest differs from the reference"
    assert digest_terms(ts) == got, "kernel digest is not deterministic"


def test_parameter_digest_detects_one_word():
    if _skip():
        return
    from isoexec.integrations.verl.digest import parameter_digest

    ps = [(f"p{i}", torch.randn(1000 + i, device="cuda").to(torch.bfloat16)) for i in range(50)]
    a = parameter_digest(ps)
    assert parameter_digest(ps) == a
    ps[17][1].view(torch.int16)[123] ^= 1
    assert parameter_digest(ps) != a


def bench():
    """~900 TP4-local Qwen3.8-27B-like params (~5 G bf16 elements): old per-param terms vs one launch."""
    shapes = []
    for _ in range(48):  # GDN layers (TP4 local)
        shapes += [(4120, 5120), (5120, 1536), (1536,), (5120,), (5120,), (16,), (16,), (4, 3072),
                   (4352, 5120), (5120, 4352), (5120,), (5120,)]
    for _ in range(16):  # attention layers
        shapes += [(3584, 5120), (5120, 1536), (256,), (256,), (5120,), (4352, 5120), (5120, 4352), (5120,), (5120,)]
    shapes += [(62080, 5120), (62080, 5120), (5120,)]
    ps = [torch.empty(s, device="cuda", dtype=torch.bfloat16).normal_() for s in shapes]
    n = sum(p.numel() for p in ps)

    def old():  # integrations/verl/digest.py::_device_terms at 18c3db82, per parameter
        out = []
        for p in ps:
            words = p.view(-1).view(torch.int16)
            total = torch.zeros((), dtype=torch.int64, device="cuda")
            squares = torch.zeros((), dtype=torch.float64, device="cuda")
            for start in range(0, words.numel(), 1 << 26):
                part = words[start:start + (1 << 26)]
                total += part.sum(dtype=torch.int64)
                squares += part.float().square().sum(dtype=torch.float64)
            out.append((int(total), float(squares)))
        return out

    for name, fn in (("old per-param", old), ("one launch", lambda: digest_terms(ps))):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print(f"[digest bench] {len(ps)} params, {n / 1e9:.2f} G elements: {name} {(time.perf_counter() - t0) / 3 * 1e3:.1f} ms")


if __name__ == "__main__":
    if _skip():
        print("SKIP: needs a CUDA device")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    if "--bench" in sys.argv:
        bench()
