"""The eager attention binding executes without either host runtime, even with the engine flag on.

Run in a fresh process under the pinned GPU Python. This proves the numerical binding, not the
legacy process-wide aten installer (which still imports vLLM for non-attention overrides).
"""

from __future__ import annotations

import importlib.abc
import sys


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"vllm", "megatron"}:
            raise ImportError(f"runtime forbidden in eager attention: {fullname}")
        return None


def main():
    assert not any(m.split(".", 1)[0] in {"vllm", "megatron"} for m in sys.modules)
    sys.meta_path.insert(0, NoRuntime())
    import torch

    from isoexec.ops.attention.gqa_sm90.integration import trainer_core as attention_core_varlen
    from isoexec.runtimes.trainer_installs import apply_trainer_env

    apply_trainer_env(None)
    dev = torch.device("cuda", 0)
    torch.manual_seed(47)
    count = 0
    for hq, hk, d in ((4, 2, 64), (4, 2, 128), (2, 1, 256), (16, 4, 256)):
        for length in (17, 129, 300):
            q = torch.randn(length, hq, d, dtype=torch.bfloat16, device=dev, requires_grad=True)
            k = torch.randn(length, hk, d, dtype=torch.bfloat16, device=dev, requires_grad=True)
            v = torch.randn(length, hk, d, dtype=torch.bfloat16, device=dev, requires_grad=True)
            cu = torch.tensor([0, length], dtype=torch.int32, device=dev)
            out = attention_core_varlen(q, k, v, cu, length, d**-0.5)
            with torch.no_grad():
                score = attention_core_varlen(q, k, v, cu, length, d**-0.5)
            assert torch.equal(out, score), (hq, hk, d, length)
            out.float().square().sum().backward()
            assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
            count += 1
    assert not any(m.split(".", 1)[0] in {"vllm", "megatron"} for m in sys.modules)
    print(f"RESULT: PASS eager attention runtime-free; {count} score/forward/backward geometries")


if __name__ == "__main__":
    main()
