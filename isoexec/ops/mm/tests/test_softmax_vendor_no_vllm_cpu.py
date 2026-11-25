"""Run the eager router on CPU in a subprocess that blocks all vLLM imports.
Executing the route exposes lazy runtime dependencies that import-only checks cannot detect."""

from __future__ import annotations

import pathlib
import subprocess
import sys

# Child cwd must contain the isoexec package so python -c can import it without inherited PYTHONPATH.
_REPO = pathlib.Path(__file__).resolve().parents[4]

_CHILD = r'''
import sys, types

class _NoVllm:
    """Refuse `vllm` and every submodule, wherever the import comes from."""
    def find_module(self, name, path=None):
        return None
    def find_spec(self, name, path=None, target=None):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError(f"vllm is not installed in this process (blocked by {__file__})")
        return None

sys.meta_path.insert(0, _NoVllm())
sys.modules["vllm"] = None          # `import vllm` -> ImportError, the specified spelling

# Belt and braces: prove the blocker actually blocks before trusting the rest of the test.
try:
    import vllm  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("BLOCKER FAILED: `import vllm` succeeded in the child")
try:
    from vllm.model_executor.determinism import batch_invariant  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("BLOCKER FAILED: the vLLM submodule import succeeded in the child")

import torch

# 1. The vendored symbol itself.
from isoexec.ops.mm.mm_batch_invariant import softmax_batch_invariant

x = torch.tensor([[0.5, -1.25, 3.0, 0.0], [-4.0, -4.0, -4.0, -4.0]], dtype=torch.float32)
p = softmax_batch_invariant(x, -1, torch.float32)
assert p.dtype is torch.float32, p.dtype
assert torch.allclose(p.sum(-1), torch.ones(2)), p.sum(-1)

# 2. The reference router path, which is what actually used to import vLLM.
from isoexec.ops.moe.eager import route_mechanics_megatron
from isoexec.schema import Mlp

spec = Mlp(intermediate_size=128, num_experts=4, top_k=2, routing="softmax_renorm", activation="silu_mul")
logits = torch.tensor(
    [[0.1, 0.9, -0.3, 0.4], [2.0, 2.0, 1.0, -1.0], [-5.0, -5.0, -5.0, -5.0]], dtype=torch.float32
)
res = route_mechanics_megatron(logits, spec, True)
assert res.indices.shape == (3, 2), res.indices.shape
assert res.probs.dtype is torch.float32, res.probs.dtype
assert torch.allclose(res.probs.sum(-1), torch.ones(3)), res.probs.sum(-1)

# 3. Nothing anywhere in that path replaced the poisoned entry.
assert sys.modules.get("vllm") is None, "something imported vllm after all"
assert not [m for m in sys.modules if m.startswith("vllm.")], [m for m in sys.modules if m.startswith("vllm.")]
print("OK no-vllm router")
'''


def test_eager_router_runs_without_vllm():
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, f"child failed:\nSTDOUT\n{out.stdout}\nSTDERR\n{out.stderr}"
    assert "OK no-vllm router" in out.stdout, out.stdout


if __name__ == "__main__":  # pragma: no cover - runnable in the runtime venv, which has no pytest
    test_eager_router_runs_without_vllm()
    print("PASS test_softmax_vendor_no_vllm_cpu")
