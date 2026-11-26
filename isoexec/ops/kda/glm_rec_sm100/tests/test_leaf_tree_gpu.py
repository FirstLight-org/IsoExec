"""The vectorized o_proj leaf tree of the recurrent recipe's engine is the chunkwise package's leaf program.

This is the one model-independent test of the source tree's ``test_engine_rec_gpu.py`` (the others build an
``isoexec.nn.GLMKimiDeltaAttention``, which this tree does not have, and ``ops/`` may not import ``isoexec.nn``
here). The composed layer of both reducers is covered by ``isoexec/ops/kda/tests/test_schedule_invariance_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_rec_sm100 import engine
from isoexec.ops.kda.glm_rec_sm100.tests._fixtures import require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def test_vectorized_leaf_tree_equals_the_cpr_leaf_program_bitwise():
    from isoexec.ops.kda.glm_sm100 import engine as cpr

    for leaves, root in ((8, False), (4, True), (2, True), (1, False)):
        for T in (1, 5, 128):
            y = torch.randn(T, 1024, device="cuda").bfloat16()
            w = (torch.randn(512, 1024, device="cuda") * 0.05).bfloat16()
            got = engine.o_proj_leaves(y, w, leaves, root)
            want = cpr.o_proj_leaves(y, w, leaves, root)
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), (leaves, T)
