"""GPU relations of the group tree: R-A9 and R-A10.

R-A9: G workers (one CTA per group, fp32 partials) + the combine kernel are bit-exact against the
emulation's canonical adjacent-pair tree for every G in 1..8, and the emulation's one-worker form
(``reference.tree_fold_stack``) equals its G-worker form bit for bit.  R-A10: the result for a fixed G
does not depend on the batch or on which other rows share the launch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs an SM100 device")

from isoexec.ops.attention.dsa_sm100 import reference as ref  # noqa: E402
from isoexec.ops.attention.dsa_sm100.fwd import dsa_sparse_attn  # noqa: E402
from isoexec.ops.attention.dsa_sm100.tests._prims import DevicePrims  # noqa: E402

import functools as _ft
import os as _os
#: DSA_TEST_RECIPE=2 runs this battery on recipe v2 (sm100_dsa_sparse_v2, lazy rescale) against the
#: lazy emulation; counts go to the *_v2.json file.
RECIPE = int(_os.environ.get("DSA_TEST_RECIPE", "1"))
dsa_sparse_attn = _ft.partial(dsa_sparse_attn, recipe=RECIPE)
_emul = _ft.partial(ref.sparse_attn_emul, lazy=RECIPE == 2)
from isoexec.ops.attention.dsa_sm100.tests.test_attn_gpu import Case, SCALE, _bits_out, _count  # noqa: E402

#: (G, K): every group a whole number of 64-slot tiles
GK = [(1, 2048), (2, 2048), (3, 1920), (4, 2048), (5, 1920), (6, 1920), (7, 1792), (8, 2048)]


@pytest.mark.parametrize("G,K", GK)
def test_ra9_g_workers_equal_the_tree(G, K):
    case = Case([9000, 2500, 300], K=K, seed=20 + G)
    case.lists[2, 150:] = -1                              # whole groups of pads: empty partials
    out, lse = dsa_sparse_attn(case.q, case.cache, case.bt, case.lists, SCALE, groups=G)
    mism = n = 0
    for b in range(case.B):
        rows, valid = case.rows_of(b)
        q = case.q[b].float().cpu().numpy()
        eo, el, _ = _emul(q, rows, valid, SCALE, groups=G, prims=DevicePrims)
        eo1, el1, _ = _emul(q, rows, valid, SCALE, groups=G, prims=DevicePrims, one_worker=True)
        assert np.array_equal(eo, eo1) and np.array_equal(el.view(np.uint32), el1.view(np.uint32))
        go = _bits_out(out[b])
        gl = lse[b].cpu().numpy()
        mism += int((go != eo).sum()) + int((gl.view(np.uint32) != el.view(np.uint32)).sum())
        n += go.size + gl.size
    _count(f"R-A9 G workers == tree (G={G})", n, mism)
    assert mism == 0


def test_ra10_batch_and_plan_independence():
    case = Case([6000, 4000, 8000, 1000], K=2048, seed=40)
    for G in (2, 4, 8):
        full_o, full_l = dsa_sparse_attn(case.q, case.cache, case.bt, case.lists, SCALE, groups=G)
        bad = 0
        for b in range(case.B):
            o1, l1 = dsa_sparse_attn(case.q[b:b + 1], case.cache, case.bt[b:b + 1], case.lists[b:b + 1], SCALE, groups=G)
            bad += int((o1.view(torch.int16) != full_o[b:b + 1].view(torch.int16)).sum())
            bad += int((l1.view(torch.int32) != full_l[b:b + 1].view(torch.int32)).sum())
        _count(f"R-A10 batch composition at G={G}", full_o.numel() + full_l.numel(), bad)
        assert bad == 0
