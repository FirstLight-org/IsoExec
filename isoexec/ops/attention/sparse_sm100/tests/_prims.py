"""The device primitives of the fp32 emulation (``reference.CpuPrims``'s interface), evaluated on the
GPU under this package's profile: exp2f, the root's division and the root's lse expression.  The
tensor-core accumulation stays the CPU model (exact k-step block, one round-to-nearest), which R-I2
validated for the block-scaled indexer and R-A2 validates for kind::f16."""

from __future__ import annotations

import numpy as np
import torch

from isoexec.ops.attention.sparse_sm100 import reference as ref
from isoexec.ops.attention.sparse_sm100._extension import extension


class DevicePrims:
    exact = True

    @staticmethod
    def _t(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()

    @staticmethod
    def exp2f(x):
        x = np.asarray(x, dtype=np.float32)
        return extension("prims").exp2f(DevicePrims._t(x.reshape(-1))).cpu().numpy().reshape(x.shape)

    @staticmethod
    def divf(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.broadcast_to(np.asarray(b, dtype=np.float32), a.shape)
        return extension("prims").div(DevicePrims._t(a.reshape(-1)), DevicePrims._t(b.reshape(-1))).cpu().numpy().reshape(a.shape)

    @staticmethod
    def lse(L, M):
        L = np.asarray(L, dtype=np.float32)
        return extension("prims").lse(DevicePrims._t(L.reshape(-1)), DevicePrims._t(np.asarray(M, np.float32).reshape(-1))).cpu().numpy().reshape(L.shape)

    logf = staticmethod(ref.CpuPrims.logf)
    mma_step = staticmethod(ref.CpuPrims.mma_step)
    mma_f16 = staticmethod(ref.CpuPrims.mma_f16)
