# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified for IsoExec: only ``get_compute_capability`` is kept from flashinfer/utils.py.
from __future__ import annotations

import torch


def get_compute_capability(device: torch.device):
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    return torch.cuda.get_device_capability(device.index)
