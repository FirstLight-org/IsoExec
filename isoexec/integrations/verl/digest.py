"""Parameter digests for the W0/W1 receipts (did the weights change between versions?).

Production runs digest on the device: per parameter, three integer sums over its 16-bit words (plain,
squared and position-hashed; ``ops/digest``), all parameters in ONE kernel launch with one copy to the
host, folded with the names into one SHA-256 over a few bytes per tensor. That costs milliseconds;
the receipt only has to tell two versions of one side apart, and the per-step bitwise logprob comparison
is the actual trainer/engine equality check. Full-row (debug) runs keep the byte-exact SHA-256 over
every parameter, which copies the whole model to the host.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch

from isoexec.ops.digest import digest_terms


@torch.no_grad()
def parameter_digest(named_parameters: Iterable[tuple[str, torch.Tensor]]) -> str:
    from .full_distribution import full_rows_enabled

    exact = full_rows_enabled()
    digest = hashlib.sha256()
    items = sorted(named_parameters, key=lambda item: item[0])
    on_device = [] if exact else [p for _, p in items if p.device.type == "cuda"]
    terms = iter(digest_terms(on_device)) if on_device else iter(())
    for name, parameter in items:
        digest.update(name.encode())
        if exact or parameter.device.type != "cuda":
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        else:
            s1, s2, s3 = next(terms)
            digest.update(f"{parameter.dtype}:{tuple(parameter.shape)}:{s1}:{s2}:{s3}".encode())
    return digest.hexdigest()


__all__ = ["parameter_digest"]
