# ---------------------------------------------------------------------------------------------
# VENDORED KERNEL -- DO NOT EDIT BY HAND.
# upstream_module: fla.utils._compat
# upstream_dist:   flash-linear-attention / fla-core 0.5.2  (tag v0.5.2, commit 9c8e42e762fce087c27b673af4922795d9edb85e)
# upstream_url:    https://github.com/fla-org/flash-linear-attention (MIT; see NOTICE)
# provenance:      ops/gdn/shared/vendored_fla_bwd/PROVENANCE.json
# local changes:   only the rewrite_rules recorded in PROVENANCE.json (import redirects into this
#                  tree); vendored_sha256 pins this file.
# why vendored:    the trainer's GDN backward runs these kernels; IsoExec owns the source it runs
#                  instead of importing the fla package at runtime.
# ---------------------------------------------------------------------------------------------
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools
import importlib.metadata
import inspect
import logging
import os
import shutil
from importlib.util import find_spec
from pathlib import Path

import triton
from packaging import version as package_version

from ._config import FLA_CACHE_RESULTS

logger = logging.getLogger(__name__)

TRITON_ABOVE_3_4_0 = package_version.parse(triton.__version__) >= package_version.parse("3.4.0")
TRITON_ABOVE_3_5_1 = package_version.parse(triton.__version__) >= package_version.parse("3.5.1")
TRITON_ABOVE_3_7_1 = package_version.parse(triton.__version__) >= package_version.parse("3.7.1")

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


@functools.cache
def find_spec_cached(name):
    return find_spec(name)


@functools.cache
def has_usable_nvcc() -> bool:
    """Whether a usable nvcc compiler is available for TileLang's JIT.

    Mirrors the guesses in ``tilelang.env._find_cuda_home`` (env
    CUDA_HOME/CUDA_PATH, nvcc on PATH, the ``nvidia-cuda-nvcc`` wheel,
    /usr/local/cuda), but verifies the nvcc binary actually exists —
    only ``nvidia-cuda-nvcc`` >= 13.0 ships it, the ``-cu12`` variant
    installs just ptxas.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None and (Path(cuda_home) / "bin" / "nvcc").exists():
        return True
    if shutil.which("nvcc") is not None:
        return True
    try:
        files = importlib.metadata.files("nvidia-cuda-nvcc") or []
    except importlib.metadata.PackageNotFoundError:
        files = []
    if any(f.name in ("nvcc", "nvcc.exe") for f in files):
        return True
    if (Path("/usr/local/cuda") / "bin" / "nvcc").exists():
        return True

    logger.info(
        "[FLA Backend] TileLang is installed but no usable nvcc compiler was found; falling back to Triton. "
        "Install a CUDA toolkit or nvidia-cuda-nvcc, or set FLA_TILELANG=0 to disable TileLang explicitly."
    )
    return False
